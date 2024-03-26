import utils
import torch
import torch.nn.functional as F
from fastai.vision.models import resnet18, resnet34
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# NOTE: we can change/add to this list
compared_models = {"resnet18": resnet18, "resnet34": resnet34}

class ImageCommandModel(nn.Module):
    """Initializes the CommandModel class."""

    def __init__(self, architecture_name: str, pretrained: bool):
        super(ImageCommandModel, self).__init__()
        cnn_constructor = compared_models[architecture_name]
        weights = "IMAGENET1K_V1" if pretrained else None
        self.cnn = cnn_constructor(weights=weights)

        # Layers to combine image and command input
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, img, cmd):
        """Performs a forward pass through the model."""

        # Pass the image data to the cnn
        x_image = self.cnn(img)

        # Returns a new tensor from the cmd data
        x_command = cmd.unsqueeze(1)

        # Concatenate cmd and image in the 1st dimension
        x = torch.cat((x_image, x_command), dim=1)

        # Apply the ReLU function element-wise to the linearly transformed img+cmd data
        x = self.r1(self.fc1(x))

        # Apply the linear transformation to the data
        x = self.fc2(x)

        # The loss function applies softmax to the output of the model
        return x

class ImageActionTransformer(nn.Module):
    """
    This is our big transformer model! It takes a list of images and their corresponding actions to make a prediction.

    Parameters:
        num_encoder_layers (int): Number of encoder layers in the transformer.
        nhead (int): Number of heads in the multiheadattention models.
        d_model (int): Dimension of the input image features to the transformer. Resnet-18 gives 512
        dim_feedforward (int): Dimension of the feedforward network model.
        dropout (float): Dropout value.
        num_actions (int): Number of possible actions.
    """
    def __init__(self, num_encoder_layers, nhead, d_model, dim_feedforward, dropout, num_actions):
        super(ImageActionTransformer, self).__init__()
        self.actions = num_actions # The number of possible actions the model can take
        self.cnn = CNNFeatureExtractor()
        self.cmd_projection = nn.Linear(num_actions, d_model)

        # Transformer Layers
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)

        # Output Layer
        self.output_layer = nn.Linear(d_model, self.actions)
   
    def forward(self, img, cmd):
        """
        Performs a forward pass through the model.
        
        Parameters:
            img (PIL image or numpy array): A batch of images.
            cmd (Tensor): A batch of one hot encoded vectors representing discrete actions.
        Returns:
            Tensor: The model's prediction for each image-action pair.
                
        """
        # Apply transformations and feature extraction to images
        img_features = self.cnn(img)

        # Combine image features and commands
        cmd_one_hot = torch.nn.functional.one_hot(torch.tensor(cmd).to(torch.int64), self.actions)
        cmd_embedding = self.cmd_projection(cmd_one_hot.float())
        combined_features = img_features + cmd_embedding
        combined_features = combined_features.unsqueeze(0)

        # Pass through transformer
        transformer_output = self.transformer_encoder(combined_features)

        # Pass through a fully connected output layer, which will return an action list
        output = self.output_layer(transformer_output.squeeze(0))

        return output

class HybridModel(nn.Module):
    """
    This model combines both the ImageTransformer and ActionLSTM. It is a more efficient approach
    than just throwing all of the data into a really big transformer. It uses a small transformer
    with the same inner workings as the larger one, but it has a maximum sequence cutoff that limits
    how far back it looks for images. The long term dependencies necessary for exploration are captured 
    by an lstm that only stores the actions that the robot took.

    Large sequence sizes with small max sequence sizes will be the best combo, since the sequence contains the 
    long term dependicies for the lstm

    Parameters:
        num_encoder_layers (int): Number of encoder layers in the transformer. We don't need many because the
                                 dependencies are not relatively short
        nhead (int): Number of heads in the multiheadattention models.
        d_model (int): Dimension of the input image features to the transformer. Resnet-18 gives 512
        dim_feedforward (int): Dimension of the feedforward network model.
        dropout (float): Dropout value.
        num_actions (int): Number of possible actions.
        max_sequence_len (int): Determines how many images the transformer looks at at a time.
        hidden_size_lstm (int): Size of the hidden layers in the lstm. 
        num_layers_lstm (int): Number of layers in lstm. 1 is usually good enough, unless its a very complex pattern.
    """
    def __init__(self, num_encoder_layers, nhead, d_model, dim_feedforward,
                    dropout, num_actions, max_sequence_len, hidden_size_lstm,
                    num_layers_lstm):
        
        super(HybridModel, self).__init__()
        self.transformer = ImageTransformer(num_encoder_layers=num_encoder_layers, nhead=nhead, 
                                            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, 
                                            num_actions=num_actions, max_sequence_len=max_sequence_len)
        self.lstm = ActionLSTM(num_actions=num_actions, hidden_size=hidden_size_lstm, 
                               num_layers=num_layers_lstm)
        self.projection_layer = nn.Linear(hidden_size_lstm, d_model)
        self.weights_transformer = torch.nn.Parameter(torch.randn(d_model))
        self.weights_lstm = torch.nn.Parameter(torch.randn(d_model))
        self.output_layer = nn.Linear(d_model, num_actions)
    
    def forward(self, batch):
        """
        Performs a forward pass through the model.
        
        Parameters:
            sequences (tensor): batch x sequences x channels + commands x H x W
        """
        
        images = batch[0] # [B x max_seq x C x H x W]
        commands = batch[1]  # [B x seq x actions]

        while images.dim() < 5:
            images.unsqueeze(0)
        while commands.dim() < 3:
            commands.unsqueeze(0)
        
        # Push data through models and apply activations
        transformer_out = self.transformer(images, commands) # [B x 512]
        transformer_out = F.leaky_relu(transformer_out)
        
        lstm_out = self.lstm(commands) # [B x hidden_size_lstm]
        lstm_out = F.leaky_relu(lstm_out)
        
        lstm_projection = self.projection_layer(lstm_out) # [B x 512]
        lstm_projection = F.leaky_relu(lstm_projection)
        
        # Weighted combination
        merged_output = (transformer_out * self.weights_transformer.unsqueeze(0)) + (lstm_projection * self.weights_lstm.unsqueeze(0))
        merged_output = F.relu(merged_output) # [B x 512]
        
        # Output Layer and softmax
        output = self.output_layer(merged_output) # [B x 3]
        
        return output

class ImageTransformer(nn.Module):
    """
    Transformer that processes only a small number of images relative to the
    ImageActionTransformer. To be used in combination with ActionLSTM.

    Lower number of encoder layers, since we don't need as many long term dependencies.

    We also implement a maximum sequence length to restrict the amount of images processed.
    """
    def __init__(self, num_encoder_layers, nhead, d_model, dim_feedforward,
                  dropout, num_actions, max_sequence_len):
        super(ImageTransformer, self).__init__()
        self.actions = num_actions
        self.max_sequence_len = max_sequence_len
        self.cnn = CNNFeatureExtractor()
        self.cmd_projection = nn.Linear(num_actions, d_model)

        # Allows transformer to understand sequence of images
        positional_encoding = utils.create_positional_encoding(max_sequence_len, d_model)
        self.register_buffer('positional_encoding', positional_encoding)

        # Transformer Layers
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)

        # Pool each sequence
        self.attention_pooling = AttentionPooling(512)
        
        # Output Layer
        self.output_layer = nn.Linear(d_model, self.actions)

    def forward(self, img, cmd):

        batched = False
        if img.dim() == 5:
            batched = True
        while cmd.dim() < 3:
            cmd.unsqueeze(0)
        while img.dim() < 5:
            img = img.unsqueeze(0)

        batch_size = img.shape[0]
        seq_length = img.shape[1]

        # resize images
        pad_length = 0
        if seq_length == self.max_sequence_len:
            pass
        elif seq_length > self.max_sequence_len:
            # If sequence length exceeds the max, truncate it
            img = img[:, -self.max_sequence_len:]
        else:
            # Pad the sequence if it's shorter than the max length
            pad_length = self.max_sequence_len - seq_length
            padded_img_shape = (batch_size, pad_length, *img.shape[2:])
            img = torch.cat([torch.zeros(padded_img_shape, device=img.device), img], dim=1)
        
        # Resize commands
        cmd_seq_len = cmd.shape[-2]
        if cmd_seq_len > self.max_sequence_len:
            cmd = cmd[:, -self.max_sequence_len:]
        elif cmd_seq_len < self.max_sequence_len:
            pad = self.max_sequence_len - self.max_seq_length
            padding = (batch_size, pad, cmd.shape[-1])
            cmd = torch.cat([torch.zeros(padding, device=cmd.device), cmd], dim=1)
            
        # attention mask 
        attention_mask = utils.create_attention_mask(batch_size, self.max_sequence_len, pad_length, cmd)
    
        # Combine image features and command embeddings
        img_features = self.cnn(img)
        cmd_embedding = self.cmd_projection(cmd.float())
        combined_features = img_features + cmd_embedding

        #add positional encoding
        combined_features += self.positional_encoding
        
        if batched == False: # unbatch for single items
            combined_features.squeeze(0)
            attention_mask.squeeze(0)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(combined_features, src_key_padding_mask=attention_mask)
        
        # [BxSeqx512] -> pool -> [Bx512] for one prediction per sequence
        pooled_output = self.attention_pooling(transformer_output)

        return pooled_output

class ActionLSTM(nn.Module):
    """
    This LSTM model will be used to capture the patterns in the actions for use in prediction alongside images.
    It can take any number of sequence length, allowing unlimited long term dependencies and capturing changes over time.

    A LSTM is good at 'forgetting' information that is no longer relevant. ex. traveling in a straight line down a hallway
    """
    def __init__(self, num_actions, hidden_size, num_layers):
        super(ActionLSTM, self).__init__()
        # LSTM Layers
        self.lstm = nn.LSTM(input_size=num_actions, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
    def forward(self, cmd):
        cmd = cmd.float()
        _, (final_layer, _) = self.lstm(cmd)
        return final_layer[-1] # Return the hidden state from the last layer


class CNNFeatureExtractor(nn.Module):
    """
    Extract features from images

    Converts PIL images or numpy arrays into tensors
    """
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        
        # Use ResNet18 for image feature extraction, remove the final layer to get feature vector
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

    
    def forward(self, imgs):
        original_dim = imgs.dim()
        while imgs.dim() < 5:
            imgs.unsqueeze(0)
       
        batch_size, seq_len, c, h, w = imgs.size()
        
        # reshape to treat the sequence as part of the batch
        imgs_reshaped = imgs.view(batch_size * seq_len, c, h, w)
        
        # Remove extra channels
        imgs_reshaped = imgs_reshaped[:,:3,:,:]
        
        # process batch through the CNN
        extracted_features = self.feature_extractor(imgs_reshaped)
        
        #flatten
        features_flat = torch.flatten(extracted_features, start_dim=1)
        
        # return batch dimension
        features = features_flat.view(batch_size, seq_len, -1)

        # Adjust features to match original input dimensions
        if original_dim == 3:  # Single image [C, H, W]
            features = features.squeeze(0).squeeze(0)  # Remove batch and sequence dimensions
        elif original_dim == 4:  # Sequence of images [B, C, H, W]
            features = features.squeeze(0)  # Remove Batch dimension
        
        return features

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionPooling, self).__init__()
        # Initialize a query vector that will be used to compute attention scores for each timestep
        self.query = nn.Parameter(torch.randn(feature_dim, 1))
        
    def forward(self, x):
        batch_size, sequence_length, feature_dim = x.size()

        # Compute attention scores 
        attn_scores = torch.matmul(x.view(-1, feature_dim), self.query).view(batch_size, sequence_length)
        
        # Apply softmax to get attention weights 
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Use attention weights to compute a weighted sum of features across timesteps
        weighted_sum = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # Shape: [batch_size, feature_dim]
        
        return weighted_sum
