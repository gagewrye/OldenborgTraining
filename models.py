from fastai.vision.models import resnet18
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import resnet18
import torch.nn.functional as functional

class CNNFeatureExtractor(nn.Module):
    """
    Extract features from images

    Converts PIL images or numpy arrays into tensors
    """
    def __init__(self, d_model=512):
        super(CNNFeatureExtractor, self).__init__()
        
        # Use ResNet18 for image feature extraction, remove the final layer to get feature vector
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.adapt_features = nn.Linear(d_model * 7 * 7, d_model)
    
    def forward(self, imgs):
        # Extract features
        features = self.feature_extractor(imgs)
        features = torch.flatten(features, start_dim=1)
        
        # Adapt features to match the model inputs
        return self.adapt_features(features)


class ImageActionTransformer(nn.Module):
    """
    This is our big transformer model! It takes a list of images and their corresponding actions to make a prediction.

    Inputs:
        img (PIL image or numpy array): A batch of images.
        cmd (Tensor or int list): A batch of actions represented as integer indices. Ex. [0,4,1,4,3,2] where each integer represents a discrete action.
        
    Parameters:
        num_encoder_layers (int): Number of encoder layers in the transformer.
        nhead (int): Number of heads in the multiheadattention models.
        d_model (int): Dimension of the input image features to the transformer. Resnet-18 gives 512
        dim_feedforward (int): Dimension of the feedforward network model.
        dropout (float): Dropout value.
        num_actions (int): Number of possible actions.
    """
    def __init__(self, num_encoder_layers=6, nhead=8, d_model=512, dim_feedforward=2048, dropout=0.1, num_actions=3): # TODO: Change params
        super(ImageActionTransformer, self).__init__()
        self.actions = num_actions # The number of possible actions the model can take
        self.cnn = CNNFeatureExtractor(d_model=d_model)

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
            cmd (Tensor or int list): A batch of actions represented as integer indices. Ex. [0,4,1,4,3,2] where each integer represents a discrete action.
        
        Returns:
            Tensor: The model's prediction for each image-action pair.
                
        """
        # Apply transformations and feature extraction to images
        img_features = self.cnn(img)

        # Convert the actions to one hot torch vectors  Ex. [2,0] -> [[0,0,1,0],[1,0,0,0]]
        # Each vector represents the action taken in one hot form. This helps us weight actions evenly and better differentiate between each action.
        cmd_embedding = functional.one_hot(cmd.to(torch.int64), self.actions)

        # Combine image features and command embeddings
        combined_features = torch.cat((img_features, cmd_embedding), dim=1)
        combined_features = combined_features.unsqueeze(0)

        # Pass through transformer
        transformer_output = self.transformer_encoder(combined_features)

        # Pass through a fully connected output layer, which will return an action list
        output = self.output_layer(transformer_output.squeeze(0))

        return output

class ActionLSTM(nn.Module):
    """
    This LSTM model will be used to capture the patterns in the actions for use in prediction alongside images.

    A LSTM is good at 'forgetting' information that is no longer relevant. ex. traveling in a straight line down a hallway
    """
    def __init__(self, num_actions=3, hidden_size=64, num_layers=2):
        super(ActionLSTM, self).__init__()
        self.actions = num_actions

        # LSTM Layers
        self.lstm = nn.LSTM(input_size=num_actions, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
    def forward(self, cmd):
        cmd_embedding = functional.one_hot(cmd.to(torch.int64), self.actions)
        _, (final_layer, _) = self.lstm(cmd_embedding)
        return final_layer[-1] # Return the hidden state from the last layer

class ImageTransformer(nn.Module):
    """
    Transformer that processes only a small number of images relative to the
    ImageActionTransformer. To be used in combination with ActionLSTM.

    Lower number of encoder layers, since we don't need as many long term dependencies.

    We also implement a maximum sequence length to restrict the amount of images processed.

    """
    def __init__(self, num_encoder_layers=2, nhead=4, d_model=512, dim_feedforward=2048,
                  dropout=0.1, num_actions=3, max_sequence_len=16):
        super(ImageTransformer, self).__init__()
        self.actions = num_actions
        self.max_sequence_len = max_sequence_len
        self.cnn = CNNFeatureExtractor(d_model=d_model)

        # Transformer Layers
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)

        # Output Layer
        self.output_layer = nn.Linear(d_model, self.actions)

    def forward(self, img, cmd):

        # Restrict sequence length. There should be one less cmd than img
        if img.size() > self.max_sequence_len:
            img = img[:, -self.max_sequence_len]
        if cmd.size() > self.max_sequence_len-1:
            cmd = cmd[:, -self.max_sequence_len-1]

        img_features = self.cnn(img)
        cmd_embedding = functional.one_hot(cmd.to(torch.int64), self.actions)

        # Combine image features and command embeddings
        combined_features = torch.cat((img_features, cmd_embedding), dim=1)
        combined_features = combined_features.unsqueeze(0)

        # Pass through transformer
        transformer_output = self.transformer_encoder(combined_features)

        # Pass through a fully connected output layer, which will predict an action
        _, (final_layer, _) = self.output_layer(transformer_output.squeeze(0))

        # Pull out the last hidden layer
        return final_layer[-1]

class HybridModel(nn.Module):
    """
    This model combines both the ImageTransformer and ActionLSTM. It is a more efficient approach
    than just throwing all of the data into a really big transformer. It uses a small transformer
    with the same inner workings as the larger one, but it has a maximum sequence cutoff that limits
    how far back it looks for images. The long term dependencies necessary for exploration are captured 
    by an lstm that only stores the actions that the robot took.

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
    def __init__(self, num_encoder_layers=2, nhead=4, d_model=512, dim_feedforward=2048,
                    dropout=0.1, num_actions=3, max_sequence_len=8, hidden_size_lstm=64,
                    num_layers_lstm=2):
        
        super(HybridModel, self).__init__()
        self.transformer = ImageTransformer(num_encoder_layers=num_encoder_layers, nhead=nhead, 
                                            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, 
                                            num_actions=num_actions, max_sequence_len=max_sequence_len)
        self.lstm = ActionLSTM(num_actions=num_actions, hidden_size=hidden_size_lstm, 
                               num_layers=num_layers_lstm)

        self.output_layer = nn.Linear(d_model + hidden_size_lstm, num_actions)
    
    def forward(self, images, commands):
        # Push data through models
        transformer_out = self.transformer(images, commands)
        lstm_out = self.lstm(commands)

        # Combine
        combined_output = torch.cat((transformer_out, lstm_out), dim=1)
        combined_output = combined_output.unsqueeze(0)

        output = self.output_layer(combined_output)
        
        return output
