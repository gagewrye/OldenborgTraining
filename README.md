# OldenborgModel

Train and perform inference on Oldenborg datasets.

## Workflow

1. Generate data using some other tools (e.g., [`BoxNav`](https://github.com/arcslaboratory/boxnav/)).
2. Upload data using `upload_data.py`.
3. Train model using `train.py`.
4. Perform inference using `inference.py`.

For example,

~~~bash
# Runs the navigator in Python and Unreal Engine and generates a dataset
python boxsim.py wandering --save_images DATA_PATH

# Uploads the dataset to the server
python upload_data.py wandering-data wandering-data-project "Wandering data to..." DATA_PATH

# Trains the model
python training.py wandering-model wandering-data-project "Wandering model to..." resnet18 wandering-data

# Performs inference
python inference.py ...
~~~

## Windows

For inference on Windows, I had to create an environment with the following:

~~~bash
conda create --name oldenborg
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
mamba install fastai
~~~
