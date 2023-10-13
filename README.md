# OldenborgModel

Train and perform inference on Oldenborg datasets.

## Workflow

1. Generate data using some other tools (e.g., [`BoxNav`](https://github.com/arcslaboratory/boxnav/)).
2. Upload data using `upload_data.py`.
3. Train model using `training.py`.
4. Perform inference using `inference.py`.

For example,

~~~bash
# Runs the navigator in Python and Unreal Engine and generates a dataset
# This will run on a system that can run Unreal Engine
python boxsim.py wandering --save_images data/

# Uploads the dataset to the server
# You can upload from wherever the data is generated (probably the same system as above)
python upload_data.py PerfectStaticData TestingWorkflow "I am using this project to test the upload, train, then inference workflow." ../scr2023/data/PerfectStaticTextures/

# Trains the model
# This should be run on a system with a GPU (e.g., our server)
python training.py PerfectStaticModel TestingWorkflow "Testing training..." resnet18 PerfectStaticData

# Performs inference
# This will run on a system that can run Unreal Engine
python inference.py PerfectStaticInference TestingWorkflow "Testing inference..." PerfectStaticModel-resnet18-PerfectStaticData-rep00 InferenceImages
~~~

## Windows

For inference on Windows, I had to create an environment with the following:

~~~bash
conda create --name oldenborg
conda activate oldenborg
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
mamba install fastai
~~~
