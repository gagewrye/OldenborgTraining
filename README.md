# OldenborgModel

Don't save model in the repository.

## Dependencies
Must have fastai installed.

## Training Models

Specify the data you want to train on and any models you want to train in [run_training.py](https://github.com/christymarc/OldenborgTraining/blob/inference_training/run_training.py).

To train all models listed in the run_training.py script, run the following:

~~~bash
python run_training.py
~~~

## Inference

After copying over trained models into a folder called "models" in the cloned OldenborgTraining repository, run the following command to test:

~~~bash
python inference.py <model> <port> <path to folder where all UE5 projects are stored>
~~~
