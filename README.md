# OldenborgModel

Train and perform inference on Oldenborg datasets.

## Windows

For inference on Windows, I had to create an environment with the following:

~~~bash
conda create --name oldenborg
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
mamba install fastai
~~~
