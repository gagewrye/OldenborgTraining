from subprocess import run

dataset = "/data/clark/data/2ndRunHighRes"

compared_models = [
    "resnet18",
    "xresnext50",
    "xresnext18",
    "alexnet",
    "densenet121",
]

for model in compared_models:
    print(model)
    
    run(
        [
            "python",
            "training.py",
            "--dataset",
            dataset,
            "--model",
            model
        ]
    )
