from subprocess import run
import os

model_directory = "models"
port = "9000"
project_folder = "C:\\Users\\ARCS\\Documents\\Unreal Projects"

for model in os.listdir(model_directory):
    print(model)
    run(
        [
            "python",
            "inference.py",
            model,
            port,
            project_folder
        ]
    )