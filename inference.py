from time import sleep
from fastai.vision.all import *
from ue5env import UE5EnvWrapper
from pathlib import WindowsPath
from math import radians
from argparse import ArgumentParser


def get_action_from_filename(filename):
    return filename.split("_")[0]


def main():
    env = UE5EnvWrapper(port=8500)
    # TODO make this work
    # learn = create_vision_model(resnet18, n_out=3)
    learn = load_learner(f"./models/{args.model}")
    # path to where UE5 saves images
    image_path = args.path_to_unreal_image

    movement_increment = 50
    rotation_increment = radians(5)
    while env.is_connected():
        env.save_image(0)
        clss, clss_idx, probs = learn.predict(image_path)
        print(clss)
        if clss == "right":
            env.right(rotation_increment)
        elif clss == "left":
            env.left(rotation_increment)
        elif clss == "forward":
            env.forward(rotation_increment)
        elif clss == "back":
            env.back(movement_increment)
        sleep(2)


argparser = ArgumentParser("Run inference on model")
argparser.add_argument("model", type=str, help="name of model to load")
argparser.add_argument(
    "path_to_unreal_image", type=str, help="path to photo unreal saved"
)
args = argparser.parse_args()

if __name__ == "__main__":
    main()
