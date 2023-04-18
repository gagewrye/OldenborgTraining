from time import sleep
from fastai.vision.all import *
from ue5env import UE5EnvWrapper

import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import sys

from math import radians
from argparse import ArgumentParser

import time


def get_action_from_filename(filename):
    return filename.split("_")[0]


def main():
    argparser = ArgumentParser("Run inference on model")
    argparser.add_argument("model", type=str, help="name of model to load")
    argparser.add_argument("port", type=int, help="port for Unreal Engine 5 connection")
    argparser.add_argument(
        "path_to_projects", type=str, help="path to where Unreal Engine 5 projects are stored"
    )
    args = argparser.parse_args()

    env = UE5EnvWrapper(port=args.port)
    # TODO make this work
    # learn = create_vision_model(resnet18, n_out=3)
    learn = load_learner(f"./models/{args.model}")

    # path to where UE5 saves projects
    project_path = args.path_to_projects
    # get the connected project's name
    project_name = env.get_project_name()
    # path in project folder to saved folder
    project_saved_path = project_name + "\\Saved"
    # path in project folder to screenshots according to default configurations
    screenshot_folder = "Screenshots"
    project_screenshot_path = project_saved_path + "\\" + screenshot_folder

    image_folder = os.path.join(project_path, project_screenshot_path)
    if (os.path.isdir(image_folder)):
        sys.exit("ERROR: The \"Screenshots\" folder exists in the Saved directory of UE5 connected project file, please delete this folder and run again.")

    # movement_increment = 50
    rotation_increment = radians(5)
    image_num = 0
    while env.is_connected():
        env.save_image(0)
        image_path = os.path.join(image_folder, 'WindowsEditor\HighresScreenshot{num:0{width}}.png'.format(num=image_num, width=5))
        print(image_path)
        clss, clss_idx, probs = learn.predict(image_path)
        print(clss)
        if clss == "right":
            env.right(rotation_increment)
        elif clss == "left":
            env.left(rotation_increment)
        elif clss == "forward":
            env.forward()
        elif clss == "back":
            env.back()
        image_num += 1
        if image_num == 3: break
        sleep(1)

    os.chdir(os.path.join(project_path, project_saved_path))
    current_time = time.localtime()
    new_folder_name = screenshot_folder + time.strftime("-%Y-%m-%d-%H%M", current_time)
    os.rename(screenshot_folder, pathlib.Path(new_folder_name))
    print(f"Inference images saved to: {project_saved_path}\\{new_folder_name}")
        

if __name__ == "__main__":
    main()
