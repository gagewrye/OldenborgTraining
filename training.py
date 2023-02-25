from fastai.vision.all import *
from pathlib import Path
from argparse import ArgumentParser
import os


def get_action_from_filename(filename):
    return filename.split("_")[0]


def main():
    arg_parser = ArgumentParser("Train a model")
    arg_parser.add_argument("dataset", type=str, help="dataset location")
    args = arg_parser.parse_args()

    dataset_path = Path(args.dataset)
    path = Path(os.getcwd())
    filenames = get_image_files(dataset_path)
    dls = ImageDataLoaders.from_name_func(
        path,
        valid_pct=0.2,
        item_tfms=Resize(224),
        bs=32,
        label_func=get_action_from_filename,
        fnames=filenames,
    )

    print("Validation dataset size:", len(dls.valid_ds))
    print("Training dataset size:", len(dls.train_ds))

    learn = vision_learner(dls, resnet18, metrics=accuracy)

    learn.fine_tune(5)

    learn.export("./models/testModel.pkl")


if __name__ == "__main__":
    main()
