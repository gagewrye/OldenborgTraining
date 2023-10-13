import pathlib
import platform
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from time import sleep

from fastai.callback.wandb import WandbCallback
from fastai.vision.learner import load_learner
from ue5osc import Communicator

import wandb
from utils import y_from_filename  # noqa: F401 (needed for fastai load_learner)


@contextmanager
def set_posix_windows():
    # NOTE: This is a workaround for a bug in fastai/pathlib on Windows
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def parse_args():
    arg_parser = ArgumentParser("Track performance of trained networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run inference results.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")
    arg_parser.add_argument("wandb_model", help="Path to the model to evaluate.")

    arg_parser.add_argument("output_dir", help="Directory to store saved images.")
    arg_parser.add_argument(
        "--movement_amount",
        type=float,
        default=120.0,
        help="Movement forward per action.",
    )
    arg_parser.add_argument(
        "--rotation_amount",
        type=float,
        default=10.0,
        help="Rotation per action (in degrees for ue5osc).",
    )
    arg_parser.add_argument(
        "--max-actions",
        type=int,
        default=500,
        help="Maximum number of actions to take.",
    )
    return arg_parser.parse_args()


def main():
    args = parse_args()

    wandb_entity = "arcslaboratory"
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    wandb_notes = args.wandb_notes
    wandb_model = args.wandb_model

    run = wandb.init(
        job_type="inference",
        entity=wandb_entity,
        name=wandb_name,
        project=wandb_project,
        notes=wandb_notes,
    )

    if run is None:
        raise Exception("wandb.init() failed")

    # Download the fastai learner
    artifact = run.use_artifact(f"{wandb_model}:latest", type="model")
    model_dir = artifact.download()
    model_filename = Path(model_dir) / wandb_model

    # Load the learner and its model
    # TODO: this doesn't load the "best" model, but the last one
    # We should probably also download the weights and load them manually
    if platform.system() == "Windows":
        with set_posix_windows():
            model = load_learner(model_filename)
    else:
        model = load_learner(model_filename)

    # TODO: temporary fix? (we might remove callback on training side)
    model.remove_cb(WandbCallback)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if output directory is empty
    if any(output_dir.iterdir()):
        print("Output directory is not empty. Aborting.")
        return

    with Communicator("127.0.0.1", ue_port=7447, py_port=7001) as ue:
        print("Connected to", ue.get_project_name())
        print("Saving images to", output_dir)
        ue.reset()

        for action_step in range(args.max_actions):
            # Save image
            image_filename = f"{output_dir}/{action_step:04}.png"
            ue.save_image(image_filename)
            sleep(0.5)

            # Predict correct action
            action_to_take, action_index, action_probs = model.predict(image_filename)
            action_prob = action_probs[action_index]

            print(f"Moving {action_to_take} with probabilities {action_prob:.2f}")

            # Take action
            # TODO: prevent cycling actions (e.g., left followed by right)
            match action_to_take:
                case "forward":
                    ue.move_forward(args.movement_amount)
                case "left":
                    ue.rotate_left(args.rotation_amount)
                case "right":
                    ue.rotate_right(args.rotation_amount)
                case _:
                    raise ValueError(f"Unknown action: {action_to_take}")


if __name__ == "__main__":
    main()
