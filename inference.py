from argparse import ArgumentParser
from math import radians
from pathlib import Path

from fastai.vision.learner import load_learner
from ue5osc import Communicator


def parse_args():
    arg_parser = ArgumentParser("Track performance of trained networks.")
    arg_parser.add_argument("model", help="Path to the model to evaluate.")
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
        default=radians(10),
        help="Rotation per action.",
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

    model = load_learner(args.model)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Communicator("127.0.0.1", ue_port=7447, py_port=7001) as ue:
        for action_index in range(args.max_actions):
            # Save image
            image_filename = f"{args.output_dir}/{action_index}.png"
            ue.save_image(image_filename)

            # Predict correct action
            predicted_action = model.predict(image_filename)[0]

            # Take action
            match predicted_action:
                case "forward":
                    ue.move_forward(args.movement_amount)
                case "left":
                    ue.rotate_left(args.rotation_amount)
                case "right":
                    ue.rotate_right(args.rotation_amount)
                case _:
                    raise ValueError(f"Unknown action: {predicted_action}")


if __name__ == "__main__":
    main()
