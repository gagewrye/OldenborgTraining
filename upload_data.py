from argparse import ArgumentParser

import wandb

wandb.login()

arg_parser = ArgumentParser("Upload the given dataset to wandb.")

arg_parser.add_argument("wandb_name", help="Name of run and artifact.")
arg_parser.add_argument("wandb_project", help="Wandb project name.")
arg_parser.add_argument("wandb_notes", help="Wandb run description.")
arg_parser.add_argument("data_dir", help="Directory containing dataset.")

args = arg_parser.parse_args()

run = wandb.init(
    job_type="dataset-upload",
    entity="arcslaboratory",
    project=args.wandb_project,
    notes=args.wandb_notes,
)

if run is None:
    raise Exception("wandb.init() failed")

artifact = wandb.Artifact(name=args.wandb_name, type="dataset")
artifact.add_dir(local_path=args.data_dir, name="data")
run.log_artifact(artifact)
