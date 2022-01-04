#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases 

author: Youhee
date: Jan 2022
"""
import argparse
import logging
import rich
from rich.logging import RichHandler
import wandb

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(
	level=logging.INFO, 
	format=FORMAT, 
	datefmt="[%X]", 
	handlers=[RichHandler()]
)

logger = logging.getLogger("rich")






def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Downloading {args.artifact_name} artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # Drop the duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    
    ######################
    # YOUR CODE HERE     #
    ######################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifacat", 
        type=str,
        help="Fully-qaulified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="The name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=## INSERT TYPE HERE: str, float or int,
        help=## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
