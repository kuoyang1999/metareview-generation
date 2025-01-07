import logging
import os
import wandb
from datetime import datetime

def init_wandb(training_args, config, timestamp):
    """
    Attempt to login and initialize W&B if WANDB_API_KEY is present
    and if training_args.use_wandb is True.
    """
    if not training_args.use_wandb:
        logging.info("W&B logging is disabled via --use_wandb=False.")
        return False

    if not os.getenv("WANDB_API_KEY"):
        logging.warning("WANDB_API_KEY not found in environment. Skipping W&B init.")
        return False

    try:
        # Login with the stored API key from the environment
        wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
        wandb.init(project="meta-review", config=config, name=f"{timestamp}", dir="/logs")
        logging.info("W&B initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize W&B: {e}")
        return False