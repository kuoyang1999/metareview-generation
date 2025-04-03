import logging
import os
import wandb
from datetime import datetime

def init_wandb(training_args, config, timestamp, data_args):
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
        # Set the W&B base URL if provided in environment
        wandb_base_url = os.getenv("WANDB_BASE_URL")
        if wandb_base_url:
            os.environ["WANDB_BASE_URL"] = wandb_base_url
            logging.info(f"Using custom W&B base URL: {wandb_base_url}")

        # Login with the stored API key from the environment
        if training_args.local_rank == 0:
            wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
            
            # Create run name with conference info if available
            run_name = timestamp
            if data_args.conferences:
                conf_str = '_'.join(data_args.conferences)
                run_name = f"{conf_str}_{timestamp}"
                
            wandb.init(project="meta-review", config=config, name=run_name)
            logging.info("W&B initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize W&B: {e}")
        return False