import random
import numpy as np
import torch
import logging
import os

def fix_randomness(seed):
    """Fix the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # To achieve deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def starting_logs(dataset, da_method, exp_log_dir, src_id, trg_id, run_id):
    """Set up logging and directories for your experiment."""
    scenario = f"{src_id}_to_{trg_id}_run_{run_id}"
    scenario_log_dir = os.path.join(exp_log_dir, scenario)
    os.makedirs(scenario_log_dir, exist_ok=True)

    logger = logging.getLogger(scenario)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(scenario_log_dir, 'log.txt'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Adding a handler to a logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Starting new scenario: {scenario}")
    logger.info(f"Dataset: {dataset}, DA method: {da_method}")

    return logger, scenario_log_dir

class AverageMeter:
    """Calculates and stores average and current values ​​for tracking metrics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
        else:
            self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
