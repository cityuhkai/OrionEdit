# Legacy fields that are still supported for backward compatibility, but not used in the new pipeline.
# Modified from the training script provided by diffusers FLUX.1 dev.

import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


def seed_everything(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    
def seconds_to_human_readable(seconds):
    # Calculate the number of days, hours, minutes, and seconds
    seconds = int(seconds)
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    
    hours = seconds // 3600
    seconds %= 3600
    
    minutes = seconds // 60
    seconds %= 60
    
    # Format the output as "days days, hh:mm:ss"
    time_str = f"{days:02} {hours:02}:{minutes:02}:{seconds:02}"
    
    return time_str

def plot_loss(step_losses,img_dir):
    plt.figure()
    #if len(step_losses) > 10000:
        #step_losses = step_losses[::]
    plt.plot(step_losses,'b',label = 'loss')        
    plt.ylabel('loss')
    plt.xlabel('per 100 step')
    plt.legend()
    plt.savefig(os.path.join(img_dir,"train_loss.jpg")) 

def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(
                    save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)