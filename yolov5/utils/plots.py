# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Plotting utils."""
import math
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.general import LOGGER

# Settings
RANK = int(os.getenv("RANK", -1))
matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only

def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if ("Detect" not in module_type) and (
        "Segment" not in module_type
    ):  # 'Detect' for Object Detect task,'Segment' for Segment task
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            LOGGER.info(f"Saving {f}... ({n}/{channels})")
            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save
