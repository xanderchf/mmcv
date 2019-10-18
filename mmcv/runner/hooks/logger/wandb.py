import os.path as osp

import torch

from ...utils import master_only
from .base import LoggerHook
import numpy as np

PALLETE = np.asarray([
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]], dtype=np.uint8)

class WandBLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 project_name=''):
        super(WandBLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir
        self.project_name = project_name

    @master_only
    def before_run(self, runner):
        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'wandb')
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        wandb.init(project=self.project_name, dir=self.log_dir)

    @master_only
    def log(self, runner):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        for var in runner.log_buffer.output:
            if var.startswith('vis'):
                # log labels
                indices = runner.log_buffer.output[var] + 1
                indices[indices == 256] = 0
                wandb.log({var: [
                    wandb.Image(PALLETE[indices], caption=var)
                ]}, step=runner.iter)
            elif var.startswith('img'):
                # log images
                wandb.log({var: [
                    wandb.Image(np.transpose(runner.log_buffer.output[var], (1, 2, 0)), caption=var)
                ]}, step=runner.iter)
            else:
                # log scalar
                wandb.log({var: runner.log_buffer.output[var]}, step=runner.iter)

    @master_only
    def after_run(self, runner):
        pass
