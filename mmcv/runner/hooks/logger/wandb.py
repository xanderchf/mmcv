import os.path as osp

import torch

from ...utils import master_only
from .base import LoggerHook
import numpy as np
import matplotlib.pyplot as plt
from ....image import imdenormalize
import matplotlib.patches as mpatches


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

def tensor2imgs(tensor, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True):
    num_imgs = tensor.shape[0]
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = tensor.transpose(1, 2, 0)
    return imdenormalize(
        img, mean, std, to_bgr=to_rgb).astype(np.uint8).transpose(2,0,1)

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

    def draw_bbox(self, img, bboxes):
        shape = img.shape
        fig = plt.figure(figsize=(16, 9.2), dpi=80)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        ax.axis('off')
        plt.imshow(img)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.data.cpu().numpy()
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='r', facecolor='none',
                fill=False, alpha=0.75
            ))
        fig.canvas.draw_idle()
        return np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), np.uint8).reshape((736, 1280, 4))

    @master_only
    def log(self, runner):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        for var in runner.log_buffer.output:
            if var.startswith('vis'):
                if 'info' in var:
                    continue
                # log labels
                indices = runner.log_buffer.output[var] + 1
                indices[indices == 256] = 0
                wandb.log({var: [
                    wandb.Image(PALLETE[indices], caption=var)
                ]}, step=runner.iter)
            elif var.startswith('img'):
                # log images
                img = tensor2imgs(runner.log_buffer.output[var]).transpose(1, 2, 0)
                # detection
                # if 'vis_det_gt_info' in runner.log_buffer.output:
                #     img = self.draw_bbox(img, runner.log_buffer.output['vis_det_gt_info'][0])


                wandb.log({var: [
                    wandb.Image(img, caption=var)
                ]}, step=runner.iter)
            else:
                # log scalar
                wandb.log({var: runner.log_buffer.output[var]}, step=runner.iter)

    @master_only
    def after_run(self, runner):
        pass
