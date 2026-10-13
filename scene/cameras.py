#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    # 【修改点 1】：在参数列表末尾加上 learn_focal 开关和焦距边界 D_min, D_max
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, f_stop, focal_dist_m, focal_length_mm, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", time=0,
                 mask=None, depth=None, learn_focal=False, D_min=None, D_max=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time

        # ================= [严格模式：核心光学量纲换算] =================
        self.focal_dist_m = float(focal_dist_m)
        f_m = float(focal_length_mm) / 1000.0 
        self.lens_focal_meters = f_m
        
        # 拆掉 1e-5 的保护！如果光圈值 f_stop 给的是 0，直接触发 ZeroDivisionError 报错！
        self.aperture_m = f_m / float(f_stop)
        # ================================================================

        # ================= [新增：动态可学习物理焦距机制] =================
        self.learn_focal = learn_focal
        self.gt_focus_dist = float(focal_dist_m) # 永远保存真实的/初始的焦距作为GT

        if self.learn_focal and D_min is not None and D_max is not None:
            self.D_min = float(D_min)
            self.D_max = float(D_max)
            # 初始化一个可学习的参数。0.0 经过 sigmoid 就是 0.5，恰好在区间正中间
            self.focus_raw = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device="cuda"))
        else:
            # 合成数据集模式下，不开辟内存，不进行求导
            self.focus_raw = None
        # ================================================================

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.original_image = image.clamp(0.0, 1.0)[:3,:,:]
        # breakpoint()
        # .to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
            # .to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
                                                #   , device=self.data_device)
        # self.original_image = self.original_image.cuda()
        self.depth = depth
        self.mask = mask
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # .cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # 【修改点 2】：新增动态获取焦距的接口，供前向渲染使用
    def get_dynamic_focus(self):
        if self.learn_focal and self.focus_raw is not None:
            # 将无界参数通过 Sigmoid 严格束缚在物理区间内
            w = torch.sigmoid(self.focus_raw)
            return self.D_min + w * (self.D_max - self.D_min)
        else:
            # 合成数据集（或未开启学习）：直接雷打不动地返回 GT
            return self.gt_focus_dist


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height  
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time