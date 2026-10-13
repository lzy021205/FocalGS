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

import os
import numpy as np

from utils.timer import Timer

from torch.utils.data import DataLoader
from utils.loader_utils import FineSampler, get_stamp_list
import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, LikelihoodLoss, factor_loss
from gaussian_renderer import prefilter_voxel, render , network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import math
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace, SUPPRESS
from arguments import ModelParams, PipelineParams, OptimizationParams,ModelHiddenParams
import matplotlib.pyplot as plt
# ================= [新增 Import] =================
from utils.loss_utils import SmoothLoss
from utils.graphics_utils import inverse_warp_images
# =================================================

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def get_shifted_camera(cam, trans_dist):
    import copy
    import torch
    new_cam = copy.deepcopy(cam)
    new_W2C = new_cam.world_view_transform.clone()
    new_W2C[3, 0] += trans_dist  # 沿 X 轴平移
    new_cam.world_view_transform = new_W2C
    new_cam.full_proj_transform = torch.matmul(new_W2C.unsqueeze(0), new_cam.projection_matrix.unsqueeze(0)).squeeze(0)
    new_cam.camera_center = new_cam.world_view_transform.inverse()[3, :3]
    return new_cam
def draw_img(img, idx):
    ldr = img.detach().cpu()
    img_tan = ldr.reshape(-1)
    plt.hist(np.array(img_tan), bins=50)
    plt.xlabel('Max_value',fontsize=18)
    plt.ylabel('Images',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"ldr_{idx}.png",format='png')
    plt.close()

def show_depth(image,name):
    import os  # 确保 os 模块可用
    os.makedirs('test_images', exist_ok=True)  # <--- 【新增】如果文件夹不存在，就自动建一个！
    depth_max = torch.max(image)
    depth_min = torch.min(image)
    image = (image - depth_min) / (depth_max - depth_min + 0.0001)
    show_image = Image.fromarray(np.transpose(np.array(image.tile(3,1,1).detach().clamp(0,1).cpu() * 255 ).astype(np.uint8), (1, 2, 0)))
    show_image.save(f'test_images/rgb_{name}.png')

def show_image(image,name):
    from PIL import Image
    os.makedirs('test_images', exist_ok=True)
    show_image = Image.fromarray(np.transpose(np.array(image.detach().clamp(0,1).cpu() * 255 ).astype(np.uint8), (1, 2, 0)))
    show_image.save(f'test_images/rgb_{name}.png')

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')



def training(dataset, hyper, opt, pipe, frames_start_end, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname,args):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)

    # [核心修改] 显式传递 use_deblur 开关
    gaussians = GaussianModel( hyper, opt,dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              use_deblur=dataset.use_deblur) # <--- [修改点]

    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, frames_start_end)
    timer.start()


    scene_reconstruction(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer, args)
                         
    if tb_writer:
        tb_writer.close()
    



def scene_reconstruction(dataset, hyper, opt,  pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer, args):
    first_iter = 0

    gaussians.training_setup(opt,stage)


    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, train_iter), desc="Training progress")
    first_iter += 1
    # ================= [新增：双目 Loss 全局初始化] =================
    # Create tensors for binocular consistency
    train_cams = scene.getTrainCameras()
    image_height = train_cams[0].image_height
    image_width = train_cams[0].image_width
    row_indices = torch.arange(0, image_height).view(-1, 1).repeat(1, image_width).cuda()
    column_indices = torch.arange(0, image_width).repeat(image_height, 1).cuda()
    warp_mask = torch.ones((1, image_height, image_width), dtype=torch.float32).cuda()

    smooth_loss = SmoothLoss()
    # ================================================================
    # ================= [新增 1：焦距专属优化器初始化] =================
    focus_params = []
    for cam in scene.getTrainCameras():
        if hasattr(cam, 'learn_focal') and cam.learn_focal and cam.focus_raw is not None:
            focus_params.append(cam.focus_raw)
    
    if len(focus_params) > 0:
        # 学习率设为 1e-2，让它能较快地在物理区间内滑动寻优
        optimizer_focus = torch.optim.Adam([{'params': focus_params, 'lr': 1e-2, 'name': 'focal_dist'}])
    else:
        optimizer_focus = None
    # =================================================================
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        import copy
        train_cams = scene.getTrainCameras()
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=1,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    if stage == "coarse" and opt.zerostamp_init :
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack, 0) # 20
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False

    last_iteration = 0
    cur_index = 0
    for iteration in range(first_iter, train_iter + 1):
        # network gui not available in scaffold-gs yet
        # if network_gui.conn == None:
        #     network_gui.try_connect()

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # dynerf's branch
        if  opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                #print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []
            while idx < opt.batch_size :
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue


        viewpoint_cam = viewpoint_cams[0]
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)  #
        retain_grad = (iteration < opt.update_until and iteration >= 0)
# ================= [新增代码 1：Warm-up 策略] =================
        # 如果开启了去模糊，且在前 3000 次迭代（预热期）
        # 强制临时关闭去模糊功能，让 3DGS 专注于生长几何结构
        # 这样绿叶就会尝试用很多小点去拟合，而不是依赖 GTnet
        #if dataset.use_deblur:
         #   if iteration < 3000:
          #      gaussians.use_deblur = False
           # else:
            #    gaussians.use_deblur = True
        # ============================================================
        
        
        # [核心修改 1] 训练时开启 is_training 标志 (Render 函数已自动读取 pc.get_color_mlp.training)
        render_pkg = render( viewpoint_cam, gaussians, pipe, background, iteration=iteration, stage=stage, visible_mask=voxel_visible_mask, retain_grad=retain_grad)

        image, depth_map, viewspace_point_tensor, visibility_filter, offset_selection_mask, scaling, opacity, neural_points=                 render_pkg["render"], render_pkg["depth_map"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"],                     render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["neural_points"]
        
        base_scaling = render_pkg.get("base_scaling", render_pkg["scaling"])
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image)[0])
        
        psnr_ = psnr(image, gt_image).mean().double()
        #[核心修改：双重惩罚机制]
        if base_scaling.numel() > 0:
            # 1. 主惩罚：基础几何体积（系数大，逼迫底模极致 Sharp）
            base_scaling_reg = base_scaling.prod(dim=1).mean()
            # 2. 副惩罚：最终膨胀体积（系数小，像皮筋一样防止 GTnet 梯度爆炸野蛮生长）
            final_scaling_reg = scaling.prod(dim=1).mean()
        else:
            base_scaling_reg = 0.0
            final_scaling_reg = 0.0

        
        # ================= [新增：Opacity Decay (透明度惩罚)] =================
        # 利用 Loss 正则化来实现 Scaffold-GS 架构下等效的“透明度衰减”
        opacity_reg = 0.0
        # 建议在 3000 步（预热结束、基础几何刚开始成型时）启动“扣血”机制
        if iteration > 3000 and opacity.numel() > 0:
            opacity_reg = opacity.mean()
        # =====================================================================

        # 将 opacity_reg 汇入总 Loss。
        # 这里的 0.01 权重相当于每次给透明度施加一个温和但坚定的向下的梯度（等效于乘 0.995）
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + \
               0.01 * base_scaling_reg + 0* final_scaling_reg + \
               args.opacity_reg_weight * opacity_reg
                 
        anisotropy_loss = 0.0
        lambda_aniso = getattr(args, "lambda_aniso", 0.0)
        if lambda_aniso > 0.0 and scaling.numel() > 0:
            max_scales, _ = torch.max(scaling, dim=1)
            min_scales, _ = torch.min(scaling, dim=1)
            anisotropy_ratio = max_scales / (min_scales + 1e-7)
            anisotropy_loss = torch.mean(torch.nn.functional.relu(anisotropy_ratio - 3.0))
            loss = loss + lambda_aniso * anisotropy_loss
        
        
                
# ================= [全新替换：DNGaussian 双目一致性 Loss] =================
        disparity_loss = 0.0
        
        # 核心策略：延迟介入！前 15000 步让网络安心长结构，后面再用双目施加刚性约束！
        shift_cam_start = args.shift_cam_start 
        
        if args.enable_shifted_disparity_loss and iteration > shift_cam_start:
            # 1. 随机平移相机
            cam_trans_dist = 0.4
            trans_dist = torch.rand(1) * cam_trans_dist
            import random
            trans_dist = (trans_dist * random.choice([-1.0, 1.0])).item()
            
            shifted_cam = get_shifted_camera(viewpoint_cam, trans_dist) # 确保你有这个辅助函数
            
            # 2. 渲染平移画面
            with torch.no_grad():
                render_pkg_shifted = render(shifted_cam, gaussians, pipe, background, iteration=iteration, stage=stage, visible_mask=voxel_visible_mask)
                shifted_image = render_pkg_shifted["render"]
            
            # 3. 提取深度并升维到 4D [1, 1, H, W]
            # depth_map 出来通常是 [1, H, W]
            depth = depth_map.unsqueeze(0) 
            
            # 4. 计算视差 (保持 4D 结构 [1, 1, H, W])
            focal_x = viewpoint_cam.image_width / (2.0 * math.tan(viewpoint_cam.FoVx * 0.5))
            disparity = focal_x * (-trans_dist) / (depth + 1e-5)
            
            # 5. 为了传给 warp 函数，把 4D 降回 3D 或者适应 warp 函数
            # 根据我之前给你的 inverse_warp_images，它能接受 image 为 [1, 3, H, W], disparity 为 [1, 1, H, W]
            gt_image_4d = gt_image.unsqueeze(0)        # [1, 3, H, W]
            shifted_img_4d = shifted_image.unsqueeze(0)# [1, 3, H, W]
            warp_mask_4d = warp_mask.unsqueeze(0)      # [1, 1, H, W] 假设 warp_mask 初始是 [1, H, W]
            
            # 6. 进行视差扭曲
            warped_image = inverse_warp_images(shifted_img_4d, disparity, row_indices, column_indices)
            shift_mask = inverse_warp_images(warp_mask_4d, disparity, row_indices, column_indices)
            
# 7. 调用官方 SmoothLoss，并将 mask 提前乘入图像，完美适配普通 l1_loss！
            disparity_loss = (l1_loss(warped_image * shift_mask, gt_image_4d * shift_mask) +
                              0.05 * smooth_loss(disparity * shift_mask, gt_image_4d))
            
            loss += disparity_loss
            
            if iteration % 1000 == 0:
                print(f"[Debug] Iter {iteration} | Disparity Loss: {disparity_loss.item():.4f}")
        # ==========================================================================
        
        if stage == "fine" and hyper.time_smoothness_weight != 0 and not opt.hash :
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss


        loss.backward()
        # ================= [新增 2：物理焦距梯度接续与转换] =================
        if optimizer_focus is not None and hasattr(viewpoint_cam, 'learn_focal') and viewpoint_cam.learn_focal:
            # 等待从 render_pkg 中提取未来 CUDA 返回的梯度
            grad_focus_from_cuda = render_pkg.get("grad_focus", None)
            if grad_focus_from_cuda is not None:
                # 链式法则求导：d(D_focus) / d(focus_raw)
                w = torch.sigmoid(viewpoint_cam.focus_raw)
                d_focus_d_raw = (viewpoint_cam.D_max - viewpoint_cam.D_min) * w * (1.0 - w)
                
                # 将底层物理梯度转化为网络参数的梯度，并挂载到 autograd 引擎中
                viewpoint_cam.focus_raw.grad = grad_focus_from_cuda * d_focus_d_raw
        # ====================================================================
        for param in gaussians.dynamic_module.parameters():
            if param.grad is not None:
                param.grad.nan_to_num_()

        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            return
            # os.execv(sys.executable, [sys.executable] + sys.argv)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            total_point = gaussians.get_anchor.shape[0]
            if iteration % 10 == 0 :
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}" ,
                                          "psnr": f"{psnr_:.{2}f}" ,
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            
                # ================= [新增 TensorBoard 记录：训练曲线] =================
                if tb_writer:
                    # 1. 核心 Loss 曲线
                    tb_writer.add_scalar('Train/Total_Loss', loss.item(), iteration)
                    tb_writer.add_scalar('Train/L1_Loss', Ll1.item(), iteration)
                    tb_writer.add_scalar('Train/SSIM_Loss', ssim_loss.item(), iteration)
                    tb_writer.add_scalar('Train/PSNR', psnr_.item(), iteration)
                    
                    # 2. Penalty monitoring (Check if regularization works)
                    
                    # Log Base Scaling Reg (Main constraint)
                    if isinstance(base_scaling_reg, torch.Tensor):
                        tb_writer.add_scalar('Penalty/Base_Scaling_Reg', base_scaling_reg.item(), iteration)
                    else:
                        tb_writer.add_scalar('Penalty/Base_Scaling_Reg', base_scaling_reg, iteration)
                        
                    # Log Final Scaling Reg (Sub constraint)
                    if isinstance(final_scaling_reg, torch.Tensor):
                        tb_writer.add_scalar('Penalty/Final_Scaling_Reg', final_scaling_reg.item(), iteration)
                    else:
                        tb_writer.add_scalar('Penalty/Final_Scaling_Reg', final_scaling_reg, iteration)
                        
                    if isinstance(opacity_reg, torch.Tensor):
                        tb_writer.add_scalar('Penalty/Opacity_Reg', opacity_reg.item(), iteration)
                    else:
                        tb_writer.add_scalar('Penalty/Opacity_Reg', opacity_reg, iteration)

                    if isinstance(disparity_loss, torch.Tensor):
                        tb_writer.add_scalar('Penalty/Disparity_Loss', disparity_loss.item(), iteration)
                    elif disparity_loss != 0.0:
                        tb_writer.add_scalar('Penalty/Disparity_Loss', disparity_loss, iteration)
                        
                    # 3. 几何生长监控 (极其关键：监控是否点数爆炸)
                    tb_writer.add_scalar('Geometry/Anchor_Count', total_point, iteration)
                    if neural_points is not None:
                        tb_writer.add_scalar('Geometry/Neural_Gaussian_Count', neural_points.shape[0], iteration)
                # =====================================================================

            if iteration == train_iter:
                progress_bar.close()

            if stage == "fine" :
                training_report(args, gaussians, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),                                 testing_iterations, scene, render, [ pipe , background  ], stage, scene.dataset_type, logger)

            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.update_until and iteration > opt.start_stat : #   and stage == "fine":
                # add statis
                gaussians.training_statis(  neural_points ,viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                # densification
                if iteration >= opt.update_from  :  # and iteration % opt.update_interval == 0 :  # 1500 开始致密化
                    gaussians.adjust_anchor(iteration, check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)



            if iteration % 100 == 0 :
                torch.cuda.empty_cache()

            elif iteration == opt.update_until:  # 15000 删除缓存
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                del gaussians.grad_max
                del gaussians.grad_max_points
                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < train_iter :
                gaussians.optimizer.step()
                gaussians.dy_optimizer.step()
                gaussians.dy_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.zero_grad(set_to_none = True)
                # ================= [新增 3：焦距优化器步进] =================
                if optimizer_focus is not None:
                    optimizer_focus.step()
                    optimizer_focus.zero_grad(set_to_none=True)
                # ==========================================================
        
        if iteration % 500 == 0:
    # 拿到经过 exp 激活的真实物理尺度，并提取前 3 维的最大值
           max_scale = gaussians.get_scaling[:, :3].max().item()
           mean_scale = gaussians.get_scaling[:, :3].mean().item()
           print(f"\n[Debug] Iter {iteration} | Max Anchor Offset Bound: {max_scale:.6f} | Mean: {mean_scale:.6f}") 
           
           # ================= [新增 TensorBoard 记录：尺度监控] =================
           if tb_writer:
                tb_writer.add_scalar('Stats/Max_Anchor_Scale', max_scale, iteration)
                tb_writer.add_scalar('Stats/Mean_Anchor_Scale', mean_scale, iteration)
            # =====================================================================





def prepare_output_and_logger(expname):

    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        os.makedirs(args.model_path+"/logs/", exist_ok = True)
        tb_writer = SummaryWriter(args.model_path +"/logs/" )
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(args , gaussians, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,                     scene : Scene, renderFunc, renderArgs, stage, dataset_type, logger):

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        #
        validation_configs = (
            # {'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
            {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras())) ] },
            # {'name': 'test', 'cameras' : [scene.getTestCameras()[0]  ] },
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, gaussians, renderArgs[0], renderArgs[1])  # 判断是否可见

                    # 1. 先进行渲染，获取完整的打包数据 (保存为 render_result)
                    render_result = renderFunc(viewpoint, scene.gaussians,stage=stage, iteration = iteration, visible_mask = voxel_visible_mask, retain_grad = True, *renderArgs)
                    
                    # 2. 提取并裁剪 RGB 图像用于算 Loss
                    image = torch.clamp(render_result["render"], 0.0, 1.0)

                    # 3. 打印信息并提取深度图 (放在渲染之后执行！)
                    if config['name'] == 'test' and idx == 0: 
                        print(f"\n================ [FOV Check] ================")
                        print(f"Test Camera Name: {viewpoint.image_name}")
                        print(f"Test FOV X: {viewpoint.FoVx:.4f}")
                        print(f"Test FOV Y: {viewpoint.FoVy:.4f}")
                        print(f"Test Camera Center:\n{viewpoint.camera_center}")
                        print(f"Test Timestamp: {viewpoint.time:.4f}")
                        print(f"=============================================\n")
                        
                        # 从我们刚刚获取的 render_result 中提取深度
                        depth_map = render_result["depth_map"]
                        show_depth(depth_map, f"iter_{iteration}")

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    psnr_ = psnr(image, gt_image).mean().double()
                    psnr_test += psnr_
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                logger.info("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))
                
               # ================= [新增 TensorBoard 记录：验证集性能] =================
                if tb_writer:
                    # 区分这是 test 集合还是 train 集合的评测
                    prefix = "Test" if config['name'] == 'test' else "Train_Eval"
                    tb_writer.add_scalar(f'{prefix}/PSNR', psnr_test, iteration)
                    tb_writer.add_scalar(f'{prefix}/L1_Loss', l1_test, iteration)
                # =====================================================================


        torch.cuda.empty_cache()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize();t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)

        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)

        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })

    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)

            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)

        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--frames_start_end", type=int, nargs=2, default=[0, 20], help="Start and end frames")
    parser.add_argument("--shift_cam_start", type=int, default=15000)
    parser.add_argument("--enable_shifted_disparity_loss", action="store_true", default=False)
    parser.add_argument("--opacity_reg_weight", type=float, default=0.01)
    parser.add_argument("--lambda_aniso", type=float, default=0.0)
    parser.add_argument("--use_learning_focus", action="store_true", default=False, help=SUPPRESS)



    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        import mmcv
        from utils.general_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    args.save_iterations = list(dict.fromkeys(args.save_iterations))
    # enable logging

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)
    logger.info(f'args: {args}')


    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.frames_start_end, args.test_iterations, args.save_iterations,              args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.model_path, args)
