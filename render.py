#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import os
import imageio
import numpy as np
import torch
from scene import Scene
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser, SUPPRESS
from arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams ,ModelHiddenParams
from gaussian_renderer import GaussianModel
from gaussian_renderer import prefilter_voxel, render
from time import time
import threading
import concurrent.futures
from PIL import Image
import json
import torchvision.transforms.functional as TF

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(opt, model_path, name, iteration, views, gaussians, pipeline, background, cam_type, render_blur=False, source_path="", sharp_views=None):
    folder_suffix = "_blurry" if render_blur else ""
    base_path = os.path.join(model_path, name, f"ours_{iteration}{folder_suffix}")
    
    render_path = os.path.join(base_path, "renders")
    gts_path = os.path.join(base_path, "sharp_gt")
    blur_path = os.path.join(base_path, "blurry_gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(blur_path, exist_ok=True)

    render_list = []
    gt_list = []
    blur_list = [] 
    
    sharp_dict = {v.image_name: v for v in sharp_views} if sharp_views else {}
    
    # ==========================================================
    # [核心修复区]：双重 JSON 读取，完美建立 (清晰名 -> 模糊路径) 的映射
    # ==========================================================
    test_blurry_dict = {}
    if name == "test" and source_path:
        json_sharp = os.path.join(source_path, "transforms_test.json")
        json_blur = os.path.join(source_path, "transforms_test_blurry.json")
        
        if os.path.exists(json_sharp) and os.path.exists(json_blur):
            with open(json_sharp, 'r', encoding='utf-8') as fs, open(json_blur, 'r', encoding='utf-8') as fb:
                s_data = json.load(fs)
                b_data = json.load(fb)
                for s_frame, b_frame in zip(s_data.get('frames', []), b_data.get('frames', [])):
                    sharp_fname = os.path.splitext(os.path.basename(s_frame['file_path']))[0]
                    b_path = os.path.join(source_path, b_frame['file_path'])
                    if not os.path.exists(b_path):
                        b_path = os.path.join(source_path, "images", os.path.basename(b_frame['file_path']))
                    
                    test_blurry_dict[sharp_fname] = b_path

    print("point nums:", gaussians._anchor.shape[0])

    t_list = [] # 创建一个时间记录表

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
    
        # ======================== [纯渲染耗时诊断模式] ========================
        # 1. 清空显存占用统计，准备开始计时
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        time1 = time()
        
        # 2. 核心渲染过程
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)  
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        rendering = render(view, gaussians, pipeline, background, stage="fine", 
                           visible_mask=voxel_visible_mask, retain_grad=retain_grad, 
                           force_deblur=render_blur)["render"]
        
        # 3. 渲染结束，立即同步计时
        torch.cuda.synchronize()
        time2 = time()
        
        # 4. 打印该帧的 GPU 渲染时间 (仅在前 10 帧显示，避免刷屏)
        if idx < 10:
            print(f"Frame {idx} Pure Render Time: {(time2 - time1) * 1000:.2f} ms")
        
        t_list.append(time2 - time1)
        render_list.append(rendering) # 保留以便后续保存
        # =========================================================================

        # 以下是你原本完好保留的业务逻辑，用于准备保存的图像数据
        if cam_type != "PanopticSports":
            original_img = view.original_image[0:3, :, :]
        else:
            original_img = view['image'].cuda()
            
        if name == "train":
            blur_list.append(original_img)
            if view.image_name in sharp_dict:
                s_view = sharp_dict[view.image_name]
                sharp_img = s_view.original_image[0:3, :, :] if cam_type != "PanopticSports" else s_view['image'].cuda()
                gt_list.append(sharp_img)
            else:
                gt_list.append(original_img)
        elif name == "test":
            gt_list.append(original_img)
            
            if view.image_name in test_blurry_dict:
                try:
                    b_path = test_blurry_dict[view.image_name]
                    b_img = Image.open(b_path).convert("RGB")
                    blur_list.append(TF.to_tensor(b_img).cuda())
                except Exception as e:
                    print(f"   [WARN] 读图失败: {e}")
                    blur_list.append(original_img) 
            else:
                blur_list.append(original_img)

    # 循环结束后，切片抛弃前 5 帧 Warm-up，计算平均值
    t = np.array(t_list[5:])
    if len(t) > 0:
        fps = 1.0 / t.mean()
        print(f'[{name}] Method FPS (w/o warmup): \033[1;35m{fps:.5f}\033[0m')

    print(f"writing {name} renders...")
    multithread_write(render_list, render_path)
    print(f"writing {name} sharp GTs...")
    multithread_write(gt_list, gts_path)
    print(f"writing {name} blurry GTs...")
    multithread_write(blur_list, blur_path)


def render_sets( opt , hyper, dataset : ModelParams, frames_start_end, iteration : int, pipeline : PipelineParams,  skip_train : bool, skip_test : bool, skip_video: bool, render_blur: bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(hyper, opt,dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              use_deblur=dataset.use_deblur)
        
        scene = Scene(dataset, gaussians, frames_start_end = frames_start_end, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        cam_type=scene.dataset_type
        bg_color = [1,1,1]  if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(opt, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, cam_type, render_blur, source_path=dataset.source_path, sharp_views=scene.getVideoCameras())
        if not skip_test:
            render_set(opt, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, cam_type, render_blur, source_path=dataset.source_path)
        if not skip_video:
            render_set(opt, dataset.model_path, "video2", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, cam_type, render_blur)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--render_blur", action="store_true", default=False,
                        help="Enable physical DoF rendering at inference using camera focal, aperture, and focus metadata.")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--frames_start_end", type=int, nargs=2, default=[0, 300])
    parser.add_argument("--mid_focal", action="store_true")
    parser.add_argument("--mid_time", type=float, default=0.5)

    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.general_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    safe_state(args.quiet)

    render_sets(op.extract(args), hp.extract(args),  model.extract(args), args.frames_start_end, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.render_blur)
