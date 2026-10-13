#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips 
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import sys
import numpy as np

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if not fname.endswith(('.png', '.jpg', '.jpeg')):
            continue
        render = Image.open(os.path.join(renders_dir, fname))
        gt = Image.open(os.path.join(gt_dir, fname))
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate_(model_path, iteration=30000):
    test_root = os.path.join(model_path, "test")
    if not os.path.exists(test_root):
        print(f"[Error] Test directory not found: {test_root}")
        return

    target_dir = os.path.join(test_root, f"ours_{iteration}")
    if not os.path.exists(target_dir):
        print(f"[Error] Render folder not found: {target_dir}")
        return

    print(f"Selecting render folder: {target_dir}")

    gt_dir = os.path.join(target_dir, "sharp_gt")
    renders_dir = os.path.join(target_dir, "renders")

    renders, gts, image_names = readImages(renders_dir, gt_dir)
    
    print("Initializing LPIPS (VGG)...")
    lpips_fn = lpips.LPIPS(net='vgg').to("cuda")

    psnrs = []
    ssims = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Calculating Metrics"):
        p = psnr(renders[idx], gts[idx])
        psnrs.append(p.item())

        s_out = ssim(renders[idx], gts[idx])
        if isinstance(s_out, (tuple, list)):
            s_val = s_out[0]
        else:
            s_val = s_out
        
        if torch.is_tensor(s_val):
            s_val = s_val.item()
        ssims.append(s_val)

        l = lpips_fn(renders[idx], gts[idx])
        lpipss.append(l.item())

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ssim = torch.tensor(ssims).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()

    print("\n" + "="*30)
    print(f"  Avg PSNR  : {avg_psnr:.4f}")
    print(f"  Avg SSIM  : {avg_ssim:.4f}")
    print(f"  Avg LPIPS : {avg_lpips:.4f}")
    print("="*30)

    results_data = {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips,
        "path": target_dir
    }
    
    json_path = os.path.join(model_path, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
        
    print(f"Metrics saved to: {json_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--iteration", default=30000, type=int)
    
    args = parser.parse_args(sys.argv[1:])
    evaluate_(args.model_path, args.iteration)
