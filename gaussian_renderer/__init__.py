#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch
from einops import repeat
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import tinycudann as tcnn
import torch.nn.functional as F

def generate_full_temporal_gaussians(viewpoint_camera, pc : GaussianModel,  visible_mask=None, is_training=False,  timestamp = None , opt_thro = 0.0 , force_deblur=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]  # [N,3]
    
    # [修改] 增加 timestamp_idx 计算，用于 Time Embedding
    if timestamp == None:
        timestamp = torch.tensor([viewpoint_camera.time], dtype=torch.float32).to(anchor.device).repeat(anchor.shape[0], 1)
        timestamp_idx = torch.tensor([min(int(viewpoint_camera.time * 300), 299)], dtype=torch.long).to(anchor.device).repeat(anchor.shape[0])
    else :
        timestamp = torch.tensor(timestamp).to(anchor.device).repeat(anchor.shape[0],1)
        timestamp_idx = (timestamp * 300).long().squeeze(1).clamp(min=0, max=299)
    
    sta_feat = pc._anchor_feat[visible_mask]  # [N,32]          

    if pc.hash:
        dy_feat, dynamic_mask = pc.dynamic_module(anchor, timestamp)
        feat = dynamic_mask * dy_feat + (1.0 - dynamic_mask) * sta_feat
    else:
        dy_feat, _ = pc.hexplane(anchor,timestamp)
        feat = sta_feat + dy_feat
    
    # get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center.cuda().unsqueeze(0)
    ob_dist = ob_view.norm(dim=1, keepdim=True) 
    ob_view = ob_view / ob_dist

    neural_opacity = pc.get_opacity_mlp(feat)  # [N,32+3]

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = ( neural_opacity > opt_thro )  # 
    mask = mask.view(-1)

    # select opacity                                
    opacity = neural_opacity[mask] 

    # [REVERTED] 去掉拼接，直接传 feat
    # 这样就和 ZoomDyGS 原版逻辑完全一致
    color = pc.get_color_mlp(feat)    
    
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3]) # [mask]

    # The [:,:3] controls the step size of offset. The [:,3:] serves as the base scale for neural gaussian's shape.
    scale_rot = pc.get_cov_mlp(feat) 
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    # 直接获取 offsets，不需要 raw_offsets 和 tanh
    offsets = pc.get_offset_mlp(feat).view([-1,3])   
    
    grid_scaling = pc.get_scaling[visible_mask] # [N,6] 

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  
    
    # 恢复纯净拼接：直接拼接 offsets
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets  = masked.split([6, 3, 3, 7, 3], dim = -1 )

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) 
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]  
    xyz = repeat_anchor + offsets
    
    base_scaling = scaling.clone()


    if scaling.shape[0]==0:
        pass

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, base_scaling
    else:
        return xyz, color, opacity, scaling, rot, None, None, base_scaling
    

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt_thro = 0.0, stage="coarse", scaling_modifier = 1.0, iteration = 30000 , retain_grad=False , render_anchor=False, visible_mask=None, force_deblur=False):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        if iteration  >= 10000 and iteration <= 20000:
            opt_thro = (iteration / 1000) *0.001 - 0.01
        elif iteration > 20000:
            opt_thro = 0.01
    else:
        opt_thro = 0.01

    # 👇 [修改] 清理了这里的幽灵空格
    xyz, color, opacity, scaling, rot, neural_opacity, mask, base_scaling = generate_full_temporal_gaussians(viewpoint_camera, pc, visible_mask=visible_mask, is_training=is_training, opt_thro=opt_thro, force_deblur=force_deblur)
    # [EXPORT] Intercept true 3D Gaussians and export as standard PLY for viewers
    # ==========================================================
    import os
    if os.environ.get("EXPORT_REAL_PLY") == "1" and not getattr(pc, "has_exported_real_ply", False):
        import numpy as np
        from plyfile import PlyData, PlyElement

        print(f"\n[Export] Intercepting real MLP Gaussian points (Current Timestamp: {viewpoint_camera.time:.3f})...")

        # 1. Move true dynamic attributes from GPU back to CPU
        _xyz = xyz.detach().cpu().numpy()
        _opacity_raw = opacity.detach().cpu().numpy()
        _scale_raw = scaling.detach().cpu().numpy()
        _rot = rot.detach().cpu().numpy()

        # 2. Reverse Activations! (CRITICAL FIX FOR VIEWERS)
        # Viewers apply exp(scale), so we MUST export log(scale)
        _scale = np.log(_scale_raw + 1e-10)

        # Viewers apply sigmoid(opacity), so we MUST export inverse sigmoid (logit)
        _opacity_clipped = np.clip(_opacity_raw, 1e-6, 1.0 - 1e-6)
        _opacity = np.log(_opacity_clipped / (1.0 - _opacity_clipped))

        # 3. Color conversion: Viewers expect SH DC components
        _rgb = color.detach().cpu().numpy()
        _f_dc = (_rgb - 0.5) / 0.28209479177387814

        # 4. Pad with dummy normals
        _normals = np.zeros_like(_xyz)

        # 5. Pack into strictly formatted official 3DGS structure
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                 ('opacity', 'f4'),
                 ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                 ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]

        elements = np.empty(_xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((_xyz, _normals, _f_dc, _opacity, _scale, _rot), axis=1)
        elements[:] = list(map(tuple, attributes))
        vertex_element = PlyElement.describe(elements, 'vertex')

        # 6. Save as PLY file
        out_name = os.path.join(pc.model_path if hasattr(pc, 'model_path') else ".", f"REAL_Gaussians_time_{viewpoint_camera.time:.2f}.ply")
        PlyData([vertex_element]).write(out_name)
        print(f"[Export] SUCCESS! Saved {_xyz.shape[0]} fully decoded Gaussian points to: {out_name}\n")

        pc.has_exported_real_ply = True
    # ==========================================================

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # ================= [修改：支持动态可学习的严格物理参数] =================
    aperture_meters = viewpoint_camera.aperture_m
    
    # 【新增 1】：获取动态焦距！如果开了学习开关，这里拿到的是 sigmoid 后的值
    if hasattr(viewpoint_camera, 'get_dynamic_focus'):
        focus_dist_m = viewpoint_camera.get_dynamic_focus()
    else:
        focus_dist_m = viewpoint_camera.focal_dist_m
        
    f_meters = viewpoint_camera.lens_focal_meters
    f_pix = viewpoint_camera.image_width / (2.0 * tanfovx)

    # 【新增 2】：准备用来接焦距梯度的“外卖盒”
    # 注意：为了让 CUDA 也能传回来梯度，即使我们在推理时（或没开学习），也先给个 Dummy
    # 只要在 python 端不挂 required_grad，它就不会影响计算图
    grad_focus_container = torch.zeros(1, dtype=torch.float32, device="cuda")
    # ====================================================================

    use_physical_deblur = (pc.use_deblur and is_training) or bool(force_deblur)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        use_deblur=use_physical_deblur,
        # ================= [注入严格的物理参数] =================
        lens_aperture=aperture_meters,      
        lens_focus_depth=focus_dist_m,      
        lens_focal_meters=f_meters,
        lens_focal_pixels=f_pix,
        
        # 【新增 3】：把接梯度的空盒子传进设置里 (如果 C++ 那边还没改这行会报错，所以先预留)
        # 注意：此处假设你的 C++ 接口已经支持传入 grad_focus
        grad_focus=grad_focus_container
        # =======================================================
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    rendered_sta_image = None
    
    
    
    #[修改] 正确接住 4 个返回值！
    rendered_image, radii, depth_map, alpha_map = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_sta_image": rendered_sta_image, 
                "depth_map": depth_map,  # 👇 [修改] 抛出深度图
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "base_scaling": base_scaling,
                "scaling": scaling,
                "neural_points":xyz,
                "rotations": rot,     
                "colors": color,
                "grad_focus": grad_focus_container       
                }
    else:
        return {"render": rendered_image,
                "depth_map": depth_map,  # 👇 [修改] 测试阶段也抛出深度图
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    tan_half_fovx = math.tan(viewpoint_camera.FoVx * 0.5)
    W_sensor = 0.00454  # 4.54mm (物理传感器宽度)
    f_meters = W_sensor / (2.0 * tan_half_fovx)
    f_pix = viewpoint_camera.image_width / (2.0 * tan_half_fovx)
    
    dummy_grad_focus = torch.zeros(1, dtype=torch.float32, device="cuda")
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        
        # [NEW] 新增的物理参数与开关
        use_deblur=False,  # 你的开关！
        lens_aperture=0.0,         # 光圈
        lens_focus_depth=2.96,     # 对焦平面
        lens_focal_meters=f_meters,# 物理焦距
        lens_focal_pixels=f_pix,
        grad_focus=dummy_grad_focus    # 像素焦距
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python: 
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation  # [N,4]

    radii_pure = rasterizer.visible_filter(means3D = means3D,  
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
