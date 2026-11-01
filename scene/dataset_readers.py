#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import math
import os
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, read_points3D_binary, read_points3D_text
import subprocess 
import sqlite3    
import shutil     
import open3d as o3d


def rotmat2qvec(R):
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R.flat
    K = np.array([
        [R00 - R11 - R22, 0.0, 0.0, 0.0],
        [R01 + R10, R11 - R00 - R22, 0.0, 0.0],
        [R02 + R20, R12 + R21, R22 - R00 - R11, 0.0],
        [R21 - R12, R02 - R20, R10 - R01, R00 + R11 + R22]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0: q *= -1
    return q

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array

    f_stop: float
    focal_dist_m: float
    focal_length_mm: float


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def auto_run_colmap(path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sparse_dir = os.path.join(path, "colmap_sparse", "0")
    if os.path.exists(os.path.join(sparse_dir, "points3D.bin")):
        print("[Auto-COLMAP] Found points3D.bin; skipping reconstruction.")
        return

    print("[Auto-COLMAP] Building the sparse initialization from known poses...")
    db_path = os.path.join(path, "database.db")
    if os.path.exists(db_path): os.remove(db_path)
    if os.path.exists(sparse_dir): shutil.rmtree(sparse_dir)
    os.makedirs(sparse_dir, exist_ok=True)

    try:
        print("[Auto-COLMAP] Step 1/4: Extracting image features...")
        subprocess.run(["colmap", "feature_extractor", "--database_path", "database.db", "--image_path", "images", "--ImageReader.single_camera", "0"], cwd=path, check=True, stdout=subprocess.DEVNULL)
        print("[Auto-COLMAP] Step 2/4: Matching image features...")
        subprocess.run(["colmap", "exhaustive_matcher", "--database_path", "database.db"], cwd=path, check=True, stdout=subprocess.DEVNULL)

        print("[Auto-COLMAP] Step 3/4: Reading registered image IDs...")
        json_path = os.path.join(path, "transforms_train.json")
        with open(json_path, 'r') as f: data = json.load(f)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, camera_id, name FROM images")
        name_to_ids = {name: (img_id, cam_id) for img_id, cam_id, name in cursor.fetchall()}
        conn.close()
        
        colmap_text_dir = os.path.join(path, "colmap_text")
        os.makedirs(colmap_text_dir, exist_ok=True)
        cam_f = open(os.path.join(colmap_text_dir, "cameras.txt"), "w")
        img_f = open(os.path.join(colmap_text_dir, "images.txt"), "w")
        open(os.path.join(colmap_text_dir, "points3D.txt"), "w").close()
        
        default_angle_x = data.get("camera_angle_x", 0.7)
        for frame in data["frames"]:
            img_name = os.path.basename(frame["file_path"])
            if not img_name.endswith(".png"): img_name += ".png"
            if img_name not in name_to_ids: continue
                
            img_id, cam_id = name_to_ids[img_name]
            img_path = os.path.join(path, "images", img_name)
            img = Image.open(img_path)
            W, H = img.size
            
            fl_x = 0.5 * W / math.tan(0.5 * frame.get("camera_angle_x", default_angle_x))
            cam_f.write(f"{cam_id} PINHOLE {W} {H} {fl_x} {fl_x} {W/2.0} {H/2.0}\n")
            
            c2w = np.array(frame["transform_matrix"])
            c2w[0:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            

            qvec = rotmat2qvec(w2c[:3, :3])
            T = w2c[:3, 3]
            
            img_f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} {cam_id} {img_name}\n\n")
            
        cam_f.close()
        img_f.close()

        print("[Auto-COLMAP] Step 4/4: Triangulating points from the JSON poses...")
        subprocess.run(["colmap", "point_triangulator", "--database_path", "database.db", "--image_path", "images", "--input_path", "colmap_text", "--output_path", "colmap_sparse/0"], cwd=path, check=True, stdout=subprocess.DEVNULL)
        print("[Auto-COLMAP] Sparse initialization is ready.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[Auto-COLMAP ERROR] COLMAP failed: {e}"); sys.exit(1)


def readBlenderWithColmapCloud(path, white_background, eval, extension=".png"):
    auto_run_colmap(path)
    print("Loading Scene: Sim3 Aligned Test Cameras to COLMAP World...")

    all_fovs = []
    name_to_fov = {}
    train_names = set()
    
    def parse_json(json_name, name_set):
        j_path = os.path.join(path, json_name)
        if os.path.exists(j_path):
            with open(j_path, 'r') as f:
                data = json.load(f)
                df = data.get("camera_angle_x", 0.7)
                for frame in data["frames"]:
                    img_name = os.path.basename(frame["file_path"])
                    if not img_name.endswith(".png"): img_name += ".png"
                    fov = frame.get("camera_angle_x", df)
                    all_fovs.append(fov)
                    name_to_fov[img_name] = fov
                    if name_set is not None: name_set.add(img_name)
            return data
        return None
                    
    train_data = parse_json("transforms_train.json", train_names)
    test_data = parse_json("transforms_test.json", None)
    max_fov, min_fov = (max(all_fovs), min(all_fovs)) if all_fovs else (1.0, 0.1)

    cam_ext = read_extrinsics_binary(os.path.join(path, "colmap_sparse/0/images.bin"))
    cam_int = read_intrinsics_binary(os.path.join(path, "colmap_sparse/0/cameras.bin"))

    train_cam_infos = []
    extr_list = cam_ext.values() if isinstance(cam_ext, dict) else cam_ext
    sorted_extrinsics = sorted(extr_list, key=lambda x: x.name)

    colmap_centers_dict = {}

    for extr in sorted_extrinsics:
        intr = cam_int[extr.camera_id] if isinstance(cam_int, dict) else next(i for i in cam_int if i.id == extr.camera_id)
        img_name = extr.name
        uid = intr.id
        
        R_col = np.transpose(qvec2rotmat(extr.qvec))
        T_col = np.array(extr.tvec)
        W2C_col = np.eye(4); W2C_col[:3, :3] = R_col.T; W2C_col[:3, 3] = T_col
        col_center = np.linalg.inv(W2C_col)[:3, 3]
        colmap_centers_dict[img_name] = col_center

        current_fov = name_to_fov.get(img_name, 0.7)
        log_cur = np.log(max(current_fov, 1e-6))
        log_min = np.log(max(min_fov, 1e-6))
        log_max = np.log(max(max_fov, 1e-6))
        norm_time = float(np.clip((log_max - log_cur) / (log_max - log_min), 0, 1)) if (log_max - log_min) > 1e-6 else 0.0
        
        FovX, FovY = current_fov, focal2fov(fov2focal(current_fov, intr.width), intr.height)

        image_path = os.path.join(path, "images", img_name)
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA")) / 255.0
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        arr = im_data[:,:,:3] * im_data[:, :, 3:4] + bg * (1 - im_data[:, :, 3:4])
        image_pil = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

        train_cam_infos.append(CameraInfo(uid=uid, R=R_col, T=T_col, FovY=FovY, FovX=FovX, image=image_pil,
                              image_path=image_path, image_name=Path(img_name).stem, 
                              width=intr.width, height=intr.height, time=norm_time, mask=None))

    pts_blender = []
    pts_colmap = []
    if train_data:
        for frame in train_data["frames"]:
            img_name = os.path.basename(frame["file_path"])
            if not img_name.endswith(".png"): img_name += ".png"
            if img_name in colmap_centers_dict:
                c2w = np.array(frame["transform_matrix"])
                c2w[0:3, 1:3] *= -1 
                pts_blender.append(c2w[:3, 3])
                pts_colmap.append(colmap_centers_dict[img_name])

    pts_blender = np.array(pts_blender)
    pts_colmap = np.array(pts_colmap)

    def umeyama(src, dst):
        mu_s = src.mean(axis=0)
        mu_d = dst.mean(axis=0)
        src_c = src - mu_s
        dst_c = dst - mu_d
        var_s = np.var(src, axis=0).sum()
        var_d = np.var(dst, axis=0).sum()
        K = src_c.T @ dst_c
        U, D, Vh = np.linalg.svd(K)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vh) < 0: S[2,2] = -1
        R_align = U @ S @ Vh
        scale_align = np.trace(np.diag(D) @ S) / var_s
        t_align = mu_d - scale_align * (R_align @ mu_s)
        return R_align, t_align, scale_align

    R_a, T_a, S_a = umeyama(pts_blender, pts_colmap)
    print(f"Alignment calculated: Scale = {S_a:.4f}")

    test_cam_infos = []
    if eval and test_data:
        df = test_data.get("camera_angle_x", 0.7)
        for idx, frame in enumerate(test_data["frames"]):
            img_name = os.path.basename(frame["file_path"])
            if not img_name.endswith(".png"): img_name += ".png"
            image_path = os.path.join(path, "images", img_name)
            if not os.path.exists(image_path): continue
                
            c2w_blend = np.array(frame["transform_matrix"])
            c2w_blend[0:3, 1:3] *= -1 
            blender_center = c2w_blend[:3, 3]
            blender_rot = c2w_blend[:3, :3]
            
            colmap_center = S_a * (R_a @ blender_center) + T_a
            colmap_rot = R_a @ blender_rot
            
            c2w_new = np.eye(4)
            c2w_new[:3, :3] = colmap_rot
            c2w_new[:3, 3] = colmap_center
            w2c_new = np.linalg.inv(c2w_new)
            
            R_test = np.transpose(w2c_new[:3,:3])
            T_test = w2c_new[:3, 3]

            current_fov = frame.get("camera_angle_x", df)
            log_cur = np.log(max(current_fov, 1e-6))
            norm_time = float(np.clip((log_max - log_cur) / (log_max - log_min), 0, 1)) if (log_max - log_min) > 1e-6 else 0.0

            image = Image.open(image_path)
            width, height = image.size
            FovX, FovY = current_fov, focal2fov(fov2focal(current_fov, width), height)
            
            im_data = np.array(image.convert("RGBA")) / 255.0
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            arr = im_data[:,:,:3] * im_data[:, :, 3:4] + bg * (1 - im_data[:, :, 3:4])
            image_pil = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

            test_cam_infos.append(CameraInfo(uid=len(train_cam_infos)+idx, R=R_test, T=T_test, FovY=FovY, FovX=FovX, image=image_pil,
                                  image_path=image_path, image_name=Path(img_name).stem, 
                                  width=width, height=height, time=norm_time, mask=None))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    colmap_bin_path = os.path.join(path, "colmap_sparse/0/points3D.bin")
    final_ply_path = os.path.join(path, "points3d.ply")
    xyz, rgb, _ = read_points3D_binary(colmap_bin_path)
    
    storePly(final_ply_path, xyz, rgb)
    pcd = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))

    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos, video_cameras=train_cam_infos, nerf_normalization=nerf_normalization, ply_path=final_ply_path, maxtime=1.0)


def auto_run_multifocal_colmap(path, eval_mode=False):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sparse_dir = os.path.join(path, "sparse", "0")
    
    if os.path.exists(os.path.join(sparse_dir, "points3D.bin")) or os.path.exists(os.path.join(sparse_dir, "points3D.txt")):
        print("[Auto-COLMAP] sparse/0 already exists. Skipping reconstruction.")
        return

    print("\n[Auto-COLMAP] Starting multi-focal triangulation pipeline...")
    db_path = os.path.join(path, "database.db")
    os.makedirs(sparse_dir, exist_ok=True)

    try:
        if not os.path.exists(db_path):

            json_path = os.path.join(path, "transforms_train.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            

            list_file = os.path.join(path, "colmap_image_list.txt")
            with open(list_file, 'w') as f:
                for frame in train_data["frames"]:
                    img_name = os.path.basename(frame["file_path"])
                    if not img_name.endswith(".png") and not img_name.endswith(".jpg"): 
                        img_name += ".png"
                    f.write(f"{img_name}\n")
            print("[Auto-COLMAP] Step 1/4: Feature extraction (Forcing GPU)...")
            subprocess.run([
                "colmap", "feature_extractor", 
                "--database_path", "database.db", 
                "--image_path", "images", 
                "--image_list_path", "colmap_image_list.txt",
                "--ImageReader.single_camera", "0", 
                "--SiftExtraction.use_gpu", "1"
            ], cwd=path, check=True)
            
            print("[Auto-COLMAP] Step 2/4: Exhaustive matching (Forcing GPU)...")
            subprocess.run(["colmap", "exhaustive_matcher", "--database_path", "database.db", "--SiftMatching.use_gpu", "1"], cwd=path, check=True)

        print("[Auto-COLMAP] Step 3/4: Injecting EXACTLY transforms_train.json for Blurry Prior...")
        json_path = os.path.join(path, "transforms_train.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cannot find {json_path}!")
            
        with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, camera_id, name FROM images")
        name_to_ids = {name: (img_id, cam_id) for img_id, cam_id, name in cursor.fetchall()}
        conn.close()
        
        text_dir = os.path.join(path, "colmap_text")
        os.makedirs(text_dir, exist_ok=True)
        cam_f = open(os.path.join(text_dir, "cameras.txt"), "w")
        img_f = open(os.path.join(text_dir, "images.txt"), "w")
        open(os.path.join(text_dir, "points3D.txt"), "w").close()

        default_fov = data.get("camera_angle_x", 0.7)
        for frame in data["frames"]:
            img_name = os.path.basename(frame["file_path"])
            if not img_name.endswith(".png") and not img_name.endswith(".jpg"): img_name += ".png" 
            if img_name not in name_to_ids: continue
            
            img_id, cam_id = name_to_ids[img_name]
            img_path = os.path.join(path, "images", img_name)
            img = Image.open(img_path)
            W, H = img.size
            
            fov = frame.get("camera_angle_x", default_fov)
            fl_x = 0.5 * W / math.tan(0.5 * fov)
            cam_f.write(f"{cam_id} PINHOLE {W} {H} {fl_x} {fl_x} {W/2.0} {H/2.0}\n")
            
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 0] /= np.linalg.norm(c2w[:3, 0])
            c2w[:3, 1] /= np.linalg.norm(c2w[:3, 1])
            c2w[:3, 2] /= np.linalg.norm(c2w[:3, 2])
            c2w[0:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            qvec = rotmat2qvec(w2c[:3, :3])
            T = w2c[:3, 3]
            
            img_f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} {cam_id} {img_name}\n\n")
            
        cam_f.close()
        img_f.close()

        print("[Auto-COLMAP] Step 4/4: Final geometric triangulation...")
        subprocess.run(["colmap", "point_triangulator", "--database_path", "database.db", "--image_path", "images", "--input_path", "colmap_text", "--output_path", "sparse/0"], cwd=path, check=True)
        print("[Auto-COLMAP] Success! Point cloud strictly built on blurry train images.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[Auto-COLMAP ERROR] COLMAP process failed: {e}")
        sys.exit(1)


def readColmapSceneInfo(path, images, eval, lod, frames_start_end=[0,300], llffhold=8):
    print("Loading Scene: Strict 3-Way Splitting (Train Blur, Test Sharp, Train Sharp)...")
    

    auto_run_multifocal_colmap(path, eval_mode=eval)
    
    def load_json(filename):
        j_path = os.path.join(path, filename)
        if os.path.exists(j_path):
            with open(j_path, 'r', encoding='utf-8') as f: return json.load(f)
        return None


    train_data = load_json("transforms_train.json")
    test_data = load_json("transforms_test.json")
    train_sharp_data = load_json("transforms_train_sharp.json")

    if not train_data:
        raise FileNotFoundError("Missing transforms_train.json! Please use the new dataset script.")


    all_fovs = []
    for d in [train_data, test_data, train_sharp_data]:
        if d: all_fovs.extend([f.get("camera_angle_x", 0.7) for f in d.get("frames", [])])
    max_fov, min_fov = (max(all_fovs), min(all_fovs)) if all_fovs else (1.0, 0.1)

    reading_dir = "images" if images is None else images


    all_focals = []
    for d in [train_data, test_data, train_sharp_data]:
        if d: all_focals.extend([f.get("focal_length_mm", 50.0) for f in d.get("frames", [])])
    max_focal, min_focal = (max(all_focals), min(all_focals)) if all_focals else (50.0, 50.0)


    def parse_cameras(data_dict, uid_offset=0):
        cam_infos = []
        if not data_dict: return cam_infos
        

        for idx, frame in enumerate(data_dict["frames"]):
            img_name = os.path.basename(frame["file_path"])
            if not img_name.endswith(".png") and not img_name.endswith(".jpg"): img_name += ".png"
            image_path = os.path.join(path, reading_dir, img_name)
            if not os.path.exists(image_path): continue
            

            image = Image.open(image_path)
            width, height = image.size
            

            FovX = frame.get("camera_angle_x", 0.7)
            FovY = focal2fov(fov2focal(FovX, width), height)


            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 0] /= np.linalg.norm(c2w[:3, 0])
            c2w[:3, 1] /= np.linalg.norm(c2w[:3, 1])
            c2w[:3, 2] /= np.linalg.norm(c2w[:3, 2])
            c2w[0:3, 1:3] *= -1  
            w2c = np.linalg.inv(c2w)
            
            qvec = rotmat2qvec(w2c[:3, :3])
            R = np.transpose(qvec2rotmat(qvec))
            T = w2c[:3, 3]


            current_focal = frame.get("focal_length_mm", 50.0)
            
            if (max_focal - min_focal) > 1e-4:
                log_cur = np.log(max(current_focal, 1e-6))
                log_min = np.log(max(min_focal, 1e-6))
                log_max = np.log(max(max_focal, 1e-6))
                

                norm_time = float(np.clip((log_cur - log_min) / (log_max - log_min), 0.0, 1.0))
            else:
                norm_time = 0.0


            im_data = np.array(image.convert("RGBA")) / 255.0
            bg = np.array([0, 0, 0])
            arr = im_data[:,:,:3] * im_data[:, :, 3:4] + bg * (1 - im_data[:, :, 3:4])
            image_pil = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
            
            if idx == 0:
              print(f"\n[Debug Camera 0] Image: {img_name}")
              print(f"  - Image Size: {width} x {height}")
              print(f"  - Focal Length (mm) from JSON: {current_focal}")
              print(f"  - Calculated FovX: {FovX:.4f} rad ({(FovX * 180 / np.pi):.2f} deg)")
              print(f"  - Calculated FovY: {FovY:.4f} rad ({(FovY * 180 / np.pi):.2f} deg)")

              fx_calc = width / (2 * math.tan(FovX / 2))
              print(f"  - Reverse-calculated fx (pixels): {fx_calc:.2f}")
            
            cam_infos.append(CameraInfo(uid=uid_offset + idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                                        image=image_pil, image_path=image_path, 
                                        image_name=Path(img_name).stem, 
                                        width=width, height=height, time=norm_time, mask=None,
                                        f_stop=frame.get("f_stop", 2.0),
                                        focal_dist_m=frame.get("focal_dist_m", 5.0),
                                        focal_length_mm=frame.get("focal_length_mm", 50.0)))
        return sorted(cam_infos, key=lambda x: x.image_name)


    train_cam_infos = parse_cameras(train_data, uid_offset=0)
    test_cam_infos = parse_cameras(test_data, uid_offset=len(train_cam_infos))
    train_sharp_cam_infos = parse_cameras(train_sharp_data, uid_offset=len(train_cam_infos) + len(test_cam_infos))

    print(f"[Dataset Loaded] Train(Blurry)={len(train_cam_infos)} | Test(Sharp)={len(test_cam_infos)} | Train-Eval(Sharp)={len(train_sharp_cam_infos)}")


    nerf_normalization = getNerfppNorm(train_cam_infos)

    try:
        xyz, rgb, _ = read_points3D_binary(os.path.join(path, "sparse/0/points3D.bin"))
    except:
        xyz, rgb, _ = read_points3D_text(os.path.join(path, "sparse/0/points3D.txt"))

    ply_path = os.path.join(path, "points3d.ply")
    storePly(ply_path, xyz, rgb)
    pcd = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))


    return SceneInfo(point_cloud=pcd, 
                     train_cameras=train_cam_infos,
                     test_cameras=test_cam_infos,
                     video_cameras=train_sharp_cam_infos,
                     nerf_normalization=nerf_normalization, 
                     ply_path=ply_path, 
                     maxtime=1.0)


# [Pure Colmap Version]: No JSON, extracts poses and time from cameras.bin


# [Pure Colmap Version]: No JSON, extracts poses and time from cameras.bin

def readPureColmapSceneInfo(path, images, eval, lod, frames_start_end=[0,300], llffhold=8):
    print("Loading Scene: Pure COLMAP mode (Extracting poses and focal lengths directly from binaries)...")
    
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    extr_list = cam_extrinsics.values() if isinstance(cam_extrinsics, dict) else cam_extrinsics


    all_fovs = []
    for extr in extr_list:
        intr = cam_intrinsics[extr.camera_id] if isinstance(cam_intrinsics, dict) else next(i for i in cam_intrinsics if i.id == extr.camera_id)
        all_fovs.append(focal2fov(intr.params[0], intr.width))
    max_fov, min_fov = (max(all_fovs), min(all_fovs)) if all_fovs else (1.0, 0.1)

    cam_infos_unsorted = []
    for extr in extr_list:
        intr = cam_intrinsics[extr.camera_id] if isinstance(cam_intrinsics, dict) else next(i for i in cam_intrinsics if i.id == extr.camera_id)
        height = intr.height
        width = intr.width
        uid = intr.id


        R_raw = qvec2rotmat(extr.qvec)
        T_raw = np.array(extr.tvec)


        w2c_raw = np.eye(4)
        w2c_raw[:3, :3] = R_raw
        w2c_raw[:3, 3] = T_raw
        c2w = np.linalg.inv(w2c_raw)


        c2w[:3, 0] /= np.linalg.norm(c2w[:3, 0])
        c2w[:3, 1] /= np.linalg.norm(c2w[:3, 1])
        c2w[:3, 2] /= np.linalg.norm(c2w[:3, 2])


        w2c_clean = np.linalg.inv(c2w)
        qvec_clean = rotmat2qvec(w2c_clean[:3, :3])
        R = np.transpose(qvec2rotmat(qvec_clean))
        T = w2c_clean[:3, 3]


        FovX = focal2fov(intr.params[0], width)
        FovY = focal2fov(intr.params[1] if intr.model == "PINHOLE" else intr.params[0], height)

        if (max_fov - min_fov) > 1e-4:
            log_cur = np.log(max(FovX, 1e-6))
            log_min = np.log(max(min_fov, 1e-6))
            log_max = np.log(max(max_fov, 1e-6))
            norm_time = float(np.clip((log_max - log_cur) / (log_max - log_min), 0.0, 1.0))
        else:
            norm_time = 0.0

        image_path = os.path.join(path, reading_dir, os.path.basename(extr.name))
        if not os.path.exists(image_path): continue
            

        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA")) / 255.0
        bg = np.array([0, 0, 0])
        arr = im_data[:,:,:3] * im_data[:, :, 3:4] + bg * (1 - im_data[:, :, 3:4])
        image_pil = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
        
        if "00008" in image_name or "00004" in image_name:
            print(f"[Debug Check] Image: {image_name} | COLMAP FovX: {FovX:.4f} | norm_time: {norm_time:.4f} | T_colmap: {T}")
        cam_infos_unsorted.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image_pil,
                              image_path=image_path, image_name=image_name, width=width, height=height, time=norm_time, mask=None))

    cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

    train_cam_infos = []
    test_cam_infos = []
    if eval:
        test_json_path = os.path.join(path, "transforms_test.json")
        test_bak_path = os.path.join(path, "transforms_test.json.bak")
        target_json = test_json_path if os.path.exists(test_json_path) else test_bak_path
        
        if os.path.exists(target_json):
            print(f"[Dataset Split] Reading strict Novel View Split from {os.path.basename(target_json)}...")
            test_filenames = set(os.path.basename(f["file_path"]).split(".")[0] for f in json.load(open(target_json))["frames"])
            train_cam_infos = [c for c in cam_infos if c.image_name not in test_filenames]
            test_cam_infos = [c for c in cam_infos if c.image_name in test_filenames]
        else:
            print("[Dataset Split] WARNING: Test list not found. Falling back to LLFF 1/8.")
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    try:
        xyz, rgb, _ = read_points3D_binary(os.path.join(path, "sparse/0/points3D.bin"))
    except:
        xyz, rgb, _ = read_points3D_text(os.path.join(path, "sparse/0/points3D.txt"))

    storePly(os.path.join(path, "points3d.ply"), xyz, rgb)
    pcd = BasicPointCloud(points=xyz, colors=rgb / 255.0, normals=np.zeros_like(xyz))
    
    print("\n================ [DEBUG TEST CAMERAS] ================")
    for i, c in enumerate(test_cam_infos):
        print(f"Render ID: {i:05d}.png | Orig Name: {c.image_name} | FOV_X: {c.FovX:.4f} | Time: {c.time:.4f} | T: {c.T}")
    print("======================================================\n")
    
    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos, video_cameras=train_cam_infos, nerf_normalization=nerf_normalization, ply_path=os.path.join(path, "points3d.ply"), maxtime=1.0)

def readBlenderSceneInfo(path, white_background, eval):
    """Load Blender/JSON data and map focal length to the conditioning coordinate."""
    print("Loading Blender Scene with Global Zoom-Time Mapping...")
    
    def load_json_data(filename):
        json_path = os.path.join(path, filename)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f: return json.load(f)
        return None


    train_content = load_json_data("transforms_train.json")
    test_content = load_json_data("transforms_test.json")
    
    if train_content is None:
        train_content = load_json_data("transforms.json")
        if train_content is None: 
            raise FileNotFoundError(f"No transforms found in {path}")

    train_frames = train_content["frames"]
    test_frames = test_content["frames"] if test_content else []


    all_fovs = []
    def collect_fovs(frames, defaults):
        for frame in frames: 
            fov = frame.get("camera_angle_x", defaults.get("camera_angle_x"))
            if fov is not None:
                all_fovs.append(fov)
            
    collect_fovs(train_frames, train_content)
    if test_frames: 
        collect_fovs(test_frames, test_content)
        
    if not all_fovs:

        print("[Warning] No camera_angle_x found, using default range.")
        max_fov, min_fov = 1.0, 0.1
    else:
        max_fov, min_fov = max(all_fovs), min(all_fovs)
        
    print(f"[Global Range] MaxFOV: {max_fov:.4f} -> Time 0.0 | MinFOV: {min_fov:.4f} -> Time 1.0")

    def parse_cameras(frames, defaults):
        cam_infos = []
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])

            if not os.path.exists(cam_name):
                if os.path.exists(cam_name + ".png"): cam_name += ".png"
                elif os.path.exists(cam_name + ".jpg"): cam_name += ".jpg"
            
            if not os.path.exists(cam_name):
                print(f"Warning: Image {cam_name} not found, skipping.")
                continue


            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1 # OpenGL -> OpenCV
            w2c = np.linalg.inv(c2w)
            R, T = np.transpose(w2c[:3,:3]), w2c[:3, 3]


            current_fov = frame.get("camera_angle_x", defaults.get("camera_angle_x"))
            
            log_cur = np.log(max(current_fov, 1e-6))
            log_min = np.log(max(min_fov, 1e-6))
            log_max = np.log(max(max_fov, 1e-6))
            
            if (log_max - log_min) > 1e-6:
                norm_time = (log_max - log_cur) / (log_max - log_min)
            else:
                norm_time = 0.0
                
            norm_time = float(np.clip(norm_time, 0.0, 1.0))


            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)


            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image_pil = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")


            FovX = current_fov
            FovY = focal2fov(fov2focal(FovX, image_pil.size[0]), image_pil.size[1])


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_pil,
                                        image_path=image_path, image_name=image_name, width=image_pil.size[0], height=image_pil.size[1],
                                        time=norm_time, mask=None))
        return cam_infos

    print("Parsing Train Cameras...")
    train_cam_infos = parse_cameras(train_frames, train_content)
    test_cam_infos = parse_cameras(test_frames, test_content) if test_frames else []


    print("Computing Scene Normalization based on TRAIN data...")
    train_cam_centers = []
    for info in train_cam_infos:
        W2C = np.eye(4); W2C[:3, :3] = info.R.T; W2C[:3, 3] = info.T
        train_cam_centers.append(np.linalg.inv(W2C)[:3, 3])
    
    scene_center = np.mean(np.stack(train_cam_centers), axis=0)
    dist = np.linalg.norm(train_cam_centers - scene_center, axis=1)
    scene_radius = np.max(dist)
    
    translate = -scene_center
    scale_factor = 1.1 / scene_radius if scene_radius > 1e-6 else 1.0
    
    print(f"[Auto-Process] Center: {scene_center}, Radius: {scene_radius}, Scale: {scale_factor}")

    def update_camera_list(old_list):
        new_list = []
        for cam in old_list:
            W2C = np.eye(4); W2C[:3, :3] = cam.R.T; W2C[:3, 3] = cam.T
            C2W = np.linalg.inv(W2C)
            center_new = (C2W[:3, 3] + translate) * scale_factor
            T_new = -np.dot(cam.R.T, center_new)
            new_list.append(cam._replace(T=T_new))
        return new_list

    train_cam_infos = update_camera_list(train_cam_infos)
    test_cam_infos = update_camera_list(test_cam_infos)


    nerf_normalization = getNerfppNorm(train_cam_infos)
    print("Generating Centered Frustum Point Cloud...")
    
    num_pts = 100_000
    init_radius = nerf_normalization['radius']
    
    points = (np.random.random((num_pts, 3)) - 0.5) * init_radius * 2.0
    colors = np.random.random((num_pts, 3))
    
    pcd = BasicPointCloud(points=points, colors=colors, normals=np.zeros((num_pts, 3)))
    
    ply_path = os.path.join(path, "points3d_centered.ply")
    if not os.path.exists(ply_path): 
        try:

            storePly(ply_path, pcd.points, pcd.colors * 255.0)
        except:
            pass

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=1.0)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    #"Colmap": readPureColmapSceneInfo,
    "Blender": readColmapSceneInfo, 
}