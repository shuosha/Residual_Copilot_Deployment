import sys, os
import cv2
import numpy as np
import json
import trimesh
import time
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, "/home/shuosha/projects/FoundationPose")
from Utils import draw_xyz_axis # type: ignore
from estimater import FoundationPose # type: ignore

import logging
logging.getLogger().setLevel(logging.CRITICAL)

H, W = 480, 848

# Maps task name → (held_asset_obj, fixed_asset_obj)
TASK_ASSET_NAMES = {
    "gearmesh": ("gear.obj", "gear_base.obj"),
    "peginsert": ("peg.obj", "peg_base.obj"),
    "nutthread": ("nut.obj", "nut_base.obj"),
}
def get_mask(initial_scene, background_scene, corners_list): 
    """
    This function should return a mask based on the initial scene and the background scene.
    For now, we will return None as a placeholder.
    """
    # --- 1) Load images ----------------------------------------
    color  = initial_scene        # your current RGB frame
    bg_bgr = background_scene   # your background image (same size)

    H, W = color.shape[:2]

    # --- 2) Background subtraction mask ----------------------
    #    a) absolute difference & grayscale
    diff = cv2.absdiff(color, bg_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #    b) threshold to binary (0/1)
    _, mask_bs = cv2.threshold(gray, 50, 1, cv2.THRESH_BINARY)

    #    c) clean small holes / speckles
    kernel  = np.ones((5,5), np.uint8)
    mask_bs = cv2.morphologyEx(mask_bs, cv2.MORPH_CLOSE, kernel)

    # --- 3) Quadrilateral ROI mask ----------------------------
    #    Reorder corners into CCW winding so fillPoly produces a
    #    proper quad even if they were clicked in arbitrary order.
    pts = np.array(corners_list, dtype=np.float64)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(angles)].astype(np.int32).reshape((-1, 1, 2))

    quad_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(quad_mask, [pts], color=1)

    # --- 4) Combine masks --------------------------------------
    #    final mask is 1 only where both are 1
    mask_final = (mask_bs & quad_mask).astype(np.uint8)
    return mask_final

class StateEstimator():
    def __init__(self, foundation_pose_dir=None, task_name=None):
        super().__init__()
        assert foundation_pose_dir is not None, "Foundation pose directory must be provided."
        self.foundation_pose_dir = foundation_pose_dir

        # Resolve task name from directory basename if not provided
        if task_name is None:
            task_name = os.path.basename(foundation_pose_dir)
        assert task_name in TASK_ASSET_NAMES, \
            f"Unknown task '{task_name}', expected one of {list(TASK_ASSET_NAMES.keys())}"
        held_obj_name, fixed_obj_name = TASK_ASSET_NAMES[task_name]

        self.object_list = ["held_asset", "fixed_asset"]
        self.num_objects = len(self.object_list)

        self.empty_scene = cv2.imread(os.path.join(foundation_pose_dir, "empty_scene.png"))
        with open(os.path.join(foundation_pose_dir, "extrinsics.json"), 'r') as f:
            front2base = json.load(f)["cam2base"]["front2base"]
        with open(os.path.join(foundation_pose_dir, "intrinsics.json"), 'r') as f:
            intrinsics = json.load(f)["front"]
        self.cam_k = np.array([
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0]
        ])
        self.cam2base = np.array(front2base).reshape(4, 4)
        with open(os.path.join(foundation_pose_dir, "corners.json"), "r") as f:
            self.corners_dict = json.load(f)

        held_mesh = trimesh.load(os.path.join(foundation_pose_dir, held_obj_name))
        self.held_asset_estimator = FoundationPose(
            model_pts=held_mesh.vertices,
            model_normals=held_mesh.vertex_normals,
            mesh=held_mesh,
            debug_dir=foundation_pose_dir,
            debug=0
        )

        fixed_mesh = trimesh.load(os.path.join(foundation_pose_dir, fixed_obj_name))
        self.fixed_asset_estimator = FoundationPose(
            model_pts=fixed_mesh.vertices,
            model_normals=fixed_mesh.vertex_normals,
            mesh=fixed_mesh,
            debug_dir=foundation_pose_dir,
            debug=0
        )

    def estimate_object_poses(self, bgr, depth, obj_name, retrack=False):
        assert obj_name in ["fixed_asset", "held_asset"], "obj_name must be 'fixed_asset' or 'held_asset'"

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = depth / 1000.0 # convert to meters

        if obj_name == "held_asset":
            estimator = self.held_asset_estimator
        else:
            estimator = self.fixed_asset_estimator

        # todo add restart estimator if lost
        if estimator.pose_last is None or retrack:
            corners = self.corners_dict[obj_name]
            mask = get_mask(bgr, self.empty_scene, corners)
            obj2cam = estimator.register(
                K=self.cam_k,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=5,
            )
        else:
            obj2cam = estimator.track_one(
                K=self.cam_k,
                rgb=rgb,
                depth=depth,
                iteration=5,
            )
        
        return obj2cam

    def draw_detected_objects(self, bgr_image, obj2cam):
        bgr = bgr_image.copy()
        vis = draw_xyz_axis(
            color=bgr,
            ob_in_cam=obj2cam,
            K=self.cam_k,
            is_input_rgb=False,
            scale=0.05,
        )

        # Implement object detection and drawing logic here
        return vis
    
    def project_points_on_image(self, image, points_base, color=(0,0,255)) -> np.ndarray:
        """
        image: (H,W,3) uint8
        points_base: (N,3) in base frame
        """
        # invert T_cam_base to get base->cam
        T_base_cam = np.linalg.inv(self.cam2base)
        R_bc = T_base_cam[:3, :3]  # rotation from base to cam
        t_bc = T_base_cam[:3, 3]   # translation from base to cam

        # 1) base -> camera
        pts_base = points_base.T  # (3,N)
        pts_cam = R_bc @ pts_base + t_bc[:, None]  # (3,N)

        X = pts_cam[0, :]
        Y = pts_cam[1, :]
        Z = pts_cam[2, :]

        # keep only points in front of the camera
        mask = Z > 0
        X, Y, Z = X[mask], Y[mask], Z[mask]

        # 2) project using intrinsics: u = fx*X/Z + cx, v = fy*Y/Z + cy
        fx, fy = self.cam_k[0, 0], self.cam_k[1, 1]
        cx, cy = self.cam_k[0, 2], self.cam_k[1, 2]

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        # round and convert to int
        u = u.astype(int)
        v = v.astype(int)

        h, w = image.shape[:2]
        # optionally filter to only those inside the image
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[valid]
        v = v[valid]

        # 3) draw on the image
        img_vis = image.copy()
        for px, py in zip(u, v):
            cv2.circle(img_vis, (px, py), 3, color, -1)

        return img_vis 

    def draw_triangle_from_base_points(
        self,
        image: np.ndarray,
        fingertip_pos: np.ndarray,
        base_pos: np.ndarray,
        final_pos: np.ndarray,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Project three 3D points (in base frame) onto the RGB image and draw:
        - fingertip -> base    : blue
        - fingertip -> final   : green
        - base -> final        : red
        """
        img = image.copy()

        # --- 1) Build base->camera transform from cam->base ---
        # T_cam_base maps camera -> base, so invert to get base -> camera
        T_base_cam = np.linalg.inv(self.cam2base)
        R_bc = T_base_cam[:3, :3]  # rotation base->cam
        t_bc = T_base_cam[:3, 3]   # translation base->cam

        def project_point(p_base: np.ndarray):
            """Project a single 3D point in base frame to pixel coords (u,v)."""
            # ensure shape (3,)
            p_base = np.asarray(p_base).reshape(3)

            # base -> camera
            p_cam = R_bc @ p_base + t_bc
            X, Y, Z = p_cam

            if Z <= 0:
                return None  # behind camera, skip

            fx, fy = self.cam_k[0, 0], self.cam_k[1, 1]
            cx, cy = self.cam_k[0, 2], self.cam_k[1, 2]

            u = int(fx * (X / Z) + cx)
            v = int(fy * (Y / Z) + cy)

            h, w = img.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                return None  # outside image
            return (u, v)

        # --- 2) Project the three points ---
        pt_f = project_point(fingertip_pos)
        pt_b = project_point(base_pos)
        pt_g = project_point(final_pos)  # "goal"

        # --- 3) Draw segments if both endpoints are valid ---
        # colors are BGR in OpenCV
        if pt_f is not None and pt_b is not None:
            cv2.line(img, pt_f, pt_b, (255, 0, 0), thickness)   # blue: fingertip->base
        if pt_f is not None and pt_g is not None:
            cv2.line(img, pt_f, pt_g, (0, 255, 0), thickness)   # green: fingertip->final
        if pt_b is not None and pt_g is not None:
            cv2.line(img, pt_b, pt_g, (0, 0, 255), thickness)   # red: base->final

        return img

    def draw_mask_regions(self, img, color=(0,255,0), radius=2, gap=10):
        """
        Draw a dotted quadrilateral on img.

        corners: list or np.array of 4 (x,y) points in order
        """
        for obj in ["held_asset", "fixed_asset"]:
            corners = self.corners_dict[obj]
            pts = np.array(corners, dtype=np.int32)
            assert pts.shape == (4,2)

            # Loop through edges 0→1→2→3→0
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i+1) % 4]
                self.draw_dotted_line(img, p1, p2, color=color, radius=radius, gap=gap)

        return img
    
    def draw_dotted_line(self, img, p1, p2, color=(0,255,0), radius=2, gap=10):
        """
        Draw a dotted line from p1 to p2 on 'img'.

        p1, p2 : (x,y)
        color  : BGR tuple
        radius : radius of dots
        gap    : pixel spacing between dots
        """
        p1 = np.array(p1)
        p2 = np.array(p2)

        # vector and length
        diff = p2 - p1
        length = np.linalg.norm(diff)
        if length == 0:
            return img

        direction = diff / length
        num_dots = int(length // gap)

        for i in range(num_dots + 1):
            pt = (p1 + direction * i * gap).astype(int)
            cv2.circle(img, tuple(pt), radius, color, -1)

        return img
