#!/usr/bin/python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import open3d as o3d
import torch

FOCAL_LENGTH = 1734.04
BASELINE = 529.50
WIDTH = 1920
HEIGHT = 1080
N_DISP = 192
VMIN = 55
VMAX = 160


def pcd_from_stereo(imgL: np.ndarray, imgR: np.ndarray) -> o3d.geometry.PointCloud:
   """
   Input
   imgL: Left image
   imgR: Right image
   Returns
   pcd_stereo: Point cloud
   """

   """
   INSTRUCTIONS
   Write this function to do the following:
   
   1. Compute the disparity map using the stereo block matching algorithm as in the opencv tutorial
   2. Normalize the disparity map to the range specified by VMIN and VMAX
   3. Compute the depth map using the formula: depth = FOCAL_LENGTH * BASELINE / disparity
   4. Normalize the depth map to the range 0-255 and invert it
   5. Convert the depth map to open3d image
   6. Convert the open3d image to point cloud and store it in the variable pcd_stereo (you might have to change the data type of the depth map to np.float32 if the open3d function throws an error)
   """

   ##################################################################################################################################
   stereo = cv.StereoBM_create(numDisparities=N_DISP, blockSize=15)
   disparity = stereo.compute(imgL, imgR)
   disparity_norm = cv.normalize(disparity, None, alpha=VMIN, beta=VMAX, norm_type=cv.NORM_MINMAX)
   depth = np.divide(FOCAL_LENGTH * BASELINE, disparity_norm)
   depth_norm = cv.normalize(depth, None, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
   cv.imwrite("data/depth_stereo.png", depth_norm * 255.0)
   pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_norm), o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FOCAL_LENGTH, FOCAL_LENGTH, WIDTH // 2, HEIGHT // 2))

   ################################################################################################################################################

   return pcd


def pcd_from_midas(img: np.ndarray) -> o3d.geometry.PointCloud:
   """
   Input
   img: RGB image
   Returns
   pcd_midas: Point cloud
   """

   """
   INSTRUCTIONS
   Write this function to do the following:
   1. Follow the midas depth estimation tutorial to get the depth map
   2. Convert the depth map to open3d image
   3. Convert the open3d image to point cloud and store in the pcd_midas variable
   """
   ##################################################################################################################################
   midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
   midas.to('cpu')
   midas.eval()
   img_input = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   img_input = cv.resize(img_input, (384, 384))  # Resize the input image to 384x384
   img_input = img_input.transpose(2, 0, 1)
   img_input = torch.from_numpy(img_input).float() / 255.0
   img_input = img_input.unsqueeze(0)
   with torch.no_grad():
       depth_midas = midas(img_input)
   depth_midas = depth_midas.squeeze().cpu().numpy()
   depth_midas = (depth_midas - depth_midas.min()) / (depth_midas.max() - depth_midas.min())
   depth_midas = cv.bilateralFilter(depth_midas, 9, 75, 75)
   depth_midas_viz = (depth_midas * 255).astype(np.uint8)
   cv.imwrite("data/depth_midas.png", depth_midas_viz)
   pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_midas.astype(np.float32)), o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FOCAL_LENGTH, FOCAL_LENGTH, WIDTH // 2, HEIGHT // 2))
   ################################################################################################################################################

   return pcd


if __name__ == "__main__":
    """
    DO NOT CHANGE ANYTHING HERE
    """
    stereo_pcd = pcd_from_stereo(
        cv.imread("data/im0.png", cv.IMREAD_GRAYSCALE),
        cv.imread("data/im1.png", cv.IMREAD_GRAYSCALE),
    )
    o3d.io.write_point_cloud("data/pointcloud_stereo.ply", stereo_pcd)

    midas_pcd = pcd_from_midas(cv.imread("data/im0.png"))
    o3d.io.write_point_cloud("data/pointcloud_midas.ply", midas_pcd)

    # Visualize point clouds
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(stereo_pcd)
    # vis.add_geometry(midas_pcd)
    # vis.run()
    # vis.destroy_window()