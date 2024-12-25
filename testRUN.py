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
   depth_norm = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
   cv.imwrite("data/depth_stereo.png", depth_norm)
   depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
   intrinsics = o3d.camera.PinholeCameraIntrinsic(
       width=WIDTH, height=HEIGHT, fx=FOCAL_LENGTH, fy=FOCAL_LENGTH, cx=WIDTH / 2, cy=HEIGHT / 2
   )
   pcd_stereo = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics)
   pcd_stereo.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
   ################################################################################################################################################

   return pcd_stereo


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
   #################################################################################################################################
   midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
   midas.to('cpu')
   midas.eval()
   img_input = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   img_input = cv.resize(img_input, (256, 256))
   img_input = img_input.transpose(2, 0, 1)
   img_input = torch.from_numpy(img_input).float() / 255.0
   img_input = img_input.unsqueeze(0)
   with torch.no_grad():
       depth_midas = midas(img_input)
   depth_midas = depth_midas.squeeze().cpu().numpy()
   depth_midas = cv.resize(depth_midas, (WIDTH, HEIGHT))
   depth_midas_norm = cv.normalize(depth_midas, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
   cv.imwrite("data/depth_midas.png", depth_midas_norm)
   depth_o3d = o3d.geometry.Image(depth_midas.astype(np.float32))
   intrinsics = o3d.camera.PinholeCameraIntrinsic(
       width=WIDTH, height=HEIGHT, fx=FOCAL_LENGTH, fy=FOCAL_LENGTH, cx=WIDTH / 2, cy=HEIGHT / 2
   )
   pcd_midas = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics)
   pcd_midas.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
   ################################################################################################################################################

   return pcd_midas


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
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(stereo_pcd)
    vis.add_geometry(midas_pcd)
    vis.run()
    vis.destroy_window()