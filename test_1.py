#!/usr/bin/python3

import numpy as np
import cv2 as cv
import open3d as o3d
import torch


FOCAL_LENGTH = 1734.04
BASELINE = 529.50
WIDTH = 1920
HEIGHT = 1080
N_DISP = 192
VMIN = 55
VMAX = 160


def pcd_from_stereo(imgL: np.ndarray, imgR: np.ndarray) -> np.ndarray:
    stereo = cv.StereoBM_create(numDisparities=N_DISP, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    disparity_norm = cv.normalize(disparity, None, alpha=VMIN, beta=VMAX, norm_type=cv.NORM_MINMAX)
    depth = np.divide(FOCAL_LENGTH * BASELINE, disparity_norm)
    depth_norm = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    return depth_norm

def pcd_from_midas(img: np.ndarray) -> np.ndarray:
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
    # Convert depth_midas to a valid range (e.g., 0 to 255)
    depth_midas = (depth_midas - depth_midas.min()) / (depth_midas.max() - depth_midas.min()) * 255
    return depth_midas.astype(np.uint8)

if __name__ == "__main__":
        
    stereo_depth = pcd_from_stereo(
        cv.imread("data/im0.png", cv.IMREAD_GRAYSCALE),
        cv.imread("data/im1.png", cv.IMREAD_GRAYSCALE)
    )
    cv.imwrite("data/depth_stereo.png", stereo_depth)
    
    midas_depth = pcd_from_midas(cv.imread("data/im0.png"))
    cv.imwrite("data/depth_midas.png", midas_depth)
