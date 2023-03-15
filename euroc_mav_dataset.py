"""Visualize the vision room 1 easy of the EuRoC dataset."""

import open3d as o3d
import numpy as np
import pandas as pd

pcd = o3d.io.read_point_cloud("data/euroc_mav/V1_01_easy/pointcloud0/data_colored.ply")

data = np.asarray(pcd.points)

df = pd.read_csv(
    "data/euroc_mav/V1_01_easy/state_groundtruth_estimate0/data.csv", index_col=0
)

lines = [(i, i + 1) for i in range(len(df) - 1)]

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(
        np.array(df[[" p_RS_R_x [m]", " p_RS_R_y [m]", " p_RS_R_z [m]"]])
    ),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector([[1.0, 0, 0]] * len(lines))


o3d.visualization.draw_geometries(
    [pcd, line_set],
    zoom=0.02,
    front=[1, 0, 0],
    up=[0, 0, 1],
    lookat=data.mean(axis=0).tolist(),
)
