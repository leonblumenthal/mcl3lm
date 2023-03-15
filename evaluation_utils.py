import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import numpy as np


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray


@dataclass
class Keyframe:
    pose: Pose
    corners: np.ndarray
    descriptors: np.ndarray


@dataclass
class Landmark:
    position: np.ndarray
    observations: dict[int, int]


@dataclass
class Map:
    keyframes: dict[int, Keyframe]
    landmarks: dict[int, Landmark]


Trajectory = list[Pose]


def read_map_dump(path: str = "map_dump"):
    with open(path) as f:
        lines = f.read().splitlines()

    keyframes_start = lines.index("keyframes")
    landmarks_start = lines.index("landmarks")

    vo_map = Map({}, {})

    keyframe_lines = lines[keyframes_start + 1 : landmarks_start]
    while keyframe_lines:
        try:
            split_index = keyframe_lines.index("")
        except ValueError:
            split_index = len(keyframe_lines)

        ls = keyframe_lines[:split_index]
        keyframe_lines = keyframe_lines[split_index + 1 :]

        index = int(ls[0])
        pose_values = [float(s) for s in ls[1].split(" ")]
        t = np.array(pose_values[:3])
        R = Rotation.from_quat(
            [pose_values[4], pose_values[5], pose_values[6], pose_values[3]]
        ).as_matrix()
        corners = []
        descriptors = []
        for line in ls[2:]:
            x, y, d = line.split(" ")
            corners.append((int(x), int(y)))
            descriptors.append([bool(int(s)) for s in d])
        corners = np.array(corners)
        descriptors = np.array(descriptors)

        vo_map.keyframes[index] = Keyframe(Pose(R, t), corners, descriptors)

    landmark_lines = lines[landmarks_start + 1 :]
    while landmark_lines:
        try:
            split_index = landmark_lines.index("")
        except ValueError:
            split_index = len(landmark_lines)

        ls = landmark_lines[:split_index]
        landmark_lines = landmark_lines[split_index + 1 :]

        index = int(ls[0])
        position = np.array([float(s) for s in ls[1].split(" ")])

        if np.linalg.norm(position) > 50:
            print("Skipped ", index)
            continue

        observations = {}
        for line in ls[2:]:
            keyframe_index, corner_index = line.split(" ")
            observations[int(keyframe_index)] = int(corner_index)

        vo_map.landmarks[index] = Landmark(position, observations)

    return vo_map


# def read_trajectory_dump(path: str) -> Trajectory:
#     df = pd.read_csv(path)

#     trajectory = []
#     for _, (px, py, pz, qw, qx, qy, qz) in df.iterrows():
#         trajectory.append(
#             Pose(Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), np.array([px, py, pz]))
#         )

#     return trajectory


def read_trajectory_dump(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df.iloc[:, :3]


def read_ply(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def read_colord_ply(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def read_ground_truth_trajectory(
    sensor_path: str,
    data_path: str,
    cam_data_path: str,
    initial_cam_indices: list[int] = [],
) -> tuple[np.ndarray, list]:
    with open(sensor_path) as f:
        sensor_data = yaml.safe_load(f)
    sensor_T = np.array(sensor_data["T_BS"]["data"]).reshape((4, 4))
    sensor_R = sensor_T[:3, :3]
    sensor_t = sensor_T[:3, 3]

    gt_df = pd.read_csv(data_path, index_col=0)
    gt_ts = gt_df.iloc[:, :3].to_numpy()
    gt_Rs = Rotation.from_quat(gt_df.iloc[:, [4, 5, 6, 3]].to_numpy()).as_matrix()

    cam_df = pd.read_csv(cam_data_path, index_col=0)

    def cam_to_gt(index):
        return np.argmin(np.abs(cam_df.index[index] - gt_df.index))

    if initial_cam_indices:
        cam_indices = initial_cam_indices + [
            cam_to_gt(i) for i in range(initial_cam_indices[-1] + 1, len(cam_df))
        ]
    else:
        cam_indices = [cam_to_gt(i) for i in range(len(cam_df))]

    trajectory = gt_ts[cam_indices] + gt_Rs[cam_indices] @ sensor_t
    initial_poses = [
        (
            Rotation.from_matrix(gt_Rs[cam_to_gt(i)] @ sensor_R).as_quat()[
                [3, 0, 1, 2]
            ],
            gt_ts[cam_to_gt(i)] + gt_Rs[cam_to_gt(i)] @ sensor_t,
        )
        for i in initial_cam_indices
    ]

    return trajectory, initial_poses


def sample_config(random_config: dict) -> dict:
    config = {}
    for key, value in random_config.items():
        if isinstance(value, tuple):
            a, b = value
            if isinstance(a, int) and isinstance(b, int):
                config[key] = np.random.randint(a, b)
            else:
                config[key] = np.random.uniform(a, b)
        elif isinstance(value, dict):
            config[key] = sample_config(value)
        else:
            config[key] = value

    return config


def get_random_values(random_config: dict, actual_config: dict) -> list:
    out = []
    for key, value in random_config.items():
        if isinstance(value, dict):
            ret = get_random_values(value, actual_config[key])
            if ret:
                out += [f"{key}.{v}" for v in ret]
        elif isinstance(value, tuple):
            out.append(f"{key}: {actual_config[key]}")
    return out


def get_cam_data(path: str):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["resolution"], data["intrinsics"], data["distortion_coefficients"]
