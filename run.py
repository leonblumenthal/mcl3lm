"""Run the approach interactively with the vision room 1 easy of the EuRoC dataset.

Hold the spacebar to jump the enxt keyframe.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from evaluation_utils import get_cam_data, read_ground_truth_trajectory, read_ply

initial_keyframe_index_1 = 100
initial_keyframe_index_2 = 120
gt_trajectory, initial_poses = read_ground_truth_trajectory(
    "data/euroc_mav/V1_01_easy/cam0/sensor.yaml",
    "data/euroc_mav/V1_01_easy/state_groundtruth_estimate0/data.csv",
    "data/euroc_mav/V1_01_easy/cam0/data.csv",
    [initial_keyframe_index_1, initial_keyframe_index_2],
)
gt_pcl = read_ply("data/euroc_mav/V1_01_easy/pointcloud0/data.ply")

(min_x, min_y, min_z), (max_x, max_y, max_z) = np.quantile(
    gt_pcl, [0.0001, 0.9999], axis=0
)

(width, height), (fx, fy, cx, cy), (k1, k2, p1, p2) = get_cam_data(
    "data/euroc_mav/V1_01_easy/cam0/sensor.yaml"
)

config = {
    "verbose": True,
    "visual": True,
    "do_dump_map": False,
    "geometric_map": {
        "ply_path": "data/euroc_mav/V1_01_easy/pointcloud0/data.ply",
        "vertices_bounds": {
            "min_x": min_x,
            "min_y": min_y,
            "min_z": min_z,
            "max_x": max_x,
            "max_y": max_y,
            "max_z": max_z,
        },
        "voxel_size": 0.3,
    },
    "visual_odometry": {
        "images_directory": "data/euroc_mav/V1_01_easy/cam0/data",
        "images_ending": "png",
        "camera": {
            "width": width,
            "height": height,
            "parameters": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "k1": k1,
                "k2": k2,
                "p1": p1,
                "p2": p2,
            },
        },
        "initial_keyframe_index_1": initial_keyframe_index_1,
        "initial_keyframe_index_2": initial_keyframe_index_2,
        "initial_pose_1": {
            "qw": initial_poses[0][0][0],
            "qx": initial_poses[0][0][1],
            "qy": initial_poses[0][0][2],
            "qz": initial_poses[0][0][3],
            "x": initial_poses[0][1][0],
            "y": initial_poses[0][1][1],
            "z": initial_poses[0][1][2],
        },
        "initial_pose_2": {
            "qw": initial_poses[1][0][0],
            "qx": initial_poses[1][0][1],
            "qy": initial_poses[1][0][2],
            "qz": initial_poses[1][0][3],
            "x": initial_poses[1][1][0],
            "y": initial_poses[1][1][1],
            "z": initial_poses[1][1][2],
        },
        "initialize": {
            "keypoints": {
                "max_num": 760,
                "quality_level": 0.012,
                "min_distance_between": 9,
                "edge_margin": 20,
                "descriptor_angle_patch_radius": 15,
            },
            "matching": {
                "max_distance": 80,
                "max_second_to_first_distance_ratio": 1.18,
                "ransac": {
                    "threshold": 5e-5,
                    "min_num_inliers": 20,
                    "max_num_iterations": 100,
                },
            },
            "max_num_bundle_adjustment_iterations": 100,
        },
        "next": {
            "keypoints": {
                "max_num": 750,
                "quality_level": 0.011,
                "min_distance_between": 9,
                "edge_margin": 20,
                "descriptor_angle_patch_radius": 15,
            },
            "matching": {
                "max_image_distance": 23,
                "max_descriptor_distance": 84,
                "max_second_to_first_distance_ratio": 1.16,
                "ransac": {"max_num_iterations": 100, "image_distance_threshold": 3.6},
            },
            "keyframe": {
                "is_next": {
                    "min_distance_to_last": 0.3,
                    "max_num_landmark_keypoint_inliers": 60,
                    "min_num_landmark_keypoint_inliers": 18,
                },
                "max_num_keyframes": 29,
                "matching": {
                    "max_distance": 86,
                    "max_second_to_first_distance_ratio": 1.17,
                    "ransac": {
                        "threshold": 5e-5,
                        "min_num_inliers": 20,
                        "max_num_iterations": 100,
                    },
                    "max_epipolar_error": 3e-3,
                },
                "max_num_bundle_adjustment_iterations": 100,
            },
        },
    },
    "alignment": {
        "do_alignment": True,
        "align": {
            "distance_threshold_start": 2.0,
            "distance_threshold_end": 1.0,
            "max_num_alignment_steps": 10,
            "min_num_vertices_in_voxel": 10,
            "standard_deviation_scale": 3,
            "max_num_optimization_iterations": 20,
        },
    },
}

out_path = Path("out/test")

if out_path.exists():
    shutil.rmtree(out_path)
out_path.mkdir(parents=True)

with open(out_path / "config.json", "w") as f:
    json.dump(config, f)


parser = argparse.ArgumentParser()
parser.add_argument("evaluation_binary_path", type=str)
args = parser.parse_args()

subprocess.run([args.evaluation_binary_path, out_path, out_path / "config.json"])
