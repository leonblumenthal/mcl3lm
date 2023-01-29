#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string_view>
#include <vector>
#include <algorithm>
#include "queue"

#include "ceres/ceres.h"
#include "pcl_map.h"
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "filesystem"
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "features.h"

#include <opencv2/calib3d.hpp>
#include "reports.h"
#include "visualization_utils.h"
#include "visual_odometry.h"
#include "io_utils.h"
#include "alignment.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

  ////////////////////////////////////////////////////////////////
  // Config

  const fs::path output_path{argv[1]};
  std::ifstream config_data_file(argv[2]);
  const json config_data = json::parse(config_data_file);
  config_data_file.close();

  // General
  const bool verbose{config_data["verbose"]};
  const bool visual{config_data["visual"]};
  const bool do_dump_map{config_data["do_dump_map"]};


  // Geometric map
  const std::string ply_path{config_data["geometric_map"]["ply_path"]};
  const Bounds vertices_bounds{
      config_data["geometric_map"]["vertices_bounds"]["min_x"],
      config_data["geometric_map"]["vertices_bounds"]["min_y"],
      config_data["geometric_map"]["vertices_bounds"]["min_z"],
      config_data["geometric_map"]["vertices_bounds"]["max_x"],
      config_data["geometric_map"]["vertices_bounds"]["max_y"], config_data["geometric_map"]["vertices_bounds"]["max_z"]
  };
  const double voxel_size{config_data["geometric_map"]["voxel_size"]};


  // Visual odometry;
  const std::string images_directory{config_data["visual_odometry"]["images_directory"]};
  const std::string images_ending{config_data["visual_odometry"]["images_ending"]};
  const Camera<double> camera{
      config_data["visual_odometry"]["camera"]["width"], config_data["visual_odometry"]["camera"]["height"], {
          config_data["visual_odometry"]["camera"]["parameters"]["fx"],
          config_data["visual_odometry"]["camera"]["parameters"]["fy"],
          config_data["visual_odometry"]["camera"]["parameters"]["cx"],
          config_data["visual_odometry"]["camera"]["parameters"]["cy"],
          config_data["visual_odometry"]["camera"]["parameters"]["k1"],
          config_data["visual_odometry"]["camera"]["parameters"]["k2"],
          config_data["visual_odometry"]["camera"]["parameters"]["p1"],
          config_data["visual_odometry"]["camera"]["parameters"]["p2"]
      }};
  const int initial_keyframe_index_1{config_data["visual_odometry"]["initial_keyframe_index_1"]},
      initial_keyframe_index_2{config_data["visual_odometry"]["initial_keyframe_index_2"]};
  const Pose initial_pose_1{
      Eigen::Quaterniond{
          config_data["visual_odometry"]["initial_pose_1"]["qw"],
          config_data["visual_odometry"]["initial_pose_1"]["qx"],
          config_data["visual_odometry"]["initial_pose_1"]["qy"], config_data["visual_odometry"]["initial_pose_1"]["qz"]
      }, Eigen::Vector3d(
          config_data["visual_odometry"]["initial_pose_1"]["x"],
          config_data["visual_odometry"]["initial_pose_1"]["y"],
          config_data["visual_odometry"]["initial_pose_1"]["z"]
      )
  };
  const Pose initial_pose_2{
      Eigen::Quaterniond{
          config_data["visual_odometry"]["initial_pose_2"]["qw"],
          config_data["visual_odometry"]["initial_pose_2"]["qx"],
          config_data["visual_odometry"]["initial_pose_2"]["qy"], config_data["visual_odometry"]["initial_pose_2"]["qz"]
      }, Eigen::Vector3d(
          config_data["visual_odometry"]["initial_pose_2"]["x"],
          config_data["visual_odometry"]["initial_pose_2"]["y"],
          config_data["visual_odometry"]["initial_pose_2"]["z"]
      )
  };
  const VisualOdometryConfig visual_odometry_config{{{
                                                         config_data["visual_odometry"]["initialize"]["keypoints"]["max_num"],
                                                         config_data["visual_odometry"]["initialize"]["keypoints"]["quality_level"],
                                                         config_data["visual_odometry"]["initialize"]["keypoints"]["min_distance_between"],
                                                         config_data["visual_odometry"]["initialize"]["keypoints"]["edge_margin"],
                                                         config_data["visual_odometry"]["initialize"]["keypoints"]["descriptor_angle_patch_radius"]
                                                     }, {
                                                         config_data["visual_odometry"]["initialize"]["matching"]["max_distance"],
                                                         config_data["visual_odometry"]["initialize"]["matching"]["max_second_to_first_distance_ratio"],
                                                         {
                                                             config_data["visual_odometry"]["initialize"]["matching"]["ransac"]["threshold"],
                                                             config_data["visual_odometry"]["initialize"]["matching"]["ransac"]["min_num_inliers"],
                                                             config_data["visual_odometry"]["initialize"]["matching"]["ransac"]["max_num_iterations"],
                                                         }},
                                                     config_data["visual_odometry"]["initialize"]["max_num_bundle_adjustment_iterations"]

                                                    }, {{
                                                            config_data["visual_odometry"]["next"]["keypoints"]["max_num"],
                                                            config_data["visual_odometry"]["next"]["keypoints"]["quality_level"],
                                                            config_data["visual_odometry"]["next"]["keypoints"]["min_distance_between"],
                                                            config_data["visual_odometry"]["next"]["keypoints"]["edge_margin"],
                                                            config_data["visual_odometry"]["next"]["keypoints"]["descriptor_angle_patch_radius"]
                                                        }, {
                                                            config_data["visual_odometry"]["next"]["matching"]["max_image_distance"],
                                                            config_data["visual_odometry"]["next"]["matching"]["max_descriptor_distance"],
                                                            config_data["visual_odometry"]["next"]["matching"]["max_second_to_first_distance_ratio"],
                                                            {
                                                                config_data["visual_odometry"]["next"]["matching"]["ransac"]["max_num_iterations"],
                                                                config_data["visual_odometry"]["next"]["matching"]["ransac"]["image_distance_threshold"],
                                                            }}, {{
                                                                     config_data["visual_odometry"]["next"]["keyframe"]["is_next"]["min_distance_to_last"],
                                                                     config_data["visual_odometry"]["next"]["keyframe"]["is_next"]["max_num_landmark_keypoint_inliers"],
                                                                     config_data["visual_odometry"]["next"]["keyframe"]["is_next"]["min_num_landmark_keypoint_inliers"],
                                                                 },
                                                                 config_data["visual_odometry"]["next"]["keyframe"]["max_num_keyframes"],
                                                                 {
                                                                     config_data["visual_odometry"]["next"]["keyframe"]["matching"]["max_distance"],
                                                                     config_data["visual_odometry"]["next"]["keyframe"]["matching"]["max_second_to_first_distance_ratio"],
                                                                     {
                                                                         config_data["visual_odometry"]["next"]["keyframe"]["matching"]["ransac"]["threshold"],
                                                                         config_data["visual_odometry"]["next"]["keyframe"]["matching"]["ransac"]["min_num_inliers"],
                                                                         config_data["visual_odometry"]["next"]["keyframe"]["matching"]["ransac"]["max_num_iterations"],
                                                                     }},
                                                                 config_data["visual_odometry"]["next"]["keyframe"]["max_num_bundle_adjustment_iterations"],
                                                        }}};


  // Alignment
  const bool do_alignment{config_data["alignment"]["do_alignment"]};
  const AlignConfig align_config{
      static_cast<double>(config_data["alignment"]["align"]["distance_threshold_start"]) * voxel_size,
      static_cast<double>(config_data["alignment"]["align"]["distance_threshold_end"]) * static_cast<double>(config_data["alignment"]["align"]["distance_threshold_start"]) * voxel_size,
      config_data["alignment"]["align"]["max_num_alignment_steps"],
      config_data["alignment"]["align"]["min_num_vertices_in_voxel"],
      config_data["alignment"]["align"]["standard_deviation_scale"],
      config_data["alignment"]["align"]["max_num_optimization_iterations"],
  };


  ////////////////////////////////////////////////////////////////
  // Execution

  // General

  if (do_dump_map) fs::create_directories(output_path / "visual_odometry_maps");

  // Generate the geometric map from the pointcloud.
  GeometricMap geometric_map{voxel_size, vertices_bounds};
  geometric_map.initialize(ply_path);


  // Initialize the visual odometry with the two initial images and poses.
  const std::vector<std::string> image_paths = get_image_paths(images_directory, images_ending);
  VisualOdometry visual_odometry{camera, true, visual_odometry_config};
  const std::optional<VisualOdometryInitializeReport> initialize_report = visual_odometry.initialize(
      cv::imread(image_paths[initial_keyframe_index_1], cv::IMREAD_GRAYSCALE),
      cv::imread(image_paths[initial_keyframe_index_2], cv::IMREAD_GRAYSCALE),
      initial_pose_1,
      initial_pose_2,
      initial_keyframe_index_1,
      initial_keyframe_index_2
  );

  if (initialize_report) {
    if (verbose) initialize_report->print();

    ReportWriter initialize_report_writer(output_path / "initialize_report.csv");
    initialize_report_writer.write(initialize_report.value());
  }
  if (visual) {
    show_keyframes(
        {initial_keyframe_index_1, initial_keyframe_index_2}, visual_odometry.map, initial_keyframe_index_2, camera
    );
  }


  // Run the visual odometry and align with the geometric map everytime a new keyframe was added.
  std::vector<Pose> raw_trajectory{
      visual_odometry.map.keyframes[initial_keyframe_index_1].pose,
      visual_odometry.map.keyframes[initial_keyframe_index_2].pose
  };
  raw_trajectory.reserve(image_paths.size());
  std::vector<Pose> transformed_trajectory{
      visual_odometry.map.keyframes[initial_keyframe_index_1].pose,
      visual_odometry.map.keyframes[initial_keyframe_index_2].pose
  };
  transformed_trajectory.reserve(image_paths.size());

  Sophus::Sim3d local_to_map_transformation{Eigen::Matrix4d::Identity()};
  std::vector<Sophus::Sim3d> transformations{local_to_map_transformation};

  ReportWriter next_report_writer(output_path / "next_report.csv");
  ReportWriter keyframe_report_writer(output_path / "keyframe_report.csv");

  int last_keyframe_index = -1;
  for (int frame_index = initial_keyframe_index_2 + 1; frame_index < image_paths.size(); ++frame_index) {
    // Align local visual odometry map with global geometric map.
    if (visual_odometry.last_keyframe_index != last_keyframe_index) {
      last_keyframe_index = visual_odometry.last_keyframe_index;

      if (do_dump_map) {
        dump_map(
            visual_odometry.map,
            output_path / fmt::format("visual_odometry_maps/{}", last_keyframe_index));
      }

      if (do_alignment) {
        // TODO: Check why optimization costs.
        align(local_to_map_transformation, geometric_map, visual_odometry.map, align_config);

        transformations.push_back(local_to_map_transformation);
      }

      if (visual) {
        show_keyframes(
            {last_keyframe_index}, visual_odometry.map, last_keyframe_index, camera, "Last keyframe pre alignment"
        );
      }

    }

    // Run visual odometry on new image.
    cv::Mat image = cv::imread(image_paths[frame_index], cv::IMREAD_GRAYSCALE);
    auto [success, next_report, keyframe_report] = visual_odometry.next(image);

    if (next_report) {
      if (verbose) next_report->print();
      next_report_writer.write(next_report.value());
    }
    if (keyframe_report) {
      if (verbose) keyframe_report->print();
      keyframe_report_writer.write(keyframe_report.value());
    }

    if (!success) {
      std::printf("Failure at frame {}\n", frame_index);
      if (!success) break;
    }

    raw_trajectory.push_back(visual_odometry.current_pose);
    if (do_alignment) {
      Pose transformed_pose = visual_odometry.current_pose;
      transformed_pose.translation() = local_to_map_transformation * transformed_pose.translation();
      transformed_trajectory.push_back(transformed_pose);
    }
  }

  dump_trajectory(
      raw_trajectory, output_path / "raw_trajectory.csv", {initial_keyframe_index_1, initial_keyframe_index_2}
  );

  if (do_alignment) {
    dump_trajectory(
        transformed_trajectory,
        output_path / "transformed_trajectory.csv",
        {initial_keyframe_index_1, initial_keyframe_index_2}
    );
    dump_transformations(transformations, output_path / "transformations.csv");
  }

  return 0;
}

