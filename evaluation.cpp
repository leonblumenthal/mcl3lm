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

namespace fs = std::filesystem;

int main() {

  ////////////////////////////////////////////////////////////////
  // Config

  // General
  const fs::path output_path{"../out/test"};
  const bool verbose{true};
  const bool visual{false};

  // Geometric map
  const std::string ply_path{"../data/euroc_mav/V1_01_easy/pointcloud0/data.ply"};
  const Bounds vertices_bounds{-4.73, -3.49, -0.04, 4.10, 5.16, 4.13};
  const double voxel_size{0.25};

  // Visual odometry;
  const std::string images_directory{"../data/euroc_mav/V1_01_easy/cam0/data"};
  const std::string images_ending{"png"};
  const Camera<double> camera{
      752.0, 480.0, {458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05}};
  const int initial_keyframe_index_1{100}, initial_keyframe_index_2{120};
  const Pose initial_pose_1
      {Eigen::Quaterniond{-0.4277372, 0.62498984, -0.54384451, 0.36147164}, {0.87029955, 2.2052097, 0.92827237}};
  const Pose initial_pose_2
      {Eigen::Quaterniond{-0.44759128, 0.60619323, -0.53792524, 0.377926253}, {0.97336398, 2.25578283, 1.06085795}};
  const VisualOdometryConfig visual_odometry_config{};

  // Alignment
  const bool do_alignment{false};
  const AlignConfig align_config;


  ////////////////////////////////////////////////////////////////
  // Execution

  // General
  fs::remove_all(output_path);
  fs::create_directories(output_path / "visual_odometry_maps/pre_alignment");
  if (do_alignment) fs::create_directories(output_path / "visual_odometry_maps/post_alignment");

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
  std::vector<Pose> trajectory{visual_odometry.map.keyframes[initial_keyframe_index_1].pose, visual_odometry.map.keyframes[initial_keyframe_index_2].pose};
  trajectory.reserve(image_paths.size());

  Sophus::Sim3d local_to_map_transformation{Eigen::Matrix4d::Identity()};
  std::vector<Sophus::Sim3d> transformations{local_to_map_transformation};

  ReportWriter next_report_writer(output_path / "next_report.csv");
  ReportWriter keyframe_report_writer(output_path / "keyframe_report.csv");

  int last_keyframe_index = -1;
  for (int frame_index = initial_keyframe_index_2 + 1; frame_index < image_paths.size(); ++frame_index) {
    // Align local visual odometry map with global geometric map.
    if (visual_odometry.last_keyframe_index != last_keyframe_index) {
      last_keyframe_index = visual_odometry.last_keyframe_index;

      dump_map(
          visual_odometry.map, output_path / fmt::format("visual_odometry_maps/pre_alignment/{}", last_keyframe_index));

      if (do_alignment) {
        // TODO: Check why optimization costs.
        align(local_to_map_transformation, geometric_map, visual_odometry.map, align_config);

        dump_map(
            visual_odometry.map,
            output_path / fmt::format("visual_odometry_maps/post_alignment/{}", last_keyframe_index));
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

    Pose transformed_pose = visual_odometry.current_pose;
    transformed_pose.translation() = local_to_map_transformation * transformed_pose.translation();
    trajectory.push_back(transformed_pose);
  }

  dump_trajectory(trajectory, output_path / "trajectory.csv");
  if (do_alignment) {
    dump_transformations(transformations, output_path / "transformations.csv");
  }

  return 0;
}

