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
#include <sophus/se3.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>

#include "filesystem"
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "visual_odometry_utils.h"
#include "pcl_map.h"
#include "visualization_utils.h"

#include <opencv2/calib3d.hpp>
#include "io_utils.h"
#include "camera.h"
#include <fmt/core.h>
#include "visual_odometry.h"
#include "reports.h"

int main() {

  const Camera camera{
      752.0, 480.0, {458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05}};
  const std::vector<std::string> image_paths = get_image_paths("../data/euroc_mav/V1_01_easy/cam0/data", "png");
  // TODO: Poses seem to be wrong. Bundle adjustment works hard to correct them.
  // Specify initial two poses to start with correct scale.
  const std::pair<int, int> initial_keyframe_indices{100, 120};
  const Pose initial_pose_1
      {Eigen::Quaterniond{-0.4277372, 0.62498984, -0.54384451, 0.36147164}, {0.87029955, 2.2052097, 0.92827237}};
  const Pose initial_pose_2
      {Eigen::Quaterniond{-0.44759128, 0.60619323, -0.53792524, 0.377926253}, {0.97336398, 2.25578283, 1.06085795}};

  VisualOdometry vo{camera};
  auto initialize_report = vo.initialize(
      cv::imread(image_paths[initial_keyframe_indices.first], cv::IMREAD_GRAYSCALE),
      cv::imread(image_paths[initial_keyframe_indices.second], cv::IMREAD_GRAYSCALE),
      initial_pose_1,
      initial_pose_2
  );
  if (initialize_report) {
    initialize_report->print();

    ReportWriter initialize_report_writer("../initialize_report.csv");
    initialize_report_writer.write(initialize_report.value());
  }


  // TODO: Remove
  show_keyframes({0, 1}, vo.map, 1, vo.camera);

  std::vector<Pose> trajectory{vo.map.keyframes[0].pose, vo.map.keyframes[1].pose};
  trajectory.reserve(image_paths.size());

  trajectory.push_back(vo.map.keyframes[0].pose);
  trajectory.push_back(vo.map.keyframes[1].pose);

  ReportWriter next_report_writer("../next_report.csv");
  ReportWriter keyframe_report_writer("../keyframe_report.csv");

  for (int i = 121; i < image_paths.size(); ++i) {
    cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
    auto [success, next_report, keyframe_report] = vo.next(image);

    if (next_report) {
      next_report->print();
      next_report_writer.write(next_report.value());
    }
    if (keyframe_report) {
      keyframe_report->print();
      keyframe_report_writer.write(keyframe_report.value());
    }

    if (!success) break;

    trajectory.push_back(vo.current_pose);
  }

  dump_trajectory(trajectory);

  return 0;

}

