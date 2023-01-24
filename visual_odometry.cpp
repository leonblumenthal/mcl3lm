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

#include <glog/logging.h>

#include "visual_odometry_utils.h"
#include "pcl_map.h"
#include "visualization_utils.h"

#include <opencv2/calib3d.hpp>
#include "io_utils.h"
#include "camera.h"

Camera camera{
    752.0, 480.0, {458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05}};
const std::vector<std::string> image_paths = get_image_paths("../data/euroc_mav/V1_01_easy/cam0/data", "png");

int main() {
  // Specify initial two poses to start with correct scale.
  const std::pair<int, int> initial_keyframe_indices{100, 120};
  std::pair<Sophus::SE3d, Sophus::SE3d> initial_poses{
      Sophus::SE3d{
          Eigen::Quaterniond{0.061541, -0.827324, -0.058484, -0.555272}, Eigen::Vector3d{0.880183, 2.140444, 0.949612,},
      }, Sophus::SE3d{
          Eigen::Quaterniond{0.063929, -0.809801, -0.049845, -0.581077},
          Eigen::Vector3d{0.981139, 2.190511, 1.081517}}};

  // Compute keypoints and descriptors for first two images.
  cv::Mat initial_image_1 = cv::imread(image_paths[initial_keyframe_indices.first], cv::IMREAD_GRAYSCALE);
  cv::Mat initial_image_2 = cv::imread(image_paths[initial_keyframe_indices.second], cv::IMREAD_GRAYSCALE);
  auto [initial_keypoints_1, initial_descriptors_1] = compute_keypoints_and_descriptors(initial_image_1);
  auto [initial_keypoints_2, initial_descriptors_2] = compute_keypoints_and_descriptors(initial_image_2);

  // Match initial keypoints and compute initial landmark positions with know initial relative pose.
  IndexMatches initial_matches = match(initial_descriptors_1, initial_descriptors_2);
  std::cout << initial_matches.size() << std::endl;
  auto [inliers, initial_relative_pose] =
      find_inliers_ransac(initial_matches, initial_keypoints_1, initial_keypoints_2, camera);
  initial_matches = inliers;
  std::cout << initial_matches.size() << std::endl;
  // TODO
  initial_poses.first = Pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  initial_poses.second = initial_relative_pose;
  initial_relative_pose = initial_poses.first.inverse() * initial_poses.second;
  Positions initial_landmark_positions =
      triangulate(initial_matches, initial_keypoints_1, initial_keypoints_2, camera, initial_relative_pose);

  std::cout << initial_relative_pose.unit_quaternion() << std::endl;
  std::cout << initial_relative_pose.translation() << std::endl;


  // Initialize map with first two keyframes and triangulated landmarks observed from both frames.
  Map map;
  map.keyframes[initial_keyframe_indices.first] = {
      initial_poses.first, initial_keypoints_1, initial_descriptors_1
  };
  map.keyframes[initial_keyframe_indices.second] = {
      initial_poses.second, initial_keypoints_2, initial_descriptors_2
  };
  for (int i = 0; i < initial_matches.size(); ++i) {
    const auto &match = initial_matches[i];
    map.landmarks[map.next_landmark_index++] = {
        initial_landmark_positions[i],
        {{initial_keyframe_indices.first, match.first}, {initial_keyframe_indices.second, match.second}}};
  }

  std::cout << "Initialized map from keyframes " << initial_keyframe_indices.first << " and "
            << initial_keyframe_indices.second << std::endl;
  std::cout << "#Landmarks: " << map.landmarks.size() << std::endl;

  dump_map(map, "../map_dump_pre");

  std::cout << (map.keyframes[initial_keyframe_indices.first].pose.translation() - map.keyframes[initial_keyframe_indices.second].pose.translation()).norm() << std::endl;

  bundle_adjustment(map, camera, 100);

  std::cout << (map.keyframes[initial_keyframe_indices.first].pose.translation() - map.keyframes[initial_keyframe_indices.second].pose.translation()).norm() << std::endl;

  dump_map(map);

  // show_keyframes(image_paths, {100, 120}, map, camera);

  int last_keyframe_index = initial_keyframe_indices.second;
  Pose current_pose = map.keyframes[last_keyframe_index].pose;

  // Handle new images.
  for (int new_image_index = last_keyframe_index + 1; new_image_index < image_paths.size(); ++new_image_index) {

    // Compute keypoints and descriptors for new image.
    cv::Mat new_image = cv::imread(image_paths[new_image_index], cv::IMREAD_GRAYSCALE);
    auto [new_keypoints, new_descriptors] = compute_keypoints_and_descriptors(new_image);

    // Localize new image by matching with landmarks projected into last pose.
    auto [new_pose, landmark_matches] = localize(new_keypoints, new_descriptors, map, current_pose, camera);
    current_pose = new_pose;

    if (landmark_matches.size() < 5) {
      std::cout << "Less than 5 landmark matches at image " << new_image_index << std::endl;
      break;
    }

    const double
        distance_to_last = (current_pose.translation() - map.keyframes[last_keyframe_index].pose.translation()).norm();
    std::cout << new_image_index << " - distance to last keyframe: " << distance_to_last << ", #landmarck matches: " << landmark_matches.size() << std::endl;

    // Add new keyframe.
    if (distance_to_last >= 1 && (landmark_matches.size() < 20 || last_keyframe_index <= new_image_index - 10)) {

      Keyframe &last_keyframe = map.keyframes[last_keyframe_index];

      std::unordered_set<int> last_used_corner_indices;
      for (const auto &[_, landmark] : map.landmarks) {
        if (landmark.observations.contains(last_keyframe_index)) {
          last_used_corner_indices.insert(landmark.observations.at(last_keyframe_index));
        }
      }
      // std::cout << "# last unused corners: " << last_keyframe.keypoints.size() - last_used_corner_indices.size()
      //           << std::endl;

      std::unordered_set<int> new_used_corner_indices;
      for (const auto &[_, corner_index] : landmark_matches) {
        new_used_corner_indices.insert(corner_index);
      }
      // std::cout << "# new unused corners: " << new_keypoints.size() - landmark_matches.size() << std::endl;

      std::vector<Eigen::Vector2i> last_unused_corners;
      std::vector<int> last_unused_corner_indices;
      std::vector<std::bitset<256>> last_unused_descriptors;
      for (int i = 0; i < last_keyframe.keypoints.size(); ++i) {
        if (last_used_corner_indices.contains(i)) continue;
        last_unused_corners.push_back(last_keyframe.keypoints[i]);
        last_unused_corner_indices.push_back(i);
        last_unused_descriptors.push_back(last_keyframe.descriptors[i]);
      }

      std::vector<Eigen::Vector2i> new_unused_corners;
      std::vector<int> new_unused_corner_indices;
      std::vector<std::bitset<256>> new_unused_descriptors;
      for (int i = 0; i < new_keypoints.size(); ++i) {
        if (new_used_corner_indices.contains(i)) continue;
        new_unused_corners.push_back(new_keypoints[i]);
        new_unused_corner_indices.push_back((i));
        new_unused_descriptors.push_back(new_descriptors[i]);
      }

      // TODO: Better inlier filtering.
      auto nl_matches = match(new_unused_descriptors, last_unused_descriptors, 70, 1.2);
      // TODO: nl_points are not normalized to current pose.
      auto [nlr_inliers, _] =
          find_inliers_ransac(nl_matches, new_unused_corners, last_unused_corners, camera);
      auto [nl_inliers, nl_points] = find_inliers_essential(
          new_unused_corners, last_unused_corners, nlr_inliers, current_pose, last_keyframe.pose, camera
      );

      std::cout << "# inliers: " << nl_matches.size() << " -> " << nlr_inliers.size() << " -> " << nl_inliers.size()
                << std::endl;

      auto nl_projected = camera.project(nl_points);

      // Add new keyframe.
      map.keyframes[new_image_index] = Keyframe{current_pose, new_keypoints, new_descriptors};

      // TODO: Add observations of earlier keyframes?
      // Add new landmarks with new and last observations.
      for (int i = 0; i < nl_inliers.size(); ++i) {
        map.landmarks[map.next_landmark_index++] = Landmark{
            current_pose * nl_points[i], {{new_image_index, new_unused_corner_indices[nl_inliers[i].first]},
                                          {last_keyframe_index, last_unused_corner_indices[nl_inliers[i].second]},
            }};

      }

      // Add current observations.
      for (const auto &[landmark_index, corner_index] : landmark_matches) {
        map.landmarks[landmark_index].observations[new_image_index] = corner_index;
      }

      dump_map(map, "../map_dump_pre");

      bundle_adjustment(map, camera);

      dump_map(map);

      current_pose = map.keyframes[new_image_index].pose;

      last_keyframe_index = new_image_index;

    }
    //
    // cv::Mat rgb_image = cv::imread(image_paths[new_image_index]);
    // draw_new_pose(rgb_image, current_pose, landmark_matches, new_keypoints, map, camera);
    // cv::imshow("", rgb_image);
    // cv::waitKey(0);

  }
  //
  //     for (int i = 0; i < nl_inliers.size(); ++i) {
  //       const auto &[new_index, last_index] = nl_inliers[i];
  //       const auto &p = nl_projected[i];
  //       cv::drawMarker(
  //           rgb_image_new,
  //           cv::Point2d(new_unused_corners[new_index].x(), new_unused_corners[new_index].y()),
  //           CV_RGB(255, 255, 0),
  //           cv::MarkerTypes::MARKER_TILTED_CROSS,
  //           5,
  //           1
  //       );
  //       cv::drawMarker(
  //           rgb_image_new,
  //           cv::Point2d(nl_projected[i].x(), nl_projected[i].y()),
  //           CV_RGB(255, 255, 0),
  //           cv::MarkerTypes::MARKER_DIAMOND,
  //           5,
  //           1
  //       );
  //       cv::line(
  //           rgb_image_new,
  //           cv::Point2d(new_unused_corners[new_index].x(), new_unused_corners[new_index].y()),
  //           cv::Point2d(nl_projected[i].x(), nl_projected[i].y()),
  //           CV_RGB(255, 255, 0));
  //     }
  //
  //     cv::Mat rgb_image_last = cv::imread(image_paths[last_keyframe_index]);
  //
  //     for (int i = 0; i < last_keyframe.corners.size(); ++i) {
  //
  //       cv::drawMarker(
  //           rgb_image_last,
  //           cv::Point2d(last_keyframe.corners[i].x(), last_keyframe.corners[i].y()),
  //           last_used_corner_indices.contains(i) ? CV_RGB(0, 255, 0) : CV_RGB(255, 0, 0),
  //           cv::MarkerTypes::MARKER_TILTED_CROSS,
  //           5,
  //           1
  //       );
  //     }
  //     for (const auto &[_, landmark] : map.landmarks) {
  //       auto local_p = last_keyframe.pose.inverse() * landmark.position;
  //       if (local_p.z() < 0) continue;
  //       auto image_p = camera.project(local_p);
  //       if (image_p.x() < 0 || image_p.y() < 0 || image_p.x() > camera.width || image_p.y() > camera.height)
  //         continue;
  //       cv::drawMarker(
  //           rgb_image_last,
  //           cv::Point2d(image_p.x(), image_p.y()),
  //           landmark.observations.contains(last_keyframe_index) ? CV_RGB(0, 255, 0) : CV_RGB(0, 0, 255),
  //           cv::MarkerTypes::MARKER_DIAMOND,
  //           5,
  //           1
  //       );
  //
  //     }
  //
  //     for (int i = 0; i < nl_inliers.size(); ++i) {
  //       const auto &[new_index, last_index] = nl_inliers[i];
  //       const auto &p = nl_projected[i];
  //       cv::drawMarker(
  //           rgb_image_new,
  //           cv::Point2d(new_unused_corners[new_index].x(), new_unused_corners[new_index].y()),
  //           CV_RGB(0, 255, 255),
  //           cv::MarkerTypes::MARKER_TILTED_CROSS,
  //           5,
  //           1
  //       );
  //       cv::drawMarker(
  //           rgb_image_new,
  //           cv::Point2d(nl_projected[i].x(), nl_projected[i].y()),
  //           CV_RGB(0, 255, 255),
  //           cv::MarkerTypes::MARKER_DIAMOND,
  //           5,
  //           1
  //       );
  //       cv::line(
  //           rgb_image_new,
  //           cv::Point2d(new_unused_corners[new_index].x(), new_unused_corners[new_index].y()),
  //           cv::Point2d(nl_projected[i].x(), nl_projected[i].y()),
  //           CV_RGB(0, 255, 255));
  //
  //       const auto last_p = camera.project(last_keyframe.pose.inverse() * current_pose * nl_points[i]);
  //       cv::drawMarker(
  //           rgb_image_last,
  //           cv::Point2d(last_p.x(), last_p.y()),
  //           CV_RGB(0, 255, 255),
  //           cv::MarkerTypes::MARKER_DIAMOND,
  //           5,
  //           1
  //       );
  //       cv::drawMarker(
  //           rgb_image_last,
  //           cv::Point2d(last_unused_corners[last_index].x(), last_unused_corners[last_index].y()),
  //           CV_RGB(0, 255, 255),
  //           cv::MarkerTypes::MARKER_TILTED_CROSS,
  //           5,
  //           1
  //       );
  //       cv::line(
  //           rgb_image_last,
  //           cv::Point2d(last_unused_corners[last_index].x(), last_unused_corners[last_index].y()),
  //           cv::Point2d(last_p.x(), last_p.y()),
  //           CV_RGB(0, 255, 255));
  //     }
  //
  //     cv::imshow("last", rgb_image_last);
  //
  //     last_keyframe_index = new_image_index;
  //
  //   }
  //
  //   cv::imshow("", rgb_image_new);
  //
  //   if (new_image_index >= 500) cv::waitKey(0);
  //
  // }

}

