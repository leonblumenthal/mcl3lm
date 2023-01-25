#ifndef VISUAL_ODOMETRY_H_
#define VISUAL_ODOMETRY_H_

#include "visualization_utils.h"
#include "features.h"
#include "filesystem"
#include <Eigen/Dense>
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>
#include <vector>
#include "camera.h"
#include <fmt/core.h>
#include "visualization_utils.h"

struct VisualOdometry {

  const Camera<> camera;

  Map map;
  Pose current_pose;
  int last_keyframe_index;
  int current_frame_index;

  std::queue<int> keyframe_indices;

  explicit VisualOdometry(const Camera<> &camera) : camera(camera) {
    current_pose = Pose::rotX(0);
  }

  void initialize(const cv::Mat &image_1, const cv::Mat &image_2, double distance_between) {

    constexpr int max_num_keypoints = 500;
    constexpr double keypoint_quality_level = 0.01;
    constexpr int min_distance_between_keypoints = 8;
    constexpr int keypoint_edge_margin = 16;
    constexpr int descriptor_angle_patch_radius = 15;
    auto [keypoints_1, descriptors_1] = compute_keypoints_and_descriptors(
        image_1,
        max_num_keypoints,
        keypoint_quality_level,
        min_distance_between_keypoints,
        keypoint_edge_margin,
        descriptor_angle_patch_radius
    );
    auto [keypoints_2, descriptors_2] = compute_keypoints_and_descriptors(
        image_2,
        max_num_keypoints,
        keypoint_quality_level,
        min_distance_between_keypoints,
        keypoint_edge_margin,
        descriptor_angle_patch_radius
    );

    constexpr int max_distance = 100;
    constexpr double max_second_to_first_distance_ratio = 1.1;
    IndexMatches matches = match(descriptors_1, descriptors_2, max_distance, max_second_to_first_distance_ratio);

    constexpr double ransac_threshold = 5e-6;
    constexpr int ransac_min_num_inliers = 16;
    constexpr int ransac_max_iterations = 100;
    auto [inlier_matches, relative_pose] = find_inliers_ransac(
        matches, keypoints_1, keypoints_2, camera, ransac_threshold, ransac_min_num_inliers, ransac_max_iterations
    );
    relative_pose.translation() *= distance_between;

    map.keyframes[0] = {
        Pose::rotX(0), keypoints_1, descriptors_1, image_1
    };
    map.keyframes[1] = {
        relative_pose, keypoints_2, descriptors_2, image_2
    };

    Positions landmark_positions = triangulate(inlier_matches, map.keyframes[0], map.keyframes[1], camera);
    for (int i = 0; i < inlier_matches.size(); ++i) {
      const auto &match = inlier_matches[i];
      map.landmarks[map.next_landmark_index++] = {
          landmark_positions[i], {{0, match.first}, {1, match.second}}};
    }

    constexpr int max_num_iterations = 100;
    bundle_adjustment(map, camera, max_num_iterations);

    current_pose = map.keyframes[1].pose;
    last_keyframe_index = 1;
    current_frame_index = 1;

    keyframe_indices.push(0);
    keyframe_indices.push(1);

    fmt::print("Initialized VO Map with {} landmarks:\n", map.landmarks.size());
    fmt::print("  #keypoints: {}, {}\n", keypoints_1.size(), keypoints_2.size());
    fmt::print(
        "  #inliers: {} ({:.0f}% matches)\n", inlier_matches.size(), 100.0 * inlier_matches.size() / matches.size());
    fmt::print("\n");
  }

  bool next(const cv::Mat &image) {
    ++current_frame_index;

    constexpr int max_num_keypoints = 500;
    constexpr double keypoint_quality_level = 0.01;
    constexpr int min_distance_between_keypoints = 8;
    // This is also influenced by the patterns and should not be smaller than 20.
    constexpr int keypoint_edge_margin = 20;
    constexpr int descriptor_angle_patch_radius = 15;
    auto [keypoints, descriptors] = compute_keypoints_and_descriptors(
        image,
        max_num_keypoints,
        keypoint_quality_level,
        min_distance_between_keypoints,
        keypoint_edge_margin,
        descriptor_angle_patch_radius
    );

    std::unordered_map<int, Point> landmark_points = project_landmarks(map.landmarks, current_pose, camera);

    constexpr double max_image_distance = 20;
    constexpr double max_descriptor_distance = 70;
    constexpr double max_second_to_first_distance_ratio = 1.2;
    IndexMatches landmark_keypoint_matches = match_landmarks(
        keypoints,
        descriptors,
        map,
        landmark_points,
        max_image_distance,
        max_descriptor_distance,
        max_second_to_first_distance_ratio
    );

    constexpr int ransac_max_iterations = 100;
    constexpr double ransac_image_distance_threshold = 3;
    auto [pose, landmark_keypoint_inliers] = localize(
        keypoints, map, landmark_keypoint_matches, camera, ransac_max_iterations, ransac_image_distance_threshold
    );

    const double
        distance_to_last_keyframe = (map.keyframes[last_keyframe_index].pose.translation() - pose.translation()).norm();
    const double distance_to_last_frame = (current_pose.translation() - pose.translation()).norm();
    const Pose pose_to_last_keyframe = map.keyframes[last_keyframe_index].pose.inverse() * pose;
    const Pose pose_to_last_frame = current_pose.inverse() * pose;

    fmt::print("Frame {}:\n", current_frame_index);
    fmt::print(
        "  #visible landmarks: {} ({:.0f}% all)\n",
        landmark_points.size(),
        100.0 * landmark_points.size() / map.landmarks.size());
    fmt::print("  #keypoints: {}\n", keypoints.size());
    fmt::print(
        "  #matches: {} ({:.0f}% keypoints, {:.0f}% visible landmarks)\n",
        landmark_keypoint_matches.size(),
        100.0 * landmark_keypoint_matches.size() / keypoints.size(),
        100.0 * landmark_keypoint_matches.size() / landmark_points.size());
    fmt::print(
        "  #inliers: {} ({:.0f}% matches, {:.0f}% visible landmarks)\n",
        landmark_keypoint_inliers.size(),
        100.0 * landmark_keypoint_inliers.size() / landmark_keypoint_matches.size(),
        100.0 * landmark_keypoint_inliers.size() / landmark_points.size());
    fmt::print("  distance to last frame: {:.2}, keyframe: {:.2}\n", distance_to_last_frame, distance_to_last_keyframe);
    fmt::print(
        "  translation to last frame: {:.4f}, {:.4f}, {:.4f}, keyframe: {:.4f}, {:.4f}, {:.4f}\n",
        pose_to_last_frame.translation().x(),
        pose_to_last_frame.translation().y(),
        pose_to_last_frame.translation().z(),
        pose_to_last_keyframe.translation().x(),
        pose_to_last_keyframe.translation().y(),
        pose_to_last_keyframe.translation().z());
    fmt::print(
        "  rotation to last frame: {:.4f}, {:.4f}, {:.4f}, keyframe: {:.4f}, {:.4f}, {:.4f}\n",
        pose_to_last_frame.angleX(),
        pose_to_last_frame.angleY(),
        pose_to_last_frame.angleZ(),
        pose_to_last_keyframe.angleX(),
        pose_to_last_keyframe.angleY(),
        pose_to_last_keyframe.angleZ());

    if (landmark_keypoint_inliers.size() < 5) return false;

    current_pose = pose;



    // TODO: This value depends on the initial scale.
    const double min_distance_to_last_keyframe = 0.2;
    const int max_num_landmark_keypoint_inliers = 60;
    const int min_num_landmark_keypoint_inliers = 20;
    bool is_next_keyframe = (distance_to_last_keyframe > min_distance_to_last_keyframe
        && landmark_keypoint_inliers.size() < max_num_landmark_keypoint_inliers)
        || landmark_keypoint_inliers.size() < min_num_landmark_keypoint_inliers;

    if (is_next_keyframe) {

      int new_keyframe_index = current_frame_index;
      map.keyframes[new_keyframe_index] = Keyframe{current_pose, keypoints, descriptors, image};
      for (const auto &[landmark_index, keypoint_index] : landmark_keypoint_inliers) {
        map.landmarks[landmark_index].observations[new_keyframe_index] = keypoint_index;
      }

      keyframe_indices.push(new_keyframe_index);

      int num_removed_landmarks = map.landmarks.size();
      int removed_keyframe_index = -1;
      constexpr int max_num_keyframes = 10;
      // This should only be executed once actually.
      if (keyframe_indices.size() > max_num_keyframes) {
        removed_keyframe_index = keyframe_indices.front();
        keyframe_indices.pop();
        remove_keyframe(removed_keyframe_index, map);
      }
      num_removed_landmarks -= map.landmarks.size();

      std::unordered_set<int> new_used_keypoint_indices = get_used_keypoint_indices(map, new_keyframe_index);
      std::unordered_set<int> last_used_keypoint_indices = get_used_keypoint_indices(map, last_keyframe_index);

      Keyframe &new_keyframe = map.keyframes[new_keyframe_index];
      Keyframe &last_keyframe = map.keyframes[last_keyframe_index];

      constexpr int max_distance = 70;
      constexpr double max_second_to_first_distance_ratio = 1.2;
      IndexMatches new_last_matches = match(
          new_keyframe.descriptors,
          last_keyframe.descriptors,
          max_distance,
          max_second_to_first_distance_ratio,
          new_used_keypoint_indices,
          last_used_keypoint_indices
      );

      // cv::Mat new_rgb_image;
      // cv::Mat last_rgb_image;
      // cv::cvtColor(image, new_rgb_image, cv::COLOR_GRAY2BGR);
      // cv::cvtColor(last_keyframe.image, last_rgb_image, cv::COLOR_GRAY2BGR);


      constexpr double ransac_threshold = 5e-6;
      constexpr int ransac_min_num_inliers = 16;
      constexpr int ransac_max_iterations = 100;
      auto [new_last_inliers_ransac, _] = find_inliers_ransac(
          new_last_matches,
          new_keyframe.keypoints,
          last_keyframe.keypoints,
          camera,
          ransac_threshold,
          ransac_min_num_inliers,
          ransac_max_iterations
      );
      constexpr double max_epipolar_error = 1e-3;
      IndexMatches new_last_inliers_epipolar =
          find_inliers_epipolar(new_last_inliers_ransac, new_keyframe, last_keyframe, camera, max_epipolar_error);

      Positions new_landmark_positions = triangulate(new_last_inliers_epipolar, new_keyframe, last_keyframe, camera);
      for (int i = 0; i < new_last_inliers_epipolar.size(); ++i) {
        const auto &[new_keypoint_index, last_keypoint_index] = new_last_inliers_epipolar[i];
        map.landmarks[map.next_landmark_index++] = {
            new_landmark_positions[i],
            {{new_keyframe_index, new_keypoint_index}, {last_keyframe_index, last_keypoint_index}}};
      }

      constexpr int max_num_iterations = 20;
      bundle_adjustment(map, camera, max_num_iterations);

      current_pose = new_keyframe.pose;

      if (removed_keyframe_index > -1) {
        fmt::print("Removed keyframe {} and {} landmarks\n", removed_keyframe_index, num_removed_landmarks);
      }
      fmt::print("New keyframe {}, matching with {}\n", new_keyframe_index, last_keyframe_index);
      const int num_new_unused_keypoints = new_keyframe.keypoints.size() - new_used_keypoint_indices.size();
      const int num_last_unused_keypoints = last_keyframe.keypoints.size() - last_used_keypoint_indices.size();
      fmt::print(
          "  #unused keypoints: {} ({:.2f}% all), {} ({:.2f}% all)\n",
          num_new_unused_keypoints,
          100.0 * (num_new_unused_keypoints) / new_keyframe.keypoints.size(),
          num_last_unused_keypoints,
          100.0 * (num_last_unused_keypoints) / last_keyframe.keypoints.size());
      fmt::print(
          "  #matches: {}, ({:.2f}% new unused), ({:.2f}% last unused)\n",
          new_last_matches.size(),
          100.0 * (new_last_matches.size()) / num_new_unused_keypoints,
          100.0 * (new_last_matches.size()) / num_last_unused_keypoints
      );
      fmt::print(
          "  #inliers ransac: {} ({:.2f}% matches)\n",
          new_last_inliers_ransac.size(),
          100.0 * (new_last_inliers_ransac.size()) / new_last_matches.size());
      fmt::print(
          "  #inliers epipolar: {} ({:.2f}% inliers ransac), ({:.2f}% matches)\n",
          new_last_inliers_epipolar.size(),
          100.0 * (new_last_inliers_epipolar.size()) / new_last_inliers_ransac.size(),
          100.0 * (new_last_inliers_epipolar.size()) / new_last_matches.size());
      fmt::print("  #total keyframes: {}\n", map.keyframes.size());
      fmt::print("  #total landmarks: {}\n", map.landmarks.size());

      // TODO: Remove
      // show_keyframes({last_keyframe_index, new_keyframe_index}, map, new_keyframe_index, camera);

      last_keyframe_index = new_keyframe_index;

    }

    return true;
  }

};

#endif //VISUAL_ODOMETRY_H_
