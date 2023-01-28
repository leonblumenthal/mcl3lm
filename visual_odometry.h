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
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>
#include <vector>
#include "camera.h"
#include <fmt/core.h>
#include "visualization_utils.h"
#include "reports.h"

struct VisualOdometry {

  const Camera<> camera;

  const VisualOdometryConfig config;

  Map map;
  Pose current_pose;
  int last_keyframe_index;
  int current_frame_index;

  std::queue<int> keyframe_indices;

  bool return_reports;

  VisualOdometry(
      const Camera<> &camera, bool return_reports = true, const VisualOdometryConfig &config = VisualOdometryConfig())
      : camera(camera), return_reports(return_reports), config(config), current_pose(Pose::rotX(0)) {
  }

  std::optional<VisualOdometryInitializeReport> initialize(
      const cv::Mat &image_1, const cv::Mat &image_2, const Pose &pose_1, const Pose &pose_2, const int index_1, const int index_2
  ) {

    auto [keypoints_1, descriptors_1] = initialize_compute_keypoints_and_descriptors(image_1);
    auto [keypoints_2, descriptors_2] = initialize_compute_keypoints_and_descriptors(image_2);

    IndexMatches matches = initialize_match(descriptors_1, descriptors_2);

    auto [inlier_matches, relative_pose] = initialize_find_inliers_ransac(
        matches, keypoints_1, keypoints_2
    );
    // relative_pose.translation() *= (pose_1.translation() - pose_2.translation()).norm();

    map.keyframes[index_1] = {
        pose_1, keypoints_1, descriptors_1, image_1
    };
    map.keyframes[index_2] = {
        pose_2, keypoints_2, descriptors_2, image_2
        // pose_1 * relative_pose, keypoints_2, descriptors_2, image_2
    };

    Positions landmark_positions = vo_utils::triangulate(inlier_matches, map.keyframes[index_1], map.keyframes[index_2], camera);
    for (int i = 0; i < inlier_matches.size(); ++i) {
      const auto &match = inlier_matches[i];
      map.landmarks[map.next_landmark_index++] = {
          landmark_positions[i], {{index_1, match.first}, {index_2, match.second}}};
    }

    ceres::Solver::Summary bundle_adjustment_summary = initialze_bundle_adjustment();

    current_pose = map.keyframes[index_2].pose;
    last_keyframe_index = index_2;
    current_frame_index = index_2;

    keyframe_indices.push(index_1);
    keyframe_indices.push(index_2);

    if (return_reports) {
      return VisualOdometryInitializeReport{
          keypoints_1.size(), keypoints_2.size(), matches.size(), inlier_matches.size(), bundle_adjustment_summary,
      };
    }

  }

  std::tuple<bool,
             std::optional<VisualOdometryNextReport>,
             std::optional<VisualOdometryKeyframeReport>> next(const cv::Mat &image) {
    ++current_frame_index;

    auto [keypoints, descriptors] = next_compute_keypoints_and_descriptors(image);

    std::unordered_map<int, Point> landmark_points = vo_utils::project_landmarks(map.landmarks, current_pose, camera);

    IndexMatches landmark_keypoint_matches = next_match_landmarks(keypoints, descriptors, landmark_points);

    auto [pose, landmark_keypoint_inliers] = next_localize(keypoints, landmark_keypoint_matches);

    std::optional<VisualOdometryNextReport> next_report;
    std::optional<VisualOdometryKeyframeReport> keyframe_report;
    if (return_reports) {
      next_report = {
          current_frame_index, keypoints.size(), landmark_points.size(), map.landmarks.size(),
          landmark_keypoint_matches.size(), landmark_keypoint_inliers.size(),
          (map.keyframes[last_keyframe_index].pose.translation() - pose.translation()).norm(),
          (current_pose.translation() - pose.translation()).norm()
      };
    }

    if (landmark_keypoint_inliers.size() < 5) return {false, next_report, keyframe_report};

    current_pose = pose;

    const double
        distance_to_last_keyframe = (map.keyframes[last_keyframe_index].pose.translation() - pose.translation()).norm();

    bool is_next_keyframe = (distance_to_last_keyframe > config.next.keyframe.is_next.min_distance_to_last
        && landmark_keypoint_inliers.size() < config.next.keyframe.is_next.max_num_landmark_keypoint_inliers)
        || landmark_keypoint_inliers.size() < config.next.keyframe.is_next.min_num_landmark_keypoint_inliers;

    if (is_next_keyframe) {

      // TODO: Add comparison with all keyframes.

      int new_keyframe_index = current_frame_index;
      map.keyframes[new_keyframe_index] = Keyframe{current_pose, keypoints, descriptors, image};
      for (const auto &[landmark_index, keypoint_index] : landmark_keypoint_inliers) {
        map.landmarks[landmark_index].observations[new_keyframe_index] = keypoint_index;
      }

      keyframe_indices.push(new_keyframe_index);

      size_t num_removed_landmarks = map.landmarks.size();
      int removed_keyframe_index = -1;
      // This should only be executed once actually.
      if (keyframe_indices.size() > config.next.keyframe.max_num_keyframes) {
        removed_keyframe_index = keyframe_indices.front();
        keyframe_indices.pop();
        vo_utils::remove_keyframe(removed_keyframe_index, map);
      }
      num_removed_landmarks -= map.landmarks.size();

      std::unordered_set<int> new_used_keypoint_indices = vo_utils::get_used_keypoint_indices(map, new_keyframe_index);
      std::unordered_set<int>
          last_used_keypoint_indices = vo_utils::get_used_keypoint_indices(map, last_keyframe_index);

      Keyframe &new_keyframe = map.keyframes[new_keyframe_index];
      Keyframe &last_keyframe = map.keyframes[last_keyframe_index];

      IndexMatches new_last_matches = next_match(
          new_keyframe.descriptors, last_keyframe.descriptors, new_used_keypoint_indices, last_used_keypoint_indices
      );

      auto [new_last_inliers_ransac, _] = next_find_inliers_ransac(
          new_last_matches, new_keyframe.keypoints, last_keyframe.keypoints
      );
      IndexMatches new_last_inliers_epipolar = vo_utils::find_inliers_epipolar(
          new_last_inliers_ransac, new_keyframe, last_keyframe, camera, config.next.keyframe.matching.max_epipolar_error
      );

      Positions new_landmark_positions =
          vo_utils::triangulate(new_last_inliers_epipolar, new_keyframe, last_keyframe, camera);
      for (int i = 0; i < new_last_inliers_epipolar.size(); ++i) {
        const auto &[new_keypoint_index, last_keypoint_index] = new_last_inliers_epipolar[i];
        map.landmarks[map.next_landmark_index++] = {
            new_landmark_positions[i],
            {{new_keyframe_index, new_keypoint_index}, {last_keyframe_index, last_keypoint_index}}};
      }

      ceres::Solver::Summary bundle_adjustment_summary = next_bundle_adjustment();

      current_pose = new_keyframe.pose;

      if (return_reports) {
        keyframe_report = {
            removed_keyframe_index, num_removed_landmarks, new_keyframe_index, last_keyframe_index,
            new_keyframe.keypoints.size(), last_keyframe.keypoints.size(),
            new_keyframe.keypoints.size() - new_used_keypoint_indices.size(),
            last_keyframe.keypoints.size() - last_used_keypoint_indices.size(), new_last_matches.size(),
            new_last_inliers_ransac.size(), new_last_inliers_epipolar.size(), map.keyframes.size(),
            map.landmarks.size(), bundle_adjustment_summary
        };
      }

      last_keyframe_index = new_keyframe_index;

    }

    return {true, next_report, keyframe_report};
  }

  // Methods that call vo_utils functions with specific arguments.
 private:
  std::pair<Keypoints, Descriptors> initialize_compute_keypoints_and_descriptors(const cv::Mat &image) const {
    return vo_utils::compute_keypoints_and_descriptors(
        image,
        config.initialize.keypoints.max_num,
        config.initialize.keypoints.quality_level,
        config.initialize.keypoints.min_distance_between,
        config.initialize.keypoints.edge_margin,
        config.initialize.keypoints.descriptor_angle_patch_radius
    );
  }

  IndexMatches initialize_match(const Descriptors &descriptors_1, const Descriptors &descriptors_2) const {
    return vo_utils::match(
        descriptors_1,
        descriptors_2,
        config.initialize.matching.max_distance,
        config.initialize.matching.max_second_to_first_distance_ratio
    );
  }

  std::pair<IndexMatches, Pose> initialize_find_inliers_ransac(
      const IndexMatches &matches, const Keypoints &keypoints_1, const Keypoints &keypoints_2
  ) const {
    return vo_utils::find_inliers_ransac(
        matches,
        keypoints_1,
        keypoints_2,
        camera,
        config.initialize.matching.ransac.threshold,
        config.initialize.matching.ransac.min_num_inliers,
        config.initialize.matching.ransac.max_num_iterations
    );
  }

  ceres::Solver::Summary initialze_bundle_adjustment() {
    return vo_utils::bundle_adjustment(map, camera, config.initialize.max_num_bundle_adjustment_iterations);
  }

  std::pair<Keypoints, Descriptors> next_compute_keypoints_and_descriptors(const cv::Mat &image) const {
    return vo_utils::compute_keypoints_and_descriptors(
        image,
        config.next.keypoints.max_num,
        config.next.keypoints.quality_level,
        config.next.keypoints.min_distance_between,
        config.next.keypoints.edge_margin,
        config.next.keypoints.descriptor_angle_patch_radius
    );
  }

  IndexMatches next_match_landmarks(
      const Keypoints &keypoints, const Descriptors &descriptors, const std::unordered_map<int, Point> &landmark_points
  ) const {
    return vo_utils::match_landmarks(
        keypoints,
        descriptors,
        map,
        landmark_points,
        config.next.matching.max_image_distance,
        config.next.matching.max_descriptor_distance,
        config.next.matching.max_second_to_first_distance_ratio
    );
  }

  std::pair<Pose, IndexMatches> next_localize(const Keypoints &keypoints, const IndexMatches &matches) const {
    return vo_utils::localize(
        keypoints,
        map,
        matches,
        camera,
        config.next.matching.ransac.max_num_iterations,
        config.next.matching.ransac.image_distance_threshold
    );
  }

  IndexMatches next_match(
      const Descriptors &descriptors_1,
      const Descriptors &descriptors_2,
      const std::unordered_set<int> &used_indices_1,
      const std::unordered_set<int> &used_indices_2
  ) const {
    return vo_utils::match(
        descriptors_1,
        descriptors_2,
        config.next.keyframe.matching.max_distance,
        config.next.keyframe.matching.max_second_to_first_distance_ratio,
        used_indices_1,
        used_indices_2
    );
  }

  std::pair<IndexMatches, Pose> next_find_inliers_ransac(
      const IndexMatches &matches, const Keypoints &keypoints_1, const Keypoints &keypoints_2
  ) const {
    return vo_utils::find_inliers_ransac(
        matches,
        keypoints_1,
        keypoints_2,
        camera,
        config.next.keyframe.matching.ransac.threshold,
        config.next.keyframe.matching.ransac.min_num_inliers,
        config.next.keyframe.matching.ransac.max_num_iterations
    );
  }

  ceres::Solver::Summary next_bundle_adjustment() {
    return vo_utils::bundle_adjustment(map, camera, config.next.keyframe.max_num_bundle_adjustment_iterations);
  }

};

#endif //VISUAL_ODOMETRY_H_
