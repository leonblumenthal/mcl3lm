#ifndef VISUAL_ODOMETRY_UTILS_H_
#define VISUAL_ODOMETRY_UTILS_H_

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




// TODO:
// Find inliers with known poses using the epipolar constraint.
std::pair<std::vector<std::pair<int, int>>, std::vector<Eigen::Vector3d>> find_inliers_essential(
    const std::vector<Eigen::Vector2i> &corners_1,
    const std::vector<Eigen::Vector2i> &corners_2,
    const std::vector<std::pair<int, int>> &matches,
    const Sophus::SE3d &pose_1,
    const Sophus::SE3d &pose_2,
    const Camera<> &camera,
    const double max_epipolar_error = 1e-3
) {
  // Compute essential matrix.
  Sophus::SE3d pose_1_2 = pose_1.inverse() * pose_2;
  const Eigen::Vector3d t = pose_1_2.translation() / pose_1_2.translation().norm();
  Eigen::Matrix3d t_hat;
  t_hat << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;
  Eigen::Matrix3d E{t_hat * pose_1_2.rotationMatrix()};

  opengv::bearingVectors_t bearing_vectors_1;
  opengv::bearingVectors_t bearing_vectors_2;

  std::vector<std::pair<int, int>> inliers;
  for (const auto &[index_1, index_2] : matches) {
    Eigen::Vector3d unprojected_1 = camera.unproject(corners_1[index_1].cast<double>());
    Eigen::Vector3d unprojected_2 = camera.unproject(corners_2[index_2].cast<double>());

    if (unprojected_1.transpose() * E * unprojected_2 < max_epipolar_error) {
      inliers.emplace_back(index_1, index_2);

      bearing_vectors_1.push_back(unprojected_1);
      bearing_vectors_2.push_back(unprojected_2);
    }
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearing_vectors_1, bearing_vectors_2
  );
  Sophus::SE3d relative_pose = pose_1.inverse() * pose_2;
  adapter.sett12(relative_pose.translation());
  adapter.setR12(relative_pose.rotationMatrix());

  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < inliers.size(); ++i) {
    points.push_back(opengv::triangulation::triangulate(adapter, i));
  }

  return {inliers, points};
}


using Keypoint = Eigen::Vector2i;
using Keypoints = std::vector<Keypoint>;
using Descriptor = std::bitset<256>;
using Descriptors = std::vector<Descriptor>;
using IndexMatches = std::vector<std::pair<int, int>>;
using Position = Eigen::Vector3d;
using Positions = std::vector<Position>;
using Pose = Sophus::SE3d;
using Point = Eigen::Vector2d;

struct Landmark {
  Position position;
  std::unordered_map<int, int> observations;
};

struct Keyframe {
  Pose pose;
  Keypoints keypoints;
  Descriptors descriptors;
};

struct Map {
  std::unordered_map<int, Keyframe> keyframes;
  std::unordered_map<int, Landmark> landmarks;
  int next_landmark_index = 0;
};

bool is_in_bounds(int x, int y, const cv::Mat &image, int edge_margin = 0) {
  return (x >= edge_margin && x < image.cols - edge_margin && y >= edge_margin && y < image.rows - edge_margin);
}

// Compute sparse set of "interesting" image points.
Keypoints detect_keypoints(
    const cv::Mat &image, int max_num_keypoints, double quality_level, int min_distance_between, int edge_margin
) {
  // Use OpenCV to detect corners/keypoints.
  std::vector<cv::Point2i> all_keypoints;
  cv::goodFeaturesToTrack(
      image, all_keypoints, max_num_keypoints, quality_level, min_distance_between
  );

  // Filter out points too close to the image edge.
  Keypoints valid_keypoints;
  for (const auto &point : all_keypoints) {
    if (is_in_bounds(point.x, point.y, image, edge_margin)) {
      valid_keypoints.emplace_back(point.x, point.y);
    }
  }

  return valid_keypoints;
}

// Compute local orientation of each keypoint for rotated descriptors.
std::vector<double> compute_angles(
    const cv::Mat &image, const Keypoints &keypoints, int patch_radius
) {

  std::vector<double> angles(keypoints.size());
  for (int i = 0; i < keypoints.size(); ++i) {
    const Keypoint &keypoint = keypoints[i];

    // Compute weighted center of intensity in a circular patch around the
    // keypoint.
    double mx = 0, my = 0;
    for (int x = -patch_radius; x <= patch_radius; ++x) {
      for (int y = -patch_radius; y <= patch_radius; ++y) {
        if (x * x + y * y > patch_radius * patch_radius)
          continue;

        int intensity = static_cast<int>(
            image.at<uint8_t>(keypoint.y() + y, keypoint.x() + x));
        mx += intensity * x;
        my += intensity * y;
      }
    }

    // The angle of the keypoint is the direction of the weighted center.
    angles[i] = std::atan2(my, mx);
  };

  return angles;
}

std::tuple<int, int, int, int> rotate_descriptor_pattern(int bit_index, double sin_angle, double cos_angle) {
  int x_1 = std::round(
      cos_angle * pattern_31_x_a[bit_index] - sin_angle * pattern_31_y_a[bit_index]
  );
  int y_1 = std::round(
      sin_angle * pattern_31_x_a[bit_index] + cos_angle * pattern_31_y_a[bit_index]
  );
  int x_2 = std::round(
      cos_angle * pattern_31_x_b[bit_index] - sin_angle * pattern_31_y_b[bit_index]
  );
  int y_2 = std::round(
      sin_angle * pattern_31_x_b[bit_index] + cos_angle * pattern_31_y_b[bit_index]
  );

  return {x_1, y_1, x_2, y_2};
}

// Compute ORB descriptors.
Descriptors compute_descriptors(
    const cv::Mat &image, const Keypoints &keypoints, int angle_patch_radius
) {
  Descriptors descriptors(keypoints.size());

  // Pre-compute angles to rotate descriptors.
  std::vector<double> angles = compute_angles(image, keypoints, angle_patch_radius);

  for (int keypoint_index = 0; keypoint_index < keypoints.size(); ++keypoint_index) {
    const Keypoint &keypoint = keypoints[keypoint_index];
    const double cos_angle = std::cos(angles[keypoint_index]);
    const double sin_angle = std::sin(angles[keypoint_index]);

    // Compute each descriptor bit as binary comparison between image
    // intensities at rotated pre-defined positions.
    for (int bit_index = 0; bit_index < descriptors[keypoint_index].size(); ++bit_index) {
      auto [x_1, y_1, x_2, y_2] = rotate_descriptor_pattern(bit_index, cos_angle, sin_angle);
      uint8_t intensity_1 = image.at<uint8_t>(keypoint.y() + y_1, keypoint.x() + x_1);
      uint8_t intensity_2 = image.at<uint8_t>(keypoint.y() + y_2, keypoint.x() + x_1);
      descriptors[keypoint_index][bit_index] = intensity_1 < intensity_2;
    }
  }

  return descriptors;
}

std::pair<Keypoints, Descriptors> compute_keypoints_and_descriptors(
    const cv::Mat &image,
    int max_num_keypoints = 1000,
    double keypoint_quality_level = 0.01,
    int min_distance_between_keypoints = 8,
    int keypoint_edge_margin = 20,
    int descriptor_angle_patch_radius = 15
) {
  Keypoints keypoints = detect_keypoints(
      image, max_num_keypoints, keypoint_quality_level, min_distance_between_keypoints, keypoint_edge_margin
  );
  Descriptors descriptors = compute_descriptors(image, keypoints, descriptor_angle_patch_radius);

  return {keypoints, descriptors};
}

// Match descriptors of two frames.
IndexMatches match(
    const Descriptors &descriptors_1,
    const Descriptors &descriptors_2,
    int max_distance = 70,
    double max_second_to_first_distance_ratio = 1.1
) {

  // 1. Compute best and second-best matches for each descriptor on both frames.
  struct Match {
    int best_index = -1;
    int best_distance = INT_MAX;
    int second_best_distance = INT_MAX;
  };
  std::vector<Match> matches_1(descriptors_1.size());
  std::vector<Match> matches_2(descriptors_2.size());

  for (int index_1 = 0; index_1 < descriptors_1.size(); ++index_1) {
    Match &match_1 = matches_1[index_1];
    for (int index_2 = 0; index_2 < descriptors_2.size(); ++index_2) {
      Match &match_2 = matches_2[index_2];

      // Hamming distance.
      const int distance = static_cast<int>((descriptors_1[index_1] ^ descriptors_2[index_2]).count());

      // Not skipping here if distance > max_distance allows filtering out not distinctive enough
      // matches later if the second-best match is above max_distance, e.g.
      // invalid match: max_distance=70, best_distance=69, second_best_distance=71

      // Update best or second-best match for descriptor 1.
      if (distance < match_1.best_distance) {
        match_1.second_best_distance = match_1.best_distance;
        match_1.best_distance = distance;
        match_1.best_index = index_2;
      } else if (distance < match_1.second_best_distance) {
        match_1.second_best_distance = distance;
      }

      // Update best or second-best match for descriptor 2.
      if (distance < match_2.best_distance) {
        match_2.second_best_distance = match_2.best_distance;
        match_2.best_distance = distance;
        match_2.best_index = index_1;
      } else if (distance < match_2.second_best_distance) {
        match_2.second_best_distance = distance;
      }

    }
  }

  // 2. Filter out matches that are:
  // not bidirectional or
  // not distinctive enough (i.e. difference to second-best match too small)
  IndexMatches matches;
  for (int index_1 = 0; index_1 < matches_1.size(); ++index_1) {
    Match &match_1 = matches_1[index_1];

    // This additionally ensures that a match exists.
    if (match_1.best_distance > max_distance) continue;

    Match &match_2 = matches_2[match_1.best_index];

    // Bidirectional.
    if (match_2.best_index != index_1) continue;

    // Distinctiveness.
    if (match_1.second_best_distance < match_1.best_distance * max_second_to_first_distance_ratio
        || match_2.second_best_distance < match_2.best_distance * max_second_to_first_distance_ratio) {
      continue;
    }

    matches.emplace_back(index_1, match_1.best_index);
  }

  return matches;
}

// Unproject all keypoints into normalized positions (bearing vectors).
std::pair<opengv::bearingVectors_t, opengv::bearingVectors_t> compute_bearing_vectors(
    const IndexMatches &matches,
    const Keypoints &keypoints_1,
    const Keypoints &keypoints_2,
    const Camera<double> &camera
) {
  opengv::bearingVectors_t bearing_vectors_1(matches.size());
  opengv::bearingVectors_t bearing_vectors_2(matches.size());
  for (int i = 0; i < matches.size(); ++i) {
    const auto &[index_1, index_2] = matches[i];
    bearing_vectors_1[i] = camera.unproject(keypoints_1[index_1].cast<double>());
    bearing_vectors_2[i] = camera.unproject(keypoints_2[index_2].cast<double>());
  }

  return {bearing_vectors_1, bearing_vectors_2};
}

// Refine matches by computing a relative pose using RANSAC
std::pair<IndexMatches, Pose> find_inliers_ransac(
    const IndexMatches &matches,
    const Keypoints &keypoints_1,
    const Keypoints &keypoints_2,
    const Camera<double> &camera,
    double ransac_threshold = 5e-5,
    int ransac_min_num_inliers = 16,
    int ransac_max_iterations = 100
) {

  auto [bearing_vectors_1, bearing_vectors_2] = compute_bearing_vectors(matches, keypoints_1, keypoints_2, camera);

  // Setup OpenGV RANSAC stuff using Nister's 5-point algorithm.
  opengv::relative_pose::CentralRelativeAdapter adapter(
      bearing_vectors_1, bearing_vectors_2
  );
  opengv::sac::Ransac<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> ransac;;
  std::shared_ptr<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> relposeproblem_ptr(
      new opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem(
          adapter, opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::NISTER
      ));
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_threshold;
  ransac.max_iterations_ = ransac_max_iterations;
  // Actually run the above.
  ransac.computeModel();

  // Fail.
  if (ransac.inliers_.size() < ransac_min_num_inliers)
    return {};

  // Optimize relative pose based on inliers.
  opengv::transformation_t relative_pose = opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);
  // Expand set of inliers based on new optimized relative pose.
  ransac.sac_model_->selectWithinDistance(
      relative_pose, ransac.threshold_, ransac.inliers_
  );

  // Refine matches with inliers.
  IndexMatches inlier_matches(ransac.inliers_.size());
  for (int i = 0; i < ransac.inliers_.size(); ++i) inlier_matches[i] = matches[ransac.inliers_[i]];

  Pose pose{relative_pose.leftCols(3), relative_pose.col(3).normalized()};

  return {inlier_matches, pose};
}

// Triangulate 3D positions from matching keypoints of two frames with a known relative pose.
Positions triangulate(
    const IndexMatches &matches,
    const Keypoints &keypoints_1,
    const Keypoints &keypoints_2,
    const Camera<double> &camera,
    const Pose &relative_pose
) {
  auto [bearing_vectors_1, bearing_vectors_2] = compute_bearing_vectors(matches, keypoints_1, keypoints_2, camera);

  opengv::relative_pose::CentralRelativeAdapter adapter{
      bearing_vectors_1, bearing_vectors_2, relative_pose.translation(), relative_pose.rotationMatrix()
  };

  Positions positions(matches.size());
  for (int i = 0; i < matches.size(); ++i) {
    positions[i] = opengv::triangulation::triangulate(adapter, i);
  }

  return positions;

}

// Project visible landmark positions into an image at the given pose.
std::unordered_map<int, Point> project_landmarks(
    const std::unordered_map<int, Landmark> &landmarks, const Pose &pose, const Camera<> &camera, int min_z = 0
) {
  std::unordered_map<int, Point> landmark_points;

  for (const auto &[index, landmark] : landmarks) {
    // Transform into camera coordinate frame.
    Position local_position{pose.inverse() * landmark.position};
    // Ensure that landmark is in front of camera.
    if (local_position.z() < min_z)
      continue;

    Point image_position = camera.project(local_position);

    // Ensure that landmark is visible inside image.
    if (image_position.x() < 0 || image_position.y() < 0 || image_position.x() > camera.width
        || image_position.y() > camera.height)
      continue;

    landmark_points[index] = image_position;
  }

  return landmark_points;
}

struct BundleAdjustmentReprojectionCostFunctor {

  const Keypoint &observed_keypoint;
  const Camera<> &camera;

  BundleAdjustmentReprojectionCostFunctor(
      const Keypoint &observed_keypoint, const Camera<> &camera
  ) : observed_keypoint(observed_keypoint), camera(camera) {}

  template<class T>
  bool operator()(
      T const *const skeyframe_pose, T const *const slandmark_position, T *sResiduals
  ) const {
    // Map inputs to ceres::Jet types.
    Eigen::Map<Sophus::SE3<T> const> const keyframe_pose(skeyframe_pose);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const landmark_position(
        slandmark_position
    );
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);

    // TODO: Improve.
    // Create ceres::Jet camera.
    Camera<T> cam{
        T(camera.width), T(camera.height), CameraParameters<T>{
            T(camera.parameters.fx), T(camera.parameters.fy), T(camera.parameters.cx), T(camera.parameters.cy),
            T(camera.parameters.k1), T(camera.parameters.k2), T(camera.parameters.p1), T(camera.parameters.p2)
        }};

    // Residual is the error between point of projected landmark and observed keypoint.
    // This is a vector of two residual entries.
    Eigen::Matrix<T, 2, 1> point = cam.project(keyframe_pose.inverse() * landmark_position);
    residuals = observed_keypoint.cast<T>() - point;

    return true;
  }
};

void bundle_adjustment(
    Map &map, const Camera<> &camera, const int max_num_iterations = 20
) {

  ceres::Problem problem;

  // Add a residual block for each visible landmark position projected in each image.
  for (auto &[keyframe_index, keyframe] : map.keyframes) {
    // Specific parametrization of the pose.
    problem.AddParameterBlock(
        keyframe.pose.data(), Pose::num_parameters, new Sophus::Manifold<Sophus::SE3>());

    for (auto &[_, landmark] : map.landmarks) {

      if (!landmark.observations.contains(keyframe_index))
        continue;

      auto &observed_keypoint{
          keyframe.keypoints[landmark.observations[keyframe_index]]
      };

      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<BundleAdjustmentReprojectionCostFunctor, 2, Sophus::SE3d::num_parameters, 3>(
              new BundleAdjustmentReprojectionCostFunctor(
                  observed_keypoint, camera
              )), new ceres::HuberLoss(1.0), keyframe.pose.data(), landmark.position.data());
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = max_num_iterations;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
}

// Find a matching landmark for each keypoint.
IndexMatches match_landmarks(
    const Keypoints &keypoints,
    const Descriptors &descriptors,
    const Pose &pose,
    const Map &map,
    const Camera<> &camera,
    const double max_image_distance = 20,
    const double max_descriptor_distance = 70,
    const double max_second_to_first_distance_ratio = 1.1
) {

  std::unordered_map<int, Point> landmark_points = project_landmarks(map.landmarks, pose, camera);

  // Find closest landmark for each keypoint.
  std::unordered_map<int, int> landmark_to_keypoint;
  std::unordered_map<int, int> landmark_best_distance;
  for (int keypoint_index = 0; keypoint_index < keypoints.size(); ++keypoint_index) {
    int best_distance = INT_MAX;
    int second_distance = INT_MAX;
    int best_landmark_index = -1;

    for (const auto &[landmark_index, projected_landmark] : landmark_points) {
      // Ensure that the projected landmark and the keypoint are close together on the image.
      double image_distance = (keypoints[keypoint_index].cast<double>() - projected_landmark).norm();
      if (image_distance > max_image_distance)
        continue;

      // Compute minimum distance to observed descriptors for landmark
      int descriptor_distance = INT_MAX;
      for (const auto &[keyframe_index, descriptor_index] : map.landmarks.at(landmark_index).observations) {
        int distance =
            (descriptors[keypoint_index] ^ map.keyframes.at(keyframe_index).descriptors[descriptor_index]).count();
        descriptor_distance = std::min(descriptor_distance, distance);
      }

      if (descriptor_distance < best_distance) {
        best_landmark_index = landmark_index;
        second_distance = best_distance;
        best_distance = descriptor_distance;
      } else if (descriptor_distance < second_distance) {
        second_distance = descriptor_distance;
      }
    }

    if (best_landmark_index == -1 || best_distance > max_descriptor_distance)
      continue;

    // Check if matching landmark is unique enough.
    if (second_distance < best_distance * max_second_to_first_distance_ratio)
      continue;

    // Ensure that a landmark is only matched with the (descriptor-wise) closest
    // corner.
    if (!landmark_to_keypoint.contains(best_landmark_index)) {
      landmark_best_distance[best_landmark_index] = INT_MAX;
    }
    if (landmark_best_distance[best_landmark_index] > best_distance) {
      landmark_to_keypoint[best_landmark_index] = keypoint_index;
      landmark_best_distance[best_landmark_index] = best_distance;
    }
  }

  IndexMatches matches;
  for (const auto &[landmark_index, keypoint_index] : landmark_to_keypoint)
    matches.emplace_back(landmark_index, keypoint_index);

  return matches;
}

// TODO: Expose default parameters of internal methods.
std::pair<Pose, IndexMatches> localize(
    const Keypoints &keypoints,
    const Descriptors &descriptors,
    const Map &map,
    const Pose &current_pose,
    const Camera<> &camera,
    const int ransac_max_iterations = 100,
    const double ransac_image_distance_threshold = 3
) {

  IndexMatches matches = match_landmarks(keypoints, descriptors, current_pose, map, camera);

  opengv::bearingVectors_t bearing_vectors;
  opengv::points_t absolute_points;
  for (const auto &[landmark_index, corner_index] : matches) {
    bearing_vectors.push_back(
        camera.unproject(keypoints[corner_index].cast<double>()));
    absolute_points.push_back(map.landmarks.at(landmark_index).position);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(
      bearing_vectors, absolute_points
  );
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP
      ));
  ransac.sac_model_ = absposeproblem_ptr;
  // Set appropriate threshold based on pixel threshold and focal length of 500.
  ransac.threshold_ = 1.0 - cos(atan(ransac_image_distance_threshold / 500.0));
  ransac.max_iterations_ = ransac_max_iterations;

  ransac.computeModel(0);

  // TODO: This is not done internally?
  adapter.sett(ransac.model_coefficients_.col(3));
  adapter.setR(ransac.model_coefficients_.leftCols(3));

  opengv::transformation_t t = opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

  ransac.sac_model_->selectWithinDistance(
      t, ransac.threshold_, ransac.inliers_
  );

  Sophus::SE3d pose;
  pose.setRotationMatrix(t.leftCols(3));
  pose.translation() = t.col(3);

  std::vector<std::pair<int, int>> inliers;
  for (const auto &i : ransac.inliers_)
    inliers.emplace_back(matches[i]);

  return {pose, inliers};
}

#endif //VISUAL_ODOMETRY_UTILS_H_

