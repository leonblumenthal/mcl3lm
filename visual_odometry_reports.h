#ifndef VISUAL_ODOMETRY_REPORTS_H_
#define VISUAL_ODOMETRY_REPORTS_H_

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
#include "filesystem"

namespace fs = std::filesystem;

// struct Report {
//   virtual void print() const = 0;
//   virtual std::string get_csv_header() const = 0;
//   virtual std::string get_csv_line() const = 0;
// };
//


struct VisualOdometryInitializeReport {
  size_t num_keypoints_1;
  size_t num_keypoints_2;
  size_t num_matches;
  size_t num_inliers;
  ceres::Solver::Summary bundle_adjustment_summary;

  void print() const {
    fmt::print("Initialized VO Map with {} landmarks:\n", num_inliers);
    fmt::print("  #keypoints: {}, {}\n", num_keypoints_1, num_keypoints_2);
    fmt::print(
        "  #inliers: {} ({:.0f}% matches)\n", num_inliers, 100.0 * num_inliers / num_matches
    );
    fmt::print("{}\n", bundle_adjustment_summary.BriefReport());
  }

  std::string get_csv_header() const {
    return "num_keypoints_1,num_keypoints_2,num_matches,num_inliers,bundle_adjustment_num_iterations,bundle_adjustment_initial_cost,bundle_adjustment_final_cost,bundle_adjustment_convergence";
  }

  std::string get_csv_line() const {
    {
      return fmt::format(
          "{},{},{},{},{},{},{},{}",
          num_keypoints_1,
          num_keypoints_2,
          num_matches,
          num_inliers,
          bundle_adjustment_summary.iterations.size(),
          bundle_adjustment_summary.initial_cost,
          bundle_adjustment_summary.final_cost,
          static_cast<int>(bundle_adjustment_summary.termination_type == ceres::TerminationType::CONVERGENCE)
      );
    }

  };
};

struct VisualOdometryNextReport {
  int frame_index;
  size_t num_keypoints;
  size_t num_visible_landmarks;
  size_t num_total_landmarks;
  size_t num_landmark_keypoint_matches;
  size_t num_landmark_keypoint_inliers;
  double distance_to_last_keyframe;
  double distance_to_last_frame;

  void print() const {
    fmt::print("Frame {}:\n", frame_index);
    fmt::print(
        "  #visible landmarks: {} ({:.0f}% all)\n",
        num_visible_landmarks,
        100.0 * num_visible_landmarks / num_total_landmarks
    );
    fmt::print("  #keypoints: {}\n", num_keypoints);
    fmt::print(
        "  #matches: {} ({:.0f}% keypoints, {:.0f}% visible landmarks)\n",
        num_landmark_keypoint_matches,
        100.0 * num_landmark_keypoint_matches / num_keypoints,
        100.0 * num_landmark_keypoint_matches / num_visible_landmarks
    );
    fmt::print(
        "  #inliers: {} ({:.0f}% matches, {:.0f}% visible landmarks)\n",
        num_landmark_keypoint_inliers,
        100.0 * num_landmark_keypoint_inliers / num_landmark_keypoint_matches,
        100.0 * num_landmark_keypoint_inliers / num_visible_landmarks
    );
    fmt::print(
        "  distance to last keyframeframe: {:.2}, rame: {:.2}\n", distance_to_last_keyframe, distance_to_last_frame
    );

  }

  std::string get_csv_header() const {
    return "frame_index,num_keypoints,num_visible_landmarks,num_total_landmarks,num_landmark_keypoint_matches,num_landmark_keypoint_inliers,distance_to_last_keyframe,distance_to_last_frame";
  }

  std::string get_csv_line() const {
    return fmt::format(
        "{},{},{},{},{},{},{},{}",
        frame_index,
        num_keypoints,
        num_visible_landmarks,
        num_total_landmarks,
        num_landmark_keypoint_matches,
        num_landmark_keypoint_inliers,
        distance_to_last_keyframe,
        distance_to_last_frame
    );
  }
};

struct VisualOdometryKeyframeReport {
  int removed_keyframe_index;
  size_t num_removed_landmarks;
  int new_keyframe_index;
  int last_keyframe_index;
  size_t num_new_keypoints;
  size_t num_last_keypoints;
  size_t num_new_unused_keypoints;
  size_t num_last_unused_keypoints;
  size_t num_new_last_matches;
  size_t num_new_last_inliers_ransac;
  size_t num_new_last_inliers_epipolar;
  size_t num_total_keyframes;
  size_t num_total_landmarks;
  ceres::Solver::Summary bundle_adjustment_summary;

  void print() {
    if (removed_keyframe_index > -1) {
      fmt::print(
          "Removed keyframe {} and {} landmarks\n", removed_keyframe_index, num_removed_landmarks
      );
    }
    fmt::print("New keyframe {}, matching with {}\n", new_keyframe_index, last_keyframe_index);
    fmt::print(
        "  #unused keypoints: {} ({:.2f}% all), {} ({:.2f}% all)\n",
        num_new_unused_keypoints,
        100.0 * (num_new_unused_keypoints) / num_new_keypoints,
        num_last_unused_keypoints,
        100.0 * (num_last_unused_keypoints) / num_last_keypoints
    );
    fmt::print(
        "  #matches: {}, ({:.2f}% new unused), ({:.2f}% last unused)\n",
        num_new_last_matches,
        100.0 * (num_new_last_matches) / num_new_unused_keypoints,
        100.0 * (num_new_last_matches) / num_last_unused_keypoints
    );
    fmt::print(
        "  #inliers ransac: {} ({:.2f}% matches)\n",
        num_new_last_inliers_ransac,
        100.0 * (num_new_last_inliers_ransac) / num_new_last_matches
    );
    fmt::print(
        "  #inliers epipolar: {} ({:.2f}% inliers ransac), ({:.2f}% matches)\n",
        num_new_last_inliers_epipolar,
        100.0 * num_new_last_inliers_epipolar / num_new_last_inliers_ransac,
        100.0 * num_new_last_inliers_epipolar / num_new_last_matches
    );
    fmt::print("  #total keyframes: {}\n", num_total_keyframes);
    fmt::print(
        "  #total landmarks: {}\n", num_total_landmarks
    );
    fmt::print("{}\n", bundle_adjustment_summary.BriefReport());
  }

  std::string get_csv_header() const {
    return "removed_keyframe_index,num_removed_landmarks,new_keyframe_index,last_keyframe_index,num_new_keypoints,num_last_keypoints,num_new_unused_keypoints,num_last_unused_keypoints,num_new_last_matches,num_new_last_inliers_ransac,num_new_last_inliers_epipolar,num_total_keyframes,num_total_landmarks,bundle_adjustment_num_iterations,bundle_adjustment_initial_cost,bundle_adjustment_final_cost,bundle_adjustment_convergence";
  }

  std::string get_csv_line() const {
    return fmt::format(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        removed_keyframe_index,
        num_removed_landmarks,
        new_keyframe_index,
        last_keyframe_index,
        num_new_keypoints,
        num_last_keypoints,
        num_new_unused_keypoints,
        num_last_unused_keypoints,
        num_new_last_matches,
        num_new_last_inliers_ransac,
        num_new_last_inliers_epipolar,
        num_total_keyframes,
        num_total_landmarks,
        bundle_adjustment_summary.iterations.size(),
        bundle_adjustment_summary.initial_cost,
        bundle_adjustment_summary.final_cost,
        static_cast<int>(bundle_adjustment_summary.termination_type == ceres::TerminationType::CONVERGENCE)
    );
  }
};

// TODO: Actually do with report base class.
struct ReportWriter {
  const std::string path;
  bool has_written;

  explicit ReportWriter(const std::string &path) : path(path), has_written(false) {
    // Clear file.
    std::ofstream out{path};
    out.close();
  };

  void write(const VisualOdometryInitializeReport &report) {
    std::ofstream out{path, std::ios_base::app};

    if (!has_written) {
      out << report.get_csv_header() << std::endl;
      has_written = true;
    }

    out << report.get_csv_line() << std::endl;

    out.close();
  }

  void write(const VisualOdometryNextReport &report) {
    std::ofstream out{path, std::ios_base::app};

    if (!has_written) {
      out << report.get_csv_header() << std::endl;
      has_written = true;
    }

    out << report.get_csv_line() << std::endl;

    out.close();
  }

  void write(const VisualOdometryKeyframeReport &report) {
    std::ofstream out{path, std::ios_base::app};

    if (!has_written) {
      out << report.get_csv_header() << std::endl;
      has_written = true;
    }

    out << report.get_csv_line() << std::endl;

    out.close();
  }
};

#endif //VISUAL_ODOMETRY_REPORTS_H_
