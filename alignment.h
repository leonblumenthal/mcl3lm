#ifndef ALIGNMENT_H_
#define ALIGNMENT_H_

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

struct AlignmentCostFunctor {

  const Position &landmark_position;
  const Vertex &vertex;

  AlignmentCostFunctor(
      const Position &landmark_position, const Vertex &vertex
  ) : landmark_position(landmark_position), vertex(vertex) {}

  template<class T>
  bool operator()(
      T const *const stransformation, T *sResiduals
  ) const {
    // Map inputs to ceres::Jet types.
    Eigen::Map<Sophus::Sim3<T> const> const transformation(stransformation);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(sResiduals);

    residuals = transformation * landmark_position.cast<T>() - vertex.cast<T>();

    return true;
  }
};

Sophus::Sim3d alignment(
    std::vector<std::pair<const Position, const Vertex>> &correspondences,
    double huber_loss_parameter,
    int max_num_iterations = 20
) {

  ceres::Problem problem;

  Sophus::Sim3d transformation{Eigen::Matrix4d::Identity()};

  problem.AddParameterBlock(
      transformation.data(), Sophus::Sim3d::num_parameters, new Sophus::Manifold<Sophus::Sim3>());


  // Add a residual block for each visible landmark position projected in each image.
  for (auto &[landmark_position, vertex] : correspondences) {

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<AlignmentCostFunctor, 3, Sophus::Sim3d::num_parameters>(
            new AlignmentCostFunctor(
                landmark_position, vertex
            )), new ceres::HuberLoss(huber_loss_parameter), transformation.data());
  }

  ceres::Solver::Options options;
  options.max_num_iterations = max_num_iterations;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  ceres::Solve(
      options, &problem, &summary
  );

  std::cout << summary.BriefReport() << std::endl;

  return transformation;
}

struct AlignConfig {
  double distance_threshold_start = 0.5;
  double distance_threshold_end = 0.25;
  int max_num_alignment_steps = 10;
  int min_num_vertices_in_voxel = 10;
  double standard_deviation_scale = 3;
  int max_num_optimization_iterations = 20;
};

// TODO: Optimize in keyframe frame.
void align(
    Sophus::Sim3d &local_to_map_transformation,
    const GeometricMap &geometric_map,
    const Map &map,
    const AlignConfig &config = AlignConfig{}
) {

  std::vector<std::pair<const Position, const Vertex>> correspondences;

  for (int alignment_step = 0; alignment_step < config.max_num_alignment_steps; ++alignment_step) {
    correspondences.clear();

    double distance_threshold = config.distance_threshold_start
        - (alignment_step + 1) * (config.distance_threshold_start - config.distance_threshold_end)
            / config.max_num_alignment_steps;

    for (const auto &[landmark_index, landmark] : map.landmarks) {
      Position transformed_position = local_to_map_transformation * landmark.position;

      if (!geometric_map.is_valid(
          transformed_position, config.min_num_vertices_in_voxel, config.standard_deviation_scale
      ))
        continue;

      std::optional<Vertex> nearest_vertex = geometric_map.get_nearest_vertex(transformed_position, distance_threshold);

      if (!nearest_vertex) continue;

      correspondences.emplace_back(transformed_position, nearest_vertex.value());
    }

    Sophus::Sim3d new_transformation =
        alignment(correspondences, config.distance_threshold_end, config.max_num_optimization_iterations);

    local_to_map_transformation *= new_transformation;

    // fmt::print(
    //     "{}/{}: found {} valid correspondences with distance threshold {}\n",
    //     alignment_step + 1,
    //     config.max_num_alignment_steps,
    //     correspondences.size(),
    //     distance_threshold
    // );
  }
}

#endif //ALIGNMENT_H_
