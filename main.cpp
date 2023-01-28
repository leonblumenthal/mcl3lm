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
#include "visual_odometry_reports.h"
#include "visualization_utils.h"
#include "visual_odometry.h"
#include "io_utils.h"

struct ALignmentCostFunctor {

  const Position &landmark_position;
  const Vertex &vertex;

  ALignmentCostFunctor(
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
    int max_num_iterations = 20,
    double huber_loss_parameter = 1.0
) {

  ceres::Problem problem;

  Sophus::Sim3d transformation{Eigen::Matrix4d::Identity()};

  problem.AddParameterBlock(
      transformation.data(), Sophus::Sim3d::num_parameters, new Sophus::Manifold<Sophus::Sim3>());


  // Add a residual block for each visible landmark position projected in each image.
  for (auto &[landmark_position, vertex] : correspondences) {

    // TODO: Actual huber loss from the paper.
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ALignmentCostFunctor, 3, Sophus::Sim3d::num_parameters>(
            new ALignmentCostFunctor(
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


void align(Sophus::Sim3d &local_to_map_transformation, const GeometricMap& geometric_map, const Map &map) {
  double distance_threshold_start = 0.5;
  double distance_threshold_end = 0.25;
  int max_num_alignment_steps = 10;
  int min_num_vertices_in_voxel = 10;
  double standard_deviation_scale = 3;
  int max_num_optimization_iterations = 20;

  std::vector<std::pair<const Position, const Vertex>> correspondences;

  for (int alignment_step = 0; alignment_step < max_num_alignment_steps; ++alignment_step) {
    correspondences.clear();

    double distance_threshold = distance_threshold_start
        - (alignment_step + 1) * (distance_threshold_start - distance_threshold_end) / max_num_alignment_steps;

    for (const auto &[landmark_index, landmark] : map.landmarks) {
      Position transformed_position = local_to_map_transformation * landmark.position;

      if (!geometric_map.is_valid(transformed_position, min_num_vertices_in_voxel, standard_deviation_scale)) continue;

      std::optional<Vertex> nearest_vertex = geometric_map.get_nearest_vertex(transformed_position, distance_threshold);

      if (!nearest_vertex) continue;

      correspondences.emplace_back(transformed_position, nearest_vertex.value());
    }

    Sophus::Sim3d
        new_transformation = alignment(correspondences, max_num_optimization_iterations, distance_threshold_end);

    local_to_map_transformation *= new_transformation;

    fmt::print(
        "{}/{}: found {} valid correspondences with distance threshold {}\n",
        alignment_step + 1,
        max_num_alignment_steps,
        correspondences.size(),
        distance_threshold
    );
  }
}

int main() {

  const std::string ply_path = "../data/euroc_mav/V1_01_easy/pointcloud0/data.ply";
  const Bounds bounds{-4.73, -3.49, -0.04, 4.10, 5.16, 4.13};
  const double voxel_size = 0.25;

  GeometricMap geometric_map{voxel_size, bounds};
  geometric_map.initialize(ply_path);

  const Camera camera{
      752.0, 480.0, {458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05}};
  const std::vector<std::string> image_paths = get_image_paths("../data/euroc_mav/V1_01_easy/cam0/data", "png");
  // TODO: Poses seem to be wrong. Bundle adjustment works hard to correct them.
  // Specify initial two poses to start with correct scale.
  const std::pair<int, int> initial_keyframe_indices{100, 120};
  // TODO: Somethin is messed up the the wxyz order, either here or in the notebook.
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

  // dump_map(vo.map);

  // show_keyframes({0, 1}, vo.map, 1, vo.camera);


  // TODO: Optimize in keyframe frame.
  Sophus::Sim3d local_to_map_transformation{Eigen::Matrix4d::Identity()};

  // align(local_to_map_transformation, geometric_map, vo.map);

  std::cout << local_to_map_transformation.rotationMatrix() << std::endl;
  std::cout << local_to_map_transformation.translation() << std::endl;
  std::cout << local_to_map_transformation.scale() << std::endl;

  {


    // Pose pose = vo.current_pose;
    // Pose pose = initial_pose_2;
    // Sophus::SE3d{Eigen::Quaterniond{0.061541, -0.827324, -0.058484, -0.555272}, {0.87029955, 2.2052097, 0.92827237}};
    // Pose pose = Pose::rotX(0);
    // pose.translation() = {0.87029955, 2.2052097, 0.92827237};
    // Eigen::Matrix3d rot;
    // rot << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    // pose.setRotationMatrix(rot);
    //
    // pose = pose * Pose::rotY(-14.0 / 180 * EIGEN_PI) * Pose::rotX(-21.0 / 180 * EIGEN_PI);

    // std::cout << pose.unit_quaternion() << std::endl;
    // std::cout << pose.translation() << std::endl;
    // std::cout << pose.rotationMatrix() << std::endl;

    // cv::Mat image = cv::imread(image_paths[initial_keyframe_indices.second]);
    //
    // for (int x_index = 0; x_index < num_x_voxels; ++x_index) {
    //   for (int y_index = 0; y_index < num_y_voxels; ++y_index) {
    //     for (int z_index = 0; z_index < num_z_voxels; ++z_index) {
    //       if (vertices_voxels[x_index][y_index][z_index].empty()) continue;
    //
    //       // Eigen::Vector3d offset{
    //       //     x_index * voxel_size + voxels_bounds.min_x, y_index * voxel_size + voxels_bounds.min_y,
    //       //     z_index * voxel_size + voxels_bounds.min_z
    //       // };
    //       //
    //       // std::vector<Eigen::Vector3d> local_corners = {
    //       //     pose.inverse() * (Eigen::Vector3d{0, 0, 0} * voxel_size + offset),
    //       //     pose.inverse() * (Eigen::Vector3d{1, 0, 0} * voxel_size + offset),
    //       //     pose.inverse() * (Eigen::Vector3d{1, 1, 0} * voxel_size + offset),
    //       //     pose.inverse() * (Eigen::Vector3d{0, 1, 0} * voxel_size + offset),
    //       //     pose.inverse() * (Eigen::Vector3d{0, 0, 1} * voxel_size + offset),
    //       //     pose.inverse() * (Eigen::Vector3d{0, 0, 1} * voxel_size + offset),
    //       //     pose.inverse() * (Eigen::Vector3d{0, 0, 1} * voxel_size + offset),
    //       //     pose.inverse() * (Eigen::Vector3d{0, 0, 1} * voxel_size + offset)
    //       // };
    //
    //
    //       Position local_position = pose.inverse() * local_distribution_voxels[x_index][y_index][z_index].mean;
    //       if (local_position.z() <= 0.1)
    //         continue;
    //
    //       Point image_position = camera.project(local_position);
    //
    //       if (image_position.x() < 0 || image_position.y() < 0 || image_position.x() > camera.width
    //           || image_position.y() > camera.height)
    //         continue;
    //
    //       Position local_position_x = pose.inverse() * (local_distribution_voxels[x_index][y_index][z_index].mean
    //           + local_distribution_voxels[x_index][y_index][z_index].axes.col(0)
    //               * local_distribution_voxels[x_index][y_index][z_index].standard_deviation.x());
    //       Position local_position_y = pose.inverse() * (local_distribution_voxels[x_index][y_index][z_index].mean
    //           + local_distribution_voxels[x_index][y_index][z_index].axes.col(1)
    //               * local_distribution_voxels[x_index][y_index][z_index].standard_deviation.y());
    //       Position local_position_z = pose.inverse() * (local_distribution_voxels[x_index][y_index][z_index].mean
    //           + local_distribution_voxels[x_index][y_index][z_index].axes.col(2)
    //               * local_distribution_voxels[x_index][y_index][z_index].standard_deviation.z());
    //       Point image_position_x = camera.project(local_position_x);
    //       Point image_position_y = camera.project(local_position_y);
    //       Point image_position_z = camera.project(local_position_z);
    //
    //       cv::Point2d cv_point{image_position.x(), image_position.y()};
    //       cv::Point2d cv_point_x{image_position_x.x(), image_position_x.y()};
    //       cv::Point2d cv_point_y{image_position_y.x(), image_position_y.y()};
    //       cv::Point2d cv_point_z{image_position_z.x(), image_position_z.y()};
    //
    //
    //       // cv::drawMarker(image, cv_point, CV_RGB(0, 0, 0), cv::MARKER_DIAMOND, 5);
    //       // cv::drawMarker(image, cv_point_x, CV_RGB(255, 0, 0), cv::MARKER_STAR, 2);
    //       // cv::drawMarker(image, cv_point_y, CV_RGB(0, 255, 0), cv::MARKER_STAR, 2);
    //       // cv::drawMarker(image, cv_point_z, CV_RGB(0, 0, 255), cv::MARKER_STAR, 2);
    //       if (local_position_x.z() > 0.1)
    //         cv::line(image, cv_point, cv_point_x, CV_RGB(255, 0, 255), 1);
    //       if (local_position_y.z() > 0.1)
    //         cv::line(image, cv_point, cv_point_y, CV_RGB(255, 0, 255), 1);
    //       if (local_position_z.z() > 0.1)
    //         cv::line(image, cv_point, cv_point_z, CV_RGB(255, 0, 255), 1);
    //
    //       //
    //       // const Vertices &vertices = vertices_voxels[x_index][y_index][z_index];
    //       //
    //       // for (const Vertex& vertex : vertices) {
    //       //   Position local_position = initial_pose_1.inverse() * vertex;
    //       //   if (local_position.z() < 0)
    //       //     continue;
    //       //
    //       //   Point image_position = camera.project(local_position);
    //       //
    //       //   if (image_position.x() < 0 || image_position.y() < 0 || image_position.x() > camera.width
    //       //       || image_position.y() > camera.height)
    //       //     continue;
    //       //
    //       //   cv::Point2d cv_point{image_position.x(), image_position.y()};
    //       //   cv::drawMarker(image, cv_point, CV_RGB(0, 0, 255), cv::MARKER_DIAMOND, 5);
    //       //   break;
    //       // }
    //     }
    //   }
    // }
    //
    // cv::imshow("", image);
    // cv::waitKey(0);

  }


  // TODO: Remove
  // show_keyframes({0, 1}, vo.map, 1, vo.camera);


  std::vector<Pose> trajectory{vo.map.keyframes[0].pose, vo.map.keyframes[1].pose};
  trajectory.reserve(image_paths.size());

  trajectory.push_back(vo.map.keyframes[0].pose);
  trajectory.push_back(vo.map.keyframes[1].pose);

  ReportWriter next_report_writer("../next_report.csv");
  ReportWriter keyframe_report_writer("../keyframe_report.csv");

  int last_keyframe_index = 1;

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


    if (last_keyframe_index != vo.last_keyframe_index) {
      last_keyframe_index = vo.last_keyframe_index;
      // align(local_to_map_transformation, geometric_map, vo.map);
      // show_keyframes({last_keyframe_index}, vo.map, last_keyframe_index, camera);
    }

    // TODO: Transform entire pose.
    Pose transformed_pose = vo.current_pose;
    transformed_pose.translation() = local_to_map_transformation * transformed_pose.translation();
    trajectory.push_back(transformed_pose);
  }

  dump_trajectory(trajectory);

  return 0;

}