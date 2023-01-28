#ifndef PCL_MAP_H_
#define PCL_MAP_H_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>
#include <fmt/core.h>
#include <sophus/se3.hpp>
#include <optional>
#include <float.h>

using Vertex = Eigen::Vector3d;
using Vertices = std::vector<Eigen::Vector3d>;

struct Bounds {
  double min_x;
  double min_y;
  double min_z;
  double max_x;
  double max_y;
  double max_z;
};

template<typename T> using Voxels = std::vector<std::vector<std::vector<T>>>;

struct LocalDistribution {
  // The eigenvectors in Axes are not ordered by value.
  Eigen::Matrix3d axes;
  Eigen::Vector3d mean;
  Eigen::Vector3d standard_deviation;
};

// Load vertices from a .ply file.
// Properties must start with x, y, z.
Vertices load_vertices(const std::string &ply_path, int limit = 0) {
  std::ifstream pcl_file(ply_path);

  // Check if first line is "ply".
  std::string line;
  getline(pcl_file, line);
  if (line != "ply") {
    std::cerr << ply_path << " does not start with \"ply\"" << std::endl;
    // TODO: Handle better.
    return {};
  }

  // Skip header.
  for (; getline(pcl_file, line);) {
    if (line == "end_header") break;
  }

  Vertices vertices;

  std::vector<std::string> parts;
  for (; getline(pcl_file, line);) {
    std::stringstream sstream{line};
    parts.clear();
    for (std::string part; getline(sstream, part, ' ');) parts.push_back(part);

    Eigen::Vector3d vertex;
    vertex << std::stod(parts[0]), std::stod(parts[1]), std::stod(parts[2]);

    vertices.push_back(vertex);

    // Stop if limit is reached.
    if (limit > 0 && vertices.size() == limit) break;
  }

  return vertices;
}

// Compute "local distribution" aka PCA.
LocalDistribution compute_local_distribution(const Vertices &vertices) {

  // Compute mean.
  Vertex mean = Vertex::Zero();
  for (const Vertex &vertex : vertices) {
    mean += vertex;
  }
  mean /= static_cast<double>(vertices.size());

  // Compute covariance matrix.
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const Vertex &vertex : vertices) {
    Vertex centered_vertex = vertex - mean;
    covariance += centered_vertex * centered_vertex.transpose();
  }
  covariance /= static_cast<double>(vertices.size()) - 1;

  // Compute eigenvalues and eigenvectors.
  Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(covariance);

  Eigen::Matrix3d eigenvectors = eigen_solver.eigenvectors().real();
  Eigen::Vector3d standard_deviation = eigen_solver.eigenvalues().real().cwiseSqrt();

  return {eigenvectors, mean, standard_deviation};
}

template<typename T>
Voxels<T> initialize_voxels(size_t num_x, size_t num_y, size_t num_z) {
  std::vector<T> z_voxels(num_z);
  std::vector<std::vector<T>> y_voxels(num_y, z_voxels);
  Voxels<T> voxels(num_x, y_voxels);
  return voxels;
}

std::tuple<int, int, int> compute_voxel_index(
    const Vertex &vertex, double voxel_size, int min_x, int min_y, int min_z
) {
  int x_index = static_cast<int>(std::floor(vertex.x() / voxel_size)) - min_x;
  int y_index = static_cast<int>(std::floor(vertex.y() / voxel_size)) - min_y;
  int z_index = static_cast<int>(std::floor(vertex.z() / voxel_size)) - min_z;

  return {x_index, y_index, z_index};
}

template<typename T>
std::tuple<size_t, size_t, size_t> get_shape(const Voxels<T> voxels) {
  return {voxels.size(), voxels[0].size(), voxels[0][0].size()};
}

struct GeometricMap {
  const double voxel_size;

  int min_index_x;
  int min_index_y;
  int min_index_z;

  int num_x;
  int num_y;
  int num_z;

  Voxels<Vertices> vertices_voxels;
  Voxels<LocalDistribution> local_distributions_voxels;

  GeometricMap(double voxel_size, Bounds bounds) : voxel_size(voxel_size) {
    min_index_x = static_cast<int>(std::floor(bounds.min_x / voxel_size));
    min_index_y = static_cast<int>(std::floor(bounds.min_y / voxel_size));
    min_index_z = static_cast<int>(std::floor(bounds.min_z / voxel_size));
    int max_index_x = static_cast<int>(std::floor(bounds.max_x / voxel_size));
    int max_index_y = static_cast<int>(std::floor(bounds.max_y / voxel_size));
    int max_index_z = static_cast<int>(std::floor(bounds.max_z / voxel_size));

    num_x = max_index_x - min_index_x + 1;
    num_y = max_index_y - min_index_y + 1;
    num_z = max_index_z - min_index_z + 1;

    vertices_voxels = initialize_voxels<Vertices>(num_x, num_y, num_z);
    local_distributions_voxels = initialize_voxels<LocalDistribution>(num_x, num_y, num_z);
  }

  void initialize(const std::string &ply_path) {
    Vertices vertices = load_vertices(ply_path);

    fmt::print("Loaded {} vertices\n", vertices.size());

    Bounds voxels_bounds{
        min_index_x * voxel_size, min_index_y * voxel_size, min_index_z * voxel_size,
        (min_index_x + num_x + 1) * voxel_size, (min_index_y + num_y + 1) * voxel_size,
        (min_index_z + num_z + 1) * voxel_size
    };

    fmt::print(
        "Initialized {} voxels\n  x: {} ({:.2f}, {:.2f})\n  y: {} ({:.2f}, {:.2f})\n  z: {} ({:.2f}, {:.2f})\n",
        vertices_voxels.size() * vertices_voxels[0].size() * vertices_voxels[0][0].size(),
        num_x,
        voxels_bounds.min_x,
        voxels_bounds.max_x,
        num_y,
        voxels_bounds.min_y,
        voxels_bounds.max_y,
        num_z,
        voxels_bounds.min_z,
        voxels_bounds.max_z
    );

    int num_skipped_vertices = 0;
    for (const auto &vertex : vertices) {
      auto [x_index, y_index, z_index] = compute_voxel_index(vertex, voxel_size, min_index_x, min_index_y, min_index_z);

      if (x_index < 0 || x_index >= num_x || y_index < 0 || y_index >= num_y || z_index < 0 || z_index >= num_z) {
        ++num_skipped_vertices;
        continue;
      }

      vertices_voxels[x_index][y_index][z_index].push_back(vertex);
    }

    fmt::print(
        "Filled voxels, {} outlier vertices\n", num_skipped_vertices
    );

    // Compute PCA for each voxel.
    int num_computed_local_distributions = 0;
    for (int x_index = 0; x_index < num_x; ++x_index) {
      for (int y_index = 0; y_index < num_y; ++y_index) {
        for (int z_index = 0; z_index < num_z; ++z_index) {
          if (vertices_voxels[x_index][y_index][z_index].empty()) continue;

          local_distributions_voxels[x_index][y_index][z_index] = compute_local_distribution(
              vertices_voxels[x_index][y_index][z_index]
          );
          ++num_computed_local_distributions;
        }
      }
    }

    fmt::print("Computed {} local distributions\n", num_computed_local_distributions);
  }

  // TODO: Constant.
  bool is_valid(const Eigen::Vector3d &position, int min_num_vertices = 10, double std_scale = 3) const {
    for (const auto &[index_x, index_y, index_z] : get_neigboring_voxel_indices(position)) {
      const LocalDistribution &local_distribution = local_distributions_voxels[index_x][index_y][index_z];

      if (vertices_voxels[index_x][index_y][index_z].size() < min_num_vertices) continue;

      Eigen::Vector3d
          projection = (local_distribution.axes.transpose() * (position - local_distribution.mean)).cwiseAbs();
      if (projection.x() <= local_distribution.standard_deviation.x() * std_scale
          && projection.y() <= local_distribution.standard_deviation.y() * std_scale
          && projection.z() <= local_distribution.standard_deviation.z() * std_scale) {
        return true;
      }

    }

    return false;

  }

  std::optional<Vertex> get_nearest_vertex(const Eigen::Vector3d &position, double distance_threshold) const {

    std::tuple<int, int, int> best_voxel_index;
    int best_vertex_index = -1;
    double min_distance = DBL_MAX;
    for (const auto &[index_x, index_y, index_z] : get_neigboring_voxel_indices(position)) {
      for (int i = 0; i < vertices_voxels[index_x][index_y][index_z].size(); ++i) {
        const Vertex &vertex = vertices_voxels[index_x][index_y][index_z][i];
        double distance = (vertex - position).norm();

        if (distance > distance_threshold) continue;

        if (distance < min_distance) {
          best_voxel_index = {index_x, index_y, index_z};
          best_vertex_index = i;
          min_distance = distance;
        }
      }
    }

    if (best_vertex_index == -1) return {};

    const auto &[index_x, index_y, index_z] = best_voxel_index;

    return vertices_voxels[index_x][index_y][index_z][best_vertex_index];
  }

 private:
  std::vector<std::tuple<int, int, int>> get_neigboring_voxel_indices(const Eigen::Vector3d &position) const {
    auto [index_x, index_y, index_z] = compute_voxel_index(position, voxel_size, min_index_x, min_index_y, min_index_z);

    std::vector<std::tuple<int, int, int>> indices;

    for (int offset_x = -1; offset_x < 2; ++offset_x) {
      for (int offset_y = -1; offset_y < 2; ++offset_y) {
        for (int offset_z = -1; offset_z < 2; ++offset_z) {
          int i_x = index_x + offset_x;
          int i_y = index_y + offset_y;
          int i_z = index_z + offset_z;
          if (i_x < 0 || i_x >= num_x || i_y < 0 || i_y >= num_y || i_z < 0 || i_z >= num_z)
            continue;

          indices.emplace_back(i_x, i_y, i_z);

        }
      }
    }

    return indices;

  }

};


// void dump(
//     const Voxels<LocalDistribution> &local_distribution_voxels,
//     const Voxels<Vertices> &vertices_voxels,
//     double voxel_size,
//     const Bounds &bounds,
//     const std::string &path = "../local_distribution.csv"
// ) {
//   std::ofstream out{path};
//
//   out << fmt::format("min_x,min_y,min_z,voxel_size\n{},{},{},{}\n", bounds.min_x, bounds.min_y, bounds.min_z, voxel_size);
//   out << fmt::format(
//       "index_x,index_y,index_z,num_vertices,mean_x,mean_y,mean_z,quaternion_w,quaternion_x,quaternion_y,quaternion_z,std_x,std_y,std_z\n"
//   );
//
//   auto [num_x_voxels, num_y_voxels, num_z_voxels] = get_shape(vertices_voxels);
//   for (int x_index = 0; x_index < num_x_voxels; ++x_index) {
//     for (int y_index = 0; y_index < num_y_voxels; ++y_index) {
//       for (int z_index = 0; z_index < num_z_voxels; ++z_index) {
//         if (vertices_voxels[x_index][y_index][z_index].empty()) continue;
//
//         const Sophus::SE3d &transformation = local_distribution_voxels[x_index][y_index][z_index].transformation;
//         const Eigen::Vector3d
//             &standard_deviation = local_distribution_voxels[x_index][y_index][z_index].standard_deviation;
//
//         out << fmt::format(
//             "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
//             x_index,
//             y_index,
//             z_index,
//             vertices_voxels.size(),
//             transformation.translation().x(),
//             transformation.translation().y(),
//             transformation.translation().z(),
//             transformation.unit_quaternion().w(),
//             transformation.unit_quaternion().x(),
//             transformation.unit_quaternion().y(),
//             transformation.unit_quaternion().z(),
//             standard_deviation.x(),
//             standard_deviation.y(),
//             standard_deviation.z());
//       }
//     }
//   }
//
//   out.close();
// }

#endif //PCL_MAP_H_

