#include <Eigen/Dense>
#include <iostream>
#include "pcl_map.h"

#include <fmt/core.h>

int main() {

  const std::string ply_path = "../data/euroc_mav/V1_01_easy/pointcloud0/data.ply";
  const double voxel_size = 0.1;

  auto [vertices_voxels, voxels_bounds] = load_vertices_voxels(
      ply_path, voxel_size, {-4.73, -3.49, -0.04, 4.10, 5.16, 4.13}
  );

  Voxels<LocalDistribution> local_distribution_voxels = compute_local_distribution_voxels(vertices_voxels);

}