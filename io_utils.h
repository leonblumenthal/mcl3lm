#ifndef IO_UTILS_H_
#define IO_UTILS_H_

#include "filesystem"
#include <vector>
#include "visual_odometry_utils.h"
#include <fmt/core.h>

namespace fs = std::filesystem;

std::vector<std::string> get_image_paths(
    const std::string &directory, const std::string &file_ending, int limit = 0, bool sort = true
) {

  std::vector<std::string> paths;
  for (const auto &entry : fs::directory_iterator(directory)) {
    const fs::path &path = entry.path();
    if (path.extension().string().substr(1) == file_ending) paths.emplace_back(path.string());
  }

  if (sort) std::sort(paths.begin(), paths.end());

  if (limit > 0 and limit < paths.size()) paths = std::vector<std::string>(paths.begin(), paths.begin() + limit);

  return paths;
}

void dump_map(const Map &map, const std::string &path = "../map_dump") {
  std::ofstream out{path};

  out << "keyframes" << std::endl;
  bool first = true;
  for (const auto &[keyframe_index, keyframe] : map.keyframes) {
    if (!first)
      out << std::endl;
    first = false;

    out << keyframe_index << std::endl;
    auto &t = keyframe.pose.translation();
    auto &q = keyframe.pose.unit_quaternion();

    out << t.x() << " " << t.y() << " " << t.z() << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
        << std::endl;
    for (int i = 0; i < keyframe.keypoints.size(); ++i) {
      out << keyframe.keypoints[i].x() << " " << keyframe.keypoints[i].y() << " " << keyframe.descriptors[i]
          << std::endl;
    }
  }

  out << "landmarks" << std::endl;
  first = true;
  for (const auto &[landmark_index, landmark] : map.landmarks) {
    if (!first)
      out << std::endl;
    first = false;

    out << landmark_index << std::endl;
    out << landmark.position.x() << " " << landmark.position.y() << " " << landmark.position.z() << std::endl;
    for (const auto &[keyframe_index, corner_index] : landmark.observations) {
      out << keyframe_index << " " << corner_index << std::endl;
    }
  }

  out.close();
}

void dump_trajectory(
    const std::vector<Pose> &trajectory,
    const std::string &path = "../trajectory.csv",
    const std::vector<int> &initial_frame_indices = {0}
) {
  std::ofstream out{path};

  out << fmt::format(
      "frame_index,position_x,position_y,position_z,quaternion_w,quaternion_x,quaternion_y,quaternion_z\n"
  );

  int frame_index;
  for (int i = 0; i < trajectory.size(); ++i) {
    if (i < initial_frame_indices.size()) {
      frame_index = initial_frame_indices[i];
    } else {
      frame_index = initial_frame_indices.back() + i - initial_frame_indices.size() + 1;
    }
    const Pose &pose = trajectory[i];

    out << fmt::format(
        "{},{},{},{},{},{},{},{}\n",
        frame_index,
        pose.translation().x(),
        pose.translation().y(),
        pose.translation().z(),
        pose.unit_quaternion().w(),
        pose.unit_quaternion().x(),
        pose.unit_quaternion().y(),
        pose.unit_quaternion().z());
  }

  out.close();
}

void dump_transformations(
    const std::vector<Sophus::Sim3d> &transformations, const std::string &path = "../transformations.csv"
) {
  std::ofstream out{path};

  out << fmt::format("position_x,position_y,position_z,quaternion_w,quaternion_x,quaternion_y,quaternion_z\n");

  for (const Sophus::Sim3d &transformation : transformations) {
    out << fmt::format(
        "{},{},{},{},{},{},{}\n",
        transformation.translation().x(),
        transformation.translation().y(),
        transformation.translation().z(),
        transformation.quaternion().w(),
        transformation.quaternion().x(),
        transformation.quaternion().y(),
        transformation.quaternion().z());
  }

  out.close();
}

#endif //IO_UTILS_H_
