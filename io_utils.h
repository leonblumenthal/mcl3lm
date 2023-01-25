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

void dump_trajectory(const std::vector<Pose> &trajectory, const std::string &path = "../trajectory.csv") {
  std::ofstream out{path};

  out << fmt::format("position_x,position_y,position_z,quaternion_w,quaternion_x,quaternion_y,quaternion_z\n");

  for (const Pose &pose : trajectory) {
    out << fmt::format(
        "{},{},{},{},{},{},{}\n",
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

#endif //IO_UTILS_H_
