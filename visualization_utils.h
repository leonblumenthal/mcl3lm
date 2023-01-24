#ifndef VISUALIZATION_UTILS_H_
#define VISUALIZATION_UTILS_H_

#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "visual_odometry_utils.h"

std::vector<cv::Scalar> colors{
    CV_RGB(51, 102, 204), CV_RGB(220, 57, 18), CV_RGB(255, 153, 0), CV_RGB(16, 150, 24), CV_RGB(153, 0, 153),
    CV_RGB(0, 153, 198), CV_RGB(221, 68, 119), CV_RGB(102, 170, 0), CV_RGB(184, 46, 46), CV_RGB(49, 99, 149)
};

std::vector<cv::MarkerTypes> marker_types{
    cv::MarkerTypes::MARKER_CROSS, cv::MarkerTypes::MARKER_TILTED_CROSS, cv::MarkerTypes::MARKER_STAR,
    cv::MarkerTypes::MARKER_DIAMOND, cv::MarkerTypes::MARKER_SQUARE, cv::MarkerTypes::MARKER_TRIANGLE_UP,
    cv::MarkerTypes::MARKER_TRIANGLE_DOWN
};

void show_matches(
    const cv::Mat &image_1,
    const cv::Mat &image_2,
    const IndexMatches &matches,
    const Keypoints &keypoints_1,
    const Keypoints &keypoints_2
) {
  cv::Mat dual_image;
  cv::hconcat(image_1, image_2, dual_image);

  int color_index = 0;
  for (const auto &[index_1, index_2] : matches) {
    cv::Point2i point_1{keypoints_1[index_1].x(), keypoints_1[index_1].y()};
    cv::Point2i point_2{keypoints_2[index_2].x() + image_1.cols, keypoints_2[index_2].y()};
    cv::Scalar color = colors[color_index % colors.size()];

    cv::drawMarker(dual_image, point_1, color, cv::MARKER_STAR, 5);
    cv::drawMarker(
        dual_image, point_2, color, cv::MARKER_STAR, 5
    );
    cv::line(
        dual_image, point_1, point_2, color
    );

    ++color_index;
  }
  cv::imshow("", dual_image);
  cv::waitKey(0);
}

void draw_new_pose(
    cv::Mat &image,
    const Pose &pose,
    const IndexMatches &landmark_to_keypoint,
    const Keypoints &keypoints,
    const Map &map,
    const Camera<> &camera
) {
  std::unordered_map<int, Point> landmark_points = project_landmarks(map.landmarks, pose, camera);

  // Draw matching landmarks and keypoints in green.
  std::unordered_set<int> matched_landmark_indices;
  std::unordered_set<int> matched_keypoint_indices;
  for (const auto &[landmark_index, keypoint_index] : landmark_to_keypoint) {
    cv::Point2d cv_landmark_point{landmark_points[landmark_index].x(), landmark_points[landmark_index].y()};
    cv::drawMarker(image, cv_landmark_point, CV_RGB(0, 255, 0), cv::MARKER_DIAMOND, 5);

    cv::Point2i cv_keypoint{keypoints[keypoint_index].x(), keypoints[keypoint_index].y()};
    cv::drawMarker(image, cv_keypoint, CV_RGB(0, 255, 0), cv::MARKER_TILTED_CROSS, 5);

    cv::line(image, cv_landmark_point, cv_keypoint, CV_RGB(0, 255, 0));

    matched_landmark_indices.insert(landmark_index);
    matched_keypoint_indices.insert(keypoint_index);
  }

  // Draw unmatched landmarks in blue.
  for (const auto &[index, point] : landmark_points) {
    if (matched_landmark_indices.contains(index)) continue;
    cv::Point2d cv_point{point.x(), point.y()};
    cv::drawMarker(image, cv_point, CV_RGB(0, 0, 255), cv::MARKER_DIAMOND, 5);
  }
  // Draw unmatched keypoints in red.
  for (int i = 0; i < keypoints.size(); ++i) {
    if (matched_keypoint_indices.contains(i)) continue;
    cv::Point2i cv_point{keypoints[i].x(), keypoints[i].y()};
    cv::drawMarker(image, cv_point, CV_RGB(255, 0, 0), cv::MARKER_TILTED_CROSS, 5);
  }
}

// Draw projected landmarks and keypoints for a keyframe.
void draw_keyframe(cv::Mat &image, int keyframe_index, const Map &map, const Camera<> &camera) {
  const Keyframe &keyframe = map.keyframes.at(keyframe_index);

  std::unordered_map<int, Point> landmark_points = project_landmarks(map.landmarks, keyframe.pose, camera);

  // Draw each landmark and the matching keypoint if applicable.
  std::unordered_set<int> matched_keypoint_indices;
  for (const auto &[index, landmark] : map.landmarks) {
    if (!landmark_points.contains(index)) continue;

    // Color red for unmachted landmarks.
    cv::Scalar color = CV_RGB(0, 0, 255);
    cv::Point2d cv_landmark_point{landmark_points[index].x(), landmark_points[index].y()};

    if (landmark.observations.contains(keyframe_index)) {
      // Color green for matches.
      color = CV_RGB(0, 255, 0);
      const int keypoint_index = landmark.observations.at(keyframe_index);
      matched_keypoint_indices.insert(keypoint_index);

      // Draw keypoint and connecting line to the landmark.
      cv::Point2i cv_keypoint{keyframe.keypoints[keypoint_index].x(), keyframe.keypoints[keypoint_index].y()};
      cv::drawMarker(image, cv_keypoint, color, cv::MARKER_TILTED_CROSS, 5);
      cv::line(image, cv_landmark_point, cv_keypoint, color);
    }

    cv::drawMarker(image, cv_landmark_point, color, cv::MARKER_DIAMOND, 5);
  }

  // Draw unmachted keypoints.
  for (int i = 0; i < keyframe.keypoints.size(); ++i) {
    if (matched_keypoint_indices.contains(i)) continue;

    cv::Point2i cv_point{keyframe.keypoints[i].x(), keyframe.keypoints[i].y()};
    cv::drawMarker(image, cv_point, CV_RGB(255, 0, 0), cv::MARKER_TILTED_CROSS, 5);
  }

}

void show_keyframes(
    const std::vector<std::string> &image_paths, const std::vector<int> &indices, const Map &map, const Camera<> &camera
) {
  for (const int &index : indices) {
    cv::Mat image = cv::imread(image_paths[index]);
    draw_keyframe(image, index, map, camera);
    cv::imshow(std::to_string(index), image);
  }
  cv::waitKey(0);
}

#endif //VISUALIZATION_UTILS_H_
