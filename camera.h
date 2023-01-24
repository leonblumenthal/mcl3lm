#ifndef CAMERA_H_
#define CAMERA_H_


#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

// Pinhole camera and distortion parameters.
template<typename Scalar = double>
struct CameraParameters {
  Scalar fx;
  Scalar fy;
  Scalar cx;
  Scalar cy;
  Scalar k1;
  Scalar k2;
  Scalar p1;
  Scalar p2;
};
template<typename T> CameraParameters(T, T, T, T, T, T, T, T) -> CameraParameters<T>;

// Pinhole camera including distortion.
template<typename Scalar = double>
struct Camera {

  using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  const Scalar width;
  const Scalar height;
  const CameraParameters<Scalar> parameters;

  // Parameters required by OpenCV for the unprojection.
  const cv::Matx<Scalar, 3, 3> camera_matrix;
  const cv::Vec<Scalar, 4> distortion_coefficients;
  // TODO: Check if wrapping with Scalar is necessary
  const cv::Vec<Scalar, 3> rotation{Scalar(0), Scalar(0), Scalar(0)};
  const cv::Vec<Scalar, 3> translation{Scalar(0), Scalar(0), Scalar(0)};

  Camera(
      Scalar width, Scalar height, const CameraParameters<Scalar> &parameters
  ) : width{width}, height{height}, parameters{parameters}, camera_matrix{
      parameters.fx, Scalar(0), parameters.cx, Scalar(0), parameters.fy, parameters.cy, Scalar(0), Scalar(0), Scalar(1)
  }, distortion_coefficients{parameters.k1, parameters.k2, parameters.p1, parameters.p2} {}

  // Transform 3D points from the camera coordinate frame into 2D points in pixel coordinates.
  std::vector<Vector2> project(const std::vector<Vector3> &points) const {
    // OpenCV is not used here because it does not work with ceres.

    std::vector<Vector2> projected_points(points.size());
    for (int i = 0; i < points.size(); ++i) projected_points[i] = project(points[i]);

    return projected_points;
  }

  // Transform a 3D point from the camera coordinate frame into 2D points in pixel coordinates.
  Vector2 project(const Vector3 &point) const {
    // Pinhole.
    Scalar x = point.x() / point.z();
    Scalar y = point.y() / point.z();

    // Distortion.
    Scalar r2 = x * x + y * y;
    const Scalar radial_distortion{static_cast<Scalar>(Scalar(1) + parameters.k1 * r2 + parameters.k2 * r2 * r2)};
    Scalar u = x * radial_distortion + Scalar(2) * parameters.p1 * x * y + parameters.p2 * (r2 + Scalar(2) * x * x);
    Scalar v = y * radial_distortion + Scalar(2) * parameters.p2 * x * y + parameters.p1 * (r2 + Scalar(2) * y * y);

    // Pinhole.
    u = parameters.fx * u + parameters.cx;
    v = parameters.fy * v + parameters.cy;

    return {u, v};
  }

  // Transform 2D points in pixel coordinates into normalized 3D points in the camera coordinate frame.
  std::vector<Vector3> unproject(const std::vector<Vector2> &points) const {
    // Convert from Eigen to OpenCV.
    // TODO: Better conversion with cv::eigen2cv?
    std::vector<cv::Point_<Scalar>> cv_points(points.size());
    for (int i = 0; i < points.size(); ++i) {
      cv_points[i] = cv::Point_<Scalar>(points[i].x(), points[i].y());
    }

    // Revert projection including distortion.
    // OpenCV is used because no closed-from solution for the distortion exists.
    std::vector<cv::Point_<Scalar>> cv_unprojected_points;
    cv::undistortPoints(
        cv_points, cv_unprojected_points, camera_matrix, distortion_coefficients
    );

    // Normalize and convert from OpenCV to Eigen.
    // TODO: Better conversion with cv::cv2eigen?
    std::vector<Vector3> unprojected_points(points.size());
    for (int i = 0; i < points.size(); ++i) {
      unprojected_points[i] = Vector3(cv_unprojected_points[i].x, cv_unprojected_points[i].y, 1).normalized();
    }

    return unprojected_points;
  }

  // Transform a 2D point from pixel coordinates into a normalized 3D point in the camera coordinate frame.
  Vector3 unproject(const Vector2 &point) const {
    return unproject({std::vector<Vector2>{point}})[0];
  }

};

#endif //CAMERA_H_
