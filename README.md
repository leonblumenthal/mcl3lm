# Monocular Camera Localization in 3D LiDAR Maps
This is an implementation of the paper "Monocular Camera Localization in 3D LiDAR Maps" [1] using C++ and Python. 


It was my final project for the practical course [Vision-based Navigation](https://vision.in.tum.de/teaching/ws2022/visnav_ws2022) by the Computer Vision Group at Technical University of Munich.

See run.py for an example how to run the approach for the EuRoC dataset. The notebooks created the runs and figures for my presentation.

# Installation

Install all dependencies from CMakeLists.txt with e.g. brew on MacOS. 

For OpenGV, I had to use a custom (hacky) installation to support C++20:
- Clone github repo
- Modify cmake to C++20
- Remove eigen allocator from types 


# References
[1] T. Caselitz, B. Steder, M. Ruhnke and W. Burgard, "Monocular camera localization in 3D LiDAR maps," 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2016, pp. 1926-1931, doi: 10.1109/IROS.2016.7759304.
