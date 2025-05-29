# lidarsim

LIDARSIM is a **header-only C library** to simulate any lidar, such as Livox, Ouster, etc.

## Description

This library allows you to simulate LiDAR sensors in a virtual 3D environment. It's optimized for performance and extensibility, making it suitable for robotics, perception pipelines, and research.

![Image](https://github.com/user-attachments/assets/d03c6c08-e91e-4793-9fa8-a21665baa878)

### Key Features

- Header-only, fast, and portable
- Optimized ray casting using AABB BVH trees
- Mesh BVH built per object (parallelizable)
- Load STL and OBJ mesh files
- Extensible to any lidar pattern (Livox, Ouster, Velodyne, etc.)
- Designed for multithreading or GPU acceleration (CUDA/ROCm)
- Simple transformation API for scene and objects

## How to Use

1. Create a lidar object and an empty scene.
2. Add scene objects (boxes, cylinders, planes, or meshes).
3. Apply transformations to objects and rebuild the scene.
4. Cast rays using the lidar to generate a point cloud.
5. Repeat step 3 to simulate motion or changes.

## Build and Run Instructions

### Prerequisites

- [CMake](https://cmake.org/)
- [PCL (Point Cloud Library)](https://pointclouds.org/) (optional)

### Installation

```bash
git clone git@github.com:stm32f303ret6/lidarsim.git
cd lidarsim
mkdir build
cd build
cmake ..
make
```
This will generate **basic.pcd**, you can view it using tools like pcl_viewer
```bash
sudo apt-get install pcl-tools
pcl_viewer basic.pcd
```
