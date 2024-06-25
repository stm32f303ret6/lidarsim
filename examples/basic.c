#define _GNU_SOURCE

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "lidarsim.h"

int main()
{
  // create lidar 
  lidar mid70;
  if (!create_lidar_from_file("../csv/mid70.csv", &mid70))
  {
    printf("Error loading lidar rays, check out the lidar csv filepath\n");
    return 0;
  }
  // create empty scene
  scene scn;
  scene_init(&scn);
  // add objects to the scene
  scene_object obj_plane = create_plane(50, 50);
  scene_add_object(&scn, obj_plane);

  scene_object obj_mesh;
  if (!create_mesh_from_file("../examples/meshes/hatchback.obj", &obj_mesh, 0.0254, 0.0254, 0.0254))
  {
    printf("Error loading mesh, check out the mesh filepath\n");
    return 0;
  }
  scene_add_object(&scn, obj_mesh);

  // update objects transform
  scn.objects[1].transform = create_transformation_matrix(15, 0, 0.0, 0, 0, 0, "XYZ"); // x,y,z,R,P,Y
  mid70.transform = create_transformation_matrix(0, 1, 1, 0, 0, 0, "XYZ");

  // update scene and rebuild
  scene_update(&scn, mid70);
  scene_build(&scn);

  // ray cast the scene and get the pointcloud
  pointcloud cloud = cast_rays(mid70, scn, 0, 400000);
  printf("Ray intersection count: %d \n", cloud.point_count);

  // save the pointcloud as .pcd file, and free memory
  save_as_pcd("basic.pcd", &cloud);
  pointcloud_free(&cloud);

  lidar_free(&mid70);
  scene_free(&scn);
  printf("\n");
  return 0;
}
