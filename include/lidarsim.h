/*
 * LIDARSIM
 * By Ricardo Casimiro
 * Initial release v1.0 2024-06-25 
 * Description:
    Header-only library to simulate any lidar, such as Livox, Ouster, etc.
 * How to use:
    (1) Create a lidar object and an empty scene.
    (2) Create scene objects like boxes, cylinders, meshes, and add them to the scene.
    (3) Update the transformation of the objects in the scene (including the lidar). All objects in the scene are referred to the world frame (0,0,0). Then rebuild the scene.
    (4) Ray cast the scene with the lidar and get the intersected points to generate a point cloud.
    (5) You can repeat the process from step (3).
 * Features:
    - Fast, portable, and easy to use. 
    - Supports C17 and C++17
    - Ray casting queries are optimized using AABB BVH trees for every object in the scene.
    - Ray casting queries can be parallelized using threads or GPU.
    - Each mesh object has its own AABB BVH tree, which can be built in parallel using threads or GPU.
    - Support to load STL and OBJ files
    - Extensible to any type of lidar laser pattern.
    - Easy to add support for GPU using CUDA or ROCm.
    - Easy to add support for SIMD vectorization.
*/

#ifndef UTILS_H_
#define UTILS_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

/* __STRUTCS__ */

// MATH STRUCTS
typedef struct
{
  float x, y, z;
} vec3;
typedef struct
{
  vec3 min, max;
} AABB;
typedef struct
{
  vec3 vertices[3];
  AABB aabb;
  vec3 edge1, edge2, centroid;
} triangle;

typedef struct
{
  vec3 ori;
  vec3 dir;
  vec3 inv_dir;
} ray;

typedef struct
{
  float m[4][4];
} mat4x4;

// GEOMETRIC PRIMITIVES STRUCTS

typedef struct
{
  float radius, _k;
} Sphere;

typedef struct
{
  float radius;
  float height;
} Cylinder;

typedef struct
{
  vec3 size;
  vec3 normals[3];
} Box;

typedef struct {
  vec3 _normal;  // Plane normal vector
  float _d;      // Plane constant
  float x_size, y_size;
} Plane;
typedef struct bvhmesh_node
{
  AABB box;
  uint16_t object_index; // Index to the scene object
  triangle *tri;         // Pointer to a triangle, if this node represents a triangle
  struct bvhmesh_node *left;
  struct bvhmesh_node *right;
} bvhmesh_node;
typedef struct
{
  triangle *original_triangles;
  triangle *transformed_triangles;
  uint32_t triangle_count;
  bvhmesh_node *root;
} Mesh;

// SCENE STRUCTS
typedef enum
{
  SPHERE,
  CYLINDER,
  BOX,
  PLANE,
  MESH
} object_type;

typedef struct
{
  object_type type;
  union
  {
    Sphere sphere;
    Cylinder cylinder;
    Box box;
    Plane plane;
    Mesh mesh;
  };
  mat4x4 transform, _inverse;
  vec3 _pos, _pos_neg;
  AABB aabb;
} scene_object;
// BVH AABB TREE STRUCTS
typedef struct bvh_node
{
  AABB box;
  uint16_t index;
  struct bvh_node *left;
  struct bvh_node *right;
} bvh_node;
typedef struct
{
  scene_object *objects;
  bvh_node *root;
  uint16_t current_size;
  uint32_t max_size;
} scene;
typedef struct
{
  float t;           // distance from ray origin to intersection
  scene_object *obj; // the closest intersected object
} ray_result;
// BVH AABB TREE STRUCTS FOR MESHES
typedef struct
{
  float t;       // distance from ray origin to intersection
  triangle *tri; // the closest intersected triangle
} ray_triangle_result;

typedef struct
{
  ray *rays;
  uint32_t ray_count;
  mat4x4 transform;
} lidar;
typedef struct
{
  vec3 *points;
  uint32_t point_count;
} pointcloud;
/* __FUNCTIONS__ */
// math
static inline vec3 vec3_sub(vec3 a, vec3 b);
static inline vec3 vec3_cross(vec3 a, vec3 b);
static inline float vec3_dot(vec3 a, vec3 b);
static inline vec3 vec3_scalar(vec3 v, float factor);
static inline vec3 vec3_add(vec3 a, vec3 b);
static inline vec3 vec3_negate(vec3 v);
static inline vec3 vec3_multiply(vec3 a, vec3 b);
static inline void scale_vertex(vec3 *vertex, float scale_x, float scale_y, float scale_z);
void mat4x4_identity(mat4x4 *matrix);
mat4x4 create_transformation_matrix(float x, float y, float z, float R, float P, float Y, const char *order);
mat4x4 create_rotation_matrix(const char order[], float R, float P, float Y);
static inline void apply_transformation_to_mesh(const mat4x4 *transform,
                                                Mesh *mesh);
// ray intersection
uint8_t ray_sphere(ray *r, scene_object *obj, float *t);
uint8_t ray_obb(ray *r, scene_object *obb, float *t);
uint8_t ray_aabb(ray *r, AABB *box, float *t);
uint8_t ray_triangle(const ray *r, const triangle *triangle, float *t);
// scene
void scene_init(scene *scn);
void scene_add_object(scene *scn, scene_object new_object);
void scene_update(scene *scn, lidar lidar_data);
void scene_build(scene *scn);
void scene_free(scene *scn);
void calculate_plane_from_matrix(mat4x4 *matrix, Plane *plane);
void calculate_plane_aabb(scene_object *obj);
// bvh aabb
int bvh_comparator(const void *a, const void *b, void *arg);
static inline void bvh_unionAABB(const AABB *a, const AABB *b, AABB *result);
bvh_node *bvh_build_tree(scene_object **objects, int start, int end);
void bvh_free_tree(bvh_node *node);
AABB bvh_calculate_aabb(scene_object *obj);
void bvh_update_npearest_intersection(ray_result *closest, float t,
                                      scene_object *obj);
void bvh_traverse_ray_aabb(bvh_node *node, ray *r, scene *scn,
                           ray_result *closest);
void print_bvh_tree(bvh_node *node, int depth);
// bvh aabb mesh
static inline void bvhmesh_calculate_triangle_aabb(triangle *tri);
bvhmesh_node *bvhmesh_build_tree(triangle *triangles, int start, int end,
                                 int depth);
void bvhmesh_free(bvhmesh_node *node);
void bvhmesh_update_nearest_intersection(ray_triangle_result *closest, float t,
                                         triangle *tri);
void bvhmesh_traverse_ray_aabb(bvhmesh_node *node, ray *r,
                               ray_triangle_result *closest);
// utils
ray *read_rays_from_csv(const char *csv_filepath, int *ray_count,
                        int csv_row_count);
void save_as_pcd(const char *filename, const pointcloud *cloud);
uint8_t create_lidar_from_file(const char *filename, lidar *lidar_data);
uint8_t create_mesh_from_file(const char *filename, scene_object *scene, float scale_x, float scale_y, float scale_z);
void pointcloud_free(pointcloud *cloud);
void lidar_free(lidar *lidar_data);
double tic();
void toc(const char *message, double start);
scene_object create_sphere(float radius);
scene_object create_plane(float x_size, float y_size);
scene_object create_cylinder(float radius, float height);
scene_object create_box(float width, float length, float height);
pointcloud cast_rays(lidar lidar_data, scene scn, uint32_t start, uint32_t end);
static inline vec3 cast_vec3(float x, float y, float z) {
    vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}
// MATH FUNCTIONS
static inline vec3 vec3_sub(vec3 a, vec3 b)
{
  return cast_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static inline vec3 vec3_cross(vec3 a, vec3 b)
{
  return cast_vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}
static inline float vec3_dot(vec3 a, vec3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline vec3 vec3_scalar(vec3 v, float factor)
{
  return cast_vec3(v.x * factor, v.y * factor, v.z * factor);
}
static inline vec3 vec3_add(vec3 a, vec3 b)
{
  return cast_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline vec3 vec3_negate(vec3 v) { return cast_vec3(-v.x, -v.y, -v.z); }
static inline vec3 vec3_multiply(vec3 a, vec3 b)
{
  return cast_vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
static inline void scale_vertex(vec3 *vertex, float scale_x, float scale_y, float scale_z) {
    vertex->x *= scale_x;
    vertex->y *= scale_y;
    vertex->z *= scale_z;
}
void mat4x4_identity(mat4x4 *matrix) {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == j) {
        matrix->m[i][j] = 1.0f;
      } else {
        matrix->m[i][j] = 0.0f;
      }
    }
  }
}
mat4x4 mat4x4_mul(const mat4x4 *a, const mat4x4 *b) {
    mat4x4 result;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.m[i][j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                result.m[i][j] += a->m[i][k] * b->m[k][j];
            }
        }
    }
    return result;
}

uint8_t mat4x4_invert(mat4x4 *result, const mat4x4 *m) {
    float s0 = m->m[0][0] * m->m[1][1] - m->m[1][0] * m->m[0][1];
    float s1 = m->m[0][0] * m->m[1][2] - m->m[1][0] * m->m[0][2];
    float s2 = m->m[0][0] * m->m[1][3] - m->m[1][0] * m->m[0][3];
    float s3 = m->m[0][1] * m->m[1][2] - m->m[1][1] * m->m[0][2];
    float s4 = m->m[0][1] * m->m[1][3] - m->m[1][1] * m->m[0][3];
    float s5 = m->m[0][2] * m->m[1][3] - m->m[1][2] * m->m[0][3];

    float c5 = m->m[2][2] * m->m[3][3] - m->m[3][2] * m->m[2][3];
    float c4 = m->m[2][1] * m->m[3][3] - m->m[3][1] * m->m[2][3];
    float c3 = m->m[2][1] * m->m[3][2] - m->m[3][1] * m->m[2][2];
    float c2 = m->m[2][0] * m->m[3][3] - m->m[3][0] * m->m[2][3];
    float c1 = m->m[2][0] * m->m[3][2] - m->m[3][0] * m->m[2][2];
    float c0 = m->m[2][0] * m->m[3][1] - m->m[3][0] * m->m[2][1];

    float det = (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);
    if (det == 0)
        return 0; // Matrix is not invertible
    float invdet = 1.0f / det;

    result->m[0][0] = ( m->m[1][1] * c5 - m->m[1][2] * c4 + m->m[1][3] * c3) * invdet;
    result->m[0][1] = (-m->m[0][1] * c5 + m->m[0][2] * c4 - m->m[0][3] * c3) * invdet;
    result->m[0][2] = ( m->m[3][1] * s5 - m->m[3][2] * s4 + m->m[3][3] * s3) * invdet;
    result->m[0][3] = (-m->m[2][1] * s5 + m->m[2][2] * s4 - m->m[2][3] * s3) * invdet;

    result->m[1][0] = (-m->m[1][0] * c5 + m->m[1][2] * c2 - m->m[1][3] * c1) * invdet;
    result->m[1][1] = ( m->m[0][0] * c5 - m->m[0][2] * c2 + m->m[0][3] * c1) * invdet;
    result->m[1][2] = (-m->m[3][0] * s5 + m->m[3][2] * s2 - m->m[3][3] * s1) * invdet;
    result->m[1][3] = ( m->m[2][0] * s5 - m->m[2][2] * s2 + m->m[2][3] * s1) * invdet;

    result->m[2][0] = ( m->m[1][0] * c4 - m->m[1][1] * c2 + m->m[1][3] * c0) * invdet;
    result->m[2][1] = (-m->m[0][0] * c4 + m->m[0][1] * c2 - m->m[0][3] * c0) * invdet;
    result->m[2][2] = ( m->m[3][0] * s4 - m->m[3][1] * s2 + m->m[3][3] * s0) * invdet;
    result->m[2][3] = (-m->m[2][0] * s4 + m->m[2][1] * s2 - m->m[2][3] * s0) * invdet;

    result->m[3][0] = (-m->m[1][0] * c3 + m->m[1][1] * c1 - m->m[1][2] * c0) * invdet;
    result->m[3][1] = ( m->m[0][0] * c3 - m->m[0][1] * c1 + m->m[0][2] * c0) * invdet;
    result->m[3][2] = (-m->m[3][0] * s3 + m->m[3][1] * s1 - m->m[3][2] * s0) * invdet;
    result->m[3][3] = ( m->m[2][0] * s3 - m->m[2][1] * s1 + m->m[2][2] * s0) * invdet;
    return 1; // Inversion successful
}

static inline vec3 mat4x4_mul_vec3(mat4x4 m, vec3 v)
{
  vec3 res;
  res.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3];
  res.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3];
  res.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3];
  return res;
}
// Function to calculate centroid
static inline void calculate_centroid(triangle *tri)
{
  tri->centroid.x = (tri->vertices[0].x + tri->vertices[1].x +
                     tri->vertices[2].x) /
                    3.0f;
  tri->centroid.y = (tri->vertices[0].y +
                     tri->vertices[1].y + tri->vertices[2].y) /
                    3.0f;
  tri->centroid.z =
      (tri->vertices[0].z + tri->vertices[1].z + tri->vertices[2].z) / 3.0f;
}
void calculate_plane_from_matrix(mat4x4 *matrix, Plane *plane) {
  // Assuming the matrix is in column-major order
  plane->_normal.x = matrix->m[0][2];
  plane->_normal.y = matrix->m[1][2];
  plane->_normal.z = matrix->m[2][2];

  // Normalize the normal vector
  float length = sqrt(plane->_normal.x * plane->_normal.x +
                      plane->_normal.y * plane->_normal.y +
                      plane->_normal.z * plane->_normal.z);
  
  plane->_normal.x /= length;
  plane->_normal.y /= length;
  plane->_normal.z /= length;

  // Extract the d constant
  plane->_d = matrix->m[3][2];
}

// apply_transformation_to_mesh function with macro optimization
static inline void apply_transformation_to_mesh(const mat4x4 *transform,
                                                Mesh *mesh)
{

  for (uint32_t i = 0; i < mesh->triangle_count; i++)
  {
    triangle *tr = &mesh->transformed_triangles[i];
    triangle *orig_tr = &mesh->original_triangles[i];

    for (int j = 0; j < 3; j++)
    {
      vec3 *vertex = &orig_tr->vertices[j];
      vec3 *result = &tr->vertices[j];
      *result = mat4x4_mul_vec3(*transform, *vertex);
      // TRANSFORM_VERTEX(*vertex, *result, transform->m);
    }
    calculate_centroid(tr);
    bvhmesh_calculate_triangle_aabb(tr);
    tr->edge1 = vec3_sub(tr->vertices[1], tr->vertices[0]);
    tr->edge2 = vec3_sub(tr->vertices[2], tr->vertices[0]);
  }
}
mat4x4 create_rotation_matrix(const char order[], float R, float P, float Y)
{
    mat4x4 rotation = {{{1, 0, 0, 0},
                        {0, 1, 0, 0},
                        {0, 0, 1, 0},
                        {0, 0, 0, 1}}};
    // Compute sine and cosine of the Euler angles
    float cosR = cos(R), sinR = sin(R);
    float cosP = cos(P), sinP = sin(P);
    float cosY = cos(Y), sinY = sin(Y);
    // Roll (R) matrix
    mat4x4 roll = {{{1, 0, 0, 0},
                    {0, cosR, -sinR, 0},
                    {0, sinR, cosR, 0},
                    {0, 0, 0, 1}}};
    // Pitch (P) matrix
    mat4x4 pitch = {{{cosP, 0, sinP, 0},
                     {0, 1, 0, 0},
                     {-sinP, 0, cosP, 0},
                     {0, 0, 0, 1}}};
    // Yaw (Y) matrix
    mat4x4 yaw = {{{cosY, -sinY, 0, 0},
                   {sinY, cosY, 0, 0},
                   {0, 0, 1, 0},
                   {0, 0, 0, 1}}};
    // Apply rotations in the specified order
    for (int i = 0; order[i] != '\0'; i++)
    {
        switch (order[i])
        {
        case 'X':
            rotation = mat4x4_mul(&rotation, &pitch);
            break;
        case 'Y':
            rotation = mat4x4_mul(&rotation, &yaw);
            break;
        case 'Z':
            rotation = mat4x4_mul(&rotation, &roll);
            break;
        }
    }
    return rotation;
}

// Create transformation matrix with translation and rotation
mat4x4 create_transformation_matrix(float x, float y, float z, float R, float P, float Y, const char *order)
{
    mat4x4 mat;
    mat4x4 rotation = create_rotation_matrix(order, R, P, Y);
    // Combine translation and rotation into the final matrix
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            mat.m[i][j] = rotation.m[i][j];
        }
    }
    mat.m[0][3] = x;
    mat.m[1][3] = y;
    mat.m[2][3] = z;
    mat.m[3][0] = 0.0f;
    mat.m[3][1] = 0.0f;
    mat.m[3][2] = 0.0f;
    mat.m[3][3] = 1.0f;
    return mat;
}
// SINGLE RAY_OBJECT INTERSECTION
uint8_t ray_sphere(ray *r, scene_object *obj, float *t)
{
  float half_b = obj->_pos_neg.x * r->dir.x + obj->_pos_neg.y * r->dir.y +
                 obj->_pos_neg.z * r->dir.z;
  float discriminant = half_b * half_b - obj->sphere._k;
  if (discriminant < 0)
    return 0; // no intersection
  float sqrt_discriminant = sqrt(discriminant);
  // Find the nearest t (t1 or t2) that is positive
  float root = -half_b - sqrt_discriminant;
  if (root < 0)
  {
    root = -half_b + sqrt_discriminant;
    if (root < 0)
      return 0; // Both roots are negative, no intersection
  }
  *t = root;
  return 1;
}

uint8_t ray_obb(ray *r, scene_object *obb, float *t)
{
  float t_min = -INFINITY;
  float t_max = INFINITY;

  float obb_pos[3] = {obb->_pos.x, obb->_pos.y, obb->_pos.z};
  float ray_orig[3] = {r->ori.x, r->ori.y, r->ori.z};
  float ray_dir[3] = {r->dir.x, r->dir.y, r->dir.z};
  float obb_size[3] = {obb->box.size.x, obb->box.size.y, obb->box.size.z};

  for (int i = 0; i < 3; i++)
  {
    float *n = &obb->box.normals[i].x;
    float e = n[0] * (obb_pos[0] - ray_orig[0]) +
              n[1] * (obb_pos[1] - ray_orig[1]) +
              n[2] * (obb_pos[2] - ray_orig[2]);
    float f = n[0] * ray_dir[0] + n[1] * ray_dir[1] + n[2] * ray_dir[2];

    if (f > 1e-6f || f < -1e-6f)
    {
      float t1 = (e + obb_size[i]) / f;
      float t2 = (e - obb_size[i]) / f;

      if (t1 > t2)
      {
        float temp = t1;
        t1 = t2;
        t2 = temp;
      }

      if (t1 > t_min)
        t_min = t1;
      if (t2 < t_max)
        t_max = t2;

      if (t_min > t_max)
        return 0;
      if (t_max < 0)
        return 0;
    }
    else if (-e - obb_size[i] > 0 || -e + obb_size[i] < 0)
    {
      return 0;
    }
  }

  *t = t_min > 0 ? t_min : t_max;
  return 1;
}
uint8_t ray_plane(ray *r, scene_object *obj, float *t) {
  float denom = obj->plane._normal.x * r->dir.x + obj->plane._normal.y * r->dir.y + obj->plane._normal.z * r->dir.z;
  if (denom > 1e-6 || denom < -1e-6) {
    *t = (obj->_pos.x * obj->plane._normal.x + obj->_pos.y * obj->plane._normal.y + obj->_pos.z * obj->plane._normal.z) / denom;
    if (*t >= 0) {
      float dx = r->dir.x * (*t) - obj->_pos.x;
      float dy = r->dir.y * (*t) - obj->_pos.y;
      if (dx >= -obj->plane.x_size / 2 && dx <= obj->plane.x_size / 2 &&
          dy >= -obj->plane.y_size / 2 && dy <= obj->plane.y_size / 2) 
        return 1; // Intersection within bounds
      
  }
  }
  return 0; // No intersection
}

uint8_t ray_aabb(ray *r, AABB *box, float *t)
{
  float tmin = box->min.x * r->inv_dir.x;
  float tmax = box->max.x * r->inv_dir.x;

  if (tmin > tmax)
  {
    float temp = tmin;
    tmin = tmax;
    tmax = temp;
  }

  float tymin = box->min.y * r->inv_dir.y;
  float tymax = box->max.y * r->inv_dir.y;

  if (tymin > tymax)
  {
    float temp = tymin;
    tymin = tymax;
    tymax = temp;
  }

  if ((tmin > tymax) || (tymin > tmax))
    return 0;

  if (tymin > tmin)
    tmin = tymin;

  if (tymax < tmax)
    tmax = tymax;

  float tzmin = box->min.z * r->inv_dir.z;
  float tzmax = box->max.z * r->inv_dir.z;

  if (tzmin > tzmax)
  {
    float temp = tzmin;
    tzmin = tzmax;
    tzmax = temp;
  }

  if ((tmin > tzmax) || (tzmin > tmax))
    return 0;
  *t = tmin;
  return 1;
}

uint8_t ray_cylinder(ray *r, scene_object *obj, float *t)
{
  // Transform the ray into the local space of the cylinder
  vec3 local_ori = mat4x4_mul_vec3(obj->_inverse, r->ori);
  vec3 local_dir = mat4x4_mul_vec3(obj->_inverse, r->dir);

  // Calculate quadratic equation components for intersection with infinite
  // cylinder
  float a = local_dir.x * local_dir.x + local_dir.z * local_dir.z;
  float b = 2.0f * (local_dir.x * local_ori.x + local_dir.z * local_ori.z);
  float c = local_ori.x * local_ori.x + local_ori.z * local_ori.z -
            obj->cylinder.radius * obj->cylinder.radius;

  float discriminant = b * b - 4 * a * c;
  if (discriminant < 0)
  {
    return 0;
  }

  float sqrt_discriminant = sqrtf(discriminant);
  float t0 = (-b - sqrt_discriminant) / (2.0f * a);
  float t1 = (-b + sqrt_discriminant) / (2.0f * a);

  if (t0 > t1)
  {
    float temp = t0;
    t0 = t1;
    t1 = temp;
  }

  // Check intersection points with the finite height of the cylinder
  float y0 = local_ori.y + t0 * local_dir.y;
  if (y0 >= 0 && y0 <= obj->cylinder.height)
  {
    *t = t0;
    return 1;
  }

  float y1 = local_ori.y + t1 * local_dir.y;
  if (y1 >= 0 && y1 <= obj->cylinder.height)
  {
    *t = t1;
    return 1;
  }

  // Check intersection with cylinder caps
  if (local_dir.y != 0)
  {
    float t_cap_bottom = -local_ori.y / local_dir.y;
    vec3 p_bottom = vec3_add(local_ori, vec3_scalar(local_dir, t_cap_bottom));
    if (p_bottom.x * p_bottom.x + p_bottom.z * p_bottom.z <=
        obj->cylinder.radius * obj->cylinder.radius)
    {
      *t = t_cap_bottom;
      return 1;
    }

    float t_cap_top = (obj->cylinder.height - local_ori.y) / local_dir.y;
    vec3 p_top = vec3_add(local_ori, vec3_scalar(local_dir, t_cap_top));
    if (p_top.x * p_top.x + p_top.z * p_top.z <=
        obj->cylinder.radius * obj->cylinder.radius)
    {
      *t = t_cap_top;
      return 1;
    }
  }

  return 0;
}

uint8_t ray_triangle(const ray *r, const triangle *tri, float *t)
{
  vec3 v0v1 = tri->edge1;
  vec3 v0v2 = tri->edge2;
  vec3 pvec = {r->dir.y * v0v2.z - r->dir.z * v0v2.y,
               r->dir.z * v0v2.x - r->dir.x * v0v2.z,
               r->dir.x * v0v2.y - r->dir.y * v0v2.x};
  float det = v0v1.x * pvec.x + v0v1.y * pvec.y + v0v1.z * pvec.z;

  // If the determinant is near zero, the ray lies in the plane of the triangle
  if ((det < 0 ? -det : det) < 1e-8)
    return 0;

  float invDet = 1 / det;

  float u = (-tri->vertices[0].x * pvec.x - tri->vertices[0].y * pvec.y - tri->vertices[0].z * pvec.z) * invDet;
  if (u < 0 || u > 1)
    return 0;

  vec3 qvec = {-tri->vertices[0].y * v0v1.z + tri->vertices[0].z * v0v1.y,
               -tri->vertices[0].z * v0v1.x + tri->vertices[0].x * v0v1.z,
               -tri->vertices[0].x * v0v1.y + tri->vertices[0].y * v0v1.x};
  float v =
      (r->dir.x * qvec.x + r->dir.y * qvec.y + r->dir.z * qvec.z) * invDet;
  if (v < 0 || u + v > 1)
    return 0;

  *t = (v0v2.x * qvec.x + v0v2.y * qvec.y + v0v2.z * qvec.z) * invDet;
  if (*t < 0)
    return 0;
  return 1;
}

// SCENE FUNCTIONS
void scene_init(scene *scn)
{
  scn->objects = NULL;
  scn->current_size = 0;
  scn->max_size = 0;
}

void scene_add_object(scene *scn, scene_object new_object)
{
  if (scn->current_size == scn->max_size)
  {
    if (scn->max_size == 0)
    {
      scn->max_size = 1; // Start with size 1
    }
    else
    {
      scn->max_size *= 2;
    }
    scn->objects = (scene_object *)realloc(
        scn->objects, scn->max_size * sizeof(scene_object));
    if (scn->objects == NULL)
    {
      // Handle reallocation error
      exit(1);
    }
  }
  scn->objects[scn->current_size++] = new_object;
}
void scene_update(scene *scn, lidar lidar_data)
{
  for (int i = 0; i < scn->current_size; i++)
  {
    scene_object *obj = &scn->objects[i];
    
    mat4x4 inv;
    if (mat4x4_invert(&inv, &lidar_data.transform)) {
        obj->transform = mat4x4_mul(&inv, &obj->transform);
    }
    else printf("cannot inverse matrix transformation for %d object", i);


    // Process based on object type
    switch (obj->type)
    {
    case MESH:
      apply_transformation_to_mesh(&obj->transform, &obj->mesh);
      obj->mesh.root = bvhmesh_build_tree(obj->mesh.transformed_triangles, 0,
                                          obj->mesh.triangle_count - 1, 0);
      break;

    case SPHERE:
      obj->_pos = cast_vec3(obj->transform.m[0][3], obj->transform.m[1][3],
                         obj->transform.m[2][3]);
      obj->sphere._k = vec3_dot(obj->_pos, obj->_pos) -
                       obj->sphere.radius * obj->sphere.radius;
      obj->_pos_neg = vec3_negate(obj->_pos);
      break;

    case BOX:
      obj->_pos = cast_vec3(obj->transform.m[0][3], obj->transform.m[1][3],
                         obj->transform.m[2][3]);
      obj->box.normals[0] =
          cast_vec3(obj->transform.m[0][0], obj->transform.m[0][1],
                 obj->transform.m[0][2]);
      obj->box.normals[1] =
          cast_vec3(obj->transform.m[1][0], obj->transform.m[1][1],
                 obj->transform.m[1][2]);
      obj->box.normals[2] =
          cast_vec3(obj->transform.m[2][0], obj->transform.m[2][1],
                 obj->transform.m[2][2]);
      break;

    case CYLINDER:
      obj->_pos = cast_vec3(obj->transform.m[0][3], obj->transform.m[1][3],
                         obj->transform.m[2][3]);
      if(!mat4x4_invert(&obj->_inverse, &obj->transform))
        printf("cannot invert matrix");
      break;
    
    case PLANE:

      calculate_plane_from_matrix(&obj->transform, &obj->plane);
      obj->_pos = cast_vec3(obj->transform.m[0][3], obj->transform.m[1][3],
                         obj->transform.m[2][3]);
      break;
      

    default:
      obj->_pos = cast_vec3(obj->transform.m[0][3], obj->transform.m[1][3],
                         obj->transform.m[2][3]);
      break;
    }

    // Calculate AABB for each object
    obj->aabb = bvh_calculate_aabb(obj);
  }
}

void scene_build(scene *scn)
{
  scene_object **object_ptrs = (scene_object **)malloc(scn->current_size * sizeof(scene_object *));
  if (object_ptrs == NULL)
  {
    fprintf(stderr, "Failed to allocate memory for object pointers\n");
    return;
  }
  for (int i = 0; i < scn->current_size; i++)
  {
    object_ptrs[i] = &scn->objects[i];
  }
  // Now call bvh_build_tree with the array of pointers
  scn->root = bvh_build_tree(object_ptrs, 0, scn->current_size - 1);
  free(object_ptrs);
}
void scene_free(scene *scn)
{
  for (int i = 0; i < scn->current_size; i++)
  {
    if (scn->objects[i].type == MESH)
    {
      free(scn->objects[i].mesh.original_triangles);
      free(scn->objects[i].mesh.transformed_triangles);
      bvhmesh_free(scn->objects[i].mesh.root);
    }
  }
  bvh_free_tree(scn->root);
  free(scn->objects);
  scn->objects = NULL;
  scn->current_size = 0;
  scn->max_size = 0;
}

int bvh_comparator(const void *a, const void *b, void *arg)
{
  int axis = *(int *)arg;
  const scene_object *objA = *(const scene_object **)a;
  const scene_object *objB = *(const scene_object **)b;

  float minA, minB;
  switch (axis)
  {
  case 0: // X-axis
    minA = objA->aabb.min.x;
    minB = objB->aabb.min.x;
    break;
  case 1: // Y-axis
    minA = objA->aabb.min.y;
    minB = objB->aabb.min.y;
    break;
  case 2: // Z-axis
    minA = objA->aabb.min.z;
    minB = objB->aabb.min.z;
    break;
  default:
    return 0; // Should not happen
  }

  return (minA < minB) ? -1 : (minA > minB) ? 1
                                            : 0;
}

static inline void bvh_unionAABB(const AABB *a, const AABB *b, AABB *result)
{
  result->min.x = (a->min.x < b->min.x) ? a->min.x : b->min.x; // more efficient than using fmin
  result->min.y = (a->min.y < b->min.y) ? a->min.y : b->min.y;
  result->min.z = (a->min.z < b->min.z) ? a->min.z : b->min.z;
  result->max.x = (a->max.x > b->max.x) ? a->max.x : b->max.x;
  result->max.y = (a->max.y > b->max.y) ? a->max.y : b->max.y;
  result->max.z = (a->max.z > b->max.z) ? a->max.z : b->max.z;
}

bvh_node *bvh_build_tree(scene_object **objects, int start, int end)
{
  if (start > end)
    return NULL;

  bvh_node *node = (bvh_node *)malloc(sizeof(bvh_node));
  if (!node)
  {
    fprintf(stderr, "Memory allocation failed\n");
    return NULL;
  }

  if (start == end)
  {
    // For a single object, directly use its AABB.
    node->box = objects[start]->aabb;
    node->index = start; // Set to the index of the object
    node->left = NULL;
    node->right = NULL;
  }
  else
  {
    // Choose the axis to split along (e.g., x-axis here)
    int axis = rand() % 3; // Randomly choose axis for better balance

    // Sort the objects based on their AABBs along the chosen axis
    // qsort(objects + start, end - start + 1, sizeof(scene_object *),
    // bvh_comparator) ;
    /*qsort_r(objects + start, end - start + 1, sizeof(scene_object *),
            (int (*)(const void *, const void *, void *))bvh_comparator,
            (void *)&axis);*/
    qsort_r(objects + start, end - start + 1, sizeof(scene_object *), bvh_comparator, (void *)&axis);

    // Calculate the midpoint
    int mid = start + (end - start) / 2;

    // Recursively build subtrees
    node->left = bvh_build_tree(objects, start, mid);
    node->right = bvh_build_tree(objects, mid + 1, end);

    // Calculate node's AABB as the union of its children's AABBs
    if (node->left && node->right)
    {
      bvh_unionAABB(&node->left->box, &node->right->box, &node->box);
    }
    else if (node->left)
    {
      node->box = node->left->box;
    }
    else if (node->right)
    {
      node->box = node->right->box;
    }
    node->index = -1; // This is an internal node
  }

  return node;
}

void bvh_free_tree(bvh_node *node)
{
  if (node == NULL)
  {
    return; // Base case: if the node is NULL, just return
  }

  bvh_free_tree(node->left);  // Recursively free the left subtree
  bvh_free_tree(node->right); // Recursively free the right subtree

  free(node); // Free the current node after its children have been freed
}

AABB bvh_calculate_aabb(scene_object *obj)
{
  AABB aabb;
  switch (obj->type)
  {
  case SPHERE:
   {
    aabb.min.x = obj->_pos.x - obj->sphere.radius;
    aabb.min.y = obj->_pos.y - obj->sphere.radius;
    aabb.min.z = obj->_pos.z - obj->sphere.radius;
    aabb.max.x = obj->_pos.x + obj->sphere.radius;
    aabb.max.y = obj->_pos.y + obj->sphere.radius;
    aabb.max.z = obj->_pos.z + obj->sphere.radius;
    break;
   }

  case CYLINDER:
   {
    aabb.min.x = obj->_pos.x - obj->cylinder.radius;
    aabb.min.z = obj->_pos.z - obj->cylinder.height / 2.0;
    aabb.min.y = obj->_pos.y - obj->cylinder.radius;
    aabb.max.x = obj->_pos.x + obj->cylinder.radius;
    aabb.max.z = obj->_pos.z + obj->cylinder.height / 2.0;
    aabb.max.y = obj->_pos.y + obj->cylinder.radius;
    break;
  }
  case BOX:
   {
    vec3 rot_matrix[3] = {{obj->transform.m[0][0], obj->transform.m[0][1],
                           obj->transform.m[0][2]},
                          {obj->transform.m[1][0], obj->transform.m[1][1],
                           obj->transform.m[1][2]},
                          {obj->transform.m[2][0], obj->transform.m[2][1],
                           obj->transform.m[2][2]}};

    vec3 min = obj->_pos;
    vec3 max = obj->_pos;

    float extents[3] = {fabs(rot_matrix[0].x) * obj->box.size.x +
                            fabs(rot_matrix[1].x) * obj->box.size.y +
                            fabs(rot_matrix[2].x) * obj->box.size.z,
                        fabs(rot_matrix[0].y) * obj->box.size.x +
                            fabs(rot_matrix[1].y) * obj->box.size.y +
                            fabs(rot_matrix[2].y) * obj->box.size.z,
                        fabs(rot_matrix[0].z) * obj->box.size.x +
                            fabs(rot_matrix[1].z) * obj->box.size.y +
                            fabs(rot_matrix[2].z) * obj->box.size.z};

    min.x -= extents[0];
    max.x += extents[0];
    min.y -= extents[1];
    max.y += extents[1];
    min.z -= extents[2];
    max.z += extents[2];

    aabb.min = min;
    aabb.max = max;
    break;
  }
  case PLANE:
  {
    vec3 min = obj->_pos;
    vec3 max = obj->_pos;
    vec3 corners[4];
    float halfX = obj->plane.x_size / 2.0;
    float halfY = obj->plane.y_size / 2.0;

    // Define corners in local plane space
    corners[0] = cast_vec3(-halfX, halfY, 0);
    corners[1] = cast_vec3(halfX, halfY, 0);
    corners[2] = cast_vec3(-halfX, -halfY, 0);
    corners[3] = cast_vec3(halfX, -halfY, 0);

    // Transform corners to world space
    for (int i = 0; i < 4; i++) {
        corners[i] = mat4x4_mul_vec3(obj->transform, corners[i]);
    }

    // Compute AABB from transformed corners
    aabb.min = corners[0];
    aabb.max = corners[0];
    for (int i = 1; i < 4; i++) {
        if (corners[i].x < min.x) aabb.min.x = corners[i].x;
        if (corners[i].y < min.y) aabb.min.y = corners[i].y;
        if (corners[i].z < min.z) aabb.min.z = corners[i].z;
        if (corners[i].x > max.x) aabb.max.x = corners[i].x;
        if (corners[i].y > max.y) aabb.max.y = corners[i].y;
        if (corners[i].z > max.z) aabb.max.z = corners[i].z;
    }
    aabb.max.z = 0;
    break;
    }
  case MESH:
  {
    aabb.min = cast_vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    aabb.max = cast_vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (uint32_t i = 0; i < obj->mesh.triangle_count; i++)
    {
      triangle tr = obj->mesh.transformed_triangles[i];
      for (int j = 0; j < 3; j++)
      {
        vec3 vertex = tr.vertices[j];

        // Update the min and max values
        if (vertex.x < aabb.min.x)
          aabb.min.x = vertex.x;
        if (vertex.y < aabb.min.y)
          aabb.min.y = vertex.y;
        if (vertex.z < aabb.min.z)
          aabb.min.z = vertex.z;

        if (vertex.x > aabb.max.x)
          aabb.max.x = vertex.x;
        if (vertex.y > aabb.max.y)
          aabb.max.y = vertex.y;
        if (vertex.z > aabb.max.z)
          aabb.max.z = vertex.z;
      }
    }
    break;
  }
  }
  return aabb;
}

void bvh_update_nearest_intersection(ray_result *closest, float t,
                                     scene_object *obj)
{
  if (t < closest->t && t >= 0)
  {
    closest->t = t;
    closest->obj = obj;
  }
}

void check_and_update_intersection(ray *r, scene_object *obj,
                                   ray_result *closest)
{
  float t_hit = FLT_MAX;
  uint8_t hit = 0;

  switch (obj->type)
  {
  case MESH:
  {
    ray_triangle_result closest_tri;
    closest_tri.t = FLT_MAX;
    closest_tri.tri = NULL;
    bvhmesh_traverse_ray_aabb(obj->mesh.root, r, &closest_tri);
    if (closest_tri.t < closest->t && closest_tri.t > 0)
    {
      bvh_update_nearest_intersection(closest, closest_tri.t, obj);
    }
    break;
  }
  case SPHERE:
    hit = ray_sphere(r, obj, &t_hit);
    break;
  case CYLINDER:
    hit = ray_cylinder(r, obj, &t_hit);
    break;
  case BOX:
    hit = ray_obb(r, obj, &t_hit);
    break;
  case PLANE:
    hit = ray_plane(r, obj, &t_hit);
    break;
  }

  if (hit && t_hit < closest->t && t_hit > 0)
  {
    bvh_update_nearest_intersection(closest, t_hit, obj);
  }
}

void bvh_traverse_ray_aabb(bvh_node *node, ray *r, scene *scn,
                           ray_result *closest)
{
  if (node == NULL)
  {
    return;
  }

  float t;
  if (!ray_aabb(r, &node->box, &t))
  {
    return;
  }

  if (node->left == NULL && node->right == NULL)
  {
    scene_object *obj = &scn->objects[node->index];
    check_and_update_intersection(r, obj, closest);
  }
  else
  {
    bvh_traverse_ray_aabb(node->left, r, scn, closest);
    bvh_traverse_ray_aabb(node->right, r, scn, closest);
  }
}


static inline void bvhmesh_calculate_triangle_aabb(triangle *tri)
{
  float min_x, min_y, min_z, max_x, max_y, max_z;
  min_x = max_x = tri->vertices[0].x;
  min_y = max_y = tri->vertices[0].y;
  min_z = max_z = tri->vertices[0].z;

  float x1 = tri->vertices[1].x;
  float y1 = tri->vertices[1].y;
  float z1 = tri->vertices[1].z;
  float x2 = tri->vertices[2].x;
  float y2 = tri->vertices[2].y;
  float z2 = tri->vertices[2].z;

  if (x1 < min_x)
    min_x = x1;
  if (y1 < min_y)
    min_y = y1;
  if (z1 < min_z)
    min_z = z1;
  if (x1 > max_x)
    max_x = x1;
  if (y1 > max_y)
    max_y = y1;
  if (z1 > max_z)
    max_z = z1;

  if (x2 < min_x)
    min_x = x2;
  if (y2 < min_y)
    min_y = y2;
  if (z2 < min_z)
    min_z = z2;
  if (x2 > max_x)
    max_x = x2;
  if (y2 > max_y)
    max_y = y2;
  if (z2 > max_z)
    max_z = z2;

  tri->aabb.min.x = min_x;
  tri->aabb.min.y = min_y;
  tri->aabb.min.z = min_z;
  tri->aabb.max.x = max_x;
  tri->aabb.max.y = max_y;
  tri->aabb.max.z = max_z;
}

// Comparator function for qsort
int compare_x(const void *a, const void *b) {
    const triangle *t1 = (const triangle *)a;
    const triangle *t2 = (const triangle *)b;
    return (t1->centroid.x < t2->centroid.x) ? -1 : (t1->centroid.x > t2->centroid.x) ? 1 : 0;
}

int compare_y(const void *a, const void *b) {
    const triangle *t1 = (const triangle *)a;
    const triangle *t2 = (const triangle *)b;
    return (t1->centroid.y < t2->centroid.y) ? -1 : (t1->centroid.y > t2->centroid.y) ? 1 : 0;
}

int compare_z(const void *a, const void *b) {
    const triangle *t1 = (const triangle *)a;
    const triangle *t2 = (const triangle *)b;
    return (t1->centroid.z < t2->centroid.z) ? -1 : (t1->centroid.z > t2->centroid.z) ? 1 : 0;
}

// Partition function for quickselect
int partition(triangle *triangles, int low, int high, int axis, int (*comparator)(const void *, const void *)) {
    triangle pivot = triangles[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (comparator(&triangles[j], &pivot) <= 0) {
            i++;
            triangle temp = triangles[i];
            triangles[i] = triangles[j];
            triangles[j] = temp;
        }
    }
    triangle temp = triangles[i + 1];
    triangles[i + 1] = triangles[high];
    triangles[high] = temp;
    return i + 1;
}

// Quickselect algorithm to find the median
void quickselect(triangle *triangles, int low, int high, int k, int axis, int (*comparator)(const void *, const void *)) {
    if (low < high) {
        int pivotIndex = partition(triangles, low, high, axis, comparator);
        if (pivotIndex == k) return;
        else if (pivotIndex > k) quickselect(triangles, low, pivotIndex - 1, k, axis, comparator);
        else quickselect(triangles, pivotIndex + 1, high, k, axis, comparator);
    }
}

// Function to build the BVH tree
bvhmesh_node *bvhmesh_build_tree(triangle *triangles, int start, int end, int depth) {
    if (start > end) return NULL;

    bvhmesh_node *node = (bvhmesh_node *)malloc(sizeof(bvhmesh_node));
    if (!node) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }
    node->tri = NULL;

    if (start == end) {
        node->tri = &triangles[start];
        node->box = node->tri->aabb;
        node->left = NULL;
        node->right = NULL;
    } else {
        int axis = depth % 3;  // Cycle through axes: X, Y, Z
        if (axis == 0)
            quickselect(triangles, start, end, (start + end) / 2, axis, compare_x);
        else if (axis == 1)
            quickselect(triangles, start, end, (start + end) / 2, axis, compare_y);
        else
            quickselect(triangles, start, end, (start + end) / 2, axis, compare_z);

        int mid = start + (end - start) / 2;

        node->left = bvhmesh_build_tree(triangles, start, mid, depth + 1);
        node->right = bvhmesh_build_tree(triangles, mid + 1, end, depth + 1);

        if (node->left && node->right) {
            bvh_unionAABB(&node->left->box, &node->right->box, &node->box);
        } else if (node->left) {
            node->box = node->left->box;
        } else if (node->right) {
            node->box = node->right->box;
        }
    }

    return node;
}

void bvhmesh_free(bvhmesh_node *node)
{
  if (node == NULL)
  {
    return; // Base case: if the node is NULL, just return
  }

  bvhmesh_free(node->left);  // Recursively free the left subtree
  bvhmesh_free(node->right); // Recursively free the right subtree

  free(node); // Free the current node after its children have been freed
}

void bvhmesh_update_nearest_intersection(ray_triangle_result *closest, float t,
                                         triangle *tri)
{
  if (t < closest->t && t >= 0)
  {
    closest->t = t;
    closest->tri = tri;
  }
}

void bvhmesh_traverse_ray_aabb(bvhmesh_node *node, ray *r,
                               ray_triangle_result *closest)
{
  if (node == NULL)
  {
    return;
  }

  float t;
  if (!ray_aabb(r, &node->box, &t) || t >= closest->t)
  {
    return; // No intersection or further than the closest found
  }

  if (node->tri)
  {
    float t_hit;
    if (ray_triangle(r, node->tri, &t_hit) && t_hit < closest->t &&
        t_hit > 0)
    {
      bvhmesh_update_nearest_intersection(closest, t_hit, node->tri);
    }
  }
  else
  {
    bvhmesh_traverse_ray_aabb(node->left, r, closest);
    bvhmesh_traverse_ray_aabb(node->right, r, closest);
  }
}

/* __UTILS FUNCTIONS__ */

uint8_t create_lidar_from_file(const char *filename, lidar *lidar_data)
{
  FILE *file = fopen(filename, "r");
  if (!file)
  {
    perror("Failed to open file");
    return 0;
  }

  char buffer[1024];
  if (!fgets(buffer, sizeof(buffer), file))
  { // Skip header
    fclose(file);
    return 0;
  }

  lidar_data->ray_count = 0;
  lidar_data->rays = NULL;
  ray *temp_rays = NULL;
  uint32_t count = 0;

  while (fgets(buffer, sizeof(buffer), file))
  {
    temp_rays = (ray *)realloc(lidar_data->rays, sizeof(ray) * (count + 1));
    if (!temp_rays)
    {
      perror("Failed to allocate memory");
      free(lidar_data->rays);
      fclose(file);
      return -1;
    }
    lidar_data->rays = temp_rays;

    char *dir_x_str = strtok(buffer, ",");
    char *dir_y_str = strtok(NULL, ",");
    char *dir_z_str = strtok(NULL, "\n");

    vec3 dir = {(float)atof(dir_x_str), (float)atof(dir_y_str), (float)atof(dir_z_str)};
    vec3 inv_dir = {1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z};

    lidar_data->rays[count].ori = cast_vec3(0.0f, 0.0f, 0.0f);
    lidar_data->rays[count].dir = dir;
    lidar_data->rays[count].inv_dir = inv_dir;

    count++;
  }
  mat4x4_identity(&lidar_data->transform);
  lidar_data->ray_count = count;
  fclose(file);
  return 1;
}

uint8_t create_mesh_from_file(const char *filename, scene_object *scene, float scale_x, float scale_y, float scale_z) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    if (strstr(filename, ".stl")) {
        // Skip the 80-byte header
        fseek(file, 80, SEEK_SET);

        // Read the number of triangles
        unsigned int numTriangles = 0;
        if (fread(&numTriangles, sizeof(unsigned int), 1, file) != 1) {
            fclose(file);
            return 0;
        }

        printf("Number of triangles: %u\n", numTriangles);

        // Initialize the scene_object and its mesh
        scene->type = MESH;
        scene->mesh.original_triangles = (triangle *)malloc(numTriangles * sizeof(triangle));
        scene->mesh.transformed_triangles = (triangle *)malloc(numTriangles * sizeof(triangle));
        if (scene->mesh.original_triangles == NULL || scene->mesh.transformed_triangles == NULL) {
            if (scene->mesh.original_triangles)
                free(scene->mesh.original_triangles);
            fclose(file);
            return 0;
        }
        scene->mesh.triangle_count = numTriangles;

        // Read triangles
        for (unsigned int i = 0; i < numTriangles; i++) {
            // Ignore the normal vector (3 floats)
            fseek(file, 3 * sizeof(float), SEEK_CUR);

            // Read vertices
            for (int j = 0; j < 3; j++) {
                if (fread(&scene->mesh.original_triangles[i].vertices[j], sizeof(vec3), 1, file) != 1) {
                    free(scene->mesh.original_triangles);
                    free(scene->mesh.transformed_triangles);
                    fclose(file);
                    return 0;
                }
                // Scale the vertex
                scale_vertex(&scene->mesh.original_triangles[i].vertices[j], scale_x, scale_y, scale_z);
            }

            // Ignore attribute byte count (2 bytes)
            fseek(file, 2, SEEK_CUR);
        }

    } else if (strstr(filename, ".obj")) {
        char line[256];
        unsigned int numVertices = 0, numTriangles = 0;
        vec3 *vertices = NULL;

        // First pass: count vertices and faces
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "v ", 2) == 0) {
                numVertices++;
            } else if (strncmp(line, "f ", 2) == 0) {
                numTriangles++;
            }
        }

        // Allocate memory
        scene->type = MESH;
        vertices = (vec3 *)malloc(numVertices * sizeof(vec3));
        scene->mesh.original_triangles = (triangle *)malloc(numTriangles * sizeof(triangle));
        scene->mesh.transformed_triangles = (triangle *)malloc(numTriangles * sizeof(triangle));
        if (vertices == NULL || scene->mesh.original_triangles == NULL || scene->mesh.transformed_triangles == NULL) {
            free(vertices);
            free(scene->mesh.original_triangles);
            free(scene->mesh.transformed_triangles);
            fclose(file);
            return 0;
        }

        scene->mesh.triangle_count = numTriangles;

        // Second pass: read vertices and faces
        rewind(file);
        unsigned int vertexIndex = 0, triangleIndex = 0;
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "v ", 2) == 0) {
                sscanf(line + 2, "%f %f %f", &vertices[vertexIndex].x, &vertices[vertexIndex].y, &vertices[vertexIndex].z);
                // Scale the vertex
                scale_vertex(&vertices[vertexIndex], scale_x, scale_y, scale_z);
                vertexIndex++;
            } else if (strncmp(line, "f ", 2) == 0) {
                unsigned int vIndex[3];
                int matches = sscanf(line + 2, "%u/%*u/%*u %u/%*u/%*u %u/%*u/%*u", &vIndex[0], &vIndex[1], &vIndex[2]);
                if (matches != 3) {
                    matches = sscanf(line + 2, "%u/%*u %u/%*u %u/%*u", &vIndex[0], &vIndex[1], &vIndex[2]);
                }
                if (matches != 3) {
                    matches = sscanf(line + 2, "%u %u %u", &vIndex[0], &vIndex[1], &vIndex[2]);
                }
                if (matches != 3) {
                    printf("Error reading face at line: %s\n", line);
                    free(vertices);
                    free(scene->mesh.original_triangles);
                    free(scene->mesh.transformed_triangles);
                    fclose(file);
                    return 0;
                }

                for (int j = 0; j < 3; j++) {
                    if (vIndex[j] < 1 || vIndex[j] > numVertices) {
                        printf("Invalid vertex index %u at line: %s\n", vIndex[j], line);
                        free(vertices);
                        free(scene->mesh.original_triangles);
                        free(scene->mesh.transformed_triangles);
                        fclose(file);
                        return 0;
                    }
                    scene->mesh.original_triangles[triangleIndex].vertices[j] = vertices[vIndex[j] - 1];
                }
                triangleIndex++;
            }
        }

        free(vertices);
    } else {
        fclose(file);
        printf("Unsupported file format\n");
        return 0;
    }

    fclose(file);
    return 1; // Success
}

void save_as_pcd(const char *filename, const pointcloud *cloud)
{
  FILE *file = fopen(filename, "w");
  if (file == NULL)
  {
    perror("Error opening file");
    return;
  }

  // Write the header
  fprintf(file, "# .PCD v.7 - Point Cloud Data file format\n");
  fprintf(file, "VERSION .7\n");
  fprintf(file, "FIELDS x y z\n");
  fprintf(file, "SIZE 4 4 4\n");
  fprintf(file, "TYPE F F F\n");
  fprintf(file, "COUNT 1 1 1\n");
  fprintf(file, "WIDTH %d\n", cloud->point_count); // Use %ld for long type
  fprintf(file, "HEIGHT 1\n");
  fprintf(file, "VIEWPOINT 0 0 0 1 0 0 0\n");
  fprintf(file, "POINTS %d\n", cloud->point_count); // Use %ld for long type
  fprintf(file, "DATA ascii\n");

  // Write the point data
  for (uint32_t i = 0; i < cloud->point_count; ++i)
  {
    fprintf(file, "%f %f %f\n", cloud->points[i].x, cloud->points[i].y,
            cloud->points[i].z);
  }

  fclose(file);
}

void pointcloud_free(pointcloud *cloud)
{
  if (cloud != NULL && cloud->points != NULL)
  {
    free(cloud->points);    // Free the memory allocated for points
    cloud->points = NULL;   // Set the pointer to NULL to avoid dangling pointers
    cloud->point_count = 0; // Reset the point count
  }
}
void lidar_free(lidar *lidar_data)
{
  if (lidar_data != NULL && lidar_data->rays != NULL)
  {
    free(lidar_data->rays); // Free the memory allocated for the rays
    lidar_data->rays =
        NULL;                  // Set the rays pointer to NULL to avoid dangling pointers
    lidar_data->ray_count = 0; // Reset the ray count
  }
}

double tic() { return (double)clock(); }
void toc(const char *message, double start)
{
  double end = (double)clock();
  double cpu_time_used =
      ((end - start) / CLOCKS_PER_SEC) * 1000; // Convert to milliseconds
  printf("  -%s took %f ms to execute.\n", message,
         cpu_time_used);
  // printf("\033[31mHello, World!\033[0m\n") ;
}


scene_object create_sphere(float radius) {
  scene_object obj;
  obj.type = SPHERE;
  obj.sphere.radius = radius;
  mat4x4_identity(&obj.transform);
  return obj;
}
scene_object create_plane(float x_size, float y_size) {
  scene_object obj;
  obj.type = PLANE;
  obj.plane.x_size = x_size;
  obj.plane.y_size = y_size;
  mat4x4_identity(&obj.transform);
  return obj;
}

scene_object create_cylinder(float radius, float height) {
  scene_object obj;
  obj.type = CYLINDER;
  obj.cylinder.radius = radius;
  obj.cylinder.height = height;
  mat4x4_identity(&obj.transform);
  return obj;
}

scene_object create_box(float width, float length, float height) {
  scene_object obj;
  obj.type = BOX;
  obj.box.size = cast_vec3(width, length, height);
  mat4x4_identity(&obj.transform);
  return obj;
}

pointcloud cast_rays(lidar lidar_data, scene scn, uint32_t start, uint32_t end) {
  pointcloud cloud;
  uint32_t count = 0;

  // Allocate memory based on the number of rays (this is an upper bound)
  cloud.points = (vec3 *)malloc(sizeof(vec3) * lidar_data.ray_count);
  if (!cloud.points) {
    cloud.point_count = 0;
    return cloud; // return an empty pointcloud if allocation fails
  }

  for (uint32_t i = start; i < end; i++) {
    ray_result closest;
    closest.t = FLT_MAX;
    closest.obj = NULL;
    bvh_traverse_ray_aabb(scn.root, &lidar_data.rays[i], &scn, &closest);

    if (closest.obj != NULL) {
      vec3 intersection = {
          lidar_data.rays[i].ori.x + closest.t * lidar_data.rays[i].dir.x,
          lidar_data.rays[i].ori.y + closest.t * lidar_data.rays[i].dir.y,
          lidar_data.rays[i].ori.z + closest.t * lidar_data.rays[i].dir.z};
      cloud.points[count] = intersection;
      count++;
    }
  }

  cloud.point_count = count;
  // Optionally resize the allocated memory to fit the actual number of points
  cloud.points = (vec3 *)realloc(cloud.points, sizeof(vec3) * count);
  return cloud;
}

#endif // UTILS_H_
