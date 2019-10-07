#include <stdint.h>
#define inum int64_t
void distance_transform(inum ny, inum nx, uint8_t * mask, double * posmap, double * dists);
void distance_from_points(inum npix, double * posmap, inum npoint, double * points, double * dists, int * areas);
void distance_from_points_separable(inum ny, inum nx, double * ypos, double * xpos, inum npoint, double * points, double * dists, int * areas);
inum find_edges(inum ny, inum nx, uint8_t * mask, inum ** edges);
inum find_edges_labeled(inum ny, inum nx, int * labels, inum ** edges);
void distance_from_points_treerings_separable(int ny, int nx, double * ypos, double * xpos, inum npoint, double * point_pos, int * point_y, int * point_x, double * dists, int * domains);
