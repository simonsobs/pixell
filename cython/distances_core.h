#include <stdint.h>
#define inum int64_t
void distance_from_points_simple(inum npix, double * posmap, inum npoint, double * points, double * dists, int * areas);
void distance_from_points_simple_separable(inum ny, inum nx, double * ypos, double * xpos, inum npoint, double * points, double * dists, int * areas);
inum find_edges(inum ny, inum nx, uint8_t * mask, inum ** edges);
inum find_edges_labeled(inum ny, inum nx, int * labels, inum ** edges);
void distance_from_points_bubble(int ny, int nx, double * posmap, inum npoint, double * point_pos, int * point_pix, double * dists, int * domains);
void distance_from_points_bubble_separable(int ny, int nx, double * ypos, double * xpos, inum npoint, double * point_pos, int * point_pix, double * dists, int * domains);
