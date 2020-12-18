cdef extern from "distances_core.h":
	ctypedef long long int inum
	ctypedef unsigned char uint8_t
	void distance_from_points_simple(inum npix, double * posmap, inum npoint, double * points, double * dists, int * area)
	void distance_from_points_simple_separable(inum ny, inum nx, double * ypos, double * xpos, inum npoint, double * points, double * dists, int * area)
	void distance_from_points_bubble(int ny, int nx, double * posmap, inum npoint, double * point_pos, int * point_pix, double rmax, double * dists, int * domains)
	void distance_from_points_bubble_separable(int ny, int nx, double * ypos, double * xpos, inum npoint, double * point_pos, int * point_pix, double rmax, double * dists, int * domains)
	inum find_edges(inum ny, inum nx, uint8_t * mask, inum ** edges)
	inum find_edges_labeled(inum ny, inum nx, int * labels, inum ** edges)

	void distance_from_points_cellgrid(int ny, int nx, double * ypos, double * xpos, inum npoint, double * point_pos, int * point_pix, int bsize_y, int bsize_x, double rmax, double dr, int separable, double * dists, int * domains)

	# Healpix stuff below
	ctypedef struct healpix_info:
		int nside, ny
		inum npix, ncap
		int * nx
		inum * off
		int  * shift
		double *ra0
		double *dec
		double *cos_dec
		double *sin_dec

	healpix_info * build_healpix_info(int nside)
	void free_healpix_info(healpix_info * geo)
	void unravel_healpix(healpix_info * geo, inum npoint, inum * pix1d, int * pix2d)
	void ravel_healpix(healpix_info * geo, inum npoint, int * pix2d, inum * pix1d)
	void get_healpix_neighs(healpix_info * geo, int y, int x, int * oy, int * ox)
	inum find_edges_healpix(healpix_info * geo, uint8_t * mask, int ** edges)
	inum find_edges_labeled_healpix(healpix_info * geo, int * labels, int ** edges)
	void distance_from_points_bubble_healpix(healpix_info * geo, inum npoint, double * point_pos, int * point_pix, double rmax, double * dists, int * domains)
	void distance_from_points_heap_healpix(healpix_info * geo, inum npoint, double * point_pos, int * point_pix, double rmax, double * dists, int * domains)
