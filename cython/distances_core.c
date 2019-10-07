#include <stdlib.h>
#include <math.h>
#include "distances_core.h"
#include <sys/time.h>
#include <stdio.h>

double wall_time() { struct timeval tv; gettimeofday(&tv,0); return tv.tv_sec + 1e-6*tv.tv_usec; }

// The simple functions are too slow to serve as the basis for a distance transform.
// I can't afford the full O(npix*nedge) scaling.
//
// A faster algorithm would be one that starts from the pixels closest to each point,
// and then gradually grows outwards by processing neighbors. This would have an O(npix)
// scaling. Two variants:
// 1. Always visit the point with the shortest distance so far. Simple update logic and
//    minimal number of visits to each pixel, but will require a heap data structure,
//    which could be slow.
// 2. Work in a set of passes. For each pass one updates the neighbors of pixels in the
//    most recent pass. The set up updated pixels that resulted in a shorter distance
//    than before become the basis for the next pass. Stop when the list of candidates
//    becomes empty. This will revisit pixels more times than necessary, but avoids
//    using a heap.
// This is a good fit for the distance transform. It can also be used to implement
// distance_from_points, but only if a starting set of pixels is available, making
// the interface a bit more clunky. The algorithm is hard to parallelize.
//
// Alternative: coarse grid. Make a grid several times lower resolution and find the
// domains for points given by the center of each coarse pixel that includes an original point.
// For each pixel in a given coarse cell, we only need to consider points that correspond
// to the domain of it or one of its neighbors. This might be a bit easier to implement
// than the first algorithm, and also more parallelizable, but will require more distnaces
// to be computed total, and isn't that straightforward anyway. It might also not be 100% accurate.

// coordinate ordering: dec,ra

void distance_from_points_simple(inum npix, double * posmap, inum npoint, double * points, double * dists, int * areas)
{
	// Compute the distance from each entry in postmap to the closest entry in points, storing the
	// result in dists. Use Vincenty formula for distances. It's a bit slower than the simplest formula, but
	// it's very stable. Can optimize later
	// Precompute cos and sin dec for the edge pixels
	double * edge_cos_dec = realloc(NULL, sizeof(double)*npoint);
	double * edge_sin_dec = realloc(NULL, sizeof(double)*npoint);
	for(inum j = 0; j < npoint; j++) {
		double dec = points[j];
		edge_cos_dec[j] = cos(dec);
		edge_sin_dec[j] = sin(dec);
	}
	for(inum i = 0; i < npix; i++) {
		double pix_dec = posmap[i];
		double pix_ra  = posmap[i+npix];
		double pix_cos_dec = cos(pix_dec);
		double pix_sin_dec = sin(pix_dec);
		for(inum j = 0; j < npoint; j++) {
			double edge_ra = points[j+npoint];
			double dra = pix_ra - edge_ra;
			double cos_dra = cos(dra);
			double sin_dra = sin(dra);
			// a = pix, b = edge
			double y1 = edge_cos_dec[j]*sin_dra;
			double y2 = pix_cos_dec*edge_sin_dec[j]-pix_sin_dec*edge_cos_dec[j]*cos_dra;
			double y = sqrt(y1*y1+y2*y2);
			double x = pix_sin_dec*edge_sin_dec[j] + pix_cos_dec*edge_cos_dec[j]*cos_dra;
			double d = atan2(y,x);
			if(j == 0 || d < dists[i]) {
				dists[i] = d;
				if(areas) areas[i] = j;
			}
		}
	}
	free(edge_cos_dec);
	free(edge_sin_dec);
}

void distance_from_points_simple_separable(inum ny, inum nx, double * ypos, double * xpos, inum npoint, double * points, double * dists, int * areas)
{
	// Compute the distance from each entry in postmap to the closest entry in points, storing the
	// result in dists. Use Vincenty formula for distances. It's a bit slower than the simplest formula, but
	// it's very stable. Can optimize later
	// Precompute cos and sin dec for the edge pixels
	double t1 = wall_time();
	double * edge_cos_dec = realloc(NULL, sizeof(double)*npoint);
	double * edge_sin_dec = realloc(NULL, sizeof(double)*npoint);
	for(inum j = 0; j < npoint; j++) {
		double dec = points[j];
		edge_cos_dec[j] = cos(dec);
		edge_sin_dec[j] = sin(dec);
	}
	double t2 = wall_time();
	#pragma omp parallel for
	for(inum y = 0; y < ny; y++) {
		double pix_dec = ypos[y];
		double pix_cos_dec = cos(pix_dec);
		double pix_sin_dec = sin(pix_dec);
		for(inum x = 0; x < nx; x++) {
			double pix_ra  = xpos[x];
			inum i = y*nx+x;
			for(inum j = 0; j < npoint; j++) {
				double edge_ra = points[j+npoint];
				double dra = pix_ra - edge_ra;
				double cos_dra = cos(dra);
				double sin_dra = sin(dra);
				// a = pix, b = edge
				double y1 = edge_cos_dec[j]*sin_dra;
				double y2 = pix_cos_dec*edge_sin_dec[j]-pix_sin_dec*edge_cos_dec[j]*cos_dra;
				double wy = sqrt(y1*y1+y2*y2);
				double wx = pix_sin_dec*edge_sin_dec[j] + pix_cos_dec*edge_cos_dec[j]*cos_dra;
				double d = atan2(wy,wx);
				if(j == 0 || d < dists[i]) {
					dists[i] = d;
					if(areas) areas[i] = j;
				}
			}
		}
	}
	double t3 = wall_time();
	fprintf(stderr, "%8.4f %8.4f\n", t2-t1, t3-t2);
	free(edge_cos_dec);
	free(edge_sin_dec);
}

double dist_vincenty_helper(double ra1, double cos_dec1, double sin_dec1, double ra2, double cos_dec2, double sin_dec2) {
	double dra = ra2 - ra1;
	double cos_dra = cos(dra);
	double sin_dra = sin(dra);
	double y1 = cos_dec1*sin_dra;
	double y2 = cos_dec2*sin_dec1-sin_dec2*cos_dec1*cos_dra;
	double y  = sqrt(y1*y1+y2*y2);
	double x  = sin_dec2*sin_dec1 + cos_dec2*cos_dec1*cos_dra;
	double d  = atan2(y,x);
	return d;
}

typedef struct { inum n, cap; int * y, * x; } PointVec;
PointVec * pointvec_new() {
	PointVec * v = realloc(NULL, sizeof(PointVec));
	v->n   = 0;
	v->cap = 1024;
	v->y   = realloc(NULL, v->cap*sizeof(int));
	v->x   = realloc(NULL, v->cap*sizeof(int));
	return v;
}
void pointvec_push(PointVec * v, int y, int x) {
	if(v->n >= v->cap) {
		v->cap *= 2;
		v->y = realloc(v->y, v->cap*sizeof(int));
		v->x = realloc(v->x, v->cap*sizeof(int));
	}
	v->y[v->n] = y;
	v->x[v->n] = x;
	v->n++;
}
void pointvec_free(PointVec * v) { free(v->y); free(v->x); free(v); }
void pointvec_swap(PointVec ** a, PointVec ** b) { PointVec * tmp = *a; *a = *b; *b = tmp; }

#define nneigh 4
void distance_from_points_bubble(int ny, int nx, double * posmap, inum npoint, double * point_pos, int * point_pix, double * dists, int * domains)
{
	// Compute the distance from each entry in postmap to the closest entry in points, storing the
	// result in dists. Works by starting from the closest pixels to the points, then working outwards
	// through neighbors, keeping track of the shortest distance to each pixel and which point that
	// corresponded to. It is possible for this approach to fail for the case of very narrow (less than
	// a pixel wide) domains, but this will only result in a tiny error in the distance, so it's acceptable.
	inum npix = (inum)ny*nx;
	double * point_dec = point_pos;
	double * point_ra  = point_pos+npoint;
	int    * point_y   = point_pix;
	int    * point_x   = point_pix+npoint;
	double * pix_dec   = posmap;
	double * pix_ra    = posmap+npix;
	
	// Fill dists and domains with unvisited values
	for(inum i = 0; i < ny*nx; i++) {
		dists[i] = 1e300; // would use inf, but harder to generate
		domains[i] = -1;
	}

	// Precompute cos and sin dec for the points
	double * point_cos_dec = realloc(NULL, sizeof(double)*npoint);
	double * point_sin_dec = realloc(NULL, sizeof(double)*npoint);
	for(inum j = 0; j < npoint; j++) {
		double dec = point_dec[j];
		point_cos_dec[j] = cos(dec);
		point_sin_dec[j] = sin(dec);
	}

	// offsets in neigborhood search
	int yoffs[nneigh] = { 0,  0, -1, +1};
	int xoffs[nneigh] = {-1, +1,  0,  0};

	// These data structures will keep track of which points we're visiting
	PointVec * curr = pointvec_new();
	PointVec * next = pointvec_new();
	// Initialize our working set of points to the input points
	for(inum i = 0; i < npoint; i++) {
		int y = point_y[i], x = point_x[i];
		inum pix = (inum)y*nx+x;
		double dist = dist_vincenty_helper(point_ra[i], point_cos_dec[i], point_sin_dec[i], pix_ra[pix], cos(pix_dec[pix]), sin(pix_dec[pix]));
		pointvec_push(curr, y, x);
		dists[y*nx+x]   = dist;
		domains[y*nx+x] = i;
	}

	inum it = 0;
	while(curr->n > 0) {
		//fprintf(stderr, "%5ld %10ld\n", it, curr->n);
		// For each of our current points, see if we can improve on their neighbors
		for(inum i = 0; i < curr->n; i++) {
			int y = curr->y[i], x = curr->x[i];
			inum pix    = (inum)y*nx+x;
			inum ipoint = domains[pix];
			for(int oi = 0; oi < nneigh; oi++) {
				int y2 = y+yoffs[oi], x2 = x+xoffs[oi];
				// Handle edge wrapping. This doesn't cover all the ways wrapping can happen, though...
				if(y2 < 0) { y2 += ny; } else if(y2 >= ny) { y2 -= ny; }
				if(x2 < 0) { x2 += nx; } else if(x2 >= nx) { x2 -= nx; }
				inum pix2 = y2*nx+x2;
				if(domains[pix2] == ipoint) continue;
				double cand_dist = dist_vincenty_helper(point_ra[ipoint], point_cos_dec[ipoint], point_sin_dec[ipoint], pix_ra[pix2], cos(pix_dec[pix2]), sin(pix_dec[pix2]));
				if(cand_dist < dists[pix2]) {
					dists[pix2]   = cand_dist;
					domains[pix2] = ipoint;
					pointvec_push(next, y2, x2);
				}
			}
		}
		pointvec_swap(&curr, &next);
		next->n = 0;
		it++;
	}

	// Done. Free all our structurs
	free(point_cos_dec);
	free(point_sin_dec);
	pointvec_free(curr);
	pointvec_free(next);
}
#undef nneigh

#define nneigh 4
void distance_from_points_bubble_separable(int ny, int nx, double * ypos, double * xpos, inum npoint, double * point_pos, int * point_pix, double * dists, int * domains)
{
	// Compute the distance from each entry in postmap to the closest entry in points, storing the
	// result in dists. Works by starting from the closest pixels to the points, then working outwards
	// through neighbors, keeping track of the shortest distance to each pixel and which point that
	// corresponded to. It is possible for this approach to fail for the case of very narrow (less than
	// a pixel wide) domains, but this will only result in a tiny error in the distance, so it's acceptable.
	double * point_dec = point_pos;
	double * point_ra  = point_pos+npoint;
	int    * point_y   = point_pix;
	int    * point_x   = point_pix+npoint;
	
	// Fill dists and domains with unvisited values
	for(inum i = 0; i < ny*nx; i++) {
		dists[i] = 1e300; // would use inf, but harder to generate
		domains[i] = -1;
	}

	// Precompute cos and sin dec for the points, as well as for the relatively
	// few dec values we have along the y axis due to our separable pixelization.
	double * point_cos_dec = realloc(NULL, sizeof(double)*npoint);
	double * point_sin_dec = realloc(NULL, sizeof(double)*npoint);
	for(inum j = 0; j < npoint; j++) {
		double dec = point_dec[j];
		point_cos_dec[j] = cos(dec);
		point_sin_dec[j] = sin(dec);
	}
	double * pix_cos_dec  = realloc(NULL, sizeof(double)*ny);
	double * pix_sin_dec  = realloc(NULL, sizeof(double)*ny);
	for(int y = 0; y < ny; y++) {
		pix_cos_dec[y] = cos(ypos[y]);
		pix_sin_dec[y] = sin(ypos[y]);
	}

	// offsets in neigborhood search
	int yoffs[nneigh] = { 0,  0, -1, +1};
	int xoffs[nneigh] = {-1, +1,  0,  0};

	// These data structures will keep track of which points we're visiting
	PointVec * curr = pointvec_new();
	PointVec * next = pointvec_new();
	// Initialize our working set of points to the input points
	for(inum i = 0; i < npoint; i++) {
		int y = point_y[i], x = point_x[i];
		double dist = dist_vincenty_helper(point_ra[i], point_cos_dec[i], point_sin_dec[i], xpos[x], pix_cos_dec[y], pix_sin_dec[y]);
		pointvec_push(curr, y, x);
		dists[y*nx+x]   = dist;
		domains[y*nx+x] = i;
	}

	inum it = 0;
	while(curr->n > 0) {
		//fprintf(stderr, "%5ld %10ld\n", it, curr->n);
		// For each of our current points, see if we can improve on their neighbors
		for(inum i = 0; i < curr->n; i++) {
			int y = curr->y[i], x = curr->x[i];
			inum pix    = (inum)y*nx+x;
			inum ipoint = domains[pix];
			for(int oi = 0; oi < nneigh; oi++) {
				int y2 = y+yoffs[oi], x2 = x+xoffs[oi];
				// Handle edge wrapping. This doesn't cover all the ways wrapping can happen, though...
				if(y2 < 0) { y2 += ny; } else if(y2 >= ny) { y2 -= ny; }
				if(x2 < 0) { x2 += nx; } else if(x2 >= nx) { x2 -= nx; }
				inum pix2 = y2*nx+x2;
				if(domains[pix2] == ipoint) continue;
				double cand_dist = dist_vincenty_helper(point_ra[ipoint], point_cos_dec[ipoint], point_sin_dec[ipoint], xpos[x2], pix_cos_dec[y2], pix_sin_dec[y2]);
				if(cand_dist < dists[pix2]) {
					dists[pix2]   = cand_dist;
					domains[pix2] = ipoint;
					pointvec_push(next, y2, x2);
				}
			}
		}
		pointvec_swap(&curr, &next);
		next->n = 0;
		it++;
	}

	// Done. Free all our structurs
	free(point_cos_dec);
	free(point_sin_dec);
	free(pix_cos_dec);
	free(pix_sin_dec);
	pointvec_free(curr);
	pointvec_free(next);
}
#undef nneigh



#define push(vec,cap,n,i) { if(n >= cap) { cap *= 2; vec = realloc(vec, sizeof(inum)*cap); } vec[n++] = i; }
inum find_edges(inum ny, inum nx, uint8_t * mask, inum ** edges)
{
	// Return the pixels defining the boundary of the zero regions in mask. These
	// are the pixels with value 0 that have a non-zero pixel neighbor. The beyond
	// edge area counts as non-zero for simplicity, since we don't know how things
	// might wrap around.
	// Start with the boundary
	inum y, x, i, n = 0, capacity = 0x100;
	inum * edges_ = realloc(NULL, sizeof(inum)*capacity);
	for(i = 0; i < nx; i++)            if(mask[i] == 0) push(edges_, capacity, n, i);
	for(i = (ny-1)*nx; i < nx*nx; i++) if(mask[i] == 0) push(edges_, capacity, n, i);
	for(i = nx; i < ny*nx; i += nx)    if(mask[i] == 0) push(edges_, capacity, n, i);
	for(i = nx-1; i < ny*nx; i += nx)  if(mask[i] == 0) push(edges_, capacity, n, i);
	// Then do the interior
	for(y = 1; y < ny-1; y++)
	for(x = 1; x < nx-1; x++) {
		i = y*nx+x;
		if(mask[i] == 0 && (mask[i-1] != 0 || mask[i+1] != 0 || mask[i-nx] != 0 || mask[i+nx] != 0))
			push(edges_, capacity, n, i);
	}
	*edges = realloc(edges_, sizeof(inum)*n);
	return n;
}

inum find_edges_labeled(inum ny, inum nx, int32_t * labels, inum ** edges)
{
	// Return the pixels defining the boundary of the non-zero regions in mask. These
	// are the pixels with a non-zero value that have a different-valued neighbor. The beyond
	// edge area counts as different for simplicity, since we don't know how things
	// might wrap around. Allocates and fills *edges with the indices of the pixels at the edge 
	// of the regions, and elabs with the label 
	// Start with the boundary
	inum y, x, i, n = 0, capacity = 0x100;
	inum * edges_ = realloc(NULL, sizeof(inum)*capacity);
	for(i = 0; i < nx; i++)            if(labels[i]) push(edges_, capacity, n, i);
	for(i = (ny-1)*nx; i < ny*nx; i++) if(labels[i]) push(edges_, capacity, n, i);
	for(i = nx; i < ny*nx; i += nx)    if(labels[i]) push(edges_, capacity, n, i);
	for(i = nx-1; i < ny*nx; i += nx)  if(labels[i]) push(edges_, capacity, n, i);
	// Then do the interior
	for(y = 1; y < ny-1; y++)
	for(x = 1; x < nx-1; x++) {
		i = y*nx+x;
		if(labels[i] && (labels[i-1] != labels[i] || labels[i+1] != labels[i] || labels[i-nx] != labels[i] || labels[i+nx] != labels[i]))
			push(edges_, capacity, n, i);
	}
	*edges = realloc(edges_, sizeof(inum)*n);
	return n;
}
#undef push
