#define _GNU_SOURCE
#include "distances_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define pi 3.141592653589793238462643383279502884197
#define nneigh 4

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
	double * edge_cos_dec = realloc(NULL, (inum)npoint*sizeof(double));
	double * edge_sin_dec = realloc(NULL, (inum)npoint*sizeof(double));
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
	double * edge_cos_dec = realloc(NULL, (inum)npoint*sizeof(double));
	double * edge_sin_dec = realloc(NULL, (inum)npoint*sizeof(double));
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
	v->y   = realloc(NULL, (inum)v->cap*sizeof(int));
	v->x   = realloc(NULL, (inum)v->cap*sizeof(int));
	return v;
}
void pointvec_push(PointVec * v, int y, int x) {
	if(v->n >= v->cap) {
		v->cap *= 2;
		v->y = realloc(v->y, (inum)v->cap*sizeof(int));
		v->x = realloc(v->x, (inum)v->cap*sizeof(int));
	}
	v->y[v->n] = y;
	v->x[v->n] = x;
	v->n++;
}
void pointvec_free(PointVec * v) { free(v->y); free(v->x); free(v); }
void pointvec_swap(PointVec ** a, PointVec ** b) { PointVec * tmp = *a; *a = *b; *b = tmp; }

void distance_from_points_bubble(int ny, int nx, double * posmap, inum npoint, double * point_pos, int * point_pix, double rmax, double * dists, int * domains)
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

	// Allow us to disable rmax by setting it to zero
	if(rmax <= 0) rmax = 1e300; // might consider using inf
	
	// Fill dists and domains with unvisited values
	for(inum i = 0; i < ny*nx; i++) {
		dists[i] = rmax;
		domains[i] = -1;
	}

	// Precompute cos and sin dec for the points
	double * point_cos_dec = realloc(NULL, (inum)npoint*sizeof(double));
	double * point_sin_dec = realloc(NULL, (inum)npoint*sizeof(double));
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
				if(cand_dist < dists[pix2] && cand_dist < rmax) {
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

void distance_from_points_bubble_separable(int ny, int nx, double * ypos, double * xpos, inum npoint, double * point_pos, int * point_pix, double rmax, double * dists, int * domains)
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

	// Allow us to disable rmax by setting it to zero
	if(rmax <= 0) rmax = 1e300; // might consider using inf
	
	// Fill dists and domains with unvisited values
	for(inum i = 0; i < ny*nx; i++) {
		dists[i] = rmax;
		domains[i] = -1;
	}

	// Precompute cos and sin dec for the points, as well as for the relatively
	// few dec values we have along the y axis due to our separable pixelization.
	double * point_cos_dec = realloc(NULL, (inum)npoint*sizeof(double));
	double * point_sin_dec = realloc(NULL, (inum)npoint*sizeof(double));
	for(inum j = 0; j < npoint; j++) {
		double dec = point_dec[j];
		point_cos_dec[j] = cos(dec);
		point_sin_dec[j] = sin(dec);
	}
	double * pix_cos_dec  = realloc(NULL, (inum)ny*sizeof(double));
	double * pix_sin_dec  = realloc(NULL, (inum)ny*sizeof(double));
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
				// Stop exploration if we're not the best so far, or if we are beyond rmax
				if(cand_dist < dists[pix2] && cand_dist < rmax) {
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

healpix_info * build_healpix_info(int nside) {
	healpix_info * geo = malloc(sizeof(healpix_info));
	geo->nside   = nside;
	geo->npix    = 12*(inum)nside*nside;
	geo->ncap    = 2*nside*(nside+1);
	geo->ny      = 4*nside-1;
	geo->nx      = calloc(geo->ny, sizeof(inum));
	geo->off     = calloc(geo->ny, sizeof(inum));
	geo->shift   = calloc(geo->ny, sizeof(int));
	geo->ra0     = calloc(geo->ny, sizeof(double));
	geo->dec     = calloc(geo->ny, sizeof(double));
	geo->cos_dec = calloc(geo->ny, sizeof(double));
	geo->sin_dec = calloc(geo->ny, sizeof(double));
	// North Polar cap
	for(int y = 0; y < nside; y++) {
		geo->nx[y]    = 4*(y+1);
		geo->off[y]   = 2*(inum)y*(y+1);
		geo->shift[y] = 1;
		geo->ra0[y]   = pi/geo->nx[y];
		geo->dec[y]   = pi/2 - 2*asin((y+1)/(sqrt(6.)*nside));
		geo->cos_dec[y] = cos(geo->dec[y]);
		geo->sin_dec[y] = sin(geo->dec[y]);
	}
	// Middle region
	for(int y = nside; y < 3*nside-1; y++) {
		geo->nx[y]    = 4*nside;
		geo->off[y]   = geo->ncap + 4*(inum)nside*(y-nside);
		geo->shift[y] = (y-nside)&1;
		geo->ra0[y]   = geo->shift[y] ? pi/geo->nx[y] : 0;
		geo->sin_dec[y] = (2*nside-(y+1))*(8.0*nside)/geo->npix;
		geo->cos_dec[y] = sqrt(1-geo->sin_dec[y]*geo->sin_dec[y]);
		geo->dec[y]   = asin(geo->sin_dec[y]);
	}
	// South polar cap
	for(int y = 3*nside-1; y < 4*nside-1; y++) {
		int y2 = 4*nside-2-y;
		geo->nx[y]    = geo->nx[y2];
		geo->off[y]   = geo->npix - geo->off[y2] - geo->nx[y2];
		geo->shift[y] = geo->shift[y2];
		geo->ra0[y]   = geo->ra0[y2];
		geo->dec[y]   = -geo->dec[y2];
		geo->cos_dec[y] =  geo->cos_dec[y2];
		geo->sin_dec[y] = -geo->sin_dec[y2];
	}
	return geo;
}

void free_healpix_info(healpix_info * geo) {
	if(geo->nx)      { free(geo->nx);      geo->nx      = NULL; }
	if(geo->off)     { free(geo->off);     geo->off     = NULL; }
	if(geo->shift)   { free(geo->shift);   geo->shift   = NULL; }
	if(geo->ra0)     { free(geo->ra0);     geo->ra0     = NULL; }
	if(geo->dec)     { free(geo->dec);     geo->dec     = NULL; }
	if(geo->cos_dec) { free(geo->cos_dec); geo->cos_dec = NULL; }
	if(geo->sin_dec) { free(geo->sin_dec); geo->sin_dec = NULL; }
	free(geo);
}

#if 0
// Being able to get all 8 neighbors would be useful for speeding up
// the bubble algorithm. This is not obvious, but because geodesics
// tend to get L-shaped towards the poles, it's better to explore
// faster in the cardinal directions than in the diagonal directions.
// I started implementing this, but it's too finicky. It seems like
// healpy is finally implementing their own distance transform anyway.

int get_xoff(healpix_info * geo, int y, int y2, int s) {
	// Get the correction we must add to x when moving from y to y2,
	// assuming that we're in split s
	int xoff = (geo->nx[y2]-geo->nx[y])/4*s;
	int eshift1 = geo->shift[y];
	int eshift2 = ego->shift[y2];
	if(y >= geo->nside && y < 3*geo->nside-1) eshift1 = -eshift1;
	if(y2>= geo->nside && y2< 3*geo->nside-1) eshift2 = -eshift2;
	xoff += 
}

// Get the 8 neighbors of the pixel (y,x). This is pretty complicated!
// The order is NW NE SW SE N S E W. Yes, this is a stupid order.
void get_healpix_neighs(healpix_info * geo, int y, int x, int * oy, int * ox) {
	// West and East are simple, since we have iso-latitude rings
	oy[6] = y+0; oy[7] = y+0;
	ox[6] = x-1; ox[7] = x+1;
	if(y >= geo->nside && y < 3*geo->nside-1) { // The middle band
		oy[0] = y-1; oy[1] = y-1; oy[2] = y+1; oy[3] = y+1;
		if(!geo->shift[y]) {
			ox[0] = x-1; ox[1] =   x; ox[2] = x-1; ox[3] =   x;
		} else {
			ox[0] =   x; ox[1] = x+1; ox[2] =   x; ox[3] = x+1;
		}
	} else { // one of the caps
		// left  edge at 0, nx/4, 2*nx/4 and 3*nx/4: x%(nx/4)==0
		// right edge at nx/4-1, 2*nx/4-1, etc.      (x+nx/4-1)%(nx/4) == 0
		int sw    = geo->nx[y]/4;
		int left  = x%sw == 0;
		int right = (x+1)%sw == 0;
		int s     = x/sw;
		if(y < geo->nside) { // north cap
			// Southwards is simple here
			oy[2] = y+1; oy[3] = y+1;
			oy[5] = y+2; ox[5] = x+2*s;
			// But north can be hard
			if (left) { oy[0] =   y; oy[4] =   y-1; ox[0] =   x-1; ox[4] = x-1-s; }
			else      { oy[0] = y-1; oy[4] =   y-2; ox[0] = x-1-s; ox[4] = x-2*s; }
			if (right){ oy[1] =   y; oy[4] =   y-1; ox[1] =   x+1; ox[4] = x+2-s; }
			else      { oy[1] = y-1; ox[1] = x+0-s; }


			// Extra complication from the case where our neighbor is in the middle band
			if(y == geo->nside-1) { ox[2] =   x; ox[3] =   x+1; }
			else                  { ox[2] = x+s; ox[3] = x+1+s; }
			if (left) { oy[0] =   y; ox[0] =   x-1; }
			else      { oy[0] = y-1; ox[0] = x-1-s; }
			if (right){ oy[1] =   y; ox[1] =   x+1; }
			else      { oy[1] = y-1; ox[1] = x+0-s; }

			// what determines the x shift?
			// dx = s*(onx-nx)/4 + 


		} else { // south cap
			// Up is simple here
			oy[0] = y-1; oy[1] = y-1;
			if(y == 3*geo->nside-1) { ox[0] =   x; ox[1] =   x+1; }
			else                    { ox[0] = x+s; ox[1] = x+1+s; }
			if (left) { oy[2] =   y; ox[2] =   x-1; }
			else      { oy[2] = y+1; ox[2] = x-1-s; }
			if (right){ oy[3] =   y; ox[3] =   x+1; }
			else      { oy[3] = y+1; ox[3] = x+0-s; }
		}
	}
	// Now handle the cardinal directions E, W are simple. Healpix disqualifies some of
	// these, but they're harmless to us. And since healpix rows really are iso-latitude
	// this choice is arguably more correct.
	oy[6] = y+0; oy[7] = y+0;
	ox[6] = x-1; ox[7] = x+1;
	// N, S are harder as we're switching rows. In the middle region NS is always
	// a step of 2. It will therefore take us to an row that has the same shift as
	// what we started on.
	if(y == 0) { // north pole
		oy[4] =   0; oy[5] = 2
		ox[4] = x+2; ox[5] = geo->off[2]+1+x*2;
	} else if(y < geo->nside) { // north cap
		int sw = geo->nx[y]/4;
		int  s = x/sw;
		int left  = x%sw == 0;
		int right = (x+1)%sw == 0;



	}


	//
	if(y > geo->nside && y < 3*geo->nside-2) { // middle with middle neighs
		oy[4] = y-2; oy[5] = y+2;
		ox[4] = x+0; ox[5] = x+0;
	} 

	// Handle shifted rows and wrapping
	for(int i = 0; i < nneigh; i++) {
		int w = geo->nx[oy[i]];
		if     (ox[i] < 0)  ox[i] += w;
		else if(ox[i] >= w) ox[i] -= w;
	}
}
#endif

// Get the 4 neighbors of the pixel (y,x). This is pretty complicated!
void get_healpix_neighs(healpix_info * geo, int y, int x, int * oy, int * ox) {
	if(y >= geo->nside && y < 3*geo->nside-1) { // The middle band
		oy[0] = y-1; oy[1] = y-1; oy[2] = y+1; oy[3] = y+1;
		if(!geo->shift[y]) {
			ox[0] = x-1; ox[1] =   x; ox[2] = x-1; ox[3] =   x;
		} else {
			ox[0] =   x; ox[1] = x+1; ox[2] =   x; ox[3] = x+1;
		}
	} else { // one of the caps
		// left  edge at 0, nx/4, 2*nx/4 and 3*nx/4: x%(nx/4)==0
		// right edge at nx/4-1, 2*nx/4-1, etc.      (x+nx/4-1)%(nx/4) == 0
		int sw    = geo->nx[y]/4;
		int left  = x%sw == 0;
		int right = (x+1)%sw == 0;
		int s     = x/sw;
		if(y < geo->nside) { // north cap
			oy[2] = y+1; oy[3] = y+1;
			// Extra complication from the case where our neighbor is in the middle band
			if(y == geo->nside-1) { ox[2] =   x; ox[3] =   x+1; }
			else                  { ox[2] = x+s; ox[3] = x+1+s; }
			if (left) { oy[0] =   y; ox[0] =   x-1; }
			else      { oy[0] = y-1; ox[0] = x-1-s; }
			if (right){ oy[1] =   y; ox[1] =   x+1; }
			else      { oy[1] = y-1; ox[1] = x+0-s; }
		} else { // south cap
			// Up is simple here
			oy[0] = y-1; oy[1] = y-1;
			if(y == 3*geo->nside-1) { ox[0] =   x; ox[1] =   x+1; }
			else                    { ox[0] = x+s; ox[1] = x+1+s; }
			if (left) { oy[2] =   y; ox[2] =   x-1; }
			else      { oy[2] = y+1; ox[2] = x-1-s; }
			if (right){ oy[3] =   y; ox[3] =   x+1; }
			else      { oy[3] = y+1; ox[3] = x+0-s; }
		}
	}
	// Handle shifted rows and wrapping
	for(int i = 0; i < nneigh; i++) {
		int w = geo->nx[oy[i]];
		if     (ox[i] < 0)  ox[i] += w;
		else if(ox[i] >= w) ox[i] -= w;
	}
}

void distance_from_points_bubble_healpix(healpix_info * geo, inum npoint, double * point_pos, int * point_pix, double rmax, double * dists, int * domains)
{
	// unwrap flattened arrays
	double * point_dec = point_pos;
	double * point_ra  = point_pos+npoint;
	int    * point_y   = point_pix;
	int    * point_x   = point_pix+npoint;
	// offsets in neigborhood search
	int yneigh[nneigh];
	int xneigh[nneigh];
	// Precompute cos and sin dec for the points
	double * point_cos_dec = realloc(NULL, (inum)npoint*sizeof(double));
	double * point_sin_dec = realloc(NULL, (inum)npoint*sizeof(double));
	for(inum j = 0; j < npoint; j++) {
		double dec = point_dec[j];
		point_cos_dec[j] = cos(dec);
		point_sin_dec[j] = sin(dec);
	}

	// Allow us to disable rmax by setting it to zero
	if(rmax <= 0) rmax = 1e300; // might consider using inf
	
	// Fill dists and domains with unvisited values
	for(inum i = 0; i < geo->npix; i++) {
		dists[i] = rmax;
		domains[i] = -1;
	}

	// These data structures will keep track of which points we're visiting
	PointVec * curr = pointvec_new();
	PointVec * next = pointvec_new();
	// Initialize our working set of points to the input points
	for(inum i = 0; i < npoint; i++) {
		int y = point_y[i], x = point_x[i];
		inum pix = geo->off[y]+x;
		double dra  = 2*pi/geo->nx[y];
		double dist = dist_vincenty_helper(point_ra[i], point_cos_dec[i], point_sin_dec[i],
				geo->ra0[y]+dra*x, geo->cos_dec[y], geo->sin_dec[y]);
		pointvec_push(curr, y, x);
		dists[pix]   = dist;
		domains[pix] = i;
	}

	inum it = 0;
	while(curr->n > 0) {
		//fprintf(stderr, "%5ld %10ld\n", it, curr->n);
		// For each of our current points, see if we can improve on their neighbors
		for(inum i = 0; i < curr->n; i++) {
			int y = curr->y[i], x = curr->x[i];
			inum pix    = geo->off[y]+x;
			inum ipoint = domains[pix];
			// Find out who our neighbors are
			get_healpix_neighs(geo, y, x, yneigh, xneigh);
			for(int oi = 0; oi < nneigh; oi++) {
				int y2 = yneigh[oi], x2 = xneigh[oi];
				inum pix2 = geo->off[y2]+x2;
				if(domains[pix2] == ipoint) continue;
				double dra  = 2*pi/geo->nx[y2];
				double cand_dist = dist_vincenty_helper(point_ra[ipoint], point_cos_dec[ipoint], point_sin_dec[ipoint],
						geo->ra0[y2]+dra*x2, geo->cos_dec[y2], geo->sin_dec[y2]);
				// Stop exploration if we're not the best so far, or if we are beyond rmax
				if(cand_dist < dists[pix2] && cand_dist < rmax) {
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

typedef struct { double v; int y, x; } HeapEntry;
typedef struct { inum n, cap; HeapEntry * data; } PointHeap;
PointHeap * pointheap_new() {
	PointHeap * heap = realloc(NULL, sizeof(PointHeap));
	heap->n    = 0;
	heap->cap  = 1024;
	heap->data = realloc(NULL, (inum)heap->cap*sizeof(HeapEntry));
	return heap;
}
void pointheap_free(PointHeap * heap) {
	free(heap->data);
	free(heap);
}
void heapentry_swap(HeapEntry * a, HeapEntry * b) { HeapEntry tmp = *a; *a = *b; *b = tmp; }
void pointheap_push(PointHeap * heap, HeapEntry e) {
	// First append it to the end
	if(heap->n >= heap->cap) {
		heap->cap *= 2;
		heap->data = realloc(heap->data, (inum)heap->cap*sizeof(HeapEntry));
	}
	heap->data[heap->n] = e;
	// Then update the heap condition
	inum i = heap->n, p;
	while(i) {
		p = i>>1;
		if(heap->data[i].v >= heap->data[p].v) break;
		heapentry_swap(&heap->data[i], &heap->data[p]);
		i = p;
	}
	heap->n++;
}
HeapEntry pointheap_pop(PointHeap * heap) {
	HeapEntry res = heap->data[0];
	// Replace root with last element
	heap->data[0] = heap->data[heap->n-1];
	heap->n--;
	// Then update the heap condition
	inum p = 0, c1, c2;
	while(p < heap->n/2) {
		c1 = p<<1; c2 = c1+1;
		if(heap->data[c1].v < heap->data[p].v) {
			if(c2 >= heap->n || heap->data[c1].v < heap->data[c2].v) {
				heapentry_swap(&heap->data[p], &heap->data[c1]);
				p = c1;
			} else {
				heapentry_swap(&heap->data[p], &heap->data[c2]);
				p = c2;
			}
		} else if(c2 < heap->n && heap->data[c2].v < heap->data[p].v) {
			heapentry_swap(&heap->data[p], &heap->data[c2]);
			p = c2;
		} else break;
	}
	return res;
}

void distance_from_points_heap_healpix(healpix_info * geo, inum npoint, double * point_pos, int * point_pix, double rmax, double * dists, int * domains)
{
	// unwrap flattened arrays
	double * point_dec = point_pos;
	double * point_ra  = point_pos+npoint;
	int    * point_y   = point_pix;
	int    * point_x   = point_pix+npoint;
	// offsets in neigborhood search
	int yneigh[nneigh];
	int xneigh[nneigh];
	// Precompute cos and sin dec for the points
	double * point_cos_dec = realloc(NULL, (inum)npoint*sizeof(double));
	double * point_sin_dec = realloc(NULL, (inum)npoint*sizeof(double));
	for(inum j = 0; j < npoint; j++) {
		double dec = point_dec[j];
		point_cos_dec[j] = cos(dec);
		point_sin_dec[j] = sin(dec);
	}

	// Allow us to disable rmax by setting it to zero
	if(rmax <= 0) rmax = 1e300; // might consider using inf
	
	// Fill dists and domains with unvisited values
	for(inum i = 0; i < geo->npix; i++) {
		dists[i] = rmax;
		domains[i] = -1;
	}

	// This data structure lets us keep the working set of points sorted
	PointHeap * heap = pointheap_new();
	// Initialize our working set of points to the input points
	for(inum i = 0; i < npoint; i++) {
		int y = point_y[i], x = point_x[i];
		inum pix = geo->off[y]+x;
		double dra  = 2*pi/geo->nx[y];
		double dist = dist_vincenty_helper(point_ra[i], point_cos_dec[i], point_sin_dec[i],
				geo->ra0[y]+dra*x, geo->cos_dec[y], geo->sin_dec[y]);
		HeapEntry entry = { dist, y, x };
		pointheap_push(heap, entry);
		dists[pix]   = dist;
		domains[pix] = i;
	}

	inum it = 0;
	while(heap->n > 0) {
		HeapEntry current = pointheap_pop(heap);
		int y = current.y, x = current.x;
		inum pix    = geo->off[y]+x;
		inum ipoint = domains[pix];
		// Find out who our neighbors are
		get_healpix_neighs(geo, y, x, yneigh, xneigh);
		for(int oi = 0; oi < nneigh; oi++) {
			int y2 = yneigh[oi], x2 = xneigh[oi];
			inum pix2 = geo->off[y2]+x2;
			if(domains[pix2] == ipoint) continue;
			double dra  = 2*pi/geo->nx[y2];
			double cand_dist = dist_vincenty_helper(point_ra[ipoint], point_cos_dec[ipoint], point_sin_dec[ipoint],
					geo->ra0[y2]+dra*x2, geo->cos_dec[y2], geo->sin_dec[y2]);
			// Stop exploration if we're not the best so far, or if we are beyond rmax
			if(cand_dist < dists[pix2] && cand_dist < rmax) {
				dists[pix2]   = cand_dist;
				domains[pix2] = ipoint;
				HeapEntry next = { cand_dist, y2, x2 };
				pointheap_push(heap, next);
			}
		}
		it++;
	}

	// Done. Free all our structurs
	free(point_cos_dec);
	free(point_sin_dec);
	pointheap_free(heap);
}

#define push(vec,cap,n,i) { if(n >= cap) { cap *= 2; vec = realloc(vec, (inum)cap*sizeof(inum)); } vec[n++] = i; }
inum find_edges(inum ny, inum nx, uint8_t * mask, inum ** edges)
{
	// Return the pixels defining the boundary of the zero regions in mask. These
	// are the pixels with value 0 that have a non-zero pixel neighbor. The beyond
	// edge area counts as non-zero for simplicity, since we don't know how things
	// might wrap around.
	// Start with the boundary
	inum y, x, i, n = 0, capacity = 0x100;
	inum * edges_ = realloc(NULL, (inum)capacity*sizeof(inum));
	for(i = 0; i < nx; i++)            if(mask[i] == 0) push(edges_, capacity, n, i);
	for(i = (ny-1)*nx; i < ny*nx; i++) if(mask[i] == 0) push(edges_, capacity, n, i);
	for(i = nx; i < ny*nx; i += nx)    if(mask[i] == 0) push(edges_, capacity, n, i);
	for(i = nx-1; i < ny*nx; i += nx)  if(mask[i] == 0) push(edges_, capacity, n, i);
	// Then do the interior
	for(y = 1; y < ny-1; y++)
	for(x = 1; x < nx-1; x++) {
		i = y*nx+x;
		if(mask[i] == 0 && (mask[i-1] != 0 || mask[i+1] != 0 || mask[i-nx] != 0 || mask[i+nx] != 0))
			push(edges_, capacity, n, i);
	}
	*edges = realloc(edges_, (inum)n*sizeof(inum));
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
	inum * edges_ = realloc(NULL, (inum)capacity*sizeof(inum));
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
	*edges = realloc(edges_, (inum)n*sizeof(inum));
	return n;
}

inum find_edges_healpix(healpix_info * geo, uint8_t * mask, int ** edges) {
	// Ironically, since going from pixel index to row, col in healpix is complicated,
	// we will return a 2d yx for this normally 1d pixelization
	PointVec * points = pointvec_new();
	int yneigh[nneigh], xneigh[nneigh];
	for(int y = 0; y < geo->ny; y++) {
		for(int x = 0; x < geo->nx[y]; x++) {
			get_healpix_neighs(geo, y, x, yneigh, xneigh);
			inum i = geo->off[y]+x;
			if(mask[i] != 0) continue;
			for(int j = 0; j < nneigh; j++) {
				int y2 = yneigh[j], x2 = xneigh[j];
				inum i2 = geo->off[y2]+x2;
				if(mask[i2] != 0) {
					// We're a non-zero neighbor of a zero element, which must have been a border
					pointvec_push(points, y, x);
					break;
				}
			}
		}
	}
	// Ok, we're done. Copy over our information into a single edges array,
	// which will be 2*nedge long, with the first half being y and the
	// second x
	inum n = points->n;
	*edges = calloc(2*n, sizeof(int));
	memcpy(*edges,   points->y, (size_t)n*sizeof(int));
	memcpy(*edges+n, points->x, (size_t)n*sizeof(int));
	pointvec_free(points);
	return n;
}

inum find_edges_labeled_healpix(healpix_info * geo, int32_t * labels, int ** edges) {
	// Ironically, since going from pixel index to row, col in healpix is complicated,
	// we will return a 2d yx for this normally 1d pixelization
	PointVec * points = pointvec_new();
	int yneigh[nneigh], xneigh[nneigh];
	for(int y = 0; y < geo->ny; y++) {
		for(int x = 0; x < geo->nx[y]; x++) {
			get_healpix_neighs(geo, y, x, yneigh, xneigh);
			inum i = geo->off[y]+x;
			if(labels[i] == 0) continue;
			for(int j = 0; j < nneigh; j++) {
				int y2 = yneigh[j], x2 = xneigh[j];
				inum i2 = geo->off[y2]+x2;
				if(labels[i2] != labels[i]) {
					// We're a non-zero neighbor of a zero element, which must have been a border
					pointvec_push(points, y, x);
					break;
				}
			}
		}
	}
	inum n = points->n;
	*edges = calloc(2*n, sizeof(int));
	//for(int i = 0; i < n; i++) (*edges)[i]   = points->y[i];
	//for(int i = 0; i < n; i++) (*edges)[n+i] = points->x[i];
	memcpy(*edges,   points->y, (size_t)n*sizeof(int));
	memcpy(*edges+n, points->x, (size_t)n*sizeof(int));
	pointvec_free(points);
	return n;
}

int32_t isqrt(inum x) {
	inum res = (inum)sqrt(x+0.5);
	// Healpix has these extra checks, which only matter for very large numbers
	if(x > (1LL<<50)) {
		if(res*res>x) res -= 1;
		else if((res+1)*(res+1)<=x) res += 1;
	}
	return (int32_t)res;
}

int32_t pix2y_healpix(healpix_info * geo, inum pix) {
	inum pix2 = geo->npix-1-pix;
	if(pix < geo->ncap) {
		// North polar cap. Here each row starts at off = 2*y*(y+1), so y**2+y - off/2 = 0
		// y = (-1+sqrt(1+2*off))/2 rounded down
		return (-1+isqrt(1+2*pix))/2;
	} else if(pix2 < geo->ncap) {
		// South polar cap
		return 4*geo->nside-2-(-1+isqrt(1+2*pix2))/2;
	} else {
		// Middle area
		return geo->nside + (pix - geo->ncap)/(4*geo->nside);
	}
}

void ravel_healpix(healpix_info * geo, inum npoint, int32_t * pix2d, inum * pix1d) {
	// Given pix2d[{y,x},npoint] 2d healpix indices, retulrn pix1d[npoint] standard 1d healpix indices
	int32_t * y = pix2d;
	int32_t * x = pix2d + npoint;
	#pragma omp parallel for
	for(inum i = 0; i < npoint; i++)
		pix1d[i] = geo->off[y[i]]+x[i];
}

void unravel_healpix(healpix_info * geo, inum npoint, inum * pix1d, int32_t * pix2d) {
	// Given pix1d[npoint] standard 1d healpix indices, return pix2d[{y,x},npoint]
	int32_t * y = pix2d;
	int32_t * x = pix2d + npoint;
	#pragma omp parallel for
	for(inum i = 0; i < npoint; i++) {
		y[i] = pix2y_healpix(geo, pix1d[i]);
		x[i] = pix1d[i] - geo->off[y[i]];
	}
}

