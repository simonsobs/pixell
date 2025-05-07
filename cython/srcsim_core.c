// This file provides a low-level implementation of object (point source, cluster
// ect.) simulation. Given a catalog of positions, peak amplitudes and radial profiles
// it paints these on a sky map.

// 1. For each source, decide the maximum relevant radius for it,
//    using its profile and peak amplitude.
// 2. We split the map into cells, e.g. 16x16 pixels
// 3. Decide which cells each source overlaps, building a list
//    of object indices for each cell. This is probably faster
//    for larger cells. Have a second coarser tiling to help with this.
//    To identify the cells, let rmax be the maximum relevant radius
//    for the object, and include it if dist(obj,cell_center) < rmax+cell_rmax
//    cell_rmax can be found by computing the distance from a cell to its neighbors.
// 4. OMP loop over each cell
// 5. Copy out the pixel data for the cell from the big map: cell_map
// 6. For each object in the cell, make a zeroed scratch buffer the same size of cell_map.
//    Loop over each pixel in this and compute the distance to the object,
//    and use this to interpolate the profile value here. Might want to support several
//    interpolations, but the baseline is non-equidistant linear interpolation, like
//    what np.interp supports. Hopefully this won't be too slow.
//    Multiply the profile value by the peak amplitude and write to the scratch buffer.
// 7. merge the scratch buffer into cell_map using the combination operation, which
//    can be = += max= min= etc.
// 8. copy cell_map back into the full map

// Alternative approach:
// For each source build a rectangular area big enough to hold all
// relevant pixels, simulate it there, and then merge this into the map.
// I think this is worse because it could be hard to estimate those
// pixels ingeneral, and because one needs to avoid clobbering in the second step.

// TODO: Check if restrict keyword helps

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "srcsim_core.h"

// Precompute profile levels down to 1e-10 of the abspeak. Beyond that
// the whole profile is used.
#define prof_base   10
#define prof_levels 11
// Only use profile precompution if we have significantly fewer profiles than objects
#define prof_prec_lim 10

// I thought this was part of math.h, but apparently it's optional
#define M_PI 3.14159265358979323846

// Forward declaration for all the implementation details that don't
// belong in the header

float sum_map(float * m, int ny, int nx) {
	float sum = 0;
	for(int y = 0; y < ny; y++)
	for(int x = 0; x < nx; x++)
		sum += m[y*nx+x];
	return sum;
}

float * measure_amax(int nobj, int ncomp, float ** amps);
float * measure_rmax(int nobj, float * amaxs, int * prof_ids, int * prof_ns, float ** prof_rs, float ** prof_vs, float vmin, float rmax, int * prof_info);
int * build_prof_info(int nprof, int * prof_ns, float ** prof_vs);

int assign_cells(
		int nobj,         // Number of objects
		float * obj_decs, // Object coordinates
		float * obj_ras,  //
		int   * obj_ys,   // Object pixel coordinates
		int   * obj_xs,   //
		float * rmaxs,    // Max relevant radius for each object
		int ny, int nx,   // Map dimensions
		int separable,    // Are ra/dec separable?
		float * pix_decs, // Coordinates of map pixels
		float * pix_ras,  //
		int csize,        // Cell size
		int **cell_nobj,  // Output parameter. Number of objects for each cell
		int ***cell_objs, // Output parameter. Ids of objects in each cell
		int ***cell_boxes // Output parameter. {y1,y2,x1,x1} for each cell.
	);

void process_cell(
		int nobj,         // Number of objects in this cell
		int * objs,       // ids of those objects
		int * box,        // {y1,y2,x1,x2} pixel bounding box
		float * obj_decs, // [nobj_tot]. Coordinates of objects
		float * obj_ras,  // [nobj_tot]
		float ** amps,    // [ncomp][nobj_tot]
		int * prof_ids,   // Profile id for each object id
		int * prof_ns,    // Number of points for each profile id
		float ** prof_rs, // [nprof][prof_n]. R values in each profile
		float ** prof_vs, // [nprof][prof_n]. Profile values.
		int prof_equi,    // are profiles equi-spaced?
		int op,           // The operation to perform when merging object signals
		int ncomp, int ny, int nx, // Number of components
		int separable,    // Are ra/dec separable?
		int transpose,    // Whether to do the transpose operation: map -> amp
		float * pix_decs, // [ny*nx]. Coordinates of objects
		float * pix_ras,  // [ny*nx]
		float ** imap,    // [ncomp,ny*nx]. The input map
		float ** omap     // [ncomp,ny*nx]. The output map. Can be the same as the input map
	);

float * extract_map(float ** imap, int * box, int ncomp, int iny, int inx, int ystep, int xstep);
float * extract_coords(float * imap, int * box, int iny, int inx, int ystep, int xstep);
void insert_map(float * imap, float ** omap, int * box, int ncomp, int ony, int onx);

void paint_object(
		float obj_dec,             // object coordinates
		float obj_ra,              //
		float * restrict amps,     // [ncomp], e.g. T, Q, U.
		int prof_n,                // number of sample points in profile
		float * restrict prof_rs,  // radial coordinate for each sample point
		float * restrict prof_vs,  // profile value for each sample point
		int prof_equi,             // are profiles equi_spaced?
		int ncomp, int ny, int nx, // cell dimensions
		int transpose,             // Whether to do the transpose operation: map -> amp
		float * restrict pix_decs, // pixel coordinates
		float * restrict pix_ras,  //
		float * restrict map       // map to overwrite
	);

void merge_cell(int n, int op, float * restrict source, float * restrict target);
float evaluate_profile(int n, float * rs, float * vs, float r, int equi);
int binary_search(int n, float * rs, float r);
int equi_search(int n, float * rs, float r);

float calc_dist(float dec1, float ra1, float dec2, float ra2);
float calc_grad(int i, int n, int s, float * v);

void calc_pix_shape(int y, int x, int ny, int nx, int separable, float * pix_decs, float * pix_ras, float * ysize, float * xsize);
void estimate_bounding_box(
		int   obj_y,      // object pixel coordinates
		int   obj_x,      //
		float rmax,       // max relevant radius for object
		int ny, int nx,   // map dimensions
		int separable,    // Are ra/dec separable?
		float * pix_decs, // coordinates of map pixels
		float * pix_ras,  //
		int * box         // {y1,y2,x1,x2} in pixels.
	);
void pixbox2cellbox(int * pixbox, int csize, int ncy, int ncx, int * cellbox);
void wrap_box(int * box, int ny, int nx);

typedef struct IntList { int n, cap; int * vals; } IntList;
IntList * intlist_new();
void intlist_push(IntList * v, int val);
void intlist_free(IntList * v);
void intlist_swap(IntList ** a, IntList ** b);

float *** calloc_pppf(int n1, int n2, int n3);
void free_pppf(float *** a, int n1, int n2);
void reduce_amps(float *** work_amps, float ** amps, int nwork, int ncomp, int nobj);

double wall_time() { struct timeval tv; gettimeofday(&tv,0); return tv.tv_sec + 1e-6*tv.tv_usec; }

void sim_objects(
		int nobj,         // Number of objects
		float * obj_decs, // [nobj]. Coordinates of objects
		float * obj_ras,  // [nobj]
		int   * obj_ys,   // [nobj]. Pixel coordinates of objects. Theoretically redundant,
		int   * obj_xs,   // [nobj], but useful in practice since we don't have the wcs here.
		float ** amps,    // [ncomp][nobj]. Peak amplitude. comp = stokes for example
		int nprof,        // Number of unique profiles
		int * prof_ids,   // Profile id for each object
		int * prof_ns,    // [nprof]. Samples in each profile
		float ** prof_rs, // [nprof][prof_n]. R values in each profile
		float ** prof_vs, // [nprof][prof_n]. Profile values.
		int prof_equi,    // are profiles equi-spaced?
		float vmin,       // Lowest value to simulate, in amplitude units = map units
		float rmax,       // Maximum radius to consider, even if vmin would want more
		int op,           // The operation to perform when merging object signals
		int ncomp, int ny, int nx,// Map dimensions
		int separable,    // Are ra/dec separable?
		int transpose,    // Whether to do the transpose operation, which reads from map and writes to amps
		float *  pix_decs,// [ny*nx]
		float *  pix_ras, // [ny*nx]
		float ** imap,    // [ncomp,ny*nx]. The input map
		float ** omap,    // [ncomp,ny*nx]. The output map. Can be the same as the input map.
		int csize,        // cell size. These are processed in parallel. E.g. 32 for 32x32 cells
		double * times    // Time taken in the different steps
	) {
	// 1. Measure the maximum radius for each source. prof_info is an optimization for the common
	// case where the same profile is shared between many objects. In this case it can make sense
	// to precompute information about roughly where in the profile relevant values start.
	double t1 = wall_time();
	float * amaxs = measure_amax(nobj, ncomp, amps);
	int   * prof_info = nobj > nprof*prof_prec_lim ? build_prof_info(nprof, prof_ns, prof_vs) : NULL;
	float * rmaxs = measure_rmax(nobj, amaxs, prof_ids, prof_ns, prof_rs, prof_vs, vmin, rmax, prof_info);
	free(amaxs);
	free(prof_info);
	// 2. Find which objects are relevant for which cells
	double t2 = wall_time();
	int *cell_nobj, **cell_objs, **cell_boxes; // [ncell], [ncell][objs] and [ncell][{y1,y2,x1,x2}]
	int ncell = assign_cells(nobj, obj_decs, obj_ras, obj_ys, obj_xs, rmaxs, ny, nx, separable, pix_decs, pix_ras, csize, &cell_nobj, &cell_objs, &cell_boxes);
	double t3 = wall_time();
	// 3. Process each cell
	if(transpose) {
		int nthread = omp_get_max_threads();
		float *** work_amps = calloc_pppf(nthread, ncomp, nobj);
		#pragma omp parallel
		{
			int thread = omp_get_thread_num();
			#pragma omp for
			for(int ci = 0; ci < ncell; ci++)
				process_cell(cell_nobj[ci], cell_objs[ci], cell_boxes[ci], obj_decs, obj_ras, work_amps[thread], prof_ids, prof_ns, prof_rs, prof_vs, prof_equi, op, ncomp, ny, nx, separable, transpose, pix_decs, pix_ras, imap, omap);
		}
		reduce_amps(work_amps, amps, nthread, ncomp, nobj);
		free_pppf(work_amps, nthread, ncomp);
	} else {
		#pragma omp parallel for
		for(int ci = 0; ci < ncell; ci++)
			process_cell(cell_nobj[ci], cell_objs[ci], cell_boxes[ci], obj_decs, obj_ras, amps, prof_ids, prof_ns, prof_rs, prof_vs, prof_equi, op, ncomp, ny, nx, separable, transpose, pix_decs, pix_ras, imap, omap);
	}
	double t4 = wall_time();
	times[0] = t2-t1;
	times[1] = t3-t2;
	times[2] = t4-t3;
	// Clean up stuff
	for(int ci = 0; ci < ncell; ci++) {
		free(cell_objs[ci]);
		free(cell_boxes[ci]);
	}
	free(cell_objs);
	free(cell_boxes);
	free(cell_nobj);
	free(rmaxs);
}

float * measure_amax(int nobj, int ncomp, float ** amps) {
	float * amaxs = calloc(nobj, sizeof(float));
	#pragma omp parallel for
	for(int i = 0; i < nobj; i++) {
		float amax = fabsf(amps[0][i]);
		for(int c = 1; c < ncomp; c++)
			amax = fmaxf(amax, fabsf(amps[c][i]));
		amaxs[i] = amax;
	}
	return amaxs;
}

// prof_info is conceptually a 2d array with the one-past-the-end
// index of the relevant part of the profile for exponentially spaced
// value cutoffs.
int * build_prof_info(int nprof, int * prof_ns, float ** prof_vs) {
	int * prof_info = calloc(nprof*prof_levels, sizeof(int));
	#pragma omp parallel for
	for(int i = 0; i < nprof; i++) {
		// First measure the maximum value and index
		float vmax = fabsf(prof_vs[i][0]);
		int imax = 0;
		for(int ri = 1; ri < prof_ns[i]; ri++) {
			float v = fabsf(prof_vs[i][ri]);
			if(v > vmax) {
				imax = ri;
				vmax = v;
			}
		}
		// This corresponds to cases where the cutoff is bigger than
		// any value the beam can reach. Should really set it to 0
		// in this case, but easier for measure_rmax if se set it to
		// 1
		prof_info[i*prof_levels+0] = 1;
		// The last level just goes all the way out to the end
		prof_info[i*prof_levels+prof_levels-1] = prof_ns[i];
		// Populate the others by inwards from the edge
		int ri = prof_ns[i]-1;
		for(int li = prof_levels-2; li > 0; li--) {
			float vlim = vmax * powf(prof_base, -li);
			for(; ri > imax && fabsf(prof_vs[i][ri]) < vlim; ri--);
			prof_info[i*prof_levels+li] = ri;
		}
	}
	return prof_info;
}

int prof_info_lookup(int * prof_info, int prof, float vrel) {
	// Calculate what power of our base (e.g. power of ten) we are
	// away from the profile peak
	int ind = (int)ceilf(-logf(vrel)/logf(prof_base));
	// Cap to [0:nlevel]
	if(ind < 0) ind = 0;
	else if(ind >= prof_levels) ind = prof_levels-1;
	// And look up the relevant starting point in the profile
	return prof_info[prof_levels*prof+ind];
}

float * measure_rmax(int nobj, float * amaxs, int * prof_ids, int * prof_ns, float ** prof_rs, float ** prof_vs, float vmin, float rmax, int * prof_info) {
	float * rmaxs = calloc(nobj, sizeof(float));
	#pragma omp parallel for
	for(int oi = 0; oi < nobj; oi++) {
		int pid    = prof_ids[oi];
		int n      = prof_ns[pid];
		float * rs = prof_rs[pid];
		float * vs = prof_vs[pid];
		float vrel = vmin/amaxs[oi];
		// Don't assume that the profile is monotonic. This is expensive, since we have to loop
		// through lots of potentially tiny values in the wings of the profile to find the last
		// non-negligible point. Assuming that profiles tend to be repeated, I could precompute
		// the inds where profiles reach a set of predefined multiples of their peak height. Then
		// this loop could use that to reduce the loop range for the individual objects.
		int i = prof_info ? prof_info_lookup(prof_info, pid, vrel)-1 : n-1;
		for(; i > 0 && fabsf(vs[i]) < vrel; i--);
		rmaxs[oi] = rs[i];
		if(rmax > 0) rmaxs[oi] = fminf(rmaxs[oi], rmax);
	}
	return rmaxs;
}

// This old version did not support profiles that chance sign or start at zero
/*
float * measure_rmax(int nobj, float * amaxs, int * prof_ids, int * prof_ns, float ** prof_rs, float ** prof_vs, float vmin, float rmax) {
	float * rmaxs = calloc(nobj, sizeof(float));
	#pragma omp parallel for
	for(int oi = 0; oi < nobj; oi++) {
		int pid    = prof_ids[oi];
		int n      = prof_ns[pid];
		float * rs = prof_rs[pid];
		float * vs = prof_vs[pid];
		float vrel = vmin/amaxs[oi];
		if     (vs[0]   <  vrel) rmaxs[oi] = 0;
		else if(vs[n-1] >= vrel) rmaxs[oi] = rs[n-1];
		else {
			int i;
			for(i = n-1; i > 0 && vs[i] < vrel; i--);
			rmaxs[oi] = rs[i];
			if(rmax > 0) rmaxs[oi] = fminf(rmaxs[oi], rmax);
		}
	}
	return rmaxs;
}
*/


int assign_cells(
		int nobj,         // Number of objects
		float * obj_decs, // Object coordinates
		float * obj_ras,  //
		int   * obj_ys,   // Object pixel coordinates
		int   * obj_xs,   //
		float * rmaxs,    // Max relevant radius for each object
		int ny, int nx,   // Map dimensions
		int separable,    // Are ra/dec separable?
		float * pix_decs, // Coordinates of map pixels
		float * pix_ras,  //
		int csize,        // Cell size
		int **cell_nobj,  // Output parameter. Number of objects for each cell
		int ***cell_objs, // Output parameter. Ids of objects in each cell
		int ***cell_boxes // Output parameter. {y1,y2,x1,x1} for each cell.
	) {
	// 1. Allocate our cell lists
	int ncy   = (ny+csize-1)/csize;
	int ncx   = (nx+csize-1)/csize;
	int ncell = ncy*ncx;
	IntList ** cell_list = calloc(ncell, sizeof(IntList*));
	for(int ci = 0; ci < ncell; ci++)
		cell_list[ci] = intlist_new();
	// 2. For each object estimate its pixel bounding box, and turn that into a
	//    cell bounding box. We do this all at once so we can use openmp
	int * cellboxes = calloc(nobj*4, sizeof(int));
	#pragma omp parallel for
	for(int oi = 0; oi < nobj; oi++) {
		int pixbox[4];
		estimate_bounding_box(obj_ys[oi], obj_xs[oi], rmaxs[oi], ny, nx, separable, pix_decs, pix_ras, pixbox);
		// This also handles wrapping such that the start will always be
		// positive, and the end will be at most start beyond the end.
		// This means that the sloppy wrapping we do that doesn't know about
		// the real wrapping length of the sky will not cover any tile more than
		// once.
		pixbox2cellbox(pixbox, csize, ncy, ncx, cellboxes+4*oi);
		wrap_box(cellboxes+4*oi, ncy, ncx);
	}
	// 3. For each cell in each object's cell box, register the object in that cell.
	for(int oi = 0; oi < nobj; oi++) {
		int cy1 = cellboxes[4*oi+0], cy2 = cellboxes[4*oi+1];
		int cx1 = cellboxes[4*oi+2], cx2 = cellboxes[4*oi+3];
		int cy_wrap, cx_wrap;
		for(int cy = cy1; cy < cy2; cy++) {
			cy_wrap = cy >= ncy ? cy-ncy : cy;
			for(int cx = cx1; cx < cx2; cx++) {
				cx_wrap = cx >= ncx ? cx-ncx : cx;
				int ci = cy_wrap*ncx+cx_wrap;
				intlist_push(cell_list[ci], oi);
			}
		}
	}
	// 4. Measure the active cells
	IntList * active   = intlist_new();
	IntList * inactive = intlist_new();
	for(int ci = 0; ci < ncell; ci++)
		if(cell_list[ci]->n > 0)
			intlist_push(active, ci);
		else
			intlist_push(inactive, ci);
	int nactive = active->n;

	// 5. Transfer to output
	*cell_nobj  = calloc(nactive, sizeof(int));
	*cell_objs  = calloc(nactive, sizeof(int*));
	*cell_boxes = calloc(nactive, sizeof(int*));
	for(int ai = 0; ai < nactive; ai++) {
		int ci = active->vals[ai];
		(*cell_nobj)[ai] = cell_list[ci]->n;
		(*cell_objs)[ai] = cell_list[ci]->vals;
		int cy = ci/ncx, cx = ci%ncx;
		int y1 = cy*csize, y2 = (cy+1)*csize; if(y2>ny) y2 = ny;
		int x1 = cx*csize, x2 = (cx+1)*csize; if(x2>nx) x2 = nx;
		int * box = calloc(4, sizeof(int));
		box[0] = y1; box[1] = y2; box[2] = x1; box[3] = x2;
		(*cell_boxes)[ai] = box;
	}

	// Call intlist_free only on inactive cells, since we've given
	// away ownership of the values ofr the active ones
	for(int i = 0; i < inactive->n; i++)
		intlist_free(cell_list[inactive->vals[i]]);
	for(int i = 0; i < active->n; i++)
		free(cell_list[active->vals[i]]);
	intlist_free(active);
	intlist_free(inactive);
	free(cell_list);
	free(cellboxes);
	// Finally return the number of cells
	return nactive;
}

int mod(int a, int b) { int c = a % b; return c < 0 ? c+b : c; }
int floor_div(int a, int b) { int c = a / b; return c*b > a ? c-1 : c; }

// Convert y1 y2 x1 x2 -> cy1 cy2 cx1 cx2. The ranges are
// half-open like in python
void pixbox2cellbox(int * pixbox, int csize, int ncy, int ncx, int * cellbox) {
	// Go from raw pixel bounds to raw cell bounds
	int cy1 = floor_div(pixbox[0], csize), cy2 = floor_div(pixbox[1]-1, csize)+1;
	int cx1 = floor_div(pixbox[2], csize), cx2 = floor_div(pixbox[3]-1, csize)+1;
	// Put into output
	cellbox[0] = cy1; cellbox[1] = cy2;
	cellbox[2] = cx1; cellbox[3] = cx2;
}

// wrap an integer box such that the start point is >= 1, and
// the end point wraps no more than once around the sky.
void wrap_box(int * box, int ny, int nx) {
	int y1 = box[0], y2 = box[1], x1 = box[2], x2 = box[3];
	int h  = y2-y1, w = x2-x1;
	if(h > ny) h = ny;
	if(w > nx) w = nx;
	y1 = mod(y1, ny); y2 = y1+h;
	x1 = mod(x1, nx); x2 = x1+w;
	box[0] = y1; box[1] = y2; box[2] = x1; box[3] = x2;
}

void process_cell(
		int nobj,         // Number of objects in this cell
		int * objs,       // ids of those objects
		int * box,        // {y1,y2,x1,x2} pixel bounding box
		float * obj_decs, // [nobj_tot]. Coordinates of objects
		float * obj_ras,  // [nobj_tot]
		float ** amps,    // [ncomp][nobj_tot]
		int * prof_ids,   // Profile id for each object id
		int * prof_ns,    // Number of points for each profile id
		float ** prof_rs, // [nprof][prof_n]. R values in each profile
		float ** prof_vs, // [nprof][prof_n]. Profile values.
		int prof_equi,    // are profiles equi-spaced?
		int op,           // The operation to perform when merging object signals
		int ncomp, int ny, int nx,// Map dimensions
		int separable,    // Are ra/dec separable?
		int transpose,    // Whether to do the transpose operation: map -> amp
		float * pix_decs, // [nx] if seprable else [ny*nx]. Coordinates of objects
		float * pix_ras,  // [ny] if seprable else [ny*nx]
		float ** imap,    // [ncomp,ny*nx]. The input map
		float ** omap     // [ncomp,ny*nx]. The output map. Can be the same as the input map
	) {
	int y1 = box[0], y2 = box[1], x1 = box[2], x2 = box[3];
	int cny = y2-y1, cnx = x2-x1, npix = cny*cnx, ntot = ncomp*npix;
	// 1. Copy out the pixels
	float * cell_data = extract_map(imap, box, ncomp, ny, nx, nx, 1);
	float * cell_ras, * cell_decs;
	if(separable) {
		cell_decs = extract_coords(pix_decs, box, ny, nx, 1, 0);
		cell_ras  = extract_coords(pix_ras,  box, ny, nx, 0, 1);
	} else {
		cell_decs = extract_coords(pix_decs, box, ny, nx, nx, 1);
		cell_ras  = extract_coords(pix_ras,  box, ny, nx, nx, 1);
	}
	float * cell_work = calloc(ncomp*cny*cnx, sizeof(float));
	float * amp = calloc(ncomp, sizeof(float));
	// 2. Process each object
	if(transpose) {
		for(int oi = 0; oi < nobj; oi++) {
			int obj = objs[oi];
			int pid = prof_ids[obj];
			for(int ci = 0; ci < ncomp; ci++) amp[ci] = 0;
			// 3. Paint object onto work-space
			paint_object(obj_decs[obj], obj_ras[obj], amp, prof_ns[pid], prof_rs[pid], prof_vs[pid], prof_equi, ncomp, cny, cnx, transpose, cell_decs, cell_ras, cell_data);
			// 4. Update global amps
			for(int ci = 0; ci < ncomp; ci++)
				amps[ci][obj] += amp[ci];
		}
	} else {
		for(int oi = 0; oi < nobj; oi++) {
			int obj = objs[oi];
			for(int ci = 0; ci < ncomp; ci++)
				amp[ci] = amps[ci][obj];
			int pid = prof_ids[obj];
			// 3. Paint object onto work-space
			paint_object(obj_decs[obj], obj_ras[obj], amp, prof_ns[pid], prof_rs[pid], prof_vs[pid], prof_equi, ncomp, cny, cnx, transpose, cell_decs, cell_ras, cell_work);
			// 4. Merge work-space with cell data
			merge_cell(ntot, op, cell_work, cell_data);
		}
		// 5. Copy back into map
		insert_map(cell_data, omap, box, ncomp, ny, nx);
	}
	free(amp);
	free(cell_data);
	free(cell_decs);
	free(cell_ras);
	free(cell_work);
}

// Copy out a box from imap, returning it as omap. NB: No wrapping, but
// vales are clamped
float * extract_map(float ** imap, int * box, int ncomp, int iny, int inx, int ystep, int xstep) {
	int ny = box[1]-box[0], nx = box[3]-box[2];
	int npix = ny*nx, ntot = npix*ncomp;
	float * omap = calloc(ntot, sizeof(float));
	for(int c = 0; c < ncomp; c++) {
		float * im = imap[c];
		float * om = omap+c*npix;
		for(int oy = 0, iy = box[0]; oy < ny; oy++, iy++) {
			if(iy < 0 || iy >= iny) continue;
			for(int ox = 0, ix = box[2]; ox < nx; ox++, ix++) {
				if(ix < 0 || ix >= inx) continue;
				om[oy*nx+ox] = im[iy*ystep+ix*xstep];
			}
		}
	}
	return omap;
}

float * extract_coords(float * imap, int * box, int iny, int inx, int ystep, int xstep) { return extract_map(&imap, box, 1, iny, inx, ystep, xstep); }

void insert_map(float * imap, float ** omap, int * box, int ncomp, int ony, int onx) {
	int ny = box[1]-box[0], nx = box[3]-box[2];
	int npix = ny*nx, ntot = npix*ncomp;
	for(int c = 0; c < ncomp; c++) {
		float * im = imap+c*npix;
		float * om = omap[c];
		for(int iy = 0, oy = box[0]; iy < ny; iy++, oy++) {
			if(oy < 0 || oy >= ony) continue;
			for(int ix = 0, ox = box[2]; ix < nx; ix++, ox++) {
				if(ox < 0 || ox >= onx) continue;
				om[oy*onx+ox] = im[iy*nx+ix];
			}
		}
	}
}

void paint_object(
		float obj_dec,             // object coordinates
		float obj_ra,              //
		float * restrict amps,     // [ncomp], e.g. T, Q, U.
		int prof_n,                // number of sample points in profile
		float * restrict prof_rs,  // radial coordinate for each sample point
		float * restrict prof_vs,  // profile value for each sample point
		int prof_equi,             // are profiles equi-spaced?
		int ncomp, int ny, int nx, // cell dimensions
		int transpose,
		float * restrict pix_decs, // pixel coordinates
		float * restrict pix_ras,  //
		float * restrict map       // map to overwrite
	) {
	int npix = ny*nx;
	for(int y = 0; y < ny; y++) {
		for(int x = 0; x < nx; x++) {
			int pix    = y*nx+x;
			float r    = calc_dist(pix_decs[pix], pix_ras[pix], obj_dec, obj_ra);
			float prof = evaluate_profile(prof_n, prof_rs, prof_vs, r, prof_equi);
			for(int ci = 0; ci < ncomp; ci++) {
				if(transpose) amps[ci] += map[ci*npix+pix]*prof;
				else   map[ci*npix+pix] = amps[ci]*prof;
			}
		}
	}
}


void merge_cell(int n, int op, float * restrict source, float * restrict target) {
	switch(op) {
		case OP_ADD:
			for(int i = 0; i < n; i++)
				target[i] += source[i];
			break;
		case OP_MAX:
			for(int i = 0; i < n; i++)
				if(source[i] > target[i]) target[i] = source[i];
			break;
		case OP_MIN:
			for(int i = 0; i < n; i++)
				if(source[i] < target[i]) target[i] = source[i];
			break;
	}
}

float evaluate_profile(int n, float * rs, float * vs, float r, int equi) {
	int i1 = equi ? equi_search(n, rs, r) : binary_search(n, rs, r);
	if(i1 < 0) return vs[0];
	int i2 = i1+1;
	if(i2 >= n) return 0;
	float x = (r-rs[i1])/(rs[i2]-rs[i1]);
	return vs[i1] + (vs[i2]-vs[i1])*x;
}

// Returns i such that rs[i] <= r < rs[i+1]. rs must be sorted.
// This function is responsible for 46% of the total run time.
int binary_search(int n, float * rs, float r) {
	if(r <  rs[0])   return -1;
	if(r >= rs[n-1]) return  n;
	int a = 0, b = n-1;
	// will maintain r inside interval rs[a]:rs[b]
	while(b > a+1) {
		int c = (a+b)/2;
		if(r < rs[c]) b = c;
		else          a = c;
	}
	return a;
}
int equi_search(int n, float * rs, float r) {
	int i = (int)(r/rs[1]);
	if(i < 0) return -1;
	if(r >= n) return n;
	return i;
}

// Compute angular distance using vincenty formula. Quite heavy, but
// accurate at all distances. Can be sped up by precmputing cos(dec)
// and sin(dec). Might be worth it if this is the bottleneck.
float calc_dist(float dec1, float ra1, float dec2, float ra2) {
	float cos_dec1 = cos(dec1);
	float sin_dec1 = sin(dec1);
	float cos_dec2 = cos(dec2);
	float sin_dec2 = sin(dec2);
	float dra = ra2 - ra1;
	float cos_dra = cos(dra);
	float sin_dra = sin(dra);
	float y1 = cos_dec1*sin_dra;
	float y2 = cos_dec2*sin_dec1-sin_dec2*cos_dec1*cos_dra;
	float y  = sqrt(y1*y1+y2*y2);
	float x  = sin_dec2*sin_dec1 + cos_dec2*cos_dec1*cos_dra;
	float d  = atan2(y,x);
	return d;
}

float calc_grad(int i, int n, int s, float * v) {
	float dv, di;
	// Handle edge cases
	if     (i <= 0  ) { dv = v[s      ]-v[0      ]; di = 1; }
	else if(i >= n-1) { dv = v[s*(n-1)]-v[s*(n-2)]; di = 1; }
	else              { dv = v[s*(i+1)]-v[s*(i-1)]; di = 2; }
	// Handle angle cut
	dv = fmod(dv + M_PI, 2*M_PI) - M_PI;
	return dv/di;
}

// This function looks slow. Can be sped up by precomputing gradients.
// A coarse grid should suffice, just make sure to include the edges of
// the map
void calc_pix_shape_general(int y, int x, int ny, int nx, float * pix_decs, float * pix_ras, float * ysize, float * xsize) {
	y = y < 0 ? 0 : y >= ny ? ny-1 : y;
	x = x < 0 ? 0 : x >= nx ? nx-1 : x;
	float ddec_dy = calc_grad(y, ny, nx, pix_decs+x);
	float ddec_dx = calc_grad(x, nx,  1, pix_decs+nx*y);
	float dra_dy  = calc_grad(y, ny, nx, pix_ras+x);
	float dra_dx  = calc_grad(x, nx,  1, pix_ras+nx*y);
	float c       = cos(pix_decs[y*nx+x]);
	*ysize = sqrt((c*dra_dy)*(c*dra_dy)+ddec_dy*ddec_dy);
	*xsize = sqrt((c*dra_dx)*(c*dra_dx)+ddec_dx*ddec_dx);
}

void calc_pix_shape_separable(int y, int x, int ny, int nx, float * pix_decs, float * pix_ras, float * ysize, float * xsize) {
	y = y < 0 ? 0 : y >= ny ? ny-1 : y;
	x = x < 0 ? 0 : x >= nx ? nx-1 : x;
	float ddec_dy = calc_grad(y, ny,  1, pix_decs);
	float dra_dx  = calc_grad(x, nx,  1, pix_ras);
	float c       = cos(pix_decs[y]);
	*ysize        = fabsf(ddec_dy);
	*xsize        = fabsf(dra_dx*c);
}

void calc_pix_shape(int y, int x, int ny, int nx, int separable, float * pix_decs, float * pix_ras, float * ysize, float * xsize) {
	if(separable) return calc_pix_shape_separable(y, x, ny, nx, pix_decs, pix_ras, ysize, xsize);
	else          return calc_pix_shape_general  (y, x, ny, nx, pix_decs, pix_ras, ysize, xsize);
}

void estimate_bounding_box(
		int   obj_y,      // object pixel coordinates
		int   obj_x,      //
		float rmax,       // max relevant radius for object
		int ny, int nx,   // map dimensions
		int separable,    // are ra/dec separable?
		float * pix_decs, // coordinates of map pixels
		float * pix_ras,  //
		int * box         // {y1,y2,x1,x2} in pixels.
	) {
	// 1. Find the height and width of the object's pixel
	float dy0, dx0;
	calc_pix_shape(obj_y, obj_x, ny, nx, separable, pix_decs, pix_ras, &dy0, &dx0);
	// 2. Use this to define a preliminary rectangle
	int Dy = (int)fabsf(rmax/dy0)+1;
	int Dx = (int)fabsf(rmax/dx0)+1;
	// 3. and visit its four corners, measuring the smallest dy
	//    and dx for all of them
	float dy = dy0, dx = dx0;
	for(int oy = -1; oy <= 1; oy += 2)
	for(int ox = -1; ox <= 1; ox += 2) {
		calc_pix_shape(obj_y+Dy*oy, obj_x+Dx*ox, ny, nx, separable, pix_decs, pix_ras, &dy0, &dx0);
		if(dy0 < dy) dy = dy0;
		if(dx0 < dx) dx = dx0;
	}
	float tol = fmaxf(fmaxf(dy, dx)*1e-6, 1e-12);
	dy = fmaxf(dy, tol);
	dx = fmaxf(dx, tol);
	// 4. Use this to define a final rectangle
	Dy = (int)(rmax/dy)+1;
	Dx = (int)(rmax/dx)+1;
	box[0] = obj_y - Dy;
	box[1] = obj_y + Dy+1;
	box[2] = obj_x - Dx;
	box[3] = obj_x + Dx+1;
}

IntList * intlist_new() {
	IntList * v = malloc(sizeof(IntList));
	v->n   = 0;
	v->cap = 64;
	v->vals= malloc((long)v->cap*sizeof(int));
	return v;
}
void intlist_push(IntList * v, int val) {
	if(v->n >= v->cap) {
		v->cap *= 2;
		v->vals = realloc(v->vals, (long)v->cap*sizeof(int));
	}
	v->vals[v->n++] = val;
}
void intlist_free(IntList * v) { free(v->vals); free(v); }
void intlist_swap(IntList ** a, IntList ** b) { IntList * tmp = *a; *a = *b; *b = tmp; }


// This is surprisingly verbose...
float *** calloc_pppf(int n1, int n2, int n3) {
	float *** a = calloc(n1, sizeof(float**));
	for(int i1 = 0; i1 < n1; i1++) {
		a[i1] = calloc(n2, sizeof(float*));
		for(int i2 = 0; i2 < n2; i2++)
			a[i1][i2] = calloc(n3, sizeof(float));
	}
	return a;
}

void free_pppf(float *** a, int n1, int n2) {
	for(int i1 = 0; i1 < n1; i1++) {
		a[i1] = calloc(n2, sizeof(float*));
		for(int i2 = 0; i2 < n2; i2++)
			free(a[i1][i2]);
		free(a[i1]);
	}
	free(a);
}

void reduce_amps(float *** work_amps, float ** amps, int nwork, int ncomp, int nobj) {
	for(int w = 0; w < nwork; w++) {
		for(int c = 0; c < ncomp; c++) {
			for(int o = 0; o < nobj; o++)
				amps[c][o] += work_amps[w][c][o];
			}
	}
}

// This is equivalent to i%n, but should be faster for the common case where
// i is either in range or just outside
int wrap(int i, int n) {
	while(i <  0) i += n;
	while(i >= n) i -= n;
	return i;
}

// Radial binning code. Has some things in common with sim_objects, but not enough to
// be part of that function
void radial_sum(
		int nobj,         // Number of objects
		float * obj_decs, // [nobj]. Coordinates of objects
		float * obj_ras,  // [nobj]
		int   * obj_ys,   // [nobj]. Pixel coordinates of objects. Theoretically redundant,
		int   * obj_xs,   // [nobj], but useful in practice since we don't have the wcs here.
		int     nbin,     // Number of radial bins
		float * rs,       // [nbin+1]. The bin edges. Minimum length 2. First must be 0 in equi
		int   equi,       // are bins equi-spaced?
		int ncomp, int ny, int nx,// Map dimensions
		int separable,    // Are ra/dec separable?
		float *  pix_decs,// [ny*nx]
		float *  pix_ras, // [ny*nx]
		float ** imap,    // [ncomp,ny*nx]. The input map
		float *** bins,   // [nobj,ncomp,nbin]. The values in each bin (output)
		double * times    // Time taken in the different steps
	) {
	// 1. Figure out how far away we need to consider each object
	float rmax    = rs[nbin-1];
	// 2. Loop over each object
	double t1     = wall_time();
	#pragma omp parallel for
	for(int oi = 0; oi < nobj; oi++) {
		// 3. Figure out which pixels are relevant
		int pixbox[4];
		estimate_bounding_box(obj_ys[oi], obj_xs[oi], rmax, ny, nx, separable, pix_decs, pix_ras, pixbox);
		wrap_box(pixbox, ny, nx);
		for(int y_ = pixbox[0]; y_ < pixbox[1]; y_++) {
			// We only need to handle upper wrapping due to wrap_box
			int y = y_ >= ny ? y_-ny : y_;
			for(int x_ = pixbox[2]; x_ < pixbox[3]; x_++) {
				int x = x_ >= nx ? x_-nx : x_;
				float ra, dec;
				if(separable) { dec = pix_decs[y];      ra = pix_ras[x];      }
				else          { dec = pix_decs[y*nx+x]; ra = pix_ras[y*nx+x]; }
				float r = calc_dist(dec, ra, obj_decs[oi], obj_ras[oi]);
				int i   = equi ? equi_search(nbin, rs, r) : binary_search(nbin, rs, r);
				if(i < 0 || i >= nbin) continue;
				for(int ci = 0; ci < ncomp; ci++)
					bins[oi][ci][i] += imap[ci][y*nx+x];
			}
		}
	}
	double t2 = wall_time();
	times[0] = t2-t1;
}
