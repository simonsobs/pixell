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

enum { OP_COPY, OP_ADD, OP_MAX, OP_MIN };

void sim_objects(
		int nobj,         // Number of objects
		float * obj_ras,  // [nobj]. Coordinates of objects
		float * obj_decs, // [nobj]
		int   * obj_xs,   // [nobj]. Pixel coordinates of objects. Theoretically redundant,
		int   * obj_ys,   // [nobj], but useful in practice since we don't have the wcs here.
		float ** amps,    // [ncomp][nobj]. Peak amplitude. comp = stokes for example
		int nprof,        // Number of unique profiles
		int * prof_ids,   // Profile id for each object
		int * prof_ns,    // [nprof]. Samples in each profile
		float ** prof_rs, // [nprof][prof_n]. R values in each profile
		float ** prof_vs, // [nprof][prof_n]. Profile values.
		float tol,        // Lowest value to simulate, in amplitude units = map units
		int op,           // The operation to perform when merging object signals
		int ncomp, int ny, int nx,// Map dimensions
		float *  pix_ras, // [ny*nx]
		float *  pix_decs,// [ny*nx]
		float ** imap,    // [ncomp,ny*nx]. The input map
		float ** omap,    // [ncomp,ny*nx]. The output map. Can be the same as the input map.
		int csize,        // cell size. These are processed in parallel. E.g. 32 for 32x32 cells
	) {
	// 1. Measure the maximum radius for each source
	float * amaxs = measure_amax(nobj, ncomp, amps);
	float * rmaxs = measure_rmax(nobj, amaxs, prof_ids, prof_ns, prof_rs, prof_vs, tol);
	free(amaxs);
	// 2. Find which objects are relevant for which cells
	int *cell_nobj, **cell_objs, **cell_boxes; // [ncell], [ncell][objs] and [ncell][{y1,y2,x1,x2}]
	int ncell = assign_cells(nobj, obj_ras, obj_decs, obj_xs, obj_ys, rmaxs, ny, nx, pix_ras, pix_decs, csize, &cell_nobj, &cell_objs, &cell_boxes);
	// 3. Process each cell
	#pragma omp parallel for
	for(int ci = 0; ci < ncell; ci++)
		if(!(op == OP_ADD && cell_nobj[ci] == 0))
			process_cell(cell_nobj[ci], cell_objs[ci], cell_boxes[ci], obj_ras, obj_decs, amps, prof_ids, prof_ns, prof_rs, prof_vs, op, ncomp, pix_ras, pix_decs, imap, omap);
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
		float amax = amps[0][i];
		for(int c = 1; c < ncomp; c++)
			if(amps[c][i] > amax)
				amax = amps[c][i];
		amaxs[i] = amax;
	}
	return amaxs;
}

float * measure_rmax(int nobj, float * amaxs, int * prof_ids, int * prof_ns, float ** prof_rs, float ** prof_vs, float tol) {
	float * rmaxs = calloc(nobj, sizeof(float));
	#pragma omp parallel for
	for(int oi = 0; oi < nobj; oi++) {
		int pid    = prof_ids[oi];
		float * rs = prof_rs[pid];
		float * vs = prof_vs[pid];
		if     (vs[0]   <  tol) rmaxs[oi] = 0;
		else if(vs[n-1] >= tol) rmaxs[oi] = rs[n-1];
		else {
			int i;
			for(i = n-1; i > 0 && vs[i] < tol; i++);
			rmaxs[oi] = rs[i];
		}
	}
	return rmaxs;
}

int assign_cells(
		int nobj,         // Number of objects
		float * obj_ras,  // Object coordinates
		float * obj_decs, //
		float * obj_xs,   // Object pixel coordinates
		float * obj_ys,   //
		float * rmaxs,    // Max relevant radius for each object
		int ny, nx,       // Map dimensions
		float * pix_ras,  // Coordinates of map pixels
		float * pix_decs, //
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
		cell_list[i] = intlist_new();
	// 2. For each object estimate its pixel bounding box, and turn that into a
	//    cell bounding box. We do this all at once so we can use openmp
	int pixbox[4];
	int * cellboxes = calloc(nobj*4, sizeof(int));
	#pragma omp parallel for
	for(int oi = 0; oi < nobj; oi++) {
		estimate_bounding_box(obj_xs[oi], obj_ys[oi], rmaxs[oi], ny, nx, pix_ras, pix_decs, pixbox);
		// This also handles wrapping such that the start will always be
		// positive, and the end will be at most start beyond the end.
		// This means that the sloppy wrapping we do that doesn't know about
		// the real wrapping length of the sky will not cover any tile more than
		// once.
		pixbox2cellbox(pixbox, csize, ncy, ncx, cellboxes+4*oi);
	}
	// 3. For each cell in each object's cell box, register the object in that cell.
	for(int oi = 0; oi < nobj; oi++) {
		int cy1 = cellboxes[4*oi+0], cy2 = cellboxes[4*oi+1];
		int cx1 = cellboxes[4*oi+2], cx2 = cellboxes[4*oi+3];
		for(int cy = cy1; cy <= cy2; cy++) {
			if(cy >= ncy) cy -= ncy;
			for(int cx = cx1; cx <= cx2; cx++) {
				if(cx >= ncx) cx -= ncx;
				int ci = cy*ncx+cx;
				intlist_push(cell_list[ci], oi);
			}
		}
	}
	// 4. Transfer to output
	*cell_nobj  = calloc(ncell, int);
	*cell_objs  = calloc(ncell, int*);
	*cell_boxes = calloc(ncell, int*);
	for(int ci = 0; ci < ncell; ci++) {
		(*cell_nobj)[ci] = (*cell_list)[ci].n;
		(*cell_objs)[ci] = (*cell_list)[ci].vals;
		int cy = ci/ncx, cx = ci%ncx;
		int y1 = cy*csize, y2 = (cy+1)*csize; if(y2>ny) y2 = ny;
		int x1 = cx*csize, x2 = (cx+1)*csize; if(x2>nx) x2 = nx;
		int * box = calloc(4, sizeo(int));
		box[0] = y1; box[1] = y2; box[2] = x1; box[3] = x2;
		(*cell_boxes)[ci] = box;
	}
	// We don't call inlist_free here, because we've given away the
	// vals;
	for(int ci = 0; ci < ncell; ci++)
		free(cell_list[ci]);
	free(cell_list);
	free(cellboxes);
	// Finally return the number of cells
	return ncell;
}

void process_cell(
		nobj,             // Number of objects in this cell
		int * objs,       // ids of those objects
		int * box,        // {y1,y2,x1,x2} pixel bounding box
		float * obj_ras,  // [nobj_tot]. Coordinates of objects
		float * obj_decs, // [nobj_tot]
		float ** amps,    // [ncomp][nobj_tot]
		int * prof_ids,   // Profile id for each object id
		int * prof_ns,    // Number of points for each profile id
		float ** prof_rs, // [nprof][prof_n]. R values in each profile
		float ** prof_vs, // [nprof][prof_n]. Profile values.
		int op,           // The operation to perform when merging object signals
		int ncomp,        // Number of components
		float * pix_ras,  // [ny*nx]. Coordinates of objects
		float * pix_decs, // [ny*nx]
		float ** imap     // [ncomp,ny*nx]. The input map
		float ** omap     // [ncomp,ny*nx]. The output map. Can be the same as the input map
	) {
	int y1 = box[0], y2 = box[1], x1 = box[2], x2 = box[3];
	int ny = y2-y1, nx = x2-x1, npix = ny*nx, ntot = ncomp*npix;
	// 1. Copy out the pixels
	float * cell_data = extract_map(imap, box, ncomp);
	float * cell_ras  = extract_coords(pix_ras,  box);
	float * cell_decs = extract_coords(pix_decs, box);
	float * cell_work = calloc(ncomp*ny*nx, sizeof(float));
	float * amp = calloc(ncomp, sizeof(float));
	// 2. Process each object
	for(int oi = 0; oi < nobj; oi++) {
		int obj = objs[oi];
		for(int ci = 0; ci < ncomp; ci++)
			amp[ci] = amps[ci,oi];
		int pid = prof_ids[obj];
		// 3. Paint object onto work-space
		paint_object(obj_ras[obj], obj_decs[obj], amp, prof_ns[pid], prof_rs[pid], prof_vs[pid], cell_ras, cell_decs, cell_work);
		// 4. Merge work-space with cell data
		merge_cell(ntot, op, cell_work, cell_data);
	}
	// 5. Copy back into map
	insert_map(cell_data, omap, box, ncomp);
	free(amp);
	free(cell_data);
	free(cell_ras);
	free(cell_decs);
	free(cell_work);
}

void paint_object(
		float obj_ra,     // object coordinates
		float obj_dec,    //
		float * amps,     // [ncomp], e.g. T, Q, U.
		int prof_n,       // number of sample points in profile
		float * prof_rs,  // radial coordinate for each sample point
		float * prof_vs,  // profile value for each sample point
		int ncomp, int ny, int nx,// cell dimensions
		float * pix_ras,  // pixel coordinates
		float * pix_decs, //
		float * map       // map to overwrite
	) {
	int npix = ny*nx;
	for(int y = 0; y < ny; y++) {
		for(int x = 0; x < nx; x++) {
			int pix    = y*nx+x;
			float ra   = pix_ras [pix];
			float dec  = pix_decs[pix];
			float r    = calc_dist(ra, dec, obj_ra, obj_dec);
			float prof = evaluate_profile(prof_n, prof_rs, prof_vs, r);
			for(int ci = 0; ci < ncomp; ci++)
				map[ci*npix+pix] = amps[ci]*prof;
		}
	}
}

void merge_cell(int n, int op, float * source, float * target) {
	switch(op) {
		case OP_COPY:
			memcpy(target, source, n*sizeof(int));
			break;
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

float evaluate_profile(int n, float * rs, float * vs, float r) {
	int i1 = binary_search(n, rs, r)
	if(i1 < 0) return vs[0];
	int i2 = i1+1;
	if(i2 >= n) return vs[n-1];
	float x = (r-rs[i1])/(rs[i2]-rs[i1]);
	return vs[i1] + (vs[i2]-vs[i1])*x;
}

// Returns i such that rs[i] < r <= rs[i+1]. rs must be sorted.
int binary_search(int n, float rs, float r) {
	if(r <= rs[0])   return -1;
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

// Compute angular distance using vincenty formula. Quite heavy, but
// accurate at all distances. Can be sped up by precmputing cos(dec)
// and sin(dec). Might be worth it if this is the bottleneck.
float calc_dist(float ra1, float dec1, float ra2, float dec2) {
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
void calc_pix_shape(int y, int x, int ny, int nx, float * pix_ras, float * pix_decs, float * ysize, float * xsize) {
	y = y < 0 ? 0 : y >= ny ? ny : y;
	x = x < 0 ? 0 : x >= nx ? nx : x;
	float dra_dy  = calc_grad(y, ny, nx, pix_ras+x);
	float dra_dx  = calc_grad(x, nx,  1, pix_ras+nx*y);
	float ddec_dy = calc_grad(y, ny, nx, pix_decs+x);
	float ddec_dx = calc_grad(x, nx,  1, pix_decs+nx*y);
	float c       = cos(pix_decs[y*nx+x]);
	*ysize = sqrt((c*dra_dy)*(c*dra_dy)+ddec_dy*ddec_dy);
	*xsize = sqrt((c*dra_dx)*(c*dra_dx)+ddec_dx*ddec_dx);
}

void estimate_bounding_box(
		int   obj_x,      // object pixel coordinates
		int   obj_y,      //
		float rmax,       // max relevant radius for object
		int ny, nx,       // map dimensions
		float * pix_ras,  // coordinates of map pixels
		float * pix decs, //
		int * box         // {y1,y2,x1,x2} in pixels.
	) {
	// 1. Find the height and width of the object's pixel
	float dy0, dx0;
	calc_grad(obj_y, obj_x, ny, nx, pix_decs, &dy0, &dx0);
	// 2. Use this to define a preliminary rectangle
	int Dy = (int)fabsf(fminf(rmax/dy,M_PI))+1;
	int Dx = (int)fabsf(fminf(rmax/dx,M_PI))+1;
	// 3. and visit its four corners, measuring the smallest dy
	//    and dx for all of them
	float dy = dy0, dx = dx0;
	for(int oy = -1; oy <= 1; oy += 2)
	for(int ox = -1; ox <= 1; ox += 2) {
		calc_grad(obj_y+Dy*oy, obj_x+Dy*ox, ny, nx, pix_decs, &dy0, &dx0);
		if(dy0 > dy) dy = dy0;
		if(dx0 > dx) dx = dx0;
	}
	// 4. Use this to define a final rectangle
	int Dy = (int)fabsf(fminf(rmax/dy,M_PI))+1;
	int Dx = (int)fabsf(fminf(rmax/dx,M_PI))+1;
	box[0] = obj_y - Dy;
	box[1] = obj_y + Dy+1;
	box[2] = obj_x - Dx;
	box[3] = obj_x + Dx+1;
}


typedef struct { int n, cap; int * vals; } IntList;
IntList * intlist_new() {
	IntList * v = malloc(sizeof(IntList));
	v->n   = 0;
	v->cap = 1024;
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
