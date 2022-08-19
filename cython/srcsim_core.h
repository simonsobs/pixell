enum { OP_ADD, OP_MAX, OP_MIN };

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
		int transpose,    // Whether to do the transpose operation: map -> amp
		float * pix_decs, // [ny] if separable else [ny*nx]
		float * pix_ras,  // [nx] if separable else [ny*nx]
		float ** imap,    // [ncomp,ny*nx]. The input map
		float ** omap,    // [ncomp,ny*nx]. The output map. Can be the same as the input map.
		int csize,        // cell size. These are processed in parallel. E.g. 32 for 32x32 cells
		double * times    // benchmarking
	);


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
	);
