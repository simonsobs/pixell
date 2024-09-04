#include "cmisc_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }

// We treat the double arrays as real here due to cython issues. Maybe I should find
// a better glue than cython. It's fragile and requires .c, .h, .pxd and .pyx files
// for everything

void alm2cl_sp(int lmax, int mmax, int64_t * mstart, float * alm1, float * alm2, float * cl) {
	int nthread = omp_get_max_threads();
	int nl      = lmax+1;
	float * buf = calloc(nthread*nl, sizeof(float));
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		if(id == 0) {
			for(int l = 0; l <= lmax; l++) {
				int64_t i = mstart[0]*2 + l*2;
				buf[nl*id+l] = alm1[i]*alm2[i]/2;
			}
		}
		#pragma omp for schedule(dynamic)
		for(int m = 1; m <= mmax; m++) {
			for(int l = m; l <= lmax; l++) {
				int64_t i = mstart[m]*2 + l*2;
				buf[nl*id+l] += alm1[i]*alm2[i] + alm1[i+1]*alm2[i+1];
			}
		}
		#pragma omp barrier
		#pragma omp for
		for(int l = 0; l < nl; l++) {
			cl[l] = 0;
			for(int i = 0; i < nthread; i++)
				cl[l] += buf[nl*i+l];
			cl[l] *= 2.0/(2*l+1);
		}
	}
	free(buf);
}

void alm2cl_dp(int lmax, int mmax, int64_t * mstart, double * alm1, double * alm2, double * cl) {
	int nthread = omp_get_max_threads();
	int nl      = lmax+1;
	double * buf = calloc(nthread*nl, sizeof(double));
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		if(id == 0) {
			for(int l = 0; l <= lmax; l++) {
				int64_t i = mstart[0]*2 + l*2;
				buf[nl*id+l] = alm1[i]*alm2[i]/2;
			}
		}
		#pragma omp for schedule(dynamic)
		for(int m = 1; m <= mmax; m++) {
			for(int l = m; l <= lmax; l++) {
				int64_t i = mstart[m]*2 + l*2;
				buf[nl*id+l] += alm1[i]*alm2[i] + alm1[i+1]*alm2[i+1];
			}
		}
		#pragma omp barrier
		#pragma omp for
		for(int l = 0; l < nl; l++) {
			cl[l] = 0;
			for(int i = 0; i < nthread; i++)
				cl[l] += buf[nl*i+l];
			cl[l] *= 2.0/(2*l+1);
		}
	}
	free(buf);
}

// Transpose a scalar alm into an output array
// The purpose of this function is to let us generate random numbers
// in m-major order, which is what's natural for numpy, and then turn
// them into l-major order afterwards.
void transpose_alm_dp(int lmax, int mmax, int64_t * mstart, double * ialm, double * oalm) {
	int ol, om;
	int64_t ii, oi;
	om = ol = 0;
	// This loop is slow because it has data dependencies. Would be nice
	// if we could calculate om,ol directly from im,il.
	for(int im = 0; im <= mmax; im++) {
		for(int il = im; il <= lmax; il++) {
			ii = mstart[im]+il;
			oi = mstart[om]+ol;
			oalm[2*oi+0] = ialm[2*ii+0];
			oalm[2*oi+1] = ialm[2*ii+1];
			om++;
			if(om > mmax || om > ol) {
				ol++;
				om = 0;
			}
		}
	}
}

void transpose_alm_sp(int lmax, int mmax, int64_t * mstart, float * ialm, float * oalm) {
	int ol, om;
	int64_t ii, oi;
	om = ol = 0;
	// This loop is slow because it has data dependencies. Would be nice
	// if we could calculate om,ol directly from im,il.
	for(int im = 0; im <= mmax; im++) {
		for(int il = im; il <= lmax; il++) {
			ii = mstart[im]+il;
			oi = mstart[om]+ol;
			oalm[2*oi+0] = ialm[2*ii+0];
			oalm[2*oi+1] = ialm[2*ii+1];
			om++;
			if(om > mmax || om > ol) {
				ol++;
				om = 0;
			}
		}
	}
}

// Multiply a scalar alm by a scalar function of l
void lmul_dp(int lmax, int mmax, int64_t * mstart, double * alm, int lfmax, const double * lfun) {
	#pragma omp parallel for
	for(int m = 0; m <= mmax; m++) {
		for(int l = m; l <= lmax; l++) {
			int64_t i = mstart[m]+l;
			double  v = l <= lfmax ? lfun[l] : 0;
			alm[2*i+0] *= v;
			alm[2*i+1] *= v;
		}
	}
}

// Multiply a scalar alm by a scalar function of l
void lmul_sp(int lmax, int mmax, int64_t * mstart, float * alm, int lfmax, const float * lfun) {
	#pragma omp parallel for
	for(int m = 0; m <= mmax; m++) {
		for(int l = m; l <= lmax; l++) {
			int64_t i = mstart[m]+l;
			float   v = l <= lfmax ? lfun[l] : 0;
			alm[2*i+0] *= v;
			alm[2*i+1] *= v;
		}
	}
}

// Multiply matrix [N,M,l] by alm [M,nalm] producing oalm[N,nalm]
void lmatmul_dp(int N, int M, int lmax, int mmax, int64_t * mstart, double ** alm, int lfmax, double ** lmat, double ** oalm) {
	int leff = min(lmax, lfmax);
	#pragma omp parallel
	{
		// We need these arrays so that we can store a whole
		// row of the output. We can't write the row to oalm
		// before the whole entry is done because we could be
		// running in-place.
		double * vreal = calloc(N, sizeof(double));
		double * vimag = calloc(N, sizeof(double));
		#pragma omp for
		for(int m = 0; m <= mmax; m++) {
			for(int l = m; l <= leff; l++) {
				int64_t i = mstart[m]+l;
				// Zero out the work arrays
				for(int r = 0; r < N; r++) { vreal[r] = vimag[r] = 0; }
				// rc,c->r
				for(int r = 0; r < N; r++) {
					for(int c = 0; c < M; c++) {
						vreal[r] += lmat[r*M+c][l] * alm[c][2*i+0];
						vimag[r] += lmat[r*M+c][l] * alm[c][2*i+1];
					}
				}
				// Copy into output array
				for(int r = 0; r < N; r++) {
					oalm[r][2*i+0] = vreal[r];
					oalm[r][2*i+1] = vimag[r];
				}
			}
		}
		free(vreal);
		free(vimag);
	}
	// Zero out parts beyond lfmax
	#pragma omp parallel for
	for(int r = 0; r < N; r++) {
		for(int m = 0; m <= mmax; m++) {
			for(int l = max(m,leff+1); l <= lmax; l++) {
				int64_t i = mstart[m]+l;
				oalm[r][2*i+0] = oalm[r][2*i+1] = 0;
			}
		}
	}
}

// Multiply matrix [N,M,l] by alm [M,nalm] producing oalm[N,nalm]
void lmatmul_sp(int N, int M, int lmax, int mmax, int64_t * mstart, float ** alm, int lfmax, float ** lmat, float ** oalm) {
	int leff = min(lmax,lfmax);
	#pragma omp parallel
	{
		// We need these arrays so that we can store a whole
		// row of the output. We can't write the row to oalm
		// before the whole entry is done because we could be
		// running in-place.
		float * vreal = calloc(N, sizeof(float));
		float * vimag = calloc(N, sizeof(float));
		#pragma omp for
		for(int m = 0; m <= mmax; m++) {
			for(int l = m; l <= lfmax; l++) {
				int64_t i = mstart[m]+l;
				// Zero out the work arrays
				for(int r = 0; r < N; r++) { vreal[r] = vimag[r] = 0; }
				// rc,c->r
				for(int r = 0; r < N; r++) {
					for(int c = 0; c < M; c++) {
						vreal[r] += lmat[r*M+c][l] * alm[c][2*i+0];
						vimag[r] += lmat[r*M+c][l] * alm[c][2*i+1];
					}
				}
				// Copy into output array
				for(int r = 0; r < N; r++) {
					oalm[r][2*i+0] = vreal[r];
					oalm[r][2*i+1] = vimag[r];
				}
			}
		}
		free(vreal);
		free(vimag);
	}
	// Zero out parts beyond lfmax
	#pragma omp parallel for
	for(int r = 0; r < N; r++) {
		for(int m = 0; m <= mmax; m++) {
			for(int l = max(m,leff+1); l <= lmax; l++) {
				int64_t i = mstart[m]+l;
				oalm[r][2*i+0] = oalm[r][2*i+1] = 0;
			}
		}
	}
}

// Copy a scalar alm from one layout to another. Parts with no
// data are not overwritten
void transfer_alm_dp(int lmax1, int mmax1, int64_t * mstart1, double * alm1, int lmax2, int mmax2, int64_t * mstart2, double * alm2) {
	int lmax = min(lmax1,lmax2);
	int mmax = min(mmax1,mmax2);
	#pragma omp parallel for
	for(int m = 0; m <= mmax; m++) {
		for(int l = m; l <= lmax; l++) {
			int64_t i1 = mstart1[m]+l;
			int64_t i2 = mstart2[m]+l;
			alm2[2*i2+0] = alm1[2*i1+0];
			alm2[2*i2+1] = alm1[2*i1+1];
		}
	}
}

void transfer_alm_sp(int lmax1, int mmax1, int64_t * mstart1, float * alm1, int lmax2, int mmax2, int64_t * mstart2, float * alm2) {
	int lmax = min(lmax1,lmax2);
	int mmax = min(mmax1,mmax2);
	#pragma omp parallel for
	for(int m = 0; m <= mmax; m++) {
		for(int l = m; l <= lmax; l++) {
			int64_t i1 = mstart1[m]+l;
			int64_t i2 = mstart2[m]+l;
			alm2[2*i2+0] = alm1[2*i1+0];
			alm2[2*i2+1] = alm1[2*i1+1];
		}
	}
}

// wcs acceleration
#define DEG (M_PI/180)

// I pass the wcs information as individual doubles to avoid having to construct
// numpy arrays on the python side. All of these have the arguments in the same
// order, regardless of which way they go, so they can be defined in a macro
//
// We only implement plain spherical coordinates here - the final coordinate
// rotation is missing. For cylindrical coordinates this means that we only
// support dec0 = 0 - the result will be wrong for other values
#define wcsdef(name) \
void name(int64_t n, double * restrict dec, double * restrict ra, \
		double * restrict y, double * restrict x, \
		double crval0, double crval1, double cdelt0, double cdelt1, \
		double crpix0, double crpix1) { \
	double ra0 = crval0*DEG, dec0 = crval1*DEG; \
	double dra = cdelt0*DEG, ddec = cdelt1*DEG; \
	double x0  = crpix0-1,   y0   = crpix1-1; \
	_Pragma("omp parallel for") \
	for(int64_t i = 0; i < n; i++) {
#define wcsend } \
}

wcsdef(wcs_car_sky2pix)
x[i] = (ra [i]-ra0 )/dra +x0;
y[i] = (dec[i]-dec0)/ddec+y0; // dec0 should be zero
wcsend

wcsdef(wcs_car_pix2sky)
ra [i] = (x[i]-x0)*dra +ra0;
dec[i] = (y[i]-y0)*ddec+dec0; // dec0 should be zero
wcsend

wcsdef(wcs_cea_sky2pix)
x[i] = (ra [i]-ra0 )/dra +x0;
y[i] = sin(dec[i])/ddec  +y0;
(void)dec0; // mark dec0 as explicitly unused
wcsend

wcsdef(wcs_cea_pix2sky)
ra [i] = (x[i]-x0)*dra +ra0;
dec[i] = asin((y[i]-y0)*ddec);
(void)dec0; // mark dec0 as explicitly unused
wcsend

void rewind_inplace(int64_t n, double * vals, double period, double ref) {
	_Pragma("omp parallel for")
	for(int64_t i = 0; i < n; i++)
		vals[i] = fmod((vals[i]+ref),period)-ref;
}
