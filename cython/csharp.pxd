from libc.stddef cimport ptrdiff_t
cdef extern from "csharp.h":
	ctypedef struct sharp_ringinfo:
		double theta, phi0, weight, cth, sth
		ptrdiff_t ofs
		int nph, stride
	ctypedef struct sharp_ringpair:
		sharp_ringinfo r1, r2
	ctypedef struct sharp_geom_info:
		sharp_ringpair * pair
		int npairs, nphmax
	ctypedef struct sharp_alm_info:
		int lmax, nm, flags
		int *mval
		ptrdiff_t *mvstart
		ptrdiff_t stride

	int SHARP_YtW=0
	int SHARP_MAP2ALM=SHARP_YtW
	int SHARP_Y=1
	int SHARP_ALM2MAP=SHARP_Y
	int SHARP_Yt=2
	int SHARP_WY=3
	int SHARP_ALM2MAP_DERIV1=4
	int SHARP_DP              = 1<<4
	int SHARP_ADD             = 1<<5
	int SHARP_REAL_HARMONICS  = 1<<6
	int SHARP_NO_FFT          = 1<<7
	int SHARP_USE_WEIGHTS     = 1<<20
	int SHARP_NO_OPENMP       = 1<<21
	int SHARP_NVMAX           = (1<<4)-1
	void sharp_make_geom_info (int nrings, int *nph, ptrdiff_t *ofs,
		int *stride, double *phi0, double *theta,
		double *wgt, sharp_geom_info **geom_info)
	void sharp_destroy_geom_info (sharp_geom_info *info)
	void sharp_make_alm_info (int lmax, int mmax, int stride,
		ptrdiff_t *mstart, sharp_alm_info **alm_info)
	void sharp_make_triangular_alm_info (int lmax, int mmax, int stride,
		sharp_alm_info **alm_info)
	void sharp_destroy_alm_info (sharp_alm_info *info)
	void sharp_execute (int type, int spin, void *alm, void *map,
		sharp_geom_info *geom_info, sharp_alm_info *alm_info, int ntrans,
		int flags, double *time, unsigned long long *opcnt)

	void sharp_make_weighted_healpix_geom_info (int nside, int stride,
		double *weight, sharp_geom_info **geom_info)
	void sharp_make_gauss_geom_info (int nrings, int nphi, double phi0,
		int stride_lon, int stride_lat, sharp_geom_info **geom_info)
	void sharp_make_fejer1_geom_info (int nrings, int nphi, double phi0,
		int stride_lon, int stride_lat, sharp_geom_info **geom_info)
	void sharp_make_cc_geom_info (int nrings, int ppring, double phi0,
		int stride_lon, int stride_lat, sharp_geom_info **geom_info)
	void sharp_make_fejer2_geom_info (int nrings, int ppring, double phi0,
		int stride_lon, int stride_lat, sharp_geom_info **geom_info)
	void sharp_make_mw_geom_info (int nrings, int ppring, double phi0,
		int stride_lon, int stride_lat, sharp_geom_info **geom_info)

cdef extern from "sharp_utils.h":
	void alm2cl_sp(sharp_alm_info * alm_info, float  * alm1, float  * alm2, float * cl)
	void alm2cl_dp(sharp_alm_info * alm_info, double * alm1, double * alm2, double * cl)
