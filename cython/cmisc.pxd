from libc.stdint cimport int64_t
cdef extern from "cmisc_core.h":
	void alm2cl_sp(int lmax, int mmax, int64_t * mstart, float  * alm1, float  * alm2, float * cl)
	void alm2cl_dp(int lmax, int mmax, int64_t * mstart, double * alm1, double * alm2, double * cl)
	void transpose_alm_dp(int lmax, int mmax, int64_t * mstart, double * ialm, double * oalm)
	void transpose_alm_sp(int lmax, int mmax, int64_t * mstart, float * ialm, float * oalm)
	void lmul_dp(int lmax, int mmax, int64_t * mstart, double * alm, int lfmax, const double * lfun)
	void lmul_sp(int lmax, int mmax, int64_t * mstart, float * alm, int lfmax, const float * lfun)
	void transfer_alm_dp(int lmax1, int mmax1, int64_t * mstart1, double * alm1, int lmax2, int mmax2, int64_t * mstart2, double * alm2)
	void transfer_alm_sp(int lmax1, int mmax1, int64_t * mstart1, float * alm1, int lmax2, int mmax2, int64_t * mstart2, float * alm2)
	void lmatmul_dp(int N, int M, int lmax, int mmax, int64_t * mstart, double ** alm, int lfmax, double ** lmat, double ** oalm)
	void lmatmul_sp(int N, int M, int lmax, int mmax, int64_t * mstart, float ** alm, int lfmax, float ** lmat, float ** oalm)
