#include "sharp_utils.h"
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }

void alm2cl_sp(sharp_alm_info * ainfo, float * alm1, float * alm2, float * cl) {
	int nthread = omp_get_max_threads();
	int nl      = ainfo->lmax+1;
	float * buf = calloc(nthread*nl, sizeof(float));
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		if(id == 0) {
			for(int l = 0; l <= ainfo->lmax; l++) {
				ptrdiff_t i = ainfo->mvstart[0]*2 + l*2;
				buf[nl*id+l] = alm1[i]*alm2[i]/2;
			}
		}
		#pragma omp for schedule(dynamic)
		for(int m = 1; m < ainfo->nm; m++) {
			for(int l = m; l <= ainfo->lmax; l++) {
				ptrdiff_t i = ainfo->mvstart[m]*2 + l*2;
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

void alm2cl_dp(sharp_alm_info * ainfo, double * alm1, double * alm2, double * cl) {
	int nthread = omp_get_max_threads();
	int nl      = ainfo->lmax+1;
	double * buf = calloc(nthread*nl, sizeof(double));
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		if(id == 0) {
			for(int l = 0; l <= ainfo->lmax; l++) {
				ptrdiff_t i = ainfo->mvstart[0]*2 + l*2;
				buf[nl*id+l] = alm1[i]*alm2[i]/2;
			}
		}
		#pragma omp for schedule(dynamic)
		for(int m = 1; m < ainfo->nm; m++) {
			for(int l = m; l <= ainfo->lmax; l++) {
				ptrdiff_t i = ainfo->mvstart[m]*2 + l*2;
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

#if 0
void alm2cl_plain_sp(sharp_alm_info * ainfo, float * alm, float * cl) {
	// Straightforward implementation. The baseline we will try to improve on.
	// We don't support stride or packing.
	for(int l = 0; l <= ainfo->lmax; l++) {
		ptrdiff_t i = ainfo->mvstart[0]*2 + l*2;
		float c     = alm[i]*alm[i];
		for(int mi = 1; mi < ainfo->nm; mi++) {
			if(ainfo->mval[mi] > l) continue;
			i  = ainfo->mvstart[mi]*2 + l*2;
			c += 2*(alm[i]*alm[i] + alm[i+1]*alm[i+1]);
		}
		cl[l] = c/(2*l+1);
	}
}

void alm2cl_mmajor_sp(sharp_alm_info * ainfo, float * alm, float * cl) {
	for(int l = 0; l <= ainfo->lmax; l++) {
		ptrdiff_t i = ainfo->mvstart[0]*2 + l*2;
		cl[l]       = alm[i]*alm[i];
	}
	for(int mi = 1; mi < ainfo->nm; mi++) {
		int m = ainfo->mval[mi];
		for(int l = m; l <= ainfo->lmax; l++) {
			ptrdiff_t i = ainfo->mvstart[mi]*2 + l*2;
			cl[l] += 2*(alm[i]*alm[i] + alm[i+1]*alm[i+1]);
		}
	}
	for(int l = 0; l <= ainfo->lmax; l++) {
		cl[l] /= (2*l+1);
	}
}

void alm2cl_mmajor2_sp(sharp_alm_info * ainfo, float * alm, float * cl) {
	int nthread = omp_get_max_threads();
	int nl      = ainfo->lmax+1;
	float * buf = calloc(nthread*nl, sizeof(float));
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		if(id == 0) {
			for(int l = 0; l <= ainfo->lmax; l++) {
				ptrdiff_t i = ainfo->mvstart[0]*2 + l*2;
				buf[nl*id+l] = alm[i]*alm[i];
			}
		}
		#pragma omp for schedule(dynamic)
		for(int m = 1; m < ainfo->nm; m++) {
			for(int l = m; l <= ainfo->lmax; l++) {
				ptrdiff_t i = ainfo->mvstart[m]*2 + l*2;
				if(m == 0) buf[nl*id+l] += alm[i]*alm[i];
				else       buf[nl*id+l] += 2*(alm[i]*alm[i] + alm[i+1]*alm[i+1]);
			}
		}
	}
	#pragma omp parallel for
	for(int l = 0; l < nl; l++) {
		cl[l] = 0;
		for(int i = 0; i < nthread; i++)
			cl[l] += buf[nl*i+l];
		cl[l] /= 2*l+1;
	}
	free(buf);
}

void alm2cl_avx_sp(sharp_alm_info * ainfo, float * alm, float * cl) {
	// AVX verison. Operate on 8 ls at a time. Simple l-major one first.
	// In theory this should be safe, as we're working on units larger than a
	// cache line. Fast operations require 32-byte alignment, but it's hard to
	// ensure that. That means that this won't be much faster than the automatic
	// version. I measure it to be 17% faster.
	#pragma omp parallel for schedule(dynamic)
	for(int l1 = 0; l1 <= ainfo->lmax; l1+=8) {
		int l2 = min(l1+8, ainfo->lmax+1);
		int nm_block;
		if(l2 == l1+8) {
			// We have a whole block worth of ls. Handle all our ls up to nm_block
			nm_block = min(l1+1, ainfo->nm);
			__m256 a1, a2, c = _mm256_setzero_ps(), fact, tmp;
			__m256i permutation = _mm256_set_epi32(7,6,3,2,5,4,1,0);
			for(int m = 0; m < nm_block; m++) {
				fact = _mm256_set1_ps(m?2:1);
				// Load our alms
				ptrdiff_t i = ainfo->mvstart[m]*2 + l1*2;
				a1 = _mm256_loadu_ps(alm + i + 0);
				a2 = _mm256_loadu_ps(alm + i + 8);
				a1 *= a1;
				a2 *= a2;
				tmp = _mm256_hadd_ps(a1, a2);
				// At this point we have 76325410, but we want 76543210
				tmp = _mm256_permutevar8x32_ps(tmp, permutation);
				c  += tmp*fact;
			}
			_mm256_storeu_ps(cl + l1, c);
		} else {
			// We're at the end of the ls, so we can't do blocks. This will
			// be handled by the general case below, but before that we need
			// to zero-initialize our values. We don't handle any ms in this case,
			nm_block = 0;
			for(int l = l1; l < l2; l++)
				cl[l] = 0;
		}
		// General case
		for(int m = nm_block; m < ainfo->nm; m++) {
			for(int l = max(m,l1); l < l2; l++) {
				ptrdiff_t i = ainfo->mvstart[m]*2 + l*2;
				cl[l] += (1+(m>0))*(alm[i]*alm[i]+alm[i+1]*alm[i+1]);
			}
		}
	}
	// And normalize
	for(int l = 0; l <= ainfo->lmax; l++)
		cl[l] /= 2*l+1;
}

#endif
