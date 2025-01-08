import cython
import numpy as np
cimport numpy as np
cimport cmisc
from libc.math cimport atan2
from libc.stdint cimport uintptr_t, int64_t

def alm2cl(ainfo, alm, alm2=None):
	"""Computes the cross power spectrum for the given alm and alm2, which
	must have the same dtype and broadcast. For example, to get the TEB,TEB
	cross spectra for a single map you would do
	 cl = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
	To get the same TEB,TEB spectra crossed with a different map it would
	be
	 cl = ainfo.alm2cl(alm1[:,None,:], alm2[None,:,:])
	In both these cases the output will be [{T,E,B},{T,E,B},nl]"""
	alm   = np.asarray(alm)
	alm2  = np.asarray(alm2) if alm2 is not None else alm
	# Unify dtypes
	dtype= np.result_type(alm, alm2)
	alm  = alm.astype (dtype, copy=False)
	alm2 = alm2.astype(dtype, copy=False)
	def getaddr(a): return a.__array_interface__["data"][0]
	# Broadcast alms. This looks scary, but just results in views of the original
	# arrays according to the documentation. Hence, it shouldn't use more memory.
	# The resulting arrays are non-contiguous, but each individual alm is still
	# contiguous. We set the writable flags not because we intend to write, but
	# to silience a false positive warning from numpy
	alm, alm2 = np.broadcast_arrays(alm, alm2)
	alm.flags["WRITEABLE"] = alm2.flags["WRITEABLE"] = True
	# I used to flatten here to make looping simple, but that caused a copy to be made
	# when combined with np.broadcast. So instead I will use manual raveling
	pshape = alm.shape[:-1]
	npre   = int(np.prod(pshape))
	cdef float[::1]  cl_single_sp, alm_single_sp1, alm_single_sp2
	cdef double[::1] cl_single_dp, alm_single_dp1, alm_single_dp2
	cdef int64_t[::1] mstart = np.ascontiguousarray(ainfo.mstart).view(np.int64)
	# A common use case is to compute TEBxTEB auto-cross spectra, where
	# e.g. TE === ET since alm1 is the same array as alm2. To avoid duplicate
	# calculations in this case we use a cache, which skips computing the
	# cross-spectrum of any given pair of arrays more than once.
	cache = {}
	if alm.dtype == np.complex64:
		cl = np.empty(alm.shape[:-1]+(ainfo.lmax+1,), np.float32)
		# We will loop over individual spectra
		for i in range(npre):
			I = np.unravel_index(i, pshape)
			# Avoid duplicate calculation
			key   = tuple(sorted([getaddr(alm[I]), getaddr(alm2[I])]))
			if key in cache:
				cl[I] = cache[key]
			else:
				alm_single_sp1 = np.ascontiguousarray(alm [I]).view(np.float32)
				alm_single_sp2 = np.ascontiguousarray(alm2[I]).view(np.float32)
				cl_single_sp = cl[I]
				cmisc.alm2cl_sp(ainfo.lmax, ainfo.mmax, &mstart[0], &alm_single_sp1[0], &alm_single_sp2[0], &cl_single_sp[0])
				cache[key] = cl[I]
		return cl
	elif alm.dtype == np.complex128:
		cl = np.empty(alm.shape[:-1]+(ainfo.lmax+1,), np.float64)
		# We will loop over individual spectra
		for i in range(npre):
			I = np.unravel_index(i, pshape)
			key   = tuple(sorted([getaddr(alm[I]), getaddr(alm2[I])]))
			if key in cache:
				cl[I] = cache[key]
			else:
				alm_single_dp1 = np.ascontiguousarray(alm [I]).view(np.float64)
				alm_single_dp2 = np.ascontiguousarray(alm2[I]).view(np.float64)
				cl_single_dp = cl[I]
				cmisc.alm2cl_dp(ainfo.lmax, ainfo.mmax, &mstart[0], &alm_single_dp1[0], &alm_single_dp2[0], &cl_single_dp[0])
				cache[key] = cl[I]
		return cl
	else:
		raise ValueError("Only complex64 and complex128 supported, but got %s" % str(alm.dtype))

# Repeated from utils to avoid dragging it in as a dependency
def nditer(shape):
	ndim = len(shape)
	I    = [0]*ndim
	while True:
		yield tuple(I)
		for dim in range(ndim-1,-1,-1):
			I[dim] += 1
			if I[dim] < shape[dim]: break
			I[dim] = 0
		else:
			break

def transpose_alm(ainfo, alm, out=None):
	"""In order to accomodate l-major ordering, which is not directoy
	supported by sharp, this function efficiently transposes Alm into
	Aml. If the out argument is specified, the transposed result will
	be written there. In order to perform an in-place transpose, call
	this function with the same array as "alm" and "out". If the out
	argument is not specified, then a new array will be constructed
	and returned."""
	if out is None: out = alm.copy()
	# Need this work array when the out and in arrays are the same.
	# Could avoid it for the non-inplace case, but for now always use it
	work = np.zeros(alm.shape[-1:], alm.dtype)
	cdef int64_t[::1] mstart = np.ascontiguousarray(ainfo.mstart).view(np.int64)
	cdef float[::1]  alm_sp, work_sp
	cdef double[::1] alm_dp, work_dp
	if out.dtype == np.complex128:
		work_dp = work.view(np.float64)
		for I in nditer(alm.shape[:-1]):
			alm_dp = alm[I].view(np.float64)
			cmisc.transpose_alm_dp(ainfo.lmax, ainfo.mmax, &mstart[0], &alm_dp[0], &work_dp[0])
			out[I] = work.view(np.complex128)
	elif out.dtype == np.complex64:
		work_sp = work.view(np.float32)
		for I in nditer(alm.shape[:-1]):
			alm_sp = alm[I].view(np.float32)
			cmisc.transpose_alm_sp(ainfo.lmax, ainfo.mmax, &mstart[0], &alm_sp[0], &work_sp[0])
			out[I] = work.view(np.complex64)
	else:
		raise ValueError("transpose_alm requires contiguous complex64 or complex128 arrays")
	return out

# I have C versions of this, but this one is more flexible and should be fast enough,
# given that it's just copying things around (and it's in cython anyway)
def transfer_alm(iainfo, ialm, oainfo, oalm=None, op=lambda a,b:b):
	"""Transfer alm from one layout to another."""
	if oalm is None:
		oalm = np.zeros(ialm.shape[:-1]+(oainfo.nelem,), ialm.dtype)
	lmax = min(iainfo.lmax, oainfo.lmax)
	mmax = min(iainfo.mmax, oainfo.mmax)
	if ialm.shape[:-1] != oalm.shape[:-1]:
		raise ValueError("ialm and oalm must agree on pre-dimensions")
	pshape = ialm.shape[:-1]
	npre = int(np.prod(pshape))
	# Numpy promotes uint64 to float64, so make an int64 view of mstart
	imstart = iainfo.mstart.view(np.int64)
	omstart = oainfo.mstart.view(np.int64)
	def transfer(dest, src, op): dest[:] = op(dest, src)
	for i in range(npre):
		I  = np.unravel_index(i, pshape)
		ia = ialm[I]; oa = oalm[I]
		for m in range(0, mmax+1):
			transfer(oa[omstart[m]+m*oainfo.stride:omstart[m]+(lmax+1)*oainfo.stride:oainfo.stride], ia[imstart[m]+m*iainfo.stride:imstart[m]+(lmax+1)*iainfo.stride:iainfo.stride], op)
	return oalm

def islastcontig(arr):
	return arr[(0,)*(arr.ndim-1)].flags["C_CONTIGUOUS"]
def aslastcontig(arr, dtype=None):
	arr = np.asarray(arr, dtype=dtype)
	if islastcontig(arr): return arr
	else: return np.ascontiguousarray(arr)

def lmul(ainfo, alm, lfun, out=None):
	import warnings
	# alm must be complex, lfun must be the corresponding real type
	alm  = np.asarray(alm)
	ctype= np.result_type(alm.dtype, 0j)
	rtype= np.zeros(1, alm.dtype).real.dtype
	# The arrays need to be contiguous along the last dimension
	alm  = aslastcontig(alm,  dtype=ctype)
	lfun = aslastcontig(lfun, dtype=rtype)
	if out is not None and not islastcontig(out) and out.dtype != ctype:
		raise ValueError("lmul's out argument must be contiguous along last axis, and have the same dtype as alm")
	# Are we doing a matmul?
	if lfun.ndim == 3 and alm.ndim == 2:
		if out is None: out = np.zeros(lfun.shape[:1]+alm.shape[1:], alm.dtype)
		if alm.dtype == np.complex128:  _lmatmul_dp(ainfo, alm, lfun, out)
		elif alm.dtype == np.complex64: _lmatmul_sp(ainfo, alm, lfun, out)
		else: raise ValueError("lmul requires complex64 or complex128 arrays")
	else:
		# Broadcast pre-dimensions, if they're compatible
		try:
			pre  = np.broadcast_shapes(alm.shape[:-1], lfun.shape[:-1])
		except ValueError:
			raise ValueError("lmul's alm and lfun's dimensions must either broadcast (when ignoring the last dimension), or have shape compatible with a matrix product (again ignoring the last dimension)")
		alm  = np.broadcast_to(alm,  pre+ alm.shape[-1:])
		lfun = np.broadcast_to(lfun, pre+lfun.shape[-1:])
		# Flatten, so the C code doesn't need to deal with variable dimensionality
		aflat= alm.reshape(-1,alm.shape[-1])
		lflat= lfun.reshape(-1,lfun.shape[-1])
		if out is None:
			out = aflat.copy()
		else:
			out = out.reshape(aflat.shape)
			out[:] = aflat
		if alm.dtype == np.complex128:  _lmul_dp(ainfo, out, lflat)
		elif alm.dtype == np.complex64: _lmul_sp(ainfo, out, lflat)
		else: raise ValueError("lmul requires complex64 or complex128 arrays")
		# Unflatten output
		out = out.reshape(pre + out.shape[-1:])
	return out

cdef _lmul_dp(ainfo, alm, lfun):
	cdef int64_t[::1] mstart = np.ascontiguousarray(ainfo.mstart).view(np.int64)
	cdef double[::1] _alm
	cdef const double[::1] _lfun
	for i in range(alm.shape[-2]):
		_alm  = alm [i].view(np.float64)
		_lfun = lfun[i].view(np.float64)
		cmisc.lmul_dp(ainfo.lmax, ainfo.mmax, &mstart[0], &_alm[0], lfun.shape[1]-1, &_lfun[0])

cdef _lmul_sp(ainfo, alm, lfun):
	cdef int64_t[::1] mstart = np.ascontiguousarray(ainfo.mstart).view(np.int64)
	cdef float  [::1] _alm
	cdef const float  [::1] _lfun
	for i in range(alm.shape[-2]):
		_alm  = alm [i].view(np.float32)
		_lfun = lfun[i].view(np.float32)
		cmisc.lmul_sp(ainfo.lmax, ainfo.mmax, &mstart[0], &_alm[0], lfun.shape[1]-1, &_lfun[0])

# Yuck! Cython is a pain. There must be an easier way to do this, but I didn't find it.
cdef _lmatmul_dp(ainfo, np.complex128_t[:,:] alm, np.float64_t[:,:,:] lfun, np.complex128_t[:,:] out):
	cdef int64_t[::1] mstart = np.ascontiguousarray(ainfo.mstart).view(np.int64)
	cdef np.uintp_t[::1] aptrs = np.empty(alm.shape[0], dtype=np.uintp)
	cdef np.uintp_t[::1] optrs = np.empty(out.shape[0], dtype=np.uintp)
	cdef np.uintp_t[::1] fptrs = np.empty(lfun.shape[0]*lfun.shape[1],dtype=np.uintp)
	cdef int i, j
	cdef int N = lfun.shape[0]
	cdef int M = lfun.shape[1]
	for i in range(N):
		for j in range(M):
			fptrs[i*M+j] = <np.uintp_t>&lfun[i,j,0]
	for i in range(N): optrs[i] = <np.uintp_t>&out[i,0]
	for j in range(M): aptrs[j] = <np.uintp_t>&alm[j,0]
	cmisc.lmatmul_dp(N, M, ainfo.lmax, ainfo.mmax, &mstart[0], <double**>&aptrs[0], lfun.shape[2]-1, <double**>&fptrs[0], <double**>&optrs[0])

cdef _lmatmul_sp(ainfo, np.complex64_t[:,:] alm, np.float32_t[:,:,:] lfun, np.complex64_t[:,:] out):
	cdef int64_t[::1] mstart = np.ascontiguousarray(ainfo.mstart).view(np.int64)
	cdef np.uintp_t[::1] aptrs = np.empty(alm.shape[0], dtype=np.uintp)
	cdef np.uintp_t[::1] optrs = np.empty(out.shape[0], dtype=np.uintp)
	cdef np.uintp_t[::1] fptrs = np.empty(lfun.shape[0]*lfun.shape[1],dtype=np.uintp)
	cdef int i, j
	cdef int N = lfun.shape[0]
	cdef int M = lfun.shape[1]
	for i in range(N):
		for j in range(M):
			fptrs[i*M+j] = <np.uintp_t>&lfun[i,j,0]
	for i in range(N): optrs[i] = <np.uintp_t>&out[i,0]
	for j in range(M): aptrs[j] = <np.uintp_t>&alm[j,0]
	cmisc.lmatmul_sp(N, M, ainfo.lmax, ainfo.mmax, &mstart[0], <float**>&aptrs[0], lfun.shape[2]-1, <float**>&fptrs[0], <float**>&optrs[0])
