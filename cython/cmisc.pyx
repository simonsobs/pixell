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
	npre   = int(np.product(pshape))
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

def islastcontig(arr):
	return arr[(0,)*(arr.ndim-1)].flags["C_CONTIGUOUS"]
def aslastcontig(arr):
	arr = np.asarray(arr)
	if islastcontig(arr): return arr
	else: return np.ascontiguousarray(arr)

def lmul(ainfo, alm, lfun, out=None):
	import warnings
	# The arrays need to be contiguous along the last dimension
	alm  = aslastcontig(alm)
	lfun = aslastcontig(lfun)
	if out is not None and not islastcontig(out):
		raise ValueError("lmul's out argument must be contiguous along last axis")
	# Are we doing a matmul?
	if lfun.ndim == 3 and alm.ndim == 2:
		if out is None: out = np.zeros(lfun.shape[:1]+alm.shape[1:], alm.dtype)
		if alm.dtype == np.complex128:  _lmatmul_dp(ainfo, alm, lfun, out)
		elif alm.dtype == np.complex64: _lmatmul_sp(ainfo, alm, lfun, out)
		else: raise ValueError("lmul requires complex64 or complex128 arrays")
	elif lfun.ndim == 1 and alm.ndim == 1:
		if out is None: out = alm.copy()
		if alm.dtype == np.complex128:  _lmul_dp(ainfo, out[None], lfun[None])
		elif alm.dtype == np.complex64: _lmul_sp(ainfo, out[None], lfun[None])
		else: raise ValueError("lmul requires complex64 or complex128 arrays")
	elif lfun.ndim == 2 and alm.ndim == 2:
		if out is None: out = alm.copy()
		if alm.dtype == np.complex128:  _lmul_dp(ainfo, out, lfun)
		elif alm.dtype == np.complex64: _lmul_sp(ainfo, out, lfun)
		else: raise ValueError("lmul requires complex64 or complex128 arrays")
	else:
		raise ValueError("lmul only supports alm,lfun shapes of [nalm],[nl], [N,nalm],[N,nl] and [N,M,nalm],[M,nl]")
	return out

cdef _lmul_dp(ainfo, alm, lfun):
	cdef int64_t[::1] mstart = np.ascontiguousarray(ainfo.mstart).view(np.int64)
	cdef double[::1] _alm, _lfun
	for i in range(alm.shape[-2]):
		_alm  = alm [i].view(np.float64)
		_lfun = lfun[i].view(np.float64)
		cmisc.lmul_dp(ainfo.lmax, ainfo.mmax, &mstart[0], &_alm[0], lfun.shape[1]-1, &_lfun[0])

cdef _lmul_sp(ainfo, alm, lfun):
	cdef int64_t[::1] mstart = np.ascontiguousarray(ainfo.mstart).view(np.int64)
	cdef float  [::1] _alm, _lfun
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

#	@cython.boundscheck(False)
#	@cython.wraparound(False)
#	cdef lmul_dp(self,np.ndarray[np.complex128_t,ndim=2] alm, np.ndarray[np.float64_t,ndim=3] lmat):
#		cdef int l, m, lm, c1, c2, ncomp, lcap
#		cdef np.ndarray[np.complex128_t,ndim=1] v
#		cdef np.ndarray[np.int64_t,ndim=1] mstart = self.mstart
#		l,m=0,0
#		ncomp = alm.shape[0]
#		lcap  = min(self.lmax+1,lmat.shape[2])
#		v = np.empty(ncomp,dtype=np.complex128)
#		#for m in prange(self.mmax+1,nogil=True,schedule="dynamic"):
#		for m in range(self.mmax+1):
#			for l in range(m, lcap):
#				lm = mstart[m]+l*self.stride
#				for c1 in range(ncomp):
#					v[c1] = alm[c1,lm]
#				for c1 in range(ncomp):
#					alm[c1,lm] = 0
#					for c2 in range(ncomp):
#						alm[c1,lm] = alm[c1,lm] + lmat[c1,c2,l] * v[c2]
#			# If lmat is too short, interpret missing values as zero
#			for l in range(lcap, self.lmax+1):
#				lm = mstart[m]+l*self.stride
#				for c1 in range(ncomp):
#					alm[c1,lm] = 0
#	@cython.boundscheck(False)
#	@cython.wraparound(False)
#	cdef lmul_sp(self,np.ndarray[np.complex64_t,ndim=2] alm, np.ndarray[np.float32_t,ndim=3] lmat):
#		cdef int l, m, lm, c1, c2, ncomp, lcap
#		cdef np.ndarray[np.complex64_t,ndim=1] v
#		cdef np.ndarray[np.int64_t,ndim=1] mstart = self.mstart
#		l,m=0,0
#		ncomp = alm.shape[0]
#		lcap  = min(self.lmax+1,lmat.shape[2])
#		v = np.empty(ncomp,dtype=np.complex64)
#		#for m in prange(self.mmax+1,nogil=True,schedule="dynamic"):
#		for m in range(self.mmax+1):
#			for l in range(m, lcap):
#				lm = mstart[m]+l*self.stride
#				for c1 in range(ncomp):
#					v[c1] = alm[c1,lm]
#				for c1 in range(ncomp):
#					alm[c1,lm] = 0
#					for c2 in range(ncomp):
#						alm[c1,lm] = alm[c1,lm] + lmat[c1,c2,l] * v[c2]
#			for l in range(lcap, self.lmax+1):
#				lm = mstart[m]+l*self.stride
#				for c1 in range(ncomp):
#					alm[c1,lm] = 0
#	def __dealloc__(self):
#		csharp.sharp_destroy_alm_info(self.info)
#
#def nalm2lmax(nalm):
#	return int((-1+(1+8*nalm)**0.5)/2)-1
#
#def transfer_alm(iainfo, ialm, oainfo, oalm=None, op=lambda a,b:b):
#	"""Transfer alm from one layout to another."""
#	if oalm is None:
#		oalm = np.zeros(ialm.shape[:-1]+(oainfo.nelem,), ialm.dtype)
#	lmax   = min(iainfo.lmax, oainfo.lmax)
#	mmax   = min(iainfo.mmax, oainfo.mmax)
#	pshape = ialm.shape[:-1]
#	npre   = int(np.product(pshape))
#	def transfer(dest, src, op): dest[:] = op(dest, src)
#	for i in range(npre):
#		I  = np.unravel_index(i, pshape)
#		ia = ialm[I]; oa = oalm[I]
#		for m in range(0, mmax+1):
#			transfer(oa[oainfo.mstart[m]+m*oainfo.stride:oainfo.mstart[m]+(lmax+1)*oainfo.stride:oainfo.stride], ia[iainfo.mstart[m]+m*iainfo.stride:iainfo.mstart[m]+(lmax+1)*iainfo.stride:iainfo.stride], op)
#	return oalm
#
#cdef class sht:
#	cdef public map_info minfo
#	cdef public alm_info ainfo
#	def __cinit__(self, minfo, ainfo):
#		"""Construct a sharp Spherical Harmonics Transform (SHT) object, which
#		transforms between maps with pixellication given by the map_info "minfo"
#		and spherical harmonic coefficents given by alm_info "ainfo"."""
#		self.minfo, self.ainfo = minfo, ainfo
#	def alm2map(self, alm, map=None, spin=0):
#		"""Transform the given spherical harmonic coefficients "alm" into
#		a map space. If a map is specified as the "map" argument, output
#		will be written there. Otherwise, a new map will be constructed
#		and returned. "alm" has dimensions [ntrans,nspin,nalm], or
#		[nspin,nalm] or [nalm] where ntrans is the number of independent
#		transforms to perform in parallel, nspin is the number of spin
#		components per transform (1 or 2), and nalm is the number of coefficients
#		per alm."""
#		alm = np.asarray(alm)
#		ntrans, nspin = dim_helper(alm, "alm")
#		# Create a compatible output map
#		if map is None:
#			map = np.empty([ntrans,nspin,self.minfo.npix],dtype=alm.real.dtype)
#			map = map.reshape(alm.shape[:-1]+(map.shape[-1],))
#		else:
#			assert alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree"
#		execute(csharp.SHARP_ALM2MAP, self.ainfo, alm, self.minfo, map, spin=spin)
#		return map
#	def map2alm(self, map, alm=None, spin=0):
#		"""Transform the given map "map" into a harmonic space. If "alm" is specified,
#		output will be written there. Otherwise, a new alm will be constructed
#		and returned. "map" has dimensions [ntrans,nspin,npix], or
#		[nspin,npix] or [npix] where ntrans is the number of independent
#		transforms to perform in parallel, nspin is the number of spin
#		components per transform (1 or 2), and npix is the number of pixels per map."""
#		map = np.asarray(map)
#		ntrans, nspin = dim_helper(map, "map")
#		# Create a compatible output map
#		if alm is None:
#			alm = np.empty([ntrans,nspin,self.ainfo.nelem],dtype=np.result_type(map.dtype,0j))
#			alm = alm.reshape(map.shape[:-1]+(alm.shape[-1],))
#		else:
#			assert alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree"
#		execute(csharp.SHARP_MAP2ALM, self.ainfo, alm, self.minfo, map, spin=spin)
#		return alm
#	def alm2map_der1(self, alm, map=None):
#		"""Compute derivative maps for the given scalar alms. If "map" is passed,
#		output will be written there. Otherwise a new map will be constructed
#		and returned. "alm" has dimensions [ntrans,nalm] or [nalm], and the map
#		has dimensions [ntrans,2,npix] or [2,npix], where ntrans is the number
#		of transformations to perform in parallel, and npix is the number of pixels
#		in the map."""
#		alm = np.asarray(alm)
#		ntrans = 1 if alm.ndim == 1 else alm.shape[0]
#		if map is None:
#			map = np.empty([ntrans,2,self.minfo.npix],dtype=alm.real.dtype)
#			map = map.reshape(alm.shape[:-1]+map.shape[-2:])
#		else:
#			assert map.ndim >= 2, "map must be at least 2d"
#			assert map.shape[-2] == 2, "Second to last dimension of map must have length 2"
#			assert alm.shape[:-1]==map.shape[:-2], "alm.shape[:-1] != map.shape[:-2]"
#		if alm.ndim == 2:
#			# Insert spin axis to ensure dimension check in execute passes.
#			alm = alm.reshape(alm.shape[0], 1, alm.shape[1])
#		execute(csharp.SHARP_ALM2MAP_DERIV1, self.ainfo, alm, self.minfo, map, spin=0)
#		return map
#	def map2alm_adjoint(self, alm, map=None, spin=0):
#		"""The adjoint of the map2alm operation. This operation is closely
#		related to alm2map. It reads from alm and writes to map."""
#		alm = np.asarray(alm)
#		ntrans, nspin = dim_helper(alm, "alm")
#		# Create a compatible output map
#		if map is None:
#			map = np.empty([ntrans,nspin,self.minfo.npix],dtype=alm.real.dtype)
#			map = map.reshape(alm.shape[:-1]+(map.shape[-1],))
#		else:
#			assert alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree"
#		execute(csharp.SHARP_WY, self.ainfo, alm, self.minfo, map, spin=spin)
#		return map
#	def alm2map_adjoint(self, map, alm=None, spin=0):
#		"""The adjoint of the alm2map operation. This operation is closely
#		related to map2alm, but is a simple sum instead of a surface integral.
#		It reads from map and writes to alm."""
#		map = np.asarray(map)
#		ntrans, nspin = dim_helper(map, "map")
#		# Create a compatible output map
#		if alm is None:
#			alm = np.empty([ntrans,nspin,self.ainfo.nelem],dtype=np.result_type(map.dtype,0j))
#			alm = alm.reshape(map.shape[:-1]+(alm.shape[-1],))
#		else:
#			assert alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree"
#		execute(csharp.SHARP_Yt, self.ainfo, alm, self.minfo, map, spin=spin)
#		return alm
#
## alm and map have the formats:
##  [nlm]:           spin=0,    ntrans=1
##  [ns,nlm]:        spin=ns>1, ntrans=1
##  [ntrans,ns,nlm]: spin=ns>1, ntrans=ntrans
## So to do many spin 0 transforms in parallel, you would pass alm with
## the shape [:,1,:], which can be created from a 2d alm by alm[:,None]
#def dim_helper(a, name):
#	assert a.ndim > 0 and a.ndim <= 3, name + " must be [nlm], [ntrf*ncomp,nlm] or [ntrf,ncomp,nlm]"
#	if a.ndim == 1:
#		ntrans, nspin = 1, 1
#	elif a.ndim == 2:
#		ntrans, nspin = 1, a.shape[0]
#	elif a.ndim == 3:
#		ntrans, nspin = a.shape[:2]
#	assert nspin < 3, name + " spin axis must have length 1 or 2 (T and P must be done separately)"
#	return ntrans, nspin
#
#def execute(type, alm_info ainfo, alm, map_info minfo, map, spin):
#	assert isinstance(alm, np.ndarray), "alm must be a numpy array"
#	assert isinstance(map, np.ndarray), "map must be a numpy array"
#	cdef int i
#	ntrans, nspin = dim_helper(alm, "alm")
#	ntrans, ncomp = dim_helper(map, "map")
#	assert(spin == 0 and nspin != 1 or spin > 0 and nspin != 2,
#		"Dimension -2 of maps and alms must be 2 for spin transforms and 1 for scalar transforms.")
#	try:
#		type = typemap[type]
#	except KeyError:
#		pass
#	alm3 = alm.reshape(ntrans,nspin,-1)
#	map3 = map.reshape(ntrans,ncomp,-1)
#	if map.dtype == np.float64:
#		execute_dp(type, spin, ainfo, alm3, minfo, map3)
#	else:
#		execute_sp(type, spin, ainfo, alm3, minfo, map3)
#
#cpdef execute_dp(int type, int spin, alm_info ainfo, np.ndarray[np.complex128_t,ndim=3] alm, map_info minfo, np.ndarray[np.float64_t,ndim=3] map):
#	cdef int ntrans = map.shape[0]
#	cdef int nmap = map.shape[0]*map.shape[1]
#	cdef int nalm = alm.shape[0]*alm.shape[1]
#	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]	 aptrs = np.empty(alm.shape[1],dtype=np.uintp)
#	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]	 mptrs = np.empty(map.shape[1],dtype=np.uintp)
#
#	cdef int i, j
#	for i in range(ntrans):
#		for j in range(alm.shape[1]):
#			aptrs[j] = <np.uintp_t>&alm[i,j,0]
#		for j in range(map.shape[1]):
#			mptrs[j] = <np.uintp_t>&map[i,j,0]
#		execute_helper(type, ainfo, aptrs, minfo, mptrs,
#			       spin, csharp.SHARP_DP)
#
#cpdef execute_sp(int type, int spin, alm_info ainfo, np.ndarray[np.complex64_t,ndim=3,mode="c"] alm, map_info minfo, np.ndarray[np.float32_t,ndim=3,mode="c"] map):
#	cdef int ntrans = map.shape[0]
#	cdef int nmap = map.shape[0]*map.shape[1]
#	cdef int nalm = alm.shape[0]*alm.shape[1]
#	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]	 aptrs = np.empty(alm.shape[1],dtype=np.uintp)
#	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]	 mptrs = np.empty(map.shape[1],dtype=np.uintp)
#
#	cdef int i, j
#	for i in range(ntrans):
#		for j in range(alm.shape[1]):
#			aptrs[j] = <np.uintp_t>&alm[i,j,0]
#		for j in range(map.shape[1]):
#			mptrs[j] = <np.uintp_t>&map[i,j,0]
#		execute_helper(type, ainfo, aptrs, minfo, mptrs,
#			       spin, 0)
#
#cdef check_cont_dp(np.ndarray[np.float64_t,ndim=1,mode="c"] map_row, np.ndarray[np.complex128_t,ndim=1,mode="c"] alm_row): pass
#cdef check_cont_sp(np.ndarray[np.float32_t,ndim=1,mode="c"] map_row, np.ndarray[np.complex64_t, ndim=1,mode="c"] alm_row): pass
#
#typemap = { "map2alm": csharp.SHARP_MAP2ALM, "alm2map": csharp.SHARP_ALM2MAP, "alm2map_der1": csharp.SHARP_ALM2MAP_DERIV1, "map2alm_adjoint": csharp.SHARP_WY, "alm2map_adjoint": csharp.SHARP_Yt }
#
#cdef execute_helper(int type,
#		alm_info ainfo, np.ndarray[np.uintp_t,ndim=1] alm,
#		map_info minfo, np.ndarray[np.uintp_t,ndim=1] map,
#		int spin=0, int flags=0):
#	csharp.sharp_execute(type, spin, <void*>&alm[0], <void*>&map[0],
#			minfo.geom, ainfo.info, flags, NULL, NULL)
#
#cdef csharp.sharp_geom_info * make_geometry_helper(
#		int ntheta,
#		np.ndarray[int,ndim=1] nphi,
#		np.ndarray[csharp.ptrdiff_t,ndim=1] offsets,
#		np.ndarray[int,ndim=1] stride,
#		np.ndarray[double,ndim=1] phi0,
#		np.ndarray[double,ndim=1] theta,
#		np.ndarray[double,ndim=1] weight):
#	cdef csharp.sharp_geom_info * geom
#	csharp.sharp_make_geom_info(ntheta, &nphi[0], &offsets[0], &stride[0], &phi0[0], &theta[0], &weight[0], &geom)
#	return geom
#
#cdef csharp.sharp_alm_info * make_alm_helper(int lmax, int mmax, int stride, np.ndarray[csharp.ptrdiff_t,ndim=1] mstart):
#	cdef csharp.sharp_alm_info * info
#	csharp.sharp_make_alm_info(lmax, mmax, stride, &mstart[0], &info)
#	return info
#
#cdef csharp.sharp_alm_info * make_triangular_alm_helper(int lmax, int mmax, int stride):
#	cdef csharp.sharp_alm_info * info
#	csharp.sharp_make_triangular_alm_info(lmax, mmax, stride, &info)
#	return info
