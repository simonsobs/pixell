import cython
import numpy as np
cimport numpy as np
cimport csharp
from libc.math cimport atan2
from libc.stdint cimport uintptr_t
#from cython.parallel import prange, parallel

cdef class map_info:
	"""This class is a thin wrapper for the sharp geom_info struct, which represents
	the pixelization used in spherical harmonics transforms. It can represent an
	abritrary number of constant-latitude rings at arbitrary latitudes. Each ring
	has an arbitrary number of equidistant points in latitude, with an abritrary
	offset in latitude."""
	cdef csharp.sharp_geom_info * geom
	cdef readonly int nrow
	cdef readonly int npix
	cdef readonly np.ndarray theta
	cdef readonly np.ndarray nphi
	cdef readonly np.ndarray phi0
	cdef readonly np.ndarray offsets
	cdef readonly np.ndarray stride
	cdef readonly np.ndarray weight
	def __cinit__(self, theta, nphi=0, phi0=0, offsets=None, stride=None, weight=None):
		"""Construct a new sharp map geometry consiting of N rings in co-latitude, at
		co-latitudes theta[N], with the rings having nphi[N] points each, with the
		first point at longitude phi0[N]. Each ring has pixel stride stride[N], and
		the pixel index of the first pixel in each ring is offsets[N]. For map2alm
		transforms, weight[N] specifies the integral weights for each row. These
		are complicated and depend on the pixel layout."""
		theta = np.asarray(theta, dtype=np.float64)
		assert theta.ndim == 1, "theta must be one-dimensional!"
		ntheta = len(theta)
		nphi  = np.asarray(nphi, dtype=np.int32)
		assert nphi.ndim < 2, "nphi must be 0 or 1-dimensional"
		if nphi.ndim == 0:
			nphi = np.zeros(ntheta,dtype=np.int32)+(nphi or 2*ntheta)
		assert len(nphi) == ntheta, "theta and nphi arrays do not agree on number of rings"
		phi0 = np.asarray(phi0, dtype=np.float64)
		assert phi0.ndim < 2, "phi0 must be 0 or 1-dimensional"
		if phi0.ndim == 0:
			phi0 = np.zeros(ntheta,dtype=np.float64)+phi0
		if offsets is None:
			offsets = np.concatenate([[0],np.cumsum(nphi.astype(np.int64)[:-1])])
		if stride  is None:
			stride  = np.zeros(ntheta,dtype=np.int32)+1
		if weight  is None:
			weight  = np.zeros(ntheta,dtype=np.float64)+1
		self.geom = make_geometry_helper(ntheta, nphi, offsets, stride, phi0, theta, weight)
		# Store publicly accessible view of the internal geometry. This
		# is kept by going out of sync via readonly and writable=False
		self.npix = np.sum(nphi)
		self.nrow = len(nphi)
		for v in [theta,nphi,phi0,offsets,stride,weight]: v.flags.writeable = False
		self.theta, self.nphi, self.phi0, self.offsets, self.stride, self.weight = theta, nphi, phi0, offsets, stride, weight
	def __dealloc__(self):
		csharp.sharp_destroy_geom_info(self.geom)
	def select_rows(self, rows):
		return map_info(self.theta[rows], self.nphi[rows], self.phi0[rows], self.offsets[rows], self.stride[rows], self.weight[rows])

def map_info_healpix(int nside, int stride=1, weights=None):
	"""Construct a new sharp map geometry for the HEALPix pixelization in the RING
	scheme, with the given nside parameter (which does not need to be a power of
	2 in this case). The optional weights array specifies quadrature weights
	*relative* to the default weights, so you will get sensible results even without
	specifying weights."""
	nring = 4*nside-1
	if weights is None: weights = np.zeros(nring)+1
	assert len(weights) >= nring, "incorrect length of weights array. need 4*nside-1"
	cdef np.ndarray[np.float64_t,ndim=1] w = weights
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_weighted_healpix_geom_info (nside, stride, &w[0], &geom)
	return map_info_from_geom_and_free(geom)

def map_info_gauss_legendre(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Gauss-Legendre pixelization. Optimal
	weights are computed automatically. The pixels are laid out in nrings rings of
	constant colatitude, each with nphi pixels equally spaced in longitude, with the
	first pixel in each ring at longitude phi0."""
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_gauss_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom_and_free(geom)

def map_info_clenshaw_curtis(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Cylindrical Equi-rectangular pixelization
	with nrings iso-colatitude rings with nphi pixels each, such that the first and last
	rings have colatitude 0 and pi respectively. This corresponds to Clenshaw-Curtis
	quadrature."""
	cdef int inphi = 2*(nrings-1) if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_cc_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom_and_free(geom)

def map_info_fejer1(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Cylindrical Equi-rectangular pixelization
	with nrings iso-colatitude rings with nphi pixels each, such that the first and last
	rings have colatitude 0.5*pi/nrings and pi-0.5*pi/nrings respectively.
	This corresponds to Frejer's first quadrature."""
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_fejer1_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom_and_free(geom)

def map_info_fejer2(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Cylindrical Equi-rectangular pixelization
	with nrings iso-colatitude rings with nphi pixels each, such that the first and last
	rings have colatitude pi/(nrings+1) and pi-pi/(nrings+1) respectively.
	This corresponds to Frejer's second quadrature."""
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_fejer2_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom_and_free(geom)

def map_info_mw(int nrings, nphi=None, double phi0=0, stride_lon=None, stride_lat=None):
	"""Construct a new sharp map geometry with Cylindrical Equi-rectangular pixelization
	with nrings iso-colatitude rings with nphi pixels each, such that the first and last
	rings have colatitude pi/(2*nrings-1) and pi respectively.
	This function does *NOT* define any quadrature weights!"""
	cdef int inphi = 2*nrings if nphi is None else nphi
	cdef int slon  = 1 if stride_lon is None else stride_lon
	cdef int slat  = inphi*slon if stride_lat is None else stride_lat
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_mw_geom_info(nrings, inphi, phi0, slon, slat, &geom)
	return map_info_from_geom_and_free(geom)

cdef map_info_from_geom(csharp.sharp_geom_info * geom):
	"""Constructs a map_info from a gemoetry pointer."""
	cdef int pair
	cdef int ring = 0
	cdef int npairs = geom.npairs
	cdef np.ndarray[np.float64_t,ndim=1] theta   = np.empty(2*npairs,dtype=np.float64)
	cdef np.ndarray[np.float64_t,ndim=1] phi0    = np.empty(2*npairs,dtype=np.float64)
	cdef np.ndarray[np.float64_t,ndim=1] weight  = np.empty(2*npairs,dtype=np.float64)
	cdef np.ndarray[np.int32_t,ndim=1]   nphi    = np.empty(2*npairs,dtype=np.int32)
	cdef np.ndarray[np.int32_t,ndim=1]   stride  = np.empty(2*npairs,dtype=np.int32)
	cdef np.ndarray[np.int64_t,ndim=1]   offsets = np.empty(2*npairs,dtype=np.int64)
	cdef csharp.sharp_ringinfo * info
	# This should have been as simple as for info in (geom.pair[pair].r1,geom.pair[pair].r2)
	cdef np.uintp_t infop
	cdef np.ndarray[np.uintp_t,ndim=1] tmp = np.empty(2,dtype=np.uintp)
	for pair in range(npairs):
		tmp[0] = <np.uintp_t>&geom.pair[pair].r1
		tmp[1] = <np.uintp_t>&geom.pair[pair].r2
		for infop in tmp:
			info = <csharp.sharp_ringinfo*> infop
			if info.nph >= 0:
				theta[ring]  = atan2(info.sth,info.cth)
				phi0[ring]   = info.phi0
				weight[ring] = info.weight
				nphi[ring]   = info.nph
				stride[ring] = info.stride
				offsets[ring]= info.ofs
				ring += 1
	cdef np.ndarray[np.int_t,ndim=1] order = np.argsort(offsets[:ring])
	return map_info(theta[order],nphi[order],phi0[order],offsets[order],stride[order],weight[order])

cdef map_info_from_geom_and_free(csharp.sharp_geom_info * geom):
	res = map_info_from_geom(geom)
	csharp.sharp_destroy_geom_info(geom)
	return res

cdef class alm_info:
	cdef csharp.sharp_alm_info * info
	cdef readonly int lmax
	cdef readonly int mmax
	cdef readonly int stride
	cdef readonly int nelem
	cdef readonly np.ndarray mstart
	def __cinit__(self, lmax=None, mmax=None, nalm=None, stride=1, layout="triangular"):
		"""Constructs a new sharp spherical harmonic coefficient layout information
		for the given lmax and mmax. The layout defaults to triangular, but
		can be changed by explicitly specifying layout, either as a string
		naming layout (triangular or rectangular), or as an array containing the
		index of the first l for each m. Once constructed, an alm_info is immutable.
		The layouts are all m-major, with all the ls for each m consecutive."""
		if lmax is not None: lmax = int(lmax)
		if mmax is not None: mmax = int(mmax)
		if nalm is not None: nalm = int(nalm)
		if isinstance(layout,basestring):
			if layout == "triangular" or layout == "tri":
				if lmax is None: lmax = nalm2lmax(nalm)
				if mmax is None: mmax = lmax
				m = np.arange(mmax+1)
				mstart = stride*(m*(2*lmax+1-m)//2)
			elif layout == "rectangular" or layout == "rect":
				if lmax is None: lmax = int(nalm**0.5)-1
				if mmax is None: mmax = lmax
				mstart = np.arange(mmax+1)*(lmax+1)*stride
			else:
				raise ValueError("unkonwn layout: %s" % layout)
		else:
			mstart = layout
		self.info  = make_alm_helper(lmax,mmax,stride,mstart)
		self.lmax  = lmax
		self.mmax  = mmax
		self.stride= stride
		self.nelem = np.max(mstart + (lmax+1)*stride)
		if nalm is not None:
			assert self.nelem == nalm, "lmax must be explicitly specified when lmax != mmax"
		self.mstart= mstart
		self.mstart.flags.writeable = False
	def lm2ind(self, l, m):
		return self.mstart[m]+l*self.stride
	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_map(self):
		"""Return the explicit [nelem,{l,m}] mapping this alm_info represents."""
		cdef np.ndarray[np.int64_t,ndim=2] mapping = np.empty([self.nelem,2],np.int64)
		cdef np.ndarray[np.int64_t,ndim=1] mstart = self.mstart
		cdef int l, m, i
		l, m = 0, 0
		for m in range(self.mmax+1):
			for l in range(m, self.lmax+1):
				i = mstart[m]+l*self.stride
				mapping[i,0] = l
				mapping[i,1] = m
		return mapping
	def transpose_alm(self, alm, out=None):
		"""In order to accomodate l-major ordering, which is not directoy
		supported by sharp, this function efficiently transposes Alm into
		Aml. If the out argument is specified, the transposed result will
		be written there. In order to perform an in-place transpose, call
		this function with the same array as "alm" and "out". If the out
		argument is not specified, then a new array will be constructed
		and returned."""
		if out is None: out = alm.copy()
		o2d = out.reshape(-1,out.shape[-1])
		# These are not really in-place at the moment: They still
		# use a large work array internally.
		if out.dtype == np.complex128:
			self.transpose_alm_dp(o2d)
		else:
			self.transpose_alm_sp(o2d)
		return out
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef transpose_alm_dp(self,np.ndarray[np.complex128_t,ndim=2] alm):
		cdef int l, m, i, j, comp
		cdef np.complex128_t v
		cdef np.ndarray[np.int64_t,ndim=1] mstart = self.mstart
		cdef np.ndarray[np.complex128_t,ndim=1] work = np.empty(alm.shape[1],alm.dtype)
		for comp in range(alm.shape[0]):
			l,m,i = 0,0,0
			for i in range(alm.shape[1]):
				j = mstart[m]+l*self.stride
				work[j] = alm[comp,i]
				m = m+1
				if m > self.mmax or m > l:
					l = l+1
					m = 0
			for i in range(alm.shape[1]):
				alm[comp,i] = work[i]
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef transpose_alm_sp(self,np.ndarray[np.complex64_t,ndim=2] alm):
		cdef int l, m, i, j, comp
		cdef np.complex64_t v
		cdef np.ndarray[np.int64_t,ndim=1] mstart = self.mstart
		cdef np.ndarray[np.complex64_t,ndim=1] work = np.empty(alm.shape[1],alm.dtype)
		for comp in range(alm.shape[0]):
			l,m = 0,0
			for i in range(alm.shape[1]):
				j = mstart[m]+l*self.stride
				work[j] = alm[comp,i]
				m = m+1
				if m > self.mmax or m > l:
					l = l+1
					m = 0
			for i in range(alm.shape[1]):
				alm[comp,i] = work[i]
	def alm2cl(self, alm, alm2=None):
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
		# A common use case is to compute TEBxTEB auto-cross spectra, where
		# e.g. TE === ET since alm1 is the same array as alm2. To avoid duplicate
		# calculations in this case we use a cache, which skips computing the
		# cross-spectrum of any given pair of arrays more than once.
		cache = {}
		if alm.dtype == np.complex64:
			cl = np.empty(alm.shape[:-1]+(self.lmax+1,), np.float32)
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
					csharp.alm2cl_sp(self.info, &alm_single_sp1[0], &alm_single_sp2[0], &cl_single_sp[0])
					cache[key] = cl[I]
			return cl
		elif alm.dtype == np.complex128:
			cl = np.empty(alm.shape[:-1]+(self.lmax+1,), np.float64)
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
					csharp.alm2cl_dp(self.info, &alm_single_dp1[0], &alm_single_dp2[0], &cl_single_dp[0])
					cache[key] = cl[I]
			return cl
		else:
			raise ValueError("Only complex64 and complex128 supported, but got %s" % str(alm.dtype))
	def lmul(self, alm, lmat, out=None):
		"""Computes res[a,lm] = lmat[a,b,l]*alm[b,lm], where lm is the position of the
		element with (l,m) in the alm array, as defined by this class."""
		if out is None: out = alm.copy()
		if out.ndim == 1: out = out[None]
		if lmat.ndim == 1:
			lmat = np.eye(out.shape[0])[:,:,None]*lmat
		lmat = lmat.astype(out.real.dtype, copy=False)
		if out.dtype == np.complex128:
			self.lmul_dp(out, lmat)
		else:
			self.lmul_sp(out, lmat)
		return out.reshape(alm.shape)
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef lmul_dp(self,np.ndarray[np.complex128_t,ndim=2] alm, np.ndarray[np.float64_t,ndim=3] lmat):
		cdef int l, m, lm, c1, c2, ncomp, lcap
		cdef np.ndarray[np.complex128_t,ndim=1] v
		cdef np.ndarray[np.int64_t,ndim=1] mstart = self.mstart
		l,m=0,0
		ncomp = alm.shape[0]
		lcap  = min(self.lmax+1,lmat.shape[2])
		v = np.empty(ncomp,dtype=np.complex128)
		#for m in prange(self.mmax+1,nogil=True,schedule="dynamic"):
		for m in range(self.mmax+1):
			for l in range(m, lcap):
				lm = mstart[m]+l*self.stride
				for c1 in range(ncomp):
					v[c1] = alm[c1,lm]
				for c1 in range(ncomp):
					alm[c1,lm] = 0
					for c2 in range(ncomp):
						alm[c1,lm] += lmat[c1,c2,l]*v[c2]
			# If lmat is too short, interpret missing values as zero
			for l in range(lcap, self.lmax+1):
				lm = mstart[m]+l*self.stride
				for c1 in range(ncomp):
					alm[c1,lm] = 0
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef lmul_sp(self,np.ndarray[np.complex64_t,ndim=2] alm, np.ndarray[np.float32_t,ndim=3] lmat):
		cdef int l, m, lm, c1, c2, ncomp, lcap
		cdef np.ndarray[np.complex64_t,ndim=1] v
		cdef np.ndarray[np.int64_t,ndim=1] mstart = self.mstart
		l,m=0,0
		ncomp = alm.shape[0]
		lcap  = min(self.lmax+1,lmat.shape[2])
		v = np.empty(ncomp,dtype=np.complex64)
		#for m in prange(self.mmax+1,nogil=True,schedule="dynamic"):
		for m in range(self.mmax+1):
			for l in range(m, lcap):
				lm = mstart[m]+l*self.stride
				for c1 in range(ncomp):
					v[c1] = alm[c1,lm]
				for c1 in range(ncomp):
					alm[c1,lm] = 0
					for c2 in range(ncomp):
						alm[c1,lm] += lmat[c1,c2,l]*v[c2]
			for l in range(lcap, self.lmax+1):
				lm = mstart[m]+l*self.stride
				for c1 in range(ncomp):
					alm[c1,lm] = 0
	def __dealloc__(self):
		csharp.sharp_destroy_alm_info(self.info)

def nalm2lmax(nalm):
	return int((-1+(1+8*nalm)**0.5)/2)-1

cdef class sht:
	cdef public map_info minfo
	cdef public alm_info ainfo
	def __cinit__(self, minfo, ainfo):
		"""Construct a sharp Spherical Harmonics Transform (SHT) object, which
		transforms between maps with pixellication given by the map_info "minfo"
		and spherical harmonic coefficents given by alm_info "ainfo"."""
		self.minfo, self.ainfo = minfo, ainfo
	def alm2map(self, alm, map=None, spin=0):
		"""Transform the given spherical harmonic coefficients "alm" into
		a map space. If a map is specified as the "map" argument, output
		will be written there. Otherwise, a new map will be constructed
		and returned. "alm" has dimensions [ntrans,nspin,nalm], or
		[nspin,nalm] or [nalm] where ntrans is the number of independent
		transforms to perform in parallel, nspin is the number of spin
		components per transform (1 or 2), and nalm is the number of coefficients
		per alm."""
		alm = np.asarray(alm)
		ntrans, nspin = dim_helper(alm, "alm")
		# Create a compatible output map
		if map is None:
			map = np.empty([ntrans,nspin,self.minfo.npix],dtype=alm.real.dtype)
			map = map.reshape(alm.shape[:-1]+(map.shape[-1],))
		else:
			assert alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree"
		execute(csharp.SHARP_ALM2MAP, self.ainfo, alm, self.minfo, map, spin=spin)
		return map
	def map2alm(self, map, alm=None, spin=0):
		"""Transform the given map "map" into a harmonic space. If "alm" is specified,
		output will be written there. Otherwise, a new alm will be constructed
		and returned. "map" has dimensions [ntrans,nspin,npix], or
		[nspin,npix] or [npix] where ntrans is the number of independent
		transforms to perform in parallel, nspin is the number of spin
		components per transform (1 or 2), and npix is the number of pixels per map."""
		map = np.asarray(map)
		ntrans, nspin = dim_helper(map, "map")
		# Create a compatible output map
		if alm is None:
			alm = np.empty([ntrans,nspin,self.ainfo.nelem],dtype=np.result_type(map.dtype,0j))
			alm = alm.reshape(map.shape[:-1]+(alm.shape[-1],))
		else:
			assert alm.shape[:-1]==map.shape[:-1], "all but last index of map and alm must agree"
		execute(csharp.SHARP_MAP2ALM, self.ainfo, alm, self.minfo, map, spin=spin)
		return alm
	def alm2map_der1(self, alm, map=None):
		"""Compute derivative maps for the given scalar alms. If "map" is passed,
		output will be written there. Otherwise a new map will be constructed
		and returned. "alm" has dimensions [ntrans,nalm] or [nalm], and the map
		has dimensions [ntrans,2,npix] or [2,npix], where ntrans is the number
		of transformations to perform in parallel, and npix is the number of pixels
		in the map."""
		alm = np.asarray(alm)
		ntrans = 1 if alm.ndim == 1 else alm.shape[0]
		if map is None:
			map = np.empty([ntrans,2,self.minfo.npix],dtype=alm.real.dtype)
			map = map.reshape(alm.shape[:-1]+map.shape[-2:])
		else:
			assert map.ndim >= 2, "map must be at least 2d"
			assert map.shape[-2] == 2, "Second to last dimensino of map must have length 2"
			assert alm.shape[:-1]==map.shape[:-2], "alm.shape[:-1] != map.shape[:-2]"
		execute(csharp.SHARP_ALM2MAP_DERIV1, self.ainfo, alm, self.minfo, map, spin=0)
		return map

# alm and map have the formats:
#  [nlm]:           spin=0,    ntrans=1
#  [ns,nlm]:        spin=ns>1, ntrans=1
#  [ntrans,ns,nlm]: spin=ns>1, ntrans=ntrans
# So to do many spin 0 transforms in parallel, you would pass alm with
# the shape [:,1,:], which can be created from a 2d alm by alm[:,None]
def dim_helper(a, name):
	assert a.ndim > 0 and a.ndim <= 3, name + " must be [nlm], [ntrf*ncomp,nlm] or [ntrf,ncomp,nlm]"
	if a.ndim == 1:
		ntrans, nspin = 1, 1
	elif a.ndim == 2:
		ntrans, nspin = 1, a.shape[0]
	elif a.ndim == 3:
		ntrans, nspin = a.shape[:2]
	assert nspin < 3, name + " spin axis must have length 1 or 2 (T and P must be done separately)"
	return ntrans, nspin

def execute(type, alm_info ainfo, alm, map_info minfo, map, spin):
	assert isinstance(alm, np.ndarray), "alm must be a numpy array"
	assert isinstance(map, np.ndarray), "map must be a numpy array"
	cdef int i
	ntrans, nspin = dim_helper(alm, "alm")
	ntrans, ncomp = dim_helper(map, "map")
	assert(spin == 0 and nspin != 1 or spin > 0 and nspin != 2,
		"Dimension -2 of maps and alms must be 2 for spin transforms and 1 for scalar transforms.")
	try:
		type = typemap[type]
	except KeyError:
		pass
	alm3 = alm.reshape(ntrans,nspin,-1)
	map3 = map.reshape(ntrans,ncomp,-1)
	if map.dtype == np.float64:
		execute_dp(type, spin, ainfo, alm3, minfo, map3)
	else:
		execute_sp(type, spin, ainfo, alm3, minfo, map3)

cpdef execute_dp(int type, int spin, alm_info ainfo, np.ndarray[np.complex128_t,ndim=3] alm, map_info minfo, np.ndarray[np.float64_t,ndim=3] map):
	cdef int ntrans = map.shape[0]
	cdef int nmap = map.shape[0]*map.shape[1]
	cdef int nalm = alm.shape[0]*alm.shape[1]
	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]      aptrs = np.empty(nalm,dtype=np.uintp)
	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]      mptrs = np.empty(nmap,dtype=np.uintp)
	cdef int i, j
	for i in range(ntrans):
		for j in range(alm.shape[1]):
			aptrs[i*alm.shape[1]+j] = <np.uintp_t>&alm[i,j,0]
		for j in range(map.shape[1]):
			mptrs[i*map.shape[1]+j] = <np.uintp_t>&map[i,j,0]
	execute_helper(type, ainfo, aptrs, minfo, mptrs, spin, ntrans, csharp.SHARP_DP)

cpdef execute_sp(int type, int spin, alm_info ainfo, np.ndarray[np.complex64_t,ndim=3,mode="c"] alm, map_info minfo, np.ndarray[np.float32_t,ndim=3,mode="c"] map):
	cdef int ntrans = map.shape[0]
	cdef int nmap = map.shape[0]*map.shape[1]
	cdef int nalm = alm.shape[0]*alm.shape[1]
	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]      aptrs = np.empty(nalm,dtype=np.uintp)
	cdef np.ndarray[np.uintp_t,ndim=1,mode="c"]      mptrs = np.empty(nmap,dtype=np.uintp)
	cdef int i, j
	for i in range(ntrans):
		for j in range(alm.shape[1]):
			aptrs[i*alm.shape[1]+j] = <np.uintp_t>&alm[i,j,0]
		for j in range(map.shape[1]):
			mptrs[i*map.shape[1]+j] = <np.uintp_t>&map[i,j,0]
	execute_helper(type, ainfo, aptrs, minfo, mptrs, spin, ntrans, 0)

cdef check_cont_dp(np.ndarray[np.float64_t,ndim=1,mode="c"] map_row, np.ndarray[np.complex128_t,ndim=1,mode="c"] alm_row): pass
cdef check_cont_sp(np.ndarray[np.float32_t,ndim=1,mode="c"] map_row, np.ndarray[np.complex64_t, ndim=1,mode="c"] alm_row): pass

typemap = { "map2alm": csharp.SHARP_MAP2ALM, "alm2map": csharp.SHARP_ALM2MAP, "alm2map_der1": csharp.SHARP_ALM2MAP_DERIV1 }

cdef execute_helper(int type,
		alm_info ainfo, np.ndarray[np.uintp_t,ndim=1] alm,
		map_info minfo, np.ndarray[np.uintp_t,ndim=1] map,
		int spin=0, int ntrans=1, int flags=0):
	csharp.sharp_execute(type, spin, <void*>&alm[0], <void*>&map[0],
			minfo.geom, ainfo.info, ntrans, flags, NULL, NULL)

cdef csharp.sharp_geom_info * make_geometry_helper(
		int ntheta,
		np.ndarray[int,ndim=1] nphi,
		np.ndarray[csharp.ptrdiff_t,ndim=1] offsets,
		np.ndarray[int,ndim=1] stride,
		np.ndarray[double,ndim=1] phi0,
		np.ndarray[double,ndim=1] theta,
		np.ndarray[double,ndim=1] weight):
	cdef csharp.sharp_geom_info * geom
	csharp.sharp_make_geom_info(ntheta, &nphi[0], &offsets[0], &stride[0], &phi0[0], &theta[0], &weight[0], &geom)
	return geom

cdef csharp.sharp_alm_info * make_alm_helper(int lmax, int mmax, int stride, np.ndarray[csharp.ptrdiff_t,ndim=1] mstart):
	cdef csharp.sharp_alm_info * info
	csharp.sharp_make_alm_info(lmax, mmax, stride, &mstart[0], &info)
	return info

cdef csharp.sharp_alm_info * make_triangular_alm_helper(int lmax, int mmax, int stride):
	cdef csharp.sharp_alm_info * info
	csharp.sharp_make_triangular_alm_info(lmax, mmax, stride, &info)
	return info
