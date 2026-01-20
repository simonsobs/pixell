"""This module provides functions for taking into account the curvature of the
full sky."""
from __future__ import print_function, division
import numpy as np, os, warnings
from . import enmap, powspec, wcsutils, utils, bunch

from . import cmisc
# Initialize DUCC's thread num variable from OMP's if it's not already set.
# This must be done before importing ducc0 for the first time. Doing this
# limits wasted memory from ducc allocating too big a thread pool. For computes
# with many cores, this can save GBs of memory.
utils.setenv("DUCC0_NUM_THREADS", utils.getenv("OMP_NUM_THREADS"), keep=True)
import ducc0

class ShapeError(Exception): pass

def rand_map(shape, wcs, ps, lmax=None, dtype=np.float64, seed=None, spin=[0,2], method="auto", verbose=False):
	"""Generates a CMB realization with the given power spectrum for an enmap
	with the specified shape and WCS. This is identical to enlib.rand_map, except
	that it takes into account the curvature of the full sky. This makes it much
	slower and more memory-intensive. The map should not cross the poles."""
	# Ensure everything has the right dimensions and restrict to relevant dimensions
	ps = utils.atleast_3d(ps)
	if not ps.shape[0] == ps.shape[1]: raise ShapeError("ps must be [ncomp,ncomp,nl] or [nl]")
	if not (len(shape) == 2 or len(shape) == 3): raise ShapeError("shape must be (ncomp,ny,nx) or (ny,nx)")
	ncomp = 1 if len(shape) == 2 else shape[-3]
	ps = ps[:ncomp,:ncomp]

	ctype = np.result_type(dtype,0j)
	if verbose: print("Generating alms with seed %s up to lmax=%d dtype %s" % (str(seed), lmax, np.dtype(ctype).char))
	alm   = rand_alm_healpy(ps, lmax=lmax, seed=seed, dtype=ctype)
	if verbose: print("Allocating output map shape %s dtype %s" % (str((ncomp,)+shape[-2:]), np.dtype(dtype).char))
	map   = enmap.empty((ncomp,)+shape[-2:], wcs, dtype=dtype)
	alm2map(alm, map, spin=spin, method=method, verbose=verbose)
	if len(shape) == 2: map = map[0]
	return map

def pad_spectrum(ps, lmax):
	ps  = np.asarray(ps)
	ops = np.zeros(ps.shape[:-1]+(lmax+1,),ps.dtype)
	ops[...,:ps.shape[-1]] = ps[...,:ps.shape[-1]]
	return ops

def rand_alm_healpy(ps, lmax=None, seed=None, dtype=np.complex128):
	import healpy
	if seed is not None: np.random.seed(seed)
	ps  = np.asarray(ps)
	if lmax is None: lmax = ps.shape[-1]-1
	# Handle various shaped input spectra
	if   ps.ndim == 1: wps = ps[None,None]
	elif ps.ndim == 2: wps = powspec.sym_expand(ps, scheme="diag")
	elif ps.ndim == 3: wps = ps
	else: raise ValueError("ps must be either [nl], [nspec,nl] or [ncomp,ncomp,nl] in rand_alm_healpy")
	# Flatten, since healpy wants only the non-redundant components in the diagonal-first scheme
	fps = powspec.sym_compress(wps, scheme="diag")
	alm = np.asarray(healpy.synalm(fps, lmax=lmax, new=True))
	# Produce scalar output for scalar inputs
	if ps.ndim == 1: alm = alm[0]
	return alm

def rand_alm(ps, ainfo=None, lmax=None, seed=None, dtype=np.complex128, m_major=True, return_ainfo=False):
	"""This is a replacement for healpy.synalm. It generates the random
	numbers in l-major order before transposing to m-major order in order
	to allow generation of low-res and high-res maps that agree on large
	scales. It uses 2/3 of the memory of healpy.synalm, and has comparable
	speed."""
	rtype      = np.zeros([0],dtype=dtype).real.dtype
	wps, ainfo = prepare_ps(ps, ainfo=ainfo, lmax=lmax)
	alm = rand_alm_white(ainfo, pre=[wps.shape[0]], seed=seed, dtype=dtype, m_major=m_major)
	# Scale alms by spectrum, taking into account which alms are complex
	ps12 = enmap.multi_pow(wps, 0.5)
	ainfo.lmul(alm, (ps12/2**0.5).astype(rtype, copy=False), alm)
	alm[:,:ainfo.lmax+1].imag  = 0
	alm[:,:ainfo.lmax+1].real *= 2**0.5
	if ps.ndim == 1: alm = alm[0]
	if return_ainfo: return alm, ainfo
	else: return alm

######################################
### Spherical harmonics transforms ###
######################################

def alm2map(alm, map, spin=[0,2], deriv=False, adjoint=False,
		copy=False, method="auto", ainfo=None, verbose=False, nthread=None,
		epsilon=1e-6, pix_tol=1e-6, locinfo=None, tweak=False):
	"""Spherical harmonics synthesis. Transform from harmonic space to real space.

	Parameters
	----------
	alm: complex64 or complex128 numpy array with shape [...,ncomp,nelem],
	 [ncomp,nelem] or [nelem]. Spin transforms will be applied to the ncomp
	 axis, controlled by the spin argument below.
	map: float32 or float64 enmap with shape [...,ncomp,ny,nx], [ncomp,ny,nx]
	 or [ny,nx]. All but last two dimensions must match alm.
	 Will be overwritten unless copy is True

	Options
	-------
	spin: list of spins. These describe how to handle the [ncomp] axis.
	 0: scalar transform. Consumes one element in the component axis
	 not 0: spin transform. Consumes two elements from the component axis.
	 For example, if you have a TEB alm [3,nelem] and want to transform it
	 to a TQU map [3,ny,nx], you would use spin=[0,2] to perform a scalar
	 transform for the T component and a spin-2 transform for the Q,U
	 components. Another example. If you had an alm [5,nelem] and map
	 [5,ny,nx] and the first element was scalar, the next pair spin-1
	 and the next pair spin-2, you woudl use spin=[0,1,2]. default:[0,2]
	deriv: If true, instead calculates the d/ddec and d/dra derivatives
	 of the map corresponding to the alms. In this case the alm must have
	 shape [...,nelem] or [nelem] and the map must have shape
	 [...,2,ny,nx] or [2,ny,nx]. default: False
	adjoint: If true, instead calculates the adjoint of the
	 alm2map operation. This reads from map and writes to alm. default: False
	copy: If true, writes to a copy of map instead of overwriting the
	 map argument. The resulting map is returned.
	method: Select the spherical harmonics transform method:
	 "2d": Use ducc's "2d" transforms. These are fast and accurate, but
	  require full-sky CAR maps with one of a limited set of pixel layouts
	  (CC, F1, MW, MWflip, DH, F2), see the ducc documentation. Maps with
	  partial sky coverage compatible with these pixelizations will be
	  temporarily padded to full sky before the transform. For other maps,
	  this method will fail.
	 "cyl": Use ducc's standard transforms. These work for any cylindrical
	  projection where pixels are equi-spaced and evenly divide the sky
	  along each horizontal line. Maps with partial sky coverage will be
	  temporarily padded horizontally as necessary.
	 "general": Use ducc's general transforms. These work for any pixelization,
	  but are significantly more expensive, both in terms of time and memory.
	 "auto": Automatically choose "2d", "cyl" or "general". This is the default.,
	ainfo: alm_info object containing information about the alm layout.
	 default: standard triangular layout,
	verbose: If True, prints information about what's being done
	nthread: Number of threads to use. Defaults to OMP_NUM_THREADS.
	epsilon: The desired fractional accuracy. Used for interpolation
	 in the "general" method. Default: 1e-6.
	pix_tol: Tolerance for matching a pixel layout with a predefined one,
	 in fractions of a pixel. Default: 1e-6.
	locinfo: Information about the coordinates and validity of each pixel.
	 Only relevant for the "general" method. Computed via calc_locinfo if missing.
	 If you're doing multiple transforms with the same geometry, you can
	 speed things up by precomputing this and passing it in here.

	Returns
	-------
	The resulting map. This will be the same object as the map argument,
	or a copy if copy == True."""
	if tweak: warnings.warn("The tweak argument is deprecated and does nothing after the libsharp→ducc transition. It will be removed in a future version")
	minfo = analyse_geometry(map.shape, map.wcs, tol=pix_tol)
	if method == "auto": method = get_method(map.shape, map.wcs, minfo=minfo)
	if   method == "2d":
		if verbose: print("method: 2d")
		return alm2map_2d(alm, map, ainfo=ainfo, minfo=minfo, spin=spin, deriv=deriv, copy=copy,
			verbose=verbose, adjoint=adjoint, nthread=nthread, pix_tol=pix_tol)
	elif method == "cyl":
		if verbose: print("method: cyl")
		return alm2map_cyl(alm, map, ainfo=ainfo, minfo=minfo, spin=spin, deriv=deriv, copy=copy,
			verbose=verbose, adjoint=adjoint, nthread=nthread, pix_tol=pix_tol)
	elif method == "general":
		if verbose: print("method: general")
		return alm2map_general(alm, map, ainfo=ainfo, spin=spin, deriv=deriv, copy=copy,
			verbose=verbose, adjoint=adjoint, nthread=nthread, epsilon=epsilon,
			locinfo=locinfo)
	else:
		raise ValueError("Unrecognized alm2map method '%s'" % str(method))

def alm2map_adjoint(map, alm=None, spin=[0,2], deriv=False,
		copy=False, method="auto", ainfo=None, verbose=False, nthread=None,
		epsilon=None, pix_tol=1e-6, locinfo=None):
	"""The adjoint of map2alm. Forwards to map2alm; see its docstring for details"""
	return alm2map(alm, map, spin=spin, deriv=deriv, adjoint=True,
		copy=copy, method=method, ainfo=ainfo, verbose=verbose, nthread=nthread,
		epsilon=epsilon, pix_tol=pix_tol, locinfo=locinfo)

def alm2map_pos(alm, pos=None, loc=None, ainfo=None, map=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, epsilon=None):
	"""Like alm2map, but operates directly on arbitrary positions instead of an enmap.
	The positions are given either with the pos argument or the loc argument.
	 pos: [{dec,ra},...] in radians
	 loc: [...,{codec,ra}] in radians. codec is pi/2 - dec, ra must be positive
	The underlying implementation uses loc, so if pos is passed an internal loc will be
	built. See alm2map for the meaning of the other arguments."""
	if adjoint:
		if copy and alm is not None: alm = alm.copy()
	else:
		if copy and map is not None: map = map.copy()
	if loc is None:
		# The disadvantage of passing pos instead of loc is that we end up
		# making a copy in the convention ducc wants
		loc = np.moveaxis(np.asarray(pos),0,-1).copy(order="C")
		# This should use less memory than writing loc[:,0] = np.pi/2-loc[:,0]
		loc[...,0] *= -1
		loc[...,0] += np.pi/2
		# Should use rewind here, but this is more efficient
		loc[loc[...,1]<0,1] += 2*np.pi
		# Support arbitrary pre-dimensions for loc (post-dimensions for pos)
	lpre = loc.shape[:-1]
	loc  = loc.reshape(-1,2)
	if deriv: oshape = alm.shape[:-1]+(2,len(loc))
	else:     oshape = alm.shape[:-1]+(len(loc),)
	if map is None:
		map = np.zeros(oshape, utils.real_dtype(alm.dtype))
	for I in utils.nditer(map.shape[:-2]):
		alm2map_raw_general(alm[I], map[I], loc, ainfo=ainfo, spin=spin, deriv=deriv,
				nthread=nthread, verbose=verbose, epsilon=epsilon, adjoint=adjoint)
	# Reshape to reflect the dimensions pos/loc
	map = map.reshape(map.shape[:-1]+lpre)
	if adjoint: return alm
	else:       return map

def map2alm(map, alm=None, lmax=None, spin=[0,2], deriv=False, adjoint=False,
		copy=False, method="auto", ainfo=None, verbose=False, nthread=None,
		niter=0, epsilon=None, pix_tol=1e-6, weights=None, locinfo=None, tweak=False):
	"""Spherical harmonics analysis. Transform from real space to harmonic space.

	Parameters
	----------
	map: float32 or float64 enmap with shape [...,ncomp,ny,nx], [ncomp,ny,nx]
	 or [ny,nx]. All but last two dimensions must match alm.
	alm: complex64 or complex128 numpy array with shape [...,ncomp,nelem],
	 [ncomp,nelem] or [nelem]. Spin transforms will be applied to the ncomp
	 axis, controlled by the spin argument below.
	 Will be overwritten unless copy is True

	Options
	-------
	spin: list of spins. These describe how to handle the [ncomp] axis.
	 0: scalar transform. Consumes one element in the component axis
	 not 0: spin transform. Consumes two elements from the component axis.
	 For example, if you have a TEB alm [3,nelem] and want to transform it
	 to a TQU map [3,ny,nx], you would use spin=[0,2] to perform a scalar
	 transform for the T component and a spin-2 transform for the Q,U
	 components. Another example. If you had an alm [5,nelem] and map
	 [5,ny,nx] and the first element was scalar, the next pair spin-1
	 and the next pair spin-2, you woudl use spin=[0,1,2]. default:[0,2]
	deriv: If true, instead calculates the d/ddec and d/dra derivatives
	 of the map corresponding to the alms. In this case the alm must have
	 shape [...,nelem] or [nelem] and the map must have shape
	 [...,2,ny,nx] or [2,ny,nx]. default: False
	adjoint: If true, instead calculates the adjoint of the
	 map2alm operation. This reads from alm and writes to map. default: False
	copy: If true, writes to a copy of map instead of overwriting the
	 map argument. The resulting map is returned.
	method: Select the spherical harmonics transform method:
	 "2d": Use ducc's "2d" transforms. These are fast and accurate, but
	  require full-sky CAR maps with one of a limited set of pixel layouts
	  (CC, F1, MW, MWflip, DH, F2), see the ducc documentation. Maps with
	  partial sky coverage compatible with these pixelizations will be
	  temporarily padded to full sky before the transform. For other maps,
	  this method will fail.
	 "cyl": Use ducc's standard transforms. These work for any cylindrical
	  projection where pixels are equi-spaced and evenly divide the sky
	  along each horizontal line. Maps with partial sky coverage will be
	  temporarily padded horizontally as necessary.
	 "general": Use ducc's general transforms. These work for any pixelization,
	  but are significantly more expensive, both in terms of time and memory.
	 "auto": Automatically choose "2d", "cyl" or "general". This is the default.,
	ainfo: alm_info object containing information about the alm layout.
	 default: standard triangular layout,
	verbose: If True, prints information about what's being done
	nthread: Number of threads to use. Defaults to OMP_NUM_THREADS.
	niter: The number of Jacobi iteration steps to perform when
	 estimating the map2alm integral. Should ideally be controlled via epsilon,
	 but is manual for now. Only relevant for the "cyl" and "general" methods.
	 Time proportional to 1+2*niter. For a flat spectrum, niter=0 typically results in
	 std(alm-alm_true)/std(alm_true) ≈ 1e-5, improving to 1e-8 by niter=3.
	 Default: 0
	epsilon: The desired fractional accuracy. Used for interpolation
	 in the "general" method. Default: 1e-6.
	pix_tol: Tolerance for matching a pixel layout with a predefined one,
	 in fractions of a pixel. Default: 1e-6.
	weights: Integration weights to use. Only used for methods "cyl" and "general".
	 Defaults to ducc's grid weights if available, otherwise the pixel area.
	 Somewhat heavy to compute and store for the "general" method, so if you're
	 performing multiple map2alm operations with the same geometry, consider
	 precomputing them and passing them with this argument. Must have the
	 same shape as locinfo.loc for the "general" method.
	locinfo: Information about the coordinates and validity of each pixel.
	 Only relevant for the "general" method. Computed via calc_locinfo if missing.
	 If you're doing multiple transforms with the same geometry, you can
	 speed things up by precomputing this and passing it in here.
	Returns
	-------
	The resulting alm. This will be the same object as the alm argument,
	or a copy if copy == True."""
	if tweak: warnings.warn("The tweak argument is deprecated and does nothing after the libsharp→ducc transition. It will be removed in a future version")
	minfo = analyse_geometry(map.shape, map.wcs, tol=pix_tol)
	if method == "auto": method = get_method(map.shape, map.wcs, minfo=minfo)
	if   method == "2d":
		if verbose: print("method: 2d")
		return map2alm_2d(map, alm, ainfo=ainfo, minfo=minfo, lmax=lmax, spin=spin, deriv=deriv, copy=copy,
			verbose=verbose, adjoint=adjoint, nthread=nthread, pix_tol=pix_tol)
	elif method == "cyl":
		if verbose: print("method: cyl")
		return map2alm_cyl(map, alm, ainfo=ainfo, minfo=minfo, lmax=lmax, spin=spin, deriv=deriv, copy=copy,
			verbose=verbose, adjoint=adjoint, nthread=nthread, niter=niter,
			pix_tol=pix_tol, weights=weights)
	elif method == "general":
		if verbose: print("method: pos")
		return map2alm_general(map, alm, ainfo=ainfo, lmax=lmax, spin=spin, deriv=deriv, copy=copy,
			verbose=verbose, adjoint=adjoint, nthread=nthread, epsilon=epsilon,
			locinfo=locinfo, weights=weights)
	else:
		raise ValueError("Unrecognized alm2map method '%s'" % str(method))

def map2alm_adjoint(alm, map, lmax=None, spin=[0,2], deriv=False,
		copy=False, method="auto", ainfo=None, verbose=False, nthread=None,
		niter=0, epsilon=1e-6, pix_tol=1e-6, weights=None, locinfo=None):
	"""The adjoint of alm2map. Forwards to map2alm. See its docstring for details"""
	return map2alm(map=map, alm=alm, lmax=lmax, spin=spin, deriv=deriv, adjoint=True,
		copy=copy, method=method, ainfo=ainfo, verbose=verbose, nthread=nthread,
		niter=niter, epsilon=epsilon, pix_tol=pix_tol, weights=weights, locinfo=locinfo)

def alm2map_healpix(alm, healmap=None, spin=[0,2], deriv=False, adjoint=False,
		copy=False, ainfo=None, nside=None, theta_min=None, theta_max=None, nthread=None):
	"""Projects the given alm[...,ncomp,nalm] onto the given healpix map
	healmap[...,ncomp,npix]."""
	dtype      = utils.native_dtype(utils.real_dtype(alm.dtype))
	alm, ainfo = prepare_alm(alm, ainfo, dtype=dtype)
	healmap    = prepare_healmap(healmap, nside, alm.shape[:-1], dtype)
	alm_full   = utils.atleast_Nd(alm, 2 if deriv else 3)
	map_full   = utils.atleast_Nd(healmap, 3)
	alm_full   = utils.fix_zero_strides(alm_full)
	map_full   = utils.fix_zero_strides(map_full)
	# Check if shapes are consistent
	if deriv and (alm_full.shape[:-1] != map_full.shape[:-2] or map_full.shape[-2] != 2):
		raise ValueError("When deriv is True, alm must have shape [...,nelem] and map shape [...,2,npix]")
	if not deriv and (alm_full.shape[:-1] != map_full.shape[:-1]):
		raise ValueError("alm must have shape [...,[ncomp,]nelem] and map shape [...,[ncomp,]npix]")
	if adjoint: func = ducc0.sht.experimental.adjoint_synthesis
	else:       func = ducc0.sht.experimental.synthesis
	nside   = npix2nside(map_full.shape[-1])
	rinfo   = get_ring_info_healpix(nside)
	rinfo   = apply_minfo_theta_lim(rinfo, theta_min, theta_max)
	nthread = int(utils.fallback(utils.getenv("OMP_NUM_THREADS", nthread),0))
	kwargs  = {"theta":rinfo.theta, "nphi":rinfo.nphi, "phi0":rinfo.phi0,
		"ringstart":rinfo.offsets, "lmax":ainfo.lmax, "mmax":ainfo.mmax,
		"mstart": ainfo.mstart, "nthreads":nthread}
	# Loop over pre-dimensions
	for I in utils.nditer(map_full.shape[:-2]):
		if deriv:
			func(alm=alm_full[I], map=map_full[I], mode="DERIV1", spin=1, **kwargs)
			# Flip sign of theta derivative to get dec derivative
			map_full[I+(0,)] *= -1
		else:
			for s, j1, j2 in enmap.spin_helper(spin, alm_full[I].shape[-2]):
				Ij = I+(slice(j1,j2),)
				func(alm=alm_full[Ij], map=map_full[Ij], spin=s, **kwargs)
	if adjoint: return alm
	else:       return healmap

def map2alm_healpix(healmap, alm=None, ainfo=None, lmax=None, spin=[0,2], weights=None, deriv=False, copy=False, verbose=False, adjoint=False, niter=0, theta_min=None, theta_max=None, nthread=None):
	"""map2alm for healpix maps. Similar to healpy's map2alm. See the map2alm docstring for details."""
	if adjoint:
		if copy and map is not None: map = map.copy()
	else:
		if copy and alm is not None: alm = alm.copy()
	alm, ainfo = prepare_alm(alm=alm, ainfo=ainfo, lmax=lmax, pre=healmap.shape[:-1], dtype=utils.native_dtype(healmap.dtype))
	alm_full   = utils.atleast_Nd(alm, 2 if deriv else 3)
	map_full   = utils.atleast_Nd(healmap, 3)
	alm_full   = utils.fix_zero_strides(alm_full)
	map_full   = utils.fix_zero_strides(map_full)
	nside      = npix2nside(map_full.shape[-1])
	rinfo      = get_ring_info_healpix(nside)
	rinfo      = apply_minfo_theta_lim(rinfo, theta_min, theta_max)
	nthread    = int(utils.fallback(utils.getenv("OMP_NUM_THREADS",nthread),0))
	kwargs     = {"theta":rinfo.theta, "nphi":rinfo.nphi, "phi0":rinfo.phi0,
		"ringstart":rinfo.offsets, "lmax":ainfo.lmax, "mmax":ainfo.mmax,
		"mstart": ainfo.mstart, "nthreads":nthread}
	if weights is None: weights = 4*np.pi/rinfo.npix
	# Helper for weights multiplication
	def wmul(map_flat, weights): return map_flat*weights
	# Iterate over all the predimensions
	for I in utils.nditer(map_full.shape[:-2]):
		if deriv:
			def Y(alm):   return ducc0.sht.experimental.synthesis(alm=alm, mode="DERIV1", spin=1, **kwargs)
			def YT(map):  return ducc0.sht.experimental.adjoint_synthesis(map=map, mode="DERIV1", spin=1, **kwargs)
			def YTW(map): return YT(wmul(map,weights))
			def WY(alm):  return wmul(Y(alm),weights)
			decflip = np.array([-1,1])[:,None,None]
			if adjoint: map_full[I] = jacobi_inverse(YT, WY, utils.fix_zero_strides(alm_full[I][None]), niter=niter)*decflip
			# does this need an [0] at the end like the other versions have?
			else:       alm_full[I] = jacobi_inverse(Y, YTW, map_full[I]*decflip, niter=niter)
		else:
			for s, j1, j2 in enmap.spin_helper(spin, alm_full.shape[-2]):
				Ij = I+(slice(j1,j2),)
				def Y(alm):   return ducc0.sht.experimental.synthesis(alm=alm, spin=s, **kwargs)
				def YT(map):  return ducc0.sht.experimental.adjoint_synthesis(map=map, spin=s, **kwargs)
				def YTW(map): return YT(wmul(map,weights))
				def WY(alm):  return wmul(Y(alm),weights)
				if adjoint: map_full[Ij] = jacobi_inverse(YT, WY, alm_full[Ij], niter=niter)
				else:       alm_full[Ij] = jacobi_inverse(Y, YTW, map_full[Ij], niter=niter)
	if adjoint: return healmap
	else:       return alm

# Class used to specify alm layout. Compatible with the old one from the libsharp-based
# implementation.
class alm_info:
	def __init__(self, lmax=None, mmax=None, nalm=None, stride=1, layout="triangular"):
		"""Constructs a new spherical harmonic coefficient layout information
		for the given lmax and mmax. The layout defaults to triangular, but
		can be changed by explicitly specifying layout, either as a string
		naming layout (triangular or rectangular), or as an array containing the
		index of the first l for each m. Can be used as the ainfo argument in map2alm
		and alm2map."""
		if lmax is not None: lmax = int(lmax)
		if mmax is not None: mmax = int(mmax)
		if nalm is not None: nalm = int(nalm)
		if isinstance(layout,str):
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
		self.lmax  = lmax
		self.mmax  = mmax
		self.stride= int(stride)
		self.nelem = int(np.max(mstart) + (lmax+1)*stride)
		self.nreal = lmax**2+2*lmax+2
		if nalm is not None:
			assert self.nelem == nalm, "lmax must be explicitly specified when lmax != mmax"
		self.mstart= mstart.astype(np.uint64, copy=False)
	@property
	def nl(self): return self.lmax+1
	@property
	def nm(self): return self.mmax+1
	def lm2ind(self, l, m):
		return (self.mstart[m].astype(int, copy=False)+l*self.stride).astype(int, copy=False)
	def get_map(self):
		"""Return the explicit [nelem,{l,m}] mapping this alm_info represents."""
		raise NotImplementedError
	def transpose_alm(self, alm, out=None):
		"""In order to accomodate l-major ordering, which is not directly
		supported, this function efficiently transposes Alm into
		Aml. If the out argument is specified, the transposed result will
		be written there. In order to perform an in-place transpose, call
		this function with the same array as "alm" and "out". If the out
		argument is not specified, then a new array will be constructed
		and returned."""
		return cmisc.transpose_alm(self, alm, out=out)
	def alm2cl(self, alm, alm2=None):
		"""Computes the cross power spectrum for the given alm and alm2, which
		must have the same dtype and broadcast. For example, to get the TEB,TEB
		cross spectra for a single map you would do
		 cl = ainfo.alm2cl(alm[:,None,:], alm[None,:,:])
		To get the same TEB,TEB spectra crossed with a different map it would
		be
		 cl = ainfo.alm2cl(alm1[:,None,:], alm2[None,:,:])
		In both these cases the output will be [{T,E,B},{T,E,B},nl].
        The returned cls start at ell=0."""
		return cmisc.alm2cl(self, alm, alm2=alm2)
	def lmul(self, alm, lmat, out=None):
		"""Computes res[a,lm] = lmat[a,b,l]*alm[b,lm], where lm is the position of the
		element with (l,m) in the alm array, as defined by this class."""
		return cmisc.lmul(self, alm, lmat, out=out)
	def __repr__(self):
		return "alm_info(lmax=%s,mmax=%s,mstart=%s)" % (str(self.lmax),str(self.mmax),str(self.mstart))

def get_method(shape, wcs, minfo=None, pix_tol=1e-6):
	"""Return which method map2alm and alm2map will use for the given
	enmap geometry. Returns either "2d", "cyl" or "general"."""
	if minfo is None: minfo = analyse_geometry(shape, wcs, tol=pix_tol)
	# Decide which method to use. Some cyl cases can be handled with 2d.
	# Consider doing that in the future. Not that important for alm2map,
	# but could help for map2alm.
	if   minfo.case == "general": method = "general"
	elif minfo.case == "2d":      method = "2d"
	else:                         method = "cyl"
	return method

# Quadrature weights

def quad_weights(shape, wcs, pix_tol=1e-6):
	"""Return the quadrature weights to use for map2alm operations for the given geometry.
	Only valid for a limited number of cylindrical geometries recognized by ducc. Returns
	weights[ny] where ny is shape[-2]. For cases where quadrature weights aren't available,
	it's a pretty good approximation to just use the pixel area."""
	minfo = analyse_geometry(shape, wcs, tol=pix_tol)
	if minfo.ducc_geo.name is None:
		raise ValueError("Quadrature weights not available for geometry %s,%s" % (str(shape),str(wcs)))
	ny      = shape[-2]+np.sum(minfo.ypad)
	weights = ducc0.sht.experimental.get_gridweights(minfo.ducc_geo.name, ny)
	weights = weights[minfo.ypad[0]:len(weights)-minfo.ypad[1]]
	if minfo.flip: weights = weights[::-1]
	weights/= minfo.ducc_geo.nx
	return weights

#####################
### 1d Transforms ###
#####################

def profile2harm(br, r, lmax=None, oversample=1, left=None, right=None):
	"""This is an alternative to healpy.beam2bl. In my tests it's a bit more
	accurate and about 3x faster, most of which is spent constructing
	the quadrature. It does use some interpolation internally, though, so
	there might be cases where it's less accurate. Transforms the
	function br(r) to bl(l). br has shape [...,nr], and the output will have
	shape [...,nl]. Implemented using sharp SHTs with one
	pixel per row and mmax=0. r is in radians and must be in ascending order."""
	br    = np.asarray(br)
	r     = np.asarray(r)
	# 1. We will implement this using a SHT. Start by setting up its parameters
	# Clenshaw-curtis sample points
	dr    = (r[-1]-r[0])/(len(r)-1)
	nfull = utils.nint(np.pi/dr)+1
	dr    = np.pi/(nfull-1)
	ncut  = int(np.ceil(r[-1]/dr))
	if lmax is None: lmax = int(nfull//2-1)
	l     = np.arange(lmax+1)
	rinfo = get_ring_info_radial(np.arange(ncut)*dr)
	# Get the ring weights
	weights = ducc0.sht.experimental.get_gridweights("CC", nfull)[:ncut]
	# This is to support br[...,nr] instead of just br[nr]
	harm  = np.zeros(br.shape[:-1]+(lmax+1,), br.dtype)
	for I in utils.nditer(br.shape[:-1]):
		# 2. Interpolate br to the rinfo geometry. Simple linear interpolation.
		map = np.interp(rinfo.theta, r, br[I], left=left, right=right).reshape(1,-1)
		alm = ducc0.sht.experimental.adjoint_synthesis(
				map=map*weights, theta=rinfo.theta, nphi=rinfo.nphi,
				phi0=rinfo.phi0, ringstart=rinfo.offsets, spin=0, lmax=lmax, mmax=0)[0]
		harm[I] = alm.real * (4*np.pi/(2*l+1))**0.5
	return harm

def harm2profile(bl, r):
	"""The inverse of profile2beam or healpy.beam2bl. *Much* faster
	than these (150x faster in my test case). Should be exact too."""
	bl = np.asarray(bl)
	r  = np.asarray(r)
	l  = np.arange(bl.shape[-1])
	rinfo = get_ring_info_radial(r)
	alm   = bl * ((2*l+1)/(4*np.pi))**0.5 + 0j
	br    = np.zeros(bl.shape[:-1]+(r.size,), bl.dtype)
	for I in utils.nditer(bl.shape[:-1]):
		ducc0.sht.experimental.synthesis(
				alm=alm[I][None], map=br[I][None], theta=rinfo.theta, nphi=rinfo.nphi, phi0=rinfo.phi0,
				ringstart=rinfo.offsets, spin=0, lmax=bl.shape[-1]-1, mmax=0)[0]
	return br

def prof2alm(profile, dir=[0, np.pi/2], spin=0, geometry="CC", nthread=None, norot=False):
	"""Calculate the alms for a 1d equispaced profile[...,n] oriented along the
	given [ra,dec] on the sky."""
	nthread= int(utils.fallback(utils.getenv("OMP_NUM_THREADS",nthread),0))
	profile= np.asarray(profile)
	dtype  = profile.dtype
	lmax   = get_ducc_maxlmax(geometry, profile.shape[-1])
	# Set up output arrays
	iainfo = alm_info(lmax=lmax, mmax=0)
	oainfo = alm_info(lmax=lmax, mmax=lmax if not norot else 0)
	ctype  = utils.complex_dtype(dtype)
	oalm   = np.zeros(profile.shape[:-1]+(oainfo.nelem,), ctype)
	for s, I in enmap.spin_pre_helper(spin, profile.shape[:-1]):
		# ducc has problems with None-axes, so fix that
		prof   = utils.fix_zero_strides(profile[I][...,None])
		alm    = ducc0.sht.experimental.analysis_2d(map=prof, spin=s, lmax=lmax, mmax=0, geometry=geometry, nthreads=nthread)
		if not norot:
			# Expand to full mmax to prepare for rotation
			alm    = transfer_alm(iainfo, alm, oainfo)
			# Rotate to target coordinate system
			alm    = rotate_alm(alm, 0, np.pi/2-dir[1], dir[0], nthread=nthread)
		oalm[I] = alm
	return oalm

#####################
###### Helpers ######
#####################

def npix2nside(npix):
	return utils.nint((npix/12)**0.5)

def prepare_healmap(healmap, nside=None, pre=(), dtype=np.float64):
	if healmap is not None: return healmap
	return np.zeros(pre + (12*nside**2,), dtype)

def apply_minfo_theta_lim(minfo, theta_min=None, theta_max=None):
	if theta_min is None and theta_max is None: return minfo
	mask = np.full(minfo.nrow, True, bool)
	if theta_min is not None: mask &= minfo.theta >= theta_min
	if theta_max is not None: mask &= minfo.theta <= theta_max
	res = minfo.copy()
	for key in ["theta", "nphi", "phi0"]: res[key] = res[key][mask]
	return res

def fill_gauss(arr, bsize=0x10000):
	rtype = np.zeros([0],arr.dtype).real.dtype
	arr   = arr.reshape(-1).view(rtype)
	for i in range(0, arr.size, bsize):
		arr[i:i+bsize] = np.random.standard_normal(min(bsize,arr.size-i))

def prepare_ps(ps, ainfo=None, lmax=None):
	ps    = np.asarray(ps)
	if ainfo is None:
		if lmax is None: lmax = ps.shape[-1]-1
		if lmax > ps.shape[-1]-1: ps = pad_spectrum(ps, lmax)
		ainfo = alm_info(lmax)
	if   ps.ndim == 1: wps = ps[None,None]
	elif ps.ndim == 2: wps = powspec.sym_expand(ps, scheme="diag")
	elif ps.ndim == 3: wps = ps
	else: raise ValueError("power spectrum must be [nl], [nspec,nl] or [ncomp,ncomp,nl]")
	return wps, ainfo

def rand_alm_white(ainfo, pre=None, alm=None, seed=None, dtype=np.complex128, m_major=True):
	if seed is not None:     np.random.seed(seed)
	if alm is None:
		if pre is None: alm = np.empty(ainfo.nelem, dtype)
		else:           alm = np.empty(tuple(pre)+(ainfo.nelem,), dtype)
	fill_gauss(alm)
	# Transpose numbers to make them m-major.
	if m_major: ainfo.transpose_alm(alm,alm)
	return alm

def almxfl(alm,lfilter=None,ainfo=None,out=None):
	"""Filter alms isotropically. Unlike healpy (at time of writing),
	this function allows leading dimensions in the alm, and also allows
	the filter to be specified as a function instead of an array.

	Args:
	    alm: (...,N) ndarray of spherical harmonic alms
	    lfilter: either an array containing the 1d filter to apply starting with ell=0
	    and separated by delta_ell=1, or a function mapping multipole ell to the 
	    filtering expression.
	    ainfo: If ainfo is provided, it is an alm_info describing the layout 
	    of the input alm. Otherwise it will be inferred from the alm itself.

	Returns:
	    falm: The filtered alms a_{l,m} * lfilter(l)
	"""
	alm   = np.asarray(alm)
	ainfo = alm_info(nalm=alm.shape[-1]) if ainfo is None else ainfo
	if callable(lfilter):
		l = np.arange(ainfo.lmax+1.0)
		lfilter = lfilter(l)
	return ainfo.lmul(alm, lfilter, out=out)

def filter(imap,lfilter,ainfo=None,lmax=None):
	"""Filter a map isotropically by a function.
	Returns alm2map(map2alm(alm * lfilt(ell),lmax))

	Args:
	    imap: (...,Ny,Nx) ndmap stack of enmaps.
	    lmax: integer specifying maximum multipole beyond which the alms are zeroed
	    lfilter: either an array containing the 1d filter to apply starting with ell=0
	    and separated by delta_ell=1, or a function mapping multipole ell to the 
	    filtering expression.
	    ainfo: If ainfo is provided, it is an alm_info describing the layout 
	of the input alm. Otherwise it will be inferred from the alm itself.

	Returns:
	    omap: (...,Ny,Nx) ndmap stack of filtered enmaps
	"""
	return alm2map(almxfl(map2alm(imap,ainfo=ainfo,lmax=lmax,spin=0),lfilter=lfilter,ainfo=ainfo),enmap.empty(imap.shape,imap.wcs,dtype=imap.dtype),spin=0,ainfo=ainfo)


def alm2cl(alm, alm2=None, ainfo=None):
	"""Compute the power spectrum for alm, or if alm2 is given, the cross-spectrum
	between alm and alm2, which must broadcast.

	Some example usage, where the notation a[{x,y,z},n,m] specifies that the array
	a has shape [3,n,m], and the 3 entries in the first axis should be interpreted
	as x, y and z respectively.

	1. cl[nl] = alm2cl(alm[nalm])
	   This just computes the standard power spectrum of the given alm, resulting in
	   a single 1d array.
	2. cl[nl] = alm2cl(alm1[nalm], alm2[nalm])
	   This compues the 1d cross-spectrum between the 1d alms alm1 and alm2.
	3. cl[{T,E,B},{T,E,B},nl] = alm2cl(alm[{T,E,B},None,nalm], alm[None,{T,E,B},nalm])
	   This computes the 3x3 polarization auto-spectrum for a 2d polarized alm.
	4. cl[{T,E,B},{T,E,B},nl] = alm2cl(alm1[{T,E,B},None,nalm], alm2[None,{T,E,B},nalm])
	   As above, but gives the 3x3 polarization cross-spectrum between two 2d alms.

	The output is in the shape one would expect from numpy broadcasting. For example,
	in the last example, the TE power spectrum would be found in cl[0,1], and the
	ET power spectrum (which is different for the cross-spectrum case) is in cl[1,0].
	If a Healpix-style compressed spectrum is desired, use pixell.powspec.sym_compress.
	"""
	alm = np.asarray(alm)
	ainfo = alm_info(nalm=alm.shape[-1]) if ainfo is None else ainfo
	return ainfo.alm2cl(alm, alm2=alm2)

euler_angs={}
euler_angs[("gal","equ")] = np.array([57.06793215,  62.87115487, -167.14056929])*utils.degree
euler_angs[("equ","gal")] = -euler_angs[("gal","equ")][::-1]
def rotate_alm(alm, psi, theta, phi, lmax=None, method="auto", nthread=None, inplace=False):
	"""Rotate the given alm[...,:] via the zyz rotations given by euler angles
	psi, theta and phi. See curvedsky.euler_angs for some predefined angles.
	The underlying implementation is provided by ducc0 or healpy. This is controlled
	with the "method" argument, which can be "ducc0", "healpy" or "auto". For "auto"
	it uses ducc0 if available, otherwise healpy. The resulting alm is returned.
	If inplace=True, then the input alm will be modified in place (but still returned).
	The number of threads to use is controlled with the nthread argument. If this is
	0 (the default), then the number of threads is given by the value of the OMP_NUM_THREADS
	variable."""
	if not inplace:  alm  = alm.copy()
	if lmax is None: lmax = nalm2lmax(alm.shape[-1])
	if method == "auto": method = utils.first_importable("ducc0", "healpy")
	if method == "ducc0":
		nthread = int(utils.fallback(utils.getenv("OMP_NUM_THREADS",nthread),0))
		for I in utils.nditer(alm.shape[:-1]):
			alm[I] = ducc0.sht.rotate_alm(alm[I], lmax=lmax, psi=psi, theta=theta, phi=phi, nthreads=nthread)
	elif method == "healpy":
		import healpy
		for I in utils.nditer(alm.shape[:-1]):
			healpy.rotate_alm(alm[I], lmax=lmax, psi=psi, theta=theta, phi=phi)
	elif method is None:
		raise ValueError("No rotate_alm implementations found")
	else:
		raise ValueError("Unrecognized rotate_alm implementation '%s'" % str(method))
	return alm

def transfer_alm(iainfo, ialm, oainfo, oalm=None, op=lambda a,b:b):
	"""Copy data from ialm with layout given by iainfo to oalm with layout
	given by oainfo. If oalm is not passed, it will be allocated. In either
	case oalm is returned. If op is specified, then it defines out oalm
	is updated: oalm = op(ialm, oalm). For example, if op = lambda a,b:a+b,
	then ialm would be added to oalm instead of overwriting it."""
	return cmisc.transfer_alm(iainfo, ialm, oainfo, oalm=oalm, op=op)

##############################
### Implementation details ###
##############################

def alm2map_2d(alm, map, ainfo=None, minfo=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, pix_tol=1e-6):
	"""Helper function for alm2map. See its docstring for details"""
	if copy:
		if adjoint and alm is not None: alm = alm.copy()
		else:       map = map.copy()
	if adjoint: alm, ainfo = prepare_alm(alm=alm, ainfo=ainfo, pre=map.shape[:-2], dtype=utils.native_dtype(map.dtype))
	minfo = analyse_geometry(map.shape, map.wcs, tol=pix_tol)
	# Loop over pre-pre-dimensions. ducc usually doesn't do anything clever with
	# these, so looping in python is cheap
	for I in utils.nditer(map.shape[:-3]):
		# Pad as necessary
		pad  = ((minfo.ypad[0],minfo.xpad[0]),(minfo.ypad[1],minfo.xpad[1]))
		tmap = map2buffer(map[I], minfo.flip, pad)
		alm2map_raw_2d(alm[I], tmap, ainfo=ainfo, spin=spin, deriv=deriv, nthread=nthread, verbose=verbose, adjoint=adjoint)
		# Copy out if necessary
		if not adjoint:
			map[I] = buffer2map(tmap, minfo.flip, pad)
	if adjoint: return alm
	else:       return map

def alm2map_cyl(alm, map, ainfo=None, minfo=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, pix_tol=1e-6):
	"""Helper function for alm2map. See its docstring for details"""
	if copy:
		if adjoint and alm is not None: alm = alm.copy()
		else:       map = map.copy()
	if adjoint: alm, ainfo = prepare_alm(alm=alm, ainfo=ainfo, pre=map.shape[:-2], dtype=utils.native_dtype(map.dtype))
	if minfo is None: minfo = analyse_geometry(map.shape, map.wcs, tol=pix_tol)
	# Loop over pre-pre-dimensions. ducc usually doesn't do anything clever with
	# these, so looping in python is cheap
	for I in utils.nditer(map.shape[:-3]):
		# Unlike 2d, cyl is fine with a band around the sky, so y padding is not needed
		pad  = ((0,minfo.xpad[0]),(0,minfo.xpad[1]))
		tmap = map2buffer(map[I], minfo.flip, pad, obuf=not adjoint)
		alm2map_raw_cyl(alm[I], tmap, ainfo=ainfo, spin=spin, deriv=deriv, adjoint=adjoint, nthread=nthread, verbose=verbose)
		# Copy out if necessary
		if not adjoint: # and not utils.same_array(tmap, map[I]):
			map[I] = buffer2map(tmap, minfo.flip, pad)
	if adjoint: return alm
	else:       return map

def alm2map_general(alm, map, ainfo=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, locinfo=None, epsilon=None):
	"""Helper function for alm2map. See its docstring for details"""
	if copy:
		if adjoint and alm is not None: alm = alm.copy()
		else:       map = map.copy()
	if adjoint: alm, ainfo = prepare_alm(alm=alm, ainfo=ainfo, pre=map.shape[:-2], dtype=utils.native_dtype(map.dtype))
	if locinfo is None: locinfo = calc_locinfo(map.shape, map.wcs)
	# Loop over pre-pre-dimensions. ducc usually doesn't do anything clever with
	# these, so looping in python is cheap
	for I in utils.nditer(map.shape[:-3]):
		if locinfo.masked:
			mslice = (mask,) if map.ndim == 2 else (slice(None),locinfo.mask)
			tmap = np.ascontiguousarray(map[I][mslice])
		else:
			tmap = utils.postflat(map[I],2)
		alm2map_raw_general(alm[I], tmap, locinfo.loc, ainfo=ainfo, spin=spin, deriv=deriv,
				verbose=verbose, epsilon=epsilon, adjoint=adjoint, nthread=nthread)
		# Copy out map if necessary
		if not adjoint:
			if locinfo.masked:
				map[I][mslice] = tmap
			else:
				map[I] = tmap.reshape(map[I].shape)
	if adjoint: return alm
	else:       return map

def map2alm_2d(map, alm=None, ainfo=None, minfo=None, lmax=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, pix_tol=1e-6):
	"""Helper function for map2alm. See its docsctring for details."""
	if adjoint:
		if copy and map is not None: map = map.copy()
	else:
		if copy and alm is not None: alm = alm.copy()
	alm, ainfo = prepare_alm(alm=alm, ainfo=ainfo, lmax=lmax, pre=map.shape[:-2], dtype=utils.native_dtype(map.dtype))
	minfo = analyse_geometry(map.shape, map.wcs, tol=pix_tol)
	# Loop over pre-pre-dimensions. ducc usually doesn't do anything clever with
	# these, so looping in python is cheap
	for I in utils.nditer(map.shape[:-3]):
		# Pad as necessary
		pad  = ((minfo.ypad[0],minfo.xpad[0]),(minfo.ypad[1],minfo.xpad[1]))
		tmap = map2buffer(map[I], minfo.flip, pad)
		map2alm_raw_2d(tmap, alm[I], ainfo=ainfo, lmax=lmax, spin=spin, deriv=deriv, verbose=verbose, adjoint=adjoint, nthread=nthread)
		# Copy out if necessary
		if adjoint: #and not utils.same_array(tmap, map[I]):
			map[I] = buffer2map(tmap, minfo.flip, pad)
	if adjoint: return map
	else:       return alm

def map2alm_cyl(map, alm=None, ainfo=None, minfo=None, lmax=None, spin=[0,2], weights=None, deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, pix_tol=1e-6, niter=0):
	"""Helper function for map2alm. See its docsctring for details."""
	if adjoint:
		if copy and map is not None: map = map.copy()
	else:
		if copy and alm is not None: alm = alm.copy()
	alm, ainfo = prepare_alm(alm=alm, ainfo=ainfo, lmax=lmax, pre=map.shape[:-2], dtype=utils.native_dtype(map.dtype))
	minfo = analyse_geometry(map.shape, map.wcs, tol=pix_tol)
	# Get our weights, approximate or not
	if weights is None:
		if minfo.ducc_geo is not None and minfo.ducc_geo.name is not None:
			ny      = map.shape[-2]+np.sum(minfo.ypad)
			weights = ducc0.sht.experimental.get_gridweights(minfo.ducc_geo.name, ny)
			weights = weights[minfo.ypad[0]:len(weights)-minfo.ypad[1]]
			weights/= minfo.ducc_geo.nx
		else:
			weights = map.pixsizemap(separable=True, broadcastable=True)[:,0]
			if minfo.flip: weights = weights[::-1]
		weights = weights.astype(map.dtype, copy=False)
	# Loop over pre-pre-dimensions. ducc usually doesn't do anything clever with
	# these, so looping in python is cheap
	for I in utils.nditer(map.shape[:-3]):
		# Pad as necessary
		pad  = ((0,minfo.xpad[0]),(0,minfo.xpad[1]))
		tmap = map2buffer(map[I], minfo.flip, pad)
		map2alm_raw_cyl(tmap, alm[I], ainfo=ainfo, lmax=lmax, spin=spin, weights=weights, deriv=deriv, niter=niter, verbose=verbose, adjoint=adjoint, nthread=nthread)
		# Copy out if necessary
		if adjoint: # and not utils.same_array(tmap, map[I]):
			map[I] = buffer2map(tmap, minfo.flip, pad)
	if adjoint: return map
	else:       return alm

def map2alm_general(map, alm=None, ainfo=None, minfo=None, lmax=None, spin=[0,2], weights=None, deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, locinfo=None, epsilon=None, niter=0):
	"""Helper function for map2alm. See its docsctring for details."""
	if adjoint:
		if copy and map is not None: map = map.copy()
	else:
		if copy and alm is not None: alm = alm.copy()
	alm, ainfo = prepare_alm(alm=alm, ainfo=ainfo, lmax=lmax, pre=map.shape[:-2], dtype=utils.native_dtype(map.dtype))
	if locinfo is None: locinfo = calc_locinfo(map.shape, map.wcs)
	if weights is None: weights = map.pixsizemap()[locinfo.mask].astype(map.dtype, copy=False)
	for I in utils.nditer(map.shape[:-3]):
		if locinfo.masked:
			mslice = (mask,) if map.ndim == 2 else (slice(None),locinfo.mask)
			tmap = np.ascontiguousarray(map[I][mslice])
		else:
			tmap = utils.postflat(map[I],2)
		map2alm_raw_general(tmap, locinfo.loc, alm[I], ainfo=ainfo, lmax=lmax, spin=spin, deriv=deriv,
				weights=weights, adjoint=adjoint, nthread=nthread, verbose=verbose,
				niter=niter, epsilon=epsilon)
		# Copy out if necessary
		if adjoint:
			if locinfo.masked: map[I][mslice] = tmap
			else: map[I] = tmap.reshape(map[I].shape)
	if adjoint: return map
	else:       return alm

def alm2map_raw_2d(alm, map, ainfo=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None):
	"""Helper function for alm2map_2d. Usually not called directly. See the alm2map docstring for details."""
	if copy:
		if adjoint: alm = alm.copy()
		else:       map = map.copy()
	alm_full, map_full, ainfo, nthread = prepare_raw(alm, map, ainfo=ainfo, deriv=deriv, nthread=nthread)
	minfo = analyse_geometry(map.shape, map.wcs)
	if adjoint: func = ducc0.sht.experimental.adjoint_synthesis_2d
	else:       func = ducc0.sht.experimental.synthesis_2d
	# mstart is needed to support a lower lmax than the one actually used in the alm
	kwargs = {"phi0": minfo.phi0, "lmax":ainfo.lmax, "mmax":ainfo.mmax,
		"geometry": minfo.ducc_geo.name, "nthreads": nthread, "mstart": ainfo.mstart}
	# Iterate over all the predimensions. If deriv is true we have
	# alm[{pre},nalm], map[{pre},2,ny,nx]. Otherwise we have
	# alm[{pre},ncomp,nalm], map[{pre},ncomp,ny,nx]. In either case, the
	# pre-dimentions are map.shape[:-3]
	for I in utils.nditer(map_full.shape[:-3]):
		if deriv:
			func(alm=utils.fix_zero_strides(alm_full[I][None]), map=map_full[I], mode="DERIV1", spin=1, **kwargs)
			# Flip sign of theta derivative to get dec derivative
			map_full[I+(0,)] *= -1
		else:
			for s, j1, j2 in enmap.spin_helper(spin, alm_full.shape[-2]):
				Ij = I+(slice(j1,j2),)
				func(alm=alm_full[Ij], map=map_full[Ij], spin=s, **kwargs)
	if adjoint: return alm
	else:       return map

def alm2map_raw_cyl(alm, map, ainfo=None, minfo=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None):
	"""Helper function for alm2map_cyl. Usually not called directly. See the alm2map docstring for details."""
	if copy:
		if adjoint: alm = alm.copy()
		else:       map = map.copy()
	alm_full, map_full, ainfo, nthread = prepare_raw(alm, map, ainfo=ainfo, deriv=deriv, nthread=nthread)
	map_full = utils.postflat(map_full, 2) # ducc wants just 1 pixel axis
	rinfo    = get_ring_info(map.shape, map.wcs)
	if adjoint: func = ducc0.sht.experimental.adjoint_synthesis
	else:       func = ducc0.sht.experimental.synthesis
	kwargs   = {"theta":rinfo.theta, "nphi":rinfo.nphi, "phi0":rinfo.phi0,
		"ringstart":rinfo.offsets, "lmax":ainfo.lmax, "mmax":ainfo.mmax,
		"mstart": ainfo.mstart, "nthreads":nthread}
	# Iterate over all the predimensions. Why do I do this instead of just
	# passing them on to ducc all at once?
	# 1. numpy.reshape can end up silently making a copy in some cases when slicing
	#    has been done to an array. If a copy is procued, then we will end up
	#    discarding all our work for nothing
	# 2. By passing in smaller arrays to ducc, I reduce the chance that ducc will
	#    make big internal work arrays.
	# 3. According to Reinecke, ducc almost always just loop internally over pre-
	#    dimensions anyway.
	# Normally this will be called from alm2map_cyl, which already loops over pre-
	# dimensions, so this outermost loop usually does nothing
	for I in utils.nditer(map_full.shape[:-2]):
		if deriv:
			func(alm=utils.fix_zero_strides(alm_full[I][None]), map=map_full[I], mode="DERIV1", spin=1, **kwargs)
			# Flip sign of theta derivative to get dec derivative
			map_full[I+(0,)] *= -1
		else:
			for s, j1, j2 in enmap.spin_helper(spin, alm_full.shape[-2]):
				Ij = I+(slice(j1,j2),)
				func(alm=alm_full[Ij], map=map_full[Ij], spin=s, **kwargs)
	if adjoint: return alm
	else:       return map

# What about the adjoint?
# map2alm_adjoint = (Y'W)' = W'Y. So simple provided we have the right quad weights W.
# What if we only have approximate quad weights? In this case we have
#  alm0 = Y'W map
#  alm(n+1) = alm(n) + Y'W (map - Y alm(n)) = (1-Y'WY)alm(n) + Y'W map = (1+(1-Y'WY)^n) Y'W map = K(n) map
# Hence, the adjoint of K(n) is
#  K(n)' = W'Y (1+(1-Y'W'Y)^n)
# So all in all, we have
#
#  map2alm
#  0it  alm0 = asyn(w*map)
#  1it  alm1 = asyn(w*map) + alm0 - asyn(w*syn(alm0))
#  2it  alm2 = asyn(w*map) + alm1 - asyn(w*syn(alm1))
#
#  map2alm'
#  0it  map0 = w*syn(alm)
#  1it  map1 = w*syn(alm) + w*syn(alm) - w*syn(asyn(w*(syn(alm))))
#            = w*syn(alm) + map0 - w*syn(asyn(map0))
#  2it  map2 = w*syn(alm) + map1 - w*syn(asyn(map1))
# etc.
#
# So where map2alm uses jacobi with forward = syn and backward = asyn(w()),
# map2alm' uses jacobi with forward = asyn and backward = w*syn
#
# Given this, it might be best to implement map2alm_adjoint for cyl via map2alm instead
# of via alm2map, since that's where the jacobi and weighting stuff is defined.
# I didn't do that for the other ones since an adjoint was already avaliable from ducc,
# and it was more convenient to keep the read/write direction consistent.

def alm2map_raw_general(alm, map, loc, ainfo=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, epsilon=None):
	"""Helper function for alm2map_general. Usually not called directly. See the alm2map docstring for details."""
	if copy:
		if adjoint: alm = alm.copy()
		else:       map = map.copy()
	alm_full, map_full, ainfo, nthread = prepare_raw(alm, map, ainfo=ainfo, deriv=deriv, nthread=nthread, pixdims=1)
	if adjoint: func = ducc0.sht.experimental.adjoint_synthesis_general
	else:       func = ducc0.sht.experimental.synthesis_general
	if epsilon is None:
		if map.dtype == np.float64: epsilon = 1e-10
		else:                       epsilon = 1e-6
	kwargs = {"loc":loc, "lmax":ainfo.lmax, "mmax":ainfo.mmax, "nthreads":nthread, "epsilon":epsilon}
	# Iterate over all the predimensions.
	for I in utils.nditer(map_full.shape[:-2]):
		if deriv:
			func(alm=utils.fix_zero_strides(alm_full[I][None]), map=map_full[I], mode="DERIV1", spin=1, **kwargs)
			# Flip sign of theta derivative to get dec derivative
			map_full[I+(0,)] *= -1
		else:
			for s, j1, j2 in enmap.spin_helper(spin, alm_full.shape[-2]):
				Ij = I+(slice(j1,j2),)
				func(alm=alm_full[Ij], map=map_full[Ij], spin=s, **kwargs)
	if adjoint: return alm
	else:       return map

def map2alm_raw_2d(map, alm=None, ainfo=None, lmax=None, spin=[0,2], deriv=False, copy=False, verbose=False, adjoint=False, nthread=None):
	"""Helper function for map2alm_2d. Usually not called directly. See the map2alm docstring for details."""
	if adjoint:
		if copy and map is not None: map = map.copy()
	else:
		if copy and alm is not None: alm = alm.copy()
	alm_full, map_full, ainfo, nthread = prepare_raw(alm, map, ainfo=ainfo, lmax=lmax, deriv=deriv, nthread=nthread)
	minfo = analyse_geometry(map.shape, map.wcs)
	# Restrict to lmax and mmax that ducc_2d allows. Higher ones will be ignored.
	lmax  = min(ainfo.lmax, minfo.ducc_geo.lmax)
	mmax  = min(ainfo.mmax, lmax)
	if deriv:
		# Could fix this by calling adjoint_synthesis_2d with weights myself
		raise NotImplementedError("ducc does not support derivatives for map2alm operations. Can be worked around if necessary.")
	if adjoint: func = ducc0.sht.experimental.adjoint_analysis_2d
	else:       func = ducc0.sht.experimental.analysis_2d
	# mstart is needed to support a lower lmax than the one actually used in the alm
	kwargs = {"phi0": minfo.phi0, "lmax":lmax, "mmax":mmax,
		"geometry": minfo.ducc_geo.name, "nthreads": nthread, "mstart": ainfo.mstart[:mmax+1]}
	# Iterate over all the predimensions.
	for I in utils.nditer(map_full.shape[:-3]):
		if deriv:
			# Flip sign of theta derivative to get dec derivative
			decflip = np.array([-1,1])[:,None,None]
			func(alm=utils.fix_zero_strides(alm_full[I][None]), map=map_full[I]*decflip, mode="DERIV1", spin=1, **kwargs)
		else:
			for s, j1, j2 in enmap.spin_helper(spin, alm_full.shape[-2]):
				Ij = I+(slice(j1,j2),)
				func(alm=alm_full[Ij], map=map_full[Ij], spin=s, **kwargs)
	if adjoint: return map
	else:       return alm

def map2alm_raw_cyl(map, alm=None, ainfo=None, lmax=None, spin=[0,2], weights=None, deriv=False, copy=False, verbose=False, adjoint=False, niter=0, nthread=None):
	"""Helper function for map2alm_cyl. Usually not called directly. See the map2alm docstring for details."""
	if adjoint:
		if copy and map is not None: map = map.copy()
	else:
		if copy and alm is not None: alm = alm.copy()
	alm_full, map_full, ainfo, nthread = prepare_raw(alm, map, ainfo=ainfo, lmax=lmax, deriv=deriv, nthread=nthread)
	map_full = utils.postflat(map_full, 2) # ducc wants just 1 pixel axis
	rinfo    = get_ring_info   (map.shape, map.wcs)
	kwargs   = {"theta":rinfo.theta, "nphi":rinfo.nphi, "phi0":rinfo.phi0,
		"ringstart":rinfo.offsets, "lmax":ainfo.lmax, "mmax":ainfo.mmax,
		"mstart": ainfo.mstart, "nthreads":nthread}
	# Helper for weights multiplication
	def wmul(map_flat, weights):
		return (map_flat.reshape(map_flat.shape[:-1]+map.shape[-2:])*weights[:,None]).reshape(map_flat.shape)
	# Iterate over all the predimensions.
	for I in utils.nditer(map_full.shape[:-2]):
		if deriv:
			def Y(alm):   return ducc0.sht.experimental.synthesis(alm=alm, mode="DERIV1", spin=1, **kwargs)
			def YT(map):  return ducc0.sht.experimental.adjoint_synthesis(map=map, mode="DERIV1", spin=1, **kwargs)
			def YTW(map): return YT(wmul(map,weights))
			def WY(alm):  return wmul(Y(alm),weights)
			decflip = np.array([-1,1])[:,None,None]
			# The with deriv, alm has a shape of [1,nalm]. The [0] reduces this to [nalm]
			if adjoint: map_full[I] = jacobi_inverse(YT, WY, utils.fix_zero_strides(alm_full[I][None]), niter=niter)*decflip
			else:       alm_full[I] = jacobi_inverse(Y, YTW, map_full[I]*decflip, niter=niter)[0]
		else:
			for s, j1, j2 in enmap.spin_helper(spin, alm_full.shape[-2]):
				Ij = I+(slice(j1,j2),)
				def Y(alm):   return ducc0.sht.experimental.synthesis(alm=alm, spin=s, **kwargs)
				def YT(map):  return ducc0.sht.experimental.adjoint_synthesis(map=map, spin=s, **kwargs)
				def YTW(map): return YT(wmul(map,weights))
				def WY(alm):  return wmul(Y(alm),weights)
				if adjoint: map_full[Ij] = jacobi_inverse(YT, WY, alm_full[Ij], niter=niter)
				else:       alm_full[Ij] = jacobi_inverse(Y, YTW, map_full[Ij], niter=niter)
	if adjoint: return map
	else:       return alm

def map2alm_raw_general(map, loc, alm=None, ainfo=None, lmax=None, spin=[0,2], weights=None, deriv=False, copy=False, verbose=False, adjoint=False, nthread=None, niter=0, epsilon=None):
	"""Helper function for map2alm_general. Usually not called directly. See the map2alm docstring for details."""
	if adjoint:
		if copy and map is not None: map = map.copy()
	else:
		if copy and alm is not None: alm = alm.copy()
	if epsilon is None:
		if map.dtype == np.float64: epsilon = 1e-10
		else:                       epsilon = 1e-6
	alm_full, map_full, ainfo, nthread = prepare_raw(alm, map, ainfo=ainfo, lmax=lmax, deriv=deriv, nthread=nthread, pixdims=1)
	kwargs = {"loc":loc, "lmax":ainfo.lmax, "mmax":ainfo.mmax, "nthreads":nthread, "epsilon":epsilon, "mstart":ainfo.mstart, "epsilon": epsilon}
	if weights is None: weights = np.ones(1)
	def wmul(map, weights): return map*weights
	# Iterate over all the predimensions.
	for I in utils.nditer(map_full.shape[:-2]):
		if deriv:
			def Y(alm):   return ducc0.sht.experimental.synthesis_general(alm=alm, mode="DERIV1", spin=1, **kwargs)
			def YT(map):  return ducc0.sht.experimental.adjoint_synthesis_general(map=map, mode="DERIV1", spin=1, **kwargs)
			def YTW(map): return YT(map*weights)
			def WY(alm):  return wmul(Y(alm),weights)
			decflip = np.array([-1,1])[:,None,None]
			if adjoint: map_full[I] = jacobi_inverse(YT, WY, utils.fix_zero_strides(alm_full[I][None]), niter=niter)*decflip
			else:       alm_full[I] = jacobi_inverse(Y, YTW, map_full[I]*decflip, niter=niter)[0]
		else:
			for s, j1, j2 in enmap.spin_helper(spin, alm_full.shape[-2]):
				Ij = I+(slice(j1,j2),)
				def Y(alm):   return ducc0.sht.experimental.synthesis_general(alm=alm, spin=s, **kwargs)
				def YT(map):  return ducc0.sht.experimental.adjoint_synthesis_general(map=map, spin=s, **kwargs)
				def YTW(map): return YT(map*weights)
				def WY(alm):  return wmul(Y(alm),weights)
				if adjoint: map_full[Ij] = jacobi_inverse(YT, WY, alm_full[Ij], niter=niter)
				else:       alm_full[Ij] = jacobi_inverse(Y, YTW, map_full[Ij], niter=niter)
	return alm

def jacobi_inverse(forward, approx_backward, y, niter=0):
	"""Given y = forward(x), attempt to recover x using jacobi iteration
	with forward and it's approximate inverse approx_backward. niter
	controls the number of iterations. The number of calls to forward is
	niter. The number of calls to approx_backward is 1+niter.

	See minres_inverse for a function with faster convergence and better
	stopping criterion. But Jacobi's quick startup time often means it's
	finished by the time minres has gotten started, so unless high accuracy
	is needed, Jacobi might be the best choice.
	"""
	x = approx_backward(y)
	for i in range(niter):
		x -= approx_backward(forward(x)-y)
	return x

def minres_inverse(forward, approx_backward, y, epsilon=1e-6, maxiter=100, zip=None, unzip=None, verbose=False):
	"""Given y = forward(x), attempt to recover the maximum-likelihood
	solution of x = (P'N"P)"P'N"P using Minres iteration. Here forward = P
	and approx_backward = P'N". Both of these should be functions that takes a single
	argument and returns the result. Iterates until the desired accuracy given by
	epsilon is reached, or the maximum number of iterations given by maxiter is
	reached. If verbose is True, prints information about each step in the
	iteration.

	This function converges more quickly than jacobi, and has a better
	defined stopping criterion, but uses more memory and has a higher
	startup cost. Effectively this function starts two iteration steps
	behind jacobi, and takes several more steps to catch up. It is therefore
	not the fastest choice when only moderate accuracy is needed.
	"""
	rhs   = approx_backward(y)
	rtype = utils.real_dtype(rhs.dtype)
	if zip is None:
		def zip(a): return a.view(rtype).reshape(-1)
	if unzip is None:
		def unzip(x): return x.view(rhs.dtype).reshape(rhs.shape)
	def A(x):
		return zip(approx_backward(forward(unzip(x))))
	solver = utils.Minres(A, zip(rhs))
	while solver.abserr**0.5 > epsilon and solver.i < maxiter:
		solver.step()
		if verbose: print("Minres %4d %15.7e" % (solver.i, solver.abserr**0.5))
	return unzip(solver.x)

def nalm2lmax(nalm):
	return int((-1+(1+8*nalm)**0.5)/2)-1

def get_ring_info(shape, wcs, dtype=np.float64):
	"""Return information about the horizontal rings of pixels in a cylindrical pixelization.
	Used in map2alm and alm2map with the "cyl" method."""
	y = np.arange(shape[-2])
	x = y*0
	dec, ra = enmap.pix2sky(shape, wcs, [y,x])
	theta   = np.asarray(np.pi/2-dec, dtype=dtype)
	assert theta.ndim == 1, "theta must be one-dimensional!"
	ntheta = len(theta)
	nphi   = np.asarray(shape[-1], dtype=np.uint64)
	assert nphi.ndim < 2, "nphi must be 0 or 1-dimensional"
	if nphi.ndim == 0:
		nphi = np.zeros(ntheta,dtype=np.uint64)+(nphi or 2*ntheta)
	assert len(nphi) == ntheta, "theta and nphi arrays do not agree on number of rings"
	phi0 = np.asarray(ra, dtype=dtype)
	assert phi0.ndim < 2, "phi0 must be 0 or 1-dimensional"
	if phi0.ndim == 0:
		phi0 = np.zeros(ntheta,dtype=dtype)+phi0
	offsets = utils.cumsum(nphi).astype(np.uint64, copy=False)
	stride  = np.zeros(ntheta,dtype=np.int32)+1
	return bunch.Bunch(theta=theta, nphi=nphi, phi0=phi0, offsets=offsets, stride=stride, npix=np.sum(nphi), nrow=len(nphi))

def get_ring_info_healpix(nside, rings=None):
	# Which rings to work with.
	nside = int(nside)
	if rings is None: rings = np.arange(4*nside-1)
	else:             rings = np.asarray(rings)
	nring  = len(rings)
	npix   = 12*nside**2
	# Allocate output arrays
	theta  = np.zeros(nring, np.float64)
	phi0   = np.zeros(nring, np.float64)
	nphi   = np.zeros(nring, np.uint64)
	# One-based to make comparison with sharp implementation easier
	rings      = rings+1
	northrings = np.where(rings > 2*nside, 4*nside-rings, rings)
	# Handle polar cap
	cap         = np.where(northrings < nside)[0]
	theta[ cap] = 2*np.arcsin(northrings[cap]/(6**0.5*nside))
	nphi [ cap] = 4*northrings[cap]
	phi0 [ cap] = np.pi/(4*northrings[cap])
	# Handle rest
	rest        = np.where(northrings >= nside)[0]
	theta[rest] = np.arccos((2*nside-northrings[rest])*(8*nside/npix))
	nphi [rest] = 4*nside
	phi0 [rest] = np.pi/(4*nside) * (((northrings[rest]-nside)&1)==0)
	# Above assumed northern hemisphere. Fix southern
	south       = np.where(northrings != rings)[0]
	theta[south]= np.pi-theta[south]
	# Compute the starting point of each ring
	offsets     = utils.cumsum(nphi).astype(np.uint64, copy=False)
	stride      = np.ones(nring, np.int32)
	return bunch.Bunch(theta=theta, nphi=nphi, phi0=phi0, offsets=offsets, stride=stride, npix=npix, nrow=nring)

def get_ring_info_radial(r):
	"""Construct a ring info for a case where there's just one pixel in each ring.
	This is useful for radially symmetric (mmax=0) transforms."""
	theta = np.asarray(r, dtype=np.float64)
	assert theta.ndim == 1, "r must be one-dimensional!"
	n       = len(theta)
	nphi    = np.ones  (n, dtype=np.uint64)
	phi0    = np.zeros (n, dtype=np.float64)
	offsets = np.arange(n, dtype=np.uint64)
	stride  = np.ones  (n, dtype=np.int32)
	return bunch.Bunch(theta=theta, nphi=nphi, phi0=phi0, offsets=offsets, stride=stride, npix=n, nrow=n)

def flip2slice(flips):
	res = (Ellipsis,)
	for flip in flips: res = res + (slice(None,None,1-2*flip),)
	return res
def flip_geometry(shape, wcs, flips):
	return enmap.slice_geometry(shape, wcs, flip2slice(flips))
def flip_array(arr, flips):
	return arr[flip2slice(flips)]
def pad_geometry(shape, wcs, pad):
	w = int(pad[0,0] + shape[-2] + pad[1,0])
	h = int(pad[0,1] + shape[-1] + pad[1,1])
	wcs = wcs.deepcopy()
	wcs.wcs.crpix += pad[0,::-1]
	shape = shape[:-2] + (w,h)
	return shape, wcs

def analyse_geometry(shape, wcs, tol=1e-6):
	"""
	Pass in shape, wcs, and get out an info object that contains
	 case:
	   2d:      can be passed directly to synthesis_2d
	   cyl:     can be passed directly to synthesis
	   partial: can be passed to synthesis after ring-extension,
	     or synthesis_2d after full extension
	   general: only synthesis_general can be used
	 flip: [flipy,flipx] bools. Only relevant for 2d and cyl.
	   partial always needs slices, general never needs them.
	 ducc_geo: Matching ducc geometry. Most useful member is .name, which
	   can be "CC", "F1", "MW", "MWflip" "DH", "F2".
	   ducc_geo is None if this doesn't correspond to a ducc geometry.
	 ypad: [npre,npost]. Only used when padding to 2d
	 xpad: [npre,npost]. Used when case=="partial"
	"""
	# First check if we're a cylindrical geometry. If we're not, we have
	# use the general interface, and issues like flipping and extending are moot.
	# TODO: Pseudo-cylindrical projections can be handled with standard ducc synthesis,
	# so ideally our check would be less stringent than this. Supporinting them requires
	# more work, so will just do it with the general interface for now.
	separable = wcsutils.is_separable(wcs)
	divides   = utils.hasoff(360/np.abs(wcs.wcs.cdelt[0]), 0, tol=tol)
	if not separable or not divides:
		# Not cylindrical or ra does not evenly divide the sky
		return bunch.Bunch(case="general", flip=[False,False], ducc_geo=None, ypad=(0,0), xpad=(0,0), phi0=0)
	# Ok, so we're a cylindrical projection. Check if we need flipping
	flip = [wcs.wcs.cdelt[1] > 0, wcs.wcs.cdelt[0] < 0]
	# Flipped geometry
	wshape, wwcs = flip_geometry(shape, wcs, flip)
	# Get phi0 for the flipped geo
	phi0 = wcsutils.nobcheck(wwcs).wcs_pix2world(0, wshape[-2]//2, 0)[0]*utils.degree
	# Check how we fit with a predefined ducc geometry
	ducc_geo  = get_ducc_geo(wwcs, shape=wshape, tol=tol)
	# If ducc_geo exists, then this map can either be used directly in
	# analysis_2d, or it could be extended to be used in it
	if ducc_geo is not None and shape[-2] == ducc_geo.ny and shape[-1] == ducc_geo.nx and np.abs(ducc_geo.yoff) < tol:
		# We can use 2d directly, though maybe with some flipping
		return bunch.Bunch(case="2d", flip=flip, ducc_geo=ducc_geo, ypad=(0,0), xpad=(0,0), phi0=phi0)
				
	else:
		# We can't call 2d directly. But we may want to pad and then call it.
		if ducc_geo is not None: ypad = (ducc_geo.yoff, ducc_geo.ny-ducc_geo.yoff-shape[-2])
		else: ypad = (0,0)
		# Check if we have full rows, so we can call standard analysis directly
		nx = utils.nint(360/wwcs.wcs.cdelt[0])
		if shape[-1] == nx:
			# Yes, we have full rows, so can call cyl directly. But define a y slice for 2d
			# compatibility if we can, so the user can choose
			return bunch.Bunch(case="cyl", flip=flip, ducc_geo=ducc_geo, ypad=ypad, xpad=(0,0), phi0=phi0)
		else:
			# No, we don't have full rows. Define an x padding that takes us there
			xpad = (0, nx-shape[-1])
			return bunch.Bunch(case="partial", flip=flip, ducc_geo=ducc_geo, ypad=ypad, xpad=xpad, phi0=phi0)

def get_ducc_geo(wcs, shape=None, tol=1e-6):
	"""Return the ducc geometry type for the given world coordinate system
	object. Returns a bunch(name, phi0) where name is one of "CC", "F1", "MW",
	"MWflip", "DH" and "F2". "GL": gauss-legendre is not supported by wcs.
	Returns None if the wcs doesn't correspond to a ducc geometry."""
	def near(a, b): return np.abs(a-b)<tol
	def hasoff(val, off): return utils.hasoff(val, off, tol=tol)
	# Check if we need flipping. Ducc assumes increasing phi and theta,
	# which means increasing ra and decreasing dec. The rest of this
	# function assumes that things are in ducc order, such that the north
	# pole is near pix 0 and the south pole near pix N.
	flip = [wcs.wcs.cdelt[1] > 0, wcs.wcs.cdelt[0] < 0]
	_, wcs = enmap.slice_geometry(shape or (1,1), wcs,
			(slice(None,None,1-2*flip[0]),slice(None,None,1-2*flip[1])))
	# Number of x intervals in whole sky
	nx  = 360/wcs.wcs.cdelt[0]
	# Do we have a whole number of intervals if not, it's not a valid geometry
	if not hasoff(nx, 0): return None
	# Row start offset
	phi0 = wcs.wcs_pix2world(0,0,0)[0]*utils.degree
	# Pixel coordinates of north and south pole
	y1 = wcs.wcs_world2pix(0, 90,0)[1]
	y2 = wcs.wcs_world2pix(0,-90,0)[1]
	Ny = shape[-2] if shape is not None else utils.nint(y2)+1
	# This is a bit inefficient, but it doesn't matter and it
	# makes it easier to read
	if   hasoff(y1,0.0) and hasoff(y2,0.0):
		if   near(y1,-1) and near(y2,Ny): name, o1, o2 = "F2", 1, 1
		elif near(y1, 0) and near(y2,Ny): name, o1, o2 = "DH", 1, 0
		else: name, o1, o2 = "CC", 0, 0
	elif hasoff(y1,0.5) and hasoff(y2,0.5): name, o1, o2 = "F1", 0.5, 0.5
	elif hasoff(y1,0.5) and hasoff(y2,0.0): name, o1, o2 = "MW", 0.5, 0.0
	elif hasoff(y1,0.0) and hasoff(y2,0.5): name, o1, o2 = "MWflip", 0.0, 0.5
	else: return None
	ny   = utils.nint(y2-y1+1-o1-o2)
	# yoff is the y pixel offset of our first pixel from where the first pixel
	yoff = utils.nint(-y1-o1)
	# maximum lmax supported for the geometry
	lmax = get_ducc_maxlmax(name, ny)
	return bunch.Bunch(name=name, nx=utils.nint(nx), ny=ny, pole_offs=[o1,o2], phi0=phi0, yoff=yoff, lmax=lmax)

def get_ducc_maxlmax(name, ny):
	if   name == "CC": return ny-2
	elif name == "DH": return (ny-2)//2
	elif name == "F2": return (ny-1)//2
	else:              return ny-1

def calc_locinfo(shape, wcs, bsize=1000):
	"""Calculate pixel position info in the format ducc needs"""
	# posmaps can be big, bigger than the normal map itself due to being
	# double precision. So let's try to save memory by using blocking.
	# Allocate a loc array that's (nmax,2). We will truncate it to the
	# masked length in the end
	loc  = np.zeros((shape[-2]*shape[-1],2))
	mask = np.zeros(shape[-2:],bool)
	off  = 0
	for b1 in range(0, shape[-2]-1, bsize):
		b2      = min(b1+bsize, shape[-2])
		subgeo  = enmap.Geometry(shape, wcs)[b1:b2]
		subpos  = enmap.posmap(*subgeo, safe=False)
		subpos[0]  = np.pi/2 - subpos[0]
		subpos[1] += 2*np.pi*(subpos[1]<0)
		submask = np.all(np.isfinite(subpos),0)
		nok     = np.sum(submask)
		if nok < (b2-b1)*shape[-1]:
			# This is expensive for some reason, so skip it if possible
			loc[off:off+nok,:] = subpos[:,submask].T
		else:
			loc[off:off+nok,:] = subpos.reshape(2,-1).T
		mask[b1:b2] = submask
		off += nok
	# Truncate to masked length
	loc    = loc[:off]
	masked = off < shape[-2]*shape[-1]
	return bunch.Bunch(loc=loc, mask=mask, masked=masked)

def map2buffer(map, flip, pad, obuf=False):
	"""Prepare a map for ducc operations by flipping and/or padding it, returning
	the resulting map."""
	# First allocate the output buffer
	pad = np.asarray(pad)
	geo = enmap.Geometry(*map.geometry)
	geo = flip_geometry(*geo, flip)
	geo = pad_geometry(*geo, pad)
	# Use the same dtype, except force a native dtype since ducc doesn't like
	# non-native dtypes
	buf = enmap.zeros(*geo, utils.fix_dtype_mpi4py(map.dtype))
	# Then copy the input map over, unless we're a pure
	# output buffer
	if not obuf:
		buf[...,pad[0,0]:buf.shape[-2]-pad[1,0],pad[0,1]:buf.shape[-1]-pad[1,1]] = flip_array(map, flip)
	#map = flip_array(map, flip)
	#pad = np.array(pad)
	#if np.any(pad!=0):
	#	map = enmap.pad(map, pad)
	#map = enmap.samewcs(np.ascontiguousarray(map),map)
	return buf

def buffer2map(map, flip, pad):
	"""The inverse of map2buffer. Undoes flipping and padding"""
	pad = np.array(pad)
	map = map[...,pad[0,0]:map.shape[-2]-pad[1,0],pad[0,1]:map.shape[-1]-pad[1,1]]
	map = flip_array(map, flip)
	return map

def prepare_alm(alm=None, ainfo=None, lmax=None, pre=(), dtype=np.float64):
	"""Set up alm and ainfo based on which ones of them are available."""
	ctype = utils.complex_dtype(dtype)
	if alm is None:
		if ainfo is None:
			if lmax is None:
				raise ValueError("prepare_alm needs either alm, ainfo or lmax to be specified")
			ainfo = alm_info(lmax)
		alm = np.zeros(pre+(ainfo.nelem,), dtype=ctype)
	if ainfo is None:
		ainfo = alm_info(nalm=alm.shape[-1])
	alm = alm.astype(ctype, copy=False)
	return alm, ainfo

def prepare_raw(alm, map, ainfo=None, lmax=None, deriv=False, verbose=False, nthread=None, pixdims=2):
	alm, ainfo = prepare_alm(alm, ainfo, lmax=lmax, pre=map.shape[:-pixdims], dtype=utils.native_dtype(map.dtype))
	# Maybe this should be a part of map_info too
	nthread  = int(utils.fallback(utils.getenv("OMP_NUM_THREADS", nthread),0))
	# Massage to the shape the general ducc interface wants.
	alm_full = utils.atleast_Nd(alm, 2 if deriv else 3)
	map_full = utils.atleast_Nd(map, pixdims+2)
	# Wait until here to do this test, to allow some minor broadcasting support
	if deriv:
		assert map_full.ndim >= pixdims+1 and map_full.shape[-pixdims-1] == 2, "map must have shape [...,2,%s] when deriv is True" % ("nloc" if pixdims == 1 else "ny,nx")
		assert map_full.shape[:-1-pixdims] == alm_full.shape[:-1], "map and alm must agree on pre-dimensions"
	else:
		assert map_full.shape[:-pixdims] == alm_full.shape[:-1], "map and alm must agree on pre-dimensions"
	map_full = np.asarray(map_full)              # ducc doesn't accept subclasses
	# Work around ducc bug
	alm_full = utils.fix_zero_strides(alm_full)
	map_full = utils.fix_zero_strides(map_full)
	return alm_full, map_full, ainfo, nthread

def dangerous_dtype(dtype):
	return dtype.byteorder != "="

def alm_complex2real(alm, ainfo=None):
	dtype = utils.real_dtype(alm.dtype)
	if ainfo is None:
		ainfo = alm_info(nalm=alm.shape[-1])
	i = int(ainfo.mstart[1]+1)
	return np.concatenate([alm[...,:i].real,2**0.5*alm[...,i:].view(dtype)],-1)

def alm_real2complex(ralm, ainfo=None):
	ctype = utils.complex_dtype(ralm.dtype)
	if ainfo is None:
		# For complex alms, we have
		# lm: 00 10 20 30 .. L0 11 21 31 ... L1
		# (L+1)+(L+1-1)+...(1) = sum 1..L+1 = (L+1)*(L+2)/2
		# example: L=1: 00 10 11 = 3, vs. 2*3/2 = 3 ok.
		# For ralm, we instead have (L+1)+2*sum_1..L = L+1 + L(L+1) = L²+2L+2 = n
		# => L = sqrt(n-1)-1
		lmax = utils.nint((ralm.shape[-1]-1)**0.5)-1
		ainfo= alm_info(lmax=lmax)
	i = int(ainfo.mstart[1]+1)
	oalm = np.zeros(ralm.shape[:-1]+(ainfo.nelem,), ctype)
	oalm[...,:i] = ralm[...,:i]
	oalm[...,i:] = ralm[...,i:].view(ctype)/2**0.5
	return oalm
