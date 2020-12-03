"""This module provides functions for taking into account the curvature of the
full sky."""
from __future__ import print_function
from . import sharp
import numpy as np
from . import enmap, powspec, wcsutils, utils

class ShapeError(Exception): pass

def rand_map(shape, wcs, ps, lmax=None, dtype=np.float64, seed=None, oversample=2.0, spin=[0,2], method="auto", direct=False, verbose=False):
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
	alm2map(alm, map, spin=spin, oversample=oversample, method=method, direct=direct, verbose=verbose)
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
	ainfo.lmul(alm, (ps12/2**0.5).astype(rtype), alm)
	alm[:,:ainfo.lmax+1].imag  = 0
	alm[:,:ainfo.lmax+1].real *= 2**0.5
	if ps.ndim == 1: alm = alm[0]
	if return_ainfo: return alm, ainfo
	else: return alm

##########################
### Top-level wrappers ###
##########################

def alm2map(alm, map, ainfo=None, spin=[0,2], deriv=False, direct=False, copy=False, oversample=2.0, method="auto", verbose=False):
	"""Project the spherical harmonics coefficients alm[...,nalm] onto the
	enmap map[...,ny,nx].

	The map does not need to be full-sky - an intermediate map will be
	constructed for the SHT itself. If map is in a cylindrical projection, the
	intermediate map will have compatible pixels, and no interpolation is needed.
	Otherwise, the intermediate map will be oversample times higher resolution
	than the output map, and bicubic spline interpolation will be used to tranfer
	its values to the output map. This uses more memory, is slower and less
	accurate than the direct evaluation used for cylindrical projections. If
	method is "cyl" only the cylindrical method will be used, resulting in a
	ShapeError if the pixelization is not actually cylindrical. If method is
	"pos", then the slow, general method will always be used.

	If ainfo is provided, it is an alm_info describing the layout of the input alm.
	Otherwise it will be inferred from the alm itself.

	spin describes the spin of the transformation used for the polarization
	components.

	If deriv is True, then the resulting map will be the gradient of the input alms."""
	if method == "cyl":
		alm2map_cyl(alm, map, ainfo=ainfo, spin=spin, deriv=deriv, direct=direct, copy=copy, verbose=verbose)
	elif method == "pos":
		if verbose: print("Computing pixel positions %s dtype d" % str((2,)+map.shape[-2:]))
		pos = map.posmap()
		res = alm2map_pos(alm, pos, ainfo=ainfo, oversample=oversample, spin=spin, deriv=deriv, verbose=verbose)
		map[:] = res
	elif method == "auto":
		# Cylindrical method if possible, else slow pos-based method
		try:
			alm2map_cyl(alm, map, ainfo=ainfo, spin=spin, deriv=deriv, direct=direct, copy=copy, verbose=verbose)
		except ShapeError as e:
			# Wrong pixelization. Fall back on slow, general method
			if verbose: print("Computing pixel positions %s dtype d" % str((2,)+map.shape[-2:]))
			pos = map.posmap()
			res = alm2map_pos(alm, pos, ainfo=ainfo, oversample=oversample, spin=spin, deriv=deriv, verbose=verbose)
			map[:] = res
	else:
		raise ValueError("Unknown alm2map method %s" % method)
	return map

def map2alm(map, alm=None, ainfo=None, lmax=None, spin=[0,2], direct=False, copy=False,
		oversample=2.0, method="auto", rtol=None, atol=None):
	if method == "cyl":
		alm = map2alm_cyl(map, alm, ainfo=ainfo, lmax=lmax, spin=spin, direct=direct,
				copy=copy, rtol=rtol, atol=atol)
	elif method == "pos":
		raise NotImplementedError("map2alm for noncylindrical layouts not implemented")
	elif method == "auto":
		try:
			alm = map2alm_cyl(map, alm, ainfo=ainfo, lmax=lmax, spin=spin, direct=direct,
					copy=copy, rtol=rtol, atol=atol)
		except ShapeError as e:
			raise NotImplementedError("map2alm for noncylindrical layouts not implemented")
	else:
		raise ValueError("Unknown alm2map method %s" % method)
	return alm

#################################
### Position-based transforms ###
#################################

# These perform SHTs at arbitrary sample positions

def alm2map_pos(alm, pos, ainfo=None, oversample=2.0, spin=[0,2], deriv=False, verbose=False):
	"""Projects the given alms (with layout) on the specified pixel positions.
	alm[ncomp,nelem], pos[2,...] => res[ncomp,...]. It projects on a large
	cylindrical grid and then interpolates to the actual pixels. This is the
	general way of doing things, but not the fastest. Computing pos and
	interpolating takes a significant amount of time."""
	alm_full = np.atleast_2d(alm)
	if ainfo is None: ainfo = sharp.alm_info(nalm=alm_full.shape[-1])
	ashape, ncomp = alm_full.shape[:-2], alm_full.shape[-2]
	if deriv:
		# If we're computing derivatives, spin isn't allowed.
		# alm must be either [ntrans,nelem] or [nelem],
		# and the output will be [ntrans,2,ny,nx] or [2,ny,nx]
		ashape = ashape + (ncomp,)
		ncomp = 2
	tmap   = make_projectable_map_by_pos(pos, ainfo.lmax, ashape+(ncomp,), oversample, alm.real.dtype)
	alm2map_cyl(alm, tmap, ainfo=ainfo, spin=spin, deriv=deriv, direct=True, verbose=verbose)
	# Project down on our final pixels. This will result in a slight smoothing
	res = enmap.samewcs(tmap.at(pos[:2], mode="wrap"), pos)
	# Remove any extra dimensions we added
	if alm.ndim == alm_full.ndim-1: res = res[0]
	return res

##############################
### Cylindrical transforms ###
##############################

# These assume we're using a cylindrical projection, but
# not necessarily that whole rings are covered. The coordinate
# system is extended internally if necessary. minfo is built
# internally automatically.

def alm2map_cyl(alm, map, ainfo=None, spin=[0,2], deriv=False, direct=False, copy=False, verbose=False):
	"""When called as alm2map(alm, map) projects those alms onto that map.
	alms are interpreted according to ainfo if specified.

	Possible shapes:
		alm[nelem] -> map[ny,nx]
		alm[ncomp,nelem] -> map[ncomp,ny,nx]
		alm[ntrans,ncomp,nelem] -> map[ntrans,ncomp,ny,nx]
		alm[nelem] -> map[{dy,dx},ny,nx] (deriv=True)
		alm[ntrans,nelem] -> map[ntrans,{dy,dx},ny,nx] (deriv=True)

	Spin specifies the spin of the transform. Deriv indicates whether
	we will return the derivatives rather than the map itself. If
	direct is true, the input map is assumed to already cover the whole
	sky horizontally, so that no intermediate maps need to be computed.

	If copy=True, the input map is not overwritten.
	"""
	# Work on views of alm and map with shape alm_full[ntrans,ncomp,nalm]
	# and map[ntrans,ncomp/nderiv,ny,nx] to avoid lots of if tests later.
	# We undo the reshape before returning.
	alm, ainfo = prepare_alm(alm, ainfo)
	if copy: map = map.copy()
	if direct: tmap, mslices, tslices = map, [(Ellipsis,)], [(Ellipsis,)]
	else:      tmap, mslices, tslices = make_projectable_map_cyl(map, verbose=verbose)
	if verbose: print("Performing alm2map")
	alm2map_raw(alm, tmap, ainfo, map2minfo(tmap), spin=spin, deriv=deriv)
	for mslice, tslice in zip(mslices, tslices):
		map[mslice] = tmap[tslice]
	return map

def alm2map_healpix(alm, healmap=None, ainfo=None, nside=None, spin=[0,2], deriv=False, copy=False,
		theta_min=None, theta_max=None):
	"""Projects the given alm[...,ncomp,nalm] onto the given healpix map
	healmap[...,ncomp,npix]."""
	alm, ainfo = prepare_alm(alm, ainfo)
	healmap    = prepare_healmap(healmap, nside, alm.shape[:-1], alm.real.dtype)
	nside = npix2nside(healmap.shape[-1])
	minfo = sharp.map_info_healpix(nside)
	minfo = apply_minfo_theta_lim(minfo, theta_min, theta_max)
	return alm2map_raw(alm, healmap[...,None], ainfo=ainfo, minfo=minfo,
			spin=spin, deriv=deriv, copy=copy)[...,0]

def map2alm_cyl(map, alm=None, ainfo=None, lmax=None, spin=[0,2], direct=False,
		copy=False, rtol=None, atol=None):
	"""When called as map2alm_cyl(map, alm) computes the alms corresponding
	to the given map. alms will be ordered according to ainfo if specified.
	The map must be in a cylindrical projection. If no ring weights
	can be determined, it will either use an approximation or raise an
	exception, depending on the value of tolerance, which specifies the
	maximum pixel position error allowed.

	Possible shapes:
		alm[nelem] -> map[ny,nx]
		alm[ncomp,nelem] -> map[ncomp,ny,nx]
		alm[ntrans,ncomp,nelem] -> map[ntrans,ncomp,ny,nx]

	Spin specifies the spin of the transform. If direct is true, the
	input map is assumed to already cover the whole sky horizontally,
	so that no intermediate maps need to be computed.

	If copy=True, the input alm is not overwritten.
	"""
	# Work on views of alm and map with shape alm_full[ntrans,ncomp,nalm]
	# and map[ntrans,ncomp/nderiv,ny,nx] to avoid lots of if tests later.
	# We undo the reshape before returning.
	alm, ainfo = prepare_alm(alm, ainfo, lmax, map.shape[:-2], map.dtype)
	if direct: tmap, mslices, tslices = map, [(Ellipsis,)], [(Ellipsis,)]
	else:      tmap, mslices, tslices = make_projectable_map_cyl(map)
	tmap[:] = 0
	for mslice, tslice in zip(mslices, tslices):
		tmap[tslice] = map[mslice]
	# We don't have ring weights for general cylindrical projections.
	# See if our pixelization matches one with known weights.
	minfo = match_predefined_minfo(tmap, rtol=rtol, atol=atol)
	return map2alm_raw(tmap, alm, minfo, ainfo, spin=spin, copy=copy)

def map2alm_healpix(healmap, alm=None, ainfo=None, lmax=None, spin=[0,2], copy=False,
		theta_min=None, theta_max=None):
	"""Projects the given alm[...,ncomp,nalm] onto the given healpix map
	healmap[...,ncomp,npix]."""
	alm, ainfo = prepare_alm(alm, ainfo, lmax, healmap.shape[:-1], healmap.dtype)
	nside = npix2nside(healmap.shape[-1])
	minfo = sharp.map_info_healpix(nside)
	minfo = apply_minfo_theta_lim(minfo, theta_min, theta_max)
	return map2alm_raw(healmap[...,None], alm, minfo=minfo, ainfo=ainfo,
			spin=spin, copy=copy)

######################
### Raw transforms ###
######################

# These assume the maps are already in the appropriate pixelization,
# E.g. cylindrical projection with complete equi-latitude rings.
# They assume the imap is [...,ny,nx], but these dimensions are
# flattened internally, so a healpix map could be used by adding a fake
# last axis. The map does not need to be an enmap - the world coordinate
# system is ignored as minfo handles all that.

def alm2map_raw(alm, map, ainfo, minfo, spin=[0,2], deriv=False, copy=False):
	"""Direct wrapper of libsharp's alm2map. Requires ainfo and minfo
	to already be set up, and that the map and alm must be fully compatible
	with these."""
	if copy: map = map.copy()
	alm = np.asarray(alm, dtype=np.result_type(map.dtype,1j))
	alm_full = utils.to_Nd(alm, 2 if deriv else 3)
	map_full = utils.to_Nd(map, 4)
	map_flat = map_full.reshape(map_full.shape[:-2]+(-1,))
	sht      = sharp.sht(minfo, ainfo)
	# Perform the SHT
	if deriv:
		# We need alm_full[ntrans,nalm] -> map_flat[ntrans,2,npix]
		# or alm_full[nalm] -> map_flat[2,npix]
		map_flat[:] = sht.alm2map_der1(alm_full, map_flat)
		# sharp's theta is a zenith angle, but we want a declination.
		# Actually, we may need to take into account left-handed
		# coordinates too, though I'm not sure how to detect those in
		# general.
		map_flat[:,0] = -map_flat[:,0]
	else:
		for s, i1, i2 in enmap.spin_helper(spin, map_flat.shape[1]):
			map_flat[:,i1:i2,:] = sht.alm2map(alm_full[:,i1:i2,:], map_flat[:,i1:i2,:], spin=s)
	return map

def map2alm_raw(map, alm, minfo, ainfo, spin=[0,2], copy=False):
	"""Direct wrapper of libsharp's map2alm. Requires ainfo and minfo
	to already be set up, and that the map and alm must be fully compatible
	with these."""
	if not (map.dtype == np.float32 or map.dtype == np.float64): raise TypeError("Only float32 or float64 dtype supported for shts")
	if copy: alm = alm.copy()
	alm_full = utils.to_Nd(alm, 3)
	map_full = utils.to_Nd(map, 4)
	map_flat = map_full.reshape(map_full.shape[:-2]+(-1,))
	sht      = sharp.sht(minfo, ainfo)
	for s, i1, i2 in enmap.spin_helper(spin, map_flat.shape[1]):
		alm_full[:,i1:i2,:] = sht.map2alm(map_flat[:,i1:i2,:], alm_full[:,i1:i2,:], spin=s)
	return alm

### Helper function ###

def make_projectable_map_cyl(map, verbose=False):
	"""Given an enmap in a cylindrical projection, return a map with
	the same pixelization, but extended to cover a whole band in phi
	around the sky. Also returns the slice required to recover the
	input map from the output map."""
	# First check if we need flipping. Sharp wants theta,phi increasing,
	# which means dec decreasing and ra increasing.
	flipx = map.wcs.wcs.cdelt[0] < 0
	flipy = map.wcs.wcs.cdelt[1] > 0
	if flipx: map = map[...,:,::-1]
	if flipy: map = map[...,::-1,:]
	# Then check if the map satisfies the lat-ring requirements
	ny, nx = map.shape[-2:]
	vy,vx = enmap.pix2sky(map.shape, map.wcs, [np.arange(ny),np.zeros(ny)])
	hy,hx = enmap.pix2sky(map.shape, map.wcs, [np.zeros(nx),np.arange(nx)])
	dx    = hx[1:]-hx[:-1]
	dx    = dx[np.isfinite(dx)] # Handle overextended coordinates

	if not np.allclose(dx,dx[0]): raise ShapeError("Map must have constant phi spacing")
	nphi = utils.nint(2*np.pi/dx[0])
	if not np.allclose(2*np.pi/nphi,dx[0]): raise ShapeError("Pixels must evenly circumference")
	if not np.allclose(vx,vx[0]): raise ShapeError("Different phi0 per row indicates non-cylindrical enmap")
	phi0 = vx[0]
	# Make a map with the same geometry covering a whole band around the sky
	# We can do this simply by extending it in the positive pixel dimension.
	oshape = map.shape[:-1]+(nphi,)
	owcs   = map.wcs
	# Our input map could in theory cover multiple copies of the sky, which
	# would require us to copy out multiple slices.
	nslice = (nx+nphi-1)//nphi
	islice, oslice = [], []
	def negnone(x): return x if x >= 0 else None
	for i in range(nslice):
		# i1:i2 is the range of pixels in the original map to use
		i1, i2 = i*nphi, min((i+1)*nphi,nx)
		islice.append((Ellipsis, slice(i1,i2)))
		# yslice and xslice give the range of pixels in our temporary map to use.
		# This is 0:(i2-i1) if we're not flipping, but if we flip we count from
		# the opposite direction: nx-1:nx-1-(i2-i1):-1
		yslice = slice(-1,None,-1)  if flipy else slice(None)
		xslice = slice(nx-1,negnone(nx-1-(i2-i1)),-1) if flipx else slice(0,i2-i1)
		oslice.append((Ellipsis,yslice,xslice))
	if verbose: print("Allocating shape %s dtype %s intermediate map" % (str(oshape),np.dtype(map.dtype).char))
	return enmap.empty(oshape, owcs, dtype=map.dtype), islice, oslice

def make_projectable_map_by_pos(pos, lmax, dims=(), oversample=2.0, dtype=float, verbose=False):
	"""Make a map suitable as an intermediate step in projecting alms up to
	lmax on to the given positions. Helper function for alm2map."""
	# First find the theta range of the pixels, with a 10% margin
	ra_ref   = np.mean(pos[1])/utils.degree
	decrange = np.array([np.min(pos[0]),np.max(pos[0])])
	decrange = (decrange-np.mean(decrange))*1.1+np.mean(decrange)
	decrange = np.array([max(-np.pi/2,decrange[0]),min(np.pi/2,decrange[1])])
	decrange /= utils.degree
	wdec = np.abs(decrange[1]-decrange[0])
	# The shortest wavelength in the alm is about 2pi/lmax. We need at least
	# two samples per mode.
	res = 180./lmax/oversample
	# Set up an intermediate coordinate system for the SHT. We will use
	# CAR coordinates conformal on the equator, with a pixel on each pole.
	# This will give it clenshaw curtis pixelization.
	nx    = utils.nint(360/res)
	nytot = utils.nint(180/res)
	# First set up the pixelization for the whole sky. Negative cdelt to
	# make sharp extra happy. Not really necessary, but makes some things
	# simpler later.
	wcs   = wcsutils.WCS(naxis=2)
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	wcs.wcs.crval = [ra_ref,0]
	wcs.wcs.cdelt = [360./nx,-180./nytot]
	wcs.wcs.crpix = [nx/2.0+1,nytot/2.0+1]
	# Then find the subset that includes the dec range we want
	y1= utils.nint(wcs.wcs_world2pix(0,decrange[0],0)[1])
	y2= utils.nint(wcs.wcs_world2pix(0,decrange[1],0)[1])
	y1, y2 = min(y1,y2), max(y1,y2)
	# Offset wcs to start at our target range
	ny = y2-y1
	wcs.wcs.crpix[1] -= y1
	# Construct the map. +1 to put extra pixel at pole when we are fullsky
	if verbose: print("Allocating shape %s dtype %s intermediate map" % (dims+(ny+1,nx),np.dtype(dtype).char))
	tmap = enmap.zeros(dims+(ny+1,nx),wcs,dtype=dtype)
	return tmap

def map2minfo(m):
	"""Given an enmap with constant-latitude rows and constant longitude
	intervals, return a corresponding sharp map_info."""
	theta  = np.pi/2 - m[...,:,:1].posmap()[0,:,0]
	# Slice to make calculation faster. Could have just queried m.wcs cirectly here
	# instead. Offset by 1 away from bottom to avoid any pole-related problems.
	phi0   = m[...,1:2,0:1].posmap()[1,0,0]
	nphi   = m.shape[-1]
	return sharp.map_info(theta, nphi, phi0)

def match_predefined_minfo(m, rtol=None, atol=None):
	"""Given an enmap with constant-latitude rows and constant longitude
	intervals, return the libsharp predefined minfo with ringweights that's
	the closest match to our pixelization."""
	if rtol is None: rtol = 1e-3*utils.arcmin
	if atol is None: atol = 1.0*utils.arcmin
	# Make sure the colatitude ascends
	flipy  = m.wcs.wcs.cdelt[1] > 0
	if flipy: m = m[...,::-1,:]
	theta  = np.pi/2 - m[...,:,:1].posmap()[0,:,0]
	phi0   = m[...,1:2,0:1].posmap()[1,0,0]
	ntheta, nphi = m.shape[-2:]
	# First find out how many lat rings there are in the whole sky.
	# Find the first and last pixel center inside bounds
	y1   = int(np.round(m.sky2pix([np.pi/2,0])[0]))
	y2   = int(np.round(m.sky2pix([-np.pi/2,0])[0]))
	phi0 = m.pix2sky([0,0])[1]
	ny   = utils.nint(y2-y1+1)
	nx   = utils.nint(np.abs(360./m.wcs.wcs.cdelt[0]))
	# Define our candidate pixelizations
	minfos = []
	for i in range(-1,2):
		#minfos.append(sharp.map_info_gauss_legendre(ny+i, nx, phi0))
		minfos.append(sharp.map_info_clenshaw_curtis(ny+i, nx, phi0))
		minfos.append(sharp.map_info_fejer1(ny+i, nx, phi0))
		minfos.append(sharp.map_info_fejer2(ny+i, nx, phi0))
		minfos.append(sharp.map_info_mw(ny+i, nx, phi0))
	# For each pixelization find the first ring in the map
	aroffs, scores, minfos2 = [], [], []
	for minfo in minfos:
		# Find theta closest to our first theta
		i1 = np.argmin(np.abs(theta[0]-minfo.theta))
		# If we're already on the full sky, the the +1
		# pixel alternative will not work.
		if i1+len(theta) > minfo.theta.size: continue
		# Find the largest theta offset for all y in our input map
		offs = theta-minfo.theta[i1:i1+len(theta)]
		aoff = np.max(np.abs(offs))
		# Find the largest offset after applying a small pointing offset
		roff = np.max(np.abs(offs-np.mean(offs)))
		aroffs.append([aoff,roff,i1])
		scores.append(aoff/atol + roff/rtol)
		minfos2.append(minfo)
	# Choose the one with the lowest score (lowest mismatch)
	best  = np.argmin(scores)
	aoff, roff, i1 = aroffs[best]
	i2 = i1+ntheta
	if not aoff < atol: raise ShapeError("Could not find a map_info with predefined weights matching input map (abs offset %e >= %e)" % (aoff, atol))
	if not roff < rtol: raise ShapeError("Could not find a map_info with predefined weights matching input map (%rel offset e >= %e)" % (aoff, atol))
	minfo = minfos2[best]
	# Modify the minfo to restrict it to only the rows contained in m
	minfo_cut = sharp.map_info(
			minfo.theta[i1:i2],  minfo.nphi[i1:i2], minfo.phi0[i1:i2],
			minfo.offsets[i1:i2]-minfo.offsets[i1], minfo.stride[i1:i2],
			minfo.weight[i1:i2])
	if flipy:
		# Actual map is flipped in y relative to the one we computed the map info
		minfo_cut = sharp.map_info(
				minfo_cut.theta[::-1], minfo_cut.nphi[::-1], minfo_cut.phi0[::-1],
				minfo_cut.offsets[:], minfo_cut.stride[:], minfo_cut.weight[::-1])
	# Phew! Return the result
	return minfo_cut

def npix2nside(npix):
	return utils.nint((npix/12)**0.5)

def prepare_alm(alm=None, ainfo=None, lmax=None, pre=(), dtype=np.float64):
	"""Set up alm and ainfo based on which ones of them are available."""
	if alm is None:
		if ainfo is None:
			if lmax is None:
				raise ValueError("prepare_alm needs either alm, ainfo or lmax to be specified")
			ainfo = sharp.alm_info(lmax)
		alm = np.zeros(pre+(ainfo.nelem,), dtype=np.result_type(dtype,0j))
	if ainfo is None:
		ainfo = sharp.alm_info(nalm=alm.shape[-1])
	return alm, ainfo

def prepare_healmap(healmap, nside=None, pre=(), dtype=np.float64):
	if healmap is not None: return healmap
	return np.zeros(pre + (12*nside**2,), dtype)

def apply_minfo_theta_lim(minfo, theta_min=None, theta_max=None):
	if theta_min is None and theta_max is None: return minfo
	mask = np.full(minfo.nrow, True, bool)
	if theta_min is not None: mask &= minfo.theta >= theta_min
	if theta_max is not None: mask &= minfo.theta <= theta_max
	return minfo.select_rows(mask)

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
		ainfo = sharp.alm_info(lmax)
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

def almxfl(alm,lfunc,ainfo=None):
	"""Filter alms isotropically by a function.
	Returns alm * lfunc(ell)

	Args:
	    alm: (...,N) ndarray of spherical harmonic alms
	    lfunc: a function mapping multipole ell to the filtering expression
	    ainfo: If ainfo is provided, it is an alm_info describing the layout
	of the input alm. Otherwise it will be inferred from the alm itself.

	Returns:
	    falm: The filtered alms alm * lfunc(ell)
	"""
	alm   = np.asarray(alm)
	ainfo = sharp.alm_info(nalm=alm.shape[-1]) if ainfo is None else ainfo
	l = np.arange(ainfo.lmax+1.0)
	return ainfo.lmul(alm, lfunc(l))

def filter(imap,lfunc,ainfo=None,lmax=None):
	"""Filter a map isotropically by a function.
	Returns alm2map(map2alm(alm * lfunc(ell),lmax))

	Args:
	    imap: (...,Ny,Nx) ndmap stack of enmaps.
	    lmax: integer specifying maximum multipole beyond which the alms are zeroed
	    lfunc: a function mapping multipole ell to the filtering expression
	    ainfo: 	If ainfo is provided, it is an alm_info describing the layout 
	of the input alm. Otherwise it will be inferred from the alm itself.

	Returns:
	    omap: (...,Ny,Nx) ndmap stack of filtered enmaps
	"""
	return alm2map(almxfl(map2alm(imap,ainfo=ainfo,lmax=lmax,spin=0),lfunc=lfunc,ainfo=ainfo),enmap.empty(imap.shape,imap.wcs,dtype=imap.dtype),spin=0,ainfo=ainfo)
	


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
	ainfo = sharp.alm_info(nalm=alm.shape[-1]) if ainfo is None else ainfo
	return ainfo.alm2cl(alm, alm2=alm2)
