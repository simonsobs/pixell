"""This module provides functions that make it easier to write curvature-agnostic code
that looks the same whether it's operating using the flat-sky approximation (2d FFTs)
or the curved sky (SHTs and alms)"""
import numpy as np
from . import utils, enmap, curvedsky

# Unified Harmonic Transform
class UHT:
	"""The UHT class provides a Unified Harmonic Transform, which provides both
	FFTs and SHTs under a unified interface. The purpose of this class is to be able
	to write various filtering and convolution operations in a flat-sky-agnostic way.
	
	After initialization, the main purpose of the UHT object is to transform between
	pixel-space ("map") and harmonic-space ("harm") representations. In both flat-sky
	and curved-sky modes, the map-space representation is an enmap, but the harmonic
	space reprsentation differ for the two. In flat-sky the harmonic representation is
	also an enmap, while in curved-sky mode it's an alm.

	The UHT object also provides functions for working with 1d functions or r or l.
	Functions of r can be transformed to harmonic form using rprof2hprof. Due to
	the symmetry, the harmonic space reprsentation of these is a bit different from
	that of the maps. In flat mode, the result is still an enmap, but in curved mode
	it's just a 1d function of l. If you already have a function of l, it can be put
	in the standard form using the lexpand function.

	You can multiply the harmonic representation of profiles with that of the maps
	by using lmul.

	This all sounds complicated, but the point is that this UHT object takes care
	of these details. For example, here is how you would smooth a map given the
	real-space reprsentation of a beam:

		uht  = UHT(shape, wcs)
		beam = uth.rprof2hprof(br, r)
		omap = uht.harm2map(uht.lmul(beam, uht.map2harm(map)))

	Notice how none of this depended on flat/curved sky or the the details of the harmonic
	space representation.

	Explanation of terms and representations:

	* map:   An enmap with shape [...,ny,nx] representing some or all of the sky
	* rprof: Radial profiles br[...,nr] valuated at the points r[nr], typically a beam.
	* lprof: A 1d function of l: bl[...,nl]. Does not have to match the internal lmax.
	* harm:  The harmonic-space representation of a map, as produced by map2harm.
	  - flat:   A complex enmap[...,ny,nx] corresponding to the the multipoles self.l
	  - curved: Alms[...,self.lmax+1]
	* hprof: The harmonic-space representation of an isotropic function, as produced by
	  rprof2hprof or lprof2hprof. Can be multiplied with harm using lmul.
	  - flat:   An enmap[...,ny,nx] corresponding to the multipoles self.l
	  - curved: A 1d array [...,self.lmax+1]
	"""
	def __init__(self, shape, wcs, mode="auto", lmax=None, max_distortion=0.1, niter=0):
		"""Initialize a UHT object.
		Arguments:
			shape, wcs: The geometry of the enmaps the UHT object will be used on.
			mode: The flat/curved sky mode to use. "flat" selects the flat-sky approximation,
				"curved" uses full SHTs, and "auto" (the detault) selects flat or curved based on
				the estimated maximum distortion in the map.
			lmax: The maximum multipole to use in calculations. Ignored in flat-sky mode.
			max_distortion: The maximum relative scale difference across the map that's acceptable
				before curved sky is necessary."""
		self.shape, self.wcs = shape[-2:], wcs
		self.area = enmap.area(self.shape, self.wcs)
		self.fsky = self.area/(4*np.pi)
		if mode == "auto":
			dist = estimate_distortion(shape, wcs)
			if dist <= max_distortion: mode = "flat"
			else:                      mode = "curved"
		self.mode = mode
		self.quad = None
		self.niter= niter
		if mode == "flat":
			self.l    = enmap.modlmap(shape, wcs)
			self.lmax = utils.nint(np.max(self.l))
			# FIXME: In the middle of debugging this. Was
			# nper = l[0,1]*l[1,0]/pi = 1/fsky, but was inconsistent with sum_harm for curved
			self.nper = 1/self.fsky
			self.ntot = self.nper*self.shape[-2]*self.shape[-1]
		elif mode == "curved":
			if lmax is None:
				res  = np.min(np.abs(wcs.wcs.cdelt))*utils.degree
				lmax = res2lmax(res)
			self.lmax = lmax
			self.l    = np.arange(lmax+1)
			self.ainfo= curvedsky.alm_info(lmax=lmax)
			self.nper = 2*self.l+1
			self.ntot = np.sum(self.nper)
		else:
			raise ValueError("Unrecognized mode in UHT: '%s'" % (str(mode)))
	@property
	def npix(self): return self.shape[-2]*self.shape[-1]
	def map2harm(self, map, spin=0):
		if self.mode == "flat":
			return enmap.map2harm(map, spin=spin, normalize="phys")
		else:
			return curvedsky.map2alm(map, ainfo=self.ainfo, spin=spin, niter=self.niter)
	def harm2map(self, harm, spin=0):
		if self.mode == "flat":
			return enmap.harm2map(harm, spin=spin, normalize="phys").real
		else:
			rtype= np.zeros(1, harm.dtype).real.dtype
			omap = enmap.zeros(harm.shape[:-1]+self.shape, self.wcs, rtype)
			return curvedsky.alm2map(harm, omap, ainfo=self.ainfo, spin=spin)
	def harm2map_adjoint(self, map, spin=0):
		if self.mode == "flat":
			return enmap.harm2map_adjoint(map, spin=spin, normalize="phys")
		else:
			return curvedsky.alm2map_adjoint(map, ainfo=self.ainfo)
	def map2harm_adjoint(self, harm, spin=0):
		if self.mode == "flat":
			return enmap.map2harm_adjoint(harm, spin=spin, normalize="phys")
		else:
			rtype= np.zeros(1, harm.dtype).real.dtype
			omap = enmap.zeros(harm.shape[:-1]+self.shape, self.wcs, rtype)
			return curvedsky.map2alm_adjoint(harm, omap, ainfo=self.ainfo, spin=spin, niter=self.niter)
	def quad_weights(self):
		"""Returns the quadrature weights array W. This will broadcast correctly
		with maps, but may not have the same dimensions due to symmetries.
		map2harm = harm2map_adjoint * quad_weight"""
		if self.quad is None:
			if self.mode == "flat":
				self.quad = enmap.pixsizemap(self.shape, self.wcs, broadcastable=True)
			else:
				self.quad = curvedsky.quad_weights(self.shape, self.wcs)[:,None]
		return self.quad
	def rprof2hprof(self, br, r):
		"""Like map2harm, but for a 1d function of r."""
		if self.mode == "flat":
			return profile2harm_flat_2d(br, r, self.shape, self.wcs)
		else:
			return curvedsky.profile2harm(br, r, lmax=self.lmax)
	def hprof2rprof(self, harm, r):
		"""Inverse of hprof2rprof"""
		if self.mode == "flat":
			return harm2profile_flat_2d(harm+0j, r)
		else:
			return curvedsky.harm2profile(harm, r)
	def lprof2hprof(self, lprof):
		if self.mode == "flat":
			return enmap.enmap(utils.interpol(lprof, self.l[None], order=1, border="constant"), self.wcs, copy=False)
		else:
			if lprof.shape[-1] >= self.lmax+1:
				return lprof[...,:self.lmax+1]
			else:
				return np.concatenate([lprof, np.zeros(lprof.shape[:-1]+(self.lmax+1-lprof.shape[-1],), lprof.dtype)], -1)
	def hprof2harm(self, hprof):
		if self.mode == "flat":
			return hprof.copy()
		else:
			mapping = self.ainfo.get_map()
			return hprof[...,mapping[:,0]]
	def hmul(self, hprof, harm, inplace=False):
		"""Perform the multiplication hprof*harm -> harm. See the UHT class docstring for
		the meaning of these terms. In flat mode, hprof must be [ny,nx], [ncomp,ny,nx] or
		[ncomp,ncomp,ny,nx].  In curved mode, hprof must be [nl], [ncomp,nl] or [ncomp,ncomp,nl]."""
		harm = np.asanyarray(harm)
		if self.mode == "flat":
			if inplace: harm[:] = enmap.map_mul(hprof, harm)
			else:       harm    = enmap.map_mul(hprof, harm)
			return harm
		else:
			out = harm if inplace else None
			harm = harm.astype(np.result_type(harm,0j), copy=False)
			return self.ainfo.lmul(harm, hprof, out=out)
	def hrand(self, hprof):
		"""Draw a random realization of the harmonic profile hprof"""
		if self.mode == "flat":
			noise = enmap.rand_gauss_harm(self.shape, self.wcs)
			return enmap.map_mul(enmap.multi_pow(hprof/noise.pixsize(),0.5), noise)
		else:
			return curvedsky.rand_alm(hprof, lmax=self.lmax)
	def harm2powspec(self, harm, harm2=None, patch=False):
		"""Compute the pseudo power spectrum corresponding to harm, or a cross-spectrum if harm2 is given.
		Return an hprof. If patch == True, then the harm is assumed to have been measured from
		map2harm, and an fsky correction will be applied in the curved-sky case."""
		if self.mode == "flat":
			return enmap.calc_ps2d(harm, harm2)
		else:
			powspec = curvedsky.alm2cl(harm, harm2)
			if patch: powspec /= self.fsky
			return powspec
	def sum_hprof(self, hprof):
		"""Sum the total signal in a harmonic profile, typically a power spectrum"""
		hprof = np.asanyarray(hprof)
		if self.mode == "flat":
			return np.sum(hprof*self.nper,(-2,-1))
		else:
			return np.sum(hprof*self.nper,-1)
	def mean_hprof(self, hprof): return self.sum_hprof(hprof)/self.ntot
	def hprof_rpow(self, hprof, power):
		"""Raises hprof "hprof" to the power "power" in real-space.
		Effectively map2harm(harm2map(hprof)**power), but works on
		harmonic profiles instead of full fourier maps/alms.
		Maybe this function is too specific to have in this class, but
		I needed it and it wasn't trivial to get right.
		"""
		if self.mode == "flat":
			norm = enmap.area(self.shape, self.wcs)**0.5
			map  = self.harm2map(hprof/norm+0j)
			return self.map2harm(map**power)*norm
		else:
			# Estimate the resolution from the beam
			sigma = 1/max(1,np.where(hprof > np.max(hprof)*np.exp(-0.5))[0][-1])
			r     = np.arange(0, 20*sigma, sigma/10)
			rprof = self.hprof2rprof(hprof, r)
			return self.rprof2hprof(rprof**power, r)

####################
# Helper functions #
####################

def profile2harm_flat(br, r, oversample=2, pad_factor=2):
	"""Flat-sky approximation to curvedsky.profile2harm. Accurate to about 0.5% for a 1.4 arcmin
	fwhm beam. Only supports a 1d br"""
	# Build a 2d pixelization we will evaluate the ffts on
	res  = beam2res(br, r)
	rmax = beam2rmax(br, r)*pad_factor
	n    = 2*utils.nint(rmax/res*oversample)+1
	shape, wcs = enmap.geometry(pos=[0,0], res=res/oversample, shape=(n,n), proj="car")
	# Compute the 2d beam in harmonic space
	lbeam_2d = profile2harm_flat_2d(br, r, shape, wcs)
	# Reduce to equispaced standard beam
	bl_tmp, l_tmp = lbeam_2d.lbin()
	lmax = res2lmax(res)
	l  = np.arange(lmax+1)
	bl = np.interp(l, l_tmp, bl_tmp)
	return bl

def profile2harm_flat_2d(br, r, shape, wcs):
	"""Given a 1d beam br(r), compute the 2d beam transform bl(ly,lx) for
	the l-space of the map with the given shape, wcs, assuming a flat sky.
	Despite the name, it is not specific to beams - any real-space function of r
	will do. br can have arbitrary pre-dimensions [...,nr]"""
	br     = np.asarray(br)
	cpix   = np.array(shape[-2:])//2-1
	cpos   = enmap.pix2sky(shape, wcs, cpix)
	rmap   = enmap.shift(enmap.modrmap(shape, wcs, cpos), -cpix)
	bmap   = enmap.ndmap(utils.interp(rmap, r, br, right=0), wcs)
	# Normalize the beam so that l=0 corresponds to the sky mean of the beam,
	# like it is for get_lbeam_exact
	harm  = enmap.fft(bmap, normalize=False).real
	harm *= harm.pixsize()
	return harm

def harm2profile_flat_2d(harm, r=None):
	"""Inverse of profile2harm_flat_2d. harm should be an enmap.
	r is [:] in radians."""
	bmap = enmap.ifft(harm, normalize=False).real
	bmap/= harm.pixsize() * harm.npix
	cpix = np.array(harm.shape[-2:])//2-1
	cpos = bmap.pix2sky(cpix)
	bmap = enmap.shift(bmap, cpix, keepwcs=True)
	wbr, wr = bmap.rbin(center=cpos)
	if r is None: return wbr, r
	else:         return utils.interp(r, wr, wbr, right=0)

def beam2res(br, r):
	fwhm = 2*r[np.where(br>=br[0]*0.5)[0][-1]]
	res  = fwhm/3
	return res

def beam2rmax(br, r, tol=1e-5, return_index=False):
	imax = np.where(br>=br[0]*tol)[0][-1]
	if return_index: return r[imax], imax
	else:            return r[imax]

def res2lmax(res):
	"""Get the lmax needed to represent the spatial scale res in radians"""
	return utils.nint(np.pi/res)

def estimate_distortion(shape, wcs):
	"""Get the maximum distortion in the map, assuming a cylindrical projection"""
	dec1, dec2 = enmap.corners(shape, wcs)[:,0]
	rmin = min(np.cos(dec1),np.cos(dec2))
	rmax = 1 if not dec1*dec2 > 0 else max(np.cos(dec1),np.cos(dec2))
	return rmax/rmin-1
