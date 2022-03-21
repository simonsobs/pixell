import numpy as np
from . import enmap, utils, wcsutils, curvedsky, sharp, multimap

######## Wavelet basis generators ########

class Butterworth:
	"""Butterworth waveleth basis. Built from differences between Butterworth lowpass filters,
	which have a good tradeoff between harmonic and spatial localization. However it doesn't
	have the sharp boundaries in harmonic space that needlets or scale-discrete wavelets do.
	This is a problem when we want to reduce the resolution of the wavelet maps. With a discrete
	cutoff this can be done losslessly, but with these Butterworth wavelets there's always some
	tail of the basis that extneds to arbitrarily high l, making resolution reduction lossy.
	This loss is controlled with the tol parameter."""
	# 1+2**a = 1/q => a = log2(1/tol-1)
	def __init__(self, step=2, shape=7, tol=1e-3, lmin=None, lmax=None):
		self.step = step; self.shape = shape; self.tol = tol
		self.lmin = lmin; self.lmax  = lmax
		if lmax is not None:
			if lmin is None: lmin = 1
			self._finalize()
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return Butterworth(step=self.step, shape=self.shape, tol=self.tol, lmin=lmin, lmax=lmax)
	def __call__(self, i, l, half=False):
		if i == self.n-1:      profile  = np.full(l.shape, 1.0)
		else:                  profile  = self.kernel(i,   l)
		if i > 0 and not half: profile -= self.kernel(i-1, l)
		return profile**0.5
	def half(self, i, l): return self(i, l, half=True)
	def kernel(self, i, l):
		return 1/(1 + (l/(self.lmin*self.step**(i+0.5)))**(self.shape/np.log(self.step)))
	def _finalize(self):
		self.n        = int((np.log(self.lmax)-np.log(self.lmin))/np.log(self.step))
		# 1+(l/(lmin*(step**(i+0.5))))**a = 1/tol =>
		# l = (1/tol-1)**(1/a) * lmin*(step**(i+0.5))
		self.lmaxs    = np.round(self.lmin * (1/self.tol-1)**(np.log(self.step)/self.shape) * self.step**(np.arange(self.n)+0.5)).astype(int)
		self.lmaxs[-1] = self.lmax

class ButterTrim:
	"""Butterworth waveleth basis made harmonically compact by clipping off the tails.
	Built from differences between trimmed Butterworth lowpass filters. This trimming
	sacrifices some signal suppression at high radius, but this is a pretty small effect
	even with quite aggressive trimming."""
	def __init__(self, step=2, shape=7, trim=1e-2, lmin=None, lmax=None):
		self.step = step; self.shape = shape; self.trim = trim
		self.lmin = lmin; self.lmax  = lmax
		if lmax is not None:
			if lmin is None: lmin = 1
			self._finalize()
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return ButterTrim(step=self.step, shape=self.shape, trim=self.trim, lmin=lmin, lmax=lmax)
	def __call__(self, i, l, half=False):
		if i == self.n-1:      profile  = np.full(l.shape, 1.0)
		else:                  profile  = self.kernel(i,   l)
		if i > 0 and not half: profile -= self.kernel(i-1, l)
		return profile**0.5
	def half(self, i, l): return self(i, l, half=True)
	def kernel(self, i, l):
		return trim_kernel(1/(1 + (l/(self.lmin*self.step**(i+0.5)))**(self.shape/np.log(self.step))), self.trim)
	def _finalize(self):
		self.n        = int((np.log(self.lmax)-np.log(self.lmin))/np.log(self.step))
		# 1/(1+(l/(lmin*(step**(i+0.5))))**a)*(1+2*trim)-trim = 0
		# => l = ((1+2*trim)/trim-1)**(1/a) * (lmin*(step**(i+0.5)))
		self.lmaxs    = np.ceil(self.lmin * ((1+2*self.trim)/self.trim-1)**(np.log(self.step)/self.shape) * self.step**(np.arange(self.n)+0.5)).astype(int)
		self.lmaxs[-1] = self.lmax

class DigitalButterTrim:
	"""Digitized version of ButterTrim, where the smooth filters are approximated with
	a comb of top-hat functions. This makes the wavelets orthogonal, at the cost of
	introducing poisson noise into the real-space profiles. This effective noise floor
	to the real-space profile makes them couple things to arbitrary large distance at the
	0.1% level in my tests."""
	def __init__(self, step=2, shape=7, trim=1e-2, lmin=None, lmax=None):
		self.step = step; self.shape = shape; self.trim = trim
		self.lmin = lmin; self.lmax  = lmax
		if lmax is not None:
			if lmin is None: lmin = 1
			self._finalize()
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return DigitalButterTrim(step=self.step, shape=self.shape, trim=self.trim, lmin=lmin, lmax=lmax)
	def __call__(self, i, l):
		return utils.interpol(self.profiles[i], l[None], order=0)
	def half(self, i, l): raise NotImplementedError
	def kernel(self, i, l):
		return trim_kernel(1/(1 + (l/(self.lmin*self.step**(i+0.5)))**(self.shape/np.log(self.step))), self.trim)
	def _finalize(self):
		self.n        = int((np.log(self.lmax)-np.log(self.lmin))/np.log(self.step))
		# 1/(1+(l/(lmin*(step**(i+0.5))))**a)*(1+2*trim)-trim = 0
		# => l = ((1+2*trim)/trim-1)**(1/a) * (lmin*(step**(i+0.5)))
		self.lmaxs    = np.ceil(self.lmin * ((1+2*self.trim)/self.trim-1)**(np.log(self.step)/self.shape) * self.step**(np.arange(self.n)+0.5)).astype(int)
		self.lmaxs[-1] = self.lmax
		# Evaluate 1d profiles
		l        = np.arange(self.lmax)
		kernels  = np.array([np.zeros(l.size)]+[digitize(self.kernel(i,l)) for i in range(self.n-1)] + [np.full(l.size,1.0)])
		kernels  = np.sort(kernels,0)
		self.profiles = kernels[1:]-kernels[:-1] # 0 or 1, so no square root needed


class AdriSD:
	"""Scale-discrete wavelet basis provided by Adri's optweight library.
	A bit heavy to initialize."""
	def __init__(self, lamb=2, lmin=None, lmax=None):
		self.lamb = lamb; self.lmin = lmin; self.lmax = lmax
		if lmax is not None:
			if lmin is None: lmin = 1
			self._finalize()
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return AdriSD(lamb=self.lamb, lmin=lmin, lmax=lmax)
	@property
	def n(self): return len(self.profiles)
	def __call__(self, i, l):
		return np.interp(l, np.arange(self.profiles[i].size), self.profiles[i])
	def half(self, i, l): raise NotImplementedError
	def _finalize(self):
		from optweight import wlm_utils
		self.profiles, self.lmaxs = wlm_utils.get_sd_kernels(self.lamb, self.lmax, lmin=self.lmin)

##### Wavelet transforms #####

# How do I actually get a proper invertible wavelet transform? What I have now
# is not orthogonal, so while map -> wave -> map works fine, wave -> map -> wave
# leaks power between wavelet scales due to their overlap. Because the wavelets
# overlap, there are more degrees of freedom in the wavelet coefficients than
# in the input map, meaning that the transform can't be invertible.

class WaveletTransform:
	"""This class implements a wavelet tansform. It provides thw forwards and
	backwards wavelet transforms map2wave and wave2map, where map is a normal enmap
	and the wavelet coefficients are represented as multimaps."""
	def __init__(self, uht, basis=ButterTrim(), ores=None):
		"""Initialize the WaveletTransform. Arguments:
		* uht: An inscance of uharm.UHT, which specifies how to do harmonic transforms
		  (flat-sky vs. curved sky and what lmax).
		* basis: A basis-generating function, which provides the definition of the wavelet
		  filters. Defaults to ButterTrim(), which is fast to evaluate and decently local
		  both spatially and harmonically.

		Flat-sky transforms should be exact. Curved-sky transforms become slightly inaccurate
		on small patches.

		Currently the curved-sky case uses wavelet maps with twice the naively needed resolution
		to make up for the deficiency of CAR quadrature. In the future better CAR quadrature will
		be available, but it would also be possible to use gauss-legendre pixelization internally."""
		self.uht   = uht
		self.basis = basis
		ires       = np.max(enmap.pixshapebounds(uht.shape, uht.wcs))
		# Respect the lmin and lmax in the basis if they are present, but otherwise
		# determine them ourselves.
		if self.basis.lmax is None or self.basis.lmin is None:
			lmin = self.basis.lmin; lmax = self.basis.lmax
			if lmax is None: lmax = min(int(np.ceil(np.pi/ires)),uht.lmax)
			if lmin is None: lmin = min(int(np.ceil(np.pi/np.max(enmap.extent(uht.shape, uht.wcs)))),lmax)
			self.basis = basis.with_bounds(lmin, lmax)
		# Build the geometries for each wavelet scale
		if uht.mode == "flat":
			if ores is None:
				oress = np.maximum(np.pi/self.basis.lmaxs, ires)
				self.geometries = [make_wavelet_geometry_flat(uht.shape, uht.wcs, ires, ores) for ores in oress[:-1]] + [(uht.shape, uht.wcs)]
			else:
				self.geometries = [make_wavelet_geometry_flat(uht.shape, uht.wcs, ires, ores) for l in self.basis.lmaxs]
		else:
			# I thought I would need twice the resolution here, but it doesn't seem necessary
			# May be solved with ducc0 in the future.
			if ores is None:
				oress = np.maximum(np.pi/self.basis.lmaxs, ires)
				self.geometries = [make_wavelet_geometry_curved(uht.shape, uht.wcs, ores) for ores in oress]
			else:
				self.geometries = [make_wavelet_geometry_curved(uht.shape, uht.wcs, ores) for l in self.basis.lmaxs]
		self.filters, self.norms = self.build_filters()
	@property
	def shape(self): return self.uht.shape
	@property
	def wcs(self): return self.uht.shape
	@property
	def geometry(self): return self.shape, self.wcs
	@property
	def nlevel(self): return len(self.geometries)
	def map2wave(self, map, owave=None, half=False):
		"""Transform from an enmap map[...,ny,nx] to a multimap of wavelet coefficients,
		which is effectively a group of enmaps with the same pre-dimensions but varying shape.
		If owave is provided, it should be a multimap with the right shape (compatible with
		the .geometries member of this class), and will be overwritten with the result. In
		any case the resulting wavelet coefficients are returned."""
		# The half-filter is uncommon, so build it on the fly instead of precomputing to
		# not waste memory.
		filters, norms = self.build_filters(True) if half else (self.filters, self.norms)
		# Output geometry. Can't just use our existing one because it doesn't know about the
		# map pre-dimensions. There should be an easier way to do this.
		geos = [(map.shape[:-2]+tuple(shape[-2:]), wcs) for (shape, wcs) in self.geometries]
		if owave is None: owave = multimap.zeros(geos, map.dtype)
		if self.uht.mode == "flat":
			# This normalization is equivalent to True, "pix", True, but avoids the
			# redundant multiplications
			fmap = enmap.fft(map, normalize=False)
			for i, (shape, wcs) in enumerate(self.geometries):
				fsmall  = enmap.resample_fft(fmap, shape, norm=None, corner=True)
				fsmall *= filters[i] / (norms[i]**0.5 * fmap.npix)
				owave.maps[i] = enmap.ifft(fsmall, normalize=False).real
		else:
			# FIXME: Normalization is broken
			ainfo = sharp.alm_info(lmax=self.basis.lmax)
			alm   = curvedsky.map2alm(map, ainfo=ainfo, tweak=self.uht.tweak)
			for i, (shape, wcs) in enumerate(self.geometries):
				smallinfo = sharp.alm_info(lmax=self.basis.lmaxs[i])
				asmall    = sharp.transfer_alm(ainfo, alm, smallinfo)
				smallinfo.lmul(asmall, filters[i]/norms[i]**0.5, asmall)
				curvedsky.alm2map(asmall, owave.maps[i], tweak=self.uht.tweak)
		return owave
	def wave2map(self, wave, omap=None, half=False, individual=False):
		"""Transform from the wavelet coefficients wave (multimap), to the corresponding enmap.
		If omap is provided, it must have the correct geometry (the .geometry member of this class),
		and will be overwritten with the result. In any case the result is returned."""
		if individual: return self._wave2map_individual(wave, omap=omap)
		filters, norms = self.build_filters(True) if half else self.filters, self.norms
		if self.uht.mode == "flat":
			# This normalization is equivalent to True, "pix", True, but avoids the
			# redundant multiplications
			fomap = enmap.zeros(wave.pre + self.uht.shape[-2:], self.uht.wcs, np.result_type(wave.dtype,0j))
			for i, (shape, wcs) in enumerate(self.geometries):
				fsmall  = enmap.fft(wave.maps[i], normalize=False)
				fsmall *= filters[i] * (norms[i]**0.5 / fsmall.npix)
				enmap.resample_fft(fsmall, self.uht.shape, fomap=fomap, norm=None, corner=True, op=np.add)
			tmp = enmap.ifft(fomap, normalize=False).real
			if omap is None: omap    = tmp
			else:            omap[:] = tmp
			return omap
		else:
			# FIXME: Normalization is broken
			ainfo = sharp.alm_info(lmax=self.basis.lmax)
			oalm  = np.zeros(wave.pre + (ainfo.nelem,), dtype=np.result_type(wave.dtype,0j))
			for i, (shape, wcs) in enumerate(self.geometries):
				smallinfo = sharp.alm_info(lmax=self.basis.lmaxs[i])
				asmall    = curvedsky.map2alm(wave.maps[i], ainfo=smallinfo, tweak=self.uht.tweak)
				smallinfo.lmul(asmall, filters[i]*norms[i]**0.5, asmall)
				sharp.transfer_alm(smallinfo, asmall, ainfo, oalm, op=np.add)
			if omap is None:
				omap = enmap.zeros(wave.pre + self.uht.shape[-2:], self.uht.wcs, wave.dtype)
			return curvedsky.alm2map(oalm, omap, tweak=self.uht.tweak)
	def _wave2map_individual(self, wave, omap=None):
		"""Transform from the wavelet coefficients wave (multimap), to a separate enmap for
		each wavelet scale. If omap is provided, it must have the correct geometry
		((nlevel,) + the .geometry member of this class), and will be overwritten with the
		result. In any case the result is returned."""
		if self.uht.mode == "flat":
			# This normalization is equivalent to True, "pix", True, but avoids the
			# redundant multiplications
			fomap = enmap.zeros((self.nlevel,)+wave.pre + self.uht.shape[-2:], self.uht.wcs, np.result_type(wave.dtype,0j))
			for i, (shape, wcs) in enumerate(self.geometries):
				fsmall  = enmap.fft(wave.maps[i], normalize=False)
				fsmall *= self.filters[i] * (self.norms[i]**0.5 / fsmall.npix)
				enmap.resample_fft(fsmall, self.uht.shape, fomap=fomap[i], norm=None, corner=True, op=np.add)
			tmp = enmap.ifft(fomap, normalize=False).real
			if omap is None: omap    = tmp
			else:            omap[:] = tmp
			return omap
		else:
			ainfo = sharp.alm_info(lmax=self.basis.lmax)
			if omap is None:
				omap = enmap.zeros((self.nlevel,)+wave.pre + self.uht.shape[-2:], self.uht.wcs, wave.dtype)
			for i, (shape, wcs) in enumerate(self.geometries):
				smallinfo = sharp.alm_info(lmax=self.basis.lmaxs[i])
				asmall    = curvedsky.map2alm(wave.maps[i], ainfo=smallinfo, tweak=self.uht.tweak)
				smallinfo.lmul(asmall, self.filters[i]*self.norms[i]**0.5, asmall)
				curvedsky.alm2map(asmall, omap[i], tweak=self.uht.tweak)
			return omap
	def get_ls(self, i):
		"""Get the multipole indices for wavelet scale i"""
		if self.uht.mode == "flat":
			return enmap.resample_fft(self.uht.l, self.geometries[i][0], norm=None, corner=True)
		else:
			return self.uht.l
	def build_filters(self, half=False):
		basis = self.basis if not half else self.basis.half
		if self.uht.mode == "flat":
			filters = [enmap.ndmap(basis(i, self.get_ls(i)), geo[1]) for i, geo in enumerate(self.geometries)]
			norms   = np.array([np.sum(f**2)/self.uht.npix for f in filters])
		else:
			filters = [basis(i, self.get_ls(i)) for i, geo in enumerate(self.geometries)]
			norms   = [np.sum(f**2*(2*self.uht.l+1)) for f in filters]
		return filters, norms

class HaarTransform:
	"""A simple 2d Haar-ish wavelet transform. Fast due to not using harmonic space, and
	has the nice property of being orthoginal, which means that there's no mode leakage.
	Does not take the sky's curvature into account. This might not be a big deal since
	wavelets support position-dependent models anyway.

	The way this is implemented it isn't fully orthogonal, since the wavelets have slightly
	more degrees of freedom than the map.
	"""
	def __init__(self, nlevel, ref=[0,0]):
		"""Initialize the HaarTransform.
		* nlevel: The number of wavelets to compute. Each level halves the map resolution.
		* ref: The coordinates (dec,ra in radians) of a reference point that the resolution levels should try to keep at a consistent pixel location. Useful for making sure that different patches have compatible wavelet maps. Set to None to disable. Defaults to ra=0, dec=0."""
		self.nlevel = nlevel
		self.ref    = ref
	def map2wave(self, map):
		"""Transform from an enmap map[...,ny,nx] to a multimap of wavelet coefficients,
		which is effectively a group of enmaps with the same pre-dimensions but varying shape."""
		omaps = []
		for i in range(self.nlevel):
			off  = enmap.get_downgrade_offset(*map.geometry, 2, self.ref)
			down = enmap.downgrade(map, 2, off=off, inclusive=True)
			omaps.append(map-enmap.upgrade(down, 2, off=off, inclusive=True, oshape=map.shape))
			map  = down
		omaps.append(map)
		return multimap.multimap(omaps[::-1])
	def wave2map(self, wave):
		"""Transform from the wavelet coefficients wave (multimap), to the corresponding enmap."""
		omap = wave.maps[0].copy()
		for i in range(1, wave.nmap):
			off  = enmap.get_downgrade_offset(*wave.geometries[i], 2, self.ref)
			omap = wave.maps[i] + enmap.upgrade(omap, 2, off=off, inclusive=True, oshape=wave.geometries[i].shape)
		return omap

####### Helper functions #######

def trim_kernel(a, tol): return np.clip(a*(1+2*tol)-tol,0,1)

def digitize(a):
	"""Turn a smooth array with values between 0 and 1 into an on/off array
	that approximates it."""
	f = np.round(np.cumsum(a))
	return np.concatenate([[1],f[1:]!=f[:-1]])

def make_wavelet_geometry_flat(ishape, iwcs, ires, ores, margin=4):
	# I've found that, possibly due to rounding or imprecise scaling, I sometimes need to add up
	# to +2 to avoid parts of some basis functions being cut off. I add +5 to get some margin -
	# it's cheap anyway - though it would be best to get to the bottom of it.
	oshape    = (np.ceil(np.array(ishape[-2:])*ires/ores)).astype(int)+margin
	oshape    = np.minimum(oshape, ishape[-2:])
	owcs      = wcsutils.scale(iwcs, oshape[-2:]/ishape[-2:], rowmajor=True, corner=True)
	return oshape, owcs

def make_wavelet_geometry_curved(ishape, iwcs, ores, minres=2*utils.degree):
	# NOTE: This function assumes:
	# * cylindrical coordinates
	# * dec increases with y, ra decreases with x
	# The latter can be generalized with a fewe more checks.
	# We need to be able to perform SHTs on these, so we can't just generate an arbitrary
	# pixelization. Find the fullsky geometry with the desired resolution, and cut out the
	# part best matching our patch.
	res = min(np.pi/np.ceil(np.pi/ores),minres)
	# Find the bounding box of our patch, and make sure it's in bounds.
	box = enmap.corners(ishape, iwcs)
	box[:,0] = np.clip(box[:,0], -np.pi/2, np.pi/2)
	box[1,1] = box[0,1] + np.clip(box[1,1]-box[0,1],-2*np.pi,2*np.pi)
	# Build a full-sky geometry for which we have access to quadrature
	tgeo = enmap.Geometry(*enmap.fullsky_geometry(res=res))
	# Figure out how we need to crop this geometry to match our target patch
	pbox = enmap.skybox2pixbox(*tgeo, box)
	pbox[np.argmax(pbox[:,0]),0] += 1 # Make sure we include the final full-sky row
	pbox[:,1] += utils.rewind(pbox[0,1], period=tgeo.shape[1])-pbox[0,1]
	# Round to whole pixels and slice the geometry
	pbox = utils.nint(pbox)
	# Pad the pixbox with extra pixels if requested
	oshape, owcs = tgeo.submap(pixbox=pbox)
	return oshape, owcs
