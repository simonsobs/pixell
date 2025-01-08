import numpy as np
from . import enmap, utils, wcsutils, curvedsky, multimap

# Note on units:
#  The current version uses "pix" normalization. With flat-sky
#  this is equivalent to physical normalization up to a constant:
#  the average pixel area. However, on the curved sky this isn't
#  the case, and it's better to just work with physical units.
#  By physical units, I mean units where the mean of the square
#  of a wavelet map is the power spectrum value for that map's
#  typical l.

######## Wavelet basis generators ########

class Butterworth:
	"""Butterworth waveleth basis. Built from differences between Butterworth lowpass filters,
	which have a good tradeoff between harmonic and spatial localization. However it doesn't
	have the sharp boundaries in harmonic space that needlets or scale-discrete wavelets do.
	This is a problem when we want to reduce the resolution of the wavelet maps. With a discrete
	cutoff this can be done losslessly, but with these Butterworth wavelets there's always some
	tail of the basis that extends to arbitrarily high l, making resolution reduction lossy.
	This loss is controlled with the tol parameter."""
	# 1+2**a = 1/q => a = log2(1/tol-1)
	def __init__(self, step=2, shape=7, tol=1e-3, lmin=None, lmax=None):
		self.step = step; self.shape = shape; self.tol = tol
		self.lmin = lmin; self.lmax  = lmax
		if self.lmin is not None and self.lmax is not None:
			self._finalize()
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return Butterworth(step=self.step, shape=self.shape, tol=self.tol, lmin=lmin, lmax=lmax)
	def __call__(self, i, l):
		if i == self.n-1: profile  = np.full(l.shape, 1.0)
		else:             profile  = self.kernel(i,   l)
		if i > 0:         profile -= self.kernel(i-1, l)
		return profile**0.5
	def get_variance_basis(self):
		return VarButter(step=self.step, shape=self.shape, tol=self.tol, lmin=self.lmin, lmax=self.lmax)
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
		if self.lmin is not None and self.lmax is not None:
			self._finalize()
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return ButterTrim(step=self.step, shape=self.shape, trim=self.trim, lmin=lmin, lmax=lmax)
	def __call__(self, i, l):
		if i == self.n-1: profile  = np.full(l.shape, 1.0)
		else:             profile  = self.kernel(i,   l)
		if i > 0:         profile -= self.kernel(i-1, l)
		return profile**0.5
	def get_variance_basis(self):
		return VarButter(step=self.step, shape=self.shape, lmin=self.lmin, lmax=self.lmax)
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
		if self.lmin is not None and self.lmax is not None:
			self._finalize()
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return DigitalButterTrim(step=self.step, shape=self.shape, trim=self.trim, lmin=lmin, lmax=lmax)
	def __call__(self, i, l):
		return utils.interpol(self.profiles[i], l[None], order=0)
	def get_variance_basis(self):
		raise NotImplementedError
	def kernel(self, i, l):
		return trim_kernel(1/(1 + (l/(self.lmin*self.step**(i+0.5)))**(self.shape/np.log(self.step))), self.trim)
	def _finalize(self):
		self.n     = int((np.log(self.lmax)-np.log(self.lmin))/np.log(self.step))
		# 1/(1+(l/(lmin*(step**(i+0.5))))**a)*(1+2*trim)-trim = 0
		# => l = ((1+2*trim)/trim-1)**(1/a) * (lmin*(step**(i+0.5)))
		self.lmaxs = np.ceil(self.lmin * ((1+2*self.trim)/self.trim-1)**(np.log(self.step)/self.shape) * self.step**(np.arange(self.n)+0.5)).astype(int)
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
		if self.lmin is not None and self.lmin is not None:
			self._finalize()
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return AdriSD(lamb=self.lamb, lmin=lmin, lmax=lmax)
	@property
	def n(self): return len(self.profiles)
	def __call__(self, i, l):
		return np.interp(l, np.arange(self.profiles[i].size), self.profiles[i])
	def get_variance_basis(self):
		raise NotImplementedError
	def _finalize(self):
		from optweight import wlm_utils
		self.profiles, self.lmaxs = wlm_utils.get_sd_kernels(self.lamb, self.lmax, lmin=self.lmin)



class CosineNeedlet:
	"""From Coulton et al 2023 arxiv:2307.01258"""
	def __init__(self, lpeaks):
		"""
		Cosine-shaped needlets. lpeaks is a list of multipoles
		where each needlet peaks.
		"""
		self.lpeaks = lpeaks
		self.lmaxs = np.append(self.lpeaks[1:],self.lpeaks[-1])
		self.lmins = np.append(self.lpeaks[0],self.lpeaks[:-1])
		self.lmin = self.lpeaks[0]
		self.lmax = self.lpeaks[-1]
	@property
	def n(self): return len(self.lpeaks)
	def __call__(self, i, l):
		lpeaki = self.lpeaks[i]
		out = l*0.
		if i>0:
			lpeakim1 = self.lpeaks[i-1]
			sel1 = np.logical_and(l>=lpeakim1,l<lpeaki)
			out[sel1] = np.cos(np.pi*(lpeaki-l[sel1])/(lpeaki-lpeakim1)/2.)
		if i<(self.n-1):
			lpeakip1 = self.lpeaks[i+1]
			sel2 = np.logical_and(l>=lpeaki,l<lpeakip1)
			out[sel2] = np.cos(np.pi*(l[sel2]-lpeaki)/(lpeakip1-lpeaki)/2.)
		return out
		
##### Variance wavelet basis generators #####

# These are used to implement the variance wavelet transform, which
# calculates how white noise transforms under a wavelet transform.

class VarButter:
	"""Variance basis for Butterworth wavelets."""
	# 1+2**a = 1/q => a = log2(1/tol-1)
	def __init__(self, step=2, shape=7, tol=1e-3, lmin=None, lmax=None):
		self.step = step; self.shape = shape; self.tol = tol
		self.lmin = lmin; self.lmax  = lmax
		self.basis = None
		if self.lmin is not None and self.lmin is not None:
			self._finalize()
	@property
	def n(self): return self.basis.n
	@property
	def lmaxs(self): return self.basis.lmaxs
	def with_bounds(self, lmin, lmax):
		"""Return a new instance with the given multipole bounds"""
		return VarButter(step=self.step, shape=self.shape, tol=self.tol, lmin=lmin, lmax=lmax)
	def __call__(self, i, l):
		return utils.interp(l, self.l, self.kernels[i])
	def _kernel_helper(self, i, rft):
		if i < self.basis.n-1:
			F  = self.basis(i, rft.l)
		else:
			# For the final, unbound wavelength scale, add a cutoff at lmax to avoid
			# summing infinite power that isn't actually present in the map anyway
			kernel = 1/(1 + (rft.l/self.basis.lmax)**(self.basis.shape/np.log(self.basis.step)))
			F      = (kernel - self.basis.kernel(i-1, rft.l))**0.5
		F2 = rft.real2harm(rft.harm2real(F)**2)
		F2 = rft.unpad(F2)
		return F2
	def _finalize(self):
		self.basis   = Butterworth(step=self.step, shape=self.shape, tol=self.tol, lmin=self.lmin, lmax=self.lmax)
		rft = utils.RadialFourierTransform()
		self.kernels = [self._kernel_helper(i, rft) for i in range(self.n)]
		self.l       = rft.unpad(rft.l)

##### Wavelet transforms #####

# How do I actually get a proper invertible wavelet transform? What I have now
# is not orthogonal, so while map -> wave -> map works fine, wave -> map -> wave
# leaks power between wavelet scales due to their overlap. Because the wavelets
# overlap, there are more degrees of freedom in the wavelet coefficients than
# in the input map, meaning that the transform can't be invertible.

class WaveletTransform:
	"""This class implements a wavelet tansform. It provides thw forwards and
	backwards wavelet transforms map2wave and wave2map, where map is a normal enmap
	and the wavelet coefficients are represented as multimaps.

	Example usage:

	 from pixell import enmap, uharm, wavelets, enplot
	 map  = make_some_enmap()
	 # Construct a UHT object, which handles both flat-sky and curved-sky
	 # transforms. By default it selects automatically based on the map geometry,
	 # but it can be forced with the mode="flat" or mode="curved" argument.
	 uht  = uharm.UHT(map.shape, map.wcs)
	 # Construct a wavelet transform using the default ButterTrim kernels.
	 # wt.nlevel will be the number of wavelet scales in use
	 wt   = wavelets.WaveletTransform(uht)
	 # Compute the wavelet coefficients for the map. These are multimaps,
	 # which are conceptually a list of enmaps that can be accessed with
	 # wmap.maps[i], but are actually stored as a single contiguous array,
	 # and can be operated on using mathematical operations just like a
	 # enmap, e.g. wmap *= 2 etc. pixell.multimap has various functions
	 # that make it easier to work with these, which can often be used
	 # to avoid needing to loop over .maps[].
	 #
	 # The wavelet coefficients have the same units as the alms, so for
	 # a homogeneous map, the variance of each wavelet map will match the
	 # power spectrum at that map's typical scale, which can be accessed
	 # using the .lmids member of the wavelet transform object.
	 wmap  = wt.map2wave(map)
	 # Plot the 3rd last (smallest) wavelet scale.
	 enplot.pshow(wmap.maps[-3])
	 # Measure the variance for each wavelet scale, and use it
	 # to inverse-variance weight the map
	 N = multimap.var(wmap)
	 for i, w in enumerate(wmap.maps):
	   w /= N[i]
	 # Transform back to the map
	 omap = wt.wave2map(wmap)
	"""
	def __init__(self, uht, basis=ButterTrim(), ores=None, norms=None, geometries=None):
		"""Initialize the WaveletTransform. Arguments:
		* uht: An inscance of uharm.UHT, which specifies how to do harmonic transforms
		  (flat-sky vs. curved sky and what lmax).
		* basis: A basis-generating function, which provides the definition of the wavelet
		  filters. Defaults to ButterTrim(), which is fast to evaluate and decently local
		  both spatially and harmonically. The minimum and maximum multipole to use, as well
		  as the harmonic resolution, are all controlled by this.
		* ores: The resolution of the wavelet maps. If None, these are automatically
		  determined, and will be variable-resolution with the low-l maps having lower
		  resolution. If ores is a single number, then all the wavelet maps will have
		  this resolution in radians. This is inefficient, but may be useful for plotting
		  purposes. If ores is an array, then it must have length nlevel, and specifies
		  the resolution to use for each individual wavelet scale. This can be a bit
		  tricky to use since nlevel usually is calculated later. Therefore this variant
		  is probably best used with a fully initialized basis object (one where lmin and
		  lmax were passed in when the basis was constructed, after which basis.n gives
		  the number of levels).
		* norms: Used to override the wavelet normalization. Usually not used directly.

		Flat-sky transforms should be exact. Curved-sky transforms become slightly inaccurate
		(%-level, mainly scales near the Nyquist frequency) at low res and in small patches.
		I haven't tracked this down, but hopefully it isn't a big issue.
		"""
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
		# If the user doesn't specify the geometries explicitly (which they normally won't), then
		# calculate them based on ores
		self.geometries = geometries
		if self.geometries is None:
			# Determine the resolution for the wavelet maps, unless the user has
			# already specified it
			if ores is None:
				oress = np.maximum(np.pi/self.basis.lmaxs, ires)
			else:
				oress = np.zeros(self.nlevel)+ores
			# Build the geometries for each wavelet scale
			if uht.mode == "flat":
				self.geometries = [make_wavelet_geometry_flat(uht.shape, uht.wcs, ires, ores) for ores in oress[:-1]] + [(uht.shape, uht.wcs)]
			else:
				self.geometries = [make_wavelet_geometry_curved(uht.shape, uht.wcs, ores) for ores in oress]
		# Precompute our filter and normalization. This can be memory-intensive in
		# flat mode for large maps
		self.filters, self.norms, self.lmids = self._prepare_filters()
		# Override norms if provided. This is used to implement the variance wavelet transform
		if norms is not None: self.norms[:] = norms
	@property
	def shape(self): return self.uht.shape
	@property
	def wcs(self): return self.uht.shape
	@property
	def geometry(self): return self.shape, self.wcs
	@property
	def nlevel(self): return len(self.geometries)
	def map2wave(self, map, owave=None, fl = None, scales=None, fill_value=None):
		"""Transform from an enmap map[...,ny,nx] to a multimap of wavelet coefficients,
		which is effectively a group of enmaps with the same pre-dimensions but varying shape.
		If owave is provided, it should be a multimap with the right shape (compatible with
		the .geometries member of this class), and will be overwritten with the result. In
		any case the resulting wavelet coefficients are returned.

		A filter fl (either an array or a function; see curvedsky.almxfl) can be
		provided that pre-filters the map in spherical harmonic space, e.g. to
		convolve maps to a common beam.

		A list of the indices of wavelet coefficients to be calculated can be provided
		in scales; None defaults to all scales.  For wavelet coefficients that are not
		calculated, a map of zeros wil be provided instead of performing the corresponding
		harmonic to real space transform. Alternatively, a fill_value different from zero
		can be specified.
		"""
		scales = range(len(self.geometries)) if scales is None else scales
		filters, norms, lmids = self.filters, self.norms, self.lmids
		# Output geometry. Can't just use our existing one because it doesn't know about the
		# map pre-dimensions. There should be an easier way to do this.
		geos = [(map.shape[:-2]+tuple(shape[-2:]), wcs) for (shape, wcs) in self.geometries]
		if owave is None: owave = multimap.zeros(geos, map.dtype)
		if self.uht.mode == "flat":
			fmap = enmap.fft(map, normalize=False)
			if fl is not None:
				raise NotImplementedError("Pre-filtering not yet implemented for flat-sky wavelets.")				
			for i, (shape, wcs) in enumerate(self.geometries):
				if i in scales:
					fsmall  = enmap.resample_fft(fmap, shape, norm=None, corner=True)
					fsmall *= filters[i] / (norms[i]*fmap.npix)
					owave.maps[i] = enmap.ifft(fsmall, normalize=False).real
				else:
					owave.maps[i] = enmap.zeros(shape,wcs)
					if fill_value is not None: owave.maps[i][:] = np.nan
					
		else:
			ainfo = curvedsky.alm_info(lmax=self.basis.lmax)
			alm   = curvedsky.map2alm(map, ainfo=ainfo)
			if fl is not None:
				alm = curvedsky.almxfl(alm,fl)
			for i, (shape, wcs) in enumerate(self.geometries):
				if i in scales:
					smallinfo = curvedsky.alm_info(lmax=self.basis.lmaxs[i])
					asmall    = curvedsky.transfer_alm(ainfo, alm, smallinfo)
					smallinfo.lmul(asmall, filters[i]/norms[i], asmall)
					curvedsky.alm2map(asmall, owave.maps[i])
				else:
					owave.maps[i] = enmap.zeros(shape,wcs)
					if fill_value is not None: owave.maps[i][:] = fill_value
		return owave
	def wave2map(self, wave, omap=None):
		"""Transform from the wavelet coefficients wave (multimap), to the corresponding enmap.
		If omap is provided, it must have the correct geometry (the .geometry member of this class),
		and will be overwritten with the result. In any case the result is returned."""
		filters, norms, lmids = self.filters, self.norms, self.lmids
		if self.uht.mode == "flat":
			fomap = enmap.zeros(wave.pre + self.uht.shape[-2:], self.uht.wcs, np.result_type(wave.dtype,0j))
			for i, (shape, wcs) in enumerate(self.geometries):
				fsmall  = enmap.fft(wave.maps[i], normalize=False)
				fsmall *= filters[i] * (norms[i]/fsmall.npix)
				enmap.resample_fft(fsmall, self.uht.shape, fomap=fomap, norm=None, corner=True, op=np.add)
			tmp = enmap.ifft(fomap, normalize=False).real
			if omap is None: omap    = tmp
			else:            omap[:] = tmp
			return omap
		else:
			ainfo = curvedsky.alm_info(lmax=self.basis.lmax)
			oalm  = np.zeros(wave.pre + (ainfo.nelem,), dtype=np.result_type(wave.dtype,0j))
			for i, (shape, wcs) in enumerate(self.geometries):
				smallinfo = curvedsky.alm_info(lmax=self.basis.lmaxs[i])
				asmall    = curvedsky.map2alm(wave.maps[i], ainfo=smallinfo)
				smallinfo.lmul(asmall, filters[i]*norms[i], asmall)
				curvedsky.transfer_alm(smallinfo, asmall, ainfo, oalm, op=np.add)
			if omap is None:
				omap = enmap.zeros(wave.pre + self.uht.shape[-2:], self.uht.wcs, wave.dtype)
			return curvedsky.alm2map(oalm, omap)
	def get_ls(self, i):
		"""Get the multipole indices for wavelet scale i. This will be an enmap
		in if the uht is flat, otherwise it's a 1d array"""
		if self.uht.mode == "flat":
			return enmap.resample_fft(self.uht.l, self.geometries[i][0], norm=None, corner=True)
		else:
			return self.uht.l
	def get_variance_transform(self):
		return WaveletTransform(self.uht, basis=self.basis.get_variance_basis(), norms=self.norms**2, geometries=self.geometries)
	# Helper functions
	def _prepare_filters(self):
		"""Evaluate the filter basis functions for for all filter levels,
		and compute the corresponding normalization factors and average multipoles.
		Returns filters, norms, lmids"""
		filters, norms, lmids = zip(*[self._prepare_filter(i) for i in range(self.nlevel)])
		norms = np.asarray(norms)
		lmids = np.asarray(lmids)
		return filters, norms, lmids
	def _prepare_filter(self, i):
		"""Evaluate the filter basis function for filter level i,
		and compute the corresponding normalization factor and average multipole.
		Returns filter, norm, lmid"""
		ls = self.get_ls(i)
		if self.uht.mode == "flat":
			shape, wcs = self.geometries[i]
			F    = enmap.ndmap(self.basis(i, ls), wcs)
			W    = F**2/enmap.area(shape, wcs)
		else:
			F    = self.basis(i, ls)
			W    = F**2*(2*ls+1)/(4*np.pi)
		Wtot = np.sum(W)
		norm = Wtot**0.5
		lmid = np.sum(W*ls)/Wtot
		return F, norm, lmid

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
