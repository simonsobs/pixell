from __future__ import print_function
import numpy as np, scipy.ndimage, warnings, astropy.io.fits, sys, time
from . import utils, wcsutils, powspec, fft as enfft

# Things that could be improved:
#  1. We assume exactly 2 WCS axes in spherical projection in {dec,ra} order.
#     It would be nice to support other configurations too. I have for example
#     needed [det,ra] or even [time,det,ra]. Adding support for this would
#     probably necessitate breaking backwards compatibility due to units.
#     WCS uses the units specified in the fits file, but I use radians.
#     Once we allos non-degree axes, the simple pi/180 conversion I use
#     won't work for all axes. It is simpler to just go with the flow and
#     use the same units as wcs. I need to think about how this would
#     interact with fourier units. Also, reordering or removing axes
#     can be difficult. I disallow that now, but for > 2 wcs dimensions,
#     these would be useful operations.
#  2. Passing around shape, wcs, dtype all the time is tedious. A simple
#     geometry object would make this less tedious, as long as it is
#     simple to override individual properties.

extent_model = ["subgrid"]

# Python 2/3 compatibility
try: basestring
except NameError: basestring = str

# PyFits uses row-major ordering, i.e. C ordering, while the fits file
# itself uses column-major ordering. So an array which is (ncomp,ny,nx)
# will be (nx,ny,ncomp) in the file. This means that the axes in the ndmap
# will be in the opposite order of those in the wcs object.
class ndmap(np.ndarray):
	"""Implements (stacks of) flat, rectangular, 2-dimensional maps as a dense
	numpy array with a fits WCS. The axes have the reverse ordering as in the
	fits file, and hence the WCS object. This class is usually constructed by
	using one of the functions following it, much like numpy arrays. We assume
	that the WCS only has two axes with unit degrees. The ndmap itself uses
	radians for everything."""
	def __new__(cls, arr, wcs):
		"""Wraps a numpy and bounding box into an ndmap."""
		obj = np.asarray(arr).view(cls)
		obj.wcs = wcs.deepcopy()
		return obj
	def __array_finalize__(self, obj):
		if obj is None: return
		self.wcs = getattr(obj, "wcs", None)
	def __repr__(self):
		return "ndmap(%s,%s)" % (np.asarray(self), wcsutils.describe(self.wcs))
	def __str__(self): return repr(self)
	def __array_wrap__(self, arr, context=None):
		if arr.ndim < 2: return arr
		return ndmap(arr, self.wcs)
	def copy(self, order='K'):
		return ndmap(np.copy(self,order), self.wcs)
	def sky2pix(self, coords, safe=True, corner=False): return sky2pix(self.shape, self.wcs, coords, safe, corner)
	def pix2sky(self, pix,    safe=True, corner=False): return pix2sky(self.shape, self.wcs, pix,    safe, corner)
	def box(self): return box(self.shape, self.wcs)
	def posmap(self, safe=True, corner=False): return posmap(self.shape, self.wcs, safe=safe, corner=corner)
	def pixmap(self): return pixmap(self.shape, self.wcs)
	def lmap(self, oversample=1): return lmap(self.shape, self.wcs, oversample=oversample)
	def modlmap(self, oversample=1): return modlmap(self.shape, self.wcs, oversample=oversample)
	def modrmap(self, safe=True, corner=False): return modrmap(self.shape, self.wcs, safe=safe, corner=corner)
	def area(self): return area(self.shape, self.wcs)
	def pixsize(self): return pixsize(self.shape, self.wcs)
	def pixshape(self, signed=False): return pixshape(self.shape, self.wcs, signed=signed)
	def pixsizemap(self): return pixsizemap(self.shape, self.wcs)
	def extent(self, method="default", signed=False): return extent(self.shape, self.wcs, method=method, signed=signed)
	@property
	def preflat(self):
		"""Returns a view of the map with the non-pixel dimensions flattened."""
		return self.reshape(-1, self.shape[-2], self.shape[-1])
	@property
	def npix(self): return np.product(self.shape[-2:])
	@property
	def geometry(self): return self.shape, self.wcs
	def project(self, shape, wcs, order=3, mode="constant", cval=0, prefilter=True, mask_nan=True, safe=True): return project(self, shape, wcs, order, mode=mode, cval=cval, prefilter=prefilter, mask_nan=mask_nan, safe=safe)
	def at(self, pos, order=3, mode="constant", cval=0.0, unit="coord", prefilter=True, mask_nan=True, safe=True): return at(self, pos, order, mode=mode, cval=0, unit=unit, prefilter=prefilter, mask_nan=mask_nan, safe=safe)
	def autocrop(self, method="plain", value="auto", margin=0, factors=None, return_info=False): return autocrop(self, method, value, margin, factors, return_info)
	def apod(self, width, profile="cos", fill="zero"): return apod(self, width, profile=profile, fill=fill)
	def stamps(self, pos, shape, aslist=False): return stamps(self, pos, shape, aslist=aslist)
	@property
	def plain(self): return ndmap(self, wcsutils.WCS(naxis=2))
	def padslice(self, box, default=np.nan): return padslice(self, box, default=default)
	def center(self): return center(self.shape,self.wcs)
	def downgrade(self, factor): return downgrade(self, factor)
	def upgrade(self, factor): return upgrade(self, factor)
	def fillbad(self, val=0, inplace=False): fillbad(self, val=val, inplace=inplace)
	def to_healpix(self, nside=0, order=3, omap=None, chunk=100000, destroy_input=False):
		return to_healpix(self, nside=nside, order=order, omap=omap, chunk=chunk, destroy_input=destroy_input)
	def to_flipper(self, omap=None, unpack=True): return to_flipper(self, omap=omap, unpack=unpack)
	def __getitem__(self, sel):
		# Split sel into normal and wcs parts.
		sel1, sel2 = utils.split_slice(sel, [self.ndim-2,2])
		# If any wcs-associated indices are None, then we don't know how to update the
		# wcs, and assume the user knows what it's doing
		if any([s is None for s in sel2]):
			return ndmap(np.ndarray.__getitem__(self, sel), self.wcs)
		if len(sel2) > 2:
			raise IndexError("too many indices")
		# If the wcs slice includes direct indexing, so that wcs
		# axes are lost, then degrade to a normal numpy array,
		# since this class assumes that the two last axes are
		# wcs axes.
		if any([type(s) is not slice for s in sel2]):
			return np.asarray(self)[sel]
		# Otherwise we will return a full ndmap, including a
		# (possibly) sliced wcs.
		_, wcs = slice_geometry(self.shape[-2:], self.wcs, sel2)
		return ndmap(np.ndarray.__getitem__(self, sel), wcs)
	def __getslice__(self, a, b=None, c=None): return self[slice(a,b,c)]
	def submap(self, box, mode=None, wrap="auto"):
		"""Extract the part of the map inside the given coordinate box
		box : array_like
			The [[fromy,fromx],[toy,tox]] bounding box to select.
			The resulting map will have a bounding box as close
			as possible to this, but will differ slightly due to
			the finite pixel size.
		mode : str
			How to handle partially selected pixels:
			 "round": round bounds using standard rules
			 "floor": both upper and lower bounds will be rounded down
			 "ceil":  both upper and lower bounds will be rounded up
			 "inclusive": lower bounds are rounded down, and upper bounds up
			 "exclusive": lower bounds are rounded up, and upper bounds down"""
		return submap(self, box, mode=mode, wrap=wrap)
	def subinds(self, box, mode=None, cap=True):
		return subinds(self.shape, self.wcs, box=box, mode=mode, cap=cap)
	def write(self, fname, fmt=None):
		write_map(fname, self, fmt=fmt)

def submap(map, box, mode=None, wrap="auto", iwcs=None):
	"""Extract the part of the map inside the given coordinate box
	box : array_like
		The [[fromy,fromx],[toy,tox]] bounding box to select.
		The resulting map will have a bounding box as close
		as possible to this, but will differ slightly due to
		the finite pixel size.
	mode : str
		How to handle partially selected pixels:
		 "round": round bounds using standard rules
		 "floor": both upper and lower bounds will be rounded down
		 "ceil":  both upper and lower bounds will be rounded up
		 "inclusive": lower bounds are rounded down, and upper bounds up
		 "exclusive": lower bounds are rounded up, and upper bounds down
		The iwcs argument allows the wcs to be overriden. This is usually
		not necessary."""
	if iwcs is None: iwcs = map.wcs
	ibox   = subinds(map.shape, iwcs, box, mode=mode, cap=False)
	def helper(b):
		if b[2] >= 0: return False, slice(b[0],b[1],b[2])
		else:         return True,  slice(b[1]-b[2],b[0]-b[2],-b[2])
	yflip, yslice = helper(ibox[:,0])
	xflip, xslice = helper(ibox[:,1])
	oshape, owcs = slice_geometry(map.shape, iwcs, (yslice, xslice), nowrap=True)
	omap = extract(map, oshape, owcs, wrap=wrap, iwcs=iwcs)
	# Unflip if neccessary
	if yflip: omap = omap[...,::-1,:]
	if xflip: omap = omap[...,:,::-1]
	return omap

def subinds(shape, wcs, box, mode=None, cap=True):
	"""Helper function for submap. Translates the bounding
	box provided into a pixel units. Assumes rectangular
	coordinates.

	When translated to box into pixels, the result will in general have
	fractional pixels, which need to be rounded before we can do any slicing.
	To get as robust results as possible, we want
	 1. two boxes that touch should results in iboxses that also touch.
	    This means that upper and lower bounds must be handled consistently.
	    inclusive and exclusive modes break this, and should be used with caution.
	 2. tiny floating point errors should not usually be able to cause
	    the ibox to change. Most boxes will have some simple fraction of
	    a whole degree, and most have pixels with centers at a simple fraction
	    of a whole degree. Hence, it is likely that box edges will fall
	    almost exactly on an integer pixel value. floor and ceil will
	    then move us around by a whole pixel based on tiny numerical
	    jitter around this value. Hence these should be used with caution.
	These concerns leave us with mode = "round" as the only generally
	safe alternative, which is why it's default.
	"""
	if mode is None: mode = "round"
	box = np.asarray(box)
	# Translate the box to pixels
	bpix = skybox2pixbox(shape, wcs, box, include_direction=True)
	if   mode == "round": bpix = np.round(bpix)
	elif mode == "floor": bpix = np.floor(bpix)
	elif mode == "ceil":  bpix = np.ceil(bpix)
	elif mode == "inclusive": bpix = [np.floor(bpix[0]),np.ceil (bpix[1]), bpix[2]]
	elif mode == "exclusive": bpix = [np.ceil (bpix[0]),np.floor(bpix[1]), bpix[2]]
	else: raise ValueError("Unrecognized mode '%s' in subinds" % str(mode))
	bpix = np.array(bpix, int)
	if cap:
		# Make sure we stay inside our map bounds
		for b, n in zip(bpix.T,shape[-2:]):
			if b[2] > 0: b[:2] = [max(b[0],  0),min(b[1], n)]
			else:        b[:2] = [min(b[0],n-1),max(b[1],-1)]
	return bpix

def slice_geometry(shape, wcs, sel, nowrap=False):
	"""Slice a geometry specified by shape and wcs according to the
	slice sel. Returns a tuple of the output shape and the correponding
	wcs."""
	wcs = wcs.deepcopy()
	pre, shape = shape[:-2], shape[-2:]
	oshape = np.array(shape)
	# The wcs object has the indices in reverse order
	for i,s in enumerate(sel[-2:]):
		s = utils.expand_slice(s, shape[i], nowrap=nowrap)
		j = -1-i
		start = s.start if s.step > 0 else s.start + 1
		wcs.wcs.crpix[j] -= start+0.5
		wcs.wcs.crpix[j] /= s.step
		wcs.wcs.cdelt[j] *= s.step
		wcs.wcs.crpix[j] += 0.5
		oshape[i] = (s.stop-s.start+s.step-np.sign(s.step))//s.step
	return tuple(pre)+tuple(oshape), wcs

def scale_geometry(shape, wcs, scale):
	scale  = np.zeros(2)+scale
	oshape = tuple(shape[:-2])+tuple(utils.nint(shape[-2:]*scale))
	owcs   = wcsutils.scale(wcs, scale, rowmajor=True)
	return oshape, owcs

def get_unit(wcs):
	if wcsutils.is_plain(wcs): return 1
	else: return utils.degree

def box(shape, wcs, npoint=10, corner=True):
	"""Compute a bounding box for the given geometry."""
	# Because of wcs's wrapping, we need to evaluate several
	# extra pixels to make our unwinding unambiguous
	pix = np.array([np.linspace(0,shape[-2],num=npoint,endpoint=True),
		np.linspace(0,shape[-1],num=npoint,endpoint=True)])
	if corner: pix -= 0.5
	coords = wcsutils.nobcheck(wcs).wcs_pix2world(pix[1],pix[0],0)[::-1]
	if wcsutils.is_plain(wcs):
		return np.array(coords).T[[0,-1]]
	else:
		return utils.unwind(np.array(coords)*utils.degree).T[[0,-1]]

def enmap(arr, wcs=None, dtype=None, copy=True):
	"""Construct an ndmap from data.

	Parameters
	----------
	arr : array_like
		The data to initialize the map with.
		Must be at least two-dimensional.
	wcs : WCS object
	dtype : data-type, optional
		The data type of the map.
		Default: Same as arr.
	copy : boolean
		If true, arr is copied. Otherwise, a referance is kept."""
	def has_wcs(m):
		try:
			m.wcs
			return True
		except AttributeError:
			return False
	if wcs is None:
		if has_wcs(arr):
			wcs = arr.wcs
		elif isinstance(arr, list) and len(arr) > 0 and has_wcs(arr[0]):
			wcs = arr[0].wcs
		else:
			wcs = wcsutils.WCS(naxis=2)
	if copy:
		arr = np.asanyarray(arr, dtype=dtype).copy()
	return ndmap(arr, wcs)

def empty(shape, wcs=None, dtype=None):
	return enmap(np.empty(shape, dtype=dtype), wcs, copy=False)
def zeros(shape, wcs=None, dtype=None):
	return enmap(np.zeros(shape, dtype=dtype), wcs, copy=False)
def ones(shape, wcs=None, dtype=None):
	return enmap(np.ones(shape, dtype=dtype), wcs, copy=False)
def full(shape, wcs, val, dtype=None):
	return enmap(np.full(shape, val, dtype=dtype), wcs, copy=False)

def posmap(shape, wcs, safe=True, corner=False):
	"""Return an enmap where each entry is the coordinate of that entry,
	such that posmap(shape,wcs)[{0,1},j,k] is the {y,x}-coordinate of
	pixel (j,k) in the map. Results are returned in radians, and
	if safe is true (default), then sharp coordinate edges will be
	avoided."""
	pix    = np.mgrid[:shape[-2],:shape[-1]]
	return ndmap(pix2sky(shape, wcs, pix, safe, corner), wcs)

def pixmap(shape, wcs=None):
	"""Return an enmap where each entry is the pixel coordinate of that entry."""
	res = np.mgrid[:shape[-2],:shape[-1]]
	return res if wcs is None else ndmap(res,wcs)

def pix2sky(shape, wcs, pix, safe=True, corner=False):
	"""Given an array of corner-based pixel coordinates [{y,x},...],
	return sky coordinates in the same ordering."""
	pix = np.asarray(pix).astype(float)
	if corner: pix -= 0.5
	pflat = pix.reshape(pix.shape[0], -1)
	coords = np.asarray(wcsutils.nobcheck(wcs).wcs_pix2world(*(tuple(pflat)[::-1]+(0,)))[::-1])*get_unit(wcs)
	coords = coords.reshape(pix.shape)
	if safe and not wcsutils.is_plain(wcs):
		coords = utils.unwind(coords)
	return coords

def sky2pix(shape, wcs, coords, safe=True, corner=False):
	"""Given an array of coordinates [{dec,ra},...], return
	pixel coordinates with the same ordering. The corner argument
	specifies whether pixel coordinates start at pixel corners
	or pixel centers. This represents a shift of half a pixel.
	If corner is False, then the integer pixel closest to a position
	is round(sky2pix(...)). Otherwise, it is floor(sky2pix(...))."""
	coords = np.asarray(coords)/get_unit(wcs)
	cflat  = coords.reshape(coords.shape[0], -1)
	# Quantities with a w prefix are in wcs ordering (ra,dec)
	wpix = np.asarray(wcsutils.nobcheck(wcs).wcs_world2pix(*tuple(cflat)[::-1]+(0,)))
	if corner: wpix += 0.5
	if safe and not wcsutils.is_plain(wcs):
		wshape = shape[-2:][::-1]
		# Put the angle cut as far away from the map as possible.
		# We do this by putting the reference point in the middle
		# of the map.
		wrefpix = np.array(wshape)/2.
		if corner: wrefpix += 0.5
		for i in range(len(wpix)):
			wn = np.abs(360./wcs.wcs.cdelt[i])
			if safe == 1:
				wpix[i] = utils.rewind(wpix[i], wrefpix[i], wn)
			else:
				wpix[i] = utils.unwind(wpix[i], period=wn, ref=wrefpix[i])
	return wpix[::-1].reshape(coords.shape)

def skybox2pixbox(shape, wcs, skybox, npoint=10, corner=False, include_direction=False):
	"""Given a coordinate box [{from,to},{dec,ra}], compute a
	corresponding pixel box [{from,to},{y,x}]. We avoiding
	wrapping issues by evaluating a number of subpoints."""
	coords = np.array([
		np.linspace(skybox[0,0],skybox[1,0],num=npoint,endpoint=True),
		np.linspace(skybox[0,1],skybox[1,1],num=npoint,endpoint=True)])
	pix = sky2pix(shape, wcs, coords, corner=corner, safe=2)
	dir = np.sign(pix[:,1]-pix[:,0])
	res = pix[:,[0,-1]].T
	if include_direction: res = np.concatenate([res,dir[None]],0)
	return res

def project(map, shape, wcs, order=3, mode="constant", cval=0.0, force=False, prefilter=True, mask_nan=True, safe=True):
	"""Project the map into a new map given by the specified
	shape and wcs, interpolating as necessary. Handles nan
	regions in the map by masking them before interpolating.
	This uses local interpolation, and will lose information
	when downgrading compared to averaging down."""
	map  = map.copy()
	# Skip expensive operation is map is compatible
	if not force:
		if wcsutils.equal(map.wcs, wcs) and tuple(shape[-2:]) == tuple(shape[-2:]):
			return map
		elif wcsutils.is_compatible(map.wcs, wcs) and mode == "constant":
			return extract(map, shape, wcs, cval=cval)
	pix  = map.sky2pix(posmap(shape, wcs), safe=safe)
	pmap = utils.interpol(map, pix, order=order, mode=mode, cval=cval, prefilter=prefilter, mask_nan=mask_nan)
	return ndmap(pmap, wcs)

def extract(map, shape, wcs, omap=None, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None):
	"""Like project, but only works for pixel-compatible wcs. Much
	faster because it simply copies over pixels.

	Can be used in co-adding by specifying an output map and a combining
	operation. The deafult operation overwrites the output. Use
	np.ndarray.__iadd__ to get a copy-less += operation. Not that
	areas outside are not assumed to be zero if an omap is specified -
	instead those areas will simply not be operated on.

	The optional iwcs argument is there to support input maps that are
	numpy-like but can't be made into actual enmaps. The main example of
	this is a fits hdu object, which can be sliced like an array to avoid
	reading more into memory than necessary.
	"""
	# First check that our wcs is compatible
	if iwcs is None: iwcs = map.wcs
	assert wcsutils.is_compatible(iwcs, wcs), "Incompatible wcs in enmap.extract: %s vs. %s" % (str(iwcs), str(wcs))
	# Find the bounding box of the output in terms of input pixels.
	# This is simple because our wcses are compatible, so they
	# can only differ by a simple pixel offset. Here pixoff is
	# pos_input - pos_output. This is equivalent to the coordinates of
	pixoff = utils.nint((iwcs.wcs.crpix-wcs.wcs.crpix) - (iwcs.wcs.crval-wcs.wcs.crval)/iwcs.wcs.cdelt)[::-1]
	pixbox = np.array([pixoff,pixoff+np.array(shape[-2:])])
	return extract_pixbox(map, pixbox, omap=omap, wrap=wrap, op=op, cval=cval, iwcs=iwcs)

def extract_pixbox(map, pixbox, omap=None, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None):
	"""This function extracts a rectangular area from an enmap based on the
	given pixbox[{from,to,[stride]},{y,x}]. The difference between this function
	and plain slicing of the enmap is that this one supports wrapping around the
	sky. This is necessary to make things like fast thumbnail or tile extraction
	at the edge of a (horizontally) fullsky map work."""
	if iwcs is None: iwcs = map.wcs
	pixbox = np.asarray(pixbox)
	oshape, owcs = slice_geometry(map.shape, iwcs, (slice(*pixbox[:,-2]),slice(*pixbox[:,-1])), nowrap=True)
	if omap is None:
		omap = full(map.shape[:-2]+tuple(oshape[-2:]), owcs, cval, map.dtype)
	nphi = utils.nint(360/np.abs(iwcs.wcs.cdelt[0]))
	# If our map is wider than the wrapping length, assume we're a lower-spin field
	nphi *= (nphi+map.shape[-1]-1)//nphi
	if wrap is "auto": wrap = [0,nphi]
	else: wrap = np.zeros(2,int)+wrap
	for ibox, obox in utils.sbox_wrap(pixbox.T, wrap=wrap, cap=map.shape[-2:]):
		islice = utils.sbox2slice(ibox)
		oslice = utils.sbox2slice(obox)
		omap[oslice] = op(omap[oslice], map[islice])
	return omap

def at(map, pos, order=3, mode="constant", cval=0.0, unit="coord", prefilter=True, mask_nan=True, safe=True):
	if unit != "pix": pos = sky2pix(map.shape, map.wcs, pos, safe=safe)
	return utils.interpol(map, pos, order=order, mode=mode, cval=cval, prefilter=prefilter, mask_nan=mask_nan)

def argmax(map, unit="coord"):
	"""Return the coordinates of the maximum value in the specified map.
	If map has multiple components, the maximum value for each is returned
	separately, with the last axis being the position. If unit is "pix",
	the position will be given in pixels. Otherwise it will be in physical
	coordinates."""
	return _arghelper(map, np.argmax, unit)
def argmin(map, unit="coord"):
	"""Return the coordinates of the minimum value in the specified map.
	See argmax for details."""
	return _arghelper(map, np.argmin, unit)
def _arghelper(map, func, unit):
	res = func(map.reshape(-1,map.npix),-1)
	res = np.array([np.unravel_index(r, map.shape[-2:]) for r in res])
	res = res.reshape(map.shape[:-2]+(2,))
	if unit == "coord": res = pix2sky(map.shape, map.wcs, res.T).T
	return res

def rand_map(shape, wcs, cov, scalar=False, seed=None, pixel_units=False, iau=False, spin=[0,2]):
	"""Generate a standard flat-sky pixel-space CMB map in TQU convention based on
	the provided power spectrum. If cov.ndim is 4, 2D power is assumed else 1D
	power is assumed. If pixel_units is True, the 2D power spectra is assumed
	to be in pixel units, not in steradians."""
	if seed is not None: np.random.seed(seed)
	kmap = rand_gauss_iso_harm(shape, wcs, cov, pixel_units)
	if scalar:
		return ifft(kmap).real
	else:
		return harm2map(kmap, iau=iau, spin=spin)

def rand_gauss(shape, wcs, dtype=None):
	"""Generate a map with random gaussian noise in pixel space."""
	return ndmap(np.random.standard_normal(shape), wcs).astype(dtype,copy=False)

def rand_gauss_harm(shape, wcs):
	"""Mostly equivalent to np.fft.fft2(np.random.standard_normal(shape)),
	but avoids the fft by generating the numbers directly in frequency
	domain. Does not enforce the symmetry requried for a real map. If box is
	passed, the result will be an enmap."""
	return ndmap(np.random.standard_normal(shape)+1j*np.random.standard_normal(shape),wcs)

def rand_gauss_iso_harm(shape, wcs, cov, pixel_units=False):
	"""Generates a random map with component covariance
	cov in harmonic space, where cov is a (comp,comp,l) array or a
	(comp,comp,Ny,Nx) array. Despite the name, the map doesn't need
	to be isotropic since 2D power spectra are allowed.

	If cov.ndim is 4, cov is assumed to be an array of 2D power spectra.
	else cov is assumed to be an array of 1D power spectra.
	If pixel_units is True, the 2D power spectra is assumed to be in pixel units,
	not in steradians."""
	if cov.ndim==4:
		if not(pixel_units): cov = cov * np.prod(shape[-2:])/area(shape,wcs )
		covsqrt = multi_pow(cov, 0.5)
	else:
		covsqrt = spec2flat(shape, wcs, cov, 0.5, mode="constant")
	data = map_mul(covsqrt, rand_gauss_harm(shape, wcs))
	return ndmap(data, wcs)

def extent(shape, wcs, method="default", nsub=None, signed=False):
	if method == "default": method = extent_model[-1]
	if method == "intermediate":
		return extent_intermediate(shape, wcs, signed=signed)
	elif method == "subgrid":
		return extent_subgrid(shape, wcs, nsub=nsub, signed=signed)
	else:
		raise ValueError("Unrecognized extent method '%s'" % method)

def extent_intermediate(shape, wcs, signed=False):
	"""Estimate the flat-sky extent of the map as the WCS
	intermediate coordinate extent."""
	res = wcs.wcs.cdelt[::-1]*shape[-2:]*get_unit(wcs)
	if not signed: res = np.abs(res)
	return res

# Approximations to physical box size and area are needed
# for transforming to l-space. We can do this by dividing
# our map into a set of rectangles and computing the
# coordinates of their corners. The rectangles are assumed
# to be small, so cos(dec) is constant across them, letting
# us rescale RA by cos(dec) inside each. We also assume each
# rectangle to be .. a rectangle (:D), so area is given by
# two side lengths.
# The total length in each direction could be computed by
# 1. Average of top and bottom length
# 2. Mean of all row lengths
# 3. Area-weighted mean of row lengths
# 4. Some sort of compromise that makes length*height = area.
# To construct the coarser system, slicing won't do, as it
# shaves off some of our area. Instead, we must modify
# cdelt to match our new pixels: cdelt /= nnew/nold
def extent_subgrid(shape, wcs, nsub=None, safe=True, signed=False):
	"""Returns an estimate of the "physical" extent of the
	patch given by shape and wcs as [height,width] in
	radians. That is, if the patch were on a sphere with
	radius 1 m, then this function returns approximately how many meters
	tall and wide the patch is. These are defined such that
	their product equals the physical area of the patch.
	Obs: Has trouble with areas near poles."""
	if nsub is None: nsub = 16
	# Create a new wcs with (nsub,nsub) pixels
	wcs = wcs.deepcopy()
	step = (np.asfarray(shape[-2:])/nsub)[::-1]
	wcs.wcs.crpix -= 0.5
	wcs.wcs.cdelt *= step
	wcs.wcs.crpix /= step
	wcs.wcs.crpix += 0.5
	# Get position of all the corners, including the far ones
	pos = posmap([nsub+1,nsub+1], wcs, corner=True, safe=safe)
	# Apply az scaling
	scale = np.zeros([2,nsub,nsub])
	scale[1] = np.cos(0.5*(pos[0,1:,:-1]+pos[0,:-1,:-1]))
	scale[0] = 1
	ly = np.sum(((pos[:,1:,:-1]-pos[:,:-1,:-1])*scale)**2,0)**0.5
	lx = np.sum(((pos[:,:-1,1:]-pos[:,:-1,:-1])*scale)**2,0)**0.5
	# Replace invalid areas with mean
	bad = ~np.isfinite(ly) | ~np.isfinite(lx)
	ly[bad] = np.mean(ly[~bad])
	lx[bad] = np.mean(lx[~bad])
	areas = ly*lx
	# Compute approximate overall lengths
	Ay, Ax = np.sum(areas,0), np.sum(areas,1)
	Ly = np.sum(np.sum(ly,0)*Ay)/np.sum(Ay)
	Lx = np.sum(np.sum(lx,1)*Ax)/np.sum(Ax)
	res= np.array([Ly,Lx])
	if signed: res *= np.sign(wcs.wcs.cdelt[::-1])
	return res

def area(shape, wcs, nsub=0x10):
	"""Returns the area of a patch with the given shape
	and wcs, in steradians."""
	return np.prod(extent(shape, wcs, nsub=nsub))

def pixsize(shape, wcs):
	"""Returns the area of a single pixel, in steradians."""
	return area(shape, wcs)/np.product(shape[-2:])

def pixshape(shape, wcs, signed=False):
	"""Returns the height and width of a single pixel, in radians."""
	return extent(shape, wcs, signed=signed)/shape[-2:]

def pixsizemap(shape, wcs):
	"""Returns the physical area of each pixel in the map in steradians.
	Heavy for big maps."""
	# First get the coordinates of all the pixel corners
	pix  = np.mgrid[:shape[-2]+1,:shape[-1]+1]
	with utils.nowarn():
		y, x = pix2sky(shape, wcs, pix, safe=True, corner=True)
	del pix
	dy   = y[1:,1:]-y[:-1,:-1]
	dx   = x[1:,1:]-x[:-1,:-1]
	cy   = np.cos(y)
	dx  *= 0.5*(cy[1:,1:]+cy[:-1,:-1])
	del y, x, cy
	area = dy*dx
	del dy, dx
	area = np.abs(area)
	# Due to wcs fragility, we may have some nans at wraparound points.
	# Fill these with the mean non-nan value. Since most maps will be cylindrical,
	# it makes sense to do this by row
	for a in area:
		bad  = ~np.isfinite(a)
		a[bad] = np.mean(a[~bad])
	return ndmap(area, wcs)

def lmap(shape, wcs, oversample=1):
	"""Return a map of all the wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	ly, lx = laxes(shape, wcs, oversample=oversample)
	data = np.empty((2,ly.size,lx.size))
	data[0] = ly[:,None]
	data[1] = lx[None,:]
	return ndmap(data, wcs)

def modlmap(shape, wcs, oversample=1):
	"""Return a map of all the abs wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	slmap = lmap(shape,wcs,oversample=oversample)
	return np.sum(slmap**2,0)**0.5

def center(shape,wcs):
	cpix = (np.array(shape[-2:])-1)/2.
	return pix2sky(shape,wcs,cpix)

def modrmap(shape, wcs, ref="center", safe=True, corner=False):
	"""Return an enmap where each entry is the distance from center
	of that entry. Results are returned in radians, and if safe is true
	(default), then sharp coordinate edges will be avoided."""
	slmap = posmap(shape,wcs,safe=safe,corner=corner)
	if isinstance(ref,basestring):
		if ref=="center": ref = center(shape,wcs)
		else:             raise ValueError
	ref = np.array(ref)[:,None,None]
	return ndmap(utils.angdist(slmap,ref,zenith=False),wcs)


def laxes(shape, wcs, oversample=1):
	oversample = int(oversample)
	step = extent(shape, wcs, signed=True)/shape[-2:]
	ly = np.fft.fftfreq(shape[-2]*oversample, step[0])*2*np.pi
	lx = np.fft.fftfreq(shape[-1]*oversample, step[1])*2*np.pi
	if oversample > 1:
		# When oversampling, we want even coverage of fourier-space
		# pixels. Because the pixel value indicates the *center* l
		# of that pixel, we must shift ls when oversampling.
		# E.g. [0,100,200,...] oversample 2 => [-25,25,75,125,175,...],
		# not [0,50,100,150,200,...].
		# And  [0,100,200,...] os 3 => [-33,0,33,66,100,133,...]
		# In general [0,a,2a,3a,...] os n => a*(-1+(2*i+1)/n)/2
		# Since fftfreq generates a*i, the difference is a/2*(-1+1/n)
		def shift(l,a,n): return l+a/2*(-1+1./n)
		ly = shift(ly,ly[oversample],oversample)
		lx = shift(lx,lx[oversample],oversample)
	return ly, lx

def lrmap(shape, wcs, oversample=1):
	"""Return a map of all the wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	return lmap(shape, wcs, oversample=oversample)[...,:shape[-1]//2+1]

def fft(emap, omap=None, nthread=0, normalize=True):
	"""Performs the 2d FFT of the enmap pixels, returning a complex enmap."""
	res = samewcs(enfft.fft(emap,omap,axes=[-2,-1],nthread=nthread), emap)
	if normalize: res /= np.prod(emap.shape[-2:])**0.5
	return res
def ifft(emap, omap=None, nthread=0, normalize=True):
	"""Performs the 2d iFFT of the complex enmap given, and returns a pixel-space enmap."""
	res = samewcs(enfft.ifft(emap,omap,axes=[-2,-1],nthread=nthread, normalize=False), emap)
	if normalize: res /= np.prod(emap.shape[-2:])**0.5
	return res

# These are shortcuts for transforming from T,Q,U real-space maps to
# T,E,B hamonic maps. They are not the most efficient way of doing this.
# It would be better to precompute the rotation matrix and buffers, and
# use real transforms.
def map2harm(emap, nthread=0, normalize=True, iau=False, spin=[0,2]):
	"""Performs the 2d FFT of the enmap pixels, returning a complex enmap."""
	emap = samewcs(fft(emap,nthread=nthread,normalize=normalize), emap)
	if emap.ndim > 2:
		rot, s0 = None, None
		for s, i1, i2 in spin_helper(spin, emap.shape[-3]):
			if s == 0:  continue
			if s != s0: s0, rot = s, queb_rotmat(emap.lmap(), iau=iau, spin=s)
			emap[...,i1:i2,:,:] = map_mul(rot, emap[...,i1:i2,:,:])
	return emap
def harm2map(emap, nthread=0, normalize=True, iau=False, spin=[0,2]):
	if emap.ndim > 2:
		rot, s0 = None, None
		for s, i1, i2 in spin_helper(spin, emap.shape[-3]):
			if s == 0:  continue
			if s != s0: s0, rot = s, queb_rotmat(emap.lmap(), iau=iau, spin=s, inverse=True)
			emap[...,i1:i2,:,:] = map_mul(rot, emap[...,i1:i2,:,:])
	return samewcs(ifft(emap,nthread=nthread,normalize=normalize), emap).real

def queb_rotmat(lmap, inverse=False, iau=False, spin=2):
	# atan2(x,y) instead of (y,x) because Qr points in the
	# tangential direction, not radial. This matches flipperpol too.
	# This corresponds to the Healpix convention. To get IAU,
	# flip the sign of a.
	sgn = -1 if iau else 1
	a    = sgn*spin*np.arctan2(-lmap[1], lmap[0])
	c, s = np.cos(a), np.sin(a)
	if inverse: s = -s
	return samewcs(np.array([[c,-s],[s,c]]),lmap)

def rotate_pol(emap, angle, comps=[-2,-1]):
	c, s = np.cos(2*angle), np.sin(2*angle)
	res = emap.copy()
	res[...,comps[0],:,:] = c*emap[...,comps[0],:,:] - s*emap[...,comps[1],:,:]
	res[...,comps[1],:,:] = s*emap[...,comps[0],:,:] + c*emap[...,comps[1],:,:]
	return res

def map_mul(mat, vec):
	"""Elementwise matrix multiplication mat*vec. Result will have
	the same shape as vec. Multiplication happens along the first indices.
	This function is buggy when mat is not square (in the multiplication
	dimensions). This is due to the reshape at the end. I should figure out
	what code depends on that, and decide what I really want this function
	to do."""
	oshape= vec.shape
	if len(oshape) == 2: oshape = (1,)+oshape
	tvec = np.reshape(vec, oshape)
	# It is a bit clunky to get einsum to handle arbitrary numbers of dimensions.
	vpre  = "".join([chr(ord('a')+i) for i in range(len(oshape)-3)])
	mpre  = vpre[vec.ndim-(mat.ndim-1):]
	data  = np.reshape(np.einsum("%sxyzw,%syzw->%sxzw" % (mpre,vpre,vpre), mat, tvec), vec.shape)
	return samewcs(data, mat, vec)

def smooth_gauss(emap, sigma):
	"""Smooth the map given as the first argument with a gaussian beam
	with the given standard deviation in radians."""
	if sigma == 0: return emap.copy()
	f  = map2harm(emap)
	l2 = np.sum(emap.lmap()**2,0)
	f *= np.exp(-0.5*l2*sigma**2)
	return harm2map(f)

def calc_window(shape):
	"""Compute fourier-space window function. Like the other fourier-based
	functions in this module, equi-spaced pixels are assumed. Since the
	window function is separable, it is returned as an x and y part,
	such that window = wy[:,None]*wx[None,:]."""
	wy = np.sinc(np.fft.fftfreq(shape[-2]))
	wx = np.sinc(np.fft.fftfreq(shape[-1]))
	return wy, wx

def apply_window(emap, pow=1.0):
	"""Apply the pixel window function to the specified power to the map,
	returning a modified copy. Use pow=-1 to unapply the pixel window."""
	wy, wx = calc_window(emap.shape)
	return ifft(fft(emap) * wy[:,None]**pow * wx[None,:]**pow).real

def samewcs(arr, *args):
	"""Returns arr with the same wcs information as the first enmap among args.
	If no mathces are found, arr is returned as is."""
	for m in args:
		try: return ndmap(arr, m.wcs)
		except AttributeError: pass
	return arr

# Idea: Make geometry a class with .shape and .wcs members.
# Make a function that takes (foo,bar) and returns a geometry,
# there (foo,bar) can either be (shape,wcs) or (geometry,None).
# Use that to make everything that currently accepts shape, wcs
# transparently accept geometry. This will free us from having
# to drag around a shape, wcs pair all the time.
def geometry(pos, res=None, shape=None, proj="car", deg=False, pre=(), force=False, ref=None, **kwargs):
	"""Consruct a shape,wcs pair suitable for initializing enmaps.
	pos can be either a {dec,ra} center position or a [{from,to},{dec,ra}]
	bounding box. At least one of res or shape must be specified.
	If res is specified, it must either be a number, in
	which the same resolution is used in each direction,
	or {dec,ra}. If shape is specified, it must be at least [2]. All angles
	are given in radians.

	The projection type is chosen with the proj argument. The default is "car",
	corresponding to the equirectangular plate carree projection. Other valid
	projections are "cea", "zea", "gnom", etc. See wcsutils for details.

	By default the geometry is tweaked so that a standard position, typically
	ra=0,dec=0, would be at an integer logical pixel position (even if that position is
	outside the physical map). This makes it easier to generate maps that are compatible
	up to an integer pixel offset, as well as maps that are compatible with the predefined
	spherical harmonics transform ring weights. The cost of this tweaking is that the
	resulting bounding box can differ by a fraction of a pixel from the one requested.
	To force the geometry to exactly match the bounding box provided you can pass force=True.
	It is also possible to manually choose the reference pixel via the ref argument, which
	must be a dec,ra coordinate pair."""
	# We use radians by default, while wcslib uses degrees, so need to rescale.
	# The exception is when we are using a plain, non-spherical wcs, in which case
	# both are unitless. So undo the scaling in this case.
	scale = 1 if deg else 1/utils.degree
	if proj == "plain": scale *= utils.degree
	pos = np.asarray(pos)*scale
	if res is not None: res = np.asarray(res)*scale
	# Apply a standard reference points unless one is manually specified, or we
	# want to force the bounding box to exactly match the input.
	if ref is None and not force: ref = "standard"
	wcs = wcsutils.build(pos, res, shape, rowmajor=True, system=proj, ref=ref, **kwargs)
	if shape is None:
		# Infer shape. WCS does not allow us to wrap around the
		# sky, so shape mustn't be large enough to make that happen.
		# Our relevant pixel coordinates go from (-0.5,-0.5) to
		# shape-(0.5,0.5). We assume that wcs.build has already
		# assured the former. Our job is to find shape that puts
		# the top edge close to the requested value, while still
		# being valid. If we always round down, we should be safe:
		nearedge = wcsutils.nobcheck(wcs).wcs_world2pix(pos[0:1,::-1],0)[0,::-1]
		faredge  = wcsutils.nobcheck(wcs).wcs_world2pix(pos[1:2,::-1],0)[0,::-1]
		shape = tuple(np.round(faredge-nearedge).astype(int))
	return pre+tuple(shape), wcs

def fullsky_geometry(res=None, shape=None, dims=(), proj="car"):
	"""Build an enmap covering the full sky, with the outermost pixel centers
	at the poles and wrap-around points. Assumes a CAR (clenshaw curtis variant)
	projection for now."""
	assert proj == "car", "Only CAR fullsky geometry implemented"
	if shape is None:
		res   = np.zeros(2)+res
		shape = ([1*np.pi,2*np.pi]/res+0.5).astype(int)
		shape[0] += 1
	ny,nx = shape
	ny   -= 1
	wcs   = wcsutils.WCS(naxis=2)
	wcs.wcs.crval = [0,0]
	wcs.wcs.cdelt = [-360./nx,180./ny]
	wcs.wcs.crpix = [nx/2.+1,ny/2.+1]
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	return dims+(ny+1,nx+0), wcs

def create_wcs(shape, box=None, proj="cea"):
	if box is None:
		box = np.array([[-1,-1],[1,1]])*0.5*10
		if proj != "plain": box *= utils.degree
	return wcsutils.build(box, shape=shape, rowmajor=True, system=proj)

def spec2flat(shape, wcs, cov, exp=1.0, mode="constant", oversample=1, smooth="auto"):
	"""Given a (ncomp,ncomp,l) power spectrum, expand it to harmonic map space,
	returning (ncomp,ncomp,y,x). This involves a rescaling which converts from
	power in terms of multipoles, to power in terms of 2d frequency.
	The optional exp argument controls the exponent of the rescaling factor.
	To use this with the inverse power spectrum, pass exp=-1, for example.
	If apply_exp is True, the power spectrum will be taken to the exp'th
	power. Otherwise, it is assumed that this has already been done, and
	the exp argument only controls the normalization of the result.

	It is irritating that this function needs to know what kind of matrix
	it is expanding, but I can't see a way to avoid it. Changing the
	units of harmonic space is not sufficient, as the following demonstrates:
	  m = harm2map(map_mul(spec2flat(s, b, multi_pow(ps, 0.5), 0.5), map2harm(rand_gauss(s,b))))
	The map m is independent of the units of harmonic space, and will be wrong unless
	the spectrum is properly scaled. Since this scaling depends on the shape of
	the map, this is the appropriate place to do so, ugly as it is."""
	oshape= tuple(shape)
	if len(oshape) == 2: oshape = (1,)+oshape
	ls = np.sum(lmap(oshape, wcs, oversample=oversample)**2,0)**0.5
	if smooth == "auto":
		# Determine appropriate fourier-scale smoothing based on 2d fourer
		# space resolution. We wish to smooth by about this width to approximate
		# averaging over sub-grid modes
		smooth = 0.5*(ls[1,0]+ls[0,1])
		smooth /= 3.41 # 3.41 is an empirical factor
	if smooth > 0:
		cov = smooth_spectrum(cov, kernel="gauss", weight="mode", width=smooth)
	# Translate from steradians to pixels
	cov = cov * np.prod(shape[-2:])/area(shape,wcs)
	if exp != 1.0: cov = multi_pow(cov, exp)
	cov[~np.isfinite(cov)] = 0
	cov   = cov[:oshape[-3],:oshape[-3]]
	# Use order 1 because we will perform very short interpolation, and to avoid negative
	# values in spectra that must be positive (and it's faster)
	res = ndmap(utils.interpol(cov, np.reshape(ls,(1,)+ls.shape),mode=mode, mask_nan=False, order=1),wcs)
	res = downgrade(res, oversample)
	return res

def spec2flat_corr(shape, wcs, cov, exp=1.0, mode="constant"):
	oshape= tuple(shape)
	if len(oshape) == 2: oshape = (1,)+oshape
	if exp != 1.0: cov = multi_pow(cov, exp)
	cov[~np.isfinite(cov)] = 0
	cov = cov[:oshape[-3],:oshape[-3]]
	# Convert power spectrum to correlation
	ext  = extent(shape,wcs)
	rmax = np.sum(ext**2)**0.5
	res  = np.max(ext/shape[-2:])
	nr   = rmax/res
	r    = np.arange(nr)*rmax/nr
	corrfun = powspec.spec2corr(cov, r)
	# Interpolate it 2d. First get the pixel positions
	# (remember to move to the corner because this is
	# a correlation function)
	dpos = posmap(shape, wcs)
	dpos -= dpos[:,None,None,dpos.shape[-2]//2,dpos.shape[-1]//2]
	ipos = np.arccos(np.cos(dpos[0])*np.cos(dpos[1]))*nr/rmax
	corr2d = utils.interpol(corrfun, ipos.reshape((-1,)+ipos.shape), mode=mode, mask_nan=False, order=1)
	corr2d = np.roll(corr2d, -corr2d.shape[-2]//2, -2)
	corr2d = np.roll(corr2d, -corr2d.shape[-1]//2, -1)
	corr2d = ndmap(corr2d, wcs)
	return fft(corr2d).real * np.product(shape[-2:])**0.5

def smooth_spectrum(ps, kernel="gauss", weight="mode", width=1.0):
	"""Smooth the spectrum ps with the given kernel, using the given weighting."""
	ps = np.asanyarray(ps)
	pflat = ps.reshape(-1,ps.shape[-1])
	nspec,nl = pflat.shape
	# Set up the kernel array
	K = np.zeros((nspec,nl))
	l = np.arange(nl)
	if isinstance(kernel, basestring):
		if kernel == "gauss":
			K[:] = np.exp(-0.5*(l/width)**2)
		elif kernel == "step":
			K[:,:int(width)] = 1
		else:
			raise ValueError("Unknown kernel type %s in smooth_spectrum" % kernel)
	else:
		tmp = np.atleast_2d(kernel)
		K[:,:tmp.shape[-1]] = tmp[:,:K.shape[-1]]
	# Set up the weighting scheme
	W = np.zeros((nspec,nl))
	if isinstance(weight, basestring):
		if weight == "mode":
			W[:] = l[None,:]**2
		elif weight == "uniform":
			W[:] = 1
		else:
			raise ValueError("Unknown weighting scheme %s in smooth_spectrum" % weight)
	else:
		tmp = np.atleast_2d(weight)
		assert tmp.shape[-1] == W.shape[-1], "Spectrum weight must have the same length as spectrum"
	pWK = _convolute_sym(pflat*W, K)
	WK  = _convolute_sym(W, K)
	res = pWK/WK
	return res.reshape(ps.shape)

def _convolute_sym(a,b):
	sa = np.concatenate([a,a[:,-2:0:-1]],-1)
	sb = np.concatenate([b,b[:,-2:0:-1]],-1)
	fa = enfft.rfft(sa)
	fb = enfft.rfft(sb)
	sa = enfft.ifft(fa*fb,sa,normalize=True)
	return sa[:,:a.shape[-1]]

def multi_pow(mat, exp, axes=[0,1]):
	"""Raise each sub-matrix of mat (ncomp,ncomp,...) to
	the given exponent in eigen-space."""
	return samewcs(utils.eigpow(mat, exp, axes=axes), mat)

def downgrade(emap, factor):
	"""Returns enmap "emap" downgraded by the given integer factor
	(may be a list for each direction, or just a number) by averaging
	inside pixels."""
	fact = np.full(2, 1, dtype=int)
	fact[:] = factor
	tshape = emap.shape[-2:]//fact*fact
	res = np.mean(np.reshape(emap[...,:tshape[0],:tshape[1]],emap.shape[:-2]+(tshape[0]//fact[0],fact[0],tshape[1]//fact[1],fact[1])),(-3,-1))
	try: return ndmap(res, emap[...,::fact[0],::fact[1]].wcs)
	except AttributeError: return res

def upgrade(emap, factor):
	"""Upgrade emap to a larger size using nearest neighbor interpolation,
	returning the result. More advanced interpolation can be had using
	enmap.interpolate."""
	fact = np.full(2,1).astype(int)
	fact[:] = factor
	res = np.tile(emap.copy().reshape(emap.shape[:-2]+(emap.shape[-2],1,emap.shape[-1],1)),(1,fact[0],1,fact[1]))
	res = res.reshape(res.shape[:-4]+(np.product(res.shape[-4:-2]),np.product(res.shape[-2:])))
	# Correct the WCS information
	for j in range(2):
		res.wcs.wcs.crpix[j] -= 0.5
		res.wcs.wcs.crpix[j] *= fact[1-j]
		res.wcs.wcs.cdelt[j] /= fact[1-j]
		res.wcs.wcs.crpix[j] += 0.5
	return res

def pad(emap, pix, return_slice=False, wrap=False):
	"""Pad enmap "emap", creating a larger map with zeros filled in on the sides.
	How much to pad is controlled via pix. If pix is a scalar, it specifies the number
	of pixels to add on all sides. If it is 1d, it specifies the number of pixels to add
	at each end for each axis. If it is 2d, the number of pixels to add at each end
	of an axis can be specified individually."""
	pix = np.asarray(pix,dtype=int)
	if pix.ndim == 0:
		pix = np.array([[pix,pix],[pix,pix]])
	elif pix.ndim == 1:
		pix = np.array([pix,pix])
	# Exdend the wcs in each direction.
	w = emap.wcs.deepcopy()
	w.wcs.crpix += pix[0,::-1]
	# Construct a slice between the new and old map
	res = zeros(emap.shape[:-2]+tuple([s+sum(p) for s,p in zip(emap.shape[-2:],pix.T)]),wcs=w, dtype=emap.dtype)
	mslice = (Ellipsis,slice(pix[0,0],res.shape[-2]-pix[1,0]),slice(pix[0,1],res.shape[-1]-pix[1,1]))
	res[mslice] = emap
	if wrap:
		res[...,:pix[0,0],:]  = res[...,-pix[0,0]-pix[1,0]:-pix[1,0],:]
		res[...,-pix[1,0]:,:] = res[...,pix[0,0]:pix[0,0]+pix[1,0],:]
		res[...,:,:pix[0,1]]  = res[...,:,-pix[0,1]-pix[1,1]:-pix[1,1]]
		res[...,:,-pix[1,1]:] = res[...,:,pix[0,1]:pix[0,1]+pix[1,1]]
	return (res,mslice) if return_slice else res

def find_blank_edges(m, value="auto"):
	"""Returns blanks[{front,back},{y,x}], the size of the blank area
	at the beginning and end of each axis of the map, where the argument
	"value" determines which value is considered blank. Can be a float value,
	or the strings "auto" or "none". Auto will choose the value that maximizes
	the edge area considered blank. None will result in nothing being consideered blank."""
	if value is "auto":
		# Find the median value along each edge
		medians = [np.median(m[...,:,i],-1) for i in [0,-1]] + [np.median(m[...,i,:],-1) for i in [0,-1]]
		bs = [find_blank_edges(m, med) for med in medians]
		nb = [np.product(np.sum(b,0)) for b in bs]
		blanks = bs[np.argmax(nb)]
		return blanks
	elif value is "none":
		# Don't use any values for cropping, so no cropping is done
		return np.zeros([2,2],dtype=int)
	else:
		value   = np.asarray(value)
		# Find which rows and cols consist entirely of the given value
		hitmask = np.all(np.isclose(m.T, value.T, equal_nan=True, rtol=1e-6, atol=0).T,axis=tuple(range(m.ndim-2)))
		hitrows = np.all(hitmask,1)
		hitcols = np.all(hitmask,0)
		# Find the first and last row and col which aren't all the value
		blanks  = np.array([
			np.where(~hitrows)[0][[0,-1]],
			np.where(~hitcols)[0][[0,-1]]]
			).T
		blanks[1] = m.shape[-2:]-blanks[1]-1
		return blanks

def autocrop(m, method="plain", value="auto", margin=0, factors=None, return_info=False):
	"""Adjust the size of m to be more fft-friendly. If possible,
	blank areas at the edge of the map are cropped to bring us to a nice
	length. If there there aren't enough blank areas, the map is padded
	instead. If value="none" no values are considered blank, so no cropping
	will happen. This can be used to autopad for fourier-friendliness."""
	blanks  = find_blank_edges(m, value=value)
	nblank  = np.sum(blanks,0)
	# Find the first good sizes larger than the unblank lengths
	minshape  = m.shape[-2:]-nblank+margin
	if method == "plain":
		goodshape = minshape
	elif method == "fft":
		goodshape = np.array([enfft.fft_len(l, direction="above", factors=None) for l in minshape])
	else:
		raise ValueError("Unknown autocrop method %s!" % method)
	# Pad if necessary
	adiff   = np.maximum(0,goodshape-m.shape[-2:])
	padding = [[0,0],[0,0]]
	if any(adiff>0):
		padding = [adiff,[0,0]]
		m = pad(m, padding)
		blanks[0] += adiff
		nblank = np.sum(blanks,0)
	# Then crop to goodshape
	tocrop = m.shape[-2:]-goodshape
	lower  = np.minimum(tocrop,blanks[0])
	upper  = tocrop-lower
	s      = (Ellipsis,slice(lower[0],m.shape[-2]-upper[0]),slice(lower[1],m.shape[-1]-upper[1]))
	class PadcropInfo:
		slice   = s
		pad     = padding
	if return_info:
		return m[s], PadcropInfo
	else:
		return m[s]

def padcrop(m, info):
	return pad(m, info.pad)[info.slice]

def grad(m):
	"""Returns the gradient of the map m as [2,...]."""
	return ifft(fft(m)*_widen(m.lmap(),m.ndim+1)*1j).real

def grad_pix(m):
	"""The gradient of map m expressed in units of pixels.
	Not the same as the gradient of m with resepect to pixels.
	Useful for avoiding sky2pix-calls for e.g. lensing,
	and removes the complication of axes that increase in
	nonstandard directions."""
	return grad(m)*(m.shape[-2:]/m.extent(signed=True))[(slice(None),)+(None,)*m.ndim]

def div(m):
	"""Returns the divergence of the map m[2,...] as [...]."""
	return ifft(np.sum(fft(m)*_widen(m.lmap(),m.ndim)*1j,0)).real

def _widen(map,n):
	"""Helper for gard and div. Adds degenerate axes between the first
	and the last two to give the map a total dimensionality of n."""
	return map[(slice(None),) + (None,)*(n-3) + (slice(None),slice(None))]

def apod(m, width, profile="cos", fill="zero"):
	width = np.minimum(np.zeros(2)+width,m.shape[-2:]).astype(np.int32)
	if profile == "cos":
		a = [0.5*(1-np.cos(np.linspace(0,np.pi,w))) for w in width]
	else:
		raise ValueError("Unknown apodization profile %s" % profile)
	res = m.copy()
	if fill == "mean":
		offset = np.asarray(np.mean(res,(-2,-1)))[...,None,None]
		res -= offset
	if width[0] > 0:
		res[...,:width[0],:] *= a[0][:,None]
		res[...,-width[0]:,:] *= a[0][::-1,None]
	if width[1] > 0:
		res[...,:,:width[1]] *= a[1][None,:]
		res[...,:,-width[1]:]  *= a[1][None,::-1]
	if fill == "mean":
		res += offset
	return res

def radial_average(map, center=[0,0], step=1.0):
	"""Produce a radial average of the given map that's centered on zero"""
	center = np.asarray(center)
	pos  = map.posmap()-center[:,None,None]
	rads = np.sum(pos**2,0)**0.5
	# Our resolution should be step times the highest resolution direction.
	res = np.min(map.extent()/map.shape[-2:])*step
	n   = int(np.max(rads/res))
	orads = np.arange(n)*res
	rinds = (rads/res).reshape(-1).astype(int)
	# Ok, rebin the map. We use this using bincount, which can be a bit slow
	mflat = map.reshape((-1,)+map.shape[-2:])
	mout = np.zeros((len(mflat),n))
	for i, m in enumerate(mflat):
		mout[i] = (np.bincount(rinds, weights=m.reshape(-1))/np.bincount(rinds))[:n]
	mout = mout.reshape(map.shape[:-2]+mout.shape[1:])
	return mout, orads

def padslice(map, box, default=np.nan):
	"""Equivalent to map[...,box[0,0]:box[1,0],box[0,1]:box[1,1]], except that
	pixels outside the map are treated as actually being present, but filled with
	the value given by "default". Hence, ther esult will always have size box[1]-box[0]."""
	box = np.asarray(box).astype(int)
	# Construct our output map
	wcs = map.wcs.deepcopy()
	wcs.wcs.crpix -= box[0,::-1]
	res = full(map.shape[:-2]+tuple(box[1]-box[0]), wcs, default, map.dtype)
	# Get the (possibly smaller) box for the valid pixels of the input map
	ibox = np.maximum(0,np.minimum(np.array(map.shape[-2:])[None],box))
	# Copy over the relevant region
	o, w = ibox[0]-box[0], ibox[1]-ibox[0]
	res[...,o[0]:o[0]+w[0],o[1]:o[1]+w[1]] = map[...,ibox[0,0]:ibox[1,0],ibox[0,1]:ibox[1,1]]
	return res

def tile_maps(maps):
	"""Given a 2d list of enmaps representing contiguous tiles in the
	same global pixelization, stack them into a total map and return it.
	E.g. if maps = [[a,b],[c,d]], then the result would be
	      c d
	map = a b
	"""
	# First stack the actual data:
	m = np.concatenate([np.concatenate(row,-1) for row in maps],-2)
	# Then figure out the wcs of the result. crpix counts from the
	# lower left corner, so a and the total map should have the same wcs
	m = samewcs(m, maps[0][0])
	return m

def stamps(map, pos, shape, aslist=False):
	"""Given a map, extract a set of identically shaped postage stamps with corners
	at pos[ntile,2]. The result will be an enmap with shape [ntile,...,ny,nx]
	and a wcs appropriate for the *first* tile only. If that is not the
	behavior wanted, you can specify aslist=True, in which case the result
	will be a list of enmaps, each with the correct wcs."""
	shape = np.zeros(2)+shape
	pos   = np.asarray(pos)
	res   = []
	for p in pos:
		res.append(padslice(map, [p,p+shape]))
	if aslist: return res
	res = samewcs(np.array(res),res[0])
	return res

def to_healpix(imap, omap=None, nside=0, order=3, chunk=100000, destroy_input=False):
	"""Project the enmap "imap" onto the healpix pixelization. If omap is given,
	the output will be written to it. Otherwise, a new healpix map will be constructed.
	The healpix map must be in RING order. nside controls the resolution of the output map.
	If 0, nside is chosen such that the output map is higher resolution than the input.
	This is needed to avoid losing information. To go to a lower-resolution output map,
	you should first degrade the input map. The chunk argument affects the speed/memory
	tradeoff of the function. Higher values use more memory, and might (and might not)
	give higher speed. If destroy_input is True, then the input map will be prefiltered
	in-place, which saves memory but modifies its values."""
	import healpy
	if not destroy_input and order > 1: imap = imap.copy()
	if order > 1:
		imap = utils.interpol_prefilter(imap, order=order, inplace=True)
	if omap is None:
		# Generate an output map
		if not nside:
			npix_full_cyl = 4*np.pi/imap.pixsize()
			nside = 2**int(np.floor(np.log2((npix_full_cyl/12)**0.5)))
		npix = 12*nside**2
		omap = np.zeros(imap.shape[:-2]+(npix,),imap.dtype)
	else:
		nside = healpy.npix2nside(omap.shape[-1])
	npix = omap.shape[-1]
	# Interpolate values at output pixel positions
	for i in range(0, npix, chunk):
		pos   = np.array(healpy.pix2ang(nside, np.arange(i, min(npix,i+chunk))))
		# Healpix uses polar angle, not dec
		pos[0] = np.pi/2 - pos[0]
		omap[...,i:i+chunk] = imap.at(pos, order=order, mask_nan=False, prefilter=False)
	return omap

def to_flipper(imap, omap=None, unpack=True):
	"""Convert the enmap "imap" into a flipper map with the same geometry. If
	omap is given, the output will be written to it. Otherwise, a an array of
	flipper maps will be constructed. If the input map has dimensions
	[a,b,c,ny,nx], then the output will be an [a,b,c] array with elements
	that are flipper maps with dimension [ny,nx]. The exception is for
	a 2d enmap, which is returned as a plain flipper map, not a
	0-dimensional array of flipper maps. To avoid this unpacking, pass

	Flipper needs cdelt0 to be in decreasing order. This function ensures that,
	at the cost of losing the original orientation. Hence to_flipper followed
	by from_flipper does not give back an exactly identical map to the one
	on started with.
	"""
	import flipper
	if imap.wcs.wcs.cdelt[0] > 0: imap = imap[...,::-1]
	# flipper wants a different kind of wcs object than we have.
	header = imap.wcs.to_header(relax=True)
	header['NAXIS']  = 2
	header['NAXIS1'] = imap.shape[-1]
	header['NAXIS2'] = imap.shape[-2]
	flipwcs = flipper.liteMap.astLib.astWCS.WCS(header, mode="pyfits")
	iflat = imap.preflat
	if omap is None:
		omap = np.empty(iflat.shape[:-2],dtype=object)
	for i, m in enumerate(iflat):
		omap[i] = flipper.liteMap.liteMapFromDataAndWCS(iflat[i], flipwcs)
	omap = omap.reshape(imap.shape[:-2])
	if unpack and omap.ndim == 0: return omap.reshape(-1)[0]
	else: return omap

def from_flipper(imap, omap=None):
	"""Construct an enmap from a flipper map or array of flipper maps imap.
	If omap is specified, it must have the correct shape, and the data will
	be written there."""
	imap   = np.asarray(imap)
	first  = imap.reshape(-1)[0]
	# flipper and enmap wcs objects come from different wcs libraries, so
	# they must be converted
	wcs    = wcsutils.WCS(first.wcs.header).sub(2)
	if omap is None:
		omap = empty(imap.shape + first.data.shape, wcs, first.data.dtype)
	# Copy over all components
	iflat = imap.reshape(-1)
	for im, om in zip(iflat, omap.preflat):
		om[:] = im.data
	omap = fix_endian(omap)
	return omap

############
# File I/O #
############

def write_map(fname, emap, fmt=None, extra={}):
	"""Writes an enmap to file. If fmt is not passed,
	the file type is inferred from the file extension, and can
	be either fits or hdf. This can be overriden by
	passing fmt with either 'fits' or 'hdf' as argument."""
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits"
		else: fmt = "fits"
	if fmt == "fits":
		write_fits(fname, emap, extra=extra)
	elif fmt == "hdf":
		write_hdf(fname, emap, extra=extra)
	else:
		raise ValueError

def read_map(fname, fmt=None, sel=None, box=None, pixbox=None, wrap="auto", mode=None, sel_threshold=10e6, wcs=None, hdu=None):
	"""Read an enmap from file. The file type is inferred
	from the file extension, unless fmt is passed.
	fmt must be one of 'fits' and 'hdf'."""
	toks = fname.split(":")
	fname = toks[0]
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits"
		else: fmt = "fits"
	if fmt == "fits":
		res = read_fits(fname, sel=sel, box=box, pixbox=pixbox, wrap=wrap, mode=mode, sel_threshold=sel_threshold, wcs=wcs, hdu=hdu)
	elif fmt == "hdf":
		res = read_hdf(fname, sel=sel, box=box, pixbox=pixbox, wrap=wrap, mode=mode, sel_threshold=sel_threshold, wcs=wcs)
	else:
		raise ValueError
	if len(toks) > 1:
		res = eval("res"+":".join(toks[1:]))
	return res

def read_map_geometry(fname, fmt=None, hdu=None):
	"""Read an enmap geometry from file. The file type is inferred
	from the file extension, unless fmt is passed.
	fmt must be one of 'fits' and 'hdf'."""
	toks = fname.split(":")
	fname = toks[0]
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits"
		else: fmt = "fits"
	if fmt == "fits":
		shape, wcs = read_fits_geometry(fname, hdu=hdu)
	elif fmt == "hdf":
		shape, wcs = read_hdf_geometry(fname)
	else:
		raise ValueError
	if len(toks) > 1:
		sel = eval("utils.sliceeval"+":".join(toks[1:]))[-2:]
		shape, wcs = slice_geometry(shape, wcs, sel)
	return shape, wcs

def write_fits(fname, emap, extra={}):
	"""Write an enmap to a fits file."""
	# The fits write routines may attempt to modify
	# the map. So make a copy.
	emap = enmap(emap, copy=True)
	# Get our basic wcs header
	header = emap.wcs.to_header(relax=True)
	# Add our map headers
	header['NAXIS'] = emap.ndim
	for i,n in enumerate(emap.shape[::-1]):
		header['NAXIS%d'%(i+1)] = n
	for key, val in extra.items():
		header[key] = val
	hdus   = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(emap, header)])
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		hdus.writeto(fname, clobber=True)

def read_fits(fname, hdu=None, sel=None, box=None, pixbox=None, wrap="auto", mode=None, sel_threshold=10e6, wcs=None):
	"""Read an enmap from the specified fits file. By default,
	the map and coordinate system will be read from HDU 0. Use
	the hdu argument to change this. The map must be stored as
	a fits image. If sel is specified, it should be a slice
	that will be applied to the image before reading. This avoids
	reading more of the image than necessary. Instead of sel,
	a coordinate box [[yfrom,xfrom],[yto,xto]] can be specified."""
	if hdu is None: hdu = 0
	hdu  = astropy.io.fits.open(fname)[hdu]
	ndim = len(hdu.shape)
	if hdu.header["NAXIS"] < 2:
		raise ValueError("%s is not an enmap (only %d axes)" % (fname, hdu.header["NAXIS"]))
	if wcs is None:
		with warnings.catch_warnings():
			wcs = wcsutils.WCS(hdu.header).sub(2)
	wrapper = fits_wrapper(hdu, wcs, threshold=sel_threshold)
	return read_helper(wrapper, sel=sel, box=box, pixbox=pixbox, wrap=wrap, mode=mode)

def read_fits_geometry(fname, hdu=None):
	"""Read an enmap wcs from the specified fits file. By default,
	the map and coordinate system will be read from HDU 0. Use
	the hdu argument to change this. The map must be stored as
	a fits image."""
	if hdu is None: hdu = 0
	hdu = astropy.io.fits.open(fname)[hdu]
	if hdu.header["NAXIS"] < 2:
		raise ValueError("%s is not an enmap (only %d axes)" % (fname, hdu.header["NAXIS"]))
	with warnings.catch_warnings():
		wcs = wcsutils.WCS(hdu.header).sub(2)
	shape = tuple([hdu.header["NAXIS%d"%(i+1)] for i in range(hdu.header["NAXIS"])[::-1]])
	return shape, wcs

def write_hdf(fname, emap, extra={}):
	"""Write an enmap as an hdf file, preserving all
	the WCS metadata."""
	import h5py
	emap = enmap(emap, copy=False)
	with h5py.File(fname, "w") as hfile:
		hfile["data"] = emap
		header = emap.wcs.to_header()
		for key in header:
			hfile["wcs/"+key] = header[key]
		for key, val in extra.items():
			hfile[key] = val

def read_hdf(fname, hdu=None, sel=None, box=None, pixbox=None, wrap="auto", mode=None, sel_threshold=10e6, wcs=None):
	"""Read an enmap from the specified hdf file. Two formats
	are supported. The old enmap format, which simply used
	a bounding box to specify the coordinates, and the new
	format, which uses WCS properties. The latter is used if
	available. With the old format, plate carree projection
	is assumed. Note: some of the old files have a slightly
	buggy wcs, which can result in 1-pixel errors."""
	import h5py
	with h5py.File(fname,"r") as hfile:
		data = hfile["data"]
		hwcs = hfile["wcs"]
		header = astropy.io.fits.Header()
		for key in hwcs:
			header[key] = hwcs[key].value
		if wcs is None:
			wcs = wcsutils.WCS(header).sub(2)
		wrapper = hdf_wrapper(data, wcs, threshold=sel_threshold)
		return read_helper(wrapper, sel=sel, box=box, pixbox=pixbox, wrap=wrap, mode=mode)

def read_hdf_geometry(fname):
	"""Read an enmap wcs from the specified hdf file."""
	import h5py
	with h5py.File(fname,"r") as hfile:
		hwcs = hfile["wcs"]
		header = astropy.io.fits.Header()
		for key in hwcs:
			header[key] = hwcs[key].value
		wcs   = wcsutils.WCS(header).sub(2)
		shape = hfile["data"].shape
	return shape, wcs

def read_helper(data, sel=None, box=None, pixbox=None, wrap="auto", mode=None, sel_threshold=10e6):
	"""Helper function for map reading. Handles the slicing, sky-wrapping and capping, etc."""
	if box    is not None: data = submap(data, box, wrap=wrap)
	if pixbox is not None: data = extract_pixbox(data, pixbox, wrap=wrap)
	if sel    is not None: data = data[sel]
	data = data[:] # Get rid of the wrapper if it still remains
	return data

# These wrapper classes are there to let us reuse the normal map
# extract and submap operations on fits and hdf maps without needing
# to read in all the data.

class fits_wrapper:
	def __init__(self, hdu, wcs, threshold=1e7):
		self.hdu       = hdu
		self.wcs       = wcs
		self.threshold = threshold
		self.dtype     = fix_endian(hdu.section[(slice(0,1),)*hdu.header["NAXIS"]]).dtype
	@property
	def shape(self): return self.hdu.shape
	def __getitem__(self, sel):
		_, psel = utils.split_slice(sel, [len(self.shape)-2,2])
		if len(psel) > 2: raise IndexError("too many indices")
		_, wcs = slice_geometry(self.shape[-2:], self.wcs, psel)
		if self.hdu.size > self.threshold:
			sel1, sel2 = utils.split_slice(sel, [len(self.shape)-1,1])
			res = self.hdu.section[sel1][(Ellipsis,)+sel2]
		else: res = self.hdu.data[sel]
		return ndmap(fix_endian(res), wcs)

class hdf_wrapper:
	def __init__(self, dset, wcs, threshold=1e7):
		self.dset      = dset
		self.wcs       = wcs
		self.threshold = threshold
	@property
	def shape(self): return self.dset.shape
	@property
	def dtype(self): return self.dset.shape
	def __getitem__(self, sel):
		_, psel = utils.split_slice(sel, [self.ndim-2,2])
		if len(psel) > 2: raise IndexError("too many indices")
		_, wcs = slice_geometry(self.shape[-2:], self.wcs, psel)
		if self.dset.size > self.threshold:
			sel1, sel2 = utils.split_slice(sel, [len(self.shape)-1,1])
			res = self.dset[sel1][(Ellipsis,)+sel2]
		else:
			res = self.dset.value[sel]
		return ndmap(fix_endian(res), wcs)

def fix_endian(map):
	"""Make endianness of array map match the current machine.
	Returns the result."""
	if map.dtype.byteorder not in ['=','<' if sys.byteorder == 'little' else '>']:
		map = map.byteswap(True).newbyteorder()
	return map

def shift(map, off, inplace=False):
	"""Cyclicly shift the pixels in map such that a pixel at
	position (i,j) ends up at position (i+off[0],j+off[1])"""
	if not inplace: map = map.copy()
	off = np.atleast_1d(off)
	for i, o in enumerate(off):
		if o != 0:
			map[:] = np.roll(map, o, -len(off)+i)
	return map

def fillbad(map, val=0, inplace=False):
	if not inplace: map = map.copy()
	map[~np.isfinite(map)] = val
	return map

def resample(map, oshape, off=(0,0), method="fft", mode="wrap", corner=False):
	"""Resample the input map such that it covers the same area of the sky
	with a different number of pixels given by oshape."""
	# Construct the output shape and wcs
	oshape = map.shape[:-2] + tuple(oshape)[-2:]
	owcs   = wcsutils.scale(map.wcs, np.array(oshape[-2:],float)/map.shape[-2:], rowmajor=True, corner=corner)
	off    = np.zeros(2)+off
	# Apply phase shift to realign with pixel centers. This can be seen as a half pixel shift to
	# the left in the original pixelization followed by a half pixel shift to the right in the new
	# pixelization.
	if not corner:
		off -= 0.5 - 0.5*np.array(oshape[-2:],float)/map.shape[-2:] # in output units
	if method == "fft":
		omap  = zeros(oshape, owcs, map.dtype)
		fimap = enfft.fft(map, axes=(-2,-1))
		fomap = np.zeros(oshape, fimap.dtype)
		# copy over all 4 quadrants. This would have been a single operation if the
		# fourier center had been in the middle. This could be acieved using fftshift,
		# but that would require two extra full-array shifts
		cny, cnx = np.minimum(map.shape[-2:], oshape[-2:])
		hny, hnx = cny//2, cnx//2
		fomap[...,:hny,        :hnx       ] = fimap[...,:hny,        :hnx       ]
		fomap[...,:hny,        -(cnx-hnx):] = fimap[...,:hny,        -(cnx-hnx):]
		fomap[...,-(cny-hny):, :hnx       ] = fimap[...,-(cny-hny):, :hnx       ]
		fomap[...,-(cny-hny):, -(cnx-hnx):] = fimap[...,-(cny-hny):, -(cnx-hnx):]
		if np.any(off != 0):
			fomap[:] = enfft.shift(fomap, off, axes=(-2,-1), nofft=True)
		omap[:] = enfft.ifft(fomap, axes=(-2,-1)).real
		# Normalize
		omap /= map.shape[-2]*map.shape[-1]
	elif method == "spline":
		opix  = pixmap(oshape) - off[:,None,None]
		ipix  = opix * (np.array(map.shape[-2:],float)/oshape[-2:])[:,None,None]
		omap  = ndmap(map.at(ipix, unit="pix", mode=mode), owcs)
	else:
		raise ValueError("Invalid resample method '%s'" % method)
	return omap

def spin_helper(spin, n):
	spin  = np.array(spin).reshape(-1)
	scomp = 1+(spin!=0)
	ci, i1 = 0, 0
	while True:
		i2 = min(i1+scomp[ci],n)
		if i2-i1 != scomp[ci]: raise IndexError("Unpaired component in spin transform")
		yield spin[ci], i1, i2
		if i2 == n: break
		i1 = i2
		ci = (ci+1)%len(spin)
