from __future__ import print_function
import numpy as np, scipy.ndimage, warnings, astropy.io.fits, sys, time, os, contextlib
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

# Python 2/3 compatibility
try: basestring
except NameError: basestring = str

mute = {
	"polconv_fix": True,
}

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
		"""Wraps a numpy and a wcslib world coordinate system object into an ndmap."""
		obj = np.asarray(arr).view(cls)
		obj.wcs = wcs.deepcopy()
		return obj
	def __array_finalize__(self, obj):
		if obj is None: return
		self.wcs = getattr(obj, "wcs", None)
	def __repr__(self):
		return "ndmap(%s,%s)" % (np.asarray(self), wcsutils.describe(self.wcs))
	def __str__(self): return repr(self)
	def __array_wrap__(self, arr, context=None, return_scalar=False):
		# In the future need to support `return_scalar`, but that is seemingly
		# undocumented and not actually supported in numpy 2.0? So for now we
		# just ignore it.
		if arr.ndim < 2: return arr
		return ndmap(arr, self.wcs)
	def __reduce__(self):
		reconstructor, args, state = super(ndmap, self).__reduce__()
		state += (self.wcs.to_header_string(),)
		return reconstructor, args, state
	def __setstate__(self, state):
		wcs = wcsutils.WCS(header=state[-1])
		super(ndmap, self).__setstate__(state[:-1])
		self.wcs = wcs
	def copy(self, order='K'):
		return ndmap(np.copy(self,order), self.wcs)
	def sky2pix(self, coords, safe=True, corner=False): return sky2pix(self.shape, self.wcs, coords, safe, corner)
	def pix2sky(self, pix,    safe=True, corner=False): return pix2sky(self.shape, self.wcs, pix,    safe, corner)
	def l2pix(self, ls):  return l2pix(self.shape, self.wcs, ls)
	def pix2l(self, pix): return pix2l(self.shape, self.wcs, pix)
	def contains(self, pos, unit="coord"): return contains(self.shape, self.wcs, pos, unit=unit)
	def corners(self, npoint=10, corner=True): return corners(self.shape, self.wcs, npoint=npoint, corner=corner)
	def box(self, npoint=10, corner=True): return box(self.shape, self.wcs, npoint=npoint, corner=corner)
	def pixbox_of(self,oshape,owcs): return pixbox_of(self.wcs, oshape,owcs)
	def posmap(self, safe=True, corner=False, separable="auto", dtype=np.float64): return posmap(self.shape, self.wcs, safe=safe, corner=corner, separable=separable, dtype=dtype)
	def posaxes(self, safe=True, corner=False, dtype=np.float64): return posaxes(self.shape, self.wcs, safe=safe, corner=corner, dtype=dtype)
	def pixmap(self): return pixmap(self.shape, self.wcs)
	def laxes(self, oversample=1, method="auto", broadcastable=False): return laxes(self.shape, self.wcs, oversample=oversample, method=method, broadcastable=broadcastable)
	def lmap(self, oversample=1): return lmap(self.shape, self.wcs, oversample=oversample)
	def lform(self, method="auto"): return lform(self, method=method)
	def modlmap(self, oversample=1, min=0): return modlmap(self.shape, self.wcs, oversample=oversample, min=min)
	def modrmap(self, ref="center", safe=True, corner=False): return modrmap(self.shape, self.wcs, ref=ref, safe=safe, corner=corner)
	def lbin(self, bsize=None, brel=1.0, return_nhit=False, return_bins=False, lop=None): return lbin(self, bsize=bsize, brel=brel, return_nhit=return_nhit, return_bins=return_bins, lop=lop)
	def rbin(self, center=[0,0], bsize=None, brel=1.0, return_nhit=False, return_bins=False, rop=None): return rbin(self, center=center, bsize=bsize, brel=brel, return_nhit=return_nhit, return_bins=return_bins, rop=rop)
	def area(self): return area(self.shape, self.wcs)
	def pixsize(self): return pixsize(self.shape, self.wcs)
	def pixshape(self, signed=False): return pixshape(self.shape, self.wcs, signed=signed)
	def pixsizemap(self, separable="auto", broadcastable=False): return pixsizemap(self.shape, self.wcs, separable=separable, broadcastable=broadcastable)
	def pixshapemap(self, separable="auto", signed=False): return pixshapemap(self.shape, self.wcs, separable=separable, signed=signed)
	def lpixsize(self, signed=False, method="auto"): return lpixsize(self.shape, self.wcs, signed=signed, method=method)
	def lpixshape(self, signed=False, method="auto"): return lpixshape(self.shape, self.wcs, signed=signed, method=method)
	def extent(self, method="auto", signed=False): return extent(self.shape, self.wcs, method=method, signed=signed)
	@property
	def preflat(self):
		"""Returns a view of the map with the non-pixel dimensions flattened."""
		return self.reshape(-1, self.shape[-2], self.shape[-1])
	@property
	def npix(self): return np.prod(self.shape[-2:])
	@property
	def geometry(self): return self.shape, self.wcs
	def resample(self, oshape, off=(0,0), method="fft", border="wrap", corner=True, order=3): return resample(self, oshape, off=off, method=method, border=border, corner=corner, order=order)
	def project(self, shape, wcs, mode="spline", order=3, border="constant", cval=0, safe=True): return project(self, shape, wcs, mode=mode, order=order, border=border, cval=cval, safe=safe)
	def extract(self, shape, wcs, omap=None, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None, reverse=False): return extract(self, shape, wcs, omap=omap, wrap=wrap, op=op, cval=cval, iwcs=iwcs, reverse=reverse)
	def extract_pixbox(self, pixbox, omap=None, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None, reverse=False): return extract_pixbox(self, pixbox, omap=omap, wrap=wrap, op=op, cval=cval, iwcs=iwcs, reverse=reverse)
	def insert(self, imap, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None): return insert(self, imap, wrap=wrap, op=op, cval=cval, iwcs=iwcs)
	def insert_at(self, pix, imap, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None): return insert_at(self, pix, imap, wrap=wrap, op=op, cval=cval, iwcs=iwcs)
	def at(self, pos, mode="spline", order=3, border="constant", cval=0.0, unit="coord", safe=True, ip=None): return at(self, pos, mode=mode, order=order, border=border, cval=0, unit=unit, safe=safe, ip=ip)
	def argmax(self, unit="coord"): return argmax(self, unit=unit)
	def autocrop(self, method="plain", value="auto", margin=0, factors=None, return_info=False): return autocrop(self, method, value, margin, factors, return_info)
	def apod(self, width, profile="cos", fill="zero"): return apod(self, width, profile=profile, fill=fill)
	def stamps(self, pos, shape, aslist=False): return stamps(self, pos, shape, aslist=aslist)
	def distance_from(self, points, omap=None, odomains=None, domains=False, method="cellgrid", rmax=None, step=1024): return distance_from(self.shape, self.wcs, points, omap=omap, odomains=odomains, domains=domains, method=method, rmax=rmax, step=step)
	def distance_transform(self, omap=None, rmax=None, method="cellgrid"): return distance_transform(self, omap=omap, rmax=rmax, method=method)
	def labeled_distance_transform(self, omap=None, odomains=None, rmax=None, method="cellgrid"): return labeled_distance_transform(self, omap=omap, odomains=odomains, rmax=rmax, method=method)
	@property
	def plain(self): return ndmap(self, wcsutils.WCS(naxis=2))
	def padslice(self, box, default=np.nan): return padslice(self, box, default=default)
	def center(self): return center(self.shape,self.wcs)
	def downgrade(self, factor, op=np.mean, ref=None, off=None): return downgrade(self, factor, op=op, ref=ref, off=off)
	def upgrade(self, factor,  off=None, oshape=None, inclusive=False): return upgrade(self, factor, off=off, oshape=oshape, inclusive=inclusive)
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
	def submap(self, box, mode=None, wrap="auto", recenter=False):
		"""Extract the part of the map inside the given coordinate box
		box : array_like
			The [[fromy,fromx],[toy,tox]] coordinate box to select.
			The resulting map will have bottom-left and top-right corners
			as close as possible to this, but will differ slightly due to
			the finite pixel size.
		mode : str
			How to handle partially selected pixels:
			 "round": round bounds using standard rules
			 "floor": both upper and lower bounds will be rounded down
			 "ceil":  both upper and lower bounds will be rounded up
			 "inclusive": lower bounds are rounded down, and upper bounds up
			 "exclusive": lower bounds are rounded up, and upper bounds down"""
		return submap(self, box, mode=mode, wrap=wrap, recenter=recenter)
	def subinds(self, box, mode=None, cap=True):
		return subinds(self.shape, self.wcs, box=box, mode=mode, cap=cap)
	def write(self, fname, fmt=None):
		write_map(fname, self, fmt=fmt)

def submap(map, box, mode=None, wrap="auto", recenter=False, iwcs=None):
	"""Extract the part of the map inside the given coordinate box
	box : array_like
		The [[fromy,fromx],[toy,tox]] coordinate box to select.
		The resulting map will have corners as close
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
	oshape, owcs = recenter_geo(oshape, owcs, mode=recenter)
	omap = extract(map, oshape, owcs, wrap=wrap, iwcs=iwcs)
	# Unflip if neccessary
	if yflip: omap = omap[...,::-1,:]
	if xflip: omap = omap[...,:,::-1]
	return omap

def subgeo(shape, wcs, box=None, pixbox=None, mode=None, wrap="auto", noflip=False, recenter=False):
	"""Extract the part of the geometry inside the coordinate box
	box : array_like
		The [[fromy,fromx],[toy,tox]] coordinate box to select.
		The resulting map will have corners as close
		as possible to this, but will differ slightly due to
		the finite pixel size.
	mode : str
		How to handle partially selected pixels:
		 "round": round bounds using standard rules
		 "floor": both upper and lower bounds will be rounded down
		 "ceil":  both upper and lower bounds will be rounded up
		 "inclusive": lower bounds are rounded down, and upper bounds up
		 "exclusive": lower bounds are rounded up, and upper bounds down
	"""
	if pixbox is not None: ibox = pixbox
	else: ibox = subinds(shape, wcs, box, mode=mode, noflip=noflip, cap=False)
	ogeo = slice_geometry(shape, wcs, (slice(*ibox[:,0]),slice(*ibox[:,1])), nowrap=True)
	ogeo = recenter_geo(*ogeo, mode=recenter)
	return ogeo

def subinds(shape, wcs, box, mode=None, cap=True, noflip=False, epsilon=1e-4):
	"""Helper function for submap. Translates the coordinate box provided
	into a pixel units.

	When box is translated into pixels, the result will in general have
	fractional pixels, which need to be rounded before we can do any slicing.
	To get as robust results as possible, we want
	 1. two boxes that touch should results in iboxses that also touch.
	    This means that upper and lower bounds must be handled consistently.
	    inclusive and exclusive modes break this, and should be used with caution.
	 2. tiny floating point errors should not usually be able to cause
	    the ibox to change. Most boxes will have some simple fraction of
	    a whole degree, and most have pixels with centers or pixel edges
	    at a simple fraction of a whole degree. mode="floor" or "ceil"
	    break when pixel centers are at whole values. mode="round"
	    breaks when pixel edges are at whole values. But since small
	    (but not float-precision-size) offsets from these cases are unlikely,
	    we can define safe rounding by adding an epsilon to the values
	    before rounding. As long as this epsilon is use consistently,
	    box overlap still works.
	With epsilon in place, modes "round", "floor" and "ceil" are all safe.
	We make "round" the default.
	"""
	if mode is None: mode = "round"
	box = np.asarray(box)
	# Translate the box to pixels
	bpix = skybox2pixbox(shape, wcs, box, include_direction=True)
	bpix[:2] += epsilon
	if noflip:
		for b in bpix.T:
			if b[2] < 0: b[:] = [b[1],b[0],-b[2]]
	if   mode == "round": bpix = np.round(bpix)
	elif mode == "floor": bpix = np.floor(bpix)
	elif mode == "ceil":  bpix = np.ceil(bpix)
	elif mode == "inclusive": bpix = [np.floor(bpix[0]),np.ceil (bpix[1]), bpix[2]]
	elif mode == "exclusive": bpix = [np.ceil (bpix[0]),np.floor(bpix[1]), bpix[2]]
	else: raise ValueError("Unrecognized mode '%s' in subinds" % str(mode))
	bpix = np.array(bpix, int)
	# A pixel goes from [i1-0.5:i2+0.5] with round(+eps) this becomes [i1:i2+1]
	# We therefore don't need to add 1 to get a proper slice
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
	# Convert this is a bit cumbersome, but ensures we get a tuple of
	# plain python ints instead of numpy.int64s
	oshape = tuple([int(a) for a in oshape])
	return tuple(pre)+oshape, wcs

def scale_geometry(shape, wcs, scale):
	"""Scale the geometry so that the number of pixels is scaled 
	by the factor `scale`.
	"""
	scale  = np.zeros(2)+scale
	oshape = tuple(shape[:-2])+tuple(utils.nint(shape[-2:]*scale))
	owcs   = wcsutils.scale(wcs, scale, rowmajor=True)
	return oshape, owcs

def get_unit(wcs):
	return utils.degree

def npix(shape): return shape[-2]*shape[-1]

class Geometry:
	def __init__(self, shape, wcs=None):
		try: self.shape, self.wcs = tuple(shape.shape), shape.wcs
		except AttributeError: self.shape, self.wcs = tuple(shape), wcs
		assert wcs is not None, "Geometry __init__ needs either a Geometry object or a shape, wcs pair"
	@property
	def npix(self): return self.shape[-2]*self.shape[-1]
	# Make it behave a bit like a tuple, so we can use it interchangably with a shape, wcs pair
	# for compatibility
	def __len__(self): return 2
	def __iter__(self):
		yield self.shape
		yield self.wcs
	def __getitem__(self, sel):
		if not isinstance(sel,tuple): sel = (sel,)
		shape, wcs = slice_geometry(self.shape, self.wcs, sel)
		return Geometry(shape, wcs)
	def __repr__(self):
		return "Geometry(" + str(self.shape) + ","+str(self.wcs)+")"
	@property
	def nopre(self): return Geometry(self.shape[-2:], self.wcs)
	def with_pre(self, pre): return Geometry(tuple(pre) + self.shape[-2:], self.wcs)
	def submap(self, box=None, pixbox=None, mode=None, wrap="auto", noflip=False, recenter=False):
		return Geometry(*subgeo(*self, box=box, pixbox=pixbox, mode=mode, wrap=wrap, noflip=noflip, recenter=recenter))
	def scale(self, scale):
		shape, wcs = scale_geometry(self.shape, self.wcs, scale)
		return Geometry(shape, wcs)
	def downgrade(self, factor, op=np.mean):
		shape, wcs = downgrade_geometry(self.shape, self.wcs, factor, op=op)
		return Geometry(shape, wcs)
	def copy(self):
		return Geometry(tuple(self.shape), self.wcs.deepcopy())
	def sky2pix(self, coords, safe=True, corner=False): return sky2pix(self.shape, self.wcs, coords, safe, corner)
	def pix2sky(self, pix,    safe=True, corner=False): return pix2sky(self.shape, self.wcs, pix,    safe, corner)
	def l2pix(self, ls):  return l2pix(self.shape, self.wcs, ls)
	def pix2l(self, pix): return pix2l(self.shape, self.wcs, pix)

def corners(shape, wcs, npoint=10, corner=True):
	"""Return the coordinates of the bottom left and top right corners of the
	geometry given by shape, wcs.

	If corner==True it is similar to
	enmap.pix2sky([[-0.5,shape[-2]-0.5],[-0.5,shape[-1]-0.5]]). That is, it
	return sthe coordinate of the bottom left corner of the bottom left pixel and
	the top right corner of the top right pixel. If corner==False, then it
	instead returns the corresponding pixel centers.

	It differs from the simple pix2sky calls above by handling 2*pi wrapping
	ambiguities differently. enmap.corners ensures that the coordinates returned
	are on the same side of the wrapping cut so that the coordinates of the
	two corners can be compared without worrying about wrapping. It does this
	by evaluating a set of intermediate points between the corners and counting
	and undoing any sudden jumps in coordinates it finds. This is controlled by
	the npoint option. The default of 10 should be more than enough.

	Returns [{bottom left,top right},{dec,ra}] in radians
	(or equivalent for other coordinate systems).
	e.g. an array of the form [[dec_min, ra_min ], [dec_max, ra_max]]."""
	# Because of wcs's wrapping, we need to evaluate several
	# extra pixels to make our unwinding unambiguous.
	# Could reduce code duplication a bit here, but I think it's clearer
	# when written like this
	if corner:
		pix = np.array([
			np.linspace(-0.5,shape[-2]-0.5,num=npoint,endpoint=True),
			np.linspace(-0.5,shape[-1]-0.5,num=npoint,endpoint=True)])
	else:
		pix = np.array([
			np.linspace(0,shape[-2]-1,num=npoint,endpoint=True),
			np.linspace(0,shape[-1]-1,num=npoint,endpoint=True)])
	coords = wcsutils.nobcheck(wcs).wcs_pix2world(pix[1],pix[0],0)[::-1]
	if wcsutils.is_plain(wcs):
		return np.array(coords).T[[0,-1]]*get_unit(wcs)
	else:
		return utils.unwind(np.array(coords)*get_unit(wcs),refmode="middle").T[[0,-1]]
def box(shape, wcs, npoint=10, corner=True):
	"""Alias for corners."""
	return corners(shape, wcs, npoint=npoint, corner=corner)

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
	"""
	Return an enmap with entries uninitialized (like numpy.empty).
	"""
	return enmap(np.empty(shape, dtype=dtype), wcs, copy=False)

def zeros(shape, wcs=None, dtype=None):
	"""
	Return an enmap with entries initialized to zero (like
	numpy.zeros).
	"""
	return enmap(np.zeros(shape, dtype=dtype), wcs, copy=False)

def ones(shape, wcs=None, dtype=None):
	"""
	Return an enmap with entries initialized to one (like numpy.ones).
	"""
	return enmap(np.ones(shape, dtype=dtype), wcs, copy=False)

def full(shape, wcs, val, dtype=None):
	"""
	Return an enmap with entries initialized to val (like numpy.full).
	"""
	return enmap(np.full(shape, val, dtype=dtype), wcs, copy=False)

def posmap(shape, wcs, safe=True, corner=False, separable="auto", dtype=np.float64, bsize=1e6, bcheck=False):
	"""Return an enmap where each entry is the coordinate of that entry,
	such that posmap(shape,wcs)[{0,1},j,k] is the {y,x}-coordinate of
	pixel (j,k) in the map. Results are returned in radians, and
	if safe is true (default), then sharp coordinate edges will be
	avoided. separable controls whether a fast calculation that assumes that
	ra is only a function of x and dec is only a function of y is used.
	The default is "auto", which determines this based on the wcs, but
	True or False can also be passed to control this manually.

	For even greater speed, and to save memory, consider using posaxes directly
	for cases where you know that the wcs will be separable. For separable cases,
	separable=True is typically 15-20x faster than separable=False, while posaxes
	is 1000x faster.
	"""
	res = zeros((2,)+tuple(shape[-2:]), wcs, dtype)
	if separable == "auto": separable = wcsutils.is_separable(wcs)
	if separable:
		# If posmap could return a (dec,ra) tuple instead of an ndmap,
		# we could have returned np.broadcast_arrays(dec, ra) instead.
		# That would have been as fast and memory-saving as broadcast-arrays.
		dec, ra = posaxes(shape, wcs, safe=safe, corner=corner, bcheck=bcheck)
		res[0] = dec[:,None]
		res[1] = ra[None,:]
	else:
		rowstep = int((bsize+shape[-1]-1)//shape[-1])
		for i1 in range(0, shape[-2], rowstep):
			i2  = min(i1+rowstep, shape[-2])
			pix = np.mgrid[i1:i2,:shape[-1]]
			res[:,i1:i2,:] = pix2sky(shape, wcs, pix, safe, corner, bcheck=bcheck)
	return res

def posmap_old(shape, wcs, safe=True, corner=False):
		pix    = np.mgrid[:shape[-2],:shape[-1]]
		return ndmap(pix2sky(shape, wcs, pix, safe, corner), wcs)

def posaxes(shape, wcs, safe=True, corner=False, dtype=np.float64, bcheck=False):
	y = np.arange(shape[-2])
	x = np.arange(shape[-1])
	dec = pix2sky(shape, wcs, np.array([y,y*0]), safe=safe, corner=corner, bcheck=bcheck)[0].astype(dtype, copy=False)
	ra  = pix2sky(shape, wcs, np.array([x*0,x]), safe=safe, corner=corner, bcheck=bcheck)[1].astype(dtype, copy=False)
	return dec, ra

def pixmap(shape, wcs=None):
	"""Return an enmap where each entry is the pixel coordinate of that entry."""
	res = np.mgrid[:shape[-2],:shape[-1]]
	return res if wcs is None else ndmap(res,wcs)

def pix2sky(shape, wcs, pix, safe=True, corner=False, bcheck=False):
	"""Given an array of pixel coordinates [{y,x},...],
	return sky coordinates in the same ordering."""
	pix = np.asarray(pix).astype(float)
	if corner: pix -= 0.5
	pflat = pix.reshape(pix.shape[0], -1)
	if not bcheck: wcs = wcsutils.nobcheck(wcs)
	coords = np.asarray(wcs.wcs_pix2world(*(tuple(pflat)[::-1]+(0,)))[::-1])*get_unit(wcs)
	coords = coords.reshape(pix.shape)
	if safe and not wcsutils.is_plain(wcs):
		coords[1] = utils.unwind(coords[1], refmode="middle")
	return coords

def sky2pix(shape, wcs, coords, safe=True, corner=False, bcheck=False):
	"""Given an array of coordinates [{dec,ra},...], return
	pixel coordinates with the same ordering. The corner argument
	specifies whether pixel coordinates start at pixel corners
	or pixel centers. This represents a shift of half a pixel.
	If corner is False, then the integer pixel closest to a position
	is round(sky2pix(...)). Otherwise, it is floor(sky2pix(...))."""
	coords = np.asarray(coords)/get_unit(wcs)
	cflat  = coords.reshape(coords.shape[0], -1)
	# Quantities with a w prefix are in wcs ordering (ra,dec)
	if not bcheck: wcs = wcsutils.nobcheck(wcs)
	wpix = np.asarray(wcs.wcs_world2pix(*tuple(cflat)[::-1]+(0,)))
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

def pix2l(shape, wcs, pix):
	"""Given an array of fourier-pixel coordinates [{y,x},...], returns
	the 2d fourier coordinates [{ly,lx},...]."""
	pix    = np.asanyarray(pix)
	pshape = pixshape(shape, wcs, signed=True)
	return enfft.ind2freq(np.array(shape[-2:]).T, pix.T, pshape.T/(2*np.pi)).T

def l2pix(shape, wcs, ls):
	"""Given an array of fourier-pixel coordinates [{y,x},...], returns
	the 2d fourier coordinates [{ly,lx},...]."""
	ls    = np.asanyarray(ls)
	pshape = pixshape(shape, wcs, signed=True)
	return enfft.freq2ind(np.array(shape[-2:]).T, ls.T, pshape.T/(2*np.pi)).T

def skybox2pixbox(shape, wcs, skybox, npoint=10, corner=False, include_direction=False):
	"""Given a coordinate box [{from,to},{dec,ra}], compute a
	corresponding pixel box [{from,to},{y,x}]. We avoid
	wrapping issues by evaluating a number of subpoints."""
	coords = np.array([
		np.linspace(skybox[0,0],skybox[1,0],num=npoint,endpoint=True),
		np.linspace(skybox[0,1],skybox[1,1],num=npoint,endpoint=True)])
	pix = sky2pix(shape, wcs, coords, corner=corner, safe=2)
	dir = np.sign(pix[:,1]-pix[:,0])
	res = pix[:,[0,-1]].T
	if include_direction: res = np.concatenate([res,dir[None]],0)
	return res

def pixbox2skybox(shape, wcs, pixbox):
	return pix2sky(shape, wcs, np.asanyarray(pixbox).T).T

def contains(shape, wcs, pos, unit="coord"):
	"""For the points with coordinates pos[{dec,ra},...] return whether
	each is inside the geometry given by shape, wcs"""
	if unit == "coord": pix = sky2pix(shape, wcs, pos)
	else:               pix = pos
	return np.all((pix>=0)&(pix.T<shape[-2:]).T,0)

def project(map, shape, wcs, mode="spline", order=3, border="constant",
		cval=0.0, force=False, safe=True, bsize=1000, context=50, ip=None):
	"""Project map to a new geometry.

	This function is not suited for going down in resolution, because
	only interpolation is done, not averaging. This means that if
	the output geometry has lower resolution than the input, then
	information will be lost because noise will not average down the
	way it optimally would.

	* map: enmap.ndmap of shape [...,ny,nx]
	* shape, wcs: The geometry to project to
	* mode: The interpolation mode. Same meaning as in utils.interpol.
	  Valid values are "nearest", "linear", "cubic", "spline" and "fourier".
	  "nearest" and "linear" are local interpolations, where one does
	  not need to worry about edge effects and ringing. "cubic" and
	  especially "fourier" are sensitive to the boundary conditions,
	  and maps may need to be apodized first.  Only "fourier"
	  interpolation preserves map power on all scales. The other
	  types lose a bit of power at high multipoles.
	  fourier > cubic > linear > nearest for high-l fidelity.
	  "spline" is a generalization of "nearest", "linear" and "cubic",
	  depending on the "order" argument: 0, 1 and 3.
	* order: Controls the "spline" mode. See above.
	* border: The boundary condition to assume for spline interpolation.
	  Ignored for Fourier-interpolation, which always assumes periodic
	  boundary conditions. Defaults to "constant", where areas outside
	  the map are assumed to have the constant value "cval".
	* cval: See "border".
	* force: Force interpolation, even when the input and output pixels
	  are directly compatible, so no interpolation is necessary. Normally
	  the faster enmap.extract is used in this case.
	* safe: If True (default) make extra effort to resolve 2pi sky
	  wrapping degeneracies in the coordinate conversion.
	* bsize: The interpolation is done in blocks in the y axis to save
	  memory. This argument controls how many rows are processed at once.
	* context: How much to pad each y block by. Used to avoid ringing due
	  to discontinuities at block boundaries. Defaults to 50.
	* ip: An interpolator object as returned by utils.interpolator(). If
	  provided, this interpolator is used directly, and the interpolation
	  arguments (mode, order, border, cval) are ignored. If the
	  interpolator does not count as "prefiltered" (meaning that each use of
	  the interpolator could incurr a potentially large cost regardless of
	  how few points are interpolated), then the whole map is processed in
	  one go, ignoring bsize"""
	# Skip expensive operation if map is compatible
	if not force:
		if wcsutils.equal(map.wcs, wcs) and tuple(shape[-2:]) == tuple(shape[-2:]):
			return map.copy()
		elif wcsutils.is_compatible(map.wcs, wcs) and border == "constant":
			return extract(map, shape, wcs, cval=cval)
	omap = zeros(map.shape[:-2]+shape[-2:], wcs, map.dtype)
	# Save memory by looping over rows. This won't work for non-"prefiltered" interpolators
	if ip and not ip.prefiltered: bsize=100000000
	# Avoid unneccessary padding for local cases
	if   ip or (mode == "spline" and order == 0): context = 0
	elif        mode == "spline" and order == 1 : context = 1
	elif        mode == "fourier": context = 32
	# It would have been nice to be able to use padtiles here, but
	# the input and output tilings are very different
	for i1 in range(0, shape[-2], bsize):
		i2     = min(i1+bsize, shape[-2])
		somap  = omap[...,i1:i2,:]
		pix    = map.sky2pix(somap.posmap(), safe=safe)
		if ip:
			# Can't subdivide interpolator
			band = map
		else:
			y1     = np.min(pix[0]).astype(int)-context
			y2     = np.max(pix[0]).astype(int)+context+1
			pix[0]-= y1
			band   = map.extract_pixbox([[y1,0],[y2,map.shape[-1]]])
			# Apodize if necessary
			if context > 1:
				band = apod(band, width=(context,0), fill="zero")
		# And do the interpolation
		somap[:] = utils.interpol(band, pix, mode=mode, order=order, border=border, cval=cval, ip=ip)
	return omap

def pixbox_of(iwcs,oshape,owcs):
	"""Obtain the pixbox which when extracted from a map with WCS=iwcs
	returns a map that has geometry oshape,owcs.
	"""
	# First check that our wcs is compatible
	assert wcsutils.is_compatible(iwcs, owcs), "Incompatible wcs in enmap.extract: %s vs. %s" % (str(iwcs), str(owcs))
	# Find the pixel bounding box of the output in terms of the input.
	# This is simple because our wcses are compatible, so they
	# can only differ by a simple pixel offset. Here pixoff is
	# pos_input - pos_output. This is equivalent to the coordinates of
	pixoff = utils.nint((iwcs.wcs.crpix-owcs.wcs.crpix) - (iwcs.wcs.crval-owcs.wcs.crval)/iwcs.wcs.cdelt)[::-1]
	pixbox = np.array([pixoff,pixoff+np.array(oshape[-2:])])
	return pixbox

def extract(map, shape, wcs, omap=None, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None, reverse=False):
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
	if iwcs is None: iwcs = map.wcs
	pixbox = pixbox_of(iwcs,shape,wcs)
	extracted = extract_pixbox(map, pixbox, omap=omap, wrap=wrap, op=op, cval=cval, iwcs=iwcs, reverse = reverse)
	# There is a degeneracy between crval and crpix in the wcs, so the
	# extracted map's wcs might not be identical, but is equivalent.
	# We explicitly set the wcs to be identical.
	extracted.wcs = wcs
	return extracted

def extract_pixbox(map, pixbox, omap=None, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None, reverse=False, recenter=False):
	"""This function extracts a rectangular area from an enmap based on the
	given pixbox[{from,to,[stride]},{y,x}]. The difference between this function
	and plain slicing of the enmap is that this one supports wrapping around the
	sky. This is necessary to make things like fast thumbnail or tile extraction
	at the edge of a (horizontally) fullsky map work."""
	if iwcs is None: iwcs = map.wcs
	pixbox = np.asarray(pixbox)
	if omap is None:
		oshape, owcs = slice_geometry(map.shape, iwcs, (slice(*pixbox[:,-2]),slice(*pixbox[:,-1])), nowrap=True)
		omap = full(map.shape[:-2]+tuple(oshape[-2:]), owcs, cval, map.dtype)
	nphi = utils.nint(360/np.abs(iwcs.wcs.cdelt[0]))
	# If our map is wider than the wrapping length, assume we're a lower-spin field
	nphi *= (nphi+map.shape[-1]-1)//nphi
	if utils.streq(wrap, "auto"):
		wrap = [0,0] if wcsutils.is_plain(iwcs) else [0,nphi]
	else: wrap = np.zeros(2,int)+wrap
	for ibox, obox in utils.sbox_wrap(pixbox.T, wrap=wrap, cap=map.shape[-2:]):
		islice = utils.sbox2slice(ibox)
		oslice = utils.sbox2slice(obox)
		if reverse: map [islice] = op(map[islice], omap[oslice])
		else:       omap[oslice] = op(omap[oslice], map[islice])
	# Optionally recenter cylindrical geometries so the reference point is
	# in-bounds in RA, but only do it if we're not in reverse mode,
	# since we shouldn't be writing to omap then
	if not reverse:
		omap.wcs = recenter_geo(omap.shape, omap.wcs, mode=recenter)[1]
	return omap

def insert(omap, imap, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None):
	"""Insert imap into omap based on their world coordinate systems, which
	must be compatible. Essentially the reverse of extract."""
	extract(omap, imap.shape, imap.wcs, imap, wrap="auto", op=op, cval=0, iwcs=None, reverse=True)
	return omap

def insert_at(omap, pix, imap, wrap="auto", op=lambda a,b:b, cval=0, iwcs=None):
	"""Insert imap into omap at the position given by pix. If pix is [y,x], then
	[0:ny,0:nx] in imap will be copied into [y:y+ny,x:x+nx] in omap. If pix is
	[{from,to,[stride]},{y,x}], then this specifies the omap pixbox into which to
	copy imap. Wrapping is handled the same way as in extract."""
	pixbox = np.array(pix)
	if pixbox.ndim == 1: pixbox = np.array([pixbox,pixbox+imap.shape[-2:]])
	extract_pixbox(omap, pixbox, imap, wrap=wrap, op=op, cval=cval, iwcs=iwcs, reverse=True)
	return omap

def map_union(map1, map2):
	"""Given two maps with compatible wcs but possibly covering different
	parts of the sky, return a new map that contains all pixels of both maps.
	If the input maps overlap, then those pixels will have the sum of the two maps"""
	oshape, owcs = union_geometry([map1.geometry, map2.geometry])
	omap = zeros(map1.shape[:-2]+oshape[-2:], owcs, map1.dtype)
	omap.insert(map1)
	omap.insert(map2, op=lambda a,b:a+b)
	return omap

def overlap(shape, wcs, shape2_or_pixbox, wcs2=None, wrap="auto"):
	"""Compute the overlap between the given geometry (shape, wcs) and another *compatible*
	geometry. This can be either another shape, wcs pair or a pixbox[{from,to},{y,x}].
	Returns the geometry of the overlapping region."""
	# Is it a shape or a pixbox
	tmp = np.asarray(shape2_or_pixbox)
	if   tmp.ndim == 1: pixbox = pixbox_of(wcs, shape2_or_pixbox, wcs2)
	elif tmp.ndim == 2: pixbox = shape2_or_pixbox
	else: raise ValueError("3rd argument of overlap should be an enmap, a shape tuple or a pixbox")
	# Handle wrapping
	nphi = utils.nint(360/np.abs(wcs.wcs.cdelt[0]))
	# If our map is wider than the wrapping length, assume we're a lower-spin field
	nphi *= (nphi+shape[-1]-1)//nphi
	if utils.streq(wrap, "auto"):
		wrap = [0,0] if wcsutils.is_plain(wcs) else [0,nphi]
	# Looks like the sbox stuff in utils doesn't do this, so do it ourself.
	for i in range(2):
		# If pixbox goes below 0, truncate it unless it goes negative
		# enough to reach our wrapped end.
		if pixbox[0,i] < 0 and (not wrap[i] or pixbox[0,i]+wrap[i] >= shape[-2+i]):
			pixbox[0,i] = 0
		# Similarly, if it goes above our end, truncate it unless it
		# goes far enough to reach our wrapped beginning
		if pixbox[1,i] > shape[-2+i] and (not wrap[i] or pixbox[1,i]-wrap[i] <= 0):
			pixbox[1,i] = shape[-2+i]
	# This will ensure that we get a zero shape instead of a negative one if
	# there is no overlap
	pixbox[1] = np.maximum(pixbox[1],pixbox[0])
	# Good, we now have the capped, wrapped pixbox
	oshape = tuple(pixbox[1]-pixbox[0])
	owcs   = wcs.deepcopy()
	owcs.wcs.crpix -= pixbox[0,1::-1]
	return oshape, owcs

def neighborhood_pixboxes(shape, wcs, poss, r):
	"""Given a set of positions poss[npos,{dec,ra}] in radians and a distance r in radians,
	return pixboxes[npos][{from,to},{y,x}] corresponding to the regions within a
	distance of r from each entry in poss."""
	if wcsutils.is_plain(wcs):
		rpix = r/pixsize(shape, wcs)
		centers = sky2pix(poss.T).T
		return np.moveaxis([centers-rpix,center+rpix+1],0,1)
	poss, r = utils.broadcast_arrays(poss, r, npost=(1,0))
	res     = np.zeros(poss.shape[:-1]+(2,2))
	for I in utils.nditer(poss.shape[:-1]):
		pos, r_ = poss[I], r[I]
		# Find the coordinate box we need
		dec, ra = pos[:2]
		dec1, dec2 = max(dec-r_,-np.pi/2), min(dec+r_,np.pi/2)
		with utils.nowarn():
			scale = 1/min(np.cos(dec1), np.cos(dec2))
		dra        = min(r_*scale, np.pi)
		ra1, ra2   = ra-dra, ra+dra
		box        = np.array([[dec1,ra1],[dec2,ra2]])
		# And get the corresponding pixbox
		res[I]     = skybox2pixbox(shape, wcs, box)
	# Turn ranges into from-inclusive, to-exclusive integers.
	res = utils.nint(res)
	res = np.sort(res, -2)
	res[...,1,:] += 1
	return res

def at(map, pos, mode="spline", order=3, border="constant", cval=0.0, unit="coord", safe=True, ip=None):
	if unit != "pix": pos = sky2pix(map.shape, map.wcs, pos, safe=safe)
	return utils.interpol(map, pos, mode=mode, order=order, border=border, cval=cval, ip=ip)

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
		return ifft(kmap,normalize=True).real
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
		covsqrt = spec2flat(shape, wcs, massage_spectrum(cov, shape), 0.5, border="constant")
	data = map_mul(covsqrt, rand_gauss_harm(shape, wcs))
	return ndmap(data, wcs)

def massage_spectrum(cov, shape):
	"""given a spectrum cov[nl] or cov[n,n,nl] and a shape
	(stokes,ny,nx) or (ny,nx), return a new ocov that has
	a shape compatible with shape, padded with zeros if necessary.
	If shape is scalar (ny,nx), then ocov will be scalar (nl).
	If shape is (stokes,ny,nx), then ocov will be (stokes,stokes,nl)."""
	cov = np.asarray(cov)
	if cov.ndim == 1: cov = cov[None,None]
	if len(shape) == 2: return cov[0,0]
	ocov = np.zeros((shape[0],shape[0])+cov.shape[2:])
	nmin = min(cov.shape[0],ocov.shape[0])
	ocov[:nmin,:nmin] = cov[:nmin,:nmin]
	return ocov

def extent(shape, wcs, nsub=None, signed=False, method="auto"):
	"""Returns the area of a patch with the given shape
	and wcs, in steradians."""
	if method == "auto":
		if   wcsutils.is_plain(wcs):     method = "intermediate"
		elif wcsutils.is_separable(wcs): method = "cylindrical"
		else:                            method = "subgrid"
	if   method in ["inter","intermediate"]: return extent_intermediate(shape, wcs, signed=signed)
	elif method in ["cyl",  "cylindrical"]:  return extent_cyl(shape, wcs, signed=signed)
	elif method in ["sub", "subgrid"]:       return extent_subgrid(shape, wcs, nsub=nsub, signed=signed)
	else: raise ValueError("Unrecognized method '%s' in extent()" % method)

def extent_intermediate(shape, wcs, signed=False):
	"""Estimate the flat-sky extent of the map as the WCS
	intermediate coordinate extent. This is very simple, but
	is only appropriate for very flat coordinate systems"""
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
	total_area = area(shape, wcs)
	if nsub is None: nsub = 17
	# Create a new wcs with (nsub,nsub) pixels
	wcs = wcs.deepcopy()
	step = (utils.asfarray(shape[-2:])/nsub)[::-1]
	wcs.wcs.crpix -= 0.5
	wcs.wcs.cdelt *= step
	wcs.wcs.crpix /= step
	wcs.wcs.crpix += 0.5
	# We need a representative cos(dec) for each pixel. Will use the center
	coss = np.cos(posmap([nsub,nsub], wcs, safe=False)[0])
	# Get the length of each row in the image. This will be the distance
	# from the middle of the left edge of the left-most pixel to the middle
	# of the right edge of the right-most pixel
	pixs  = np.mgrid[:nsub,:nsub+1].astype(float)
	pixs[1] -= 0.5
	decs, ras = pix2sky(nsub, wcs, pixs, safe=False)
	pix_lengths = (utils.rewind(decs[:,1:]-decs[:,:-1])**2 + (utils.rewind(ras[:,1:]-ras[:,:-1])*coss)**2)**0.5
	# Get the height of each col in the image
	pixs  = np.mgrid[:nsub+1,:nsub].astype(float)
	pixs[0] -= 0.5
	decs, ras = pix2sky(nsub, wcs, pixs, safe=False)
	pix_heights = (utils.rewind(decs[1:,:]-decs[:-1,:])**2 + (utils.rewind(ras[1:,:]-ras[:-1,:])*coss)**2)**0.5
	# The area is the sum of their product
	pix_areas  = pix_lengths*pix_heights
	# The extent should be a compromise between the different lengths and heights
	# that gives the right area.
	mean_length = np.mean(pix_lengths)*nsub
	mean_height = np.mean(pix_heights)*nsub
	# Then scale both to ensure we get the right area
	correction  = (total_area/(mean_length*mean_height))**0.5
	mean_length *= correction
	mean_height *= correction
	ext = np.array([mean_height, mean_length])
	if signed: ext *= np.sign(wcs.wcs.cdelt[::-1])
	return ext

def extent_cyl(shape, wcs, signed=False):
	"""Extent specialized for a cylindrical projection.
	Vertical: ny*cdelt[1]
	Horizontal: Each row is nx*cdelt[0]*cos(dec), but we
	want a single representative number, which will be
	some kind of average, and we're free to choose which. We choose
	the one that makes the product equal the true area.
	Area = nx*ny*cdelt[0]*cdelt[1]*mean(cos(dec)) = vertical*(nx*cdelt[0]*mean(cos)),
	so horizontal = nx*cdelt[0]*mean(cos)"""
	dec1, dec2 = pix2sky(shape, wcs, [[-0.5,shape[-2]-1+0.5],[0,0]], safe=False)[0]
	if dec1 <= dec2: ysign = 1
	else: dec1, dec2, ysign = dec2, dec1, -1
	dec1, dec2 = max(-np.pi/2, dec1), min(np.pi/2, dec2)
	mean_cos   = (np.sin(dec2)-np.sin(dec1))/(dec2-dec1)
	ext = np.array([(dec2-dec1)*ysign, shape[-1]*wcs.wcs.cdelt[0]*mean_cos*get_unit(wcs)])
	if not signed: ext = np.abs(ext)
	return ext

def area(shape, wcs, nsamp=1000, method="auto"):
	"""Returns the area of a patch with the given shape
	and wcs, in steradians."""
	if method == "auto":
		if   wcsutils.is_plain(wcs):     method = "intermediate"
		elif wcsutils.is_separable(wcs): method = "cylindrical"
		else:                            method = "contour"
	if   method in ["inter","intermediate"]: return area_intermediate(shape, wcs)
	elif method in ["cyl",  "cylindrical"]:  return area_cyl(shape, wcs)
	elif method in ["cont", "contour"]:      return area_contour(shape, wcs, nsamp=nsamp)
	else: raise ValueError("Unrecognized method '%s' in area()" % method)

def area_intermediate(shape, wcs):
	"""Get the area of a completely flat sky"""
	return np.abs(shape[-2]*shape[-1]*wcs.wcs.cdelt[0]*wcs.wcs.cdelt[1]*get_unit(wcs)**2)

def area_cyl(shape, wcs):
	"""Get the area of a cylindrical projection. Fast and exact."""
	dec1, dec2 = np.sort(pix2sky(shape, wcs, [[-0.5,shape[-2]-1+0.5],[0,0]], safe=False)[0])
	dec1, dec2 = max(-np.pi/2, dec1), min(np.pi/2, dec2)
	return (np.sin(dec2)-np.sin(dec1))*abs(wcs.wcs.cdelt[0])*shape[-1]*get_unit(wcs)

def area_contour(shape, wcs, nsamp=1000):
	"""Get the area of the map by doing a contour integral (1-sin(dec)) d(RA)
	over the closed path (dec(t), ra(t)) that bounds the valid region of
	the map, so it only works for projections where we can figure out this
	boundary. Using only d(RA) in the integral corresponds to doing a top-hat
	integral instead of something trapezoidal, but this method is fast enough
	that we can afford many points to compensate.
	The present implementation works for cases where the valid
	region of the map runs through the centers of the pixels on
	each edge or through the outer edge of those pixels (this
	detail can be different for each edge).  The former case is
	needed in the full-sky cylindrical projections that have
	pixels centered exactly on the poles.
	"""
	n2, n1 = shape[-2:]
	row_lims, col_lims = [], []
	# Ideally we want to integrate around the real outer edge
	# of our patch, which is displaced by half a pixel coordinate
	# from the pixel centers, but sometimes those coordinates are
	# not valid. The nobcheck should avoid that, but we still include
	# them to be safe. For the case where nobcheck avoids invalid values,
	# the clip() later cuts off the parts of the pixels that go outside
	# bounds. This differs from using the backup points by also handling
	# pixels that are offset from the poles by a non-half-integer amount.
	for dest_list, test_points in [
			(col_lims, [(  -0.5, 0.0), (   0.0, 0.0)]),
			(col_lims, [(n1-0.5, 0.0), (n1-1.0, 0.0)]),
			(row_lims, [(0.0,   -0.5), (0.0,    0.0)]),
			(row_lims, [(0.0, n2-0.5), (0.0, n2-1.0)]),
			]:
		for t in test_points:
			if not np.any(np.isnan(wcsutils.nobcheck(wcs).wcs_pix2world([t], 0))):
				dest_list.append(np.array(t, float))
				break
		else:
			raise ValueError("Could not identify map_boundary; last test point was %s" % t)
	# We want to draw a closed patch connecting the four corners
	# of the boundary.
	col_lims = [_c[0] for _c in col_lims]
	row_lims = [_r[1] for _r in row_lims]
	vertices = np.array([
			(col_lims[0], row_lims[0]),
			(col_lims[1], row_lims[0]),
			(col_lims[1], row_lims[1]),
			(col_lims[0], row_lims[1]),
			(col_lims[0], row_lims[0])])
	total   = 0
	tot_dra = 0
	for v0, v1 in zip(vertices[:-1], vertices[1:]):
		line_pix = np.linspace(0, 1, nsamp)[:,None] * (v1 - v0) + v0
		line = wcsutils.nobcheck(wcs).wcs_pix2world(line_pix, 0)
		# Stay within valid dec values. Used for pixels at the poles
		line[:,1] = np.clip(line[:,1], -90, 90)
		dec      = (line[1:,1] + line[:-1,1]) / 2  # average dec
		dra      = line[1:,0] - line[:-1,0]        # delta RA
		dra      = (dra+180) % 360 - 180          # safetyize branch crossing.
		total   += ((1-np.sin(dec*utils.degree)) * dra).sum()*utils.degree
	return abs(total)

def pixsize(shape, wcs):
	"""Returns the average pixel area, in steradians."""
	return area(shape, wcs)/np.prod(shape[-2:])

def pixshape(shape, wcs, signed=False):
	"""Returns the average pixel height and width, in radians."""
	return extent(shape, wcs, signed=signed)/shape[-2:]

def pixshapemap(shape, wcs, bsize=1000, separable="auto", signed=False, bcheck=False):
	"""Returns the physical width and heigh of each pixel in the map in radians.
	Heavy for big maps. Much faster approaches are possible for known pixelizations."""
	if wcsutils.is_plain(wcs):
		cdelt = wcs.wcs.cdelt
		pshape  = np.zeros([2])
		pshape[0] = wcs.wcs.cdelt[1]*get_unit(wcs)
		pshape[1] = wcs.wcs.cdelt[0]*get_unit(wcs)
		if not signed: pshape = np.abs(pshape)
		pshape  = np.broadcast_to(pshape[:,None,None], (2,)+shape[-2:])
		return ndmap(pshape, wcs)
	elif separable == True or (separable == "auto" and wcsutils.is_separable(wcs)):
		pshape = pixshapes_cyl(shape, wcs, signed=signed, bcheck=bcheck)
		pshape = np.broadcast_to(pshape[:,:,None], (2,)+shape[-2:])
		return ndmap(pshape, wcs)
	else:
		pshape = zeros((2,)+shape[-2:], wcs)
		# Loop over blocks in y to reduce memory usage
		for i1 in range(0, shape[-2], bsize):
			i2 = min(i1+bsize, shape[-2])
			pix  = np.mgrid[i1:i2+1,:shape[-1]+1]
			with utils.nowarn():
				y, x = pix2sky(shape, wcs, pix, safe=True, corner=True, bcheck=bcheck)
			del pix
			dy = y[1:,1:]-y[:-1,:-1]
			dx = x[1:,1:]-x[:-1,:-1]
			if not signed: dy, dx = np.abs(dy), np.abs(dx)
			cy = np.cos(y)
			bad= cy<= 0
			cy[bad] = np.mean(cy[~bad])
			dx *= 0.5*(cy[1:,1:]+cy[:-1,:-1])
			del y, x, cy
			# Due to wcs fragility, we may have some nans at wraparound points.
			# Fill these with the mean non-nan value. Since most maps will be cylindrical,
			# it makes sense to do this by row
			bad = ~np.isfinite(dy)
			dy[bad] = np.mean(dy[~bad])
			bad = ~np.isfinite(dx)
			dx[bad] = np.mean(dx[~bad])
			# Copy over to our output array
			pshape[0,i1:i2,:] = dy
			pshape[1,i1:i2,:] = dx
			del dx, dy
		return pshape

def pixshapes_cyl(shape, wcs, signed=False, bcheck=False):
	"""Returns the physical width and height of pixels for each row of a cylindrical
	map with the given shape, wcs, in radians, as an array [{height,width},ny]. All
	pixels in a row have the same shape in a cylindrical projection."""
	res = np.zeros([2,shape[-2]])
	ny  = shape[-2]
	# Get the dec of all the pixel edges, and use that to get the heights.
	y   = np.arange(ny+1)-0.5
	x   = y*0
	dec, ra = pix2sky(shape, wcs, [y,x], safe=False, bcheck=bcheck)
	if not np.isfinite(dec[0]):  dec[0]  = -np.pi/2 if wcs.wcs.cdelt[1] >= 0 else  np.pi/2
	if not np.isfinite(dec[-1]): dec[-1] =  np.pi/2 if wcs.wcs.cdelt[1] >= 0 else -np.pi/2
	dec = np.clip(dec, -np.pi/2, np.pi/2)
	heights = dec[1:]-dec[:-1]
	# A pixel that goes from dec1 to dec2 with a RA interval of dRA has an area of
	# (sin(dec2)-sin(dec1))*dRA. We will assign a width of area/height to each pixel,
	# such that we can simply define pixsize in terms of pixshape. That is,
	# width = dRA * (sin(dec2)-sin(dec1))/(dec2-dec1)
	dRA   = wcs.wcs.cdelt[0]*utils.degree
	sdec  = np.sin(dec)
	widths= dRA * (sdec[1:]-sdec[:-1])/heights
	res[0] = heights
	res[1] = widths
	if not signed:
		res = np.abs(res)
	return res

def pixsizemap(shape, wcs, separable="auto", broadcastable=False, bsize=1000, bcheck=False):
	"""Returns the physical area of each pixel in the map in steradians.

	If separable is True, then the map will be assumed to be in a cylindircal
	projection, where the pixel size is only a function of declination.
	This makes the calculation dramatically faster, and the resulting array
	will also use much less memory due to numpy striding tricks. The default,
	separable=auto", determines whether to use this shortcut based on the
	properties of the wcs.

	Normally the function returns a ndmap of shape [ny,nx]. If broadcastable
	is True, then it is allowed to return a smaller array that would still
	broadcast correctly to the right shape. This can be useful to save work
	if one is going to be doing additional manipulation of the pixel size
	before using it. For a cylindrical map, the result would have shape [ny,1]
	if broadcastable is True.
	"""
	if wcsutils.is_plain(wcs):
		return full(shape[-2:], wcs, np.abs(wcs.wcs.cdelt[0]*wcs.wcs.cdelt[1])*utils.degree**2)
	elif separable == True or (separable == "auto" and wcsutils.is_separable(wcs)):
		psize = np.prod(pixshapes_cyl(shape, wcs, bcheck=bcheck),0)[:,None]
		# Expand to full shape unless we are willing to accept an array
		# with smaller size that is still broadcastable to the right result
		if not broadcastable:
			psize = np.broadcast_to(psize, shape[-2:])
		return ndmap(psize, wcs)
	else:
		return pixsizemap_contour(shape, wcs, bsize=bsize, bcheck=bcheck)

def pixsizemap_contour(shape, wcs, bsize=1000, bcheck=False):
	# Loop to save memory. Numba-candidate?
	psizes = zeros(shape[-2:], wcs)
	for y1 in range(0, shape[-2], bsize):
		y2   = min(y1+bsize, shape[-2])
		# Get the pixel coordinates our pixels' corners, and
		# turn them into dec,ra
		pixs = np.mgrid[y1:y2+1,:shape[-1]+1]-0.5
		poss = pix2sky(shape, wcs, pixs, bcheck=bcheck)
		del pixs
		# Avoid impossible coordinates
		poss[0] = np.clip(poss[0], -np.pi/2, np.pi/2)
		dec, ra = poss
		# Integrate (1-sin(dec))*dRA from [0,0] to [1,0], using the
		# average msin value along this path
		msin   = 1-np.sin(dec)
		areas  = (ra[ 1:,:-1]-ra[:-1,:-1])*(msin[ 1:,:-1]+msin[:-1,:-1])/2
		# [1,0] to [1,1]
		areas += (ra[ 1:, 1:]-ra[ 1:,:-1])*(msin[ 1:, 1:]+msin[ 1:,:-1])/2
		# [1,1] to [0,1]
		areas += (ra[:-1, 1:]-ra[ 1:, 1:])*(msin[:-1, 1:]+msin[ 1:, 1:])/2
		# [0,1] to [0,0]
		areas += (ra[:-1,:-1]-ra[:-1, 1:])*(msin[:-1,:-1]+msin[:-1, 1:])/2
		psizes[y1:y2] = np.abs(areas)
	return psizes

def pixshapebounds(shape, wcs, separable="auto"):
	"""Return the minimum and maximum pixel height and width for the given
	geometry, in the form [{min,max},{y,x}]. Fast for separable geometries like
	cylindrical ones, which it will try to recognize, but this can be forced by
	setting separable to True (or disabled with False). Heavy in the general case."""
	if separable == True or (separable == "auto" and wcsutils.is_separable(wcs)):
		return utils.minmax(pixshapes_cyl(shape, wcs),1)
	else:
		return utils.minmax(pixshapemap(shape, wcs))

def lmap(shape, wcs, oversample=1, method="auto"):
	"""Return a map of all the wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	ly, lx = laxes(shape, wcs, oversample=oversample, method=method)
	data = np.empty((2,ly.size,lx.size))
	data[0] = ly[:,None]
	data[1] = lx[None,:]
	return ndmap(data, wcs)

def modlmap(shape, wcs, oversample=1, method="auto", min=0):
	"""Return a map of all the abs wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	slmap = lmap(shape,wcs,oversample=oversample, method=method)
	l = np.sum(slmap**2,0)**0.5
	if min > 0: l = np.maximum(l, min)
	return l

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
	if wcsutils.is_plain(wcs): return np.sum((slmap-ref)**2,0)**0.5
	return ndmap(utils.angdist(slmap[::-1],ref[::-1],zenith=False),wcs)

def laxes(shape, wcs, oversample=1, method="auto", broadcastable=False):
	oversample = int(oversample)
	step = extent(shape, wcs, signed=True, method=method)/shape[-2:]

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
	if broadcastable: ly, lx = ly[:,None], lx[None,:]
	return ly, lx

def lrmap(shape, wcs, oversample=1):
	"""Return a map of all the wavenumbers in the fourier transform
	of a map with the given shape and wcs."""
	return lmap(shape, wcs, oversample=oversample)[...,:shape[-1]//2+1]

def lpixsize(shape, wcs, signed=False, method="auto"):
	return np.prod(lpixshape(shape, wcs, signed=signed, method=method))

def lpixshape(shape, wcs, signed=False, method="auto"):
	return 2*np.pi/extent(shape,wcs, signed=signed, method=method)

def fft(emap, omap=None, nthread=0, normalize=True, adjoint_ifft=False, dct=False):
	"""Performs the 2d FFT of the enmap pixels, returning a complex enmap.
	If normalize is "phy", "phys" or "physical", then an additional normalization
	is applied such that the binned square of the fourier transform can
	be directly compared to theory (apart from mask corrections)
	, i.e., pixel area factors are corrected for.
	"""
	if dct: res  = samewcs(enfft.dct(emap,omap,axes=[-2,-1],nthread=nthread), emap)
	else:   res  = samewcs(enfft.fft(emap,omap,axes=[-2,-1],nthread=nthread), emap)
	norm = 1
	if normalize:
		if dct: norm /= np.prod(2*np.array(emap.shape[-2:])-1)**0.5
		else:   norm /= np.prod(emap.shape[-2:])**0.5
	if normalize in ["phy","phys","physical"]:
		if adjoint_ifft: norm /= emap.pixsize()**0.5
		else:            norm *= emap.pixsize()**0.5
	if norm != 1: res *= norm
	return res
def ifft(emap, omap=None, nthread=0, normalize=True, adjoint_fft=False, dct=False):
	"""Performs the 2d iFFT of the complex enmap given, and returns a pixel-space enmap."""
	if dct: res  = samewcs(enfft.idct(emap,omap,axes=[-2,-1],nthread=nthread, normalize=False), emap)
	else:   res  = samewcs(enfft.ifft(emap,omap,axes=[-2,-1],nthread=nthread, normalize=False), emap)
	norm = 1
	if normalize:
		if dct: norm /= np.prod(2*np.array(emap.shape[-2:])-1)**0.5
		else:   norm /= np.prod(emap.shape[-2:])**0.5
	if normalize in ["phy","phys","physical"]:
		if adjoint_fft: norm *= emap.pixsize()**0.5
		else:           norm /= emap.pixsize()**0.5
	if norm != 1: res *= norm
	return res

def dct(emap, omap=None, nthread=0, normalize=True):
	return fft(emap, omap=omap, nthread=nthread, normalize=normalize, dct=True)
def idct(emap, omap=None, nthread=0, normalize=True):
	return ifft(emap, omap=omap, nthread=nthread, normalize=normalize, dct=True)

def fft_adjoint(emap, omap=None, nthread=0, normalize=True):
	return ifft(emap, omap=omap, nthread=nthread, normalize=normalize, adjoint_fft=True)
def ifft_adjoint(emap, omap=None, nthread=0, normalize=True):
	return fft(emap, omap=omap, nthread=nthread, normalize=normalize, adjoint_ifft=True)

def idct_adjoint(emap, omap=None, nthread=0, normalize=True):
	return fft(emap, omap=omap, nthread=nthread, normalize=normalize, adjoint_ifft=True, dct=True)
def dct_adjoint(emap, omap=None, nthread=0, normalize=True):
	return ifft(emap, omap=omap, nthread=nthread, normalize=normalize, adjoint_fft=True, dct=True)

# These are shortcuts for transforming from T,Q,U real-space maps to
# T,E,B hamonic maps. They are not the most efficient way of doing this.
# It would be better to precompute the rotation matrix and buffers, and
# use real transforms.
def map2harm(emap, nthread=0, normalize=True, iau=False, spin=[0,2], adjoint_harm2map=False):
	"""Performs the 2d FFT of the enmap pixels, returning a complex enmap.
	If normalize starts with "phy" (for physical), then an additional normalization
	is applied such that the binned square of the fourier transform can
	be directly compared to theory  (apart from mask corrections)
	, i.e., pixel area factors are corrected for.
	"""
	emap = samewcs(fft(emap,nthread=nthread,normalize=normalize, adjoint_ifft=adjoint_harm2map), emap)
	if emap.ndim > 2:
		rot, s0 = None, None
		for s, i1, i2 in spin_helper(spin, emap.shape[-3]):
			if s == 0:  continue
			if s != s0: s0, rot = s, queb_rotmat(emap.lmap(), iau=iau, spin=s)
			emap[...,i1:i2,:,:] = map_mul(rot, emap[...,i1:i2,:,:])
	return emap
def harm2map(emap, nthread=0, normalize=True, iau=False, spin=[0,2], keep_imag=False, adjoint_map2harm=False):
	if emap.ndim > 2:
		emap = emap.copy()
		rot, s0 = None, None
		for s, i1, i2 in spin_helper(spin, emap.shape[-3]):
			if s == 0:  continue
			if s != s0: s0, rot = s, queb_rotmat(emap.laxes(broadcastable=True), iau=iau, spin=s, inverse=True, wcs=emap.wcs)
			emap[...,i1:i2,:,:] = map_mul(rot, emap[...,i1:i2,:,:])
	res = samewcs(ifft(emap,nthread=nthread,normalize=normalize, adjoint_fft=adjoint_map2harm), emap)
	if not keep_imag: res = res.real
	return res

def map2harm_adjoint(emap, nthread=0, normalize=True, iau=False, spin=[0,2], keep_imag=False):
	return harm2map(emap, nthread=nthread, normalize=normalize, iau=iau, spin=spin, keep_imag=keep_imag, adjoint_map2harm=True)

def harm2map_adjoint(emap, nthread=0, normalize=True, iau=False, spin=[0,2]):
	return map2harm(emap, nthread=nthread, normalize=normalize, iau=iau, spin=spin, adjoint_harm2map=True)

def queb_rotmat(lmap, inverse=False, iau=False, spin=2, wcs=None):
	# This corresponds to the Healpix convention. To get IAU,
	# flip the sign of a.
	sign = 1
	if iau:     sign = -sign
	if inverse: sign = -sign
	# More efficient to multiply in the sign before expanding to full map
	a    = spin*np.arctan2(sign*lmap[1],lmap[0])
	c, s = np.cos(a), np.sin(a)
	return samewcs(np.array([[c,-s],[s,c]]),lmap)

def rotate_pol(emap, angle, comps=[-2,-1], spin=2, axis=-3):
	"""Rotate the polarization of the given enmap "emap" by angle
	(in radians) along the given components (the last two by default)
	of the given axis (the 3rd-last axis by default). In standard enmaps
	the 3rd-last axis is holds the Stokes components of the map in the order
	T, Q, U. The spin argument controls the spin, and defaults to spin-2.
	This function is flexible enough to work with non-enmaps too."""
	if spin == 0: return emap
	axis %= emap.ndim
	c, s  = np.cos(spin*angle), np.sin(spin*angle)
	res   = emap.copy()
	pre   = (slice(None),)*axis
	res[pre+(comps[0],)] = c*emap[pre+(comps[0],)] - s*emap[pre+(comps[1],)]
	res[pre+(comps[1],)] = s*emap[pre+(comps[0],)] + c*emap[pre+(comps[1],)]
	return res

def map_mul(mat, vec):
	"""Elementwise matrix multiplication mat*vec. Result will have
	the same shape as vec. Multiplication happens along the last non-pixel
	indices."""
	# Allow scalar product, broadcasting if necessary
	mat = np.asanyarray(mat)
	if mat.ndim <= 3: return mat*vec
	# Otherwise we do a matrix product along the last axes
	ovec = samewcs(np.einsum("...abyx,...byx->...ayx", mat, vec), mat, vec)
	return ovec

def smooth_gauss(emap, sigma):
	"""Smooth the map given as the first argument with a gaussian beam
	with the given standard deviation sigma in radians. If sigma is negative,
	then the complement of the smoothed map will be returned instead (so
	it will be a highpass filter)."""
	if np.all(sigma == 0): return emap.copy()
	f  = fft(emap)
	x2 = np.sum(emap.lmap()**2*sigma**2,0)
	if sigma >= 0: f *= np.exp(-0.5*x2)
	else:          f *= 1-np.exp(-0.5*x2)
	return ifft(f).real

def inpaint(map, mask, method="nearest"):
	"""Inpaint regions in emap where mask==True based on the nearest unmasked pixels.
	Uses scipy.interpolate.griddata internally. See its documentation for the meaning of
	method. Note that if method is not "nearest", then areas where the mask touches the edge
	will be filled with NaN instead of sensible values.

	The purpose of this function is mainly to allow inapinting bad values with a
	continuous signal with the right order of magnitude, for example to allow fourier
	operations of masked data with large values near the edge of the mask (e.g. a
	galactic mask). Its goal is not to inpaint with something realistic-looking. For
	that heavier methods are needed.

	FIXME: This function is slow and not very good. Fix or remove.
	"""
	from scipy import interpolate
	# Find innermost good pixels at border of mask. These are the pixels the interpolation
	# will actually be based on, so isolating them makes things much faster than just sending
	# in every valid pixel
	border = scipy.ndimage.distance_transform_edt(~mask)==1
	pix      = map.pixmap()
	pix_good = pix[:,border].reshape(2,-1).T
	pix_bad  = pix[:,mask].reshape(2,-1).T
	if pix_good.size == 0: return map*0
	omap = map.copy()
	# Loop over each scalar component of omap
	for m in omap.preflat:
		val_good = m[border].reshape(-1)
		val_ipol = interpolate.griddata(pix_good, val_good, pix_bad, method=method)
		m[pix_bad[:,0],pix_bad[:,1]] = val_ipol
	return omap

def calc_window(shape, order=0, scale=1):
	"""Compute fourier-space pixel window function. Since the
	window function is separable, it is returned as an x and y part,
	such that window = wy[:,None]*wx[None,:]. By default the pixel
	window for interpolation order 0 mapmaking (nearest neighbor)
	is returned. Pass 1 for bilinear mapmaking's pixel window.
	The scale argument can be used to calculate the pixel window
	at non-native resolutions. For example, with scale=2 you will
	get the pixwin for a map with twice the resolution"""
	wy = utils.pixwin_1d(np.fft.fftfreq(shape[-2], scale), order=order)
	wx = utils.pixwin_1d(np.fft.fftfreq(shape[-1], scale), order=order)
	return wy, wx

def apply_window(emap, pow=1.0, order=0, scale=1, nofft=False):
	"""Apply the pixel window function to the specified power to the map,
	returning a modified copy. Use pow=-1 to unapply the pixel window.
	By default the pixel window for interpolation order 0 mapmaking
	(nearest neighbor) is applied. Pass 1 for bilinear mapmaking's pixel window."""
	wy, wx = calc_window(emap.shape, order=order, scale=scale)
	if not nofft: emap = fft(emap)
	else:         emap = emap.copy()
	emap *= wy[:,None]**pow
	emap *= wx[None,:]**pow
	if not nofft: emap = ifft(emap).real
	return emap

def unapply_window(emap, pow=1.0, order=0, scale=1, nofft=False):
	"""The inverse of apply_window. Equivalent to just flipping the sign of the pow argument."""
	return apply_window(emap, pow=-pow, order=order, scale=scale, nofft=nofft)

def samewcs(arr, *args):
	"""Returns arr with the same wcs information as the first enmap among
	args.  If no matches are found, arr is returned as is.  Will
	reference, rather than copy, the underlying array data
	whenever possible.
	"""
	for m in args:
		try: return ndmap(arr, m.wcs)
		except AttributeError: pass
	return arr

# Work-in progress potential replacement for geometry.
# Might be better to not try to replace all of geometry's functionality,
# though, and keep geometry-as-window-into-fullsky and geometry-as-independent-patch
# separate
def geometry2(pos=None, res=None, shape=None, proj="car", variant=None,
			deg=False, pre=(), ref=None, **kwargs):
	"""Construct a shape,wcs pair suitable for initializing enmaps. Some combination
	of pos, res and shape must be passed:

	* Only res given: A fullsky geometry with this resolution is constructed
	* Only shape given: Fullsky geometry with the given shape. This may result
	  in different resolution in the y and x directions if the shape is not
	  chosen carefully
	* pos and res given: A fullsky geometry with the given resolution is
	  constructed, and is then cropped using pos. pos must be [{from,to},{dec,ra}],
	  which specifies the corners of the geometry (to within a pixel).
	* pos, res and shape given: As previous, but pos is just [{dec,ra}] and
	  specifies the center of the geometry (to within a pixel), with the shape
	  being given by shape

	Other combinations are not supported. The other arguments are:

	* proj: The projection to use. Either name or name:variant, where
	  name is a WCS projection name like "CAR" (case-insensitive) and
	  variant is a pixelization variant (see next)
	* variant: A pixelization variant. Specifies how to split the full sky
	  into pixels. Valid values:
	  * "safe": pixel edge at all domain edges, e.g. both poles in dec and
	    wraparound point in ra for cylindricatl projections. This rule is
	    downgrade-safe, meaning that constructing a geometry and then
	    downgrading it will produce the same geometry as directly constructing
	    it at the downgraded resolution. This has the Fejer1 quadrature
	    rules, but has a different name to distinguish it from our original
	    implementation of Fejer1. Equivalent to the rule "hh,hh".
	  * "fejer1": pixel edge at top/bottom domain edges, but pixel center
	    at left/right domain edges. Follows the Fejer1 quadrature rules.
	    Only partially downgrade-safe: Can still SHT after downgrading, but
	    is not pixel-compatible with directly constructing a geometry at
	    the target resolution. Equivalent to the rule "00,hh".
	    This is the default, but it may be changed to "safe" in the future.
	  * "cc": pixel center at all domain edges. Follows the Clenshaw-Curtis
	    quadrature rules. Not downgrade-safe at all - cannot efficiently
	    SHT after downgrade. Equivalent to the rule "00,00".
	  * "any": No restriction on pixel center/edge placement. Allows for
	    arbitrary-resolution maps, but not fast SHTs. Equivalent to the
	    rule "**,**".
	  * A rule of the form "lr,bt", where l,r,b,t specify the pixel placement
	    for the left, right, bottom and top domain edges respectively. Each
	    can take the values
	    * "0": pixel center here (0 offset)
	    * "h": pixel edge here (half-pixel offset)
	    * "*": no restriction
	* deg: If True (defaults to False), then pos, res and ref have units of
	  degrees instead of radians.
	* pre: Tuple of pre-dimensions to prepend to the returned shape
	* ref: Reference point coordinates. This is the point the full geometry
	  is built around. E.g. for Mollweide it would be the center of the
	  Mollweide ellipse. Changing this point actually changes the projection,
	  and projections with different ref will be incompatible with each other
	  aside from special cases. Defaults to ra=dec=0 for non-azimuthal projections
	  and ra=0, dec=pi/2 for azimuthal projections. The special value "mid"
	  will use pos to set the reference point. Should probably be left alone
	  unless you have special requirements."""
	unit  = utils.degree if deg else 1
	# First build the base projection
	system, variant = wcsutils.parse_system(proj, variant=variant)
	crval = _geometry_crval(ref, pos, unit)
	#crval = None if ref is None else np.array(ref)[::-1]*unit/utils.degree
	pwcs  = wcsutils.projection(system, crval=crval)
	# Determine the pixelization. If pos is None, we will be making
	# a fullsky geometry, which is simple
	if pos is None:
		if res is not None:
			res = wcsutils.expand_res(res,flip=True)*unit/utils.degree
		oshape, owcs = wcsutils.pixelization(pwcs, shape=shape, res=res, variant=variant)
	else:
		if res is None:
			raise ValueError("geometry construction with just pos and shape is not supported. Pass in the target resolution")
		pos = np.asarray(pos)*unit
		# The coordinate order of the pos box affects the sign of res
		if pos.ndim == 1: psign = [1,-1]
		else: psign = np.sign(pos[1]-pos[0])
		res = wcsutils.expand_res(res,signs=psign,flip=True)*unit/utils.degree
		oshape, owcs = wcsutils.pixelization(pwcs, res=res, variant=variant)
		oshape, owcs = crop_geometry(oshape, owcs, box=pos, oshape=shape)
	# Add the pre-dimensions
	oshape = pre + oshape
	# For cylindrical projections we can move crval along the equator
	# without changing the meaning of the projection, but only if crval
	# is already on the equator. This avoids problems with invalid coordinates
	# at more than 180° away from the reference point. It's not possible
	# to do this for other projections because they actually change when crval
	# is moved (e.g. for mollweide it changes what point the ellipse is centered on)
	if wcsutils.is_cyl(owcs) and owcs.wcs.crval[1] == 0:
		# This choice keeps us compatible with the old fullsky_geometry implementation
		imid = np.array(oshape[-2:])//2-0.5
		vmid = pix2sky(oshape, owcs, imid)
		owcs.wcs.crval[0] = vmid[1]/utils.degree
		owcs.wcs.crpix[0] = imid[1]+1
	return oshape, owcs

def _geometry_crval(ref=None, pos=None, unit=1):
	if ref is None: return None
	if isinstance(ref, str) and ref == "mid":
		if pos is None: return None
		pos = np.asarray(pos)
		if   pos.shape == (2,):  return pos[::-1]*unit/utils.degree
		elif pos.shape == (2,2): return np.mean(pos,0)[::-1]*unit/utils.degree
		else: raise ValueError("pos must be [{from,to},{dec,ra}] or [{dec,ra}]")
	else:
		return np.array(ref)[::-1]*unit/utils.degree

def fullsky_geometry2(res=None, shape=None, pre=None, deg=False, proj="car", variant=None, dims=None):
	"""Build a fullsky geometry with the given resolution or shape. Simply forwards to
	geometry(). See its docstring for the meaning of the arguments.

	dims is an alias for pre provided for backwards compatibility"""
	return geometry2(res=res, shape=shape, deg=deg, pre=pre or dims or (), proj=proj, variant=variant)

def band_geometry2(decrange, res=None, shape=None, pre=None, deg=False, proj="car", variant=None, dims=None):
	"""Build a geometry covering a range of declinations. Equivalent to
	geometry(pos=[[decrange[0],pi],[decrange[1],-pi]], ...). See geometry's documentation
	for the meaning of the other arguments.

	dims is an alias for pre provided for backwards compatibility"""
	unit     = utils.degree if deg else 1
	decrange = (np.zeros(2)+decrange)*unit
	if decrange.shape != (2,): raise ValueError("decrange must be a number or (dec1,dec2)")
	pos      = np.array([[decrange[0],np.pi],[decrange[1],-np.pi]])/unit
	return geometry2(pos=pos, res=res, shape=shape, deg=deg, pre=pre or dims or (), proj=proj, variant=variant)

# Idea: Make geometry a class with .shape and .wcs members.
# Make a function that takes (foo,bar) and returns a geometry,
# there (foo,bar) can either be (shape,wcs) or (geometry,None).
# Use that to make everything that currently accepts shape, wcs
# transparently accept geometry. This will free us from having
# to drag around a shape, wcs pair all the time.
def geometry(pos, res=None, shape=None, proj="car", variant="cc", deg=False, pre=(), force=False, ref=None, **kwargs):
	"""Consruct a shape,wcs pair suitable for initializing enmaps.
	pos can be either a {dec,ra} center position or a [{from,to},{dec,ra}]
	array giving the bottom-left and top-right corners of the desired geometry.
	At least one of res or shape must be specified.  If res is specified, it
	must either be a number, in which the same resolution is used in each direction,
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
	resulting corners can differ by a fraction of a pixel from the one requested.
	To force the geometry to exactly match the corners provided you can pass force=True.
	It is also possible to manually choose the reference point via the ref argument, which
	must be a dec,ra coordinate pair (in radians)."""
	# TODO: This function should be generalized to support fejer1.
	# This is problematic because we can't assume that ra=0,dec=0 will be a pixel
	# center in a Fejer1 geometry. Actually we can't even assume that for CC.
	# For both cases it will happen for odd ny, but odd ny is normal for CC but
	# rare for fejer1. For fejer1 the norm will instead be to have a pixel edge
	# at ra=0,dec=0. In general the safest approach is to at least conceptually
	# start from a standardized fullsky geometry and then crop it to the target,
	# rather than to start from a standard point on the sky and then grow it
	# to the required size. This will require a complete rework of this function
	# though.
	assert variant == "cc"

	# We use radians by default, while wcslib uses degrees, so need to rescale.
	# The exception is when we are using a plain, non-spherical wcs, in which case
	# both are unitless. So undo the scaling in this case.
	scale = 1 if deg else 1/utils.degree
	pos = np.asarray(pos)*scale
	if res is not None: res = np.asarray(res)*scale
	# Apply a standard reference points unless one is manually specified, or we
	# want to force the corners to exactly match the input.
	try:
		# if it's a (dec,ra) tuple in radians, make it (ra,dec) in degrees.
		ref = (ref[1] * scale, ref[0] * scale)
		assert(len(ref) == 2)
	except (TypeError, ValueError):
		pass
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
		shape = tuple(np.round(np.abs(faredge-nearedge)).astype(int))
	return pre+tuple(shape), wcs

def fullsky_geometry(res=None, shape=None, dims=(), proj="car", variant="fejer1"):
	"""Build an enmap covering the full sky, with the outermost pixel centers
	at the poles and wrap-around points. Only the car projection is
	supported for now, but the variants CC and fejer1 can be selected using
	the variant keyword. This currently defaults to CC, but will likely
	change to fejer1 in the future."""
	assert proj == "car", "Only CAR fullsky geometry implemented"
	# Handle the CAR variants
	if   variant.lower() == "cc":     yo = 1
	elif variant.lower() == "fejer1": yo = 0
	else: raise ValueError("Unrecognized CAR variant '%s'" % str(variant))
	# Set up the shape/resolution
	if shape is None:
		res   = np.zeros(2)+res
		shape = utils.nint(([1*np.pi,2*np.pi]/res) + (yo,0))
	else:
		res = np.array([1*np.pi,2*np.pi])/(np.array(shape)-(yo,0))
	ny, nx = shape
	assert abs(res[0]*(ny-yo)-  np.pi) < 1e-8, "Vertical resolution does not evenly divide the sky; this is required for SHTs."
	assert abs(res[1]*nx     -2*np.pi) < 1e-8, "Horizontal resolution does not evenly divide the sky; this is required for SHTs."
	wcs   = wcsutils.WCS(naxis=2)
	# Note the reference point is shifted by half a pixel to keep
	# the grid in bounds, from ra=180+cdelt/2 to ra=-180+cdelt/2.
	wcs.wcs.crval = [res[1]/2/utils.degree,0]
	wcs.wcs.cdelt = [-360./nx,180./(ny-yo)]
	wcs.wcs.crpix = [nx//2+0.5,(ny+1)/2]
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	return dims+(ny,nx), wcs

def band_geometry(dec_cut, res=None, shape=None, dims=(), proj="car", variant="fejer1"):
	"""Return a geometry corresponding to a sky that had a full-sky
	geometry but to which a declination cut was applied. If dec_cut
	is a single number, the declination range will be (-dec_cut,dec_cut)
	radians, and if specified with two components, it is interpreted as
	(dec_cut_min,dec_cut_max). The remaining arguments are the same as
	fullsky_geometry and pertain to the geometry before cropping to the
	cut-sky.
	"""
	dec_cut = np.atleast_1d(dec_cut)
	if dec_cut.size == 1:
		dec_cut_min = -dec_cut[0]
		dec_cut_max = dec_cut[0]
		assert dec_cut_max>0
	elif dec_cut.size == 2:
		dec_cut_min,dec_cut_max = dec_cut
		assert dec_cut_max>dec_cut_min
	else:
		raise ValueError
	ishape,iwcs = fullsky_geometry(res=res, shape=shape, dims=dims, proj=proj, variant=variant)
	start = sky2pix(ishape,iwcs,(dec_cut_min,0))[0]
	stop = sky2pix(ishape,iwcs,(dec_cut_max,0))[0]
	Ny,_ = ishape[-2:]
	start = max(int(np.round(start)),0); stop = min(int(np.round(stop)),Ny)
	assert start>=0 and start<Ny
	assert stop>=0 and stop<Ny
	return slice_geometry(ishape,iwcs,np.s_[start:stop,:])

def thumbnail_geometry(r=None, res=None, shape=None, dims=(), proj="tan"):
	"""Build a geometry in the given projection centered on (0,0), which will
	be exactly at a pixel center.

	 r:     The radius from the center to the edges of the patch, in radians.
	 res:   The resolution of the patch, in radians.
	 shape: The target shape of the patch. Will be forced to odd numbers if necessary.

	Any two out of these three arguments must be specified. The most common usage
	will probably be to specify r and res, e.g.
	 shape, wcs = enmap.thumbnail_geometry(r=1*utils.degree, res=0.5*utils.arcmin)

	The purpose of this function is to provide a geometry appropriate for object
	stacking, etc. Ideally enmap.geometry would do this, but this specialized function
	makes it easier to ensure that the center of the coordinate system will be at
	excactly the pixel index (y,x) = shape//2+1, which was a commonly requested feature
	(even though which pixel is at the center shouldn't really matter as long as one
	takes into account the actual coordinates of each pixel).
	"""
	if wcsutils.is_plain(proj):
		ctype = ["",""]
		dirs  = [1,1]
	else:
		ctype = ["RA---%s" % proj.upper(), "DEC--%s" % proj.upper()]
		dirs  = [1,-1]
	if r is None: # res and shape given
		assert res is not None and shape is not None, "Two of r, res and shape must be given"
		res   = wcsutils.expand_res(res, dirs)
		shape = utils.nint(np.zeros(2)+shape[-2:]) # Broadcast and make sure it's an integer
		shape = shape//2*2+1                       # Force odd shape
		wcs   = wcsutils.explicit(ctype=ctype, crval=[0,0], cdelt=res[::-1]/utils.degree, crpix=shape[::-1]//2+1)
	elif shape is None: # res and r given
		assert res is not None and r is not None, "Two of r, res and shape must be given"
		res   = wcsutils.expand_res(res, dirs)
		r     = np.zeros(2)+r
		wcs   = wcsutils.explicit(ctype=ctype, crval=[0,0], cdelt=res[::-1]/utils.degree, crpix=[1,1])
		rpix  = utils.nint(np.abs(wcsutils.nobcheck(wcs).wcs_world2pix(r[None,::-1]/utils.degree,0)[0,::-1]))
		shape = 2*rpix+1
		wcs.wcs.crpix = shape[::-1]//2+1
	else: # r and shape given
		assert r is not None and shape is not None, "Two of r, res and shape must be given"
		shape = utils.nint(np.zeros(2)+shape[-2:]) # Broadcast and make sure it's an integer
		shape = shape//2*2+1                       # Force odd shape
		r     = np.zeros(2)+r
		wcs   = wcsutils.explicit(ctype=ctype, crval=[0,0], crpix=[1,1])
		rpix  = np.abs(wcsutils.nobcheck(wcs).wcs_world2pix(r[None,::-1]/utils.degree,0)[0,::-1])
		res_ratio = (shape-1)/(2*rpix)*dirs
		wcs.wcs.cdelt /= res_ratio[::-1]
		wcs.wcs.crpix  = shape[::-1]//2+1
	shape = dims+tuple(shape)
	return shape, wcs

def union_geometry(geometries):
	"""Given a list of compatible geometries, return a new geometry that's the union
	if the inputs, in the sense that it contains all the pixels that the individual ones
	contain"""
	ref      = geometries[0]
	pixboxes = [pixbox_of(ref[1],shape,wcs) for shape, wcs in geometries]
	bbox     = utils.bounding_box(pixboxes)
	oshape   = tuple(bbox[1]-bbox[0])
	owcs     = ref[1].deepcopy()
	owcs.wcs.crpix -= bbox[0,::-1]
	return oshape, owcs

def recenter_cyl(shape, wcs):
	"""Given a cylindrical geometry with the equator horizontal, move
	the reference point to the point along the equator closest to the middle
	of the geometry. This is useful when deriving a sub-geometry that strattles
	the wrap-around point of another geometry"""
	return shape, wcsutils.recenter_cyl_x(wcs, (shape[-1]-1)/2+1)

def recenter_geo(shape, wcs, mode="auto"):
	"""Return either a recentered geometry or the original one, depending
	on the value of mode, which accepts three values:
	* True: Always attempts to recenter. Will fail if it's a non-cylindrical geometry
	* False: Leaves the geometry as it is
	* "auto": Recenters only if it's a cylindrical geometry"""
	if mode == "auto": mode = wcsutils.is_separable(wcs)
	if mode not in [True,False]: raise ValueError("Mode must be 'auto', True or False")
	if mode: return recenter_cyl(shape, wcs)
	else: return shape, wcs

def create_wcs(shape, box=None, proj="cea"):
	"""Very old function. Do not use"""
	if box is None:
		box = np.array([[-1,-1],[1,1]])*0.5*10
		box *= utils.degree
	return wcsutils.build(box, shape=shape, rowmajor=True, system=proj)

def spec2flat(shape, wcs, cov, exp=1.0, border="constant", oversample=1, smooth="auto"):
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
	cov    = np.asarray(cov)
	oshape = cov.shape[:-1] + tuple(shape)[-2:]
	if cov.ndim == 1: cov = cov[None,None]
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
	# Use order 1 because we will perform very short interpolation, and to avoid negative
	# values in spectra that must be positive (and it's faster)
	res = ndmap(utils.interpol(cov, np.reshape(ls,(1,)+ls.shape),border=border, order=1),wcs)
	res = downgrade(res, oversample)
	res = res.reshape(oshape[:-2]+res.shape[-2:])
	return res

def spec2flat_corr(shape, wcs, cov, exp=1.0, border="constant"):
	cov    = np.asarray(cov)
	oshape = cov.shape[:-1] + tuple(shape)[-2:]
	if cov.ndim == 1: cov = cov[None,None]
	if exp != 1.0: cov = multi_pow(cov, exp)
	cov[~np.isfinite(cov)] = 0
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
	corr2d = utils.interpol(corrfun, ipos.reshape((-1,)+ipos.shape), border=border, order=1)
	corr2d = np.roll(corr2d, -corr2d.shape[-2]//2, -2)
	corr2d = np.roll(corr2d, -corr2d.shape[-1]//2, -1)
	corr2d = ndmap(corr2d, wcs)
	return fft(corr2d).real * np.prod(shape[-2:])**0.5

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

def calc_ps2d(harm, harm2=None):
	"""Compute the 2d power spectrum of the harmonic-space enmap "harm", as output by
	map2harm. Use map2harm with norm="phys" to get physical units in this spectrum.
	If harm2 is specified, then the cross-spectrum between harm and harm2 is computed
	instead.

	Some example usage, where the notation a[{x,y,z},n,m] specifies that the array
	a has shape [3,n,m], and the 3 entries in the first axis should be interpreted
	as x, y and z respectively.

	1. cl[nl] = calc_ps2d(harm[ny,nx])
	   This just computes the standard power spectrum of the given harm, resulting in
	   a single 2d enmap.
	2. cl[nl] = calc_ps2d(harm1[ny,nx], harm2[ny,nx])
	   This compues the 1d cross-spectrum between the 2d enmaps harm1 and harm2.
	3. cl[{T,E,B},{T,E,B},nl] = calc_ps2d(harm[{T,E,B},None,ny,nx], harm[None,{T,E,B},ny,nx])
	   This computes the 3x3 polarization auto-spectrum for a 3d polarized harmonic enmap.
	4. cl[{T,E,B},{T,E,B},nl] = calc_ps2d(harm1[{T,E,B},None,ny,nx], harm2[None,{T,E,B},ny,nx])
	   As above, but gives the 3x3 polarization cross-spectrum between two 3d harmonic enmaps.

	The output is in the shape one would expect from numpy broadcasting. For example,
	in the last example, the TE power spectrum would be found in cl[0,1], and the
	ET power spectrum (which is different for the cross-spectrum case) is in cl[1,0]."""
	harm  = np.asanyarray(harm)
	harm2 = np.asanyarray(harm2) if harm2 is not None else harm
	# Unify dtypes
	dtype = np.result_type(harm.real, harm2.real)
	def getaddr(a): return a.__array_interface__["data"][0]
	harm, harm2 = [samewcs(a, harm) for a in np.broadcast_arrays(harm, harm2)]
	# We set the writable flags not because we intend to write, but to silience a
	# false positive warning from numpy
	harm.flags["WRITEABLE"] = harm2.flags["WRITEABLE"] = True
	# I used to flatten here to make looping simple, but that caused a copy to be made
	# when combined with np.broadcast. So instead I will use manual raveling
	pshape = harm.shape[:-2]
	npre   = int(np.prod(pshape))
	# A common use case is to compute TEBxTEB auto-cross spectra, where
	# e.g. TE === ET since harm1 is the same array as harm2. To avoid duplicate
	# calculations in this case we use a cache, which skips computing the
	# cross-spectrum of any given pair of arrays more than once.
	cache = {}
	ps2d = empty(harm.shape, harm.wcs, dtype)
	# We will loop over individual spectra
	for i in range(npre):
		I = np.unravel_index(i, pshape)
		# Avoid duplicate calculation
		key = tuple(sorted([getaddr(harm[I]), getaddr(harm2[I])]))
		if key in cache:
			ps2d[I] = cache[key]
		else:
			ps2d[I] = (harm[I]*np.conj(harm2[I])).real
			cache[key] = ps2d[I]
	return ps2d

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

def get_downgrade_offset(shape, wcs, factor, ref=None):
	"""Get the pixel offset required to keep a map downgraded by the given factor
	aligned with the reference point."""
	factor  = np.zeros(2, int)+factor
	if ref is None: return np.zeros(2,int)
	else:           return utils.nint(sky2pix(shape, wcs, ref))%factor

def downgrade(emap, factor, op=np.mean, ref=None, off=None, inclusive=False):
	"""Returns enmap "emap" downgraded by the given integer factor
	(may be a list for each direction, or just a number) by averaging
	inside pixels. Returns the original map if factor is None or 1."""
	if (factor is None): return emap
	factor  = np.zeros(2, int)+factor
	if np.all(factor==1): return emap
	# Optionally apply an offset to keep different downgraded maps pixel-compatible.
	# This can be either manually specified, or inferred from reference coordinates.
	if off is None:  off = get_downgrade_offset(emap.shape, emap.wcs, factor, ref)
	else:            off = np.zeros(2, int)+off
	# Do the averaging
	omap = utils.block_reduce(emap, factor[0], off=off[0], axis=-2, inclusive=inclusive, op=op)
	omap = utils.block_reduce(omap, factor[1], off=off[1], axis=-1, inclusive=inclusive, op=op)
	# Update the wcs
	wcs  = emap[...,off[0]::factor[0],off[1]::factor[1]].wcs
	wcs.wcs.crpix += (off[1::-1]>0)*inclusive # extra downgraded pixel in front if inclusive with offset
	omap = ndmap(omap, wcs)
	return omap

def downgrade_fft(emap, factor):
	"""Like downgrade(emap, factor), but uses fourier-resampling. This avoids
	introducing both a pixel window and aliasing, but assumes periodic boundary
	conditions."""
	return resample(emap, utils.nint(np.array(emap.shape[-2:])/factor))

def upgrade_fft(emap, factor):
	"""Like upgrade(emap, factor), but uses fourier-resampling. This avoids
	introducing any sharp edges, preserving the shape of the power spectrum."""
	return resample(emap, utils.nint(np.array(emap.shape[-2:])*factor))

def upgrade(emap, factor, off=None, oshape=None, inclusive=False):
	"""Upgrade emap to a larger size using nearest neighbor interpolation,
	returning the result. More advanced interpolation can be had using
	enmap.interpolate."""
	factor  = np.zeros(2, int)+factor
	off     = np.zeros(2,int)+(0 if off is None else off)
	# If no output shape specified, guess one that had no pixels left over at the end
	if oshape is None: oshape = (np.array(emap.shape[-2:])-(off>0)*inclusive)*factor+off
	omap = utils.block_expand(emap, factor[0], oshape[-2], off=off[0], axis=-2, inclusive=inclusive)
	omap = utils.block_expand(omap, factor[1], oshape[-1], off=off[1], axis=-1, inclusive=inclusive)
	# Correct the WCS information
	omap = ndmap(omap, emap.wcs.copy())
	for j in range(2):
		omap.wcs.wcs.crpix[j] -= 0.5 + (off[1-j]>0)*inclusive
		omap.wcs.wcs.crpix[j] *= factor[1-j]
		omap.wcs.wcs.cdelt[j] /= factor[1-j]
		omap.wcs.wcs.crpix[j] += 0.5 + off[1-j]
	return omap

def downgrade_geometry(shape, wcs, factor):
	"""Returns the oshape, owcs corresponding to a map with geometry
	shape, wcs that has been downgraded by the given factor. Similar
	to scale_geometry, but truncates the same way as downgrade, and only
	supports integer factors."""
	factor = np.full(2, 1, dtype=int)*factor
	oshape = tuple(shape[-2:]//factor)
	owcs   = wcsutils.scale(wcs, 1.0/factor, rowmajor=True)
	return oshape, owcs

def upgrade_geometry(shape, wcs, factor):
	return scale_geometry(shape, wcs, factor)

def crop_geometry(shape, wcs, box=None, pixbox=None, oshape=None, recenter=False):
	if pixbox is None:
		box    = np.asarray(box)
		# Allow box and pixbox to be 1d, in which case we will
		# crop around a central point
		if box.ndim == 2: pixbox = subinds(shape, wcs, box, cap=False)
		else:             pixbox = utils.nint(sky2pix(shape, wcs, box))
	# We assume that the box selects pixel edges, so any pixel that is
	# even partially inside the box should be included. This means that
	# pixel i would be included for i-0.5, but not for i-0.6. We should
	# thefore use rounding boundaries, we just have to make sure it's
	# numerically stable
	#print("box", box/utils.degree)
	#print("mid", np.mean(box,0)/utils.degree)
	#print("pixbox", pixbox)
	#pixbox = utils.nint(pixbox)
	#print("pixbox2", pixbox)
	pixbox = np.asarray(pixbox)
	# Handle 1d case
	if pixbox.ndim == 1:
		if oshape is None: raise ValueError("crop_geometry needs an explicit output shape when given a 1d box (i.e. a single point instead of a bounding box")
		shp    = np.array(oshape[-2:])
		pixbox = np.array([pixbox-shp//2,pixbox-shp//2+shp])
	#print("pixbox3", pixbox)
	# Can now proceed assuming 2d
	oshape = tuple(shape[:-2]) + tuple(np.abs(pixbox[1]-pixbox[0]))
	owcs   = wcs.deepcopy()
	owcs.wcs.crpix -= pixbox[0,::-1]
	if recenter: owcs = wcsutils.recenter_cyl_x(owcs, oshape[-1]//2)
	return oshape, owcs

def distance_transform(mask, omap=None, rmax=None, method="cellgrid"):
	"""Given a boolean mask, produce an output map where the value in each pixel is the distance
	to the closest false pixel in the mask. See distance_from for the meaning of rmax."""
	from pixell import distances
	if omap is None: omap = zeros(mask.shape, mask.wcs)
	for i in range(len(mask.preflat)):
		edge_pix = np.array(distances.find_edges(mask.preflat[i]))
		edge_pos = mask.pix2sky(edge_pix, safe=False)
		omap.preflat[i] = distance_from(mask.shape, mask.wcs, edge_pos, rmax=rmax, method=method)
	# Distance is always zero inside mask
	omap *= mask
	return omap

def labeled_distance_transform(labels, omap=None, odomains=None, rmax=None, method="cellgrid"):
	"""Given a map of labels going from 1 to nlabel, produce an output map where the value
	in each pixel is the distance to the closest nonzero pixel in the labels, as well as a
	map of which label each pixel was closest to. See distance_from for the meaning of rmax."""
	from pixell import distances
	if omap is None: omap = zeros(labels.shape, labels.wcs)
	if odomains is None: odomains = zeros(omap.shape, omap.wcs, np.int32)
	for i in range(len(labels.preflat)):
		edge_pix = np.array(distances.find_edges_labeled(labels.preflat[i]))
		edge_pos = labels.pix2sky(edge_pix, safe=False)
		_, domains = distance_from(labels.shape, labels.wcs, edge_pos, omap=omap.preflat[i], domains=True, rmax=rmax, method=method)
		# Get the edge_pix to label mapping
		mapping = labels.preflat[i][edge_pix[0],edge_pix[1]]
		mask    = domains >= 0
		odomains.preflat[i,mask] = mapping[domains[mask]]
		# Distance is always zero inside each labeled region
		mask    = labels.preflat[i] != 0
		omap.preflat[i][mask] = 0
	return omap, odomains

def distance_from(shape, wcs, points, omap=None, odomains=None, domains=False, method="cellgrid", rmax=None, step=1024):
	"""Find the distance from each pixel in the geometry (shape, wcs) to the
	nearest of the points[{dec,ra},npoint], returning a [ny,nx] map of distances.
	If domains==True, then it will also return a [ny,nx] map of the index of the point
	that was closest to each pixel. If rmax is specified and the method is "cellgrid" or "bubble", then
	distances will only be computed up to rmax. Beyond that distance will be set to rmax
	and domains to -1. This can be used to speed up the calculation when one only cares
	about nearby areas."""
	from pixell import distances
	if wcsutils.is_plain(wcs): warnings.warn("Distance functions are not tested on plain coordinate systems.")
	if omap is None: omap = empty(shape[-2:], wcs)
	if domains and odomains is None: odomains = empty(shape[-2:], wcs, np.int32)
	points = np.asarray(points)
	if points.ndim == 1: points = points[:,None]
	assert points.ndim == 2 and len(points) == 2, "points must be [{dec,ra},npoint]"
	# Handle case where no points are specified
	if points.size == 0:
		if rmax is None: rmax = np.inf
		omap[:] = rmax
		if domains: odomains[:] = -1
		return (omap, odomains) if domains else omap
	# Ok, we have at least one point, use the normal stuff
	if wcsutils.is_separable(wcs):
		dec, ra = posaxes(shape, wcs)
		if method == "bubble":
			point_pix = utils.nint(sky2pix(shape, wcs, points))
			return distances.distance_from_points_bubble_separable(dec, ra, points, point_pix, rmax=rmax, omap=omap, odomains=odomains, domains=domains)
		elif method == "cellgrid":
			point_pix = utils.nint(sky2pix(shape, wcs, points))
			return distances.distance_from_points_cellgrid(dec, ra, points, point_pix, rmax=rmax, omap=omap, odomains=odomains, domains=domains)
		elif method == "simple":
			return distances.distance_from_points_simple_separable(dec, ra, points, omap=omap, odomains=odomains, domains=domains)
		else: raise ValueError("Unknown method '%s'" % str(method))
	else:
		# We have a general geometry, so we need the full posmap. But to avoid wasting memory we
		# can loop over chunks of the posmap.
		if method == "bubble":
			# Not sure how to slice bubble. Just do it in one go for now
			pos = posmap(shape, wcs, safe=False)
			point_pix = utils.nint(sky2pix(shape, wcs, points))
			return distances.distance_from_points_bubble(pos, points, point_pix, rmax=rmax, omap=omap, odomains=odomains, domains=domains)
		elif method == "cellgrid":
			pos = posmap(shape, wcs, safe=False)
			point_pix = utils.nint(sky2pix(shape, wcs, points))
			return distances.distance_from_points_cellgrid(pos[0], pos[1], points, point_pix, rmax=rmax, omap=omap, odomains=odomains, domains=domains)
		elif method == "simple":
			geo = Geometry(shape, wcs)
			for y in range(0, shape[-2], step):
				sub_geo = geo[y:y+step]
				pos     = posmap(*sub_geo, safe=False)
				if domains:
					distances.distance_from_points_simple(pos, points, omap=omap[y:y+step], odomains=odomains[y:y+step], domains=True)
				else:
					distances.distance_from_points_simple(pos, points, omap=omap[y:y+step])
			if domains: return omap, odomains
			else:       return omap

def distance_transform_healpix(mask, omap=None, rmax=None, method="heap"):
	"""Given a boolean healpix mask, produce an output map where the value in each pixel is the distance
	to the closest false pixel in the mask. See distance_from for the meaning of rmax."""
	import healpy
	from pixell import distances
	npix  = mask.shape[-1]
	mflat = mask.reshape(-1,npix)
	nside = healpy.npix2nside(npix)
	info  = distances.healpix_info(nside)
	if omap is None: omap = np.zeros(mflat.shape)
	for i in range(len(mflat)):
		edge_pix = distances.find_edges_healpix(info, mflat[i])
		edge_pos = np.array(healpy.pix2ang(info.nside, edge_pix))
		edge_pos[0] = np.pi/2-edge_pos[0]
		distances.distance_from_points_healpix(info, edge_pos, edge_pix, omap=omap[i], rmax=rmax, method=method)
	omap = omap.reshape(mask.shape)
	# Distance is always zero inside mask
	omap *= mask
	return omap

def labeled_distance_transform_healpix(labels, omap=None, odomains=None, rmax=None, method="heap"):
	"""Given a healpix map of labels going from 1 to nlabel, produce an output map where the value
	in each pixel is the distance to the closest nonzero pixel in the labels, as well as a
	map of which label each pixel was closest to. See distance_from for the meaning of rmax."""
	import healpy
	from pixell import distances
	npix  = labels.shape[-1]
	lflat = labels.reshape(-1,npix)
	nside = healpy.npix2nside(npix)
	info  = distances.healpix_info(nside)
	if omap is None: omap = np.zeros(lflat.shape)
	if odomains is None: odomains = np.zeros(lflat.shape)
	for i in range(len(lflat)):
		edge_pix = distances.find_edges_labeled_healpix(info, lflat[i])
		edge_pos = np.array(healpy.pix2ang(info.nside, edge_pix))
		edge_pos[0] = np.pi/2-edge_pos[0]
		_, domains = distances.distance_from_points_healpix(info, edge_pos, edge_pix, omap=omap[i], domains=True, rmax=rmax, method=method)
		# Get the edge_pix to label mapping
		mapping = lflat[i][edge_pix]
		mask    = domains >= 0
		odomains[i,mask] = mapping[domains[mask]]
		# Distance is always zero inside each labeled region
		mask = lflat[i] != 0
		omap[i][mask] = 0
	omap = omap.reshape(labels.shape)
	odomains = odomains.reshape(labels.shape)
	return omap, odomains

def distance_from_healpix(nside, points, omap=None, odomains=None, domains=False, rmax=None, method="bubble"):
	"""Find the distance from each pixel in healpix map with nside nside to the
	nearest of the points[{dec,ra},npoint], returning a [ny,nx] map of distances.
	If domains==True, then it will also return a [ny,nx] map of the index of the point
	that was closest to each pixel. If rmax is specified, then distances will only be
	computed up to rmax. Beyond that distance will be set to rmax and domains to -1.
	This can be used to speed up the calculation when one only cares about nearby areas."""
	import healpy
	from pixell import distances
	info = distances.healpix_info(nside)
	if omap is None: omap = np.empty(info.npix)
	if domains and odomains is None: odomains = np.empty(info.npix, np.int32)
	pixs = utils.nint(healpy.ang2pix(nside, np.pi/2-points[0], points[1]))
	return distances.distance_from_points_healpix(info, points, pixs, rmax=rmax, omap=omap, odomains=odomains, domains=domains, method=method)

def grow_mask(mask, r):
	"""Grow the True part of boolean mask "mask" by a distance of r radians"""
	return (~mask).distance_transform(rmax=r) < r

def shrink_mask(mask, r):
	"""Shrink the True part of boolean mask "mask" by a distance of r radians"""
	return mask.distance_transform(rmax=r) >= r

def pad(emap, pix, return_slice=False, wrap=False, value=0):
	"""Pad enmap "emap", creating a larger map with zeros filled in on the sides.
	How much to pad is controlled via pix, which har format [{from,to},{y,x}],
	[{y,x}] or just a single number to apply on all sides. E.g. pix=5 would pad
	by 5 on all sides, and pix=[[1,2],[3,4]] would pad by 1 on the bottom,
	2 on the left, 3 on the top and 4 on the right."""
	pix = np.asarray(pix,dtype=int)
	if pix.ndim == 0:
		pix = np.array([[pix,pix],[pix,pix]])
	elif pix.ndim == 1:
		pix = np.array([pix,pix])
	# Exdend the wcs in each direction.
	w = emap.wcs.deepcopy()
	w.wcs.crpix += pix[0,::-1]
	# Construct a slice between the new and old map
	res = full(emap.shape[:-2]+tuple([s+sum(p) for s,p in zip(emap.shape[-2:],pix.T)]),wcs=w, val=value, dtype=emap.dtype)
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
	if value == "auto":
		# Find the median value along each edge
		medians = [np.median(m[...,:,i],-1) for i in [0,-1]] + [np.median(m[...,i,:],-1) for i in [0,-1]]
		bs = [find_blank_edges(m, med) for med in medians]
		nb = [np.prod(np.sum(b,0)) for b in bs]
		blanks = bs[np.argmax(nb)]
		return blanks
	elif value == "none":
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

def laplace(m):
	return -ifft(fft(m)*np.sum(m.lmap()**2,0)).real

#def apod(m, width, profile="cos", fill="zero"):
#	"""Apodize the provided map. Currently only cosine apodization is
#	implemented.
#
#    Args:
#        imap: (...,Ny,Nx) or (Ny,Nx) ndarray to be apodized
#        width: The width in pixels of the apodization on each edge.
#        profile: The shape of the apodization. Only "cos" is supported.
#	"""
#	width = np.minimum(np.zeros(2)+width,m.shape[-2:]).astype(np.int32)
#	if profile == "cos":
#		a = [0.5*(1-np.cos(np.linspace(0,np.pi,w))) for w in width]
#	else:
#		raise ValueError("Unknown apodization profile %s" % profile)
#	res = m.copy()
#	if fill == "mean":
#		offset = np.asarray(np.mean(res,(-2,-1)))[...,None,None]
#		res -= offset
#	elif fill == "median":
#		offset = np.asarray(np.median(res,(-2,-1)))[...,None,None]
#		res -= offset
#	if width[0] > 0:
#		res[...,:width[0],:] *= a[0][:,None]
#		res[...,-width[0]:,:] *= a[0][::-1,None]
#	if width[1] > 0:
#		res[...,:,:width[1]] *= a[1][None,:]
#		res[...,:,-width[1]:]  *= a[1][None,::-1]
#	if fill == "mean" or fill == "median":
#		res += offset
#	return res

def apod(map, width, profile="cos", fill="zero", inplace=False):
	width = (np.zeros(2,int)+width).astype(int)
	if not inplace: map = map.copy()
	if fill == "mean":
		offset = np.mean(map,(-2,-1))[...,None,None]
		map   -= offset
	elif fill == "median":
		offset = np.median(map,(-2,-1))[...,None,None]
		map   -= offset
	# Process the axes one by one
	for i, w in enumerate(width):
		if w <= 0: continue
		if fill == "crossfade":
			x = np.arange(1,w+1,dtype=map.dtype)/(2*w+1)
		else:
			x = np.arange(1,w+1,dtype=map.dtype)/(w+1)
		if   profile == "lin": prof = apod_profile_lin(x)
		elif profile == "cos": prof = apod_profile_cos(x)
		else: raise ValueError("Unknown apodization profile '%s'" % str(profile))
		# Apply the apodization
		slice1 = (Ellipsis,)+(slice(None),)*i    +(slice(0,w),)+(slice(None),)*(1-i)
		slice2 = (Ellipsis,)+(slice(None),)*i    +(slice(-w,None),)+(slice(None),)*(1-i)
		broad  = (None,)*i+(slice(None),)+(None,)*(1-i)
		m1 = map[slice1].copy()
		m2 = map[slice2].copy()
		if fill == "crossfade":
			map[slice1] = m1*(1-prof)[::-1][broad]+m2*prof[::-1][broad]
			map[slice2] = m2*(1-prof)[broad]+m1*prof[broad]
		elif fill in ["mean", "median", "zero"]:
			map[slice1] *= prof[broad]
			map[slice2] *= prof[::-1][broad]
	# Add in offsets if necessary
	if fill in ["mean", "median"]:
		map += offset
	return map

def apod_profile_lin(x): return x
def apod_profile_cos(x): return 0.5*(1-np.cos(np.pi*x))

def apod_mask(mask, width=1*utils.degree, edge=True, profile=apod_profile_cos):
	"""Given an enmap mask that's 0 in bad regions and 1 in good regions, return an
	apodization map that's still 0 in bad regions, but transitions smoothly
	to 1 in the good region over the given width in radians. The transition
	profile is controlled by the profile argument. Regions outside the
	image are considered to be bad."""
	if edge:
		mask = mask.copy()
		mask[..., 0,:] = False; mask[...,:, 0] = False
		mask[...,-1,:] = False; mask[...,:,-1] = False
	r = mask.distance_transform(rmax=width)
	return profile(r/width)

def lform(map, method="auto"):
	"""Given an enmap, return a new enmap that has been fftshifted (unless shift=False),
	and which has had the wcs replaced by one describing fourier space. This is mostly
	useful for plotting or writing 2d power spectra.

	It could have been useful more generally, but because all "plain" coordinate systems
	are assumed to need conversion between degrees and radians, sky2pix etc. get confused
	when applied to lform-maps."""
	omap = fftshift(map)
	omap.wcs = lwcs(map.shape, map.wcs, method=method)
	return omap

def lwcs(shape, wcs, method="auto"):
	"""Build world coordinate system for l-space"""
	lres   = 2*np.pi/extent(shape, wcs, signed=True, method=method)
	ny, nx = shape[-2:]
	owcs   = wcsutils.explicit(crpix=[nx//2+1,ny//2+1], crval=[0,0], cdelt=lres[::-1])
	return owcs

def rbin(map, center=[0,0], bsize=None, brel=1.0, return_nhit=False, return_bins=False, rop=None):
	"""Radially bin map around the given center point ([0,0] by default).
	If bsize it given it will be the constant bin width. This defaults to
	the pixel size. brel can be used to scale up the bin size. This is
	mostly useful when using automatic bsize.

	Returns bvals[...,nbin], r[nbin], where bvals is the mean
	of the map in each radial bin and r is the mid-point of each bin
	"""
	r = map.modrmap(ref=center)
	if rop: r = rop(r)
	if bsize is None:
		bsize = np.min(map.extent()/map.shape[-2:])
	return _bin_helper(map, r, bsize*brel, return_nhit=return_nhit, return_bins=return_bins)

def lbin(map, bsize=None, brel=1.0, return_nhit=False, return_bins=False, lop=None):
	"""Like rbin, but for fourier space. Returns b(l),l"""
	l = map.modlmap()
	if lop: l = lop(l)
	if bsize is None: bsize = min(abs(l[0,1]),abs(l[1,0]))
	return _bin_helper(map, l, bsize*brel, return_nhit=return_nhit, return_bins=return_bins)

def _bin_helper(map, r, bsize, return_nhit=False, return_bins=False):
	"""This is very similar to a function in utils, but was sufficiently different
	that it didn't make sense to reuse that one. This is often the case with the
	binning in utils. I should clean that up, and probably base one of the new
	functions on this one."""
	# Get the number of bins
	n     = int(np.max(r/bsize))
	rinds = utils.floor((r/bsize).reshape(-1))
	# Ok, rebin the map. We do this using bincount, which can be a bit slow
	mflat = map.reshape((-1,)+map.shape[-2:])
	mout = np.zeros((len(mflat),n))
	nhit = np.bincount(rinds)[:n]
	for i, m in enumerate(mflat):
		mout[i] = np.bincount(rinds, weights=m.reshape(-1))[:n]/nhit
	mout = mout.reshape(map.shape[:-2]+mout.shape[1:])
	# What r should we assign to each bin? We could just use the bin center,
	# but since we're averaging point samples in each bin, it makes more sense
	# to assign the same average of the r values
	orads = np.bincount(rinds, weights=r.reshape(-1))[:n]/nhit
	if return_bins:
		edges = np.arange(len(orads)+1)*bsize
		orads = np.array([orads,edges[:-1],edges[1:]])
	if return_nhit: return mout, orads, nhit
	else: return mout, orads

def radial_average(map, center=[0,0], step=1.0):
	warnings.warn("radial_average has been renamed to rbin", DeprecationWarning)
	return rbin(map, center=center, brel=step)

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

def to_healpix(imap, omap=None, nside=0, order=3, chunk=100000):
	raise RuntimeError("This function has been removed. Use reproject.map2healpix().")

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
	import flipper.liteMap
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

def write_map(fname, emap, fmt=None, address=None, extra={}, allow_modify=False):
	"""Writes an enmap to file. If fmt is not passed,
	the file type is inferred from the file extension, and can
	be either fits or hdf. This can be overriden by
	passing fmt with either 'fits' or 'hdf' as argument.

	The other arguments are passed to write_fits and/or write_hdf."""
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits"
		else: fmt = "fits"
	if fmt == "fits":
		write_fits(fname, emap, extra=extra, allow_modify=allow_modify)
	elif fmt == "hdf":
		write_hdf(fname, emap, address=address, extra=extra)
	else:
		raise ValueError

def read_map(fname, fmt=None, sel=None, box=None, pixbox=None, geometry=None, wrap="auto", mode=None, sel_threshold=10e6, wcs=None, hdu=None, delayed=False, verbose=False, address=None, recenter=False):
	"""Read an enmap from file. The file type is inferred
	from the file extension, unless fmt is passed.
	fmt must be one of 'fits' and 'hdf'.

	The sel, box, pixbox, geometry, wrap, mode, and delayed arguments
	are all used by read_helper to (optionally) select a subregion of
	the map or change how it is wrapped on the sky.

	The hdu and verbose arguments are only used for FITS (see
	read_fits).  The address argument is only used for HDF (see
	read_hdf)."""
	toks = fname.split(":")
	fname = toks[0]
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".npy"):     fmt = "npy"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits"
		else: fmt = "fits"
	if fmt == "fits":
		res = read_fits(fname, sel=sel, box=box, pixbox=pixbox, geometry=geometry, wrap=wrap, mode=mode, sel_threshold=sel_threshold, wcs=wcs, hdu=hdu, delayed=delayed, recenter=recenter, verbose=verbose)
	elif fmt == "hdf":
		res = read_hdf(fname, sel=sel, box=box, pixbox=pixbox, geometry=geometry, wrap=wrap, mode=mode, sel_threshold=sel_threshold, wcs=wcs, delayed=delayed, hdu=hdu, recenter=recenter, address=address)
	elif fmt == "npy":
		res = read_npy(fname, sel=sel, box=box, pixbox=pixbox, geometry=geometry, wrap=wrap, mode=mode, sel_threshold=sel_threshold, wcs=wcs, delayed=delayed, hdu=hdu, recenter=recenter, address=address)
	else:
		raise ValueError
	if len(toks) > 1:
		res = eval("res"+":".join(toks[1:]))
	return res

def read_map_geometry(fname, fmt=None, hdu=None, address=None):
	"""Read an enmap geometry from file. The file type is inferred
	from the file extension, unless fmt is passed.
	fmt must be one of 'fits' and 'hdf'."""
	toks = fname.split(":")
	fname = toks[0]
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits.gz"
		else: fmt = "fits"
	if fmt == "fits":
		shape, wcs = read_fits_geometry(fname, hdu=hdu)
	elif fmt == "fits.gz":
		shape, wcs = read_fits_geometry(fname, hdu=hdu, quick=False)
	elif fmt == "hdf":
		shape, wcs = read_hdf_geometry(fname, address=address)
	else:
		raise ValueError
	if len(toks) > 1:
		sel = eval("utils.sliceeval"+":".join(toks[1:]))[-2:]
		shape, wcs = slice_geometry(shape, wcs, sel)
	return shape, wcs

def read_map_dtype(fname, fmt=None, hdu=None, address=None):
	toks = fname.split(":")
	fname = toks[0]
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		elif fname.endswith(".fits.gz"): fmt = "fits.gz"
		else: fmt = "fits"
	if   fmt == "fits":    return read_fits_dtype(fname, hdu=hdu)
	elif fmt == "fits.gz": return read_fits_dtype(fname, hdu=hdu, quick=False)
	elif fmt == "hdf":     return read_hdf_dtype (fname, address=address)
	else: raise ValueError

def write_map_geometry(fname, shape, wcs, fmt=None):
	"""Write an enmap geometry to file. The file type is inferred
	from the file extension, unless fmt is passed.
	fmt must be one of 'fits' and 'hdf'. Only fits is supported for now, though."""
	toks = fname.split(":")
	fname = toks[0]
	if fmt == None:
		if   fname.endswith(".hdf"):     fmt = "hdf"
		elif fname.endswith(".fits"):    fmt = "fits"
		else: fmt = "fits"
	if fmt == "fits":
		write_fits_geometry(fname, shape, wcs)
	elif fmt == "hdf":
		raise NotImplementedError("Write write_hdf_geometry not implemented yet")
	else:
		raise ValueError

def write_fits(fname, emap, extra={}, allow_modify=False):
	"""Write an enmap to a fits file."""
	# The fits write routines may attempt to modify
	# the map. So make a copy.
	if not allow_modify:
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
	if isinstance(fname, str):
		utils.mkdir(os.path.dirname(fname))
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		hdus.writeto(fname, overwrite=True)

def write_fits_geometry(fname, shape, wcs):
	"""Write just the geometry to a fits file that will only contain the header"""
	header = wcs.to_header(relax=True)
	header.insert(0, ("SIMPLE",True))
	header.insert(1, ("BITPIX",-32))
	header.insert(2, ("NAXIS",len(shape)))
	for i, s in enumerate(shape[::-1]):
		header.insert(3+i, ("NAXIS%d"%(i+1),s))
	# Dummy, but must be present
	utils.mkdir(os.path.dirname(fname))
	header.tofile(fname, overwrite=True)

def read_fits(fname, hdu=None, sel=None, box=None, pixbox=None, geometry=None, wrap="auto", mode=None, sel_threshold=10e6, wcs=None, delayed=False, recenter=False, verbose=False):
	"""Read an enmap from the specified fits file. By default,
	the map and coordinate system will be read from HDU 0. Use
	the hdu argument to change this. The map must be stored as
	a fits image. If sel is specified, it should be a slice
	that will be applied to the image before reading. This avoids
	reading more of the image than necessary. Instead of sel,
	a coordinate box [[yfrom,xfrom],[yto,xto]] can be specified."""
	if hdu is None: hdu = 0
	hdu = astropy.io.fits.open(fname)[hdu]
	ndim = len(hdu.shape)
	if hdu.header["NAXIS"] < 2:
		raise ValueError("%s is not an enmap (only %d axes)" % (str(fname), hdu.header["NAXIS"]))
	if wcs is None:
		with warnings.catch_warnings():
			wcs = wcsutils.WCS(hdu.header).sub(2)
	proxy = ndmap_proxy_fits(hdu, wcs, fname=fname, threshold=sel_threshold, verbose=verbose)
	return read_helper(proxy, sel=sel, box=box, pixbox=pixbox, geometry=geometry, wrap=wrap, mode=mode, delayed=delayed, recenter=recenter)

def read_fits_header(fname, hdu=None, quick=True):
	if hdu is None: hdu = 0
	if hdu == 0 and quick:
		# Read header only, without body
		if isinstance(fname, str):
			with open(fname, "rb") as ifile:
				header = astropy.io.fits.Header.fromstring(ifile.read(2880))
		else:
			# Handle the case where the user already has a file object
			header = astropy.io.fits.Header.fromstring(fname.read(2880))
	else:
		with utils.nowarn():
			header = astropy.io.fits.open(fname)[hdu].header
	return header

def read_fits_geometry(fname, hdu=None, quick=True):
	"""Read an enmap wcs from the specified fits file. By default,
	the map and coordinate system will be read from HDU 0. Use
	the hdu argument to change this. The map must be stored as
	a fits image."""
	header = read_fits_header(fname, hdu=hdu, quick=quick)
	if header["NAXIS"] < 2:
		raise ValueError("%s is not an enmap (only %d axes)" % (str(fname), header["NAXIS"]))
	with warnings.catch_warnings():
		wcs = wcsutils.WCS(header).sub(2)
	shape = tuple([header["NAXIS%d"%(i+1)] for i in range(header["NAXIS"])[::-1]])
	return shape, wcs

def read_fits_dtype(fname, hdu=None, quick=True):
	header = read_fits_header(fname, hdu=hdu, quick=quick)
	if "BITPIX" not in header: raise KeyError("BITPIX not defined in fits file")
	bitpix = header["BITPIX"]
	table  = {-32:np.float32, -64:np.float64, 8:np.int8, 16:np.int16, 32:np.int32, 64:np.int64}
	if bitpix not in table: raise ValueError("Unrecognized BITPIX %d" % bitpix)
	return table[bitpix]

def write_hdf(fname, emap, address=None, extra={}):
	"""Write an enmap as an hdf file, preserving all the WCS
	metadata.

	Args:
	  fname (str or h5py.Group): Filename or open h5py handle.
	  emap (ndmap): Object to store.
	  address (str): Group address within the HDF file to place
	    the result.  If None, the data are written at root level
	    after truncating the file.
	  extra (dict): additional data to write into the output.

	Notes:
	  If address is None, the output file will be replaced if it
	  exists.  If address is a string, and the target file exists,
	  the file will not be reset but anything living at that
	  address will be replaced with the encoded emap."""
	import h5py
	emap = enmap(emap, copy=False)
	if isinstance(fname, h5py.Group):
		context = contextlib.nullcontext(fname)
	else:
		utils.mkdir(os.path.dirname(fname))
		mode = "w" if address is None else "a"
		context = h5py.File(fname, mode)
	with context as hfile:
		if address is not None:
			if address in hfile:
				del hfile[address]
			hfile = hfile.create_group(address)
		hfile["data"] = emap
		header = emap.wcs.to_header()
		for key in header:
			hfile["wcs/"+key] = header[key]
		for key, val in extra.items():
			hfile[key] = val

def read_hdf(fname, hdu=None, sel=None, box=None, pixbox=None, geometry=None, wrap="auto", mode=None, sel_threshold=10e6, wcs=None, delayed=False, address=None, recenter=False):
	"""Read an enmap from the specified hdf file. Two formats
	are supported. The old enmap format, which simply used
	a bounding box to specify the coordinates, and the new
	format, which uses WCS properties. The latter is used if
	available. With the old format, plate carree projection
	is assumed. Note: some of the old files have a slightly
	buggy wcs, which can result in 1-pixel errors.

	If address is a string, the map will be loaded from that group
	address within fname.

	Note fname can be passed in as an h5py.Group (e.g. an open
	h5py.File) instead of a string, and the map will be read from that
	handle."""
	import h5py
	if isinstance(fname, h5py.Group):
		context = contextlib.nullcontext(fname)
	else:
		context = h5py.File(fname, "r")
	with context as hfile:
		if address is not None:
			hfile = hfile[address]
		data = hfile["data"][()]
		hwcs = hfile["wcs"]
		header = astropy.io.fits.Header()
		for key in hwcs:
			header[key] = fix_python3(hwcs[key][()])
		if wcs is None:
			wcs = wcsutils.WCS(header).sub(2)
		proxy = ndmap_proxy_hdf(data, wcs, fname=fname, threshold=sel_threshold)
		return read_helper(proxy, sel=sel, box=box, pixbox=pixbox, geometry=geometry, wrap=wrap, mode=mode, delayed=delayed, recenter=recenter)

def read_hdf_geometry(fname, address=None):
	"""Read an enmap wcs from the specified hdf file."""
	import h5py
	with h5py.File(fname,"r") as hfile:
		if address is not None:
			hfile = hfile[address]
		hwcs = hfile["wcs"]
		header = astropy.io.fits.Header()
		for key in hwcs:
			header[key] = hwcs[key][()]
		wcs   = wcsutils.WCS(header).sub(2)
		shape = hfile["data"].shape
	return shape, wcs

def read_hdf_dtype(fname, address=None):
	import h5py
	with h5py.File(fname,"r") as hfile:
		if address is not None:
			hfile = hfile[address]
		return hfile["data"].dtype

def read_npy(fname, hdu=None, sel=None, box=None, pixbox=None, geometry=None, wrap="auto", mode=None, sel_threshold=10e6, wcs=None, delayed=False, address=None, recenter=False):
	"""Read an enmap from the specified npy file. Only minimal support.
	No wcs information."""
	return enmap(np.load(fname), wcs)

def fix_python3(s):
	"""Convert "bytes" to string in python3, while leaving other types unmolested.
	Python3 string handling is stupid."""
	try:
		if isinstance(s, bytes): return s.decode("utf-8")
		else: return s
	except TypeError: return s

def read_helper(data, sel=None, box=None, pixbox=None, geometry=None, wrap="auto", mode=None, delayed=False, recenter=False):
	"""Helper function for map reading. Handles the slicing, sky-wrapping and capping, etc."""
	if delayed: return data # Slicing not supported yet when we want to return a proxy object
	if geometry is not None: data = extract(data, *geometry, wrap=wrap)
	if box      is not None: data = submap(data, box, wrap=wrap, recenter=recenter)
	if pixbox   is not None: data = extract_pixbox(data, pixbox, wrap=wrap, recenter=recenter)
	if sel      is not None: data = data[sel]
	data = data[:] # Get rid of the wrapper if it still remains
	data = data.copy()
	return data

# These wrapper classes are there to let us reuse the normal map
# extract and submap operations on fits and hdf maps without needing
# to read in all the data.

class ndmap_proxy:
	def __init__(self, shape, wcs, dtype, fname="<none>", threshold=1e7):
		self.fname, self.shape, self.wcs, self.dtype = fname, shape, wcs, dtype
		self.threshold = threshold
	@property
	def ndim(self): return len(self.shape)
	@property
	def geometry(self): return self.shape, self.wcs
	@property
	def npix(self): return self.shape[-2]*self.shape[-1]
	def __str__(self): return repr(self)
	def __repr__(self): return "ndmap_proxy(fname=%s, shape=%s, wcs=%s, dtype=%s)" % (str(self.fname), str(self.shape), str(self.wcs), str(self.dtype))
	def __getslice__(self, a, b=None, c=None): return self[slice(a,b,c)]
	def __getitem__(self, sel): raise NotImplementedError("ndmap_proxy must be subclassed")
	def submap(self, box, mode=None, wrap="auto", recenter=False):
		return submap(self, box, mode=mode, wrap=wrap, recenter=recenter)
	def stamps(self, pos, shape, aslist=False): return stamps(self, pos, shape, aslist=aslist)

# Copy over some methos from ndmap
for name in ["sky2pix", "pix2sky", "box", "pixbox_of", "posmap", "pixmap", "lmap", "modlmap", "modrmap", "area", "pixsize", "pixshape",
		"pixsizemap", "pixshapemap", "extent", "distance_from", "center", "extract", "extract_pixbox"]:
	setattr(ndmap_proxy, name, getattr(ndmap, name))

class ndmap_proxy_fits(ndmap_proxy):
	def __init__(self, hdu, wcs, fname="<none>", threshold=1e7, verbose=False):
		self.hdu     = hdu
		self.verbose = verbose
		# Note that 'section' is not part of some HDU types, such as CompImageHDU.
		self.use_section = hasattr(hdu, 'section')
		if self.use_section:
			dtype    = fix_endian(hdu.section[(slice(0,1),)*hdu.header["NAXIS"]]).dtype
		else:
			dtype    = fix_endian(hdu.data[(slice(0,1),)*hdu.header["NAXIS"]]).dtype
		self.stokes_flips = get_stokes_flips(hdu)
		def slist(vals):
			return ",".join([str(v) for v in vals])
		if verbose and np.any(self.stokes_flips >= 0):
			print("Converting index %s for Stokes axis %s from IAU to COSMO in %s" % (
				slist(self.stokes_flips[self.stokes_flips >= 0]),
				slist(np.where(self.stokes_flips >= 0)[0]),
				str(fname)))
		ndmap_proxy.__init__(self, hdu.shape, wcs, dtype, fname=fname, threshold=threshold)
	def __getitem__(self, sel):
		_, psel = utils.split_slice(sel, [len(self.shape)-2,2])
		if len(psel) > 2: raise IndexError("too many indices")
		_, wcs = slice_geometry(self.shape[-2:], self.wcs, psel)
		if (self.hdu.size > self.threshold) and self.use_section:
			sel1, sel2 = utils.split_slice(sel, [len(self.shape)-1,1])
			res = self.hdu.section[sel1][(Ellipsis,)+sel2]
		else: res = self.hdu.data[sel]
		# Apply stokes flips if necessary. This is a bit complicated because we have to
		# take into account that slicing might have already been done. The simplest way
		# to do this is to make a sign array with the same shape as all the pre-dimensions
		# of the raw map, and then slice that the same way.
		if np.any(self.stokes_flips >= 0):
			signs = np.full(self.shape[:-2], 1, int)
			for i, ind in enumerate(self.stokes_flips):
				if ind >= 0 and ind < self.shape[i]:
					signs[(slice(None),)*i + (ind,)] *= -1
			sel1, sel2 = utils.split_slice(sel, [len(self.shape)-2,2])
			res *= signs[sel1][...,None,None]
		return ndmap(fix_endian(res), wcs)
	def __repr__(self): return "ndmap_proxy_fits(fname=%s, shape=%s, wcs=%s, dtype=%s)" % (str(self.fname), str(self.shape), str(self.wcs), str(self.dtype))

class ndmap_proxy_hdf(ndmap_proxy):
	def __init__(self, dset, wcs, fname="<none>", threshold=1e7):
		self.dset      = dset
		ndmap_proxy.__init__(self, dset.shape, wcs, dset.dtype, fname=fname, threshold=threshold)
	def __getitem__(self, sel):
		_, psel = utils.split_slice(sel, [self.ndim-2,2])
		if len(psel) > 2: raise IndexError("too many indices")
		_, wcs = slice_geometry(self.shape[-2:], self.wcs, psel)
		if self.dset.size > self.threshold:
			sel1, sel2 = utils.split_slice(sel, [len(self.shape)-1,1])
			res = self.dset[sel1][(Ellipsis,)+sel2]
		else:
			res = self.dset[sel]
		return ndmap(fix_endian(res), wcs)
	def __repr__(self): return "ndmap_proxy_hdf(fname=%s, shape=%s, wcs=%s, dtype=%s)" % (self.fname, str(self.shape), str(self.wcs), str(self.dtype))

def fix_endian(map):
	"""Make endianness of array map match the current machine.
	Returns the result."""
	if map.dtype.byteorder not in ['=','<' if sys.byteorder == 'little' else '>']:
		map = map.byteswap(True).view(map.dtype.newbyteorder())
	map.dtype = utils.fix_dtype_mpi4py(map.dtype)
	return map

def get_stokes_flips(hdu):
	"""Given a FITS HDU, parse its header to determine which, if any, axes
	need to have their sign flip to get them in the COSMO polarization convention.
	Returns an array of length ndim, with each entry being the index of the axis
	that should be flipped, or -1 if none should be flipped."""
	ndim   = hdu.header["NAXIS"]
	# First find which index of each axis is U
	inds   = np.full(ndim, -1, int)
	noflip = np.full(ndim, -1, int)
	def get(name, ndim, i, default=None):
		nfull = name + "%d" % (ndim-i)
		return hdu.header[nfull] if nfull in hdu.header else default
	for i in range(ndim):
		ctype = get("CTYPE", ndim, i, "")
		if ctype.strip() == "STOKES":
			crpix = get("CRPIX", ndim, i, 1.0)
			crval = get("CRVAL", ndim, i, 1.0)
			cdelt = get("CDELT", ndim, i, 1.0)
			U_ind = utils.nint((3-crval)/cdelt+crpix)
			inds[i] = U_ind - 1
	# If there are no U indices (for example because there was no Stokes axis),
	# then there is nothing to flip
	if np.all(inds == -1): return noflip
	# Otherwise, check the polarization convention
	if   "POLCCONV" in hdu.header: polconv = hdu.header["POLCCONV"].strip()
	elif "POLCONV"  in hdu.header: polconv = hdu.header["POLCONV" ].strip()
	else:
		warnings.warn("FITS file has stokes axis, but no POLCCONV is specified. Assuming IAU")
		return inds
	if   polconv == "COSMO": return noflip
	elif polconv == "IAU":   return inds
	else:
		warnings.warn("Unrecognized POLCCONV '%s', assuming COSMO" % polconv)
		return noflip

def shift(map, off, inplace=False, keepwcs=False):
	"""Cyclicly shift the pixels in map such that a pixel at
	position (i,j) ends up at position (i+off[0],j+off[1])"""
	if not inplace: map = map.copy()
	off = np.atleast_1d(off)
	for i, o in enumerate(off):
		if o != 0:
			map[:] = np.roll(map, o, -len(off)+i)
	if not keepwcs:
		map.wcs.wcs.crpix += off[::-1]
	return map

def fractional_shift(map, off, keepwcs=False, nofft=False):
	"""Shift map cyclically by a non-integer amount off [{y_off,x_off}]"""
	omap = samewcs(enfft.shift(map, off, nofft=nofft), map)
	if not keepwcs:
		omap.wcs.wcs.crpix += off[::-1]
	return omap

def fftshift(map, inplace=False):
	if not inplace: map = map.copy()
	map[:] = np.fft.fftshift(map, axes=[-2,-1])
	return map

def ifftshift(map, inplace=False):
	if not inplace: map = map.copy()
	map[:] = np.fft.ifftshift(map, axes=[-2,-1])
	return map

def fillbad(map, val=0, inplace=False):
	return np.nan_to_num(map, copy=not inplace, nan=val, posinf=val, neginf=val)

def resample(map, oshape, off=(0,0), method="fft", border="wrap", corner=True, order=3):
	"""Resample the input map such that it covers the same area of the sky
	with a different number of pixels given by oshape."""
	# Construct the output shape and wcs
	oshape = map.shape[:-2] + tuple(oshape)[-2:]
	if method == "fft":
		omap  = ifft(resample_fft(fft(map, normalize=False), oshape, off=off, corner=corner, norm=1/map.npix), normalize=False).real
	elif method == "spline":
		owcs = wcsutils.scale(map.wcs, np.array(oshape[-2:],float)/map.shape[-2:], rowmajor=True, corner=corner)
		off  = np.zeros(2)+off
		if corner:
			off -= 0.5 - 0.5*np.array(oshape[-2:],float)/map.shape[-2:] # in output units
		opix  = pixmap(oshape) - off[:,None,None]
		ipix  = opix * (np.array(map.shape[-2:],float)/oshape[-2:])[:,None,None]
		omap  = ndmap(map.at(ipix, unit="pix", border=border, order=order), owcs)
	else:
		raise ValueError("Invalid resample method '%s'" % method)
	return omap

def resample_fft(fimap, oshape, fomap=None, off=(0,0), corner=True, norm="pix", op=lambda a,b:b, dummy=False):
	"""Like resample, but takes a fourier-space map as input and outputs a fourier-space map.
	unit specifies which fourier-space unit is used. "pix" corresponds to
	the standard enmap normalization (normalize=True in enmap.fft). "phys" corresponds
	to physical normalization (normalize="phys"). The fourier-units matter because some
	fourier-space units need rescaline when going from one resolution to another.
	"""
	# Construct the output shape and wcs
	oshape = fimap.shape[:-2] + tuple(oshape)[-2:]
	off    = np.zeros(2)+off
	if corner:
		# Apply phase shift to realign with pixel centers. This can be seen as a half pixel shift to
		# the left in the original pixelization followed by a half pixel shift to the right in the new
		# pixelization.
		off -= 0.5 - 0.5*np.array(oshape[-2:],float)/fimap.shape[-2:] # in output units
	if fomap is None:
		owcs  = wcsutils.scale(fimap.wcs, np.array(oshape[-2:],float)/fimap.shape[-2:], rowmajor=True, corner=corner)
		if dummy: return oshape, owcs
		fomap = zeros(oshape, owcs, fimap.dtype)
	if dummy: return oshape, owcs
	# We sadly need to care about fourier-space normalization when doing this, since
	# different-size fourier spaces can have different units. First handle explicit normalization,
	# where the factor to multiply is given directly.
	try: norm = float(norm)
	except (TypeError, ValueError):
		# Then handle various normalization conventions.
		if   norm is None:     norm = 1 # Don't do anything if None is passed. Cost free
		elif norm == "plain":  norm = fomap.npix/fimap.npix # Corresponds to normalize=False in enmap.ifft
		elif norm == "pix":    norm = (fomap.npix/fimap.npix)**0.5 # Corresponds to normalize=True, enmap.fft default
		elif norm == "phys":   norm = 1 # Corresponds to normalize="phys"
		else: raise ValueError("Unrecognized fourier unit '%s'" % str(unit))
	# copy over all 4 quadrants. This would have been a single operation if the
	# fourier center had been in the middle. This could be acieved using fftshift,
	# but that would require two extra full-array shifts
	cny, cnx = np.minimum(fimap.shape[-2:], oshape[-2:])
	hny, hnx = cny//2, cnx//2
	# This function is used to avoid paying the cost of multiplying by norm when it's one
	def transfer(dest, source, norm, op):
		if norm != 1: source = source*norm
		dest[:] = op(dest, source)
	transfer(fomap[...,:hny,       :hnx       ],fimap[...,:hny,       :hnx       ], norm, op)
	transfer(fomap[...,:hny,       -(cnx-hnx):],fimap[...,:hny,       -(cnx-hnx):], norm, op)
	transfer(fomap[...,-(cny-hny):,:hnx       ],fimap[...,-(cny-hny):,:hnx       ], norm, op)
	transfer(fomap[...,-(cny-hny):,-(cnx-hnx):],fimap[...,-(cny-hny):,-(cnx-hnx):], norm, op)
	if np.any(off != 0):
		# It's fastest to do this here when downsampling, but when upsampling
		# it's faster to do so in the fimap. And for a mix it's bad to do it both places.
		fomap[:] = enfft.shift(fomap, off, axes=(-2,-1), nofft=True)
	return fomap

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

def spin_pre_helper(spin, pre):
	"""Like spin_helper, but also handles looping over pre-dimensions"""
	# Make spin a 1d array. This will be used to
	# interpret the last axis in pre
	spin  = np.array(spin).reshape(-1)
	scomp = 1+(spin!=0)
	# Make pre an array that's at least 1d
	pre   = np.array(pre).reshape(-1)
	# Handle empty pre-dimentions
	if len(pre) == 0:
		yield 0, (None,)
		return
	n     = pre[-1]
	# Loop over pre-dimensions
	for Ipre in utils.nditer(pre[:-1]):
		ci, i1 = 0, 0
		while True:
			i2 = min(i1+scomp[ci],n)
			if i2-i1 != scomp[ci]: raise IndexError("Unpaired component in spin transform")
			Itot = Ipre + (slice(i1,i2),)
			yield spin[ci], Itot
			if i2 == n: break
			i1 = i2
			ci = (ci+1)%len(spin)

# It's often useful to be able to loop over padded tiles, do some operation on them,
# and then stitch them back together with crossfading. If would be handy to have a way to
# hide all this complexity. How about an iterator that iterates over padded tiles?
# E.g.
#  for itile, otile in zip(ipadtiles(imap), opadtiles(omap)):
#    otile[:] = fancy_stuff(itile)
# The input tile iterator is straightforward. The output iterator would
# zero out omap at the start, and then at the start of each next()
# would paste the previously yielded tile back into omap with crossfading weights applied.
# A problem with this formulation where ipadtiles and opadtiles are separate
# functions, though, is that padding arguments need to be duplicated, which can get
# tedious. Padding argument must always be consistent when iterating over input
# and output maps, so probably better to have a single function that processes
# multiple maps.
#
#  for itile, otile in padtiles(imap, omap, tshape=512, pad=30, apod=30):
#    otile[:] = foo(itile[:])
#
# When multiple maps are involved, how should it know which ones
# are output and input maps? Default:
#  1 map: input
#  2 maps: input, output
#  N maps: input, input, input, ..., output
# But have an optional argument that lets us specify the types.
#
# 3rd alternative which is cleaner but less convenient:
#  padtile = Padtiler(tshape=512, pad=30, apod=30)
#  for itile, otile in zip(padtile.in(imap), padtile.out(omap)):
#   otile[:] = foo(itile)
# This one has the advantage that it can be built once and then
# passed to other functions as a single argument. It can easily be used to
# implement #2, so can get both cheaply. #3 is less convenient in
# most cases, so #2 would be the typcial interface.

def padtiles(*maps, tshape=600, pad=60, margin=60, mode="auto", start=0, step=1):
	"""Iterate over padded tiles in one or more maps. The tiling
	will have a logical tile shape of tshape, but each yielded tile
	will be expanded with some data from its neighbors. The extra
	area consists of two parts: The padding and the margin. For
	a read-iterator these are equivalent, but for a write-iterator
	the margin will be ignored (and so can be used for apodization),
	while the padding will be used for crossfading when mergin the
	tiles together.

	Typical usage:

		for itile, otile in padtiles(imap, imap, margin=60):
			itile    = apod(itile, 60)
			otile[:] = some_filter(itile)

	This would iterate over tiles of imap and omap, with the default
	padding and a margin of 60 pixels. The margin region is used for
	apodization, and some filter is then applied to the tile, writing
	the result to the output tile. Note the use of [:] to actually write
	to otile instead of just rebinding the variable name!

	It's also possible to iterate over fewer or more maps at once.
	See the "mode" argument.

	If the tile shape does not evenly divide the map shape, then the
	last tile in each row and column will extend beyond the edge of the
	map. These pixels will be treated as enmap.extract does, with the
	potential of sky wrapping. Warning: Write-iterators for a map that
	goes all the way around the sky while the tile shape does not divide
	the map shape will incorrectly weight the wrapped tiles, so avoid this.

	Arguments:
	 * *maps: The maps to iterate over. Must have the same pixel dimensions.
	 * tshape: The tile shape. Either an integer or a (yshape,xshape) tuple.
	   Default: 600 pixels.
	 * pad: The padding. Either an integer or a (ypad,xpad) tuple. Used to
	   implement context and crossfading. Cannot be larger than half of tshape.
	   Default: 60 pixels.
	 * margin: The margin size. Either an integer or a (ymargin,xmargin) tuple.
	   Ignored in write-iterators, so suitable for apodization.
	   Default 60 pixels.
	 * mode: Specifies which maps should be read-iterated vs. write-iterated.
	   A read-iterated map will yield padded tiles from the corresponding map.
	   Writes to these tiles are discarded. A write-iterated map yields zeroed
	   tiles of the same shape as the read-iterator. Writes to these tiles are
	   used to update the corresponding map, including crossfading the overlapping
	   regions (due to the padding) such that there aren't any sharp tile boundaries
	   in the output map. mode can be either "auto" or a string of the same length
	   as maps consisting of "r" and "w" characters. If the nth character is r/w
	   then the corresponding map will be read/write-iterated. If the string is
	   "auto", then the last map will be output-iterated and all the others input-
	   iterated, unless there's only a single map in which case it will be input-
	   iterated. Default: "auto".
	 * start: Flattened tile offset to start at. Useful for mpi loops. Default: 0.
	 * step:  Flattened tile stride. Useful for mpi loops. Default: 1
	"""
	if mode == "auto":
		if   len(maps) == 0: mode = ""
		elif len(maps) == 1: mode = "r"
		else:                mode = "r"*(len(maps)-1)+"w"
	tiler = Padtiler(tshape=tshape, pad=pad, margin=margin, start=start, step=step)
	iters = []
	for map, io in zip(maps, mode):
		if   io == "r": iters.append(tiler.read (map))
		elif io == "w": iters.append(tiler.write(map))
		else: raise ValueError("Invalid mode character '%s'" % str(io))
	# Can't just return zip(*iters) because zip gives up when just
	# one iterator raises StopIteration. This doesn't allow the other
	# iterators to finish. Maybe this is hacky
	return utils.zip2(*iters)

class Padtiler:
	"""Helper class used to implement padtiles. See its docstring for details."""
	def __init__(self, tshape=600, pad=60, margin=60, start=0, step=1):
		self.tshape = tuple(np.broadcast_to(tshape, 2).astype(int))
		self.pad    = tuple(np.broadcast_to(pad,    2).astype(int))
		self.margin = tuple(np.broadcast_to(margin, 2).astype(int))
		oly, olx    = 2*np.array(self.pad,int) # overlap region size
		self.wy     = (np.arange(oly)+1)/(oly+1)
		self.wx     = (np.arange(olx)+1)/(olx+1)
		self.start  = start
		self.step   = step
	def _tbound(self, tile, tsize, n):
		pix1 = tile*tsize
		pix2 = (tile+1)*tsize
		return pix1, pix2
	def read (self, imap): return self._it_helper(imap, mode="read")
	def write(self, omap): return self._it_helper(omap, mode="write")
	def _it_helper(self, map, mode):
		# Loop over tiles
		nty, ntx = (np.array(map.shape[-2:],int)+self.tshape-1)//self.tshape
		growy, growx = np.array(self.pad) + self.margin
		oly, olx = 2*np.array(self.pad) # overlap region size
		for ti in range(self.start, nty*ntx, self.step):
			ty = ti // ntx
			tx = ti %  ntx
			y1, y2 = self._tbound(ty, self.tshape[-2], map.shape[-2])
			x1, x2 = self._tbound(tx, self.tshape[-1], map.shape[-1])
			# Construct padded pixel box and extract it
			pixbox = np.array([[y1-growy,x1-growx],[y2+growy,x2+growx]])
			tile   = map.extract_pixbox(pixbox).copy()
			if mode == "read":
				yield tile
			else:
				tile[:] = 0
				yield tile
				# Before the next iteration, take the changes the user
				# made to tile, cop off the margin, and apply crossfading
				# weights so the overlapping pad regions add up to 1, and
				# then add to the output map
				tile = tile[...,self.margin[-2]:tile.shape[-2]-self.margin[-2],self.margin[-1]:tile.shape[-1]-self.margin[-1]]
				# Apply crossfade weights
				if ty > 0: tile[...,:oly,:] *= self.wy[:,None]
				if tx > 0: tile[...,:,:olx] *= self.wx[None,:]
				if ty < nty-1: tile[...,tile.shape[-2]-oly:,:] *= self.wy[::-1,None]
				if tx < ntx-1: tile[...,:,tile.shape[-1]-olx:] *= self.wx[None,::-1]
				# And add into output map
				map.insert(tile, op=lambda a,b:a+b)
