"""This module defines shortcuts for generating WCS instances and working
with them. The bounding boxes and shapes used in this module all use
the same ordering as WCS, i.e. column major (so {ra,dec} rather than
{dec,ra}). Coordinates are assigned to pixel centers, as WCS does natively,
but bounding boxes include the whole pixels, not just their centers, which
is where the 0.5 stuff comes from."""
import numpy as np, warnings
from astropy.wcs import WCS, FITSFixedWarning

# Turn off annoying warning every time a WCS object is constructed
try:
	warnings.filterwarnings("ignore", category=FITSFixedWarning)
except AssertionError:
	# This try/catch is a hack for readthedocs builds.
	pass

# Handle annoying python3 stuff
try: basestring
except: basestring = str
def streq(x, s): return isinstance(x, basestring) and x == s

# Geometry construction redesign
#
# The old approach was build around the reference point. The idea was that
# this point would always be a pixel center, no matter which coutout of
# the sky one was looking at. The problem with this approach is that it doesn't
# generalize to downgrading, and it clashes with finer detalies such as
# distinguishing between CC and Fejer1, which care about the pixel alignment
# at the poles, not the equator where the reference point usually is.
#
# The new approach will proceed in three steps:
# 1. Specify the projection (ctype, crval) without any pixel details
# 2. Turn it into a full-sky pixelization (crpix, cdelt). This could
#    be done by specifying ny,nx or the resolution. We could here
#    issue a warning or exception if the sky isn't evenly tiled.
#    This part would care about sub-specifiers like :cc or :fejer1
# 3. Crop this to cover the target area
#
# These can all be handled in separate functions. The output from step
# 1 would be a wcs with default crpix and cdelt values.
#
# Problems:
# 1. Currently pixelization can't handle all of these:
#    * Fix left side but allow right side to float
#    * Fix right side but allow left side to float
#    * Fix both sides, but allow total width to float
#    Right now the first two are supported, but not the last.
#    For example, for CEA we can't expect a reasonable resolution
#    to reach the poles with a senible pixel offset, but we want to
#    at least make things symmetric around the equator. There's a choice
#    between trying to get a pixel edge as close to the poles as possible
#    or trying to get a pixel center as close to the poles as possible.
#    These could be written as hh adjust and 00 adjust, but hard to fit
#    this in currently.
# 2. Some projections have extra parameters, like CEA with lambda and
#    ZEA where one might want to be locally conformal. The approach where
#    one first builds the fullsky geometry and only later worried about
#    restricting it to a part of the sky clashes with this. Can support it,
#    but is it a good idea to do it automatically?

def projection(system, crval=None):
	"""Generate a pixelization-agnostic wcs"""
	system = system.lower()
	if crval is None: crval = default_crval(system)
	else: crval = np.zeros(2)+crval
	if system in ["", "plain"]: return explicit(crval=crval)
	return explicit(ctype=["RA---"+system.upper(), "DEC--"+system.upper()], crval=crval)

def pixelization(pwcs, shape=None, res=None, variant=None):
	"""Add pixel information to a wcs, returning a full-sky geometry,
	or as close to that as the projection allows."""
	# This is the hard part. Many projections have invalid areas, and
	# some have infinite size. May just have to handle the cases one by
	# one instead of trying to be general
	system   = get_proj(pwcs)
	extent, lonpole   = default_extent(system)
	variant  = variant or default_variant(system)
	offs     = parse_variant(variant)
	periodic = is_periodic(system)
	# We will now split our extent into pixels. Find the intermediate
	# coordinates of the first and last pixel center along each axis
	if shape is None:
		res = expand_res(res)
		ra1, ra2, nx, ox1, ox2 = pixelize_1d(extent[0], res=res[0],  offs=offs[0], periodic=periodic[0])
		dec1,dec2,ny, oy1, oy2 = pixelize_1d(extent[1], res=res[1],  offs=offs[1], periodic=periodic[1])
	elif res is None:
		ra1, ra2, nx, ox1, ox2 = pixelize_1d(extent[0], n=shape[-2], offs=offs[0], periodic=periodic[0])
		dec1,dec2,ny, oy1, oy2 = pixelize_1d(extent[1], n=shape[-2], offs=offs[1], periodic=periodic[0])
	else:
		raise ValueError("Either res or shape must be given to build a pixelization")
	# Now that we have the intermediate coordinates of our endpoints, we
	# can calculate cdelt and crpix
	owcs  = pwcs.deepcopy()
	owcs.wcs.cdelt = [(ra2-ra1)/(nx-1), (dec2-dec1)/(ny-1)]
	# The bottom-left corner has pixel coordinates -ox1,-oy1
	# The top-right   corner has pixel coordinates (nx-1)-ox2,(ny-1)-oy2
	# The center is the average of these
	owcs.wcs.crpix[0] = 1+((nx-1)-ox2-ox1)/2
	owcs.wcs.crpix[1] = 1+((ny-1)-oy2-oy1)/2
	if lonpole is not None:
		owcs.wcs.lonpole = lonpole
	return (ny,nx), owcs

def explicit(naxis=2, **args):
	wcs = WCS(naxis=naxis)
	for key in args:
		setattr(wcs.wcs, key, args[key])
	return wcs

def expand_res(res, signs=None, flip=False):
	"""If res is not None, expand it to length 2. If it wasn't already
	length 2, the RA sign will be inverted. If flip is True, the res order
	will be flipped before expanding"""
	if res is None: return res
	# Bleh, compensate for later flip
	if signs is None: signs = [1,-1] if flip else [-1,1]
	res = np.atleast_1d(res)
	assert res.ndim == 1, "Invalid res shape"
	assert len(res) <= 2, "Invalid res length"
	if flip: res, signs = res[::-1], signs[::-1]
	if res.size == 1: res = np.array(signs)*res[0]
	return res

def describe(wcs):
	"""Since astropy.wcs.WCS objects do not have a useful
	str implementation, this function provides a relpacement."""
	sys  = wcs.wcs.ctype[0][-3:].lower()
	n    = wcs.naxis
	fields = ("cdelt:["+",".join(["%.4g"]*n)+"],crval:["+",".join(["%.4g"]*n)+"],crpix:["+",".join(["%.2f"]*n)+"]") % (tuple(wcs.wcs.cdelt) + tuple(wcs.wcs.crval) + tuple(wcs.wcs.crpix))
	pv = wcs.wcs.get_pv()
	for p in pv:
		fields += ",pv[%d,%d]=%.3g" % p
	return "%s:{%s}" % (sys, fields)
# Add this to all WCSes in this class
WCS.__repr__ = describe
WCS.__str__ = describe

def equal(wcs1, wcs2,flags=1,tol=1e-14):
	return wcs1.wcs.compare(wcs2.wcs, flags, tol)

def nobcheck(wcs):
	res = wcs.deepcopy()
	res.wcs.bounds_check(False, False)
	return res

def is_compatible(wcs1, wcs2, tol=1e-3):
	"""Checks whether two world coordinate systems represent
	(shifted) versions of the same pixelizations, such that
	every pixel center in wcs1 correspond to a pixel center in
	wcs2. For now, they also have to have the pixels going
	in the same direction."""
	h1 = wcs1.to_header()
	h2 = wcs2.to_header()
	keys = sorted(list(set(h1.keys())&set(h2.keys())))
	for key in keys:
		if key.startswith("CRVAL") or key.startswith("CRPIX") or key.startswith("CDELT"): continue
		if key not in h2 or h2[key] != h1[key]: return False
	if np.max(np.abs(wcs1.wcs.cdelt-wcs2.wcs.cdelt))/np.min(np.abs(wcs1.wcs.cdelt)) > tol: return False
	crdelt = wcs1.wcs.crval - wcs2.wcs.crval
	cpdelt = wcs1.wcs.crpix - wcs2.wcs.crpix
	subpix = (crdelt/wcs1.wcs.cdelt - cpdelt + 0.5)%1-0.5
	if np.max(np.abs(subpix)) > tol: return False
	return True

def is_plain(wcs):
	"""Determines whether the given wcs represents plain, non-specific,
	non-wrapping coordinates or some angular coordiante system."""
	return get_proj(wcs) in ["","plain"]

def is_cyl(wcs):
	"""Returns True if the wcs represents a cylindrical coordinate system"""
	return get_proj(wcs) in ["cyp","cea","car","mer"]

def is_separable(wcs):
	return is_cyl(wcs) and wcs.wcs.crval[1] == 0

def get_proj(wcs):
	if isinstance(wcs, str): return wcs
	else:
		toks = wcs.wcs.ctype[0].split("-")
		return toks[-1].lower() if len(toks) >= 2 else ""

def parse_system(system, variant=None):
	toks = system.split(":")
	if len(toks) > 1: return toks[0].lower(), toks[1]
	else: return toks[0].lower(), variant

def scale(wcs, scale=1, rowmajor=False, corner=True):
	"""Scales the linear pixel density of a wcs by the given factor, which can be specified
	per axis. This is the same as dividing the pixel size by the same numberr
	corner controls which area is scaled. With corner=True (the default), then the
	area from the start of the first pixel to the end of the lats pixel will be scaled
	by this factor. If corner=False, then the area from the center of the first pixel
	to the center of the last pixel will be scaled. Usually the former makes most sense."""
	scale = np.zeros(2)+scale
	if rowmajor: scale = scale[::-1]
	wcs = wcs.deepcopy()
	if corner:
		wcs.wcs.crpix -= 0.5
	wcs.wcs.crpix *= scale
	wcs.wcs.cdelt /= scale
	if corner:
		wcs.wcs.crpix += 0.5
	return wcs

#def expand_res(res, default_dirs=[1,-1]):
#	res = np.atleast_1d(res)
#	assert res.ndim == 1, "Invalid res shape"
#	if res.size == 1:
#		return np.array(default_dirs)*res
#	else:
#		return res

###########################
#### Helper functions #####
###########################

def is_azimuthal(system): return system.lower() in ["arc", "zea", "sin", "tan", "azp", "slp", "stg", "zpn", "air"]

def default_crval(system):
	if is_azimuthal(system): return [0,90]
	else: return [0,0]

def default_extent(system):
	"""Return the horizontal and vertical extent of the full sky in
	degrees, and the prefered value of lonpole (or None if it should
	be left alone).  For some systems the full sky is not
	representable, in which case a reasonable compromise is returned

	"""
	system = system.lower()
	if   system in ["", "plain"]: return [1,1], None
	# Cylindrical
	if   system == "car": return [360,180], None
	elif system == "cea": return [360,360/np.pi], None
	elif system == "mer": return [360,360], None # traditional dec range gives square map
	# Zenithal
	elif system == "arc": return [360,360], 180.
	elif system == "zea": return [720/np.pi,720/np.pi], 180.
	elif system == "sin": return [360/np.pi,360/np.pi], 180. # only orthographic supported
	elif system == "tan": return [360,360], 180. # goes down to 0.158° above the horizon
	# Pseudo-cyl
	elif system == "mol": return [720*2**0.5/np.pi,360*2**0.5/np.pi], None
	elif system == "ait": return [720*2**0.5/np.pi,360*2**0.5/np.pi], None
	else: raise ValueError("Unsupported system '%s'" % str(system))

def default_variant(system):
	system = system.lower()
	return "fejer1" if system in ["car","plain",""] else "any"

def extent2bounds(extent): return [[-e/h,e/h] for e in extent]

def is_periodic(system):
	system = system.lower()
	if is_azimuthal(system) or system in ["", "plain"]:
		return [False,False]
	else:
		return [True,False]

def parse_variant(name):
	name = name.lower()
	if   name == "safe":   rule = "hh,hh" # fully-downgrade safe. What fejer1 should have been
	elif name == "fejer1": rule = "00,hh" # stays SHTable after downgrade, but not pix-comp with raw @ that res
	elif name == "cc":     rule = "00,00" # what we used for pre-DR6. Cannot SHT after downgrade
	elif name == "any":    rule = "**,**"
	else: rule = name
	toks = rule.split(",")
	if len(toks) != 2 or len(toks[0]) != 2 or len(toks[1]) != 2:
		raise ValueError("Could not recognize pixelization variant '%s'" % (str(name)))
	left  = {"0": 0, "h": 0.5, "*": None}
	right = {"0": 0, "h":-0.5, "*": None}
	try:
		return [[left[tok[0]],right[tok[1]]] for tok in toks]
	except KeyError:
		raise ValueError("Invalid character in rule '%s'" % str(rule))

class PixelizationError(Exception): pass

def pixelize_1d(w, n=None, res=None, offs=None, periodic=False, adjust=False, sign=1, tol=1e-6, eps=1e-6):
	"""Figure out how to align pixels along an interval w long such
	that there are either n pixels or the resolution is res, and with
	the given pixel offsets from the edges. Returns the coordinates of
	the center of the first and last pixel."""
	# FIXME: This is a bit poorly thought out. The concept of being
	# able to adjust the range and that of having wildcard edges should
	# be separate, but the way we've done things now there's no room in
	# parse_variant to say something like "0h" but adjustable. For now
	# I just have to hardcode that "**" means "00" but adjustable.
	o1, o2 = offs if offs is not None else (None, None)
	if res is not None:
		if res < 0: res, sign = -res, -sign
		if o1 is None and o2 is None:
			o1 = o2 = 0
			adjust = True
		if o2 is None:
			# Add a tiny number to avoid having a rounding discontinuity for common values
			# off w, res and o1
			n   = int(w/res+1-o1+eps)
		elif o1 is None:
			n   = int(w/res+1+o2+eps)
		else:
			# Both given! Can we satisfy requirement?
			nf = w/res+1-(o1-o2)
			n  = int(nf+eps)
			if adjust:
				# We're free to redefine w so things work
				w = (n+(o1+o2)-1)*res
			else:
				# Complain if the resolution and offsets are incompatible
				if not np.abs(n-nf)<tol:
					raise PixelizationError("Resolution %g does not evenly divide extent %g with offsets [%g,%g]" %
						(res, w, o1, o2))
	else:
		if o1 is None: o1 =  0.5
		if o2 is None: o2 = -0.5
		res = w/(n-1+o1-o2)
	# Finish up
	if o1 is not None:
		ra1 = -w/2+o1*res
		ra2 = ra1+(n-1)*res
	else:
		ra2 = +w/2+o2*res
		ra1 = ra2 - (n-1)*res
	# If this axis is periodic, then the last point could be equal to the first,
	# depending on the pixelization
	if periodic and np.allclose(ra2-ra1,w):
		ra2 -= res
		n   -= 1
	# Apply any sign
	ra1 *= sign
	ra2 *= sign
	return ra1, ra2, n, o1, o2

def recenter_cyl_x(wcs, x):
	"""Given a cylindrical wcs with the reference point already on the equator,
	move the reference point along the equator to the given x (counting from 1)
	returning a new wcs."""
	if not is_separable(wcs):
		raise ValueError("recenter_cyl requires a cylindrical wcs with crval on the equator")
	owcs = wcs.deepcopy()
	owcs.wcs.crpix[0]  = x
	owcs.wcs.crval[0] += (x-wcs.wcs.crpix[0])*wcs.wcs.cdelt[0]
	return owcs

def recenter_cyl_ra(wcs, ra):
	return recenter_cyl_x(wcs.wcs.crpix[0] + (ra-wcs.wcs.crval[0])/wcs.wcs.cdelt[0])

def fix_wcs(wcs, axis=0):
	"""Returns a new WCS object which has had the reference pixel moved to the
	middle of the possible pixel space."""
	res = wcs.deepcopy()
	# Find the center ra manually: mean([crval - crpix*cdelt, crval + (-crpix+shape)*cdelt])
	#  = crval + (-crpix+shape/2)*cdelt
	# What pixel does this correspond to?
	#  crpix2 = crpix + (crval2-crval)/cdelt
	# But that requires shape. Can we do without it? Yes, let's use the
	# biggest possible shape. n = 360/cdelt
	n = abs(360/wcs.wcs.cdelt[axis])
	delta_ra  = wcs.wcs.cdelt[axis]*(n/2-wcs.wcs.crpix[axis])
	delta_pix = delta_ra/wcs.wcs.cdelt[axis]
	res.wcs.crval[axis] += delta_ra
	res.wcs.crpix[axis] += delta_pix
	repr(res.wcs) # wcs not properly updated if I don't do this
	return res

def fix_cdelt(wcs):
	"""Return a new wcs with pc and cd replaced by cdelt"""
	owcs = wcs.deepcopy()
	if wcs.wcs.has_cd():
		del owcs.wcs.cd, owcs.wcs.pc
		owcs.wcs.cdelt *= np.diag(wcs.wcs.cd)
	elif wcs.wcs.has_pc():
		del owcs.wcs.cd, owcs.wcs.pc
		owcs.wcs.cdelt *= np.diag(wcs.wcs.pc)
	return owcs

# The functions below are used to implement the old
# patch-oriented geometry functions.

# The origin argument used in the wcs pix<->world routines seems to
# have to be 1 rather than the 0 one would expect. For example,
# if wcs is CAR(crval=(0,0),crpix=(0,0),cdelt=(1,1)), then
# pix2world(0,0,1) is (0,0) while pix2world(0,0,0) is (-1,-1).
#
# No! the problem is that everythin in the fits header counts from 1,
# so the default crpix should be (1,1), not (0,0). With
# CAR(crval(0,0),crpix(1,1),cdelt(1,1)) we get
# pix2world(1,1,1) = (0,0) and pix2world(0,0,0) = (0,0)

# Useful stuff to be able to do:
#  * Create a wcs from (point,res)
#  * Create a wcs from (box,res)
#  * Create a wcs from (box,shape)
#  * Create a wcs from (point,res,shape)
# Can support this by taking arguments:
#  pos: point[2] or box[2,2], mandatory
#  res: num or [2], optional
#  shape: [2], optional
# In cases where shape is not specified, the implied
# shape can be recovered from the wcs and a box by computing
# the pixel coordinates of the corners. So we don't need to return
# it.

#  1. Construct wcs from box, res (and return shape?)
#  2. Construct wcs from box, shape
#  3. Construct wcs from point, res (this is the most primitive version)

# I need to update this to work better with full-sky stuff.
# Should be easy to construct something that's part of a
# clenshaw-curtis or fejer sky.

deg2rad = np.pi/180
rad2deg = 1/deg2rad

def plain(pos, res=None, shape=None, rowmajor=False, ref=None):
	"""Set up a plain coordinate system (non-cyclical)"""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor, default_dirs=[1,1])
	w = WCS(naxis=2)
	w.wcs.crval = mid
	if streq(ref, "standard"): ref = None
	return finalize(w, pos, res, shape, ref=ref)

def car(pos, res=None, shape=None, rowmajor=False, ref=None):
	"""Set up a plate carree system. See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
	w.wcs.crval = np.array([mid[0],0])
	if streq(ref, "standard"): ref = (0,0)
	return finalize(w, pos, res, shape, ref=ref)

def cea(pos, res=None, shape=None, rowmajor=False, lam=None, ref=None):
	"""Set up a cylindrical equal area system. See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	if lam is None:
		lam = np.cos(mid[1]*deg2rad)**2
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---CEA", "DEC--CEA"]
	w.wcs.set_pv([(2,1,lam)])
	w.wcs.crval = np.array([mid[0],0])
	if streq(ref, "standard"): ref = (0,0)
	return finalize(w, pos, res, shape, ref=ref)

def mer(pos, res=None, shape=None, rowmajor=False, ref=None):
	"""Set up a mercator system. See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---MER", "DEC--MER"]
	w.wcs.crval = np.array([mid[0],0])
	if streq(ref, "standard"): ref = (0,0)
	return finalize(w, pos, res, shape, ref=ref)

def arc(pos, res=None, shape=None, rowmajor=False, ref=None):
	"""Setups up a zenithal equidistant projection.  See the build
	function for details.

	"""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---ARC", "DEC--ARC"]
	w.wcs.crval = mid
	w, ref = _apply_zenithal_ref(w, ref)
	return finalize(w, pos, res, shape, ref=ref)

def sin(pos, res=None, shape=None, rowmajor=False, ref=None):
	"""Setups up an orthographic projection.  See the build function
	for details.

	"""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
	w.wcs.crval = mid
	w, ref = _apply_zenithal_ref(w, ref)
	return finalize(w, pos, res, shape, ref=ref)

def zea(pos, res=None, shape=None, rowmajor=False, ref=None):
	"""Setups up an oblate Lambert's azimuthal equal area system.
	See the build function for details. Don't use this if you want
	a polar projection."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---ZEA", "DEC--ZEA"]
	w.wcs.crval = mid
	w, ref = _apply_zenithal_ref(w, ref)
	return finalize(w, pos, res, shape, ref=ref)

# The airy distribution is a bit different, since is needs to
# know the size of the patch.
def air(pos, res=None, shape=None, rowmajor=False, rad=None, ref=None):
	"""Setups up an Airy system. See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	if rad is None:
		if pos.ndim != 2:
			raise ValueError("Airy requires either rad or pos[2,2]")
		w = angdist(mid[0]*deg2rad,pos[0,1]*deg2rad,mid[0]*deg2rad,pos[1,1]*deg2rad)*rad2deg
		h = angdist(pos[0,0]*deg2rad,mid[1]*deg2rad,pos[1,0]*deg2rad,mid[1]*deg2rad)*rad2deg
		rad = (w+h)/4
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---AIR","DEC--AIR"]
	w.wcs.set_pv([(2,1,90-rad)])
	w, ref = _apply_zenithal_ref(w, ref)
	return finalize(w, pos, res, shape, ref=ref)

def tan(pos, res=None, shape=None, rowmajor=False, ref=None):
	"""Set up a gnomonic (tangent plane) system. See the build function for details."""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
	w = WCS(naxis=2)
	w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
	w.wcs.crval = mid
	w, ref = _apply_zenithal_ref(w, ref)
	return finalize(w, pos, res, shape, ref=ref)

systems = {"car": car, "cea": cea, "mer": mer, "air": air, "arc": arc, "sin": sin, "zea": zea, "tan": tan, "gnom": tan, "plain": plain }

def build(pos, res=None, shape=None, rowmajor=False, system="cea", ref=None, **kwargs):
	"""Set up the WCS system named by the "system" argument.
	pos can be either a [2] center position or a [{from,to},2]
	bounding box. At least one of res or shape must be specified.
	If res is specified, it must either be a number, in
	which the same resolution is used in each direction,
	or [2]. If shape is specified, it must be [2]. All angles
	are given in degrees."""
	return systems[system.lower()](pos, res, shape, rowmajor, ref=ref, **kwargs)

def validate(pos, res, shape, rowmajor=False, default_dirs=[1,-1]):
	pos = np.asarray(pos)
	if pos.shape != (2,) and pos.shape != (2,2):
		raise ValueError("pos must be [2] or [2,2]")
	if res is None and shape is None:
		raise ValueError("Atleast one of res and shape must be specified")
	if res is not None:
		res = np.atleast_1d(res)
		if res.shape == (1,):
			# If our shape has one entry, expand it to [y,x].
			# Two cases: 1. [2,2] pos given, in which case it has responsibility for
			# the coordinate directions, so we don't introduce a sign here, and
			# 2. [2] pos is given, in which case it's res's responsibility.
			if pos.shape == (2,2): res = np.zeros(2)+res
			else:                  res = np.array(default_dirs)*res
		elif res.shape != (2,):
			raise ValueError("res must be num or [2]")
	if rowmajor:
		pos = pos[...,::-1]
		if shape is not None: shape = shape[::-1]
		if res is not None: res = res[::-1]
	if shape is not None:
		shape = shape[:2]
	if res is None and pos.ndim != 2:
		raise ValueError("pos must be a bounding box if res is not specified")
	mid = pos if pos.ndim == 1 else np.mean(pos,0)
	return pos, res, shape, mid

def finalize(w, pos, res, shape, ref=None):
	"""Common logic for the various wcs builders. Fills in the reference
	pixel and resolution."""
	w.wcs.crpix = [1,1]
	if res is None:
		# Find the resolution that gives our box the required extent.
		w.wcs.cdelt = [1,1]
		corners = w.wcs_world2pix(pos,1)
		w.wcs.cdelt *= (corners[1]-corners[0])/shape
	else:
		w.wcs.cdelt = res
		if pos.ndim == 2: w.wcs.cdelt[pos[1]<pos[0]] *= -1
	if pos.ndim == 1:
		if shape is not None:
			# Place pixel origin at corner of shape centered on crval
			off = w.wcs_world2pix(pos[None],0)[0]
			w.wcs.crpix = np.array(shape)/2.0+0.5 - off
	else:
		# Make pos[0] the corner of the (0,0) pixel (counting from 0 for simplicity)
		off = w.wcs_world2pix(pos[0,None],0)[0]+0.5
		w.wcs.crpix -= off
	if ref is not None:
		# Tweak wcs so that crval is an integer number of
		# pixels away from ref.  This is most straight-forward
		# if one simply adjusts crpix.
		off = (w.wcs_world2pix(np.asarray(ref)[None], 1)[0] + 0.5) % 1 - 0.5
		w.wcs.crpix -= off
	return w

def _apply_zenithal_ref(w, ref):
	"""Input is a wcs w and ref is a position (dec,ra) or a special value
	(None, 'standard').  Returns tuple (w, ref_out).  If ref is a
	position, it is copied into w.wcs.crval and ref_out=ref.
	Otherwise, w is unmodified and ref_out=w.wcs.crval.  Also sets lonpole,
	if not already set, to 180, which is sensible default."""
	if np.isnan(w.wcs.lonpole):
		w.wcs.lonpole = 180.
	if isinstance(ref, str) and ref == 'standard':
		ref = None
	if ref is None:
		ref = w.wcs.crval
	else:
		w.wcs.crval = ref
	return w, ref

def angdist(lon1,lat1,lon2,lat2):
	return np.arccos(np.cos(lat1)*np.cos(lat2)*(np.cos(lon1)*np.cos(lon2)+np.sin(lon1)*np.sin(lon2))+np.sin(lat1)*np.sin(lat2))

def center_cyl_wcs(wcs, shape=None, off=0.5):
	"""Given the wcs for a cylindrical projection, return a new wcs
	where the reference point has been moved along the equator to
	the middle of the patch. This ensures that all pixels are within
	the standard WCS bounds. If shape is not passed, then the patch
	is assumed to cover the whole width of the sky."""
	# Can't manipulate crval if coordinates aren't separable
	if not is_separable(wcs):
		raise ValueError("Can't fix wcs for non-separable wcs")
	# Patch horizontal extent
	if shape is None: n = abs(360/wcs.wcs.cdelt[axis])
	else:             n = shape[-1]
	# x pixel of center  (1-based)
	x  = (n-1)/2+1
	# corresponding RA
	ra = wcs.wcs.crval[0] + (x-wcs.wcs.crpix[0])*wcs.wcs.cdelt[0]
	# We will allow negative RA, but we prefer small, positive values.
	# If a patch goes at most 180° away from crval, then we can require
	# 0 <= crval <= 360. But we add a small offset to not be sensitive to
	# tiny errors for our common choice of crval[0] = 0
	ra = (ra-off) % 360 + off
	# Make a new wcs with this
	owcs = wcs.deepcopy()
	owcs.wcs.crval[0] = ra
	owcs.wcs.crpix[0] = x
	return owcs

def fix_wcs(wcs, axis=0, n=None):
	"""Compatibility name for center_cyl_wcs"""
	if axis != 0: raise NotImplementedError
	return center_cyl_wcs(wcs, None if n is None else (1,n))

def fix_cdelt(wcs):
	"""Return a new wcs with pc and cd replaced by cdelt"""
	owcs = wcs.deepcopy()
	if wcs.wcs.has_cd():
		del owcs.wcs.cd, owcs.wcs.pc
		owcs.wcs.cdelt *= np.diag(wcs.wcs.cd)
	elif wcs.wcs.has_pc():
		del owcs.wcs.cd, owcs.wcs.pc
		owcs.wcs.cdelt *= np.diag(wcs.wcs.pc)
	return owcs
