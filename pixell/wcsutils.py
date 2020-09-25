"""This module defines shortcuts for generating WCS instances and working
with them. The bounding boxes and shapes used in this module all use
the same ordering as WCS, i.e. column major (so {ra,dec} rather than
{dec,ra}). Coordinates are assigned to pixel centers, as WCS does natively,
but bounding boxes include the whole pixels, not just their centers, which
is where the 0.5 stuff comes from."""
import numpy as np, warnings
from astropy.wcs import WCS, FITSFixedWarning

# Turn off annoying warning every time a WCS object is constructed
warnings.filterwarnings("ignore", category=FITSFixedWarning) 
# Handle annoying python3 stuff
try: basestring
except: basestring = str
def streq(x, s): return isinstance(x, basestring) and x == s

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

deg2rad = np.pi/180
rad2deg = 1/deg2rad

def explicit(naxis=2, **args):
	wcs = WCS(naxis=naxis)
	for key in args:
		setattr(wcs.wcs, key, args[key])
	return wcs

def describe(wcs):
	"""Since astropy.wcs.WCS objects do not have a useful
	str implementation, this function provides a relpacement."""
	sys  = wcs.wcs.ctype[0][-3:].lower()
	n    = wcs.naxis
	fields = ("cdelt:["+",".join(["%.4g"]*n)+"],crval:["+",".join(["%.4g"]*n)+"],crpix:["+",".join(["%.4g"]*n)+"]") % (tuple(wcs.wcs.cdelt) + tuple(wcs.wcs.crval) + tuple(wcs.wcs.crpix))
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
	return wcs.wcs.ctype[0] == ""

def is_cyl(wcs):
	"""Returns True if the wcs represents a cylindrical coordinate system"""
	return wcs.wcs.ctype[0].split("-")[-1] in ["CYP","CEA","CAR","MER"]

def get_proj(wcs):
	toks = wcs.wcs.ctype[0].split("-")
	return toks[1].lower() if len(toks) == 2 else ""

def scale(wcs, scale=1, rowmajor=False, corner=False):
	"""Scales the linear pixel density of a wcs by the given factor, which can be specified
	per axis. This is the same as dividing the pixel size by the same number."""
	scale = np.zeros(2)+scale
	if rowmajor: scale = scale[::-1]
	wcs = wcs.deepcopy()
	if not corner:
		wcs.wcs.crpix -= 0.5
	wcs.wcs.crpix *= scale
	wcs.wcs.cdelt /= scale
	if not corner:
		wcs.wcs.crpix += 0.5
	return wcs

# I need to update this to work better with full-sky stuff.
# Should be easy to construct something that's part of a
# clenshaw-curtis or fejer sky.

def plain(pos, res=None, shape=None, rowmajor=False, ref=None):
	"""Set up a plain coordinate system (non-cyclical)"""
	pos, res, shape, mid = validate(pos, res, shape, rowmajor)
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

systems = {"car": car, "cea": cea, "mer": mer, "air": air, "zea": zea, "tan": tan, "gnom": tan, "plain": plain }

def build(pos, res=None, shape=None, rowmajor=False, system="cea", ref=None, **kwargs):
	"""Set up the WCS system named by the "system" argument.
	pos can be either a [2] center position or a [{from,to},2]
	bounding box. At least one of res or shape must be specified.
	If res is specified, it must either be a number, in
	which the same resolution is used in each direction,
	or [2]. If shape is specified, it must be [2]. All angles
	are given in degrees."""
	return systems[system.lower()](pos, res, shape, rowmajor, ref=ref, **kwargs)

def validate(pos, res, shape, rowmajor=False):
	pos = np.asarray(pos)
	if pos.shape != (2,) and pos.shape != (2,2):
		raise ValueError("pos must be [2] or [2,2]")
	if res is None and shape is None:
		raise ValueError("Atleast one of res and shape must be specified")
	if res is not None:
		res = np.atleast_1d(res)
		if res.shape == (1,):
			res = np.array([res[0],res[0]])
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
	Otherwise, w is unmodified and ref_out=w.wcs.crval."""
	if isinstance(ref, str) and ref == 'standard':
		ref = None
	if ref is None:
		ref = w.wcs.crval
	else:
		w.wcs.crval = ref
	return w, ref

def angdist(lon1,lat1,lon2,lat2):
	return np.arccos(np.cos(lat1)*np.cos(lat2)*(np.cos(lon1)*np.cos(lon2)+np.sin(lon1)*np.sin(lon2))+np.sin(lat1)*np.sin(lat2))

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
