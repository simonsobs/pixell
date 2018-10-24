from __future__ import print_function
import numpy as np
from . import utils
import astropy.coordinates as c, astropy.units as u
# Optional dependencies are imported in the functions that
# use them. These include enlib.ephem, enlib.iers and enlib.pyfsla

# Python 2/3 compatibility
try: basestring
except NameError: basestring = str


class default_site:
	lat  = -22.9585
	lon  = -67.7876
	alt  = 5188.
	T    = 273.15
	P    = 550.
	hum  = 0.2
	freq = 150.
	lapse= 0.0065
	base_tilt =    0.0107693
	base_az   = -114.9733961

def transform(from_sys, to_sys, coords, time=55500, site=default_site, pol=None, mag=None, bore=None):
	"""Transforms coords[2,...] from system from_sys to system to_sys, where
	systems can be "hor", "cel" or "gal". For transformations involving
	"hor", the optional arguments time (in modified julian days) and site (which must
	contain .lat (rad), .lon (rad), .P (pressure, mBar), .T (temperature, K),
	.hum (humidity, 0.2 by default), .alt (altitude, m)). Returns an array
	with the same shape as the input. The coordinates are in ra,dec-ordering."""
	from_info, to_info = getsys_full(from_sys,time,site,bore=bore), getsys_full(to_sys,time,site,bore=bore)
	ihand = get_handedness(from_info[0])
	ohand = get_handedness(to_info[0])
	# Apply the specified transformation, optionally computing the induced
	# polarization rotation and apparent magnification
	def transfunc(coords):
		return transform_raw(from_info, to_info, coords, time=time, site=site, bore=bore)
	fields = []
	if pol: fields.append("ang")
	if mag: fields.append("mag")
	if pol is None and mag is None:
		if len(coords) > 2: fields.append("ang")
		if len(coords) > 3: fields.append("mag")
	meta = transform_meta(transfunc, coords[:2], fields=fields)

	# Fix the polarization convention. We use healpix
	if "ang" in fields:
		if ihand != ohand: meta.ang -= np.pi
		if ohand != 'L':   meta.ang = -meta.ang
	# Create the output array. This is a bit cumbersome because
	# each of the output columns can be either ang or mag, which
	# might or might not have previous values that need to be
	# updated. It is this way to keep backward compatibility.
	res = np.zeros((2+len(fields),) + meta.ocoord.shape[1:])
	res[:2] = meta.ocoord
	off = 2
	for i, f in enumerate(fields):
		if f == "ang":
			if len(coords) > 2: res[off+i] = coords[2] + meta.ang
			else: res[off+i] = meta.ang
		elif f == "mag":
			if len(coords) > 3: res[off+i] = coords[3] * meta.mag
			else: res[off+i] = meta.mag
	return res

def transform_meta(transfun, coords, fields=["ang","mag"], offset=5e-7):
	"""Computes metadata for the coordinate transformation functor
	transfun applied to the coordinate array coords[2,...],
	such as the induced rotation, magnification.

	Currently assumes that input and output coordinates are in
	non-zenith polar coordinates. Might generalize this later.
	"""
	if "mag_brute" in fields: ntrans = 3
	elif "ang" in fields: ntrans = 2
	else: ntrans = 1
	coords  = np.asarray(coords)
	offsets = np.array([[0,0],[1,0],[0,1]])*offset
	# Transform all the coordinates. We assume we aren't super-close to the poles
	# either before or after the transformation.
	ocoords = np.zeros((ntrans,2)+coords.shape[1:])
	ocoords = None
	for i in range(ntrans):
		# Transpose to get broadcasting right
		a = transfun((coords.T + offsets[i].T).T)
		if ocoords is None:
			ocoords = np.zeros((ntrans,)+a.shape, a.dtype)
		ocoords[i] = a

	class Result: pass
	res = Result()
	res.icoord = coords
	res.ocoord = ocoords[0]

	# Compute the individual properties we're interested in
	diff = utils.rewind(ocoords[1:]-ocoords[0,None])
	if "ang" in fields:
		# We only need the theta offset of this one. We started with
		# an offset in the [1,0] direction, and want to know how
		# far we have rotated away from this direction. This
		# Uses the IAU tangent plane angle convention:
		# http://healpix.jpl.nasa.gov/html/intronode12.htm
		# and assumes that both input and putput coordinates have the
		# same handedness. This is not always the case, for example
		# with horizontal to celestial coordinate transformations.
		# In these cases, the caller must correct there resulting angle
		# manually.
		phiscale = np.cos(ocoords[0,1])
		res.ang = np.arctan2(diff[0,1],diff[0,0]*phiscale)
	if "mag" in fields:
		res.mag = np.cos(res.icoord[1])/np.cos(res.ocoord[1])
	if "mag_brute" in fields:
		# Compute the ratio of the areas of the triangles
		# made up by the three point-sets in the input and
		# output coordinates. This ratio is always 1 when
		# using physical areas, so we instead compute the
		# apparent areas here.
		def tri_area(diff):
			return 0.5*np.abs(diff[0,0]*diff[1,1]-diff[0,1]*diff[1,0])
		res.mag = (tri_area(diff).T/tri_area(offsets[1:]-offsets[0]).T).T
	return res

def transform_raw(from_sys, to_sys, coords, time=None, site=default_site, bore=None):
	"""Transforms coords[2,...] from system from_sys to system to_sys, where
	systems can be "hor", "cel" or "gal". For transformations involving
	"hor", the optional arguments time (in modified julian days) and site (which must
	contain .lat (rad), .lon (rad), .P (pressure, mBar), .T (temperature, K),
	.hum (humidity, 0.2 by default), .alt (altitude, m)). Returns an array
	with the same shape as the input. The coordinates are in ra,dec-ordering.

	coords and time will be broadcast such that the result has the same shape
	as coords*time[None]."""
	# Prepare input and output arrays
	if time is None:
		coords = np.array(coords)[:2]
	else:
		time   = np.asarray(time)
		coords = np.asarray(coords)
		# Broadasting. A bit complicated because we want to handle
		# both time needing to broadcast and coords needing to
		time   = time + np.zeros(coords[0].shape,time.dtype)
		coords = (coords.T + np.zeros(time.shape,coords.dtype)[None].T).T
	# flatten, so the rest of the code can assume that coordinates are [2,N]
	# and time is [N]
	oshape = coords.shape
	coords= np.ascontiguousarray(coords.reshape(2,-1))
	if time is not None: time = time.reshape(-1)
	# Perform the actual coordinate transformation. There are three classes of
	# transformations here:
	# 1. To/from object-centered coordinates
	# 2. cel-hor transformation, using slalib
	# 3. cel-gal transformation, using astropy
	(from_sys,from_ref), (to_sys,to_ref) = getsys_full(from_sys,time,site,bore=bore), getsys_full(to_sys,time,site,bore=bore)
	if from_ref is not None: coords[:] = decenter(coords, from_ref[0], restore=from_ref[1])
	while True:
		if from_sys == to_sys: break
		elif from_sys == "bore":
			coords[:] = bore2tele(coords, bore)
			from_sys = "tele"
		elif from_sys == "tele" and to_sys in ["bore"]:
			coords[:] = tele2bore(coords, bore)
			from_sys = "bore"
		elif from_sys == "tele":
			coords[:] = tele2hor(coords, site, copy=False)
			from_sys  = "altaz"
		elif from_sys == "altaz" and to_sys in ["tele","bore"]:
			coords[:] = hor2tele(coords, site, copy=False)
			from_sys  = "tele"
		elif from_sys == "altaz":
			coords[:] = hor2cel(coords, time, site, copy=False)
			from_sys = "icrs"
		elif from_sys == "icrs" and to_sys in ["altaz","tele","bore"]:
			coords[:] = cel2hor(coords, time, site, copy=False)
			from_sys = "altaz"
		else:
			to_sys_astropy = nohor(to_sys)
			coords[:] = transform_astropy(from_sys, to_sys_astropy, coords)
			from_sys = to_sys_astropy
	if to_ref is not None: coords[:] = recenter(coords, to_ref[0], restore=to_ref[1])
	return coords.reshape(oshape)

def transform_astropy(from_sys, to_sys, coords):
	"""As transform, but only handles the systems supported by astropy."""
	from_sys, to_sys = getsys(from_sys), getsys(to_sys)
	if from_sys == to_sys: return coords
	unit   = u.radian
	coords = c.SkyCoord(coords[0], coords[1], frame=from_sys, unit=unit)
	coords = coords.transform_to(to_sys)
	names  = coord_names[to_sys]
	return np.asarray([
		getattr(getattr(coords, names[0]),unit.name),
		getattr(getattr(coords, names[1]),unit.name)])


def hor2cel(coord, time, site, copy=True):
	from enlib import pyfsla
	from enlib import iers
	coord  = np.array(coord, copy=copy)
	trepr  = time[len(time)/2]
	info   = iers.lookup(trepr)
	ao = pyfsla.sla_aoppa(trepr, info.dUT, site.lon*utils.degree, site.lat*utils.degree, site.alt,
		info.pmx*utils.arcsec, info.pmy*utils.arcsec, site.T, site.P, site.hum,
		299792.458/site.freq, site.lapse)
	am = pyfsla.sla_mappa(2000.0, trepr)
	# This involves a transpose operation, which is not optimal
	pyfsla.aomulti(time, coord.T, ao, am)
	return coord

def cel2hor(coord, time, site, copy=True):
	from enlib import pyfsla
	from enlib import iers
	# This is very slow for objects near the horizon!
	coord  = np.array(coord, copy=copy)
	trepr  = time[len(time)/2]
	info   = iers.lookup(trepr)
	ao = pyfsla.sla_aoppa(trepr, info.dUT, site.lon*utils.degree, site.lat*utils.degree, site.alt,
		info.pmx*utils.arcsec, info.pmy*utils.arcsec, site.T, site.P, site.hum,
		299792.458/site.freq, site.lapse)
	am = pyfsla.sla_mappa(2000.0, trepr)
	# This involves a transpose operation, which is not optimal
	pyfsla.oamulti(time, coord.T, ao, am)
	return coord

def tele2hor(coord, site, copy=True):
	coord = np.array(coord, copy=copy)
	coord = euler_rot([site.base_az*utils.degree, site.base_tilt*utils.degree, -site.base_az*utils.degree], coord)
	return coord

def hor2tele(coord, site, copy=True):
	coord = np.array(coord, copy=copy)
	coord = euler_rot([site.base_az*utils.degree, -site.base_tilt*utils.degree, -site.base_az*utils.degree], coord)
	return coord

def tele2bore(coord, bore, copy=True):
	"""Transforms coordinates [{ra,dec},...] to boresight-relative coordinates given by the boresight pointing
	[{ra,dec},...] with the same shape as coords. After the rotation, the boresight will be at the zenith;
	things above the boresight will be at 'ra'=180 and things below will be 'ra'=0."""
	coord = np.array(coord, copy=copy)
	return recenter(coord, bore)

def bore2tele(coord, bore, copy=True):
	"""Transforms coordinates [{ra,dec},...] from boresight-relative coordinates given by the boresight pointing
	[{ra,dec},...] with the same shape as coords. After the rotation, the coordinates will be in telescope
	coordinates, which are similar to horizontal coordinates."""
	coord = np.array(coord, copy=copy)
	return decenter(coord, bore)

def euler_mat(euler_angles, kind="zyz"):
	"""Defines the rotation matrix M for a ABC euler rotation,
	such that M = A(alpha)B(beta)C(gamma), where euler_angles =
	[alpha,beta,gamma]. The default kind is ABC=ZYZ."""
	alpha, beta, gamma = euler_angles
	R1 = utils.rotmatrix(gamma, kind[2])
	R2 = utils.rotmatrix(beta,  kind[1])
	R3 = utils.rotmatrix(alpha, kind[0])
	return np.einsum("...ij,...jk->...ik",np.einsum("...ij,...jk->...ik",R3,R2),R1)

def euler_rot(euler_angles, coords, kind="zyz"):
	coords = np.asarray(coords)
	co     = coords.reshape(2,-1)
	M      = euler_mat(euler_angles, kind)
	rect   = utils.ang2rect(co, False)
	rect   = np.einsum("...ij,j...->i...",M,rect)
	co     = utils.rect2ang(rect, False)
	return co.reshape(coords.shape)

    
def euler_mat(euler_angles, kind="zyz"):
    """Defines the rotation matrix M for a ABC euler rotation,
    such that M = A(alpha)B(beta)C(gamma), where euler_angles =
    [alpha,beta,gamma]. The default kind is ABC=ZYZ."""
    alpha, beta, gamma = euler_angles
    R1 = utils.rotmatrix(gamma, kind[2])
    R2 = utils.rotmatrix(beta,  kind[1])
    R3 = utils.rotmatrix(alpha, kind[0])
    return np.einsum("...ij,...jk->...ik",np.einsum("...ij,...jk->...ik",R3,R2),R1)

def euler_rot(euler_angles, coords, kind="zyz"):
    coords = np.asarray(coords)
    co     = coords.reshape(2,-1)
    M      = euler_mat(euler_angles, kind)
    rect   = utils.ang2rect(co, False)
    rect   = np.einsum("...ij,j...->i...",M,rect)
    co     = utils.rect2ang(rect, False)
    return co.reshape(coords.shape)

def recenter(angs, center, restore=False):
    """Recenter coordinates "angs" (as ra,dec) on the location given by "center",
    such that center moves to the north pole."""
    # Performs the rotation E(0,-theta,-phi). Originally did
    # E(phi,-theta,-phi), but that is wrong (at least for our
    # purposes), as it does not preserve the relative orientation
    # between the boresight and the sun. For example, if the boresight
    # is at the same elevation as the sun but 10 degrees higher in az,
    # then it shouldn't matter what az actually is, but with the previous
    # method it would.
    #
    # Now supports specifying where to recenter by specifying center as
    # lon_from,lat_from,lon_to,lat_to
    if len(center) == 4: ra0, dec0, ra1, dec1 = center
    elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], center[0]*0, center[1]*0+np.pi/2
    if restore: ra1 += ra0
    return euler_rot([ra1,dec0-dec1,-ra0], angs, kind="zyz")

def decenter(angs, center, restore=False):
    """Inverse operation of recenter."""
    if len(center) == 4: ra0, dec0, ra1, dec1 = center
    elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], center[0]*0, center[1]*0+np.pi/2
    if restore: ra1 += ra0
    return euler_rot([ra0,dec1-dec0,-ra1],  angs, kind="zyz")

def nohor(sys): return sys if sys not in ["altaz","tele","bore"] else "icrs"
def getsys(sys): return str2sys[sys.lower()] if isinstance(sys,basestring) else sys
def get_handedness(sys):
	"""Return the handedness of the coordinate system sys, as seen from inside
	the celestial sphere, in the standard IAU convention."""
	if sys in ["altaz","tele","bore"]: return 'R'
	else: return 'L'

def getsys_full(sys, time=None, site=default_site, bore=None):
	"""Handles our expanded coordinate system syntax: base[:ref[:refsys]].
	This allows a system to be recentered on a given position or object.
	The argument can either be a string of the above format (with [] indicating
	optional parts), or a list of [base, ref, refsys]. Returns a parsed
	and expanded version, where the systems have been replaced by full
	system objects (or None), and the reference point has been expanded
	into coordinates (or None), and rotated into the base system.
	Coordinates are separated by _.

	Example: Horizontal-based coordinates with the Moon centered at [0,0]
	would be hor:Moon/0_0.

	Example: Put celestial coordinates ra=10, dec=20 at horizontal coordinates
	az=0, el=0: hor:10_20:cel/0_0:hor. Yes, this is horrible.
	
	Used to be sys:center_on/center_at:sys_of_center_coordinates. But much
	more flexible to do sys:center_on:sys/center_at:sys. This syntax
	would be backwards compatible, though it's starting to get a bit clunky.

	Big hack: If the system is "sidelobe", then we will use sidelobe-oriented
	centering instead of object-oriented centering. This will result in
	a coordinate system where the boresight has the zenith-mirrored
	position of what the object would have in zenith-relative coordinates.
	"""
	if isinstance(sys, basestring): sys = sys.split(":",1)
	else:
		try: sys = list(sys)
		except TypeError: sys = [sys]
	if len(sys) < 2: sys += [None]*(2-len(sys))
	base, ref = sys
	if base == "sidelobe":
		base = "bore"
		sidelobe = True
	else: sidelobe = False
	base = getsys(base)
	prevsys = base
	#refsys = getsys(refsys) if refsys is not None else base
	if ref is None: return [base, ref]
	if isinstance(ref, basestring):
		# In general ref is ref:refsys/refto:reftosys. Here
		# ref and refto are are either an object name or a position in the format
		# lat_lon. comma would have been preferable, but we reserve that
		# for from_sys,to_sys uses for backwards compatibility with
		# existing programs.
		ref_expanded = []
		for ref_refsys in ref.split("/"):
			# In our first format, ref is a set of coordinates in degrees
			toks = ref_refsys.split(":")
			r = toks[0]
			refsys = getsys(toks[1]) if len(toks) > 1 else prevsys
			try:
				r = np.asfarray(r.split("_"))*utils.degree
				assert(r.ndim == 1 and len(r) == 2)
				r = transform_raw(refsys, base, r[:,None], time=time, site=site, bore=bore)
			except ValueError:
				# Otherwise, treat as an ephemeris object
				r = ephem_pos(r, time)
				r = transform_raw("equ", base, r, time=time, site=site, bore=bore)
			ref_expanded += list(r)
			prevsys = refsys
		ref_coords = np.array(ref_expanded)
		ref = [ref_coords, sidelobe]
	return [base, ref]

def ephem_pos(name, mjd):
	"""Given the name of an ephemeris object from pyephem and a
	time in modified julian date, return its position in ra, dec
	in radians in equatorial coordinates."""
	import ephem
	mjd = np.asarray(mjd)
	djd = mjd + 2400000.5 - 2415020
	obj = getattr(ephem, name)()
	if mjd.ndim == 0:
		obj.compute(djd)
		return np.array([float(obj.a_ra), float(obj.a_dec)])
	else:
		res = np.empty((2,djd.size))
		for i, t in enumerate(djd.reshape(-1)):
			obj.compute(t)
			res[0,i] = float(obj.a_ra)
			res[1,i] = float(obj.a_dec)
		return res.reshape((2,)+djd.shape)

def interpol_pos(from_sys, to_sys, name_or_pos, mjd, site=default_site, dt=10):
	"""Given the name of an ephemeris object or a [ra,dec]-type position
	in radians in from_sys, compute its position in the specified coordinate system for
	each mjd. The mjds are assumed to be sampled densely enough that
	interpolation will work. For ephemeris objects, positions are
	computed in steps of 10 seconds by default (controlled by the dt argument)."""
	box  = utils.widen_box([np.min(mjd),np.max(mjd)], 1e-2)
	sub_nsamp = max(3,int((box[1]-box[0])*24.*3600/dt))
	sub_mjd = np.linspace(box[0], box[1], sub_nsamp, endpoint=True)
	if isinstance(name_or_pos, basestring):
		sub_from = ephem_pos(name_or_pos, sub_mjd)
	else:
		pos = np.asarray(name_or_pos)
		assert pos.ndim == 1
		sub_from = np.zeros([2,sub_nsamp])
		sub_from[:] = np.asarray(name_or_pos)[:,None]
	sub_pos = transform_raw(from_sys, to_sys, sub_from, time=sub_mjd, site=site)
	sub_pos[1] = utils.rewind(sub_pos[1], ref="auto")
	inds = (mjd-box[0])*(sub_nsamp-1)/(box[1]-box[0])
	full_pos= utils.interpol(sub_pos, inds[None], order=3)
	return full_pos

def make_mapping(dict): return {value:key for key in dict for value in dict[key]}
str2sys = make_mapping({
	"galactic": ["gal", "galactic"],
	"icrs":     ["equ", "equatorial", "cel", "celestial", "icrs"],
	"altaz":    ["altaz", "azel", "hor", "horizontal"],
	"tele":     ["tele","telescope"],
	"bore":     ["bore","boresight"],
	"barycentrictrueecliptic": ["ecl","ecliptic","barycentrictrueecliptic"],
	})
coord_names = {
	"galactic": ["l","b"],
	"icrs": ["ra","dec"],
	"altaz":["az","alt"],
	"barycentrictrueecliptic":["lon","lat"],
	"tele":["az","alt"],
	"bore":["az","alt"],
	}
