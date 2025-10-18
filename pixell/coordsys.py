"""Plain coordinate transformations using qpoint and numpy-quaternion.
Used to implement the fast pointing interpolation in pmat.py"""
import re
import numpy as np
import qpoint
import quaternion
import copy
from . import bunch, warray, sites

# TODO:
# 1. Move to pixell.csys (any better name?)
# 2. Implement pixell.ephem (any better name? Clashes with pyephem).
#    Should provide a class for loading and caching my precomputed asteroid
#    ephemerides as well standard planet stuff from pyephem. Both should
#    be available through a common interface. May also need to build interpolation
#    for the planets on-the-fly if pyephem is too slow.
#    Maybe something like ephem.add_source(path, type), but the simple interpolator
#    should also be available as its own class, e.g. PrecompEphem.

DEG = np.pi/180

# TODO: unified planet and asteroid ephemeris in pixell? With a load method to load the asteroids?

def transform(isys, osys, coords, ctime=None, site=sites.default_site, weather=None):
	isys = expand_sys(isys, ctime=site, site=site, weather=weather)
	osys = expand_sys(osys, ctime=site, site=site, weather=weather)
	# expand_sys should return something with .base and .q properties, where .q can be None
	# 1. Undo any input rotation. I wish this could be done with /=.
	# I think this is possible by working in the inverse space, e.g.
	# coords **= -1; coords *= isys.q; coords *= hor_rots[isys.base]; coords **= -1;
	# coords = hor2equ; coords **= -1; etc. This would avoid unneccessary copies, but
	# would be confusing, and it would be hard to avoid some unnecessary inversions.
	# Could handle with .iq member. Implemented in transform2, but slower for common
	# cases.
	if not trivial_quat(isys.q):
		coords = 1/isys.q * coords
	# 2. Rotate to the target system. In general this would need pathfinding through
	# a space of stepwise transformations. But I'll just hardcode the steps here.
	# It's just the hor <-> equ step that's troublesome anyway
	if isys.base == osys.base:
		# Nothing to do. Saves computation
		pass
	elif space_sys(isys.base) and space_sys(osys.base):
		# Both in space. Simple static rotation
		coords = equ_rots[osys.base]/static_rots[isys.base] * coords
	elif space_sys(isys.base) and not space_sys(osys.base):
		# Need to cross to earth
		if not trivial_quat(equ_rots[isys.base]):
			coords = 1/equ_rots[isys.base] * coords
		coords = equ2hor(coords, ctime=ctime, site=site, weather=weather)
		if not trivial_quat(hor_rots[osys.base]):
			coords = hor_rots[osys.base] * coords
	elif not space_sys(isys.base) and space_sys(osys.base):
		# Need to cross to space
		if not trivial_quat(hor_rots[isys.base]):
			coords = 1/hor_rots[isys.base] * coords
		coords = hor2equ(coords, ctime=ctime, site=site, weather=weather)
		if not trivial_quat(equ_rots[osys.base]):
			coords = equ_rots[osys.base] * coords
	else:
		# Both on earth
		coords  = hor_rots[osys.base]/hor_rots[isys.base] * coords
	# 3. Apply any output rotation
	if not trivial_quat(osys.q):
		coords = osys.q * coords
	# Done!
	return coords

## Test version that tries to save memory by doing as many in-place
## operations as possible, at the cost of extra inversions
##
#def transform2(isys, osys, coords, ctime=None, site=sites.default_site, weather=None):
#	isys = expand_sys(isys, ctime=site, site=site, weather=weather)
#	osys = expand_sys(osys, ctime=site, site=site, weather=weather)
#	coords = coords.copy()
#	if not trivial_quat(isys.q): coords.iq *= isys.q
#	if isys.base == osys.base:
#		# Nothing to do. Saves computation
#		pass
#	elif space_sys(isys.base) and space_sys(osys.base):
#		# Both in space. Simple static rotation
#		coords.iq /= equ_rots[osys.base]/static_rots[isys.base]
#	elif space_sys(isys.base) and not space_sys(osys.base):
#		# Need to cross to earth
#		if not trivial_quat(equ_rots[isys.base]):
#			coords.iq *= equ_rots[isys.base]
#		coords = equ2hor(coords, ctime=ctime, site=site, weather=weather)
#		if not trivial_quat(hor_rots[osys.base]):
#			coords.iq /= hor_rots[osys.base]
#	elif not space_sys(isys.base) and space_sys(osys.base):
#		# Need to cross to space
#		if not trivial_quat(hor_rots[isys.base]):
#			coords.iq *= hor_rots[isys.base]
#		coords = hor2equ(coords, ctime=ctime, site=site, weather=weather)
#		if not trivial_quat(equ_rots[osys.base]):
#			coords.iq /= equ_rots[osys.base]
#	else:
#		# Both on earth
#		coords.iq /= hor_rots[osys.base]/hor_rots[isys.base]
#	# 3. Apply any output rotation
#	if not trivial_quat(osys.q):
#		coords.iq /= osys.q
#	# Done!
#	return coords

# Static transforms
equ_rots = {
	"equ": 1,
	# euler(2, Galactic._lon0_J2000.radian-np.pi)*euler(1, Galactic._ngp_J2000.dec.radian-np.pi/2)*euler(2, -Galactic._ngp_J2000.ra.radian) with astropy.coordinates.builtin_frames.galactic.Galactic
	"gal": np.quaternion(-0.488947507617903, 0.483210683963407, -0.196253758294796, -0.699229741968278),
}

hor_rots = {
	"hor": 1,
}

# Complicated transforms
def hor2equ(coords, ctime, site=sites.default_site, weather=None):
	weather = sites.expand_weather(weather, site)
	qp = qpoint.QPoint(accuracy="high", fast_math=True, mean_aber=True,
		rate_ref="always", **weather)
	q  = qp.azelpsi2bore(coords.az/DEG, coords.el/DEG, coords.psi/DEG, None, None,
		lon=site.lon, lat=site.lat, ctime=ctime)
	q  = quaternion.as_quat_array(q)
	return Coords(q=q)

def equ2hor(coords, ctime, site=sites.default_site, weather=None):
	weather = sites.expand_weather(weather, site)
	qp = qpoint.QPoint(accuracy="high", fast_math=True, mean_aber=True,
		rate_ref="always", **weather)
	# I don't recover the original roll exactly here. It's off by about half a degree
	az, el, pa = qp.radec2azel(coords.ra/DEG, coords.dec/DEG, coords.psi/DEG,
		lon=site.lon, lat=site.lat, ctime=ctime)
	return Coords(az=az*DEG, el=el*DEG, roll=pa*DEG)

class Coords:
	"""Class for representing both az,el,roll and quaternions.
	Needed to avoid unnecessary conversions between these representations
	since some functions want one format and some the other.
	Initialize with either representation. Will automatically calculate
	the other only if needed."""
	def __init__(self, az=None, el=None, roll=None, ra=None, dec=None, psi=None, q=None, iq=None):
		self._lon  = maybearr(ra)
		if az is not None: self._lon = -asfarray(az)
		self._lat  = maybearr(dec)
		if el is not None: self._lat = asfarray(el)
		self._psi  = maybearr(psi)
		if roll is not None: self._psi = asfarray(roll)+np.pi
		self._q    = maybearr(q,  default_dtype=np.quaternion)
		self._iq   = maybearr(iq, default_dtype=np.quaternion)
		if self._psi is None and self._q is None:
			# psi/roll missing. Default depends on if we're in az/el or ra/dec
			# psi and roll correspond to different angle conventions, though
			# I'm not sure about the details. Is this another consequence of
			# left-handed coordinates?
			self._psi = np.zeros_like(self._lon)
			if az is not None: self._psi += np.pi
	def __getattr__(self, name):
		# az and roll are cheap to convert, so we do them on the fly to avoid
		# having to store them, and to keep things simple
		if   name == "az":
			val = -self.ra
			def copy_back(): self.ra = val
			return warray.WatchArray(val, copy_back)
		elif name == "roll":
			val = self.psi-np.pi
			def copy_back(): self.roll = val
			return warray.WatchArray(val, copy_back)
		elif name == "theta":
			val = np.pi/2-self.lat
			def copy_back(): self.theta = val
			return warray.WatchArray(val, copy_back)
		# the others are handled via the cache system
		elif name in ["ra", "lon", "phi"]: val = self._cache("_lon", self._calc_coord)
		elif name in ["el", "dec", "lat"]: val = self._cache("_lat", self._calc_coord)
		elif name == "psi": val = self._cache("_psi",  self._calc_coord)
		elif name == "q":   val = self._cache("_q",    self._calc_q)
		elif name == "iq":  val = self._cache("_iq",   self._calc_iq)
		else: raise AttributeError(name)
		return warray.WatchArray(val, lambda: self._handle_update(name))
	def __setattr__(self, name, val):
		if   name == "az":    self._lon  = -asfarray(val)
		elif name == "roll":  self._psi = asfarray(val)+np.pi
		elif name == "theta": self._lat = np.pi/2-asfarray(val)
		elif name in ["ra", "lon", "phi"]: self._lon = asfarray(val)
		elif name in ["el", "dec", "lat"]: self._lat = asfarray(val)
		elif name == "psi": self._psi = asfarray(val)
		elif name == "q":   self._q  = asfarray(val, np.quaternion)
		elif name == "iq":  self._iq = asfarray(val, np.quaternion)
		else:
			super().__setattr__(name, val)
			return
		self._handle_update(name)
	# Add a few quaternion operations directly. This doesn't
	# let us use .iq though, so maybe just doing it manually is best
	def __mul__(self, other):
		try: return Coords(q = self.q * other.q)
		except AttributeError: return Coords(q = self.q * other)
	def __truediv__(self, other):
		try: return Coords(q = self.q / other.q)
		except AttributeError: return Coords(q = self.q / other)
	def __rmul__(self, other): return Coords(q = other * self.q)
	def __rtruediv__(self, other): return Coords(q = other / self.q)
	def __dir__(self):
		return list(dir(Coords))+["az", "el", "roll", "ra", "dec", "psi", "lon", "lat", "q", "iq"]
	@property
	def has_coords(self): return self._lon is not None
	@property
	def has_q(self): return self._q is not None
	@property
	def has_iq(self): return self._iq is not None
	def copy(self): return copy.deepcopy(self)
	def _handle_update(self, name):
		if name in ["az", "el", "roll", "ra", "dec", "psi", "lon", "lat"]:
			self.q = self.iq = None
		else:
			self._lon = self._lat = self._psi = None
			if   name != "q":  self._q  = None
			elif name != "iq": self._iq = None
	# Caching and conversion below
	def _cache(self, attr, fun):
		if getattr(self, attr) is None: fun()
		return getattr(self, attr)
	def _calc_coord(self):
		self._lon, self._lat, self._psi = decompose_lonlat(self.q)
	def _calc_q(self):
		if self.has_iq: self._q = 1/self.iq
		else: self._q = rotation_lonlat(self.ra, self.dec, self.psi)
	def _calc_iq(self):
		self._iq = 1/self.q
	def __repr__(self):
		parts = []
		if self.has_coords:
			parts.append("lon=%s, lat=%s, psi=%s" % (str(self.lon), str(self.lat), str(self.psi)))
		if self.has_q:
			parts.append("q=%s" % (str(self.q)))
		if self.has_iq:
			parts.append("iq=%s" % (str(self.iq)))
		return "Coords(" + ", ".join(parts) + ")"

def maybearr(a, default_dtype=np.float64):
	return asfarray(a, default_dtype=default_dtype) if a is not None else None

def left_handed(sys): return sys in ["hor"]
def space_sys(sys): return sys not in ["hor"]

def trivial_quat(q):
	if q is None: return True
	if np.allclose(q,np.quaternion(1,0,0,0)): return True
	return False

# angle <-> quaternion coversions. Adapted from so3g

def euler(axis, angle):
	angle = np.asarray(angle)
	shape = np.broadcast(axis, angle).shape + (4,)
	q     = np.zeros(shape)
	ahalf = angle/2
	q[...,     0] = np.cos(ahalf)
	q[...,axis+1] = np.sin(ahalf)
	q     = quaternion.as_quat_array(q)
	return q

def rotation_lonlat(lon, lat, psi=0):
	return euler(2, lon) * euler(1, np.pi/2-lat) * euler(2, psi)

def decompose_lonlat(q):
	q = quaternion.as_float_array(q)
	a, b, c, d = [q[...,i] for i in range(4)]
	psi   = np.arctan2(a*b+c*d, a*c-b*d)
	lon   = np.arctan2(c*d-a*b, a*c+b*d)
	lat   = np.pi/2 - 2*np.arctan2((b**2+c**2)**0.5, (a**2+d**2)**0.5)
	return lon, lat, psi

def expand_sys(sys, ctime=None, site=None, weather=None):
	# Parse if necessary
	if isinstance(sys, str):
		sys = parse_sys(sys)
	# Already expanded?
	if "base" in sys and "q" in sys:
		return sys
	# Our base coordinate system
	base = sys["up"]["sys"]
	# Expand pos for any objects
	qs = {}
	for key in ["up","on","to"]:
		pos = sys[key]["pos"]
		if isinstance(pos, str):
			# Ok, pos = objname. Get its coordinates
			coords = pixell.ephem(pos, ctime)
			# Transform it to our base system
			coords = transform("equ", sys[key]["sys"], ctime=ctime, site=site, weather=weather)
		else:
			# Actual coordinates
			if left_handed(sys[key]["sys"]):
				coords = Coords(az=pos[0], el=pos[1])
			else:
				coords = Coords(ra=pos[0], dec=pos[1])
		qs[key] = coords.q
	# Build up the full rotation from the qs.
	# Each quaternion represents the euler rotation ZY
	q = np.quaternion(1,0,0,0)
	# 1. Rotate the up point to the north pole, both for our
	# actual coordinates and our other points
	if not trivial_quat(qs["up"]):
		iup      = 1/qs["up"]
		q        = iup*q
		qs["on"] = iup*qs["on"]
		qs["to"] = iup*qs["to"]
	# 2. now that the up point is up, we can now do our recentering
	if not trivial_quat(qs["on"]):
		q = 1/qs["on"]*q
	if not trivial_quat(qs["to"]):
		q = qs["to"]*q
	# Tell the user if the result is trivial, so they don't waste tme
	if trivial_quat(q):
		q = None
	return bunch.Bunch(base=base, q=q)

def parse_sys(desc):
	info = {
		"up":{"sys":"equ", "pos":[0,np.pi/2]},
		"on":{"sys":"equ", "pos":[0,0]},
		"to":{"sys":"equ", "pos":[0,0]},
	}
	toks = desc.split(",")
	for i, tok in enumerate(toks):
		subs = tok.split("=")
		if i == 0 and len(subs) == 1:
			subs = ["up"]+subs
		if len(subs) != 2:
			raise ValueError("Error parsing coordinate system description '%s'" % str(desc))
		key, val = subs
		if key not in ["up", "on", "to"]:
			raise ValueError("Only up, on and to can be used when building a coordinate system, but got '%s'" % str(key))
		info[key] = _parse_sys_pos(val, default_sys=info["up"]["sys"], default_pos=info[key]["pos"])
	return info

def _parse_sys_pos(pdesc, default_sys="equ", default_pos=[0,0]):
	toks = pdesc.split(":")
	if len(toks) == 1:
		# Assume coordinates if on coordinate form
		if toks[0].startswith("["): toks = [default_sys,toks[0]]
		# Otherwise assume system name if one of the known systems.
		# NOTE: This will need to be updated if we add more systems
		elif toks[0] in ["hor", "equ", "gal"]: return {"sys":toks[0], "pos":default_pos}
		# The rest are assumed to be object names
		else: toks = [default_sys,toks[0]]
	if len(toks) != 2:
		raise ValueError("Error parsing position description '%s'" % str(pdesc))
	sys, pos = toks
	# pos can be either [ra,dec] or an object name
	if pos.startswith("[") and pos.endswith("]"):
		subs = pos[1:-1].split(",")
		if len(subs) != 2:
			raise ValueError("Coordinates must be [ra,dec] in degrees, but got '%s'" % str(pos))
		pos = [float(w)*utils.degree for w in pos.split(",")]
	else:
		# just keep it as a string, that represents the object's name.
		# This will be evaluated in eval_sys
		pass
	return {"sys":sys, "pos":pos}

def asfarray(arr, default_dtype=np.float64):
	# This was removed from numpy for some reason
	arr = np.asarray(arr)
	if np.issubdtype(arr.dtype, np.floating):
		return arr
	else:
		return arr.astype(default_dtype)
