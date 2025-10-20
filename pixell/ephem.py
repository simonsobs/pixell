"""This module provides a unified interface for several different sources of ephemeris,
including standard pyephem and precomputed asteroid ephemeris.

There are two interfaces: simple and advanced. The simple interface is based on
top-level module functions ephem.eval, ephem.add and the property ephem.bodies.
Example:

 from pixell import ephem
 # 100k samples with 10 ms spacing starting at unix time
 # 1760000000 (Oct 9 2025)
 ctimes = 1760000000 + np.arange(100000)*0.01
 radec, dist = ephem.eval("Jupiter", ctimes)
 # radec is [100k,{ra,dec}] in radians, and dist is [100k] AU.
 # Since no site was specified ephem.sites.default_site (Cerro Toco).
 # We can also specify a site manually:
 site = bunch.Bunch(lat=33.36361, lon=116.83639, alt=1872, weather="default") # Palmoar
 radec, dist = ephem.eval("Jupiter", ctimes, site=site)
 # By default only a few objects are available:
 ephem.bodies
 # > ['Ariel', 'Callisto', 'Deimos', 'Dione', 'Enceladus', 'Europa', 'Ganymede', 'Hyperion',
 'Iapetus', 'Io', 'Jupiter', 'Mars', 'Mercury', 'Mimas', 'Miranda', 'Moon', 'Neptune',
 'Oberon', 'Phobos', 'Pluto', 'Rhea', 'Saturn', 'Sun', 'Tethys', 'Titan', 'Titania',
 'Umbriel', 'Uranus']
 # We can make more available by adding another source, e.g.
 ephem.add(ephem.PrecompEphem("/path/to/precomputed/ephems"))
 # Those added later have higher priority in the lookup

The advanced interface consists of the Ephem-classes AstropyEphem,
PyephemEphem, PrecompEphem, InterpEphem and MultiEphem. These all
have the .eval(name, ctime, cartesian=False, site=None) methos and
the .bodies member, but are constructed differently. See their
docstrings for details.

The simple interface just accesses ephem.default_ephem, which
is initialized as default_ephem = MultiEphem([InterpEphem(PyephemEphem())]).
ephem.add simply calls default_ephem.add.
"""
import numpy as np, os, glob
from scipy import interpolate
from . import utils, sites

class Ephem:
	"""General interface for the Ephemeris implementations."""
	def __init__(self, bodies=[], capitalize=True):
		self.bodies     = bodies
		self.capitalize = capitalize
	def eval(self, name, ctime, cartesian=False, site=None): raise NotImplementedError

class MultiEphem(Ephem):
	"""Maintains a list of other Ephem-objects, and forwards
	any request to the last-added ephemeris that can handle
	the object-name asked for in eval."""
	def __init__(self, others=[], capitalize=True):
		"""Initialize MultiEphem with a list of other ephemerids,
		given in ascending priority. More can be added later with
		the .add method."""
		super().__init__(capitalize=capitalize)
		self.others = []
		for other in others:
			self.add(other)
	def eval(self, name, ctime, cartesian=False, site=None):
		if self.capitalize: name = name.capitalize()
		# Last-added get highest priority
		for other in self.others[::-1]:
			if name in other.bodies:
				return other.eval(name, ctime, cartesian=cartesian, site=site)
		raise KeyError("No ephemeris found for '%s'" % str(name))
	def add(self, other):
		self.others.append(other)
		self.bodies += other.bodies

class AstropyEphem(Ephem):
	"""Epemeris based on astropy's solar_system_ephemeris. Quite slow"""
	def __init__(self, ephemeris="builtin", site=None, capitalize=True):
		"""The ephemeris argument controls astropy's ephemeris source. Defaults
		to the low-accuracy "builtin" ephemeris. Other values are de432s
		(good for 1950-2050) or the heavy de430."""
		import astropy.coordinates as aco
		bodies = [name.capitalize() for name in aco.solar_system_ephemeris.bodies]
		super().__init__(bodies=bodies, capitalize=capitalize)
		self.site       = site
		self.ephemeris  = ephemeris
	def eval(self, name, ctime, cartesian=False, site=None):
		"""Runs at 79 ms/sample"""
		import astropy.time as ati, astropy.coordinates as aco
		if self.capitalize: name = name.capitalize()
		site = site or self.site or sites.default_site
		ctime= np.asarray(ctime)
		t    = ati.Time(ctime, format="unix")
		loc  = aco.EarthLocation.from_geodetic(site.lon, site.lat, site.alt)
		data = aco.get_body(name, t, location=loc, ephemeris=self.ephemeris)
		if cartesian:
			rect = np.zeros(ctime.shape+(3,))
			dcart= data.cartesian
			rect[...,0] = dcart.x.to("AU").value
			rect[...,1] = dcart.y.to("AU").value
			rect[...,2] = dcart.z.to("AU").value
			return rect
		else:
			pos = np.zeros(ctime.shape+(2,))
			r   = np.zeros(ctime.shape)
			dsph= data.spherical
			pos[...,0] = dsph.lon.radian
			pos[...,1] = dsph.lat.radian
			r[...]     = dsph.distance.to("AU").value
			return pos, r

class PyephemEphem(Ephem):
	def __init__(self, site=None, capitalize=True):
		"""Ephemeris using pyephem. About 3x as fast as astropy, but
		still slow."""
		import ephem
		bodies = ["Ariel", "Callisto", "Deimos", "Dione", "Enceladus", "Europa", "Ganymede", "Hyperion", "Iapetus", "Io", "Jupiter", "Mars", "Mercury", "Mimas", "Miranda", "Moon", "Neptune", "Oberon", "Phobos", "Pluto", "Rhea", "Saturn", "Sun", "Tethys", "Titan", "Titania", "Umbriel", "Uranus", "Venus"]
		super().__init__(bodies=bodies, capitalize=capitalize)
		self.site       = site
	def eval(self, name, ctime, cartesian=False, site=None):
		"""Given a name and ctime[...], return the
		observed position pos[...,{ra,dec}] in radian
		and distance from observer r[...] in AU for
		each time, or rect[...,3] with the observe-relative
		cartesian coordinates in AU if cartesian=True.

		If site is given, it overrides the site passed to the
		constructor. Defaults to sites.default_site.

		Runs at 23 µs/sample
		"""
		import ephem
		if self.capitalize: name = name.capitalize()
		site = site or self.site or sites.default_site
		obj  = getattr(ephem, name)()
		obs  = ephem.Observer()
		obs.lon = site.lon
		obs.lat = site.lat
		obs.elevation = site.alt
		ctime = np.asarray(ctime)
		djds  = utils.ctime2djd(ctime).reshape(-1)
		# output arguments
		pos   = np.zeros(djds.shape + (2,))
		r     = np.zeros(djds.shape)
		# Only manual looping in python, sadly
		for i, djd in enumerate(djds):
			obs.date = djd
			obj.compute(obs)
			pos[i,0] = obj.a_ra
			pos[i,1] = obj.a_dec
			r[i] = obj.earth_distance
		if cartesian:
			rect = utils.ang2rect(pos, axis=1)*r[:,None]
			rect = rect.reshape(ctime.shape+(3,))
			return rect
		else:
			pos  = pos.reshape(ctime.shape+(2,))
			r    = r  .reshape(ctime.shape)
			return pos, r

class PrecompEphem(Ephem):
	"""Ephemeris based on a directory with files containing precomputed
	positions for a set of objects. The observer site was baked into
	this precomputation, so this class ignores the site argument.
	Inflexible but very fast."""
	def __init__(self, path, capitalize=True):
		bodies= [os.path.basename(name)[:-4] for name in sorted(glob.glob(os.path.join(path, "*.npy")))]
		super().__init__(bodies=bodies, capitalize=capitalize)
		self.path  = path
		self.cache = {}
	def eval(self, name, ctime, cartesian=False, site=None):
		"""Given a name and ctime[...], return the
		observed position pos[...,{ra,dec}] in radian
		and distance from observer r[...] in AU for
		each time, or rect[...,3] with the observe-relative
		cartesian coordinates in AU if cartesian=True.

		The observer site was baked into the precomputation,
		and can't be changed at this point, so the site argument
		is ignored.

		Runs at 73 ns/sample
		"""
		if self.capitalize: name = name.capitalize()
		spline = self.get(name)
		rect   = spline(ctime)
		if cartesian:
			return rect
		else:
			pos, r = utils.rect2ang(rect, return_r=True, axis=-1)
			return pos, r
	def get(self, name):
		if name not in self.cache:
			data   = np.load(os.path.join(self.path, name + ".npy"))
			interp = interpolate.interp1d(data["ctime"], data["pos"], kind=3, axis=0)
			self.cache[name] = interp
		return self.cache[name]
	def clear(self):
		self.cache = {}

class InterpEphem(Ephem):
	"""Ephemeris that samples another ephemeris sparsely and
	interpolates between these samples. For slow ephemerides
	like AstropyEphem or PyephemEphem, this speeds them up by
	a factor of ~1000."""
	def __init__(self, other, dt=300):
		"""Interpolate another Ephem object using step size dt
		in seconds. dt should be small enough that a 3rd order
		spline approximates the motion well, including the Earth's
		motion. The default, 5 minutes, should be quite converative.
		It's good enough to get double precision accuracy for Jupiter.

		Runs at 60 ns/sample
		"""
		super().__init__(bodies=other.bodies, capitalize=other.capitalize)
		self.other = other
		self.dt    = dt
	def eval(self, name, ctime, cartesian=False, site=None):
		ctime  = np.asarray(ctime)
		tflat  = ctime.reshape(-1)
		order  = np.argsort(tflat)
		tflat  = tflat[order]
		step   = np.max(np.abs(np.diff(tflat))) if len(tflat) > 1 else 0
		if len(tflat) <= 1 or step >= self.dt:
			# Don't try to build interpolation if it would be more
			# costly than just evaluating directly!
			return self.other.eval(name, ctime, cartesian=cartesian, site=site)
		t1, t2 = utils.minmax(tflat)
		npoint = max(4,utils.ceil((t2-t1)/self.dt))
		iptime = np.linspace(t1, t2, npoint)
		data   = self.other.eval(name, iptime, cartesian=True, site=site)
		interp = interpolate.interp1d(iptime, data, kind=3, axis=0)
		rect   = np.zeros(ctime.shape+(3,))
		# Interpolate into original shape and order
		rect.reshape(-1,3)[order] = interp(tflat)
		if cartesian:
			return rect
		else:
			pos, r = utils.rect2ang(rect, return_r=True, axis=-1)
			return pos, r

# Default ephemeris
default_ephem = MultiEphem([InterpEphem(PyephemEphem())])
def eval(name, ctime, cartesian=False, site=None):
	return default_ephem.eval(name, ctime, cartesian=cartesian, site=site)
def add(ephem): default_ephem.add(ephem)
bodies = default_ephem.bodies
