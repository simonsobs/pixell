import numpy as np, numba, ducc0, time
from . import coordinates, enmap, utils, curvedsky, wcsutils

beta    = 0.001235
dir_equ = np.array([167.919, -6.936])*np.pi/180
dir_gal = np.array([263.986, 48.247])*np.pi/180
dir_ecl = np.array([171.640,-11.154])*np.pi/180
from .utils import T_cmb, h, c, k

def boost_map(map, dir=dir_equ, beta=beta, modulation="T2lin", T0=utils.T_cmb, freq=150e9,
		return_modulation=False, dipole=False, map_unit=1e-6, spin=[0,2], aberrate=True,
		modulate=True, nthread=None, coord_dtype=None, boundary="auto"):
	"""Doppler-boost (aberrate and modulate) the given input map map. The doppler boost
	goes in direction dir[{ra,dec}] with speed beta in units of c.

	spin controls how the 3rd last axis is interpreted. spin = [0,2] means that the
	first entry is scalar and the next pair is spin-2. So this works for a [TQU,ny,nx]
	map. It is safe to pass [0,2] even when the map is just scalar.

	The modulation argument covers five cases. The default is "T2lin".
	* None: No modulation or unit conversion is performed at all.
	* "plain" or "T2T": The input and output maps are in real temperature units.
	* "T2lin": The input map is real temperature, but the output map is in
	  linearized temperature units, which are proportional to the flux density.
	  This is appropriate for simulating observed CMB maps.
	* "lin2T": The opposite of T2lin. Appropriate for deaberration.
	* "lin2lin": Both input and output maps are in linearized units. Probably
	  not very useful.

	The boundary conditions are always periodic along the x axis. For the y axis
	they are controlled by the boundary argument, which has the following values:
	* "periodic": map[Ny,x] = map[0,x]
	* "fullsky":  map[Ny,x] = map[Ny-1,x+Nx/2]. This is slower and uses more memory
	  than periodic.
	* "auto": Choose fullsky if map covers close to the full sky in the y direction,
	  otherwise periodic. This is the default.

	If dipole is True, then monopole-related terms will be included in the modulation.
	This is mainly the CMB dipole.

	If return_modulation == True, then a tuple of (omap, A) will be returned, where
	A is the modulation as returned from calc_boost. This can be useful if one wants
	to compute the same map modulated at several different frequencies.
	
  The modulation and aberration steps can be individually skipped using the
  corresponding arguments. By default both are performed.
	"""
	if return_modulation: assert modulate, "Can't return modulation of modulation is disabled"
	if aberrate:
		map    = aberrate_map(map, dir=dir, beta=beta, spin=spin, nthread=nthread, boundary=boundary)
	if modulate:
		map, A = modulate_map(map, dir=dir, beta=beta, spin=spin, nthread=nthread, dipole=dipole,
			modulation=modulation, T0=T0, freq=freq, map_unit=map_unit, return_modulation=True)
	if return_modulation: return map, A
	else: return map

def deboost_map(map, dir=dir_equ, beta=beta, modulation="lin2T", T0=utils.T_cmb, freq=150e9,
		return_modulation=False, dipole=False, map_unit=1e-6, spin=[0,2], aberrate=True,
		modulate=True, nthread=None, coord_dtype=None, boundary="auto"):
	"""The inverse of boost_map. Simply calls boost map with the sign of beta flipped, and
	with a default modulation of lin2T instead of T2lin."""
	return boost_map(map, dir=dir, beta=-beta, modulation=modulation, T0=T0, freq=freq,
		return_modulation=return_modulation, dipole=dipole, map_unit=map_unit, spin=spin,
		aberrate=aberrate, modulate=modulate, nthread=nthread, coord_dtype=coord_dtype,
		boundary=boundary)

def aberrate_map(map, dir=dir_equ, beta=beta, spin=[0,2],
		nthread=None, coord_dtype=None, boundary="auto"):
	"""Perform the aberration part of the doppler boost. See
	boost_map for details."""
	if coord_dtype is None: coord_dtype = map.dtype
	return Aberrator(map.shape, map.wcs, dir=dir, beta=beta, spin=spin,
		nthread=nthread, coord_dtype=coord_dtype, boundary=boundary)(map)

def deaberrate_map(map, dir=dir_equ, beta=beta, spin=[0,2],
		nthread=None, coord_dtype=None, boundary="auto"):
	return aberrate_map(map, dir=dir, beta=-beta, spin=spin,
		nthread=nthread, coord_dtype=coord_dtype, boundary=boundary)

def modulate_map(map, dir=dir_equ, beta=beta,
		modulation="T2lin", T0=utils.T_cmb, freq=150e9, return_modulation=False,
		dipole=False, map_unit=1e-6, spin=[0,2], nthread=None):
	"""Perform the modulation part of the doppler boost. See boost_map
	for details."""
	modulator = Modulator(map.shape, map.wcs, dir=dir, beta=beta, spin=spin,
		modulation=modulation, T0=T0, freq=freq, dipole=dipole, map_unit=map_unit,
		nthread=nthread, dtype=map.dtype)
	map = modulator(map)
	if return_modulation: return map, modulator.A
	else: return map

def demodulate_map(map, dir=dir_equ, beta=beta,
		modulation="lin2T", T0=utils.T_cmb, freq=150e9, return_modulation=False,
		dipole=False, map_unit=1e-6, spin=[0,2], nthread=None):
	return modulate_map(map, dir=dir, beta=-beta, modulation=modulation,
		T0=T0, freq=freq, return_modulation=return_modulation, dipole=dipole,
		map_unit=map_unit, spin=spin, nthread=nthread)

# Classes for repeat calcuations with similar maps

class Aberrator:
	def __init__(self, shape, wcs, dir=dir_equ, beta=beta, spin=[0,2],
			nthread=None, coord_dtype=np.float64, boundary="auto"):
		"""Construct an Aberrator object, that can be used to more efficiently
		aberrate a map. E.g.

		  aberrator = Aberrator(shape, wcs)
		  for map in maps:
		    omap = aberrator(map)
		    do something with omap

		This should be faster than repeatedly calling aberrate_map because
		it precomputes the deflection information instead of needing to
		compute it from scratch each time."""
		coord_ctype = utils.complex_dtype(coord_dtype)
		nthread = int(utils.fallback(utils.getenv("OMP_NUM_THREADS",nthread),0))
		# 1. Calculate the aberration field. These are tiny
		alm_dpos = calc_boost_field(-beta, dir, nthread=nthread)
		# 2. Evaluate these on our target geometry.
		deflect = enmap.zeros(alm_dpos.shape[:-1]+shape[-2:], wcs, coord_dtype)
		curvedsky.alm2map(alm_dpos.astype(coord_ctype, copy=False), deflect, spin=1, nthread=nthread)
		# 3. Calculate the offset angles.
		rinfo = curvedsky.get_ring_info(shape, wcs)
		dphi  = np.full(shape[-2], wcs.wcs.cdelt[0]*utils.degree)
		odec, ora, gamma = ducc0.misc.get_deflected_angles(theta=rinfo.theta, phi0=rinfo.phi0,
			nphi=rinfo.nphi, dphi=dphi, ringstart=rinfo.offsets, nthreads=nthread, calc_rotation=True,
			deflect=np.asarray(deflect).reshape(2,-1).T).T
		del deflect
		odec = np.pi/2-odec
		gamma= enmap.ndmap(gamma.reshape(shape[-2:]), wcs)
		# 4. Calculate pixel coordinates of offset angles. In general this would use
		# enmap.sky2pix, but that's slow. Much faster for our typical projections.
		# Probably worth it to make overrides.
		pix = sky2pix(shape, wcs, [odec, ora]).astype(coord_dtype, copy=False)
		# Make the x pixel 0:nphi, assuming it's at most one period away
		fast_rewind(pix[1].reshape(-1), rinfo.nphi[0])
		del odec, ora
		# 5. Evaluate the map at these locations using nufft
		# Boundary condition
		if boundary == "auto": boundary = "fullsky" if fully(shape, wcs) else "periodic"
		if boundary not in ["periodic", "fullsky"]:
			raise ValueError("Unrecognized boundary '%s'" % str(boundary))
		# 6. Store for when the map is passed in later
		self.pix       = pix
		self.gamma     = gamma
		self.nthread   = nthread
		self.spin      = spin
		self.boundary  = boundary
	def __call__(self, map, spin=None, nthread=None):
		if nthread is None: nthread = self.nthread
		if spin    is None: spin    = self.spin
		shape, wcs = map.shape, map.wcs
		ydouble    = self.boundary == "fullsky"
		map = interpol_map(map, self.pix, nthread=nthread, ydouble=ydouble)
		map = enmap.ndmap (map.reshape(shape), wcs)
		# 6. Apply polarization rotation. ducc0.misc.lensing_rotate can do this,
		# but for some reason it operates on complex numbers instead of a QU field.
		# So seems like I would have to waste time and memory transforming from/to
		# this ordering. We loop over pre-dimensions to reduce memory use
		for s, I in enmap.spin_pre_helper(spin, map.shape[:-2]):
			rotate_pol(utils.fix_zero_strides(map[I]), -self.gamma, spin=s)
		return map

class Modulator:
	def __init__(self, shape, wcs, dir=dir_equ, beta=beta,
			modulation="T2lin", T0=utils.T_cmb, freq=150e9, dipole=False,
			map_unit=1e-6, spin=[0,2], dtype=np.float64, nthread=None):
		"""Construct a Modulator object, that can be used to more efficiently
		modulate a map. E.g.

		  modulator = Modulator(shape, wcs)
		  for map in maps:
		    omap = modulator(map)
		    do something with omap

		This should be faster than repeatedly calling modulate_map because
		it precomputes the modulation information instead of needing to
		compute it from scratch each time."""
		nthread = int(utils.fallback(utils.getenv("OMP_NUM_THREADS",nthread),0))
		# 1. Calculate the aberration field. These are tiny
		alm_mod = calc_boost_field(-beta, dir, nthread=nthread, modulation=True, mod_exp=-1)[1]
		alm_mod = alm_mod.astype(utils.complex_dtype(dtype), copy=False)
		# 2: Apply modulation
		A = enmap.zeros(alm_mod.shape[:-1]+shape[-2:], wcs, dtype)
		curvedsky.alm2map(alm_mod, A, spin=0, nthread=nthread)
		# Store for __call__
		self.nthread = nthread;  self.A     = A;     self.modulation = modulation
		self.T0      = T0;       self.freq  = freq;  self.dipole     = dipole
		self.dtype   = dtype;    self.spin  = spin;  self.map_unit   = map_unit
	def __call__(self, map, spin=None):
		if spin is None: spin = self.spin
		if map.dtype != self.dtype:
			warnings.warn("Modulator dtype does not match argument dtype")
		return apply_modulation(map, self.A, spin=spin, T0=self.T0, freq=self.freq,
			map_unit=self.map_unit, mode=self.modulation, dipole=self.dipole)

def calc_boost_1d(z, beta):
	"""Given the z, the cosine of the angle from the direction of travel in
	the CMB rest frame, and beta, the speed as a fraction of c, compute the
	observed z_obs after taking into account the effect of aberration, as well
	as the corresponding modulation A, such that T_obs(z_obs) = A * T_rest(z).
	Pass in -beta to get the reverse transformation, from obs to rest."""
	gamma = (1-beta**2)**-0.5
	z_obs = (z+beta)/(1+z*beta)
	np.clip(z_obs, -1, 1, out=z_obs)
	A     = 1/(gamma*(1-z_obs*beta))
	return z_obs, A

#### Helpers ####

def fully(shape, wcs, tol=0.1):
	minfo = curvedsky.analyse_geometry(shape, wcs)
	if minfo.ducc_geo is None: return False
	yfrac  = abs(shape[-2]/minfo.ducc_geo.ny)
	return yfrac > 1-tol

def beta2lmax(beta):
	# Infer lmax from beta. Empirical formula, valid to high boost factors.
	beta  = np.abs(beta)
	gamma = (1-beta**2)**-0.5
	lmax  = utils.ceil(1/(4e-3+1/gamma)**0.62*14+3.5)
	return lmax

def calc_boost_field(beta, dir, lmax=None, nthread=None, modulation=False, mod_exp=1):
	if lmax is None:
		lmax=beta2lmax(beta)
	nthread = int(utils.fallback(utils.getenv("OMP_NUM_THREADS",nthread),0))
	n      = lmax+2
	# zenith angles to sample the sky at, in fejer1
	itheta = np.arange(0,n)*np.pi/(n-1)
	iz     = np.cos(itheta)
	oz,  A = calc_boost_1d(iz, beta)
	otheta = np.arccos(oz)
	# Calculate the local position deflection. We have
	# no deflection in phi in this coordinate system
	dpos   = np.zeros([2,n])
	dpos[0]= otheta-itheta
	alm    = curvedsky.prof2alm(dpos, dir=dir, spin=1, nthread=nthread)
	if modulation:
		A  **= mod_exp
		malm = curvedsky.prof2alm(A, dir=dir, spin=0, nthread=nthread)
		return alm, malm
	else:
		return alm

def interpol_map(imap, pixs, epsilon=None, nthread=None, ydouble=False):
	ny, nx = imap.shape[-2:]
	if ydouble:
		# Make y doubled map. This is necessary for the correct boundary condition for
		# fullsky maps, but makes less sense for partial sky maps
		dmap = enmap.zeros(imap.shape[:-2]+(2*ny,nx), imap.wcs, imap.dtype)
		dmap[...,:ny,:] = imap
		dmap[...,ny:,:] = np.roll(imap[...,::-1], nx//2, -1)
	else:
		dmap = imap
	periodicity = np.array(dmap.shape[-2:])
	nthread = int(utils.fallback(utils.getenv("OMP_NUM_THREADS",nthread),0))
	pflat   = pixs.reshape(2,-1).T
	if epsilon is None:
		epsilon = 1e-5 if imap.dtype == np.float32 else 1e-12
	oarr    = np.zeros(imap.shape[:-2]+pflat.shape[:1], imap.dtype)
	for I in utils.nditer(imap.shape[:-2]):
		fmap = enmap.fft(dmap[I], normalize=False)
		oarr[I] = ducc0.nufft.u2nu(grid=np.asarray(fmap), coord=pflat, forward=False,
			epsilon=epsilon, nthreads=nthread, periodicity=periodicity, fft_order=True).real
		del fmap
	# Restore predims
	oarr = oarr.reshape(imap.shape[:-2]+oarr.shape[-1:])
	oarr/= dmap.npix
	return oarr

@numba.njit(nogil=True)
def rotate_pol(pmap, gamma, spin):
	if spin == 0: return
	qarr, uarr = pmap
	for i in range(qarr.shape[0]):
		for j in range(qarr.shape[1]):
			c = np.cos(spin*gamma[i,j])
			s = np.sin(spin*gamma[i,j])
			q =  qarr[i,j]*c + uarr[i,j]*s
			u = -qarr[i,j]*s + uarr[i,j]*c
			qarr[i,j] = q
			uarr[i,j] = u

def apply_modulation(map, A, T0=utils.T_cmb, freq=150e9, map_unit=1e-6, mode="T2lin",
		dipole=False, spin=[0,2]):
	if    mode in [None, "none"]: pass
	elif  mode in ["plain", "T2T"]:
		map *= A
		if dipole:
			utils.atleast_Nd(map,3)[...,0,:,:] += (A-1)*(T0/map_unit)
	elif  mode in ["T2lin", "lin2T", "lin2lin"]:
		# We're in linearized thermodynamic units. We assume that the map doesn't contain the
		# monopole, so we can treat it as a perturbation around the monopole. If the map
		# contains the monopole, then linearized units probably isn't the best choice
		for s, I in enmap.spin_pre_helper(spin, map.shape[:-2]):
			for comp in map[I]:
				if mode == "T2lin":
					_modulate_T2lin(comp, A, T0, freq, map_unit, spin=s, dipole=dipole)
				elif mode == "lin2T":
					_modulate_lin2T(comp, A, T0, freq, map_unit, spin=s, dipole=dipole)
				else:
					# This case probably isn't necessary
					_modulate_lin2T(comp, A*0+1, T0, freq, map_unit, spin=s, dipole=dipole)
					_modulate_T2lin(comp, A, T0, freq, map_unit, spin=s, dipole=dipole)
	else: raise ValueError("Urecognized modulation mode '%s'" % mode)
	return map

planck  = numba.njit(utils.planck)
dplanck = numba.njit(utils.dplanck)
iplanck = numba.njit(utils.iplanck_T)

# This is 4x as slow as the old one, but handles any value of beta, and also non-buggy :)
# This isn't the bottleneck anyway - aberration is - so I think it's worth it to just use this.
# In the future I can easily gain back af actor 4x with an implementation in C
@numba.njit
def _modulate_T2lin(map, A, T0=utils.T_cmb, freq=150e9, map_unit=1e-6, spin=0, dipole=False):
	scale= dplanck(freq, T=T0)
	off  = planck(freq,T0)/scale
	for y in range(map.shape[0]):
		for x in range(map.shape[1]):
			ival = np.float64(map[y,x])
			a    = np.float64(A[y,x])
			# This eats up at around 6 digits of precision for polarization, so
			# we need double precision
			oval = planck(freq, a*(ival*map_unit+T0))/scale
			if spin == 0 and dipole:
				oval -= off
			else:
				oval -= planck(freq, a*T0)/scale
			map[y,x] = oval/map_unit

@numba.njit
def _modulate_lin2T(map, A, T0=utils.T_cmb, freq=150e9, map_unit=1e-6, spin=0, dipole=False):
	scale= dplanck(freq, T=T0) # Jy/sr/K'
	off  = planck(freq,T0)     # Jy/sr
	for y in range(map.shape[0]):
		for x in range(map.shape[1]):
			a     = np.float64(A[y,x])
			ival  = np.float64(map[y,x]) # µK'
			# lin = P(T)/(dP/dT|T0) => T = P"(lin*(dP/dT|T0))
			oval  = ival*map_unit*scale # Jy/sr
			if spin == 0 and dipole:
				oval += off
			else:
				# Pre-emptively add a dipole that will be cancelled
				# when we apply the modulation below
				oval += planck(freq, 1/a*T0)
			# Go to full T units
			oval  = iplanck(freq, oval)
			# Apply modulation
			oval *= a
			oval -= utils.T_cmb
			map[y,x] = oval/map_unit

@numba.njit
def fast_rewind(arr, period, ref=None):
	if ref is None: ref = period/2
	ref2 = ref - period/2
	for i, a in enumerate(arr):
		# Try to make it branchless
		off     = a-ref
		arr[i] -= period*(off>=period)
		arr[i] += period*(off< 0)

# Should move this to enmap

def sky2pix(shape, wcs, pos):
	typ = wcs.wcs.ctype[0][-3:]
	if typ == "CAR" and wcs.wcs.crval[1] == 0:
		# Simple Plate Carree
		return np.array([
			(pos[0]-wcs.wcs.crval[1]*utils.degree)/(wcs.wcs.cdelt[1]*utils.degree)+(wcs.wcs.crpix[1]-1),
			(pos[1]-wcs.wcs.crval[0]*utils.degree)/(wcs.wcs.cdelt[0]*utils.degree)+(wcs.wcs.crpix[0]-1),
		])
	else:
		return enmap.sky2pix(shape, wcs, pos)
