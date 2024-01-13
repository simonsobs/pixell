import numpy as np, numba, ducc0
from . import coordinates, enmap, utils, curvedsky

beta    = 0.001235
dir_equ = np.array([167.919, -6.936])*np.pi/180
dir_gal = np.array([263.986, 48.247])*np.pi/180
dir_ecl = np.array([171.640,-11.154])*np.pi/180
from .utils import T_cmb, h, c, k

def boost_map(map, dir=dir_equ, beta=beta, modulation="thermo", T0=utils.T_cmb, freq=150e9,
		return_modulation=False, dipole=False, map_unit=1e-6, spin=[0,2], aberrate=True, modulate=True, nthread=None,
		coord_dtype=None):
	"""Doppler-boost (aberrate and modulate) the given input map map. The doppler boost
	goes in direction dir[{ra,dec}] with speed beta in units of c.

	spin controls how the 3rd last axis is interpreted. spin = [0,2] means that the
	first entry is scalar and the next pair is spin-2. So this works for a [TQU,ny,nx]
	map. It is safe to pass [0,2] even when the map is just scalar.

	If modulation == "thermo", then the input map is assumed to be in
	differential thermodynamic units, e.g. what CMB maps usually have, and a
	frequency-dependent gain factor is applied. In this case the T0 and freq
	arguments will be used to compute the gain factor using the thermo_boost function.
	If modulation == "plain", then the temperature modulation is directly multiplied
	with the map.
	If modulation == None, then no modulation is applied at all.

	The boundary conditions are assumed to be periodic in both directions. This will
	be slightly inaccurate at the poles for fullsky maps. This could be fixed.

	If return_modulation == True, then a tuple of (omap, A) will be returned, where
	A is the modulation as returned from calc_boost. This can be useful if one wants
	to compute the same map modulated at several different frequencies.
	
  The modulation and aberration steps can be individually skipped using the
  corresponding arguments. By default both are performed.
	"""
	if return_modulation: assert modulate, "Can't return modulation of modulation is disabled"
	if aberrate:
		map    = aberrate_map(map, dir=dir, beta=beta, spin=spin, nthread=nthread)
	if modulate:
		map, A = modulate_map(map, dir=dir, beta=beta, spin=spin, nthread=nthread,
			modulation=modulation, T0=T0, freq=freq, map_unit=map_unit, return_modulation=True)
	if return_modulation: return map, A
	else: return map

def aberrate_map(map, dir=dir_equ, beta=beta, spin=[0,2],
		nthread=None, coord_dtype=None):
	"""Perform the aberration part of the doppler boost. See
	boost_map for details."""
	if coord_dtype is None: coord_dtype = map.dtype
	return Aberrator(map.shape, map.wcs, dir=dir, beta=beta, spin=spin,
		nthread=nthread, coord_dtype=coord_dtype)(map)

def modulate_map(map, dir=dir_equ, beta=beta,
		modulation="thermo", T0=utils.T_cmb, freq=150e9, return_modulation=False,
		dipole=False, map_unit=1e-6, spin=[0,2], nthread=None):
	"""Perform the modulation part of the doppler boost. See boost_map
	for details."""
	modulator = Modulator(map.shape, map.wcs, dir=dir, beta=beta, spin=spin,
		modulation=modulation, T0=T0, freq=freq, dipole=dipole, map_unit=map_unit,
		nthread=nthread, dtype=map.dtype)
	map = modulator(map)
	if return_modulation: return map, modulator.A
	else: return map

# Classes for repeat calcuations with similar maps

class Aberrator:
	def __init__(self, shape, wcs, dir=dir_equ, beta=beta, spin=[0,2],
			nthread=None, coord_dtype=np.float64, scale_pix=True):
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
		# 2. Evaluate these on our target geometry. Hardcoded float64 because of get_deflected_angles
		deflect = enmap.zeros(alm_dpos.shape[:-1]+shape[-2:], wcs, np.float64)
		curvedsky.alm2map(alm_dpos.astype(coord_ctype, copy=False), deflect, spin=1, nthread=nthread)
		# 3. Calculate the offset angles.
		# get_deflected_angles only supports float64 :(
		rinfo = curvedsky.get_ring_info(shape, wcs)
		dphi  = np.full(shape[-2], wcs.wcs.cdelt[0]*utils.degree)
		tmp   = ducc0.misc.get_deflected_angles(theta=rinfo.theta, phi0=rinfo.phi0,
			nphi=rinfo.nphi, dphi=dphi, ringstart=rinfo.offsets, nthreads=nthread, calc_rotation=True,
			deflect=np.asarray(deflect).reshape(2,-1).T).T
		del deflect
		# We drop down to the target precision as early as possible
		odec, ora, gamma = tmp.astype(coord_dtype, copy=False)
		odec = np.pi/2-odec
		gamma= enmap.ndmap(gamma.reshape(shape[-2:]), wcs)
		del tmp
		# 4. Calculate pixel coordinates of offset angles. In general this would use
		# enmap.sky2pix, but that's slow. Much faster for our typical projections.
		# Probably worth it to make overrides.
		pix = sky2pix(shape, wcs, [odec, ora]).astype(coord_dtype, copy=False)
		# Make the x pixel 0:nphi, assuming it's at most one period away
		fast_rewind(pix[1].reshape(-1), rinfo.nphi[0])
		del odec, ora
		# 5. Evaluate the map at these locations using nufft
		if scale_pix:
			pix[0] /= shape[-2]
			pix[1] /= shape[-1]
		# 6. Store for when the map is passed in later
		self.pix       = pix
		self.scale_pix = scale_pix
		self.gamma     = gamma
		self.nthread   = nthread
		self.spin      = spin
	def __call__(self, map, spin=None, nthread=None):
		if nthread is None: nthread = self.nthread
		if spin    is None: spin    = self.spin
		shape, wcs = map.shape, map.wcs
		map = interpol_map(map, self.pix, nthread=nthread, scaled=self.scale_pix)
		map = enmap.ndmap (map.reshape(shape), wcs)
		# 6. Apply polarization rotation. ducc0.misc.lensing_rotate can do this,
		# but for some reason it operates on complex numbers instead of a QU field.
		# So seems like I would have to waste time and memory transforming from/to
		# this ordering. We loop over pre-dimensions to reduce memory use
		for s, I in enmap.spin_pre_helper(spin, map.shape[:-2]):
			rotate_pol(utils.fix_zero_strides(map[I]), self.gamma, spin=s)
		return map

class Modulator:
	def __init__(self, shape, wcs, dir=dir_equ, beta=beta,
			modulation="thermo", T0=utils.T_cmb, freq=150e9, dipole=False, tiny=False,
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
		alm_dpos, alm_mod = calc_boost_field(-beta, dir, nthread=nthread, modulation=True, mod_exp=-1)
		# 2: Apply modulation
		A = enmap.zeros(alm_mod.shape[:-1]+shape[-2:], wcs, dtype)
		curvedsky.alm2map(alm_mod.astype(utils.complex_dtype(dtype)), A, spin=0, nthread=nthread)
		# Store for __call__
		self.nthread = nthread;  self.A     = A;     self.modulation = modulation
		self.T0      = T0;       self.freq  = freq;  self.dipole     = dipole
		self.tiny    = tiny;     self.dtype = dtype; self.spin       = spin
		self.map_unit= map_unit; self.dtype = dtype
	def __call__(self, map, spin=None):
		if spin is None: spin = self.spin
		if map.dtype != self.dtype:
			warnings.warn("Modulator dtype does not match argument dtype")
		return apply_modulation(map, self.A, spin=spin, T0=self.T0, freq=self.freq,
			map_unit=self.map_unit, mode=self.modulation, dipole=self.dipole, tiny=self.tiny)

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

def beta2lmax(beta):
	# Infer lmax from beta. Empirical formula
	gamma = (1-beta**2)**-0.5
	return utils.ceil(7*gamma*(beta+1))

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

def interpol_map(imap, pixs, epsilon=None, nthread=None, scaled=False):
	# Should apparently ideally use doubling here,
	# but not worth it if we're on partial sky, which we usually
	# are. May make sense to special case it...
	# We have to rescale pixs like this because nufft doesn't support different
	# periodicity on different axes.
	if not scaled:
		pixs[0] /= imap.shape[-2]
		pixs[1] /= imap.shape[-1]
	nthread = int(utils.fallback(utils.getenv("OMP_NUM_THREADS",nthread),0))
	pflat   = pixs.reshape(2,-1).T
	if epsilon is None:
		epsilon = 1e-5 if imap.dtype == np.float32 else 1e-12
	oarr    = np.zeros(imap.shape[:-2]+pflat.shape[:1], imap.dtype)
	for I in utils.nditer(imap.shape[:-2]):
		fmap = enmap.fft(imap[I], normalize=False)
		oarr[I] = ducc0.nufft.u2nu(grid=np.asarray(fmap), coord=pflat, forward=False,
			epsilon=epsilon, nthreads=nthread, periodicity=1.0, fft_order=True).real
		del fmap
	# Restore predims
	oarr = oarr.reshape(imap.shape[:-2]+oarr.shape[-1:])
	oarr/= imap.npix
	# Scale back. Could have worked on a copy instead, but that would
	# use more memory
	if not scaled:
		pixs[0] *= imap.shape[-2]
		pixs[1] *= imap.shape[-1]
	return oarr

@numba.njit((numba.float32[:,:,:],numba.float32[:,:],numba.float32), nogil=True)
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

def apply_modulation(map, A, T0=utils.T_cmb, freq=150e9, map_unit=1e-6, mode="thermo",
		dipole=False, spin=[0,2], tiny=False):
	if    mode is None: pass
	elif  mode == "plain": map *= A
	elif  mode == "thermo":
		# We're in linearized thermodynamic units. We assume that the map doesn't contain the
		# monopole, so we can treat it as a perturbation around the monopole. If the map
		# contains the monopole, then linearized units probably isn't the best choice
		for s, I in enmap.spin_pre_helper(spin, map.shape[:-2]):
			for comp in map[I]:
				_apply_modulation_thermo(comp, A, T0, freq, map_unit, spin=s, dipole=dipole, tiny=tiny)
	else: raise ValueError("Urecognized modulation mode '%s'" % mode)
	return map

@numba.njit
def _apply_modulation_thermo(map, A, T0=utils.T_cmb, freq=150e9, map_unit=1e-6, spin=0, dipole=False, tiny=False):
	xh   = 0.5*utils.h*freq/(utils.k*T0)
	f    = xh/np.tanh(xh)-1
	t0_T = T0/map_unit
	t0   = t0_T if spin == 0 else 0
	for y in range(map.shape[0]):
		for x in range(map.shape[1]):
			a    = A[y,x]
			ival = map[y,x]
			oval = a*ival                            # basic modulation: ~100 uK*1e-3 = 0.1 uK (b^1)
			oval+= f*((a-1)**2*t0 + 2*a*(a-1)*ival)  # quad: 2.7K*1e-6*f ~ 1 uK (b^2); + ~100 uK*1e-3 ~ 0.1 uK (b^1)
			if dipole: oval  += (a-1)*t0             # dipole: 2.7K*1e-3 ~ 1000 uK (b^1)
			if tiny:   oval  += f*a**2*ival**2/t0_T  # tiny: ~100 uK*(100uK/2.7K) ~ 0.01 uK
			map[y,x] = oval

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

###### Old stuff #######

#def boost_map(imap, dir=dir_equ, beta=beta, pol=True, modulation="thermo", T0=T_cmb, freq=150e9,
#		boundary="wrap", order=3, recenter=False, return_modulation=False,
#		dipole=False, map_unit=1e-6, aberrate=True, modulate=True):
#	"""Doppler-boost (aberrate and modulate) the given input map imap. The doppler boost
#	goes in direction dir[{ra,dec}] with speed beta in units of c. If pol=True and
#	the map isn't scalar, then the map will be assumed to be [TQU], and a parallel-
#	transport-induced polarization rotation will be applied to QU.
#
#	If modulation == "thermo", then the input map is assumed to be in
#	differential thermodynamic units, e.g. what CMB maps usually have, and a
#	frequency-dependent gain factor is applied. In this case the T0 and freq
#	arguments will be used to compute the gain factor using the thermo_boost function.
#	If modulation == "plain", then the temperature modulation is directly multiplied
#	with the map.
#	If modulation == None, then no modulation is applied at all.
#
#	The boundary and order arguments control the spline interpolation used in the
#	aberration. See enmap.at() for details.
#
#	If recenter == True, then the mean map displacement due to the aberration is subtracted.
#	This removes the largest component of the aberration for small patches, which is just
#	an overall pointing shift. This can be useful for visualization purposes.
#
#	If return_modulation == True, then a tuple of (omap, A) will be returned, where
#	A is the modulation as returned from calc_boost. This can be useful if one wants
#	to compute the same map modulated at several different frequencies.
#	
#  The modulation and aberration steps can be individually skipped using the
#  corresponding arguments. By default both are performed.
#	"""
#	if imap.ndim < 3 or not aberrate: pol = False
#	opos = imap.posmap()
#	# we swap between enmap's dec,ra convention and the ra,dec convention we use here here.
#	# we use -beta because we know where the pixels in the *observed* frame is, and want
#	# to know here they should be in the raw frame. This also means that A will need to be
#	# inverted, since it now wants to take us from observed to raw too
#	ipos, A = calc_boost(opos[::-1], dir, -beta, pol=pol, recenter=recenter)
#	A **= -1
#	omap = imap
#	if aberrate: omap = apply_aberration(omap, ipos, boundary=boundary, order=order)
#	if modulate: omap = apply_modulation(omap, A, T0=T0, freq=freq, map_unit=map_unit, mode=modulation, dipole=dipole)
#	if return_modulation: return omap, A
#	else:                 return omap
#
#class Aberrator:
#	def __init__(self, shape, wcs, dir=dir_equ, beta=beta, pol=True, modulation="thermo", T0=T_cmb, freq=150e9,
#			boundary="wrap", order=3, recenter=False, dipole=False, map_unit=1e-6):
#		"""Initialize an Aberrator for maps with geometry (shape, wcs) in the
#		direction dir {ra,dec} with speed beta in units of c. If pol is True,
#		the (small) polarization rotation from the change in coordinate system
#		will also be computed.
#
#		If modulation == "thermo", then the input maps will be assumed to be in
#		differential thermodynamic units, e.g. what CMB maps usually have, and a
#		frequency-dependent gain factor will be applied. In this case the T0 and freq
#		arguments will be used to compute the gain factor using the thermo_boost function.
#		If modulation == "plain", then the temperature modulation is directly multiplied
#		with the map.
#		If modulation == None, then no modulation is applied at all.
#
#		If dipole == True, then the dipole induced from from an implied monopole with temperature
#		T0 will be included in the output.
#
#		The boundary and order arguments control the spline interpolation used in the
#		aberration. See enmap.at() for details.
#
#		In general
#		 aberrator = Aberrator(shape, wcs, ...)
#		 omap = aberrator.boost(imap)
#		is equivalent to
#		 omap = boost_map(imap, ...)
#		However, Aberrator will be more efficient when multiple maps with the
#		same geometry all need to be boosted the same way, as much of the calculation
#		can be precomputed in the constructor and reused for each map."""
#		# Save parameters for later
#		self.shape, self.wcs  = shape, wcs                                          # geometry
#		self.dir,   self.beta, self.pol, self.recenter = dir, beta, pol, recenter   # boost
#		self.boundary, self.order = boundary, order                                 # interpolation
#		self.T0, self.freq, self.dipole = T0, freq, dipole                          # modulation
#		self.map_unit, self.modulation = map_unit, modulation                       # modulation
#		# Precompute displacement
#		opos = enmap.posmap(shape, wcs)
#		ipos, A = calc_boost(opos[::-1], dir, -beta, pol=pol, recenter=recenter)
#		self.A = 1/A
#		self.ipix = enmap.ndmap(enmap.sky2pix(shape, wcs, ipos[1::-1]), wcs)
#		if pol:
#			self.cos = np.cos(2*ipos[2])
#			self.sin = np.sin(2*ipos[2])
#	def aberrate(self, imap):
#		"""Apply the aberration part of the doppler boost to the map imap"""
#		omap = enmap.samewcs(imap.at(self.ipix, unit="pix", mode=self.boundary, order=self.order), imap)
#		if self.pol and imap.ndim > 2:
#			omap1 = omap[...,1,:,:].copy()
#			omap[...,1,:,:] =  self.cos*omap1 + self.sin*omap[...,2,:,:]
#			omap[...,2,:,:] = -self.sin*omap1 + self.cos*omap[...,2,:,:]
#		return omap
#	def modulate(self, imap):
#		"""Apply the modulation part of the doppler boost to the map omap"""
#		omap = apply_modulation(imap, self.A, T0=self.T0, freq=self.freq, map_unit=self.map_unit, mode=self.modulation, dipole=self.dipole)
#		return omap
#	def boost(self, imap):
#		"""Apply the full doppler boost to the map imap"""
#		omap = self.aberrate(imap)
#		omap = self.modulate(omap)
#		return omap
#
#def apply_aberration(imap, ipos, boundary="wrap", order=3):
#	omap = enmap.samewcs(imap.at(ipos[1::-1], mode=boundary, order=order), imap)
#	if len(ipos) >= 3:
#		c,s = np.cos(2*ipos[2]), np.sin(2*ipos[2])
#		omap1 = omap[1].copy()
#		omap[1] = c*omap1 + s*omap[2]
#		omap[2] =-s*omap1 + c*omap[2]
#	return omap
#
#def apply_modulation(imap, A, T0=T_cmb, freq=150e9, map_unit=1e-6, mode="thermo",
#		dipole=False, pol=True, tiny=False):
#	if    mode is None:     return imap
#	elif  mode == "plain":  return imap*A
#	elif  mode == "thermo":
#		# We're in linearized thermodynamic units. We assume that the map doesn't contain the
#		# monopole, so we can treat it as a perturbation around the monopole. If the map
#		# contains the monopole, then linearized units probably isn't the best choice
#		iflat = imap.preflat
#		t0 = np.zeros([len(iflat),1,1])
#		if pol: t0[0] = T0/map_unit
#		else:   t0[:] = T0/map_unit
#		xh = 0.5*h*freq/(k*T0)
#		f  = xh/np.tanh(xh)-1
#		A1 = A-1
#		oflat  = A*iflat                             # basic modulation: ~100 uK*1e-3 = 0.1 uK (b^1)
#		oflat += f*(A1**2*t0 + 2*A*A1*iflat)         # quad: 2.7K*1e-6*f ~ 1 uK (b^2); + ~100 uK*1e-3 ~ 0.1 uK (b^1)
#		if dipole: oflat += A1*t0                    # dipole: 2.7K*1e-3 ~ 1000 uK (b^1)
#		if tiny:   oflat += f*A**2*iflat**2/t0[0]    # tiny: ~100 uK*(100uK/2.7K) ~ 0.01 uK
#		omap = oflat.reshape(imap.shape)
#		return omap
#	else: raise ValueError("Urecognized modulation mode '%s'" % mode)
#
#def calc_boost(pos, dir, beta, pol=True, recenter=False):
#	"""Given position angles pos_rest[{ra,dec,[phi]},...] in radians, returns
#	returns the aberrated pos_obs[{ra,dec,[phi]}], and the corresponding modulation
#	amplitude A[...] as a tuple. To get the inverse transform, from observed to
#	rest-frame coordinates, pass in -beta instead of beta.
#	
#	phi is the optional local basis rotation, which will be computed if
#	pol=True (the default). This angle needs to be taken into account for vector fields
#	like polarization.
#	
#	If recenter is True, then the mean of the position shift will be subtracted. This
#	only makes sense for transformations of points covering only a part of the sky.
#	It is intended for visualization purposes."""
#	# Flatten, so we're [{ra,dec,phi?},:]
#	pos    = np.asarray(pos)
#	res    = pos.copy().reshape(pos.shape[0],-1)
#	# First transform to a coordinate system where we're moving in the z direction.
#	res    = coordinates.transform("equ",["equ",[dir,False]],res,pol=pol)
#	if recenter: before = np.mean(res[1,::10])
#	# Apply aberration, which is a pure change of z in this system
#	z        = np.cos(np.pi/2-res[1])
#	z_obs, A = calc_boost_1d(z, beta)
#	res[1]   = np.pi/2-np.arccos(z_obs)
#	if recenter: res[1] -= np.mean(res[1,::10])-before
#	res = coordinates.transform(["equ",[dir,False]],"equ",res,pol=pol)
#	# Reshape to original shape
#	res = res.reshape(res.shape[:1]+pos.shape[1:])
#	A   = A.reshape(pos.shape[1:])
#	return res, A
#
### These would be useful for exact modulation calculations,                               ###
### but by that point one is down to errors small enough that one has to care about the    ###
### difference between brightness temperature and linearized temperature in the input map  ###
### and which of these would be expected to be gaussian                                    ###
#
#def planck(nu, T, deriv=False):
#	"""Compute the planck spectrum at frequency nu (in Hz) and temperature T (in K).
#	If deriv=True, then the derivative with respect to temperature will be returned"""
#	a = 2*h*nu**3/c**2
#	x = h*nu/(k*T)
#	e = np.exp(x)
#	b = 1/(np.exp(x)-1)
#	if not deriv: return a*b
#	else: return a*b**2*e*x/T
#
#def inv_planck(nu, I, T0=T_cmb, niter=5):
#	"""Estimate the temperature T (K) given the frequency nu (Hz) and intensity I.
#	Uses Newton iteration, and should be very accurate for typical CMB maps."""
#	T = T0
#	for i in range(niter):
#		T -= (planck(nu, T)-I)/planck(nu, T, deriv=True)
#	return T
#
##### Old stuff below here. Can be removed after a while ####
#
#
#def remap(pos, dir, beta, pol=True, modulation=True, recenter=False):
#	"""Given a set of coordinates "pos[{ra,dec]}", computes the aberration
#	deflected positions for a speed beta in units of c in the
#	direction dir. If pol=True (the default), then the output
#	will have three columns, with the third column being
#	the aberration-induced rotation of the polarization angle."""
#	pos = coordinates.transform("equ",["equ",[dir,False]],pos,pol=pol)
#	if recenter: before = np.mean(pos[1,::10])
#	# Use -beta here to get the original position from the deflection,
#	# instead of getting the deflection from the original one as
#	# aber_angle normally computes.
#	pos[1] = np.pi/2-aber_angle(np.pi/2-pos[1], -beta)
#	if recenter:
#		after = np.mean(pos[1,::10])
#		pos[1] -= after-before
#	res = coordinates.transform(["equ",[dir,False]],"equ",pos,pol=pol)
#	if modulation:
#		amp = mod_amplitude(np.pi/2-pos[1], beta)
#		res = np.concatenate([res,[amp]])
#	return res
#
#def distortion(pos, dir, beta):
#	"""Returns the local aberration distortion, defined as the
#	second derivative of the aberration displacement."""
#	pos = coordinates.transform("equ",["equ",[dir,False]],pos,pol=True)
#	return aber_deriv(np.pi/2-pos[1], -beta)-1
#
#def aberrate(imap, dir, beta, mode="wrap", order=3, recenter=False, modulation=True):
#	pol = imap.ndim > 2
#	pos = imap.posmap()
#	# The ::-1 stuff switches between dec,ra and ra,dec ordering.
#	# It is a bit confusing to have different conventions in enmap
#	# and coordinates.
#	pos = remap(pos[::-1], dir, beta, pol=pol, recenter=recenter, modulation=modulation)
#	pos[:2] = pos[1::-1]
#	pix = imap.sky2pix(pos[:2], corner=True) # interpol needs corners
#	omap= enmap.ndmap(utils.interpol(imap, pix, mode=mode, order=order), imap.wcs)
#	if pol:
#		c,s = np.cos(2*pos[2]), np.sin(2*pos[2])
#		omap[1] = c*omap[1] + s*omap[2]
#		omap[2] =-s*omap[1] + c*omap[2]
#	if modulation:
#		omap *= pos[2+pol]
#	return omap
#
#def aber_angle(theta, beta):
#	"""The aberrated angle as a function of the input angle.
#	That is: return the zenith angle of a point in the deflected
#	cmb given the zenith angle of the undeflected point."""
#	c = np.cos(theta)
#	gamma = (1-beta**2)**-0.5
#	c = (c+(gamma-1)*c+gamma*beta)/(gamma*(1+c*beta))
#	#c = (c+beta)/(1+beta*c)
#	return np.arccos(c)
#
#def mod_amplitude(theta, beta):
#	c = np.cos(theta)
#	gamma = (1-beta**2)**-0.5
#	return 1/(gamma*(1-c*beta))
#	#return 1/(1-beta*c)
#
#def aber_deriv(theta, beta):
#	B = 1-beta**2
#	C = 1-beta*np.cos(theta)
#	return B**0.5/C
