import numpy as np
from . import coordinates, enmap, utils

beta    = 0.0012301
dir_equ = np.array([167.929, -6.927])*np.pi/180
dir_gal = np.array([263.990, 48.260])*np.pi/180
dir_ecl = np.array([171.646,-11.141])*np.pi/180
from utils import T_cmb, h, c, k

def boost_map(imap, dir=dir_equ, beta=beta, pol=True, modulation="thermo", T0=T_cmb, freq=150e9,
		boundary="wrap", order=3, recenter=False, return_modulation=False,
		dipole=False, map_unit=1e-6):
	"""Doppler-boost (aberrate and modulate) the given input map imap. The doppler boost
	goes in direction dir[{ra,dec}] with speed beta in units of c. If pol=True and
	the map isn't scalar, then the map will be assumed to be [TQU], and a parallel-
	transport-induced polarization rotation will be applied to QU.

	If modulation == "thermo", then the input map is assumed to be in
	differential thermodynamic units, e.g. what CMB maps usually have, and a
	frequency-dependent gain factor is applied. In this case the T0 and freq
	arguments will be used to compute the gain factor using the thermo_boost function.
	If modulation == "plain", then the temperature modulation is directly multiplied
	with the map.
	If modulation == None, then no modulation is applied at all.

	The boundary and order arguments control the spline interpolation used in the
	aberration. See enmap.at() for details.

	If recenter == True, then the mean map displacement due to the aberration is subtracted.
	This removes the largest component of the aberration for small patches, which is just
	an overall pointing shift. This can be useful for visualization purposes.

	If return_modulation == True, then a tuple of (omap, A) will be returned, where
	A is the modulation as returned from calc_boost. This can be useful if one wants
	to compute the same map modulated at several different frequencies."""
	if imap.ndim < 3: pol = False
	opos = imap.posmap()
	# we swap between enmap's dec,ra convention and the ra,dec convention we use here here.
	# we use -beta because we know where the pixels in the *observed* frame is, and want
	# to know here they should be in the raw frame. This also means that A will need to be
	# inverted, since it now wants to take us from observed to raw too
	ipos, A  = calc_boost(opos[::-1], dir, -beta, pol=pol, recenter=recenter)
	ipos[:2] = ipos[1::-1]
	A      **= -1 # invert A to go from raw to observed
	omap = enmap.samewcs(imap.at(ipos[:2], mode=boundary, order=order), imap)
	if pol:
		c,s = np.cos(2*ipos[2]), np.sin(2*ipos[2])
		omap[1] = c*omap[1] + s*omap[2]
		omap[2] =-s*omap[1] + c*omap[2]
	omap = apply_modulation(omap, A, T0=T0, freq=freq, map_unit=map_unit, mode=modulation, dipole=dipole)
	if return_modulation: return omap, A
	else:                 return omap

def apply_modulation(imap, A, T0=T_cmb, freq=150e9, map_unit=1e-6, mode="thermo",
		dipole=False, pol=True, tiny=False):
	if    mode is None:     return imap
	elif  mode == "plain":  return imap*A
	elif  mode == "thermo":
		# We're in linearized thermodynamic units. We assume that the map doesn't contain the
		# monopole, so we can treat it as a perturbation around the monopole. If the map
		# contains the monopole, then linearized units probably isn't the best choice
		iflat = imap.preflat
		t0 = np.zeros([len(iflat),1,1])
		if pol: t0[0] = T0/map_unit
		else:   t0[:] = T0/map_unit
		xh = 0.5*h*freq/(k*T0)
		f  = xh/np.tanh(xh)-1
		A1 = A-1
		oflat  = A*iflat
		oflat += f*(A1**2*t0 + 2*A*A1*iflat)
		if dipole: oflat += A1*t0
		if tiny:   oflat += f*A**2*iflat**2/t0[0]
		omap = oflat.reshape(imap.shape)
		return omap
	else: raise ValueError("Urecognized modulation mode '%s'" % mode)

def calc_boost(pos, dir, beta, pol=True, recenter=False):
	"""Given position angles pos_rest[{ra,dec,[phi]},...] in radians, returns
	returns the aberrated pos_obs[{ra,dec,[phi]}], and the corresponding modulation
	amplitude A[...] as a tuple. To get the inverse transform, from observed to
	rest-frame coordinates, pass in -beta instead of beta.
	
	phi is the optional local basis rotation, which will be computed if
	pol=True (the default). This angle needs to be taken into account for vector fields
	like polarization.
	
	If recenter is True, then the mean of the position shift will be subtracted. This
	only makes sense for transformations of points covering only a part of the sky.
	It is intended for visualization purposes."""
	# Flatten, so we're [{ra,dec,phi?},:]
	pos    = np.asarray(pos)
	res    = pos.copy().reshape(pos.shape[0],-1)
	# First transform to a coordinate system where we're moving in the z direction.
	res    = coordinates.transform("equ",["equ",[dir,False]],res,pol=pol)
	if recenter: before = np.mean(res[1,::10])
	# Apply aberration, which is a pure change of z in this system
	z        = np.cos(np.pi/2-res[1])
	z_obs, A = calc_boost_1d(z, beta)
	res[1]   = np.pi/2-np.arccos(z_obs)
	if recenter: res[1] -= np.mean(res[1,::10])-before
	res = coordinates.transform(["equ",[dir,False]],"equ",res,pol=pol)
	# Reshape to original shape
	res = res.reshape(res.shape[:1]+pos.shape[1:])
	A   = A.reshape(pos.shape[1:])
	return res, A

def calc_boost_1d(z, beta, mixed=False):
	"""Given the z, the cosine of the angle from the direction of travel in
	the CMB rest frame, and beta, the speed as a fraction of c, compute the
	observed z_obs after taking into account the effect of aberration, as well
	as the corresponding modulation A, such that T_obs(z_obs) = A * T_rest(z).
	Pass in -beta to get the reverse transformation, from obs to rest."""
	gamma = (1-beta**2)**-0.5
	beta2 = beta if not mixed else -beta
	z_obs = (z+beta)/(1+z*beta2)
	A     = 1/(gamma*(1-z*beta))
	return z_obs, A

### These would be useful for exact modulation calculations,                               ###
### but by that point one is down to errors small enough that one has to care about the    ###
### difference between brightness temperature and linearized temperature in the input map  ###
### and which of these would be expected to be gaussian                                    ###

def planck(nu, T, deriv=False):
	"""Compute the planck spectrum at frequency nu (in Hz) and temperature T (in K).
	If deriv=True, then the derivative with respect to temperature will be returned"""
	a = 2*h*nu**3/c**2
	x = h*nu/(k*T)
	e = np.exp(x)
	b = 1/(np.exp(x)-1)
	if not deriv: return a*b
	else: return a*b**2*e*x/T

def inv_planck(nu, I, T0=T_cmb, niter=5):
	"""Estimate the temperature T (K) given the frequency nu (Hz) and intensity I.
	Used Newton iteration, and should be very accurate for typical CMB maps."""
	T = T0
	for i in range(niter):
		T -= (planck(nu, T)-I)/planck(nu, T, deriv=True)
	return T





#### Old stuff below here. Can be removed after a while ####


def remap(pos, dir, beta, pol=True, modulation=True, recenter=False):
	"""Given a set of coordinates "pos[{ra,dec]}", computes the aberration
	deflected positions for a speed beta in units of c in the
	direction dir. If pol=True (the default), then the output
	will have three columns, with the third column being
	the aberration-induced rotation of the polarization angle."""
	pos = coordinates.transform("equ",["equ",[dir,False]],pos,pol=pol)
	if recenter: before = np.mean(pos[1,::10])
	# Use -beta here to get the original position from the deflection,
	# instead of getting the deflection from the original one as
	# aber_angle normally computes.
	pos[1] = np.pi/2-aber_angle(np.pi/2-pos[1], -beta)
	if recenter:
		after = np.mean(pos[1,::10])
		pos[1] -= after-before
	res = coordinates.transform(["equ",[dir,False]],"equ",pos,pol=pol)
	if modulation:
		amp = mod_amplitude(np.pi/2-pos[1], beta)
		res = np.concatenate([res,[amp]])
	return res

def distortion(pos, dir, beta):
	"""Returns the local aberration distortion, defined as the
	second derivative of the aberration displacement."""
	pos = coordinates.transform("equ",["equ",[dir,False]],pos,pol=True)
	return aber_deriv(np.pi/2-pos[1], -beta)-1

def aberrate(imap, dir, beta, mode="wrap", order=3, recenter=False, modulation=True):
	pol = imap.ndim > 2
	pos = imap.posmap()
	# The ::-1 stuff switches between dec,ra and ra,dec ordering.
	# It is a bit confusing to have different conventions in enmap
	# and coordinates.
	pos = remap(pos[::-1], dir, beta, pol=pol, recenter=recenter, modulation=modulation)
	pos[:2] = pos[1::-1]
	pix = imap.sky2pix(pos[:2], corner=True) # interpol needs corners
	omap= enmap.ndmap(utils.interpol(imap, pix, mode=mode, order=order), imap.wcs)
	if pol:
		c,s = np.cos(2*pos[2]), np.sin(2*pos[2])
		omap[1] = c*omap[1] + s*omap[2]
		omap[2] =-s*omap[1] + c*omap[2]
	if modulation:
		omap *= pos[2+pol]
	return omap

def aber_angle(theta, beta):
	"""The aberrated angle as a function of the input angle.
	That is: return the zenith angle of a point in the deflected
	cmb given the zenith angle of the undeflected point."""
	c = np.cos(theta)
	gamma = (1-beta**2)**-0.5
	c = (c+(gamma-1)*c+gamma*beta)/(gamma*(1+c*beta))
	#c = (c+beta)/(1+beta*c)
	return np.arccos(c)

def mod_amplitude(theta, beta):
	c = np.cos(theta)
	gamma = (1-beta**2)**-0.5
	return 1/(gamma*(1-c*beta))
	#return 1/(1-beta*c)

def aber_deriv(theta, beta):
	B = 1-beta**2
	C = 1-beta*np.cos(theta)
	return B**0.5/C
