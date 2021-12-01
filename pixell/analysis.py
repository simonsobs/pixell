"""This is a module for analysing sky maps: stuff like matched filtering, feature
detection, pixel-space likelihoods etc.


Example usage of matched filter functions:

import numpy as np
from pixell import enmap, utils, uharm, analysis, curvedsky

np.random.seed(1)

# 0. Set up our geometry, a 100 uK peak point source and a 1.4 arcmin beam
pos        = [0,0]
shape, wcs = enmap.geometry(np.array([[-2,2],[2,-2]])*utils.degree, res=0.5*utils.arcmin)
pixarea    = enmap.pixsizemap(shape, wcs)
bsigma     = 1.4*utils.fwhm*utils.arcmin
signal     = 100*np.exp(-0.5*enmap.modrmap(shape, wcs, pos)**2/bsigma**2)
uht        = uharm.UHT(shape, wcs)
beam       = np.exp(-0.5*uht.l**2*bsigma**2)
fconv      = utils.dplanck(150e9, utils.T_cmb)/1e3 # uK -> mJy/sr

# 1. Matched filter for 10 uK' white noise
ivar       = 10**-2*pixarea/utils.arcmin**2
noise      = enmap.rand_gauss(shape, wcs)/ivar**0.5
map        = signal # + noise  # uncomment to actually add noise
# fconv is used to convert map and ivar from uK to mJy/sr. That way
# our output flux will be in mJy instead of the weird unit uK*sr.
rho, kappa = analysis.matched_filter_white(map*fconv, beam, ivar/fconv**2, uht)
flux  = rho.at(pos)/kappa.at(pos)
dflux = kappa.at(pos)**-0.5
print("%-15s %8.3f %8.3f %8.3f" % ("white", flux, dflux, flux/dflux))
# white              7.487    0.711   10.537

# 2. same, but with a noise power spectrum instead of a noise map. 10 uK'
# white noise has a flat noise spectrum of 10**2 * arcmin**2 = 0.46e-6
iN = 10**-2/utils.arcmin**2
rho, kappa = analysis.matched_filter_constcov(map*fconv, beam, iN/fconv**2, uht)
flux  = rho.at(pos)/kappa # kappa just a number in this case
dflux = kappa**-0.5
print("%-15s %8.3f %8.3f %8.3f" % ("constcov white", flux, dflux, flux/dflux))
# constcov white     7.486    0.711   10.534

# 3. uniform white noise, but with support for noise spectrum and position-dependence.
#    The noise units are only in ivar, so the noise spectrum is just a dimensionless 1.
iN = 1
rho, kappa = analysis.matched_filter_constcorr_lowcorr(map*fconv, beam, ivar/fconv**2, iN, uht)
# Read of the flux in mJy
flux  = rho.at(pos)/kappa.at(pos)
dflux = kappa.at(pos)**-0.5
print("%-15s %8.3f %8.3f %8.3f" % ("lowcorr white", flux, dflux, flux/dflux))
# lowcorr white      7.487    0.711   10.537

# 4. same, but with the other approximation
rho, kappa = analysis.matched_filter_constcorr_smoothivar(map*fconv, beam, ivar/fconv**2, iN, uht)
# Read of the flux in mJy
flux  = rho.at(pos)/kappa.at(pos)
dflux = kappa.at(pos)**-0.5
print("%-15s %8.3f %8.3f %8.3f" % ("smooth white", flux, dflux, flux/dflux))
# smooth white       7.487    0.711   10.537

# 5. 1/f noise power spectrum with l_knee at 2000
iN   =  10**-2/utils.arcmin**2 / (1 + ((uht.l+0.5)/2000)**-3)
rho, kappa = analysis.matched_filter_constcov(map*fconv, beam, iN/fconv**2, uht)
flux  = rho.at(pos)/kappa # kappa just a number in this case
dflux = kappa**-0.5
print("%-15s %8.3f %8.3f %8.3f" % ("constcov 1/f", flux, dflux, flux/dflux))
# constcov 1/f       7.486    0.782    9.571

# 6. 1/f noise and position-dependent depth with lowcorr
ivar  = 10**-2*pixarea/utils.arcmin**2 # Base depth
# spatial modulation with 5 arcmin wavelength horizontal sine wave
ivar *= (1+0.9*np.sin(enmap.posmap(shape, wcs)[1]/(5*utils.arcmin)))
# 1/f spectrum, but dimensionless since ivar handles the units
iN   =  1 / (1 + ((uht.l+0.5)/2000)**-3)
rho, kappa = analysis.matched_filter_constcorr_lowcorr(map*fconv, beam, ivar/fconv**2, iN, uht)
# Read of the flux in mJy
flux  = rho.at(pos)/kappa.at(pos)
dflux = kappa.at(pos)**-0.5
print("%-15s %8.3f %8.3f %8.3f" % ("lowcorr full", flux, dflux, flux/dflux))
# lowcorr full       7.491    0.782    9.578

# 7. same, but with smoothivar
rho, kappa = analysis.matched_filter_constcorr_smoothivar(map*fconv, beam, ivar/fconv**2, iN, uht)
# Read of the flux in mJy
flux  = rho.at(pos)/kappa.at(pos)
dflux = kappa.at(pos)**-0.5
print("%-15s %8.3f %8.3f %8.3f" % ("smooth full", flux, dflux, flux/dflux))
# smooth full        7.483    0.782    9.568

# 8. measuring iN from noise realizations. First build our sims
nsim   = 100
white  = enmap.rand_gauss((nsim,)+shape, wcs)
noise  = uht.harm2map(uht.map2harm(white)*iN**-0.5)*ivar**-0.5
# Measure the 2d noise power spectrum after whitening with ivar,
# and take the mean over all our sims to reduce sample variance.
# Multiplying by the mean pixel size is necessary to get the units right.
iNemp  = 1/(np.mean(np.abs(uht.map2harm(noise*ivar**0.5))**2,0) / noise.pixsize())
# If you measure iN from noise realizations that have very close to zero mean, then
# iNemp may end up with a huge value in the [0,0] (DC) component, which can cause
# issues. In that case you may need to set iNemp[0,0] = 0.
del white, noise
# Use iNemp to matched filter
rho, kappa = analysis.matched_filter_constcorr_lowcorr(map*fconv, beam, ivar/fconv**2, iNemp, uht)
# Read of the flux in mJy
flux  = rho.at(pos)/kappa.at(pos)
dflux = kappa.at(pos)**-0.5
print("%-15s %8.3f %8.3f %8.3f" % ("lowcorr full emp", flux, dflux, flux/dflux))
# lowcorr full emp   7.491    0.778    9.626

# If you don't have a a large number of sims, you can try smoothing N before inverting
# it to from iN, but this can be tricky if N has too much contrast, like a very small
# region with very high values.
"""

import numpy as np
from . import enmap, utils, uharm

def matched_filter_constcov(map, B, iN, uht=None):
	"""Apply a matched filter to the given map, assuming a constant covariance
	noise model. A constant covariance noise model is one where the pixel-pixel
	covariance is independent of the position in the map, and can be represented
	using a diagonal noise matrix in harmonic space.

	Arguments:
	map:  The [...,ny,nx] enmap to be filtered
	B:    The instrumental beam in the "hprof" format of the provided UHT
	iN:   The inverse noise power in harmonic space, in "hprof" format.
	uht:  The unified harmonic transform (UHT) to use

	Returns rho, kappa, which are [...,ny,nx] enmaps that can be used to construct:
	flux  = rho/kappa
	dflux = kappa**-0.5
	snr   = rho/kappa**0.5

	Here flux is the flux estimate in each pixel. If map is in Jy/sr, then flux
	will be in Jy. dflux is the 1 sigma flux uncertainty. snr is a map of the
	signal-to-noise ratio.
	"""
	# Npix = < n_pix n_pix' > = < Y n_harm n_harm' Y'> = Y Nharm Y'
	# Npix" = Y'" Nharm" Y", Y" = Y'W
	# Npix" = WY Nharm" Y'W. Not the same as YNharm"Y"!
	if uht is None: uht = uharm.UHT(map.shape, map.wcs)
	pixarea = enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)
	rho     = uht.map2harm_adjoint(uht.hmul(B*iN,uht.map2harm(map)))/pixarea
	kappa   = uht.sum_hprof(B**2*iN)/(4*np.pi)
	return rho, kappa

def matched_filter_white(map, B, ivar, uht=None, B2=None, high_acc=False):
	"""Apply a matched filter to the given map, assuming a constant correlation
	noise model inv(N) = ivar, where ivar = 1/pixel_variance.
	This represents pixel-uncorrelated noise.

	Arguments:
	map:  The [...,ny,nx] enmap to be filtered
	B:    The instrumental beam in the "hprof" format of the provided UHT
	ivar: The inverse of the white noise power per pixel, an [...,ny,nx] enmap
	uht:  The unified harmonic transform (UHT) to use
	B2:   The *real-space* square of the beam in "hprof" format. Make sure that the
	      beam is properly normalized to have a unit integral before squaring.
	      Optional. If it is missing, the square will be performed internally.

	Returns rho, kappa, which are [...,ny,nx] enmaps that can be used to construct:
	flux  = rho/kappa
	dflux = kappa**-0.5
	snr   = rho/kappa**0.5

	Here flux is the flux estimate in each pixel. If map is in Jy/sr, then flux
	will be in Jy. dflux is the 1 sigma flux uncertainty. snr is a map of the
	signal-to-noise ratio.
	"""
	# m = YBY"Pa + n, N" = ivar
	# rho   = (YBY"P)'N"m = P'Y"'BY' ivar m = P'WYBY"/W ivar m. Makes sense: converts to intensive units before harmonic
	# kappa = (YBY"P)'N"(YBY"P)
	#       = P'(YBY")'ivar(YBY")P
	# kappa_ii = P_ii²*sum_j (YBY")_ji ivar_jj (YBY")_ji
	#          = P_ii²*sum_j (YBY")_ji² ivar_jj
	#          = P_ii²*sum_j (Y"'BY')_ij² ivar_jj
	P = 1/enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)
	# Square the beam in real space if not provided
	if uht is None: uht = uharm.UHT(map.shape, map.wcs)
	if B2  is None: B2  = uht.hprof_rpow(B, 2)
	# TODO: kappa should have P**2. Figure out why P**1 is what works.
	rho   = P*uht.map2harm_adjoint(uht.hmul(B ,uht.harm2map_adjoint(ivar*map)))
	kappa = P*uht.map2harm_adjoint(uht.hmul(B2,uht.harm2map_adjoint(ivar)))
	return rho, kappa

def matched_filter_constcorr_lowcorr(map, B, ivar, iC, uht=None, B2=None, high_acc=False):
	"""Apply a matched filter to the given map, assuming a constant correlation
	noise model inv(N) = ivar**0.5 * iC * ivar**0.5, where ivar = 1/pixel_variance
	and iC = 1/harmonic_power(noise*ivar**0.5). This represents correlated noise
	described by iC that's modulated spatially by ivar.

	Arguments:
	map:  The [...,ny,nx] enmap to be filtered
	B:    The instrumental beam in the "hprof" format of the provided UHT
	ivar: The inverse of the white noise power per pixel, an [...,ny,nx] enmap
	iC:   The inverse power spectrum of the whitened map, map/ivar**0.5, as computed using
	      uht. Note: This will be 1/pixsize for a white noise map, not 1, due to
	      how the fourier space units work.
	uht:  The unified harmonic transform (UHT) to use
	B2:   The *real-space* square of the beam in "hprof" format. Make sure that the
	      beam is properly normalized to have a unit integral before squaring.
	      Optional. If it is missing, the square will be performed internally.
	high_acc: We approximate kappa internally. If high_acc is True, then the exacat
	      kappa will be evaluated for a single pixel, and used to rescale the approximate
	      kappa.

	Returns rho, kappa, which are [...,ny,nx] enmaps that can be used to construct:
	flux  = rho/kappa
	dflux = kappa**-0.5
	snr   = rho/kappa**0.5

	Here flux is the flux estimate in each pixel. If map is in Jy/sr, then flux
	will be in Jy. dflux is the 1 sigma flux uncertainty. snr is a map of the
	signal-to-noise ratio.

	Internally we assume that we can replace iC with its beam²-weighted
	average when computing kappa. This is a good approximation as long as
	iC isn't too steep. Handles rapid variations and holes in ivar well,
	though a hole in the middle of a source can be 20% wrong. That's still
	much better than the smoothivar approximation.
	"""
	# m = YBY"Pa + n
	# rho   = P' Y"'B'Y' V Y'" Ch" Y" V m
	# kappa = P'Y"'B'Y'  V Y'" Ch" Y" V  YBY"P
	# Pretend that we can set Ch" = alpha in kappa
	# kappa approx alpha * P'Y"'B'Y' ivar YBY"P
	# kappa_ii = alpha*P_ii²*sum_j (YBY")_ji² ivar_jj
	#  = alpha * convolve(br**2, ivar)_ii
	if uht is None: uht = uharm.UHT(map.shape, map.wcs)
	pixarea = enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)
	V = ivar**0.5
	W = uht.quad_weights()
	# Square the beam in real space if not provided
	if B2 is None: B2 = uht.hprof_rpow(B, 2)
	# Find a white approximation for iC. A B²-weighted average is accurate to
	# about 1% for lknee = 4000, worsening to about 3% by lknee = 7000. Probably
	# good enough.
	iC_white = uht.sum_hprof(B**2*iC)/uht.sum_hprof(B**2)

	rho   = uht.harm2map(uht.hmul(B,uht.harm2map_adjoint(V*uht.map2harm_adjoint(uht.hmul(iC, uht.map2harm(V*map))))))/pixarea
	kappa = uht.map2harm_adjoint(uht.hmul(B2,uht.harm2map_adjoint(ivar*W*iC_white[...,None,None])))/pixarea**2

	if high_acc:
		# Optionally find a correction factor by evaluating the exact kappa in a single pixel
		pix  = tuple(np.array(map.shape[-2:])//2)
		u    = map*0; u[...,pix[0],pix[1]] = 1
		kappa_ii = (uht.harm2map(uht.hmul(B,uht.harm2map_adjoint(V*uht.map2harm_adjoint(uht.hmul(iC,uht.map2harm(V*uht.harm2map(uht.hmul(B,uht.map2harm(u/pixarea)))))))))/pixarea)[...,pix[0],pix[1]]
		alpha  = kappa[...,pix[0],pix[1]]/kappa_ii
		kappa /= alpha[...,None,None]

	return rho, kappa

def matched_filter_constcorr_smoothivar(map, B, ivar, iC, uht=None):
	"""Apply a matched filter to the given map, assuming a constant correlation
	noise model inv(N) = ivar**0.5 * iC * ivar**0.5, where ivar = 1/pixel_variance
	and iC = 1/harmonic_power(noise*ivar**0.5). This represents correlated noise
	described by iC that's modulated spatially by ivar.

	Arguments:
	map:  The [...,ny,nx] enmap to be filtered
	B:    The instrumental beam in the "hprof" format of the provided UHT
	ivar: The inverse of the white noise power per pixel, an [...,ny,nx] enmap
	iC:   The inverse power spectrum of the whitened map, map/ivar**0.5, as computed using
	      uht. Note: This will be 1/pixsize for a white noise map, not 1, due to
	      how the fourier space units work.
	uht:  The unified harmonic transform (UHT) to use

	Returns rho, kappa, which are [...,ny,nx] enmaps that can be used to construct:
	flux  = rho/kappa
	dflux = kappa**-0.5
	snr   = rho/kappa**0.5

	Here flux is the flux estimate in each pixel. If map is in Jy/sr, then flux
	will be in Jy. dflux is the 1 sigma flux uncertainty. snr is a map of the
	signal-to-noise ratio.

	Internally we assume that the beam can be commuted past V, such that
	B'N" approx V"B'C"V". This gives us an easy analytic expression for kappa,
	but implicitly assumes that the hitcount doesn't change very rapidly
	from pixel to pixel. It breaks down completely if there is a hole at the
	peak of a source.
	"""
	# See the constcov function for a bit more math details.
	# We assume that we can commute B past V, allowing us to compute kappa directly
	if uht is None: uht = uharm.UHT(map.shape, map.wcs)
	V    = ivar**0.5
	P    = 1/enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)
	rho  = P*V*uht.map2harm_adjoint(uht.hmul(B*iC,uht.harm2map_adjoint(V*map)))
	# kappa = P'VY"'B Ch" B Y"VP = (P'V = R)(Y"'B sqrt(CH") = A)A'R' = RAA'R'
	# kappa_ii = R_ii² sum_l A_il
	kappa = ivar * (uht.sum_hprof(B**2*iC)/(4*np.pi))[...,None,None]*P
	return rho, kappa
