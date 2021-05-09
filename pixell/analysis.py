# This is a module for analysing sky maps: stuff like matched filtering, feature
# detection, pixel-space likelihoods etc.

import numpy as np
from . import enmap, utils, uharm

def matched_filter_constcov(map, B, iN, uht):
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
	pixarea = enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)
	rho     = uht.map2harm_adjoint(uht.hmul(B*iN,uht.map2harm(map)))/pixarea
	kappa   = uht.sum_hprof(B**2*iN)/(4*np.pi)
	# Equivalent to brute force unit vector bashing
	#v       = enmap.zeros(map.shape, map.wcs, map.dtype)
	#v[...,0,0] = 1
	#kappa = uht.map2harm_adjoint(uht.hmul(B**2*iN,uht.map2harm(v)))[0,0]/pixarea[0,0]**2
	return rho, kappa

def matched_filter_constcorr_lowcorr(map, B, ivar, iC, uht, B2=None, high_acc=False):
	"""Apply a matched filter to the given map, assuming a constant correlation
	noise model inv(N) = ivar**0.5 * iC * ivar**0.5, where ivar = 1/pixel_variance
	and iC = 1/harmonic_power(noise*ivar**0.5). This represents correlated noise
	described by iC that's modulated spatially by ivar.

	Arguments:
	map:  The [...,ny,nx] enmap to be filtered
	B:    The instrumental beam in the "hprof" format of the provided UHT
	ivar: The inverse of the white noise power per pixel, an [...,ny,nx] enmap
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
	pixarea = enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)
	V = ivar**0.5
	W = uht.quad_weights()
	# Square the beam in real space if not provided
	if B2 is None: B2 = uht.hprof_rpow(B, 2)
	# Find a white approximation for iC. A B²-weighted average is accurate to
	# about 1% for lknee = 4000, worsening to about 3% by lknee = 7000. Probably
	# good enough.
	iC_white = uht.sum_hprof(B**2*iC)/uht.sum_hprof(B**2)

	# FIXME: The formulas below have some ad-hoc pixarea factors in them. This makes them
	# give the right result, but is probably a symptom of using the wrong combination of
	# adjoints etc. in the rest of the formula. Get to the bottom of this so that they can
	# be removed.
	rho   = uht.harm2map(uht.hmul(B,uht.harm2map_adjoint(V*uht.map2harm_adjoint(uht.hmul(iC, uht.map2harm(V*map))))))/pixarea*pixarea
	kappa = uht.map2harm_adjoint(uht.hmul(B2,uht.harm2map_adjoint(ivar*W*iC_white[...,None,None])))/pixarea**2*pixarea

	if high_acc:
		# Optionally find a correction factor by evaluating the exact kappa in a single pixel
		pix  = tuple(np.array(map.shape[-2:])//2)
		u    = map*0; u[...,pix[0],pix[1]] = 1
		kappa_ii = (uht.harm2map(uht.hmul(B,uht.harm2map_adjoint(V*uht.map2harm_adjoint(uht.hmul(iC,uht.map2harm(V*uht.harm2map(uht.hmul(B,uht.map2harm(u/pixarea)))))))))/pixarea*pixarea)[...,pix[0],pix[1]]
		alpha  = kappa[...,pix[0],pix[1]]/kappa_ii
		kappa /= alpha[...,None,None]

	return rho, kappa

def matched_filter_constcorr_smoothivar(map, B, ivar, iC, uht):
	"""Apply a matched filter to the given map, assuming a constant correlation
	noise model inv(N) = ivar**0.5 * iC * ivar**0.5, where ivar = 1/pixel_variance
	and iC = 1/harmonic_power(noise*ivar**0.5). This represents correlated noise
	described by iC that's modulated spatially by ivar.

	Arguments:
	map:  The [...,ny,nx] enmap to be filtered
	B:    The instrumental beam in the "hprof" format of the provided UHT
	ivar: The inverse of the white noise power per pixel, an [...,ny,nx] enmap
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
	# We model N = VCV, where V is diagonal and diag(V)**2 = ivar
	# and C is diagonal in harmonic space. C can be measured by
	# whitening n: C = V"NV" = <V"nn'V"> = <cc'>, c = V"n.
	# Writing out the harmonic stuff, we have
	# N = <V"Y ch ch' Y' V"> = V"Y Ch Y' V"
	# => N" = VY'" Ch" Y" V
	pixarea = enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)
	# We assume that we can commute B past V, allowing us to compute kappa directly
	V     = ivar**0.5
	# rho = P'VY"' B Ch" Y" V map
	rho   = V*uht.map2harm_adjoint(uht.hmul(B*iC,uht.map2harm(V*map)))/pixarea
	# kappa = P'VY"'B Ch" B Y"VP = (P'V = R)(Y"'B sqrt(CH") = A)A'R' = RAA'R'
	# kappa_ii = R_ii² sum_l A_il
	kappa = ivar * (uht.sum_hprof(B**2*iC)/(4*np.pi))[...,None,None]
	return rho, kappa

#### Notes #####

# sum_j (YBY")_ji² ivar_jj
# = sum_j (Y_jl B_ll Y'_li W_i)² ivar_jj
#
# What is A_ji = Y_jl B_lm Y"_mi?
# Let's look at some map k = Br g. K = <kk'> = Br<gg'>Br' = Br G Br'
# What's this in harmonic space?
# k = Y kh, g = Y gh
# K = <kk'> = Y <kh kh'> Y' = Y KH Y' = Br<gg'>Br' = Br<Y gh gh' Y'>Br' = Br Y GH Y' Br'
# => KH = Y" Br Y GH Y' Br' Y'" = Bh GH Bh' with
# Bh = Y" Br Y <=> Br = Y Bh Y"
# So B transforms as one would expect.
#
# That's just the B matrix, though
# Bh_lm = Bl_l delta_lm
# Br_0i = Y_0l Bl_l Y"_li
# What is Y_0l? Y_0l = delta_0i Y = (Y' delta_i0)'
# Br_0i = Y"' (Bl_l Y_0l)
#
# How do we go back?
# Bl_l = Y"_li Br_0(j-i) Y_jl
#      = Y"_li Br_0k Y_(i+k)l
#      = Br_0k sum_i Y"_li Y_(i+k)l
#      = Br_0k 
#
# A_ji = Y_jl B_lm Y"_mi = (Y Bh Y")_ji = Br_ji
# kappa_ii = alpha*P_ii² * Br_ji² ivar_j
#      = alpha*(P² Br2' ivar)_i
#      = alpha*(P² (Y Br2h Y")' ivar)_i
#      = alpha*(P² Y"' Br2h Y' ivar)_i
