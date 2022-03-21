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

import numpy as np, time
from scipy import ndimage
from . import enmap, utils, uharm, wavelets, bunch

def matched_filter_constcov(map, B, iN, uht=None, spin=0):
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
	rho     = uht.map2harm_adjoint(uht.hmul(B*iN,uht.map2harm(map, spin=spin)),spin=spin)/pixarea
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



# These functions and classes represent a modular approach to
# object detection in maps. At the lowest level are Nmat objects,
# which provide matched filtering through a unified interface.
# Then there are Finder objects that are typically built using an
# Nmat object and returns a catalog given a map.
# There are also Measurer objects that are similar to Finders,
# but take a catalog of known positions as input.
# Finally, there are Modellers that go from a catalog to a map.
#
# The nice thing with this approach is that it's easy to compose
# these together to make more advanced things, like iterative
# finders etc.
#
# A limitation of the implementation below is that they're
# flat-sky only, unlike the UHT-based functions above. For me,
# that hasn't turned out to be a limitation, since I have to do
# things in tiles for other reasons anyway (e.g. memory).


# Overall abstract interfaces

class Nmat:
	def matched_filter(self, map, cache=None): raise NotImplementedError
	def simulate(self): raise NotImplementedError

class Finder:
	def __call__(self, map): raise NotImplementedError

class Measurer:
	def __call__(self, map, cat): raise NotImplementedError

class Modeller:
	def __call__(self, cat): raise NotImplementedError
	def amplitudes(self, cat): raise NotImplementedError

# Nmat implementations:

class NmatConstcov(Nmat):
	def __init__(self, iN, apod):
		"""Initialize a Constcov noise model from an inverse noise power spectrum
		enmap. iN must have shape [n,n,ny,nx]. For a simple scalar filter just
		insert scalar dimensions with None before constructing, e.g.
		iN[None,None]."""
		self.iN      = iN
		self.apod    = apod
		assert self.iN.ndim == 4, "iN   must be an enmap with 4 dims"
		self.pixsize  = enmap.pixsize(self.iN.shape, self.iN.wcs)
		self.pixratio = enmap.pixsizemap(self.iN.shape, self.iN.wcs, broadcastable=True)/self.pixsize
		self.fsky    = enmap.area(self.iN.shape, self.iN.wcs)/(4*np.pi)
	def matched_filter(self, map, beam, cache=None):
		"""Apply a matched filter to the given map [n,ny,nx], which must agree in shape
		with the beam transform beam [n,ny,nx]. Returns rho[n,ny,nx], kappa[n,n,ny,nx].
		From these the best-fit fluxes in pixel y,x can be recovered as
		np.linalg.solve(kappa[:,:,y,x],rho[:,y,x]), and the combined flux as
		rho_tot = np.sum(rho,0); kappa_tot = np.sum(kappa,(0,1)); flux_tot = rho_tot[y,x]/kappa_tot[y,x]"""
		assert map .ndim  == 3, "Map must be an enmap with 3 dims"
		assert beam.ndim  == 3, "Beam must be an enmap with 3 dims"
		assert map .shape == beam.shape, "Map and beam shape must agree"
		# Flat sky corrections. Don't work that well here. Without them
		# we have an up to 2% error with a 10° tall patch. With them this
		# is reduced to 1%. I don't understand why they don't work better here.
		flatcorr_rho, flatcorr_kappa = get_flat_sky_correction(self.pixratio)
		rho = utils.cache_get(cache, "rho_pre", lambda: enmap.map_mul(self.iN,enmap.map2harm(map*self.apod, spin=0, normalize="phys"))/self.pixsize)
		rho = enmap.map2harm_adjoint(beam*rho, spin=0, normalize="phys")*flatcorr_rho
		kappa0 = np.sum(beam[:,None]*self.iN[:,:]*beam[None,:],(-2,-1))/(4*np.pi*self.fsky)
		kappa  = np.empty_like(rho, shape=kappa0.shape+rho.shape[-2:])
		kappa[:] = kappa0[:,:,None,None]*flatcorr_kappa
		# Done! What we return will always be [n,ny,nx], [n,n,ny,nx]
		return rho, kappa
	def simulate(self):
		hN = safe_pow(self.iN, -0.5)
		r  = enmap.rand_gauss_harm(self.iN.shape[1:], self.iN.wcs).astype(utils.complex_dtype(self.iN.dtype))
		sim= enmap.ifft(enmap.map_mul(hN, r), normalize="phys").real
		return sim

class NmatConstcorr(Nmat):
	def __init__(self, iN, ivar):
		"""Initialize a Constcov noise model from an inverse noise power spectrum
		enmap. iN must have shape [n,n,ny,nx]. For a simple scalar filter just
		insert scalar dimensions with None before constructing, e.g.
		iN[None,None]."""
		self.iN      = iN
		self.ivar    = ivar
		assert self.iN  .ndim == 4, "iN   must be an enmap with 4 dims"
		assert self.ivar.ndim == 3, "ivar must be an enmap with 3 dims"
		self.pixsize  = enmap.pixsize(self.iN.shape, self.iN.wcs)
		self.pixratio = enmap.pixsizemap(self.iN.shape, self.iN.wcs, broadcastable=True)/self.pixsize
	def matched_filter(self, map, beam, beam2=None, cache=None):
		"""Apply a matched filter to the given map [n,ny,nx], which must agree in shape
		with the beam transform beam [n,ny,nx]. Returns rho[n,ny,nx], kappa[n,n,ny,nx].
		From these the best-fit fluxes in pixel y,x can be recovered as
		np.linalg.solve(kappa[:,:,y,x],rho[:,y,x]), and the combined flux as
		rho_tot = np.sum(rho,0); kappa_tot = np.sum(kappa,(0,1)); flux_tot = rho_tot[y,x]/kappa_tot[y,x]"""
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		assert beam.ndim == 3, "Beam must be an enmap with 3 dims"
		assert map.shape == beam.shape, "Map and beam shape must agree"
		V    = self.ivar**0.5
		# Square the beam in real space if not provided
		if beam2 is None: beam2 = rpow(beam, 2)
		# Find a white approximation for iN. Is doing this element-wise correct?
		iN_white = np.sum(beam[:,None]*self.iN*beam[None,:],(-2,-1))/np.sum(beam[:,None]*beam[None,:],(-2,-1))
		# Flat-sky correction factors. For a 10° tall patch we get up to 2% errors in flux without
		# them. This is reduced to 1% with them. Not sure why it doesn't do better - is the
		# gaussian approximation too limiting? Or do I have a bug?
		flatcorr_rho, flatcorr_kappa = get_flat_sky_correction(self.pixratio)
		# Numerator
		rho   = utils.cache_get(cache, "rho_pre", lambda: enmap.map2harm(flatcorr_rho*V*enmap.harm2map(enmap.map_mul(self.iN, enmap.map2harm(V*map, spin=0, normalize="phys")), spin=0, normalize="phys"), spin=0, normalize="phys")/self.pixsize)
		rho   = enmap.harm2map(beam*rho, spin=0, normalize="phys")
		# Denominator
		kappa = enmap.harm2map(enmap.map_mul(beam2,enmap.map2harm(self.ivar+0j, spin=0, normalize="phys")), spin=0, normalize="phys")/self.pixsize * flatcorr_kappa
		kappa = np.maximum(kappa,0)**0.5
		kappa = kappa[:,None]*iN_white[:,:,None,None]*kappa[None,:]
		# Done! What we return will always be [n,ny,nx], [n,n,ny,nx]
		return rho, kappa
	def simulate(self):
		hN = safe_pow(self.iN, -0.5)
		r  = enmap.rand_gauss_harm(self.iN.shape[1:], self.iN.wcs).astype(utils.complex_dtype(self.iN.dtype))
		sim= enmap.ifft(enmap.map_mul(hN, r)).real
		mask = self.ivar != 0
		sim[mask] *= self.ivar[mask]**-0.5
		return sim

class NmatWavelet(Nmat):
	def __init__(self, wt, wiN):
		"""Initialize a Wavelet noise model from a WaveletTransform object and
		a wavelet map object representing the inverse noise variance map pwer wavelet scale."""
		self.wt   = wt
		self.wiN  = wiN
	def matched_filter(self, map, beam, cache=None):
		# We get 2% flat-sky errors with a 10° tall patch without corrections, and 1% with the corrections.
		pixsize  = enmap.pixsize(map.shape, map.wcs)
		pixratio = enmap.pixsizemap(map.shape, map.wcs, broadcastable=True)/pixsize
		flatcorr_rho, flatcorr_kappa = get_flat_sky_correction(pixratio)
		# Get rho
		rho = utils.cache_get(cache, "rho_pre", lambda: enmap.map2harm(self.wt.wave2map(multimap.map_mul(self.wiN, self.wt.map2wave(map))), spin=0, normalize="phys")/pixsize)
		rho = enmap.harm2map(beam*rho, spin=0, normalize="phys")*flatcorr_rho
		# Then get kappa
		fkappa = enmap.zeros(self.wiN.pre + map.shape[-2:], map.wcs, utils.complex_dtype(map.dtype))
		for i in range(self.wt.nlevel):
			sub_Q  = self.wt.filters[i]*enmap.resample_fft(beam, self.wt.geometries[i][0], norm=None, corner=True)
			# Is it right to do this component-wise?
			sub_Q2 = rop(sub_Q, op=lambda a: a[:,None]*a[None,:])
			fsmall = sub_Q2*enmap.fft(self.wiN.maps[i], normalize=False)/self.wiN.npixs[i]
			enmap.resample_fft(fsmall, map.shape, fomap=fkappa, norm=None, corner=True, op=np.add)
		kappa = enmap.ifft(fkappa, normalize=False).real/pixsize*flatcorr_kappa
		return rho, kappa


# Finder implementations

class FinderSimple(Finder):
	def __init__(self, nmat, beam, scaling=1, save_snr=False):
		"""Initialize a simple object finder. It looks for a single class
		of object in maps with shape [nfreq,ny,nx]. It looks for objects
		with shape given by the 2d fourier profile "beam". For point
		soruces this will be an actual beam transform. For extended objects
		it will be the product of the beam transform and their shape's fourier coefficients.

		The argument "scaling" specifies how the signal is expected to change
		between frequencies, and should be either a scalar or an array of shape [nfreq].

		"nmat" should be an Nmat subclass that describes the noise properties of
		the maps to be analysed.

		This class does not do anything fancy like iterative finding etc.
		This means that it won't handle e.g. weak objects close to strong objects
		well. It also looks for just a single type of object. Use FinderIterative
		or FinderMulti (or both) to get around this."""
		self.beam   = beam
		self.nmat   = nmat
		self.scaling= scaling
		self.order  = 3
		self.grow   = 0.75*utils.arcmin
		self.grow0  = 20*utils.arcmin
		self.save_snr = save_snr
		self.snr    = None
	def __call__(self, map, snmin=5, snrel=None, penalty=None):
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		ncomp = len(map)
		dtype = [("ra","d"),("dec","d"),("snr","d"),("flux_tot","d"),("dflux_tot","d"),
				("flux","d",(ncomp,)),("dflux","d",(ncomp,))]
		if penalty is None: penalty = 1
		# Apply the matched filter
		rho, kappa = self.nmat.matched_filter(map, self.beam)
		kappa     = sanitize_kappa(kappa)
		scaling   = np.zeros(len(rho),rho.dtype)+self.scaling
		rho_tot   = np.sum(rho*scaling[:,None,None],0)
		# Build the total detection significance and find peaks
		kappa_tot = np.sum(scaling[:,None,None,None]*kappa*scaling[None,:,None,None],(0,1))
		snr_tot   = rho_tot/kappa_tot**0.5
		# Find the effective S/N threshold, taking into account any position-dependent
		# penalty and the (penalized) maximum value in the map
		if snrel   is not None: snmin = max(snmin, np.max(snr_tot/penalty)*snrel)
		snlim = snmin*penalty
		# Detect objects, and grow them a bit
		labels, nlabel = ndimage.label(snr_tot >= snlim)
		cat            = np.zeros(nlabel, dtype).view(np.recarray)
		if nlabel == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		labels = enmap.samewcs(labels, map)
		dists, labels = labels.labeled_distance_transform(rmax=self.grow0)
		labels *= dists <= self.grow
		allofthem = np.arange(1,nlabel+1)
		if len(cat) == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		# Find the position and snr of each object
		pixs    = np.array(ndimage.center_of_mass(snr_tot**2, labels, allofthem)).T
		cat.ra, cat.dec = map.pix2sky(pixs)[::-1]
		cat.snr = ndimage.maximum(snr_tot, labels, allofthem)
		del labels
		# Interpolating before solving is faster, but inaccurate. So we do the slow thing.
		# First get the total flux and its uncertainty.
		flux_tot, dflux_tot = solve_mapsys(kappa_tot, rho_tot)
		del rho_tot, kappa_tot
		cat.flux_tot        = flux_tot .at(pixs, unit="pix", order=self.order)
		cat.dflux_tot       = dflux_tot.at(pixs, unit="pix", order=0)
		del flux_tot, dflux_tot
		# Then get the per-freq versions
		flux, dflux = solve_mapsys(kappa, rho)
		del rho, kappa
		cat.flux    = flux .at(pixs, unit="pix", order=self.order).T
		cat.dflux   = dflux.at(pixs, unit="pix", order=0).T
		del flux, dflux
		# Hack
		if self.save_snr and self.snr is None: self.snr = snr_tot
		# Sort by SNR and return
		cat = cat[np.argsort(cat.snr)[::-1]]
		return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)

moo = 0
class FinderMulti(Finder):
	def __init__(self, nmat, beams, scalings, save_snr=False):
		"""Initialize FinderMulti. It's similar to FinderSimple, except
		that it looks for multiple different types of object at the same
		time. "beams" is a list of 2d fourier representations of shapes
		to look for, while scalings is a list frequency scalings with
		the same length."""
		self.nmat     = nmat
		self.beams    = beams
		if scalings is None:
			scalings = np.ones(len(beams))
		self.scalings = scalings
		self.order    = 3
		self.r        = 2*utils.arcmin
		self.save_snr = save_snr
		self.snr      = None
	def __call__(self, map, snmin=5, snrel=None, penalty=None):
		global moo; moo += 1
		#enmap.write_map("map_%02d.fits" % moo, map)
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		ncase = len(self.beams)
		ncomp = len(map)
		dtype = [("ra","d"),("dec","d"),("snr","d"),("flux_tot","d"),("dflux_tot","d"),
				("flux","d",(ncomp,)),("dflux","d",(ncomp,)),("case","i")]
		if penalty is None: penalty = 1
		# Apply the matched filter for each profile and keep track of the best beam
		# in each pixel.
		cache = {}
		snr_tot, rho, kappa, rho_tot, kappa_tot, case = None, None, None, None, None, None
		for ca, (beam, scaling) in enumerate(zip(self.beams, self.scalings)):
			def f():
				rho, kappa = self.nmat.matched_filter(map, beam, cache=cache)
				kappa      = sanitize_kappa(kappa)
				return rho, kappa
			my_rho, my_kappa = utils.cache_get(cache, "beam:"+str(id(beam)), f)
			my_rho_tot    = np.sum(my_rho*scaling[:,None,None],0)
			my_kappa_tot  = np.sum(scaling[:,None,None,None]*my_kappa*scaling[None,:,None,None],(0,1))
			my_snr_tot    = my_rho_tot/my_kappa_tot**0.5
			#enmap.write_map("snr_%02d_case_%02d.fits" % (moo, ca), my_snr_tot)
			if snr_tot is None:
				cases = enmap.full(my_snr_tot.shape, my_snr_tot.wcs, ca, np.int8)
				snr_tot, rho, kappa, rho_tot, kappa_tot= my_snr_tot, my_rho, my_kappa, my_rho_tot, my_kappa_tot
			else:
				mask = my_snr_tot > snr_tot
				cases      = enmap.samewcs(np.where(mask, ca,           cases),      map)
				snr_tot    = enmap.samewcs(np.where(mask, my_snr_tot,   snr_tot),   map)
				rho        = enmap.samewcs(np.where(mask, my_rho,       rho),       map)
				kappa      = enmap.samewcs(np.where(mask, my_kappa,     kappa),     map)
				rho_tot    = enmap.samewcs(np.where(mask, my_rho_tot,   rho_tot),   map)
				kappa_tot  = enmap.samewcs(np.where(mask, my_kappa_tot, kappa_tot), map)
			del my_rho, my_kappa, my_rho_tot, my_kappa_tot, my_snr_tot
		del cache
		#enmap.write_map("snr_%02d.fits"  % (moo), snr_tot)
		#enmap.write_map("case_%02d.fits" % (moo), cases)
		# Hack
		if self.save_snr and self.snr is None: self.snr = snr_tot
		# Find the effective S/N threshold, taking into account any position-dependent
		# penalty and the (penalized) maximum value in the map
		if snrel   is not None: snmin = max(snmin, np.max(snr_tot/penalty)*snrel)
		snlim = snmin*penalty
		labels, nlabel = ndimage.label(snr_tot >= snlim)
		allofthem      = np.arange(1,nlabel+1)
		cat            = np.zeros(nlabel, dtype).view(np.recarray)
		if nlabel == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		pixs0   = np.array(ndimage.maximum_position(snr_tot, labels, allofthem)).T
		# Make the labels circles with a constant radius. This ensures that the
		# center-of-mass is calculated over a consistent area.
		labels = enmap.samewcs(labels, map)
		labels = make_circle_labels(map.shape, map.wcs, pixs0, r=self.r)
		#enmap.write_map("labels_%02d.fits" % (moo), labels)
		if len(cat) == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		# Find the position and snr of each object. This is a bit questionable since
		# only the central pixel is guaranteed to belong to relevant case. We therefore
		# check that the center-of-mass location has the same case, and if it doesn't,
		# we fall back on just using the pixel center. This is an acceptable solution,
		# but not a good one. It means that fluxes and positions will be inaccurate for
		# the cases where we do end up taking the fallback. Hopefully those should be
		# pretty weak cases anyway. I think a better solution would require the individual
		# case information, e.g. via a two-pass approach or by saving per-case flux, dflux,
		# snr etc. maps.
		pixs    = np.array(ndimage.center_of_mass(snr_tot**2, labels, allofthem)).T
		cat.snr = ndimage.maximum(snr_tot, labels, allofthem)
		# Interpolating before solving is faster, but inaccurate. So we do the slow thing.
		flux_tot, dflux_tot = solve_mapsys(kappa_tot, rho_tot)
		case0    = cases.at(pixs0, unit="pix", order=0)
		case_com = cases.at(pixs,  unit="pix", order=0)
		flux0    = flux_tot.at(pixs0, unit="pix", order=0)
		flux_com = flux_tot.at(pixs,  unit="pix", order=self.order)
		unsafe   = (case_com != case0) | (np.abs((flux_com-flux0))/np.maximum(np.abs(flux_com),np.abs(flux0)) > 0.2)
		# Build the total part of the catalog
		cat.ra, cat.dec = map.pix2sky(np.where(unsafe, pixs0, pixs))[::-1]
		cat.case      = np.where(unsafe, case0, case_com)
		cat.flux_tot  = np.where(unsafe, flux0, flux_com)
		cat.dflux_tot = dflux_tot.at(np.where(unsafe, pixs0, pixs), unit="pix", order=0)
		del flux_tot, dflux_tot
		# Then get the per-freq versions
		flux, dflux = solve_mapsys(kappa, rho)
		del rho, kappa
		cat.flux    = np.where(unsafe, flux.at(pixs0, unit="pix", order=0), flux.at(pixs, unit="pix", order=self.order)).T
		cat.dflux   = dflux.at(np.where(unsafe, pixs0, pixs), unit="pix", order=0).T
		del flux, dflux
		# Sort by SNR and return
		cat = cat[np.argsort(cat.snr)[::-1]]
		return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)

moo = 0
class FinderMultiSafe(Finder):
	def __init__(self, nmat, beams, scalings, save_snr=False):
		"""Initialize FinderMultiSafe. It's similar to FinderSimple, except
		that it looks for multiple different types of object at the same
		time. "beams" is a list of 2d fourier representations of shapes
		to look for, while scalings is a list frequency scalings with
		the same length."""
		self.nmat     = nmat
		self.beams    = beams
		if scalings is None:
			scalings = np.ones(len(beams))
		self.scalings = scalings
		self.order    = 3
		self.rs       = np.array([get_central_radius(beam, lknee=2000) for beam in self.beams])
		print(self.rs/utils.arcmin)
		self.save_snr = save_snr
		self.snr      = None
	def __call__(self, map, snmin=5, snrel=None, penalty=None):
		global moo; moo += 1
		#enmap.write_map("map_%02d.fits" % moo, map)
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		ncase = len(self.beams)
		ncomp = len(map)
		dtype = [("ra","d"),("dec","d"),("snr","d"),("flux_tot","d"),("dflux_tot","d"),
				("flux","d",(ncomp,)),("dflux","d",(ncomp,)),("case","i")]
		if penalty is None: penalty = 1
		# Apply the matched filter for each profile and keep track of the best beam
		# in each pixel.
		cache = {}
		snr_tot, cases = None, None
		snrs, fluxs_tot, dfluxs_tot, fluxs, dfluxs = [], [], [], [], []
		for ca, (beam, scaling) in enumerate(zip(self.beams, self.scalings)):
			def f():
				rho, kappa = self.nmat.matched_filter(map, beam, cache=cache)
				kappa      = sanitize_kappa(kappa)
				return rho, kappa
			my_rho, my_kappa = utils.cache_get(cache, "beam:"+str(id(beam)), f)
			my_rho_tot    = np.sum(my_rho*scaling[:,None,None],0)
			my_kappa_tot  = np.sum(scaling[:,None,None,None]*my_kappa*scaling[None,:,None,None],(0,1))
			my_snr        = my_rho_tot/my_kappa_tot**0.5
			my_flux,     my_dflux     = solve_mapsys(my_kappa, my_rho)
			my_flux_tot, my_dflux_tot = solve_mapsys(my_kappa_tot, my_rho_tot)
			#enmap.write_map("snr_%02d_case_%02d.fits" % (moo, ca), my_snr)
			if snr_tot is None:
				cases = enmap.full(my_snr.shape, my_snr.wcs, ca, np.int8)
				snr_tot = my_snr
			else:
				mask       = my_snr > snr_tot
				cases      = enmap.samewcs(np.where(mask, ca,     cases),   map)
				snr_tot    = enmap.samewcs(np.where(mask, my_snr, snr_tot), map)
			fluxs_tot .append(my_flux_tot)
			dfluxs_tot.append(my_dflux_tot)
			fluxs .append(my_flux)
			dfluxs.append(my_dflux)
			snrs.append(my_snr)
			del my_rho, my_kappa, my_flux, my_dflux, my_flux_tot, my_dflux_tot, my_snr
		del cache
		#enmap.write_map("snr_%02d.fits"  % (moo), snr_tot)
		#enmap.write_map("case_%02d.fits" % (moo), cases)
		# Hack
		if self.save_snr and self.snr is None: self.snr = snr_tot
		# Find the effective S/N threshold, taking into account any position-dependent
		# penalty and the (penalized) maximum value in the map
		if snrel   is not None: snmin = max(snmin, np.max(snr_tot/penalty)*snrel)
		snlim = snmin*penalty
		labels, nlabel = ndimage.label(snr_tot >= snlim)
		labels         = enmap.samewcs(labels, map)
		allofthem      = np.arange(1,nlabel+1)
		cat            = np.zeros(nlabel, dtype).view(np.recarray)
		if nlabel == 0: return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)
		# Find the peak pixel for each label, and use them to reshape each label into
		# a constant-radius circle. TODO: Consider making this radius case-dependent,
		# with a radius given by the filter kernel width.
		pixs0   = np.array(ndimage.maximum_position(snr_tot, labels, allofthem)).T
		#enmap.write_map("labels_%02d.fits" % (moo), labels)
		# Find which case that corresponds to
		cat.case= cases[pixs0[0],pixs0[1]]
		cat.snr = snr_tot[pixs0[0],pixs0[1]]
		# We will now loop over cases and process the objects for each case
		# separately. This avoids the issue of averaging over pixels that correspond
		# to multiple different cases that the plain FinderMulti has.
		for ca in range(ncase):
			someofthem = allofthem[cat.case==ca]
			inds       = someofthem-1
			if len(inds) == 0: continue
			my_labels  = make_circle_labels(map.shape, map.wcs, pixs0[:,inds], inds=someofthem, r=self.rs[ca])
			pixs = np.array(ndimage.center_of_mass(snrs[ca]**2, my_labels, someofthem)).T
			cat.ra[inds], cat.dec[inds] = map.pix2sky(pixs)[::-1]
			cat. flux_tot[inds] =  fluxs_tot[ca].at(pixs, unit="pix", order=self.order)
			cat.dflux_tot[inds] = dfluxs_tot[ca].at(pixs, unit="pix", order=0)
			cat. flux[inds]     =  fluxs[ca].at(pixs, unit="pix", order=self.order).T
			cat.dflux[inds]     = dfluxs[ca].at(pixs, unit="pix", order=0).T
		# Sort by SNR and return
		cat = cat[np.argsort(cat.snr)[::-1]]
		return bunch.Bunch(cat=cat, snmin=snmin, snr=snr_tot, snlim=snlim)

class FinderIterative(Finder):
	def __init__(self, finder, modeller, maxiter=20, sntol=0.50,
			grid_max=5, grid_res=0.1*utils.degree, grid_dominance=4):
		"""Initialize an iterative object finder. This handles weaker
		objects near stronger ones by iteratively finding the strongest
		objects, subtracting them, finding weaker objects, and so on.

		Arguments:
		* finder: The Finder object to use in each step of the iterative
		  analysis, e.g. a FinderSimple or FinderMulti instance.
		* modeller: The Modeller object to use in the subtraction step.
		  Must be consistent with the catalogue returned by the finder.
		  For example, a FinderMulti must be used with a ModellerMulti.
		
		The other arguments are optional:
		* maxiter: The maximum number of iterations to run.
		* sntol:   The factor by which the detection limit decreases by
		  in each iteration.
		* grid_max: The maximum number of objects to allow in a small area.
		* grid_res: The side-length of that small area.
		* grid_dominance: Keep objects in an area that exceeded grid_max
		  if they are contaminated by less than 1/grid_dominance. Put
		  another way, the objects to keep should be at least grid_dominance
		  times higher than their contamination by other objects."""
		self.finder   = finder
		self.modeller = modeller
		self.maxiter  = maxiter
		self.sntol    = sntol
		self.grid_max = grid_max
		self.grid_res = grid_res
		self.grid_dominance = grid_dominance
	def __call__(self, map, snmin=5, snrel=None, verbose=False):
		cat      = []
		own_vals = []
		model = map*0
		# Build a grid that's used to penalize areas with too many detections
		box  = utils.widen_box(map.corners(),2*self.grid_res,relative=False)
		cgeo = enmap.geometry(box, res=self.grid_res)
		hits = enmap.zeros(*cgeo, np.int32)
		snmaxs = enmap.zeros(*cgeo, map.dtype)
		snbad  = enmap.zeros(*cgeo, bool)
		for i in range(self.maxiter):
			t1    = time.time()
			sntol = self.sntol
			bad_mask  = hits > self.grid_max
			bad_mask |= snbad
			if np.any(bad_mask):
				bad_mask[:] = ndimage.distance_transform_edt(1-bad_mask)<=1.5
			bad_mask  = bad_mask.project(*map.geometry, order=0)
			res = self.finder(map-model, snmin=snmin, snrel=sntol, penalty=1+bad_mask*1e6)
			t2  = time.time()
			if verbose:
				print("it %d snmin %5.1f nsrc %6d tot %6d time %6.2f" % (i+1, res.snmin, len(res.cat), len(res.cat)+sum([len(c) for c in cat]), t2-t1))
			if len(res.cat) == 0: break
			# Update our grid
			pix   = hits.sky2pix([res.cat.dec,res.cat.ra])
			ipix  = utils.nint(pix)
			hits += utils.bin_multi(ipix, hits.shape)
			# Measure the total snr in each cell. If this is bigger in one
			# step than it was in the previous step, then we have a model mismatch
			# blow-up. This is rare and I haven't gotten to the bottom of it, but
			# it's best to just prevent like this. Allow some tolerance to allow for
			# sources with similar amplitude very close to each other, where the negative
			# ring around one source would make the other one seem weaker at first.
			blowup_tol = 1.5
			snr_bin = utils.bin_multi(utils.nint(pix), hits.shape, res.cat.snr)
			snbad  |= (snr_bin > snmaxs*blowup_tol) & (snmaxs > 0)
			snmaxs  = np.maximum(snmaxs, snr_bin)
			# Disqualify these entries
			bad     = snbad[ipix[0],ipix[1]]
			res.cat = res.cat[~bad]
			if len(res.cat) == 0: continue
			cat.append(res.cat)
			# update total model
			model += self.modeller(res.cat)
			if res.snmin <= snmin: break
		# The final result will have an extra contamination column
		if len(cat) == 0: cat = res.cat
		else:             cat = np.concatenate(cat, 0).view(np.recarray)
		contam = np.zeros(len(cat), [("contam", "d", (len(map),))]).view(np.recarray)
		if len(cat) > 0:
			# Measure the contamination. This is defined as others' contribution
			# to our peak location relative to our own.
			own_vals = self.modeller.amplitudes(cat)
			tot_vals = model.at([cat.dec,cat.ra]).T
			contam.contam = (tot_vals-own_vals)/np.abs(own_vals)
		cat = merge_arrays([cat, contam]).view(np.recarray)
		# Disqualify sources in overly contaminated regions, unless they dominate alone.
		# Maybe this could be moved into a separate class, but single-pass finders will
		# never get very dense detectors, and the grid is needed here anyway to avoid
		# the finding being too slow.
		if self.grid_dominance and len(cat) > 0:
			# We're bad if the cell as too many hits...
			pix   = utils.nint(hits.sky2pix([cat.dec,cat.ra]))
			nhit  = hits.at(pix, unit="pix", order=0)
			bad   = nhit > self.grid_max
			# ...unless we dominate
			freq_snr    = np.abs(cat.flux/cat.dflux)
			freq_best   = np.argmax(freq_snr,1)
			best_contam = cat.contam[np.arange(len(cat)), freq_best]
			old_bad = bad.copy()
			bad  &= best_contam > 1/self.grid_dominance
			cat, cat_bad = cat[~bad], cat[bad]
			# Update model to reflect the cut sources
			if len(cat_bad) > 0:
				model -= self.modeller(cat_bad)
			print("cut %d srcs, %d remaining" % (len(cat_bad), len(cat)))
		return bunch.Bunch(cat=cat, snmin=snmin, model=model, hits=hits, bad_mask=bad_mask)

# Measurer implementations

class MeasurerSimple(Measurer):
	def __init__(self, nmat, beam, scaling=1):
		self.beam   = beam
		self.nmat   = nmat
		self.scaling= scaling
		self.order  = 3
	def __call__(self, map, icat):
		assert map.ndim == 3, "Map must be an enmap with 3 dims"
		ncomp = len(map)
		cat   = icat.copy()
		pixs  = map.sky2pix([icat.dec,icat.ra])
		# Apply the matched filter
		rho, kappa = self.nmat.matched_filter(map, self.beam)
		kappa     = sanitize_kappa(kappa)
		# Read off the total values at the given positions
		scaling   = np.zeros(len(rho),rho.dtype)+self.scaling
		rho_tot   = np.sum(rho*scaling[:,None,None],0)
		kappa_tot = np.sum(scaling[:,None,None,None]*kappa*scaling[None,:,None,None],(0,1))
		snr_tot   = rho_tot/kappa_tot**0.5
		flux_tot, dflux_tot = solve_mapsys(kappa_tot, rho_tot)
		del rho_tot, kappa_tot
		cat.snr             = snr_tot  .at(pixs, unit="pix", order=0)
		cat.flux_tot        = flux_tot .at(pixs, unit="pix", order=self.order)
		cat.dflux_tot       = dflux_tot.at(pixs, unit="pix", order=0)
		del snr_tot, flux_tot, dflux_tot
		# Read off the individual values
		flux, dflux = solve_mapsys(kappa, rho)
		del rho, kappa
		cat.flux    = flux .at(pixs, unit="pix", order=self.order).T
		cat.dflux   = dflux.at(pixs, unit="pix", order=0).T
		return bunch.Bunch(cat=cat)

class MeasurerMulti(Measurer):
	def __init__(self, measurers):
		self.measurers = measurers
	def __call__(self, map, icat):
		cat = icat.copy()
		if len(icat) == 0: return bunch.Bunch(cat=cat)
		uvals, order, edges = utils.find_equal_groups_fast(icat.case)
		for i, ca in enumerate(uvals):
			subicat = icat[order[edges[i]:edges[i+1]]]
			if len(subicat) == 0: continue
			subocat = self.measurers[i](map, subicat).cat
			cat[order[edges[i]:edges[i+1]]] = subocat
		return bunch.Bunch(cat=cat)

class MeasurerIterative(Measurer):
	def __init__(self, measurer, modeller, sntol=0.25, snscale=1):
		self.measurer = measurer
		self.modeller = modeller
		self.sntol    = sntol
		self.snscale  = snscale
		self.snmin    = 0.1 # do everything at once below this
	def __call__(self, map, icat, verbose=False):
		cat    = icat.copy()
		if cat.size == 0: return bunch.Bunch(cat=cat, model=self.modeller(cat))
		snr    = icat.snr * self.snscale
		groups = snr_split(snr, sntol=self.sntol, snmin=self.snmin)
		model  = np.zeros_like(map)
		for gi, group in enumerate(groups):
			if verbose: print("Measuring group %d with snmin %6.2f" % (gi+1, np.min(snr[group])))
			subcat = self.measurer(map-model, icat[group]).cat
			model += self.modeller(subcat)
			cat[group] = subcat
		return bunch.Bunch(cat=cat, model=model)


# Modeller implementations

class ModellerPerfreq(Modeller):
	def __init__(self, shape, wcs, beam_profiles, dtype=np.float32, nsigma=5):
		self.shape = shape
		self.wcs   = wcs
		self.dtype = dtype
		self.nsigma= nsigma
		self.beam_profiles = []
		for i, (r,b) in enumerate(beam_profiles):
			self.beam_profiles.append(np.array([r,b/np.max(b)]))
		self.areas = np.array([utils.calc_beam_area(prof) for prof in self.beam_profiles])
	def __call__(self, cat):
		from . import pointsrcs
		ncomp = len(self.beam_profiles)
		omap  = enmap.zeros((ncomp,)+self.shape[-2:], self.wcs, self.dtype)
		if len(cat) == 0: return omap
		for i in range(ncomp):
			# This just subtracts the raw measurement at each frequencies. Some frequencies may have
			# bad S/N. Maybe consider adding the scaled tot flux as a weak prior?
			srcparam = np.concatenate([cat.dec[:,None],cat.ra[:,None],cat.flux[:,i:i+1]/self.areas[i]],-1)
			pointsrcs.sim_srcs(self.shape[-2:], self.wcs, srcparam, self.beam_profiles[i], omap=omap[i], nsigma=self.nsigma)
		return omap
	def amplitudes(self, cat):
		bpeaks = np.array([prof[1,0] for prof in self.beam_profiles])
		return cat.flux * (bpeaks/self.areas)

class ModellerScaled(Modeller):
	def __init__(self, shape, wcs, beam_profiles, scaling, dtype=np.float32, nsigma=5):
		self.shape = shape
		self.wcs   = wcs
		self.dtype = dtype
		self.nsigma= nsigma
		self.scaling = scaling
		self.beam_profiles = []
		for i, (r,b) in enumerate(beam_profiles):
			self.beam_profiles.append(np.array([r,b/np.max(b)]))
		self.areas = np.array([utils.calc_beam_area(prof) for prof in self.beam_profiles])
	def __call__(self, cat):
		from . import pointsrcs
		ncomp = len(self.beam_profiles)
		omap  = enmap.zeros((ncomp,)+self.shape[-2:], self.wcs, self.dtype)
		if len(cat) == 0: return omap
		for i in range(ncomp):
			srcparam = np.concatenate([cat.dec[:,None],cat.ra[:,None],cat.flux_tot[:,None]*self.scaling[i]/self.areas[i]],-1)
			pointsrcs.sim_srcs(self.shape[-2:], self.wcs, srcparam, self.beam_profiles[i], omap=omap[i], nsigma=self.nsigma)
		return omap
	def amplitudes(self, cat):
		bpeaks = np.array([prof[1,0] for prof in self.beam_profiles])
		return cat.flux_tot[:,None]*(self.scaling*bpeaks/self.areas)

class ModellerMulti(Modeller):
	def __init__(self, modellers):
		self.modellers = modellers
	def __call__(self, cat):
		# If the cat is empty, just pass it on to the first modeller to have
		# it generate an empty map
		if len(cat) == 0: return self.modellers[0](cat)
		# Loop through the catalog entries of each type and have the corresponding
		# modeller build a model for them
		uvals, order, edges = utils.find_equal_groups_fast(cat.case)
		omap = None
		for i, ca in enumerate(uvals):
			subcat = cat[order[edges[i]:edges[i+1]]]
			if len(subcat) == 0: continue
			map    = self.modellers[ca](subcat)
			if omap is None: omap  = map
			else:            omap += map
		return omap
	def amplitudes(self, cat):
		res = np.zeros(cat.flux.shape)
		if len(cat) == 0: return res
		uvals, order, edges = utils.find_equal_groups_fast(cat.case)
		for i, ca in enumerate(uvals):
			subcat = cat[order[edges[i]:edges[i+1]]]
			res[order[edges[i]:edges[i+1]]] = self.modellers[ca].amplitudes(subcat)
		return res

# Helper functions .Should maybe be moved to utils

def sanitize_kappa(kappa, tol=1e-4, inplace=False):
	if not inplace: kappa = kappa.copy()
	for i in range(len(kappa)):
		kappa[i,i] = np.maximum(kappa[i,i], np.max(kappa[i,i])*tol)
	return kappa

def solve_mapsys(kappa, rho):
	if kappa.ndim == 2:
		flux  = rho/kappa
		dflux = kappa**-0.5
	else:
		# Check if this is slow
		flux  = enmap.samewcs(np.linalg.solve(kappa.T, rho.T).T, rho)
		dflux = enmap.samewcs(np.einsum("aayx->ayx",np.linalg.inv(kappa.T).T)**0.5, kappa)
	return flux, dflux

def get_flat_sky_correction(pixratio):
		return (0.5*(1+pixratio**2))**-0.5, 1/pixratio

def dtype_concat(dtypes):
	# numpy isn't cooperating, so I'm making this function myself
	return sum([np.dtype(dtype).descr for dtype in dtypes],[])

def merge_arrays(arrays):
	odtype = dtype_concat([a.dtype for a in arrays])
	res    = np.zeros(arrays[0].shape, odtype)
	for a in arrays:
		for key in a.dtype.names:
			res[key] = a[key]
	return res

def rpow(fmap, exp=2):
	"""Given a fourier-space map fmap corresponding to a real map, take it to the given exponent in
	*real space*, and return the fourier-space version of the result."""
	norm = fmap.area()**0.5
	map  = enmap.ifft(fmap/norm+0j, normalize="phys").real
	return enmap.fft (map**exp, normalize="phys").real*norm

def rmul(*args):
	norm = args[0].area()**0.5
	work = None
	for arg in args:
		rmap = enmap.ifft(arg/norm+0j, normalize="phys").real
		if work is None: work  = rmap
		else:            work *= rmap
	return enmap.fft(work, normalize="phys").real*norm

def rop(*args, op=np.multiply):
	norm = args[0].area()**0.5
	return enmap.fft(op(*[enmap.ifft(arg/norm+0j, normalize="phys").real for arg in args]), normalize="phys").real*norm

def snr_split(snrs, sntol=0.25, snmin=5):
	"""Given a list of S/N ratios, split split them into groups that
	can be processed together without interfering with each other.
	Returns [inds1, inds2, inds3, ...], where inds1 has the indices
	of the strongest snrs, inds2 slightly weaker snrs, and so on.
	The weakest element in a group will be at least sntol times the strongest
	element. Values below snmin are bunched into a single group."""
	v  = np.log(np.maximum(np.abs(snrs), snmin))/np.log(1/sntol)
	v -= np.max(v)+1e-9
	v  = utils.floor(v)
	return utils.find_equal_groups(v)[::-1]

def get_ref(a, tol=1e-3, default=0, n=1000):
	ref  = 0
	for i in range(2):
		vals = a[a>ref]
		if vals.size == 0: return default
		step = max(1,vals.size//n)
		ref  = np.median(vals[::step])
	return ref

def safe_pow(N, pow, bad_tol=1e-3):
	from enlib import array_ops
	v   = np.einsum("aa...->a...", N)
	ref = np.array([get_ref(vi) for vi in v])
	return array_ops.eigpow(N, pow, axes=[0,1], lim0=ref*bad_tol)

def make_circle_labels(shape, wcs, pixs, inds=None, r=2*utils.arcmin):
	if inds is None: inds = np.arange(1,len(pixs[0])+1)
	mask  = enmap.zeros(shape[-2:], wcs, np.int32)
	mask[pixs[0],pixs[1]] = inds
	dists, labels = mask.labeled_distance_transform(rmax=r)
	labels[dists >= r] = 0
	return labels

def get_central_radius(fbeam, lknee=2000, alpha=-3):
	"""Given a fourier-space beam and a filter lknee, get the radius at which
	the beam falls to zero for the first time. If would be more
	exact if this used the actual noise model, but in that case
	it's hard to know at what pixel to evaluate it, given that
	parts of the map could be empty."""
	l = fbeam.modlmap()
	fbeam = np.mean(fbeam.preflat,0)
	with utils.nowarn():
		fbeam = fbeam * (1 + (l/lknee)**alpha)**-1
	rbeam = enmap.ifft(fbeam+0j).real
	pos   = fbeam.pix2sky([0,0])
	br, r = rbeam.rbin(pos)
	br   /= br[0]
	return r[np.nonzero(br < 0)[0][0]]

#def fbeam_to_area(fbeam):
#	beam  = enmap.ifft(fbeam+0j).real
#	beam /= np.max(beam)
#	return np.sum(beam)*beam.pixsize()

# This huge comment was in NmatConstcorr. I've put it here for now
# in case it's interesting, but I'll probably delete it eventually.

# Our model that our map contains a single point source + noise,
#  m = BPa+n
# where a is the total flux in that pixel, P is something that puts all that flux in the correct
# pixel, and B is our beam, which preserves total flux but smears it out. So B[l=0] = 1. This means
# that P should have units of 1/pixarea so that Pa has units of flux per steradian.
#
# Our ML estimate of a is then
#  a = (P'B'N"BP)"P'B'N"m = rhs/kappa
#  rhs = P'B'N"m, kappa = P'B'N"BP
# In real space, each column of B would be a set of numbers that sum to 1. Near the equator
# these would be more concentrated, further away they would be broader. The peak height would
# be position-independent. The pixel size wouldn't enter here (aside from deciding how many
# pixels are hit) since the flux density is an intensive quantity.
#
# Regardless of projection B(x,y) would look like b(dist(x,y)), which is symmetric.
#
# How does the flat sky approximation enter here? Affects both B and N, but N doesn't affect
# the flux expectation value, so let's ignore it for now. Flat sky means that we replace B
# with B2(x,y) = b(dist0(x,y)), dist0(x,y) = ((x.ra-y.ra)**2*cos_dec + (x.dec-y.dec)**2)**0.5.
#
# What is (P'B2'B2P)"P'B2'BP? 
#
# BP  = a beam map centered on some location dec with total flux of 1. In car it's stretched hor by 1/cos(dec)
# B2P = similar, but stretched hor by 1/cos(dec0).
# (B2P)'(BP) is the dot product of these. Let's try for a gaussian:
# int 1/sqrt(2pi s1**2) 1/sqrt(2pi s2**2) exp(-(x*s1)**2) exp(-(x*s2)**2) dx =
# int 1/... exp(-(x*(s1**2+s2**2))) dx = sqrt(2 pi (s1**2+s2**2))/sqrt(2 pi s1**2)/sqrt(2 pi s2**2)
# = 1/sqrt(2 pi) * sqrt(s1**-2+s2**-2)
# What it should have been:
# 1/sqrt(2 pi) * sqrt(2 s1**-2)
#
# So (P'B2'B2P)"P'B2'BP = sqrt(s1**-2+s2**-2)/sqrt(s2**-2+s2**-2)
# where it should have been 1 if B2 were B. So we're off by a factor
# sqrt(cos(dec)**-2 + cos(dec0)**-2)/(2 cos(dec0)**-2)
#
# Hm, here I assumed that the normalization was different for the two cases,
# but is it? No, we have the same peak in all cases. The flat sky approximation
# only affects how we compute distances. So let's try again:
#
#   int exp(-0.5*(x*s1)**2)*exp(-0.5*(x*s2)**2) dx
# = int exp(-0.5*x**2*(s1**2+s2**2))
# = sqrt(2*pi*(s1**2+s2**2))
#
# (P'B2'B2P)"P'B2'P = sqrt(s1**2+s2**2)/sqrt(s2**2+s2**2)
# For my case that means we make a flux error of
# sqrt(cos(dec)**2+cos(dec0))/sqrt(2 cos(dec0)**2)
# = sqrt(0.5*(1+cos(dec)**2/cos(dec0)**2))
#
# This depends on it being gaussian, but let's just use it for now.
# What happens to rho and kappa separately?
# rho is wrong by sqrt(s1**2+s2**2)/sqrt(2*s1**2)
# = sqrt(cos(dec)**2+cos(dec0)**2)/sqrt(2*cos(dec)**2)
# = sqrt(0.5*(1+(cos(dec)/cos(dec0))**-2))
# kappa is wrong by sqrt(2 s2**2)/sqrt(2 s1**2) = (cos(dec)/cos(dec0))**-1
#
# But wait! We've forgotten about P! P should go as 1/pixsizemap, but
# in the flat sky it goes as pixsize. So we should keep P and P2 separate.
#
# Let's try again:
#
# rho error is:
#  rho2/rho = P2'B2'BP/(P'B'BP) = sqrt(0.5*(1+(cos(dec)/cos(dec0))**-2)) * area/area0
#           = sqrt(0.5*(1+(area/area0)**-2)) * (area/area0)
#           = sqrt(0.5*(1+(area/area0)**2))
# kappa error is
#  kappa2/kappa = (P2'B2'B2P2)'/(P'B'BP) = (cos(dec)/cos(dec0))**-1 * (area0/area)**-2
#           = area/area0
