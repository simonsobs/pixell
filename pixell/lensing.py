import numpy as np
from . import enmap, utils, powspec
try: from . import interpol
except ImportError: pass

####### Flat sky lensing #######

def lens_map(imap, grad_phi, order=3, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7):
	"""Lens map imap[{pre},ny,nx] according to grad_phi[2,ny,nx], where phi is the
	lensing potential, and grad_phi, which can be computed as enmap.grad(phi), simply
	is the coordinate displacement for each pixel. order, mode and border specify
	details of the interpolation used. See enlib.interpol.map_coordinates for details.
	If trans is true, the transpose operation is performed. This is NOT equivalent to
	delensing.

	If the same lensing field needs to be reused repeatedly, then higher efficiency
	can be gotten from calling displace_map directly with precomputed pixel positions."""
	# Converting from grad_phi to pix has roughly the same cost as calling displace_map.
	# So almost a factor 2 in speed can be won from calling displace_map directly.
	pos = imap.posmap() + grad_phi
	pix = imap.sky2pix(pos, safe=False)
	if not deriv:
		return displace_map(imap, pix, order=order, mode=mode, border=border, trans=trans)
	else:
		# displace_map deriv gives us ndim,{pre},ny,nx
		dlens_pix = displace_map(imap, pix, order=order, mode=mode, border=border, trans=trans, deriv=True)
		res = dlens_pix[0]*0
		pad = (slice(None),)+(None,)*(imap.ndim-2)+(slice(None),slice(None))
		for i in range(2):
			pos2 = pos.copy()
			pos2[i] += h
			pix2 = imap.sky2pix(pos2, safe=False)
			dpix = (pix2-pix)/h
			res += np.sum(dlens_pix * dpix[pad],0)
		return res

def delens_map(imap, grad_phi, nstep=3, order=3, mode="spline", border="cyclic"):
	"""The inverse of lens_map, such that delens_map(lens_map(imap, dpos), dpos) = imap
	for well-behaved fields. The inverse does not always exist, in which case the
	equation above will only be approximately fulfilled. The inverse is computed by
	iteration, with the number of steps in the iteration controllable through the
	nstep parameter. See enlib.interpol.map_coordinates for details on the other
	parameters."""
	grad_phi = delens_grad(grad_phi, nstep=nstep, order=order, mode=mode, border=border)
	return lens_map(imap, -grad_phi, order=order, mode=mode, border=border)

def delens_grad(grad_phi, nstep=3, order=3, mode="spline", border="cyclic"):
	"""Helper function for delens_map. Attempts to find the undisplaced gradient
	given one that has been displaced by itself."""
	alpha = grad_phi
	for i in range(nstep):
		alpha = lens_map(grad_phi, -alpha, order=order, mode=mode, border=border)
	return alpha

def displace_map(imap, pix, order=3, mode="spline", border="cyclic", trans=False, deriv=False):
	"""Displace map m[{pre},ny,nx] by pix[2,ny,nx], where pix indicates the location
	in the input map each output pixel should get its value from (float). The output
	is [{pre},ny,nx]."""
	if not deriv: omap = imap.copy()
	else:         omap = enmap.empty((2,)+imap.shape, imap.wcs, imap.dtype)
	if not trans:
		interpol.map_coordinates(imap, pix, omap, order=order, mode=mode, border=border, trans=trans, deriv=deriv)
	else:
		interpol.map_coordinates(omap, pix, imap, order=order, mode=mode, border=border, trans=trans, deriv=deriv)
	return omap

# Compatibility function. Not quite equivalent lens_map above due to taking phi rather than
# its gradient as an argument.
def lens_map_flat(cmb_map, phi_map):
	raw_pix  = cmb_map.pixmap() + enmap.grad_pix(phi_map)
	# And extract the interpolated values. Because of a bug in map_pixels with
	# mode="wrap", we must handle wrapping ourselves.
	npad = int(np.ceil(max(np.max(-raw_pix),np.max(raw_pix-np.array(cmb_map.shape[-2:])[:,None,None]))))
	pmap = enmap.pad(cmb_map, npad, wrap=True)
	return enmap.samewcs(utils.interpol(pmap, raw_pix+npad, order=4, mode="wrap"), cmb_map)

######## Curved sky lensing ########


def phi_to_kappa(phi_alm,phi_ainfo=None):
	"""Convert lensing potential alms phi_alm to
	lensing convergence alms kappa_alm, i.e.
	phi_alm * l * (l+1) / 2

	Args:
	    phi_alm: (...,N) ndarray of spherical harmonic alms of lensing potential
	    phi_ainfo: 	If ainfo is provided, it is an alm_info describing the layout 
	of the input alm. Otherwise it will be inferred from the alm itself.

	Returns:
	    kappa_alm: The filtered alms phi_alm * l * (l+1) / 2
	"""
	from . import curvedsky
	return curvedsky.almxfl(alm=phi_alm,lfunc=lambda x: x*(x+1)/2,ainfo=phi_ainfo)

def lens_map_curved(shape, wcs, phi_alm, cmb_alm, phi_ainfo=None, maplmax=None, dtype=np.float64, oversample=2.0, spin=[0,2], output="l", geodesic=True, verbose=False, delta_theta=None):
	from . import curvedsky, sharp
	# Restrict to target number of components
	oshape  = shape[-3:]
	if len(oshape) == 2: shape = (1,)+tuple(shape)
	if delta_theta is None: bsize = shape[-2]
	else:
		bsize = utils.nint(abs(delta_theta/utils.degree/wcs.wcs.cdelt[1]))
		# Adjust bsize so we don't get any tiny blocks at the end
		nblock= shape[-2]//bsize
		bsize = int(shape[-2]/(nblock+0.5))
	# Allocate output maps
	if "p" in output: phi_map   = enmap.empty(shape[-2:], wcs, dtype=dtype)
	if "k" in output:
		kappa_map = enmap.empty(shape[-2:], wcs, dtype=dtype)
		kappa_alm = phi_to_kappa(phi_alm,phi_ainfo=phi_ainfo)
		for i1 in range(0, shape[-2], bsize):
			curvedsky.alm2map(kappa_alm, kappa_map[...,i1:i1+bsize,:])
		del kappa_alm
	if "a" in output: grad_map  = enmap.empty((2,)+shape[-2:], wcs, dtype=dtype)
	if "u" in output: cmb_raw   = enmap.empty(shape, wcs, dtype=dtype)
	if "l" in output: cmb_obs   = enmap.empty(shape, wcs, dtype=dtype)
	# Then loop over dec bands
	for i1 in range(0, shape[-2], bsize):
		i2 = min(i1+bsize, shape[-2])
		lshape, lwcs = enmap.slice_geometry(shape, wcs, (slice(i1,i2),slice(None)))
		if "p" in output:
			if verbose: print("Computing phi map")
			curvedsky.alm2map(phi_alm, phi_map[...,i1:i2,:])
		if verbose: print("Computing grad map")
		if "a" in output: grad = grad_map[...,i1:i2,:]
		else: grad = enmap.zeros((2,)+lshape[-2:], lwcs, dtype=dtype)
		curvedsky.alm2map(phi_alm, grad, deriv=True)
		if "l" not in output: continue
		if verbose: print("Computing observed coordinates")
		obs_pos = enmap.posmap(lshape, lwcs)
		if verbose: print("Computing alpha map")
		raw_pos = enmap.samewcs(offset_by_grad(obs_pos, grad, pol=shape[-3]>1, geodesic=geodesic), obs_pos)
		del obs_pos, grad
		if "u" in output:
			if verbose: print("Computing unlensed map")
			curvedsky.alm2map(cmb_alm, cmb_raw[...,i1:i2,:], spin=spin)
		if verbose: print("Computing lensed map")
		cmb_obs[...,i1:i2,:] = curvedsky.alm2map_pos(cmb_alm, raw_pos[:2], oversample=oversample, spin=spin)
		if raw_pos.shape[0] > 2 and np.any(raw_pos[2]):
			if verbose: print("Rotating polarization")
			cmb_obs[...,i1:i2,:] = enmap.rotate_pol(cmb_obs[...,i1:i2,:], raw_pos[2])
		del raw_pos

	del cmb_alm, phi_alm
	# Output in same order as specified in output argument
	res = []
	for c in output:
		if   c == "l": res.append(cmb_obs.reshape(oshape))
		elif c == "u": res.append(cmb_raw.reshape(oshape))
		elif c == "p": res.append(phi_map)
		elif c == "k": res.append(kappa_map)
		elif c == "a": res.append(grad_map)
	return tuple(res)

def rand_alm(ps_lensinput, lmax=None, dtype=np.float64, seed=None, phi_seed=None, verbose=False, ncomp=None):
	from . import curvedsky, sharp
	ctype   = np.result_type(dtype,0j)
	if ncomp is not None: ps_lensinput = ps_lensinput[:1+ncomp,:1+ncomp]
	# First draw a random lensing field, and use it to compute the undeflected positions
	if verbose: print("Generating alms")
	if phi_seed is None:
		alm, ainfo = curvedsky.rand_alm(ps_lensinput, lmax=lmax, seed=seed, dtype=ctype, return_ainfo=True)
	else:
		# We want separate seeds for cmb and phi. This means we have to do things a bit more manually
		wps, ainfo = curvedsky.prepare_ps(ps_lensinput, lmax=lmax)
		alm = np.empty([1+ncomp,ainfo.nelem],ctype)
		curvedsky.rand_alm_white(ainfo, alm=alm[:1], seed=phi_seed)
		curvedsky.rand_alm_white(ainfo, alm=alm[1:], seed=seed)
		ps12 = enmap.multi_pow(wps, 0.5)
		ainfo.lmul(alm, (ps12/2**0.5).astype(dtype), alm)
		alm[:,:ainfo.lmax].imag  = 0
		alm[:,:ainfo.lmax].real *= 2**0.5
		del wps, ps12
		
	# Return phi_alm and cmb_alm
	return alm[0], alm[1:], ainfo


def rand_map(shape, wcs, ps_lensinput, lmax=None, maplmax=None, dtype=np.float64, seed=None, phi_seed=None, oversample=2.0, spin=[0,2], output="l", geodesic=True, verbose=False, delta_theta=None):
	# Restrict to target number of components
	oshape  = shape[-3:]
	if len(oshape) == 2: shape = (1,)+tuple(shape)
	ncomp   = shape[-3]
	phi_alm, cmb_alm, ainfo = rand_alm(ps_lensinput=ps_lensinput, 
									   lmax=lmax, dtype=dtype, seed=seed, 
									   phi_seed=phi_seed, verbose=verbose,
									   ncomp=ncomp)

	# Truncate alm if we want a smoother map. In taylens, it was necessary to truncate
	# to a lower lmax for the map than for phi, to avoid aliasing. The appropriate lmax
	# for the cmb was the one that fits the resolution. FIXME: Can't slice alm this way.
	#if maplmax: cmb_alm = cmb_alm[:,:maplmax]
	return lens_map_curved(shape=shape, wcs=wcs, phi_alm=phi_alm,
		cmb_alm=cmb_alm, phi_ainfo=ainfo, maplmax=maplmax,
		dtype=dtype, oversample=oversample, spin=spin,
		output=output, geodesic=geodesic, verbose=verbose,
		delta_theta=delta_theta)

def offset_by_grad(ipos, grad, geodesic=True, pol=None):
	"""Given a set of coordinates ipos[{dec,ra},...] and a gradient
	grad[{ddec,dphi/cos(dec)},...] (as returned by curvedsky.alm2map(deriv=True)),
	returns opos = ipos + grad, while properly parallel transporting
	on the sphere. If geodesic=False is specified, then an much faster
	approximation is used, which is still very accurate unless one is
	close to the poles."""
	ncomp = 2 if pol is False or pol is None and ipos.shape[0] <= 2 else 3
	opos = np.empty((ncomp,)+ipos.shape[1:])
	iflat = ipos.reshape(ipos.shape[0],-1)
	# Oflat is a flattened view of opos, so changes to oflat
	# are visible in our return value opos
	oflat = opos.reshape(opos.shape[0],-1)
	gflat = grad.reshape(grad.shape[0],-1)
	if geodesic:
		# Loop over chunks in order to conserve memory
		step = 0x100000
		for i in range(0, iflat.shape[1], step):
			# The helper function assumes zenith coordinates
			small_grad = gflat[:,i:i+step].copy(); small_grad[0] = -small_grad[0]
			small_ipos = iflat[:,i:i+step].copy(); small_ipos[0] = np.pi/2-small_ipos[0]
			# Compute the offset position and polarization rotation, the latter in the
			# form of cos and sin of the polarization rotation angle
			small_opos, small_orot = offset_by_grad_helper(small_ipos, small_grad, ncomp>2)
			oflat[0,i:i+step] = np.pi/2 - small_opos[0]
			oflat[1,i:i+step] = small_opos[1]
			# Handle rotation if necessary
			if oflat.shape[0] > 2:
				# Recover angle from cos and sin
				oflat[2,i:i+step] = np.arctan2(small_orot[1],small_orot[0])
				if iflat.shape[0] > 2:
					oflat[2,i:i+step] += iflat[2,i:i+step]
	else:
		oflat[0] = iflat[0] + gflat[0]
		oflat[1] = iflat[1] + gflat[1]/np.cos(iflat[0])
		oflat[:2] = pole_wrap(oflat[:2])
		if oflat.shape[0] > 2: oflat[2] = 0
	return opos

def offset_by_grad_helper(ipos, grad, pol):
	"""Find the new position and induced rotation
	from offseting the input positions ipos[2,nsamp]
	by grad[2,nsamp]."""
	grad = np.array(grad)
	# Decompose grad into direction and length.
	# Fix zero-length gradients first, to avoid
	# division by zero.
	grad[:,np.all(grad==0,0)] = 1e-20
	d = np.sum(grad**2,0)**0.5
	grad  /=d
	# Perform the position offset using spherical
	# trigonometry
	cosd, sind = np.cos(d), np.sin(d)
	cost, sint = np.cos(ipos[0]), np.sin(ipos[0])
	ocost  = cosd*cost-sind*sint*grad[0]
	osint  = (1-ocost**2)**0.5
	ophi   = ipos[1] + np.arcsin(sind*grad[1]/osint)
	if not pol:
		return np.array([np.arccos(ocost), ophi]), None
	# Compute the induced polarization rotation.
	A      = grad[1]/(sind*cost/sint+grad[0]*cosd)
	nom1   = grad[0]+grad[1]*A
	denom  = 1+A**2
	cosgam = 2*nom1**2/denom-1
	singam = 2*nom1*(grad[1]-grad[0]*A)/denom
	res = np.array([np.arccos(ocost), ophi]), np.array([cosgam,singam])
	return res

def pole_wrap(pos):
	"""Handle pole wraparound."""
	a = np.array(pos)
	bad = np.where(a[0] > np.pi/2)
	a[0,bad] = np.pi - a[0,bad]
	a[1,bad] = a[1,bad]+np.pi
	bad = np.where(a[0] < -np.pi/2)
	a[0,bad] = -np.pi - a[0,bad]
	a[1,bad] = a[1,bad]+np.pi
	return a
