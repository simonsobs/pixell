import numpy as np
from . import enmap, utils, powspec, curvedsky
try: from . import interpol
except ImportError: pass
import warnings

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
	return curvedsky.almxfl(alm=phi_alm,lfilter=lambda l: l*(l+1)/2,ainfo=phi_ainfo)

def kappa_to_phi(kappa_alm,kappa_ainfo=None):
	"""Convert lensing convergence alms kappa_alm to
	lensing potential alms phi_alm, i.e.
	kappa_alm / ( l * (l+1) / 2 )

	Args:
	    kappa_alm: (...,N) ndarray of spherical harmonic alms of lensing convergence
	    kappa_ainfo: If ainfo is provided, it is an alm_info describing the layout 
	of the input alm. Otherwise it will be inferred from the alm itself.

	Returns:
	    phi_alm: The filtered alms kappa_alm / ( l * (l+1) / 2 )
	"""
	from . import curvedsky
	with utils.nowarn():
		oalm = curvedsky.almxfl(alm=kappa_alm,lfilter=lambda l: 1./(l*(l+1)/2) ,ainfo=kappa_ainfo)
	return utils.remove_nan(oalm)


def kappa_to_phi(kappa_alm,kappa_ainfo=None):
	"""Convert lensing convergence alms kappa_alm to
	lensing potential alms phi_alm, i.e.
	kappa_alm / ( l * (l+1) / 2 )

	Args:
	    kappa_alm: (...,N) ndarray of spherical harmonic alms of lensing convergence
	    kappa_ainfo: If ainfo is provided, it is an alm_info describing the layout 
	of the input alm. Otherwise it will be inferred from the alm itself.

	Returns:
	    phi_alm: The filtered alms kappa_alm / ( l * (l+1) / 2 )
	"""
	from . import curvedsky
	oalm = curvedsky.almxfl(alm=kappa_alm,lfilter=lambda x: 1./(x*(x+1)/2) ,ainfo=kappa_ainfo)
	oalm[~np.isfinite(oalm)] = 0
	return oalm

def _fix_lenspyx_result(lenspyx_result, lenspyx_geom_info, shape, wcs):
	"""Lenspyx delivers results for rectangular geometries in not quite the
	format that we'd like to use. First, it delivers a tuple of 1d arrays,
	rather than a 3d array of shape (-1, ny, nx). Then, the bottom-left pixel
	corresponds to the ring with the smallest colatitude and minimum phi value,
	with phi increasing to the right. The result is always full-sky. This
	function rearranges the result into the shape we expect and copies its data
	into the right order, such that it matches the geometry obtained by 
	enmap.fullsky_geometry.

	Parameters
	----------
	lenspyx_result : 1d np.ndarray or tuple of 1d np.ndarray
		The results of a call to lenspyx.lensing functions.
	lenspyx_geom_info : (name, info_dict)
		The argument that would need to be supplied to lenspyx.get_geom:
		* name gives the rectangular projection ('cc' or 'f1')
		* info_dict gives the map shape as {'ntheta': ny, 'nphi': nx}
		Note this is a full-sky geom_info, not restricted.
	shape : (ny, nx) tuple
		Shape of output map (not necessarily full-sky).
	wcs : astropy.wcs.WCS
		WCS of output map (not necessarily full-sky).

	Returns
	-------
	(-1, ny, nx) enmap.ndmap
		The lenspyx_result projected onto the shape, wcs geometry.

	Raises
	------
	AssertionError
		If the lenspyx geometry is not shifted by an integer number of pixels
		from the corresponding pixell geometry.
	"""
	import lenspyx

	# do some sanity checks on the full-sky geom_info
	geom = lenspyx.get_geom(lenspyx_geom_info)
	phi0, nph = geom.phi0, geom.nph
	assert np.all(phi0 == phi0[0]), 'all phi0 must be the same'
	assert np.all(nph == nph[0]), 'all nph must be the same'

	# get the pixell full-sky geometry we will paste the results into
	variant = dict(cc='cc', f1='fejer1')[lenspyx_geom_info[0]]
	fs_shape = (lenspyx_geom_info[1]['ntheta'], lenspyx_geom_info[1]['nphi'])
	fs_shape, fs_wcs = enmap.fullsky_geometry(shape=fs_shape, variant=variant)

	# lenspyx delivers tuples of arrays, not arrays
	fs_inp = np.asarray(lenspyx_result).reshape((-1, *fs_shape))
	fs_out = np.zeros_like(fs_inp)

	# now cut and paste data! :(

	# first, handle the x coordinates. first find the location of phi0
	# in the full-sky geometry
	_, _phi0_ind = enmap.sky2pix(fs_shape, fs_wcs, [0, phi0[0]])
	phi0_ind = np.round(_phi0_ind).astype(int)
	assert phi0_ind == _phi0_ind, \
		('we cannot handle the case of a non-integer pixel with cut and paste '
		'but could roll the whole array by a fractional pixel using ffts, this '
		'needs to be implemented')

	# then do copy paste, handling whether phi increases
	if fs_wcs.wcs.cdelt[0] > 0: # wcs is in x,y ordering
		fs_out[..., phi0_ind:] = fs_inp[..., :shape[-1] - phi0_ind]
		fs_out[..., :phi0_ind] = fs_inp[..., shape[-1] - phi0_ind:]
	else:
		fs_out[..., phi0_ind::-1] = fs_inp[..., :phi0_ind+1]
		fs_out[..., :phi0_ind:-1] = fs_inp[..., phi0_ind+1:]

	# next, we know lenspyx delivers colatitudes, but fullsky_geometry
	# is likely the opposite of that
	if fs_wcs.wcs.cdelt[1] > 0: # wcs is in x,y ordering
		fs_out = fs_out[..., ::-1, :] # flip the y coordinates if theta increases

	# finally, extract the cutout we want
	fs_out = enmap.ndmap(fs_out, fs_wcs)
	return enmap.extract(fs_out, shape, wcs)

def _lens_map_curved_lenspyx(shape, wcs, phi_alm, cmb_alm, phi_ainfo=None, 
								dtype=np.float64, spin=[0, 2], output="l",
								epsilon=1e-7, nthreads=0, verbose=False):
	"""Lenses a CMB map given the lensing potential harmonic transform and the
	unlensed CMB harmonic transform.  By default, T, E, B spherical harmonic
	coefficients are accepted and the returned maps are T, Q, U. Unlike 
	lens_map_curved, this implements lensing using lenspyx, which is a more
	optimized/specialized/stable lensing library. This function formats lenspyx
	outputs to be drop-in replacements for lens_map_curved.

	Parameters
	----------
	shape : tuple
		Shape of the output map. Only the first pre-dimension (-3), if passed,
		is kept. 
	wcs : WCS object
		World Coordinate System object describing the map projection.
	phi_alm : array-like
		Spherical harmonic coefficients of the lensing potential.
	cmb_alm : array-like
		Spherical harmonic coefficients of the CMB. If (3, nelem) shaped, the
		coeffients are assumed to be in the form of [T, E, B] in that order,
		unless spin is 0.
	phi_ainfo : alm_info, optional
		alm_info object containing information about the alm layout. Default:
		standard triangular layout.
	dtype : data-type, optional
		Data type of the output maps. Default is np.float64.
	spin : list, optional
		List of spins. These describe how to handle the [ncomp] axis in cmb_alm.
	 	0: scalar transform. Consumes one element in the component axis
	 	not 0: spin transform. Consumes two elements from the component axis.
	 	For example, if you have a TEB alm [3,nelem] and want to transform it
	 	to a TQU map [3,ny,nx], you would use spin=[0,2] to perform a scalar
	 	transform for the T component and a spin-2 transform for the Q,U
	 	components. Another example. If you had an alm [5,nelem] and map
	 	[5,ny,nx] and the first element was scalar, the next pair spin-1
	 	and the next pair spin-2, you woudl use spin=[0,1,2]. default:[0,2]
	output : str, optional
		String which specifies which maps to output, e.g. "lu". Default is "l".
		"l" - lensed CMB map
		"u" - unlensed CMB map
		"p" - lensing potential map
		"k" - convergence map
		"a" - deflection angle maps
	epsilon : float, optional
		Target result accuracy, by default 1e-7. See lenspyx.
	nthreads : int, optional
		number of threads to use, by default 0 (os.cpu_count()). See lenspyx.
	verbose : bool, optional
		If True, prints progress information. Default is False.

	Returns
	-------
	tuple
		A tuple containing the requested output maps in the order specified by
		the `output` parameter.

	Notes
	-----
	This function assumes the cmb_alm is in a triangular layout.

	This function has a restrictive interpretation of spin. If the default
	[0, 2] is passed, the cmb_alm and output shape must have an axis size of
	2 or 3 in the (-3) axis, in which case the inputs are interpreted as T, E
	or T, E, B, respectively (see lenspyx). Otherwise, a fully spin-0 transform
	must be passed. The default should cover the vast majority of use-cases.
	"""
	import lenspyx

	# restrict to target number of components
	oshape  = shape[-3:]
	if len(oshape) == 2:
		oshape = (1, *shape)

	assert np.asarray(phi_alm).ndim == 1, \
		'Can only do 1-dimensional phi_alm, set up a loop if you have many'
	
	cmb_alm = np.atleast_2d(cmb_alm)
	assert cmb_alm.ndim <= 2, \
		'Can only do <=2-dimensional cmb_alm, set up a loop if you have many'

	# map from spin to pol 
	pol = False
	pre_shape = oshape[0]
	pre_cmb = cmb_alm.shape[0]
	if spin == [0, 2]:
		assert pre_cmb in (2, 3), \
			f'{spin=} indicates TEB but number of components in alm {pre_cmb=}' + \
			' not 2 or 3'
		assert pre_shape == 3, \
			f'{spin=} indicates TEB but number of components in map {pre_shape=}' + \
			' not 3'
		pol = True
	else:
		assert np.all(spin) == 0, \
			f'expect spin-0 transform for all {pre_cmb} components'
		assert pre_cmb == pre_shape, \
			f'expect {pre_cmb=} to be the same as {pre_cmb=}'
	
	# we need to get the "lenspyx geometry" from the "pixell geometry".
	# we know pixell will have a ducc-compatible rectangular geometry, so get
	# its parameters: number of pixels in x and y, and the name of the
	# rectangular geometry. pixell capitalizes the names, but lenspyx wants them 
	# lowercase. finally, handle cut-sky
	ducc_geo = curvedsky.analyse_geometry(shape, wcs).ducc_geo
	ny = ducc_geo.ny
	nx = ducc_geo.nx
	name = ducc_geo.name.lower()
	geom_info = (name, dict(ntheta=ny, nphi=nx)) # this is a full-sky geom_info

	# after getting the result from lenspyx, it needs to be fixed to respect
	# pixell conventions
	if 'l' in output:
		phi_lmax = curvedsky.nalm2lmax(phi_alm.shape[-1])
		d_alm = np.empty_like(phi_alm)
		lfilter = np.sqrt(np.arange(phi_lmax + 1) * np.arange(1, phi_lmax + 2))
		curvedsky.almxfl(phi_alm, lfilter, out=d_alm)
		cmb_obs = lenspyx.alm2lenmap(cmb_alm, d_alm, geom_info,
										epsilon=epsilon, verbose=verbose, 
									 	nthreads=nthreads, pol=pol)
		cmb_obs = fix_lenspyx_result(cmb_obs, geom_info, oshape, wcs)
		cmb_obs = cmb_obs.astype(dtype=dtype, copy=False)

	# possibly get extra outputs
	if 'u' in output:
		cmb_raw = enmap.empty(shape, wcs, dtype=dtype)
		if verbose:
			print("Computing unlensed map")
		curvedsky.alm2map(cmb_alm, cmb_raw, spin=spin)
	if 'p' in output:
		phi_map = enmap.empty(oshape[-2:], wcs, dtype=dtype)
		if verbose:
			print('Computing phi map')
		curvedsky.alm2map(phi_alm, phi_map)
	if 'k' in output:
		kappa_map = enmap.empty(oshape[-2:], wcs, dtype=dtype)
		kappa_alm = phi_to_kappa(phi_alm, phi_ainfo=phi_ainfo)
		curvedsky.alm2map(kappa_alm, kappa_map)
	if 'a' in output:
		grad_map = enmap.empty((2, *oshape[-2:]), wcs, dtype=dtype)
		curvedsky.alm2map(phi_alm, grad_map, deriv=True)

	# Output in same order as specified in output argument
	res = []
	for c in output:
		if   c == 'l': res.append(cmb_obs.squeeze())
		elif c == "u": res.append(cmb_raw.squeeze())
		elif c == "p": res.append(phi_map)
		elif c == "k": res.append(kappa_map)
		elif c == "a": res.append(grad_map)
	return tuple(res)

def lens_map_curved(shape, wcs, phi_alm, cmb_alm, phi_ainfo=None, dtype=np.float64, 
                    spin=[0,2], output="l", method='pixell', geodesic=True, 
                    delta_theta=None, epsilon=1e-7, nthreads=0, verbose=False):
	"""
	Lenses a CMB map given the lensing potential harmonic transform and the CMB 
	harmonic transform.  By default, T,E,B spherical harmonic coefficients are
	accepted and the returned maps are T, Q, U.
	Parameters:
	-----------
	shape : tuple
		Shape of the output map. Only the first pre-dimension (-3), if passed,
		is kept.
	wcs : WCS object
		World Coordinate System object describing the map projection.
	phi_alm : array-like
		Spherical harmonic coefficients of the lensing potential.
	cmb_alm : array-like
		Spherical harmonic coefficients of the CMB. If (3, nelem) shaped, the
		coeffients are assumed to be in the form of [T, E, B] in that order,
		unless spin is 0.
	phi_ainfo : alm_info, optional
		alm_info object containing information about the alm layout. Default: standard triangular layout.
	dtype : data-type, optional
		Data type of the output maps. Default is np.float64.
	spin : list, optional
		List of spins. These describe how to handle the [ncomp] axis in cmb_alm.
	 	0: scalar transform. Consumes one element in the component axis
	 	not 0: spin transform. Consumes two elements from the component axis.
	 	For example, if you have a TEB alm [3,nelem] and want to transform it
	 	to a TQU map [3,ny,nx], you would use spin=[0,2] to perform a scalar
	 	transform for the T component and a spin-2 transform for the Q,U
	 	components. Another example. If you had an alm [5,nelem] and map
	 	[5,ny,nx] and the first element was scalar, the next pair spin-1
	 	and the next pair spin-2, you woudl use spin=[0,1,2]. default:[0,2]
	output : str, optional
		String which specifies which maps to output, e.g. "lu". Default is "l".
		"l" - lensed CMB map
		"u" - unlensed CMB map
		"p" - lensing potential map
		"k" - convergence map
		"a" - deflection angle maps
	method : str, optional
		Select the implementation, either 'pixell' or 'lenspyx'.
	geodesic : bool, optional
		Properly parallel transport on the sphere (default). If False, a much faster
		approximation is used, which is still very accurate unless one is
		close to the poles. Only for 'pixell' method.
	delta_theta : float, optional
		Step size for the theta coordinate. Only for 'pixell' method.
	epsilon : float, optional
		Target result accuracy, by default 1e-7. Only for 'lenspyx' method.
	nthreads : int, optional
		number of threads to use, by default 0 (os.cpu_count()). Only for 'lenspyx' method.
	verbose : bool, optional
		If True, prints progress information. Default is False.
	Returns:
	--------
	tuple
		A tuple containing the requested output maps in the order specified by
		the `output` parameter.

	Notes
	-----
	For 'pixell' and 'lenspyx':
	This function assumes the cmb_alm is in a triangular layout.

	For 'lenspyx':
	This function has a restrictive interpretation of spin. If the default
	[0, 2] is passed, the cmb_alm and output shape must have an axis size of
	2 or 3 in the (-3) axis, in which case the inputs are interpreted as T, E
	or T, E, B, respectively (see lenspyx). Otherwise, a fully spin-0 transform
	must be passed. The default should cover the vast majority of use-cases.
	"""
	assert method in ('pixell', 'lenspyx'), \
		"method must be one of 'pixell' or 'lenspyx'"

	if method=='pixell':
		warnings.warn('In the future, the default will be switched to lenspyx')

		from . import curvedsky
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
			cmb_obs[...,i1:i2,:] = curvedsky.alm2map_pos(cmb_alm, raw_pos[:2], spin=spin)
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
	
	elif method == 'lenspyx':
		return _lens_map_curved_lenspyx(shape, wcs, phi_alm, cmb_alm, phi_ainfo=phi_ainfo,
										dtype=dtype, spin=spin, output=output,
										epsilon=epsilon, nthreads=nthreads, verbose=verbose)

def rand_alm(ps_lensinput, lmax=None, dtype=np.float64, seed=None, phi_seed=None, verbose=False, ncomp=None):
	from . import curvedsky
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


def rand_map(shape, wcs, ps_lensinput, lmax=None, dtype=np.float64, seed=None, phi_seed=None, spin=[0,2], output="l", geodesic=True, verbose=False, delta_theta=None):
	# Restrict to target number of components
	oshape  = shape[-3:]
	if len(oshape) == 2: shape = (1,)+tuple(shape)
	ncomp   = shape[-3]
	phi_alm, cmb_alm, ainfo = rand_alm(ps_lensinput=ps_lensinput,
		lmax=lmax, dtype=dtype, seed=seed, phi_seed=phi_seed, verbose=verbose, ncomp=ncomp)

	# Truncate alm if we want a smoother map. In taylens, it was necessary to truncate
	# to a lower lmax for the map than for phi, to avoid aliasing. The appropriate lmax
	# for the cmb was the one that fits the resolution. FIXME: Can't slice alm this way.
	#if maplmax: cmb_alm = cmb_alm[:,:maplmax]
	return lens_map_curved(shape=shape, wcs=wcs, phi_alm=phi_alm,
		cmb_alm=cmb_alm, phi_ainfo=ainfo,
		dtype=dtype, spin=spin,
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
	with utils.nowarn():
		# If sint is zero then we get a divide by zero here, but
		# the final result is still finite
		A    = grad[1]/(sind*cost/sint+grad[0]*cosd)
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
