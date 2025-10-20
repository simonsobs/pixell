from __future__ import print_function
import numpy as np
from scipy import spatial
from . import wcsutils, utils, enmap, coordinates, fft, curvedsky

# Python 2/3 compatibility
try: basestring
except NameError: basestring = str

def thumbnails(imap, coords, r=5*utils.arcmin, res=None, proj=None, apod=2*utils.arcmin,
		method="mixed", order=3, oversample=4, pol=None, oshape=None, owcs=None,
		extensive=False, verbose=False, filter=None,pixwin=False,pixwin_order=0):
	"""Given an enmap [...,ny,nx] and a set of coordinates in a numpy array
	coords with shape (n,2) and ordering [n,{dec,ra}], extract a set
	of thumbnail images [n,...,thumby,thumbx] centered on each set of
	coordinates, where n is the number of images. 
	When proj is 'tan', each of these thumbnail images is projected onto a local 
	tangent plane, removing the effect of size and shape distortions in the input 
	map.

	If oshape, owcs are specified, then the thumbnails will have this geometry,
	which should be centered on [0,0]. Otherwise, a geometry with the given
	projection (defaults to "tan" = gnomonic projection) will be constructed,
	going up to a maximum radius of r. FIXME: Defaults to "car" instead while
	enmap.pixsizemap is buggy for non-cylindrical projections.

	The reprojection involved in this operation implies interpolation. The default
	is to use fft rescaling to oversample the input pixels by the given pixel, and
	then use bicubic spline interpolation to read off the values at the output
	pixel centers. The fft oversampling can be controlled with the oversample argument.
	Values <= 1 turns this off. The other interpolation step is controlled using the
	"order" argument. 0/1/3 corresponds to nearest neighbor, bilinear and bicubic spline
	interpolation respectively.

	If pol == True, then Q,U will be rotated to take into account the change in
	the local northward direction impled in the reprojection. The default is to
	do polarization rotation automatically if the input map has a compatible shape,
	e.g. at least 3 axes and a length of 3 for the 3rd last one. TODO: I haven't
	tested this yet.

	If extensive == True (not the default), then the map is assumed to contain an
	extensive field rather than an intensive one. An extensive field is one where
	the values in the pixels depend on the size of the pixel. For example, if the
	inverse variance in the map is given per pixel, then this ivar map will be
	extensive, but if it's given in units of inverse variance per square arcmin
	then it's intensive.

	For reprojecting inverse variance maps, consider using the wrapper thumbnails_ivar,
	which makes it easier to avoid common pitfalls.
	
	If pixwin is True, the pixel window will be deconvolved."""
	# FIXME: Specifying a geometry manually is broken - see usage of r in neighborhood_pixboxes below
	# Handle arbitrary coords shape
	if proj is None: proj = "car"
	coords = np.asarray(coords)
	ishape = coords.shape[:-1]
	coords = coords.reshape(-1, coords.shape[-1])
	# If the output geometry was not given explicitly, then build one
	if oshape is None:
		if res is None: res = min(np.abs(imap.wcs.wcs.cdelt))*utils.degree/2
		oshape, owcs = enmap.thumbnail_geometry(r=r, res=res, proj=proj)
	# Check if we should be doing polarization rotation
	pol_compat = imap.ndim >= 3 and imap.shape[-3] == 3
	if pol is None: pol = pol_compat
	if pol and not pol_compat: raise ValueError("Polarization rotation requested, but can't interpret map shape %s as IQU map" % (str(imap.shape)))
	nsrc = len(coords)
	if verbose: print("Extracting %d %dx%d thumbnails from %s map" % (nsrc, oshape[-2], oshape[-1], str(imap.shape)))
	opos = enmap.posmap(oshape, owcs)
	# Get the pixel area around each of the coordinates
	rtot     = r + apod
	apod_pix = utils.nint(apod/(np.min(np.abs(imap.wcs.wcs.cdelt))*utils.degree))
	pixboxes = enmap.neighborhood_pixboxes(imap.shape, imap.wcs, coords, rtot)
	# Define our output maps, which we will fill below
	omaps = enmap.zeros((nsrc,)+imap.shape[:-2]+oshape, owcs, imap.dtype)
	for si, pixbox in enumerate(pixboxes):
		if method == "nufft" or method == "mixed" and oversample > 1:
			# Make the pixbox fft-friendly
			for i in range(2):
				pixbox[1,i] = pixbox[0,i] + fft.fft_len(pixbox[1,i]-pixbox[0,i], direction="above", factors=[2,3,5])
		ithumb = imap.extract_pixbox(pixbox)
		if extensive: ithumb /= ithumb.pixsizemap()
		ithumb = ithumb.apod(apod_pix, fill="median")
		if pixwin: ithumb = enmap.unapply_window(ithumb, order=pixwin_order)
		if filter is not None: ithumb = filter(ithumb)
		if verbose:
			print("%4d/%d %6.2f %6.2f %8.2f %dx%d" % (si+1, nsrc, coords[si,0]/utils.degree, coords[si,1]/utils.degree, np.max(ithumb), ithumb.shape[-2], ithumb.shape[-1]))
		# Oversample using fourier if requested. We do this because fourier
		# interpolation is better than spline interpolation overall
		if method == "mixed" and oversample > 1:
			fshape = utils.nint(np.array(oshape[-2:])*oversample)
			ithumb = ithumb.resample(fshape, method="fft")
		# I apologize for the syntax. There should be a better way to do this
		ipos = coordinates.transform("cel", ["cel",[[0,0,coords[si,1],coords[si,0]],False]], opos[::-1], pol=pol)
		ipos, rest = ipos[1::-1], ipos[2:]
		if   method in ["spline", "mixed"]: ipol_mode = "spline"
		elif method in ["nufft"]:           ipol_mode = "nufft"
		else: raise ValueError("Unrecognized method '%s'" % (str(method)))
		omaps[si] = ithumb.at(ipos, mode=ipol_mode, order=order)
		# Apply the polarization rotation. The sign is flipped because we computed the
		# rotation from the output to the input
		if pol: omaps[si] = enmap.rotate_pol(omaps[si], -rest[0])
	if extensive: omaps *= omaps.pixsizemap()
	# Restore original dimension
	omaps = omaps.reshape(ishape + omaps.shape[1:])
	return omaps

def thumbnails_ivar(imap, coords, r=5*utils.arcmin, res=None, proj=None,
		oshape=None, owcs=None, order=1, extensive=True, verbose=False):
	"""Like thumbnails, but for hitcounts, ivars, masks, and other quantities that
	should stay positive and local. Remember to set extensive to True if you have an
	extensive quantity, i.e. if the values in each pixel would go up if multiple pixels
	combined. An example of this is a hitcount map or ivar per pixel. Conversely, if
	you have an intensive quantity like ivar per arcmin you should set extensive=False."""
	return thumbnails(imap, coords, r=r, res=res, proj=proj, oshape=oshape, owcs=owcs,
			order=order, oversample=1, pol=False, extensive=extensive, verbose=verbose,
			pixwin=False)

def map2healpix(imap, nside=None, lmax=None, out=None, rot=None, spin=[0,2], method="harm", order=1, extensive=False, bsize=100000, nside_mode="pow2", boundary="constant", verbose=False, niter=0):
	"""Reproject from an enmap to healpix, optionally including a rotation.

	imap:  The input enmap[...,ny,nx]. Stokes along the -3rd axis if
	       present.
	nside: The nside of the healpix map to generate. Not used if
	       an output map is passed. Otherwise defaults to the same
	       resolution as the input map.
	lmax:  The highest multipole to use in any harmonic-space
	       operations. Defaults to the input maps' Nyquist limit.
	out:   An optional array [...,npix] to write the output map to.
	       The ...  part must match the input map, as must the data
	       type.
	rot:   An optional coordinate rotation to apply. Either a string
	       "isys,osys", where isys is the system to transform from,
	       and osys is the system to transform to. Currently the values
	       "cel"/"equ" and "gal" are recognized. Alternatively, a tuple of
	       3 euler zyz euler angles can be passed, in the same convention
	       as healpy.rotate_alm.
	spin:  A description of the spin of the entries along the stokes
	       axis. Defaults to [0,2], which means that the first entry
	       is spin-0, followed by a spin-2 pair (any non-zero spin
	       covers a pair of entries). If the axis is longer than
	       what's covered in the description, then it is repeated as
	       necessary. Pass spin=[0] to disable any special treatment
	       of this axis.
	method: How to interpolate between the input and output
	       pixelizations. Can be "harm" (default) or "spline".
	       "harm" maps between them using spherical harmonics
	       transforms. This preserves the power spectrum (so no window
	       function is introduced), and averages noise down when the
	       output pixels are larger than the input pixels. However, it
	       can suffer from ringing around very bright features, and an
	       all-positive input map may end up with small negative values.
	       "spline" instead uses spline interpolation to look up the
	       value in the intput map corresponding to each pixel center
	       in the output map. The spline order is controlled with the
	       "order" argument. Overall "harm" is best suited for normal
	       sky maps, while "spline" with order = 0 or 1 is best suited
	       for hitcount maps and masks.
	order: The spline order to use when method="spline".
	       0 corresponds to nearest neighbor interpolation.
	       1 corresponds to bilinear interpolation (default)
	       3 corresponds to bicubic spline interpolation.
	       0 and 1 are local and do not introduce values outside
	       the input range, but introduce some aliasing and loss of
	       power. 3 has less power loss, but still non-zero, and
	       is vulnerable to ringing.
	extensive: Whether the map represents an extensive (as opposed to
	       intensive) quantity. Extensive quantities have values
	       proportional to the pixel size, unlike intensive quantities.
	       Hitcount per pixel is an extensive quantity. Hitcount per
	       square degree is an intensive quantity, as is a temperature
	       map. Defaults to False.
	bsize: The spline method operates on batches of pixels to save memory.
	       This controls the batch size, in pixels. Defaults to 100000.
	nside_mode: Controls which restrictions apply to nside in the case where
	       it has to be inferred automatically. Can be "pow2", "mul32" and "any".
	       "pow2", the default, results in nside being a power of two, as
	       required by the healpix standard.
	       "mul32" relaxes this requirement, making a map where nside is a
	       multiple of 32. This is compatible with most healpix operations,
	       but not with ud_grade or the nest pixel ordering.
	       "any" allows for any integer nside.
	boundary: The boundary conditions assumed for the input map when
	       method="spline". Defaults to "constant", which assumes that
	       anything outsize the map has a constant value of 0. Another
	       useful value is "wrap", which assumes that the right side
	       wraps over to the left, and the top to the bottom. See
	       scipy.ndimage.distance_transform's documentation for other,
	       less useful values. method="harm" always assumes "constant"
	       regardless of this setting.
	verbose: Whether to print information about what it's doing.
	       Defaults to False, which doesn't print anything.

	Typical usage:
	* map_healpix  = map2healpix(map,  rot="cel,gal")
	* ivar_healpix = map2healpix(ivar, rot="cel,gal", method="spline", spin=[0], extensive=True)
	"""
	# Get the map's typical resolution from cdelt
	ires = np.mean(np.abs(imap.wcs.wcs.cdelt))*utils.degree
	lnyq = np.pi/ires
	if out is None:
		if nside is None:
			nside = restrict_nside(((4*np.pi/ires**2)/12)**0.5, nside_mode)
		out = np.zeros(imap.shape[:-2]+(12*nside**2,), imap.dtype)
	npix     = out.shape[-1]
	opixsize = 4*np.pi/npix
	# Might not be safe to go all the way to the Nyquist l, but looks that way to my tests.
	if lmax is None: lmax = lnyq
	if extensive:
		imap = imap * (opixsize / imap.pixsizemap(broadcastable=True)) # not /= to avoid changing original imap
	if method in ["harm", "harmonic"]:
		# Harmonic interpolation preserves the power spectrum, but can introduce ringing.
		# Probably not a good choice for positive-only quantities like hitcounts.
		# Coordinate rotation is slow.
		alm = curvedsky.map2alm(imap, lmax=lmax, spin=spin, niter=niter)
		if rot is not None:
			curvedsky.rotate_alm(alm, *rot2euler(rot), inplace=True)
		curvedsky.alm2map_healpix(alm, out, spin=spin)
		del alm
	elif method == "spline":
		# Covers both cubic spline interpolation (order=3), linear interpolation (order=1)
		# and nearest neighbor (order=0). Harmonic interpolation is preferable to cubic
		# splines, but linear and nearest neighbor may be useful. Coordinate rotation may
		# be slow.
		import healpy
		ip = utils.interpolator(imap, npre=-2, order=order, mode="spline", border=boundary)
		# Figure out if we need to compute polarization rotations
		pol = imap.ndim > 2 and any([s != 0 for s,c1,c2 in enmap.spin_helper(spin, imap.shape[-3])])
		# Batch to save memory
		for i1 in range(0, npix, bsize):
			i2   = min(i1+bsize, npix)
			opix = np.arange(i1,i2)
			pos  = healpy.pix2ang(nside, opix)[::-1]
			pos[1][:] = np.pi/2-pos[1]
			if rot is not None:
				# Not sure why the [::-1] is necessary here. Maybe psi,theta,phi vs. phi,theta,psi?
				pos = coordinates.transform_euler(inv_euler(rot2euler(rot))[::-1], pos, pol=pol)
			# The actual interpolation happens here
			vals  = imap.at(pos[1::-1], order=order, border=boundary, ip=ip)
			if rot is not None and imap.ndim > 2:
				# Update the polarization to account for the new coordinate system
				for s, c1, c2 in enmap.spin_helper(spin, imap.shape[-3]):
					if s != 0:
						vals = enmap.rotate_pol(vals, -pos[2], spin=s, comps=[c1,c2-1], axis=-2)
			out[...,i1:i2] = vals
	else:
		raise ValueError("Map reprojection method '%s' not recognized" % str(method))
	return out

def healpix2map(iheal, shape=None, wcs=None, lmax=None, out=None, rot=None, spin=[0,2], method="harm", order=1, extensive=False, bsize=100000, verbose=False, niter=0):
	"""Reproject from healpix to an enmap, optionally including a rotation.

	iheal: The input healpix map [...,npix]. Stokes along the -2nd axis if
	       present.
	shape: The (...,ny,nx) shape of the output map. Only the last two entries
	       are used, the rest of the dimensions are taken from iheal.
	       Mandatory unless an output map is passed.
	wcs  : The world woordinate system object the output map.
	       Mandatory unless an output map is passed.
	lmax:  The highest multipole to use in any harmonic-space
	       operations. Defaults to 3 times the nside of iheal.
	out:   An optional enmap [...,ny,nx] to write the output map to.
	       The ...  part must match iheal, as must the data type.
	rot:   An optional coordinate rotation to apply. Either a string
	       "isys,osys", where isys is the system to transform from,
	       and osys is the system to transform to. Currently the values
	       "cel"/"equ" and "gal" are recognized. Alternatively, a tuple of
	       3 euler zyz euler angles can be passed, in the same convention
	       as healpy.rotate_alm.
	spin:  A description of the spin of the entries along the stokes
	       axis. Defaults to [0,2], which means that the first entry
	       is spin-0, followed by a spin-2 pair (any non-zero spin
	       covers a pair of entries). If the axis is longer than
	       what's covered in the description, then it is repeated as
	       necessary. Pass spin=[0] to disable any special treatment
	       of this axis.
	method: How to interpolate between the input and output
	       pixelizations. Can be "harm" (default) or "spline".
	       "harm" maps between them using spherical harmonics
	       transforms. This preserves the power spectrum (so no window
	       function is introduced), and averages noise down when the
	       output pixels are larger than the input pixels. However, it
	       can suffer from ringing around very bright features, and an
	       all-positive input map may end up with small negative values.
	       "spline" instead uses spline interpolation to look up the
	       value in the intput map corresponding to each pixel center
	       in the output map. The spline order is controlled with the
	       "order" argument. Overall "harm" is best suited for normal
	       sky maps, while "spline" with order = 0 or 1 is best suited
	       for hitcount maps and masks.
	order: The spline order to use when method="spline".
	       0 corresponds to nearest neighbor interpolation.
	       1 corresponds to bilinear interpolation (default)
	       Higher order interpolation is not supported - use
	       method="harm" for that.
	extensive: Whether the map represents an extensive (as opposed to
	       intensive) quantity. Extensive quantities have values
	       proportional to the pixel size, unlike intensive quantities.
	       Hitcount per pixel is an extensive quantity. Hitcount per
	       square degree is an intensive quantity, as is a temperature
	       map. Defaults to False.
	bsize: The spline method operates on batches of pixels to save memory.
	       This controls the batch size, in pixels. Defaults to 100000.
	verbose: Whether to print information about what it's doing.
	       Defaults to False, which doesn't print anything.

	Typical usage:
	* map  = healpix2map(map_healpix,  shape, wcs, rot="gal,cel")
	* ivar = healpix2map(ivar_healpix, shape, wcs, rot="gal,cel", method="spline", spin=[0], extensive=True)
	"""
	iheal    = np.asarray(iheal)
	npix     = iheal.shape[-1]
	nside    = curvedsky.npix2nside(npix)
	ipixsize = 4*np.pi/npix
	if out is None:
		out = enmap.zeros(iheal.shape[:-1]+shape[-2:], wcs, dtype=iheal.dtype)
	else: shape, wcs = out.geometry
	if lmax is None: lmax = 3*nside
	if method in ["harm", "harmonic"]:
		# Harmonic interpolation preserves the power spectrum, but can introduce ringing.
		# Probably not a good choice for positive-only quantities like hitcounts.
		# Coordinate rotation is slow.
		alm = curvedsky.map2alm_healpix(iheal, lmax=lmax, spin=spin, niter=niter)
		if rot is not None:
			curvedsky.rotate_alm(alm, *rot2euler(rot), inplace=True)
		curvedsky.alm2map(alm, out, spin=spin)
		del alm
	elif method == "spline":
		# Covers linear interpolation (order=1) and nearest neighbor (order=0).
		# Coordinate rotation may be slow.
		import healpy
		if order > 1:
			raise ValueError("Only order 0 and order 1 spline interpolation supported from healpix maps")
		# Figure out if we need to compute polarization rotations
		pol  = iheal.ndim > 1 and any([s != 0 for s,c1,c2 in enmap.spin_helper(spin, iheal.shape[-2])])
		# Batch to save memory
		brow = (bsize+out.shape[-1]-1)//out.shape[-1]
		for i1 in range(0, out.shape[-2], brow):
			i2   = min(i1+brow, out.shape[-2])
			pos  = out[...,i1:i2,:].posmap().reshape(2,-1)[::-1]
			if rot is not None:
				# Not sure why the [::-1] is necessary here. Maybe psi,theta,phi vs. phi,theta,psi?
				pos = coordinates.transform_euler(inv_euler(rot2euler(rot))[::-1], pos, pol=pol)
			pos[1] = np.pi/2 - pos[1]
			if order == 0:
				# Nearest neighbor. Just read off from the pixels
				vals = iheal[...,healpy.ang2pix(nside, pos[1], pos[0])]
			else:
				# Bilinear interpolation. healpy only supports one component at a time, so loop
				vals = np.zeros(iheal.shape[:-1]+pos.shape[-1:], iheal.dtype)
				for I in utils.nditer(iheal.shape[:-1]):
					vals[I] = healpy.get_interp_val(iheal[I], pos[1], pos[0])
			if rot is not None and iheal.ndim > 1:
				# Update the polarization to account for the new coordinate system
				for s, c1, c2 in enmap.spin_helper(spin, iheal.shape[-2]):
					vals = enmap.rotate_pol(vals, -pos[2], spin=s, comps=[c1,c2-1], axis=-2)
			out[...,i1:i2,:] = vals.reshape(vals.shape[:-1]+(i2-i1,-1))
	else:
		raise ValueError("Map reprojection method '%s' not recognized" % str(method))
	if extensive:
		out *= out.pixsizemap(broadcastable=True)/ipixsize
	return out

def rot2euler(rot):
	"""Given a coordinate rotation description, return the [rotz,roty,rotz] euler
	angles it corresponds to. The rotation desciption can either be those angles
	directly, or a string of the form isys,osys"""
	gal2cel = np.array([57.06793215,  62.87115487, -167.14056929])*utils.degree
	if isinstance(rot, basestring):
		try: isys, osys = rot.split(",")
		except ValueError:
			raise ValueError("Rotation string must be of form 'isys,osys', but got '%s'" % str(rot))
		R = spatial.transform.Rotation.identity()
		# Handle input system
		if   isys in ["cel","equ"]: pass
		elif isys == "gal": R *= spatial.transform.Rotation.from_euler("zyz", gal2cel)
		else: raise ValueError("Unrecognized system '%s'" % isys)
		# Handle output system
		if   osys in ["cel","equ"]: pass
		elif osys == "gal": R *= spatial.transform.Rotation.from_euler("zyz", gal2cel).inv()
		else: raise ValueError("Unrecognized system '%s'" % osys)
		return R.as_euler("zyz")
	else:
		rot = utils.asfarray(rot)
		return rot

def inv_euler(euler): return [-euler[2], -euler[1], -euler[0]]

def restrict_nside(nside, mode="mul32", round="ceil"):
	"""Given an arbitrary Healpix nside, return one that's restricted in
	various ways according to the "mode" argument:

	"pow2":  Restrict to a power of 2. This is required for compatibility
	 with the rarely used "nest" pixel ordering in Healpix, and is the standard
	 in the Healpix world.
	"mul32": Restrict to multiple of 32, unless 12*nside**2<=1024.
	 This is enough to make the maps writable by healpy.
	"any":   No restriction

	The "round" argument controls how any rounding is done. This can be one
	of the strings "ceil" (default), "round" or "floor", or you can pass in
	a custom function(nside) -> nside.

	In all cases, the final nside is converted to an integer and capped to
	1 below.
	"""
	if isinstance(round, basestring):
		round = {"floor":np.floor, "round":np.round, "ceil":np.ceil}[round]
	if   mode == "any": nside = round(nside)
	elif mode == "mul32":
		if 12*nside**2 > 1024:
			nside = round(nside/32)*32
	elif mode == "pow2":
		nside = 2**round(np.log2(nside))
	else:
		raise ValueError("Unrecognized nside mode '%s'" % str(mode))
	nside = max(1,int(nside))
	return nside


################################
####### Old stuff below ########
################################

def centered_map(imap, res, box=None, pixbox=None, proj='car', rpix=None,
				 width=None, height=None, width_multiplier=1.,
				 rotate_pol=True, **kwargs):
	raise NotImplementedError("This function has been removed.")
	
def healpix_from_enmap_interp(imap, **kwargs):
	raise RuntimeError("This function has been removed. Use reproject.map2healpix(...method='spline').")


def healpix_from_enmap(imap, lmax, nside):
	raise RuntimeError("This function has been removed. Use reproject.map2healpix(...method='harm').")



def enmap_from_healpix(hp_map, shape, wcs, ncomp=1, unit=1, lmax=0,
					   rot="gal,equ", first=0, is_alm=False, return_alm=False, f_ell=None):
	raise RuntimeError("This function has been removed. Use reproject.healpix2map(...method='harm').")


def enmap_from_healpix_interp(hp_map, shape, wcs , rot="gal,equ",
							  interpolate=False):
	raise RuntimeError("This function has been removed. Use reproject.healpix2map(...method='spline').")



def ivar_hp_to_cyl(hmap, shape, wcs, rot=False,do_mask=True,extensive=True):
	raise NotImplementedError("This function has been removed.")

# Helper functions


def gnomonic_pole_wcs(shape, res):
	raise NotImplementedError("This function has been removed.")
 
def gnomonic_pole_geometry(width, res, height=None):
	raise NotImplementedError("This function has been removed.")


def rotate_map(imap, shape_target=None, wcs_target=None, shape_source=None,
			   wcs_source=None, pix_target=None, **kwargs):
	raise NotImplementedError("This function has been removed.")


def get_rotated_pixels(shape_source, wcs_source, shape_target, wcs_target,
					   inverse=False, pos_target=None,
					   center_target=None, center_source=None):
	raise NotImplementedError("This function has been removed.")

def cutout(imap, width=None, ra=None, dec=None, pad=1, corner=False,
		   res=None, npix=None, return_slice=False,sindex=None):
	raise NotImplementedError("This function has been removed.")


def rect_box(width, center=(0., 0.), height=None):
	raise NotImplementedError("This function has been removed.")

def get_pixsize_rect(shape, wcs):
	raise NotImplementedError("This function has been removed.")

def rect_geometry(width, res, height=None, center=(0., 0.), proj="car"):
	raise NotImplementedError("This function has been removed.")


def distribute(N,nmax):
	"""
	Distribute N things into cells as equally as possible such that 
	no cell has more than nmax things.
	"""
	actual_max = int(2.*(nmax+1)/3.)
	numcells = int(round(N*1./actual_max))
	each_cell = [actual_max]*(numcells-1)
	rem = N-sum(each_cell)
	if rem>0: each_cell.append(rem)
	assert sum(each_cell)==N
	return each_cell

def populate(shape,wcs,ofunc,maxpixy = 400,maxpixx = 400):
	"""
	Loop through tiles in a new map of geometry (shape,wcs)
	with tiles that have maximum allowed shape (maxpixy,maxpixx)
	such that each tile is populated with the result of
	ofunc(oshape,owcs) where oshape,owcs is the geometry of each
	tile.
	"""
	omap = enmap.zeros(shape,wcs)
	Ny,Nx = shape[-2:]
	tNys = distribute(Ny,maxpixy)
	tNxs = distribute(Nx,maxpixx)
	numy = len(tNys)
	numx = len(tNxs)
	sny = 0
	ntiles = numy*numx
	print("Number of tiles = ",ntiles)
	done = 0
	for i in range(numy):
		eny = sny+tNys[i]
		snx = 0
		for j in range(len(tNxs)):
			enx = snx+tNxs[j]
			sel = np.s_[...,sny:eny,snx:enx]
			oshape,owcs = enmap.slice_geometry(shape,wcs,sel)
			omap[sel] = ofunc(oshape,owcs)
			snx += tNxs[j]
			done += 1
		sny += tNys[i]
		print(done , " / ", ntiles, " tiles done...")
	return omap


def postage_stamp(inmap, ra_deg, dec_deg, width_arcmin,
		  res_arcmin, proj='gnomonic', return_cutout=False,
		  npad=3, rotate_pol=True, **kwargs):
	raise RuntimeError("postage_stamp has been removed. Please use thumbnails instead.")

