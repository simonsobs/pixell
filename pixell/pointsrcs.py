"""Point source parameter I/O. In order to simulate a point source as it appears on
the sky, we need to know its position, amplitude and local beam shape (which can
also absorb an extendes size for the source, as long as it's gaussian). While other
properties may be nice to know, those are the only ones that matter for simulating
it. This module provides functions for reading these minimal parameters from
various data files.

The standard parameters are [nsrc,nparam]:
	dec (radians)
	ra (radians)
	[T,Q,U] amplitude at center of gaussian (uK)
	beam sigma (wide  axis) (radians)
	beam sigma (short axis) (radians)
	beam orientation (wide axis from dec axis)  (radians)

What do I really need to simulate a source?

1. Physical source on the sky (pos,amps,shape)
2. Telescope response (beam in focalplane)

For a point source 1.shape would be a point. But clusters and
nearby galaxies can have other shapes. In general many profiles are
possible. Parametrizing them in a standard format may be difficult.
"""
from __future__ import print_function, division
import numpy as np, time
from astropy.io import fits
from scipy import spatial
from . import utils, enmap, srcsim, wcsutils

#### Map-space source simulation ###

# New version. Usually 10 or more times faster than the old one, and with less memory
# overhead. But interface is a bit different, so it has a new name.
def sim_objects(shape, wcs, poss, amps, profile, prof_ids=None, omap=None, vmin=None, rmax=None,
		op="add", pixwin=False, pixwin_order=0, separable="auto", transpose=False, prof_equi="auto", cache=None,
		return_times=False):
	"""Simulate radially symmetric objects with arbitrary profiles and amplitudes.
	Arguments:
	* shape, wcs: The geometry of the patch to simulate. Only shape[-2:]
	  is used. amps determines the pre-dimensions
	* poss: The positions of the objects. [{dec,ra},nobj] in radians.
	* amps: The central amplitudes of the objects. [...,nobj]. Not the same as the flux.
	* profile: The profiles to use. Either [{r,b(r)},nsamp] (with shape (2,nsamp)) or a
	  list of such, where nsamp is the size of r and b(r).  If providing a list for
	  nobj objects, the shape of the array passed is (nobj,2,nsamp) and prof_ids
	  should be np.arange(nobj).

	Optional arguments:
	* prof_ids: Which profile to use for each source. Defaults to use
	  the first profile for all. Only necessary to specify if you want to
	  simulate objects with varying profiles. If specified, it should be
	  [nobj] indices into the profile list.
	* omap: Update this map instead of constructing a new one.
	  MUST BE float32 AND C CONTIGUOUS and have shape [...,ny,nx] where ... matches amps.
	* vmin: The lowest values to bother simulating. To avoid being terribly slow,
	  profiles aren't evaluated out to infinite distance, but only once they drop down
	  to a sufficiently low level, given by vmin. This takes into account the peak
	  amplitud of each object, so it should be in map units. For example, it might
	  be reasonable to have a vmin a few times lower than the noise level of the map,
	  e.g. 0.1 µK. If not specified, then it defaults to min(abs(amps))*1e-3.
	* rmax: The maximum radius to use, in radians. Acts as a cap on the radius
	  calculated from vmin. Not applied if None or 0.
	* op: What operation to use when combining the input map with each object.
	  "add": Add linearly [default]
	  "max": Keep the max value in each pixel
	  "min": Keep the min value in each pixel
	* pixwin: Whether to apply a pixel window after simulating the objects. This
	  assumes periodic boundary consitions, so objects at the very edge will be
	  wrong.
	* separable: Whether the coordinate system's coordinate axes are indpendent,
	  such that one only needs to know y in order to calculate dec, and x to
	  calculate ra. This allows for much faster calculation of the pixel
	  coordinates. Default "auto": True for cylindrical coordinates, False otherwise.
	* cache: Dictionary to use for caching pixel coordinates. Can be useful
	  if you're doing repeated simulations on the same geometry with non-separable
	  geometry, to avoid having to recalculate the pixel coordinates all the time.

	Returns the resulting map. If omap was specified, then the same object will
	be returned (after being updated of course). In this case, the simulated
	sources will have been added (or maxed etc. depending on op) into the map.
	Otherwise, the only signal in the map will be the objects."""
	dtype = np.float32 # C extension only supports this dtype
	if separable == "auto": separable = wcsutils.is_separable(wcs)
	# Object positions
	obj_decs = np.asanyarray(poss[0], dtype=dtype, order="C")
	obj_ras  = np.asanyarray(poss[1], dtype=dtype, order="C")
	obj_ys, obj_xs = utils.nint(enmap.sky2pix(shape, wcs, poss)).astype(np.int32)
	assert obj_decs.ndim == 1 and obj_ras.ndim == 1, "poss must be [{dec,ra},nobj]"
	nobj     = len(obj_decs)
	# Object amplitudes, and number of components
	amps     = np.asanyarray(amps, dtype=dtype, order="C")
	pre      = amps.shape[:-1]
	amps_flat= amps.reshape(-1, amps.shape[-1])
	ncomp    = len(amps_flat)
	# Profiles. For point sources this would just be the beam, but extended objects
	# can have differnet profiles. We will support both [{r,b}] and [[{r,b}],[{r,b}],...]
	try: profile[0][0][0]
	except (TypeError, IndexError): profile = [profile]
	profile = [np.asanyarray(p, dtype=dtype, order="C") for p in profile]
	# If which profile to use isn't specified, default to the first one
	if prof_ids is None: prof_ids = np.zeros(nobj, np.int32)
	else: prof_ids = np.asanyarray(prof_ids, dtype=np.int32, order="C")
	if prof_equi == "auto": prof_equi = all([is_equi(prof[0]) for prof in profile])
	# If user hasn't specified how faint things to simulate, then set the limit a bit
	# below the peak of the faintest object
	if vmin is None: vmin = np.min(np.abs(amps))*1e-3
	if rmax is None: rmax = 0
	# Set up the pixel coordinates
	if separable: posmap = utils.cache_get(cache, "posmap", lambda: enmap.posaxes(shape, wcs, dtype=dtype))
	else:         posmap = utils.cache_get(cache, "posmap", lambda: enmap.posmap (shape, wcs, dtype=dtype))
	# Set up our output map
	if omap is None: omap_flat = enmap.zeros((ncomp,)+shape[-2:], wcs, dtype)
	else:            omap_flat = omap.preflat
	assert omap_flat.dtype == dtype, "omap.dtype must be np.float32"
	assert omap_flat.shape == (ncomp,)+shape[-2:], "omap must be [...,ny,nx], where [ny,nx] agrees with shape, and ... agrees with amps"
	# Whew! Actually do the work
	times = srcsim.sim_objects(omap_flat, obj_decs, obj_ras, obj_ys, obj_xs, amps_flat, profile, prof_ids, posmap, vmin, rmax=rmax, separable=separable, transpose=transpose, prof_equi=prof_equi, return_times=True)[1]
	omap = omap_flat.reshape(pre+shape[-2:])
	# NB! Since we're not padding, this fourier operation will have problems at the edges
	if pixwin: omap = enmap.apply_window(omap, order=pixwin_order)
	return (omap, times) if return_times else omap

def is_equi(r):
	"""Estimate whether the values r[:] = arange(n)*delta, allowing for
	fast index calculations. This is just a heuristic, but it is hopefully
	reliable enough."""
	return len(r) > 1 and r[0] == 0 and np.allclose(r[-1],(len(r)-1)*r[1])

def radial_sum(map, poss, bins, oprofs=None, separable="auto",
		prof_equi="auto", cache=None, return_times=False):
	"""Sum the signal in map into radial bins around a set of objects,
	returning one radial sum-profile per object.
	Arguments:
	* map: The map to read data from. [...,ny,nx]
	* poss: The positions of the objects. [{dec,ra},nobj] in radians.
	* bins: The bin edges. [nbin+1]. Faster if equi-spaced with first at 0

	Optional arguments:
	* oprofs: [obj,...,nbin] array to write result to. MUST BE float32 AND C CONTIGUOUS
	* separable: Whether the coordinate system's coordinate axes are indpendent,
	  such that one only needs to know y in order to calculate dec, and x to
	  calculate ra. This allows for much faster calculation of the pixel
	  coordinates. Default "auto": True for cylindrical coordinates, False otherwise.
	* cache: Dictionary to use for caching pixel coordinates. Can be useful
	  if you're doing repeated simulations on the same geometry with non-separable
	  geometry, to avoid having to recalculate the pixel coordinates all the time.

	Returns the resulting profiles. If oprof was specified, then the same object will
	be returned (after being updated of course)."""
	dtype = np.float32 # C extension only supports this dtype
	if separable == "auto": separable = wcsutils.is_separable(map.wcs)
	# Object positions
	obj_decs = np.asanyarray(poss[0], dtype=dtype, order="C")
	obj_ras  = np.asanyarray(poss[1], dtype=dtype, order="C")
	obj_ys, obj_xs = utils.nint(map.sky2pix(poss)).astype(np.int32)
	assert obj_decs.ndim == 1 and obj_ras.ndim == 1, "poss must be [{dec,ra},nobj]"
	nobj     = len(obj_decs)
	# map and number of components
	pre      = map.shape[:-2]
	map_flat = map.preflat
	ncomp    = len(map_flat)
	# bins
	bins     = np.asarray(bins, dtype=dtype)
	nbin     = len(bins)-1
	prof_equi = is_equi(bins) if prof_equi == "auto" else prof_equi
	# Set up the pixel coordinates
	if separable: posmap = utils.cache_get(cache, "posmap", lambda: map.posaxes(dtype=dtype))
	else:         posmap = utils.cache_get(cache, "posmap", lambda: map.posmap (dtype=dtype))
	# Set up our output
	if oprofs is None: oprofs = np.zeros((nobj,)+pre+(nbin,),dtype)
	oprofs_flat   = oprofs.reshape(nobj,ncomp,nbin)
	assert oprofs_flat.dtype == dtype, "oprofs.dtype must be np.float32"
	# Whew! Actually do the work
	times = srcsim.radial_sum(map_flat, obj_decs, obj_ras, obj_ys, obj_xs, bins, posmap, profs=oprofs_flat, separable=separable, prof_equi=prof_equi, return_times=True)[1]
	return (oprofs, times) if return_times else oprofs

def radial_bin(map, poss, bins, weights=None, separable="auto",
		prof_equi="auto", cache=None, return_times=False):
	"""Average the signal in map into radial bins for a set of objects, returning
	a radial profile for each object.
	Arguments:
	* map: The map to read data from. [...,ny,nx]
	* poss: The positions of the objects. [{dec,ra},nobj] in radians.
	* bins: The bin edges. [nbin+1]. Faster if equi-spaced with first at 0

	Optional arguments:
	* oprofs: [obj,...,nbin] array to write result to. MUST BE float32 AND C CONTIGUOUS
	* separable: Whether the coordinate system's coordinate axes are indpendent,
	  such that one only needs to know y in order to calculate dec, and x to
	  calculate ra. This allows for much faster calculation of the pixel
	  coordinates. Default "auto": True for cylindrical coordinates, False otherwise.
	* cache: Dictionary to use for caching pixel coordinates. Can be useful
	  if you're doing repeated simulations on the same geometry with non-separable
	  geometry, to avoid having to recalculate the pixel coordinates all the time.

	Returns the resulting profiles. If oprof was specified, then the same object will
	be returned (after being updated of course)."""
	if weights is not None: map = map*weights
	profs, times1 = radial_sum(map, poss, bins, separable=separable, prof_equi=prof_equi,
			cache=cache, return_times=True)
	if weights is None: weights = enmap.ones(map.shape[-2:], map.wcs, map.dtype)
	div, times2 = radial_sum(weights, poss, bins, separable=separable, prof_equi=prof_equi,
			cache=cache, return_times=True)
	# Add dimensions to div if necessary to make them broadcast.
	# For more complex broadcasting, prepare the shapes of map and weights manually
	div = div.reshape(profs.shape[:1]+(1,)*(profs.ndim-div.ndim)+profs.shape[-1:])
	profs /= div
	times = np.concatenate([times1,times2])
	return (profs, times) if return_times else profs

def sim_srcs(shape, wcs, srcs, beam, omap=None, dtype=None, nsigma=5, rmax=None, vmin=None, smul=1,
		return_padded=False, pixwin=False, pixwin_order=0, op=np.add, wrap="auto", verbose=False, cache=None,
		separable="auto", method="c"):
	"""Backwards compatibility wrapper that exposes the speed of the new sim_objects
	function using the old sim_srcs interface. For most users this should result in a
	transparent speedup of O(10x), but sim_objects does not implement 100% of the sim_srcs
	functionality, so the old python method is also available by specifying method = "python".

	Limitations of the new version:
	* only float32, C-contiguous maps supported
	* smul not supported
	* padding not supported, which impacts objects at the very edge of the map if
	  pixwin is used
	* only add, max and min supported for 'op'

	Unlike sim_srcs, sim_objects supports simulating objects with multiple
	different profiles at once, but this functionality isn't available through
	the sim_srcs interface.

	I recommend using sim_objects directly instead of relying on this wrapper in most cases.
	"""
	if method in ["c", "C"]:
		assert dtype is None or dtype == np.float32, "method 'c' only supports float32. Use method='python' if you need others"
		assert smul == 1, "method 'c' does not support smul != 1. Use method='python' if you need this"
		if vmin is None: vmin = np.exp(-0.5*nsigma**2)
		if   op == np.add or op == np.ndarray.__iadd__ or op == "add": op_ = "add"
		elif op == np.max or op == "max": op_ = "max"
		elif op == np.min or op == "min": op_ = "min"
		else: raise ValueError("method 'c' only supports op add, max or min. Use method='python' if you need more")
		if return_padded: raise ValueError("method 'c' does not support return_padded. Use method=='python' if you need this")
		ncomp = np.prod(shape[:-2], dtype=int)
		nobj  = len(srcs)
		poss  = srcs.T[:2]
		amps  = np.zeros((ncomp,nobj), np.float32)
		amps[:srcs.shape[1]-2] = srcs.T[2:2+ncomp]
		amps  = amps.reshape(shape[:-2]+(nobj,))
		# Handle beam = float, in which case a gaussian beam is made
		beam  = expand_beam(beam, nsigma, rmax)
		return sim_objects(shape, wcs, poss, amps, beam, omap=omap, vmin=vmin, rmax=rmax, op=op, pixwin=pixwin, pixwin_order=0, separable=separable, cache=cache)
	elif method == "python":
		return sim_srcs_python(shape, wcs, srcs, beam, omap=omap, dtype=dtype, nsigma=nsigma, rmax=rmax, smul=smul, return_padded=return_padded, pixwin=pixwin, pixwin_order=pixwin_order, op=op, wrap=wrap, verbose=verbose, cache=cache, separable=separable)

def sim_srcs_python(shape, wcs, srcs, beam, omap=None, dtype=None, nsigma=5, rmax=None, smul=1,
		return_padded=False, pixwin=False, pixwin_order=0, op=np.add, wrap="auto", verbose=False, cache=None,
		separable="auto"):
	"""Simulate a point source map in the geometry given by shape, wcs
	for the given srcs[nsrc,{dec,ra,T...}], using the beam[{r,val},npoint],
	which must be equispaced. If omap is specified, the sources will be
	added to it in place. All angles are in radians. The beam is only evaluated up to
	the point where it reaches exp(-0.5*nsigma**2) unless rmax is specified, in which
	case this gives the maximum radius. smul gives a factor to multiply the resulting
	source model by. This is mostly useful in conction with omap.

	The source simulation is sped up by using a source lookup grid.
	"""
	if separable == "auto": separable = wcsutils.is_separable(wcs)
	if omap is None: omap = enmap.zeros(shape, wcs, dtype)
	ishape = omap.shape
	omap   = omap.preflat
	ncomp  = omap.shape[0]
	srcs   = np.asarray(srcs)
	# Set up wrapping
	if utils.streq(wrap, "auto"):
		wrap = [0, utils.nint(360./wcs.wcs.cdelt[0])]
	# In keeping with the rest of the functions here, srcs is [nsrc,{dec,ra,T,Q,U}].
	# The beam parameters are ignored - the beam argument is used instead
	amps = srcs[:,2:2+ncomp]
	poss = srcs[:,:2].copy()
	# Rewind positions to let us use flat-sky approximation for distance calculations
	ref  = np.mean(enmap.box(shape, wcs, corner=False)[:,1])
	poss[:,1] = utils.rewind(poss[:,1], ref)
	beam = expand_beam(beam, nsigma, rmax)
	if not rmax: rmax = nsigma2rmax(beam, nsigma)
	# Pad our map by rmax, so we get the contribution from sources
	# just ourside our area. We will later split our map into cells of size cres. Let's
	# adjust the padding so we have a whole number of cells
	minshape = np.min(omap[...,5:-5:10,5:-5:10].pixshapemap()/10,(-2,-1))
	cres = np.maximum(1,utils.nint(rmax/minshape))
	epix = cres-(omap.shape[-2:]+2*cres)%cres
	padding = [cres,cres+epix]
	wmap, wslice  = enmap.pad(omap, padding, return_slice=True)
	# Overall we will have this many grid cells
	cshape = wmap.shape[-2:]//cres
	# Find out which sources matter for which cells
	srcpix = wmap.sky2pix(poss.T).T
	pixbox= np.array([[0,0],wmap.shape[-2:]],int)
	nhit, cell_srcs = build_src_cells(pixbox, srcpix, cres, wrap=wrap)
	# Optionally cache the posmap
	if cache is None or cache[0] is None: posmap = wmap.posmap(separable=separable)
	else: posmap = cache[0]
	if cache is not None: cache[0] = posmap
	model = eval_srcs_loop(posmap, poss, amps, beam, cres, nhit, cell_srcs, dtype=wmap.dtype, op=op, verbose=verbose)
	del posmap
	if pixwin: model = enmap.apply_window(model, order=pixwin_order)
	# Update our work map, through our view
	if smul != 1: model *= smul
	wmap   = op(wmap, model, wmap)
	if not return_padded:
		# Copy out
		omap[:] = wmap[wslice]
		# Restore shape
		omap = omap.reshape(ishape)
		return omap
	else:
		return wmap.reshape(ishape[:-2]+wmap.shape[-2:]), wslice

def eval_srcs_loop(posmap, poss, amps, beam, cres, nhit, cell_srcs, dtype=np.float64, op=np.add, verbose=False):
	# Loop through each cell
	ncy, ncx = nhit.shape
	model = enmap.zeros(amps.shape[-1:]+posmap.shape[-2:], posmap.wcs, dtype)
	for cy in range(ncy):
		for cx in range(ncx):
			nsrc = nhit[cy,cx]
			if verbose and nsrc > 0: print("map cell %5d/%d with %5d srcs" % (cy*ncx+cx+1, ncy*ncx, nsrc))
			if nsrc == 0: continue
			srcs  = cell_srcs[cy,cx,:nsrc]
			y1,y2 = (cy+0)*cres[0], (cy+1)*cres[0]
			x1,x2 = (cx+0)*cres[1], (cx+1)*cres[1]
			pixpos = posmap[:,y1:y2,x1:x2]
			srcpos = poss[srcs].T # [2,nsrc]
			srcamp = amps[srcs].T # [ncomp,nsrc]
			r      = utils.angdist(pixpos[::-1,None,:,:],srcpos[::-1,:,None,None])
			bpix   = (r - beam[0,0])/(beam[0,1]-beam[0,0])
			# Evaluate the beam at these locations
			bval   = utils.interpol(beam[1], bpix[None], mode="constant", order=1, mask_nan=False) # [nsrc,ry,rx]
			cmodel = srcamp[:,:,None,None]*bval
			cmodel = op.reduce(cmodel,-3)
			op(model[:,y1:y2,x1:x2], cmodel, model[:,y1:y2,x1:x2])
	return model

def sim_srcs_dist_transform(shape, wcs, srcs, beam, omap=None, dtype=None, nsigma=4, rmax=None, smul=1,
		pixwin=False, ignore_outside=False, op=np.add, verbose=False):
	"""Simulate a point source map in the geometry given by shape, wcs
	for the given srcs[nsrc,{dec,ra,T...}], using the beam[{r,val},npoint],
	which must be equispaced. Unlike sim_srcs, overalpping point sources are not supported.
	If omap is specified, the sources will be
	added to it in place. All angles are in radians. The beam is only evaluated up to
	the point where it reaches exp(-0.5*nsigma**2) unless rmax is specified, in which
	case this gives the maximum radius. smul gives a factor to multiply the resulting
	source model by. This is mostly useful in conction with omap.
	"""
	if omap is None: omap = enmap.zeros(shape, wcs, dtype)
	ishape = omap.shape
	omap   = omap.preflat
	ncomp  = omap.shape[0]
	# In keeping with the rest of the functions here, srcs is [nsrc,{dec,ra,T,Q,U}].
	# The beam parameters are ignored - the beam argument is used instead
	amps = srcs[:,2:2+ncomp]
	poss = srcs[:,:2].copy()
	br, bv = expand_beam(beam, nsigma, rmax).copy()
	rmax = nsigma2rmax(beam, nsigma)
	if ignore_outside:
		pixs = enmap.sky2pix(shape, wcs, poss.T[:2])
		pmax = 2*rmax/enmap.pixsize(shape, wcs)**0.5
		bad  = np.any(pixs <  -pmax, 0) | np.any(pixs > np.array(shape[-2:])[:,None]+pmax, 0)
		amps, poss = amps[~bad], poss[~bad]
	r, domains = enmap.distance_from(shape, wcs, poss.T[:2], domains=True, rmax=rmax)
	np.clip(domains, 0, amps.shape[0]-1, domains)
	for i in range(ncomp):
		op(omap[i], np.interp(r, br, bv, right=0) * amps[domains,i], omap[i])
	omap = omap.reshape(ishape)
	return omap

def expand_beam(beam, nsigma=5, rmax=None, nper=400):
	beam = np.asarray(beam)
	if beam.ndim == 0:
		# Build gaussian beam
		sigma = beam.reshape(-1)[0]
		if rmax is None: rmax = sigma*nsigma
		r = np.linspace(0,rmax,nsigma*nper)
		return np.array([r,np.exp(-0.5*(r/sigma)**2)])
	elif beam.ndim == 2:
		return beam
	else: raise ValueError

def nsigma2rmax(beam, nsigma):
	return beam[0,np.where(beam[1] >= np.exp(-0.5*nsigma**2))[0][-1]]

def build_src_cells(cbox, srcpos, cres, unwind=False, wrap=None):
	# srcpos is [nsrc,...,{dec,ra}]. Reshape to 3d to keep things simple.
	# will reshape back when returning
	cbox    = np.asarray(cbox)
	srcpos  = np.asarray(srcpos)
	ishape  = srcpos.shape
	srcpos  = srcpos.reshape(ishape[0],-1,ishape[-1])

	cshape  = tuple(np.ceil(((cbox[1]-cbox[0])/cres)).astype(int))
	if unwind:
		# Make the sources' ra compatible with our area
		ref     = np.mean(cbox[:,1],0)
		srcpos[:,...,1] = utils.rewind(srcpos[:,...,1], ref)
	# How big must our cell hit array be?
	nmax = max(1,np.max(build_src_cells_helper(cbox, cshape, cres, srcpos, wrap=wrap)))
	ncell, cells = build_src_cells_helper(cbox, cshape, cres, srcpos, nmax, wrap=wrap)
	# Reshape back to original shape
	ncell = ncell.reshape(ishape[1:-1]+ncell.shape[1:])
	cells = cells.reshape(ishape[1:-1]+cells.shape[1:])
	return ncell, cells

def build_src_cells_helper(cbox, cshape, cres, srcpos, nmax=0, wrap=None):
	# A cell is hit if it overlaps both horizontally and vertically
	# with the point source +- cres
	nsrc, nmid = srcpos.shape[:2]
	# ncell is [:,ncy,ncx]
	ncell = np.zeros((nmid,)+cshape,np.int32)
	if nmax > 0:
		cells = np.zeros((nmid,)+cshape+(nmax,),np.int32)
	c0 = cbox[0]; inv_dc = cshape/(cbox[1]-cbox[0]).astype(float)
	# Set up wrapping. The woff variable will contain the set of coordinate offsets we will try
	if wrap is None: wrap = [0,0]
	woffs = []
	for i, w in enumerate(wrap):
		if w == 0: woffs.append([0])
		else: woffs.append([-w,0,+w])
	for si in range(nsrc):
		for mi in range(nmid):
			pos = srcpos[si,mi]
			for woffy in woffs[0]:
				for woffx in woffs[1]:
					wpos = pos[:2] + np.array([woffy,woffx])
					i1   = (wpos[:2]-cres-c0)*inv_dc
					i2   = (wpos[:2]+cres-c0)*inv_dc+1 # +1 because this is a half-open interval
					# Don't try to update out of bounds
					i1 = np.maximum(i1.astype(int), 0)
					i2 = np.minimum(i2.astype(int), np.array(cshape))
					# Skip sources that don't hit our area at all
					if np.any(i1 >= cshape) or np.any(i2 < 0): continue
					for cy in range(i1[0],i2[0]):
						for cx in range(i1[1],i2[1]):
							if nmax > 0:
								cells[mi,cy,cx,ncell[mi,cy,cx]] = si
							ncell[mi,cy,cx] += 1
	if nmax > 0: return ncell, cells
	else: return ncell

def cellify(map, res):
	"""Given a map [...,ny,nx] and a cell resolution [ry,rx], return map
	reshaped into a cell grid [...,ncelly,ncellx,ry,rx]. The map will be
	truncated if necessary"""
	res    = np.array(res,int)
	cshape = map.shape[-2:]//res
	omap   = map[...,:cshape[0]*res[0],:cshape[1]*res[1]]
	omap   = omap.reshape(omap.shape[:-2]+(cshape[0],res[0],cshape[1],res[1]))
	omap   = utils.moveaxis(omap, -3, -2)
	return omap

def uncellify(cmap):
	omap = utils.moveaxis(cmap, -2, -3)
	omap = omap.reshape(omap.shape[:-4]+(omap.shape[-4]*omap.shape[-3],omap.shape[-2]*omap.shape[-1]))
	return omap

#### Cross-matching ####

def crossmatch(srcs1, srcs2, tol=1*utils.arcmin, safety=4):
	"""Cross-match two source catalogs based on position. Each
	source in one catalog is associated with the closest source
	in the other catalog, as long as the distance between them is
	less than the tolerance. The catalogs must be [:,{ra,dec,...}]
	in radians. Returns [nmatch,2], with the last index giving
	the index in the first and second catalog for each match."""
	vec1 = utils.ang2rect(srcs1[:,:2], axis=1)
	vec2 = utils.ang2rect(srcs2[:,:2], axis=1)
	tree1 = spatial.cKDTree(vec1)
	tree2 = spatial.cKDTree(vec2)
	groups = tree1.query_ball_tree(tree2, tol*safety)
	matches = []
	for gi, group in enumerate(groups):
		if len(group) == 0: continue
		# Get the true distance to each member in the group
		group = np.array(group)
		dists = utils.vec_angdist(vec1[gi,None,:], vec2[group,:], axis=1)
		best  = np.argmin(dists)
		if dists[best] > tol: continue
		matches.append([gi, group[best]])
	matches = np.array(matches)
	return matches

#### Source parameter I/O ####

# These functions are messy and should be cleaned up before general consumption
# Be warned if you choose to use them - they might be removed or changed in future
# releases.

def read(fname, format="auto"):
	if format == "auto": formats = ["dory_fits","dory_txt","fits","nemo","simple"]
	else:                formats = [format]
	for f in formats:
		try:
			if   f == "dory_fits":return read_dory_fits(fname)
			elif f == "dory_txt": return read_dory_txt(fname)
			elif f == "fits":     return read_fits(fname)
			elif f == "nemo":     return read_nemo(fname)
			elif f == "simple":   return read_simple(fname)
			else: raise ValueError("Unrecognized point source format '%s' for file '%s'" % (f, fname))
		except (ValueError, IOError, OSError) as e: pass
	raise IOError("Unable to read point source file '%s' with format '%s'" % (fname, f))

def read_nemo(fname):
	"""Reads the nemo ascii catalog format, and returns it as a recarray."""
	idtype = [("name","2S64"),("ra","d"),("dec","d"),("snr","d"),("npix","i"),("detfrac","d"),("template","S32"),("glat","d"),("I","d"), ("dI","d")]
	try: icat = np.loadtxt(fname, dtype=idtype)
	except (ValueError, IndexError) as e:
		idtype = [("name","2S64"),("ra","d"),("dec","d"),("snr","d"),("npix","i"),("template","S32"),("glat","d"),("I","d"), ("dI","d")]
		try: icat = np.loadtxt(fname, dtype=idtype)
		except (ValueError, IndexError) as e:
			raise IOError(e.args[0])
	odtype = [("name","S64"),("ra","d"),("dec","d"),("snr","d"),("I","d"),("dI","d"),("npix","i"),("template","S32"),("glat","d")]
	ocat = np.zeros(len(icat), odtype).view(np.recarray)
	ocat.name = np.char.add(*icat["name"].T)
	for t in odtype[1:]: ocat[t[0]] = icat[t[0]]
	return ocat

def read_simple(fname):
	try:
		cat = np.loadtxt(fname, dtype=[("ra","d"),("dec","d"),("I","d"),("dI","d")], usecols=range(4), ndmin=1).view(np.recarray)
	except ValueError:
		try:
			cat = np.loadtxt(fname, dtype=[("ra","d"),("dec","d"),("I","d")], usecols=range(3), ndmin=1).view(np.recarray)
		except ValueError as e:
			raise IOError(e.args[0])
	cat.ra  *= utils.degree
	cat.dec *= utils.degree
	return cat

def read_dory_fits(fname, hdu=1):
	d = fits.open(fname)[hdu].data
	ocat = np.zeros(len(d), dtype=[("ra","d"),("dec","d"),("I","d"),("Q","d"),("U","d")]).view(np.recarray)
	ocat.ra  = d.ra  * utils.degree
	ocat.dec = d.dec * utils.degree
	ocat.I, ocat.Q, ocat.U = d.amp.T*1e3
	return ocat

def read_dory_txt(fname):
	try:
		d = np.loadtxt(fname, usecols=[0,1,3,5,7], dtype=[("ra","d"),("dec","d"),("I","d"),("Q","d"),("U","d")]).view(np.recarray).reshape(-1)
		d.I   *= 1e3; d.Q *= 1e3; d.U *= 1e3
		d.ra  *= utils.degree
		d.dec *= utils.degree
		return d
	except (ValueError, IndexError) as e:
		raise IOError(e.args[0])

def read_fits(fname, hdu=1, fix=True):
	d = fits.open(fname)[hdu].data
	if fix:
		d = translate_dtype_keys(d, {"RADeg":"ra","decDeg":"dec","deltaT_c":"I","err_deltaT_c":"dI"})
	return d.view(np.recarray)

# Sauron catalog formats. I really need to come up with some once-and-for-all catalog format!
def format_sauron(cat):
	nfield, ncomp = cat.flux.shape[-2:]
	names  = "TQU"
	header = "#%8s %8s %9s" % ("ra", "dec", "snr_T")
	for i in range(1,ncomp): header += " %8s" % ("snr_"+names[i])
	for i in range(ncomp):   header += " %8s %7s" % ("ftot_"+names[i], "dftot_"+names[i])
	for i in range(nfield):
		for j in range(ncomp):
			header += " %8s %7s" % ("flux_"+names[j]+"%d"%(i+1), "dflux_"+names[j]+"%d"%(i+1))
	header += " %2s" % "ca"
	for i in range(nfield): header += " %7s" % ("cont_%d" % (i+1))
	header += "\n"
	res = ""
	for i in range(len(cat)):
		res += "%9.4f %8.4f" % (cat.ra[i]/utils.degree, cat.dec[i]/utils.degree)
		snr  = cat.snr[i].reshape(-1)
		res += " %9.2f" % snr[0] + " %7.2f"*(len(snr)-1) % tuple(snr[1:])
		flux = cat. flux_tot[i].reshape(-1)
		dflux= cat.dflux_tot[i].reshape(-1)
		for j in range(len(flux)):
			res += "  %8.2f %7.2f" % (flux[j], dflux[j])
		flux = cat. flux[i].reshape(-1)
		dflux= cat.dflux[i].reshape(-1)
		for j in range(len(flux)):
			res += "  %8.2f %7.2f" % (flux[j], dflux[j])
		try: res += " %2d" % (cat.case[i])
		except (KeyError, AttributeError): pass
		try:
			for j in range(len(cat.contam[i])):
				res += " %7.2f" % (cat.contam[i,j])
		except (KeyError, AttributeError): pass
		res += "\n"
	return header + res

def write_sauron(ofile, cat):
	if ofile.endswith(".fits"): write_sauron_fits(ofile, cat)
	else: write_sauron_txt (ofile, cat)

def read_sauron(ifile):
	if ifile.endswith(".fits"): return read_sauron_fits(ifile)
	else: return read_sauron_txt(ifile)

def write_sauron_fits(ofile, cat):
	from astropy.io import fits
	ocat = cat.copy()
	for field in ["ra","dec"]: ocat[field] /= utils.degree # angles in degrees
	hdu = fits.hdu.table.BinTableHDU(ocat)
	hdu.writeto(ofile, overwrite=True)

def read_sauron_fits(fname):
	from astropy.io import fits
	hdu = fits.open(fname)[1]
	cat = np.asarray(hdu.data).view(np.recarray)
	for field in ["ra","dec"]: cat[field] *= utils.degree # deg -> rad
	return cat

def write_sauron_txt(ofile, cat):
	with open(ofile, "w") as ofile:
		ofile.write(format_sauron(cat))

def read_sauron_txt(ifile, ncomp=3):
	raw   = np.loadtxt(ifile, ndmin=2)
	nrow, ncol = raw.shape
	nfreq = (ncol-2-ncomp-1)//(2*ncomp+1)
	cat_dtype  = [("ra", "d"), ("dec", "d"), ("snr", "d", (ncomp,)), ("flux_tot", "d", (ncomp,)),
			("dflux_tot", "d", (ncomp,)), ("flux", "d", (nfreq,ncomp)), ("dflux", "d", (nfreq,ncomp)),
			("case", "i"), ("contam", "d", (nfreq,))]
	ocat  = np.zeros(nrow, cat_dtype).view(np.recarray)
	ocat.ra, ocat.dec, raw = raw[:,0]*utils.degree, raw[:,1]*utils.degree, raw[:,2:]
	ocat.snr,          raw = raw[:,:ncomp], raw[:,ncomp:]
	ocat.flux_tot, ocat.dflux_tot, raw = raw[:,0:2*ncomp:2], raw[:,1:2*ncomp:2], raw[:,2*ncomp:]
	ocat.flux,     ocat.dflux,     raw = raw[:,0:2*ncomp*nfreq:2].reshape(-1,nfreq,ncomp), raw[:,1:2*ncomp*nfreq:2].reshape(-1,nfreq,ncomp), raw[:,2*ncomp*nfreq:]
	ocat.case,         raw = raw[:,0], raw[:,1:]
	ocat.contam            = raw[:,:nfreq]
	return ocat

def translate_dtype_keys(d, translation):
	descr = [(name if name not in translation else translation[name], char) for (name, char) in d.dtype.descr]
	return np.asarray(d, descr)

def src2param(srcs):
	"""Translate recarray srcs into the source fromat used for tod-level point source
	operations."""
	params = np.zeros(srcs.shape + (8,))
	params[:,0] = srcs.dec * utils.degree # yes, dec first
	params[:,1] = srcs.ra  * utils.degree
	params[:,2] = srcs.I
	if "Q" in srcs.dtype.names: params[:,3] = srcs.Q
	if "U" in srcs.dtype.names: params[:,4] = srcs.U
	# These are not used
	params[:,5] = 1 # x-scaling
	params[:,6] = 1 # y-scaling
	params[:,7] = 0 # angle
	return params
