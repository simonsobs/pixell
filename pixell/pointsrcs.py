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
import numpy as np
from astropy.io import fits
import astropy.table as atpy
from . import utils, enmap

#### Map-space source simulation ###

def sim_srcs(shape, wcs, srcs, beam, omap=None, dtype=None, nsigma=5, rmax=None, method="loop", smul=1,
		return_padded=False, pixwin=False):
	"""Simulate a point source map in the geometry given by shape, wcs
	for the given srcs[nsrc,{dec,ra,T...}], using the beam[{r,val},npoint],
	which must be equispaced. If omap is specified, the sources will be
	added to it in place. All angles are in radians. The beam is only evaluated up to
	the point where it reaches exp(-0.5*nsigma**2) unless rmax is specified, in which
	case this gives the maximum radius. smul gives a factor to multiply the resulting
	source model by. This is mostly useful in conction with omap. method can be
	"loop" or "vectorized", but "loop" is both faster and uses less memory, so there's
	no point in using the latter.

	The source simulation is sped up by using a source lookup grid.
	"""
	if omap is None: omap = enmap.zeros(shape, wcs, dtype)
	ishape = omap.shape
	omap   = omap.preflat
	ncomp  = omap.shape[0]
	# In keeping with the rest of the functions here, srcs is [nsrc,{dec,ra,T,Q,U}].
	# The beam parameters are ignored - the beam argument is used instead
	amps = srcs[:,2:2+ncomp]
	poss = srcs[:,:2].copy()
	# Rewind positions to let us use flat-sky approximation for distance calculations
	ref  = np.mean(enmap.box(shape, wcs, corner=False)[:,1])
	poss[:,1] = utils.rewind(poss[:,1], ref)
	beam = expand_beam(beam, nsigma, rmax)
	rmax = nsigma2rmax(beam, nsigma)
	# Pad our map by rmax, so we get the contribution from sources
	# just ourside our area. We will later split our map into cells of size cres. Let's
	# adjust the padding so we have a whole number of cells
	cres = utils.nint(rmax/omap.pixshape())
	epix = cres-(omap.shape[-2:]+2*cres)%cres
	padding = [cres,cres+epix]
	wmap, wslice  = enmap.pad(omap, padding, return_slice=True)
	# Overall we will have this many grid cells
	cshape = wmap.shape[-2:]/cres
	# Find out which sources matter for which cells
	srcpix = wmap.sky2pix(poss.T).T
	pixbox= np.array([[0,0],wmap.shape[-2:]],int)
	nhit, cell_srcs = build_src_cells(pixbox, srcpix, cres)
	posmap = wmap.posmap()
	model = eval_srcs_loop(posmap, poss, amps, beam, cres, nhit, cell_srcs, dtype=wmap.dtype)
	del posmap
	if pixwin: model = enmap.apply_window(model)
	# Update our work map, through our view
	if smul != 1: model *= smul
	wmap  += model
	if not return_padded:
		# Copy out
		omap[:] = wmap[wslice]
		# Restore shape
		omap = omap.reshape(ishape)
		return omap
	else:
		return wmap.reshape(ishape[:-2]+wmap.shape[-2:]), wslice

def eval_srcs_loop(posmap, poss, amps, beam, cres, nhit, cell_srcs, dtype=np.float64):
	# Loop through each cell
	ncy, ncx = nhit.shape
	model = enmap.zeros(amps.shape[-1:]+posmap.shape[-2:], posmap.wcs, dtype)
	for cy in range(ncy):
		for cx in range(ncx):
			nsrc = nhit[cy,cx]
			if nsrc == 0: continue
			srcs  = cell_srcs[cy,cx,:nsrc]
			y1,y2 = (cy+0)*cres[0], (cy+1)*cres[0]
			x1,x2 = (cx+0)*cres[1], (cx+1)*cres[1]
			pixpos = posmap[:,y1:y2,x1:x2]
			srcpos = poss[srcs].T # [2,nsrc]
			srcamp = amps[srcs].T # [ncomp,nsrc]
			diff   = pixpos[:,None,:,:]-srcpos[:,:,None,None]
			r      = (diff[0]**2 + (diff[1]*np.cos(pixpos[0,None,:,:]))**2)**0.5
			bpix   = (r - beam[0,0])/(beam[0,1]-beam[0,0])
			# Evaluate the beam at these locations
			bval   = utils.interpol(beam[1], bpix[None], mode="constant", order=1) # [nsrc,ry,rx]
			cmodel = srcamp[:,:,None,None]*bval
			cmodel = np.sum(cmodel,-3)
			model[:,y1:y2,x1:x2] += cmodel
	return model

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

def build_src_cells(cbox, srcpos, cres, unwind=False):
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
	nmax = max(1,np.max(build_src_cells_helper(cbox, cshape, cres, srcpos)))
	ncell, cells = build_src_cells_helper(cbox, cshape, cres, srcpos, nmax)
	# Reshape back to original shape
	ncell = ncell.reshape(ishape[1:-1]+ncell.shape[1:])
	cells = cells.reshape(ishape[1:-1]+cells.shape[1:])
	return ncell, cells

def build_src_cells_helper(cbox, cshape, cres, srcpos, nmax=0):
	# A cell is hit if it overlaps both horizontally and vertically
	# with the point source +- cres
	nsrc, nmid = srcpos.shape[:2]
	# ncell is [:,ncy,ncx]
	ncell = np.zeros((nmid,)+cshape,np.int32)
	if nmax > 0:
		cells = np.zeros((nmid,)+cshape+(nmax,),np.int32)
	c0 = cbox[0]; inv_dc = cshape/(cbox[1]-cbox[0]).astype(float)
	for si in range(nsrc):
		for mi in range(nmid):
			pos = srcpos[si,mi]
			i1 = (pos[:2]-cres-c0)*inv_dc
			i2 = (pos[:2]+cres-c0)*inv_dc+1 # +1 because this is a half-open interval
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
	cshape = map.shape[-2:]/res
	omap   = map[...,:cshape[0]*res[0],:cshape[1]*res[1]]
	omap   = omap.reshape(omap.shape[:-2]+(cshape[0],res[0],cshape[1],res[1]))
	omap   = utils.moveaxis(omap, -3, -2)
	return omap

def uncellify(cmap):
	omap = utils.moveaxis(cmap, -2, -3)
	omap = omap.reshape(omap.shape[:-4]+(omap.shape[-4]*omap.shape[-3],omap.shape[-2]*omap.shape[-1]))
	return omap

#### Source parameter I/O ####

def read(fname, format="auto"):
	if format == "auto": formats = ["fits","nemo","simple"]
	else:                formats = [format]
	for f in formats:
		try:
			if   f == "fits":   return read_fits(fname)
			elif f == "simple": return read_simple(fname)
			else: raise ValueError("Unrecognized point source format '%s' for file '%s'" % (f, fname))
		except (ValueError, IOError) as e: pass
	raise IOError("Unable to read point source file '%s' with format '%s'" % (fname, f))

def read_simple(fname):
	try:
		return np.loadtxt(fname, dtype=[("ra","d"),("dec","d"),("I","d"),("dI","d")], usecols=range(4), ndmin=1).view(np.recarray)
	except ValueError:
		try:
			return np.loadtxt(fname, dtype=[("ra","d"),("dec","d"),("I","d")], usecols=range(3), ndmin=1).view(np.recarray)
		except ValueError as e:
			raise IOError(e.message)

def read_fits(fname, fix=True, fix_cols = {"RADeg":"ra","decDeg":"dec","deltaT_c":"I","err_deltaT_c":"dI"}):
	"""Reads in a FITS table format catalog (e.g., as produced by nemo), converting it into a format that can
	be used by sim_srcs etc..

	Args:
		fname: Filename for .fits table
		fix: If True, rename columns according to the contents of the fix_cols dictionary
		fix_cols: A dictionary in the format {"column_to_rename": "new_column_name"}

	Returns:
		numpy recarray

	"""
	tab=atpy.Table().read(fname)
	if fix:
		for k in fix_cols.keys():
			if k in tab.keys():
				tab.rename_column(k, fix_cols[k])
	return tab.as_array().view(np.recarray)

def src2param(srcs):
	"""Translate recarray srcs into the source fromat used for tod-level point source
	operations."""
	params = np.zeros(srcs.shape + (8,))
	params[:,0] = srcs.dec * utils.degree # yes, dec first
	params[:,1] = srcs.ra  * utils.degree
	params[:,2] = srcs.I
	if "Q" in srcs.dtype.names: params[:,3] = srcs.Q
	if "U" in srcs.dtype.names: params[:,4] = srcs.Q
	# These are not used
	params[:,5] = 1 # x-scaling
	params[:,6] = 1 # y-scaling
	params[:,7] = 0 # angle
	return params

# The below should be replaced with whatever is used for SO... this is for ACT format
def read_beam(fname):
	"""Given a filename, read an equi-spaced radial beam profile.
	The file should have format [r,b(r)].
	"""
	return np.loadtxt(fname, ndmin=2).T[:2]
