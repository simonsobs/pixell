import numpy as np, os, warnings
import astropy.io.fits
from . import enmap, utils, wcsutils

class ndmaps(np.ndarray):
	"""A class for representing and working with a list of enmaps as if they were a single
	object, letting one do math operations etc. on them all at the same time. Useful for
	things like wavelet decompositions etc."""
	def __new__(cls, arr, geometries):
		"""Constructs an ndarrays object given a raw array arr[...,totpix] and
		a tuple of geometries."""
		obj = np.asarray(arr).view(cls)
		obj.geometries = tuple([enmap.Geometry(*geo).nopre for geo in geometries])
		return obj
	def __array_finalize__(self, obj):
		if obj is None: return
		self.geometries = getattr(obj, "geometries", None)
	def __repr__(self):
		return "ndmaps(%s,%s)" % (np.asarray(self), str(self.geometries))
	def __str__(self): return repr(self)
	def __array_wrap__(self, arr, context=None):
		return ndmaps(arr, self.geometries)
	def contig(self): return ndmaps(np.ascontiguousarray(self), self.geometries)
	@property
	def pre(self): return self.shape[:-1]
	@property
	def npixs(self):
		return [geo.npix for geo in self.geometries]
	@property
	def ntot(self):
		return np.sum(self.npixs)
	@property
	def nmap(self):
		return len(self.geometries)
	def copy(self, order='K'):
		return ndmaps(np.copy(self,order), self.geometries)
	@property
	def maps(self):
		return _map_view(self)
	def posmap(self, safe=True, corner=False, separable="auto", dtype=np.float64): return posmap(self.geometries, corner=corner, separable=separable, dtype=dtype)
	def pixmap(self, dtype=np.float64): return pixmap(self.geometries, dtype=dtype)
	def pixsize(self, dtype=np.float64): return pixsize(self.geometries, dtype=dtype)
	def lmap(self, oversample=1, dtype=np.float64): return lmap(self.geometries, oversample=oversample, dtype=dtype)
	def modlmap(self, oversample=1, dtype=np.float64): return modlmap(self.geometries, oversample=oversample)
	def modrmap(self, ref="center", safe=True, corner=False, dtype=np.float64): return modrmap(self.geometries, ref=ref, safe=safe, corner=corner, dtype=dtype)

class _map_view:
	"""Helper class used to implement access to the individual enmaps that make up an ndmaps object"""
	def __init__(self, mmap):
		self.multimap = mmap
		self.offs     = utils.cumsum(mmap.npixs, endpoint=True)
	def __len__(self): return self.multimap.nmap
	def __getitem__(self, sel):
		sel1, sel2 = utils.split_slice(sel, [1,self.multimap.ndim+2-1])
		if len(sel1) == 0: return self.multimap
		i    = sel1[0]
		return enmap.ndmap(self.multimap[...,self.offs[i]:self.offs[i+1]].reshape(self.multimap.pre + self.multimap.geometries[i].shape[-2:]), self.multimap.geometries[i].wcs)[sel2]
	def __setitem__(self, sel, val):
		sel1, sel2 = utils.split_slice(sel, [1,self.multimap.ndim+2-1])
		if len(sel1) == 0: return self.multimap
		i    = sel1[0]
		# This assumes that the slicing and reshaping won't cause copies. This should be the case if
		# self.multimap is contiguous. To be robust I should detect if a copy would be made, and fall back
		# on something slower in that case
		self.multimap[...,self.offs[i]:self.offs[i+1]].reshape(self.multimap.pre + self.multimap.geometries[i].shape[-2:])[sel2] = val

def multimap(maps):
	"""Construct a multimap (ndmaps instance) from a list of enmaps. These must have compatible pre-dimensions."""
	if len(maps) == 0: return ndmaps(np.zeros(0), [(0,0), enmap.zeros(0).wcs])
	for i, map in enumerate(maps):
		if map.shape[:-2] != maps[0].shape[:-2]:
			raise ValueError("Map %d in multimaps constructor has pre-shape %s, incompatible with map 0 with %s" % (i, str(map.shape[:-2]), str(maps[0].shape[:-2])))
	flat = np.concatenate([np.asarray(map).reshape(map.shape[:-2]+(-1,)) for map in maps],-1)
	geos = [map.geometry for map in maps]
	return ndmaps(flat, geos)

def _geo_helper(geometries):
	geometries = [enmap.Geometry(*geo) for geo in geometries]
	ntot = 0
	for i, geo in enumerate(geometries):
		if geo.shape[:-2] != geometries[0].shape[:-2]:
			raise ValueError("Geometry %d in zeros has pre-shape %s, incompatible with geometry 0 with %s" % (i, str(geo.shape[:-2]), str(geometries[0].shape[:-2])))
		ntot += geo.shape[-2]*geo.shape[-1]
	return geometries, ntot

def zeros(geometries, dtype=np.float64):
	"""Construct a zero-initialized multimap with the given geometries and data type"""
	if len(geometries) == 0: return ndmaps(np.zeros(0), [(0,0), enmap.zeros(0).wcs])
	geometries, ntot = _geo_helper(geometries)
	flat = np.zeros(geometries[0].shape[:-2]+(ntot,), dtype)
	return ndmaps(flat, geometries)

def empty(geometries, dtype=np.float64):
	"""Construct an uninitialized multimap with the given geometries and data type"""
	if len(geometries) == 0: return ndmaps(np.zeros(0), [(0,0), enmap.zeros(0).wcs])
	geometries, ntot = _geo_helper(geometries)
	flat = np.empty(geometries[0].shape[:-2]+(ntot,), dtype)
	return ndmaps(flat, geometries)

def full(geometries, val, dtype=None):
	"""Construct a multimap with the given geometries and data type initialized with the given value.
	If val is scalar then all maps will be filled with this value. Otherwise it must broadcast with
	geometries[0].shape[:-2]+(len(geometries),). This allows one to initialize each map with a
	separate constant.
	"""
	if len(geometries) == 0: return ndmaps(np.zeros(0), [(0,0), enmap.zeros(0).wcs])
	geometries, ntot = _geo_helper(geometries)
	# Broadcast val and geometries
	pre  = geometries[0].shape[:-2]
	val  = np.asarray(val)
	nmap = len(geometries)
	bshape = np.broadcast_shapes(val.shape, pre+(nmap,))
	val  = np.broadcast_to(val, bshape)
	pre  = val.shape[:-1]
	# Allocate the output maps
	if dtype is None: dtype = val.dtype
	flat = np.empty(pre+(ntot,), dtype)
	omaps= ndmaps(flat, geometries)
	# Fill with target values
	for i, map in enumerate(omaps.maps):
		map[:] = val[...,i]
	return omaps

def posmap(geometries, safe=True, corner=False, separable="auto", dtype=np.float64):
	"""Return a multimap containing the position map for the given geometries"""
	return multimap([enmap.posmap(*geo, safe=safe, corner=corner, separable=separable, dtype=dtype) for geo in geometries])

def pixmap(geometries, dtype=np.float64):
	"""Return a multimap containing the pixel map for the given geometries"""
	return multimap([enmap.pixmap(*geo, dtype=dtype) for geo in geometries])

def lmap(geometries, dtype=np.float64):
	"""Return a mapmultimap of all the wavenumbers in the fourier transform
	of a map with the given geometries"""
	return multimap([enmap.lmap(*geo).astype(dtype) for geo in geometries])

def modlmap(geometries, dtype=np.float64):
	"""Return a mapmultimap of all the abs wavenumbers in the fourier transform
	of a map with the given geometries"""
	return multimap([enmap.modlmap(*geo).astype(dtype) for geo in geometries])

def modrmap(geometries, ref="center", safe=True, corner=False, dtype=np.float64):
	"""Return a multimap where each entry is the distance from center
	of that entry. Results are returned in radians, and if safe is true
	(default), then sharp coordinate edges will be avoided."""
	return multimap([enmap.modrmap(*geo, ref=ref, safe=safe).astype(dtype) for geo in geometries])

def pixsize(geometries, dtype=np.float64):
	return np.array([enmap.pixsize(*geo).astype(dtype) for geo in geometries])

def pixsizemap(geometries, dtype=np.float64):
	return multimap([enmap.pixsizemap(*geo).astype(dtype) for geo in geometries])

def samegeos(arr, *args):
	"""Returns arr with the same geometries information as the first multimap among
	args.  If no matches are found, arr is returned as is.  Will
	reference, rather than copy, the underlying array data
	whenever possible.
	"""
	for m in args:
		try: return ndmaps(arr, m.geometries)
		except AttributeError: pass
	return arr

def nopre(geometries):
	"""Return a scalar version of the given geometries"""
	return tuple([enmap.Geometry(*geo).nopre for geo in geometries])

def map_mul(mat, vec):
	"""Elementwise matrix multiplication mat*vec. Result will have
	the same shape as vec. Multiplication happens along the last non-pixel
	indices."""
	# Allow scalar product, broadcasting if necessary
	mat = np.asanyarray(mat)
	if mat.ndim <= 2: return mat*vec
	# Otherwise we do a matrix product along the last axes
	ovec = samegeos(np.einsum("...abi,...bi->...ai", mat, vec), mat, vec)
	return ovec

def mean(mmap):
	"""Return the mean along the pixel direction for all maps in the multimap."""
	return np.array([np.mean(m,(-2,-1)) for m in mmap.maps])

def median(mmap):
	"""Return the median along the pixel direction for all maps in the multimap."""
	return np.array([np.median(m,(-2,-1)) for m in mmap.maps])

def max(mmap):
	"""Return the max along the pixel direction for all maps in the multimap."""
	return np.array([np.max(m,(-2,-1)) for m in mmap.maps])

def min(mmap):
	"""Return the min along the pixel direction for all maps in the multimap."""
	return np.array([np.min(m,(-2,-1)) for m in mmap.maps])

def var(mmap):
	"""Return the variance along the pixel direction for all maps in the multimap."""
	return np.array([np.var(m,(-2,-1)) for m in mmap.maps])

def std(mmap):
	"""Return the standard deviation along the pixel direction for all maps in the multimap."""
	return np.array([np.std(m,(-2,-1)) for m in mmap.maps])

def fft(mmap, omap=None, nthread=0, normalize=True, adjoint_ifft=False, dct=False):
	if omap is None: omap = mmap*0j
	return multimap([enmap.fft(im.astype(om.dtype), omap=om, nthread=nthread, normalize=normalize, adjoint_ifft=adjoint_ifft, dct=dct) for im, om in zip(mmap.maps, omap.maps)])

def ifft(mmap, omap=None, nthread=0, normalize=True, adjoint_fft=False, dct=False):
	mmap = utils.ascomplex(mmap)
	if omap is None: omap = mmap.copy()
	return multimap([enmap.ifft(im, omap=om, nthread=nthread, normalize=normalize, adjoint_fft=adjoint_fft, dct=dct) for im, om in zip(mmap.maps, omap.maps)])

# Not sure if all these are needed here

def dct(emap, omap=None, nthread=0, normalize=True):
	return fft(emap, omap=omap, nthread=nthread, normalize=normalize, dct=True)
def idct(emap, omap=None, nthread=0, normalize=True):
	return ifft(emap, omap=omap, nthread=nthread, normalize=normalize, dct=True)

def fft_adjoint(emap, omap=None, nthread=0, normalize=True):
	return ifft(emap, omap=omap, nthread=nthread, normalize=normalize, adjoint_fft=True)
def ifft_adjoint(emap, omap=None, nthread=0, normalize=True):
	return fft(emap, omap=omap, nthread=nthread, normalize=normalize, adjoint_ifft=True)

def idct_adjoint(emap, omap=None, nthread=0, normalize=True):
	return fft(emap, omap=omap, nthread=nthread, normalize=normalize, adjoint_ifft=True, dct=True)
def dct_adjoint(emap, omap=None, nthread=0, normalize=True):
	return ifft(emap, omap=omap, nthread=nthread, normalize=normalize, adjoint_fft=True, dct=True)

def map2harm(mmap, nthread=0, normalize=True, iau=False, spin=[0,2], adjoint_harm2map=False):
	return multimap([enmap.map2harm(m, nthread=nthread, normalize=normalize, iau=iau, spin=spin, adjoint_harm2map=adjoint_harm2map) for m in mmap.maps])

def harm2map(mmap, nthread=0, normalize=True, iau=False, spin=[0,2], keep_imag=False, adjoint_map2harm=False):
	return multimap([enmap.harm2map(m, nthread=nthread, normalize=normalize, iau=iau, spin=spin, adjoint_map2harm=adjoint_map2harm) for m in mmap.maps])

def map2harm_adjoint(mmap, nthread=0, normalize=True, iau=False, spin=[0,2], keep_imag=False):
	return harm2map(mmap, nthread=nthread, normalize=normalize, iau=iau, spin=spin, keep_imag=keep_imag, adjoint_map2harm=True)

def harm2map_adjoint(mmap, nthread=0, normalize=True, iau=False, spin=[0,2]):
	return map2harm(mmap, nthread=nthread, normalize=normalize, iau=iau, spin=spin, adjoint_harm2map=True)

def queb_rotmat(lmap, inverse=False, iau=False, spin=2):
	out = enmap.queb_rotmat(lmap, inverse=inverse, iau=iau, spin=spin)
	return samegeos(out,lmap)

def rotate_pol(mmap, angle, comps=[-2,-1]):
	c, s = np.cos(2*angle), np.sin(2*angle)
	res = mmap.copy()
	res[...,comps[0],:] = c*mmap[...,comps[0],:] - s*mmap[...,comps[1],:]
	res[...,comps[1],:] = s*mmap[...,comps[0],:] + c*mmap[...,comps[1],:]
	return res

def write_map(fname, mmap, extra={}):
	"""Write multimap mmap to the file fname. Each map in mmap will be written
	to its own image hdu in the file. These can be read individually with
	enmap.read_map, or together with multimap.read_map."""
	hdus = []
	for ind in range(mmap.nmap):
		emap   = mmap.maps[ind].copy()
		header = emap.wcs.to_header(relax=True)
		# Add our map headers
		header['NAXIS'] = emap.ndim
		for i,n in enumerate(emap.shape[::-1]):
			header['NAXIS%d'%(i+1)] = n
		for key, val in extra.items():
			header[key] = val
		# multimap-specific stuff
		if ind == 0:
			header["MMAPN"] = str(mmap.nmap)
			hdu = astropy.io.fits.PrimaryHDU(emap, header)
		else:
			hdu = astropy.io.fits.ImageHDU(emap, header)
		hdus.append(hdu)
	hdus = astropy.io.fits.HDUList(hdus)
	utils.mkdir(os.path.dirname(fname))
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		hdus.writeto(fname, overwrite=True)

def read_map(fname, sel=None, box=None, wrap="auto", mode=None, sel_threshold=10e6, verbose=False):
	"""Read a multimap from the file fname."""
	hdus = astropy.io.fits.open(fname)
	h0   = hdus[0].header
	nmap = int(h0["MMAPN"]) if "MMAPN" in h0 else len(hdus)
	maps = []
	for ind in range(nmap):
		with warnings.catch_warnings():
			wcs = wcsutils.WCS(hdus[ind].header).sub(2)
		proxy = enmap.ndmap_proxy_fits(hdus[ind], wcs, fname=fname, threshold=sel_threshold, verbose=verbose)
		maps.append(enmap.read_helper(proxy, sel=sel, box=box, wrap=wrap, mode=mode))
	return multimap(maps)
