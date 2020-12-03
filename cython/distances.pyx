import cython, numpy as np
from scipy import ndimage
cimport numpy as np
from libc.stdlib cimport free
# I had to call this distances_wrap instead of just distances because
# .pdx files with the same name get automatically imported into the current
# namespace, which means I would have to give stupid names to the functions here
cimport distances_wrap as c
from distances_wrap cimport inum, uint8_t

__version__ = 1.0

def distance_from_points_simple(posmap, points, omap=None, odomains=None, domains=False):
	"""distance_from_points(posmap, points, omap=None, odomains=None, domains=False)

	Given a posmap[{dec,ra},ny,nx] and a set of points[{dec,ra},npoint], computes the
	angular distance map from every pixel [ny,nx] to the nearest point. If domains==True,
	then a [ny,nx] map of the index of the nearest point is also returned. New arrays
	will be created for the output unless omap and/or odomains are specified, in which
	case they will be overwritten."""
	# Check that our inputs make sense
	posmap = np.asanyarray(posmap).astype(float, order="C", copy=False)
	points = np.asanyarray(points).astype(float, order="C", copy=False)
	assert posmap.ndim == 3 and len(posmap) == 2, "posmap must be [{dec,ra},ny,nx]"
	assert points.ndim == 2 and len(points) == 2, "points must be [{dec,ra},npoint]"
	if omap is None: omap = np.empty_like(posmap[0], dtype=np.float64)
	assert omap.ndim == 2 and omap.shape[-2:] == posmap.shape[-2:] and omap.dtype==np.float64, "omap must be [ny,nx] float64"
	if domains:
		if odomains is None: odomains = np.empty_like(posmap[0], dtype=np.int32)
		assert odomains.ndim == 2 and odomains.shape[-2:] == posmap.shape[-2:] and odomains.dtype==np.int32, "odomains must be [ny,nx] int32"
	# Prepare to call C
	cdef inum npix   = omap.size
	cdef inum npoint = points.shape[1]
	cdef double[::1] posmap_ = posmap.reshape(-1)
	cdef double[::1] points_ = points.reshape(-1)
	cdef double[::1] omap_   = omap.reshape(-1)
	cdef int[::1]    odomains_
	if not domains:
		c.distance_from_points_simple(npix, &posmap_[0], npoint, &points_[0], &omap_[0], NULL)
		return omap
	else:
		odomains_ = odomains.reshape(-1)
		c.distance_from_points_simple(npix, &posmap_[0], npoint, &points_[0], &omap_[0], &odomains_[0])
		return omap, odomains

def distance_from_points_simple_separable(ypos, xpos, points, omap=None, odomains=None, domains=False):
	"""distance_from_points_simple_separable(ypos, xpos, points, omap=None, odomains=None, domains=False)

	Like distance_from_points, but optimized for the case where the coordinate system
	is separable, as is typically the case for cylindrical projections. Instead of a full
	posmap[{dec,ra},ny,nx] it takes ypos[ny] which gives the dec of each point along the y axis
	and xpos[nx] which gives the ra of each point along the x axis. The main advantage of this
	is that one can avoid the somewhat heavy computation of the full posmap."""
	# Check that our inputs make sense
	ypos   = np.asanyarray(ypos).astype(float, order="C", copy=False)
	xpos   = np.asanyarray(xpos).astype(float, order="C", copy=False)
	points = np.asanyarray(points).astype(float, order="C", copy=False)
	assert ypos.ndim == 1, "ypos must be [ny]"
	assert xpos.ndim == 1, "xpos must be [nx]"
	assert points.ndim == 2 and len(points) == 2, "points must be [{dec,ra},npoint]"
	if omap is None: omap = np.empty_like(ypos, dtype=np.float64, shape=(len(ypos),len(xpos)))
	assert omap.ndim == 2 and omap.shape[-2:] == (len(ypos),len(xpos)) and omap.dtype==np.float64, "omap must be [ny,nx] float64"
	if domains:
		if odomains is None: odomains = np.empty_like(ypos, dtype=np.int32, shape=(len(ypos),len(xpos)))
		assert odomains.ndim == 2 and odomains.shape[-2:] == (len(ypos),len(xpos)) and odomains.dtype==np.int32, "odomains must be [ny,nx] int32"
	# Prepare to call C
	cdef inum ny = len(ypos)
	cdef inum nx = len(xpos)
	cdef inum npoint = points.shape[1]
	cdef double[::1] ypos_   = ypos.reshape(-1)
	cdef double[::1] xpos_   = xpos.reshape(-1)
	cdef double[::1] points_ = points.reshape(-1)
	cdef double[::1] omap_   = omap.reshape(-1)
	cdef int[::1]    odomains_
	if not domains:
		c.distance_from_points_simple_separable(ny, nx, &ypos_[0], &xpos_[0], npoint, &points_[0], &omap_[0], NULL)
		return omap
	else:
		odomains_ = odomains.reshape(-1)
		c.distance_from_points_simple_separable(ny, nx, &ypos_[0], &xpos_[0], npoint, &points_[0], &omap_[0], &odomains_[0])
		return omap, odomains

def fix_point_pix(posmap, point_pos, point_pix):
	"""fix_point_pix(posmap, point_pos, point_pix)

	Return a new point_pix where out of bounds points have been replaced by the closest point
	on the boundary. This uses the simple method for the pixels on the boundary. The number of pixels
	on the boundary are much lower than the pixels in the main part of the image, which partially
	makes up for the slowness of the simple method, but you can still expect slowness if too many points
	are outside the image."""
	ypos, xpos = posmap
	# Get the bad points
	bad = np.where(np.any(point_pix < 0,0) | np.any(point_pix >= np.array(ypos.shape[-2:])[:,None],0))[0]
	# If there aren't any, we can just return right away
	nbad = len(bad)
	if nbad == 0: return point_pix
	# Otherwise, run the simple method on each boundary. The simple method is slow, but the boundary is
	# has much fewer pixels.
	point_bad = point_pos[:,bad]
	pos_edges = np.array([np.concatenate([p[:,0],p[:,-1],p[0,:],p[-1,:]],-1) for p in [ypos,xpos]])
	dist, dom = distance_from_points_simple(pos_edges[:,None,:], point_bad, domains=True)
	dist, dom = dist[0], dom[0]
	# Find the minimum position for each point
	minpos1d = np.array(ndimage.minimum_position(dist, dom+1, np.arange(nbad)+1)).reshape(-1)
	# Turn the 1d minpos into a 2d pixel position
	minpos   = _unwrap_minpos(minpos1d, ypos.shape[0], ypos.shape[1])
	# Copy these into the output point_pix
	opoint_pix = point_pix.copy()
	opoint_pix[:,bad] = minpos
	return opoint_pix

def fix_point_pix_separable(ypos, xpos, point_pos, point_pix):
	"""fix_point_pix_separable(ypos, xpos, point_pos, point_pix)

	Return a new point_pix where out of bounds points have been replaced by the closest point
	on the boundary. This uses the simple method for the pixels on the boundary. The number of pixels
	on the boundary are much lower than the pixels in the main part of the image, which partially
	makes up for the slowness of the simple method, but you can still expect slowness if too many points
	are outside the image."""
	# Get the bad points
	bad = np.where(np.any(point_pix < 0,0) | (point_pix[0] >= ypos.size) | (point_pix[1] >= xpos.size))[0]
	# If there aren't any, we can just return right away
	nbad = len(bad)
	if nbad == 0: return point_pix
	# Otherwise, run the simple method on each boundary. The simple method is slow, but the boundary is
	# has much fewer pixels.
	point_bad = point_pos[:,bad]
	pos_edges = np.array([
		np.concatenate([ypos, ypos, np.full(xpos.size, ypos[0]), np.full(xpos.size, ypos[-1])]),
		np.concatenate([np.full(ypos.size, xpos[0]), np.full(ypos.size, xpos[-1]), xpos, xpos])])
	dist, dom = distance_from_points_simple(pos_edges[:,None,:], point_bad, domains=True)
	dist, dom = dist[0], dom[0]
	# Find the minimum position for each point
	minpos1d = np.array(ndimage.minimum_position(dist, dom+1, np.arange(nbad)+1)).reshape(-1)
	# Turn the 1d minpos into a 2d pixel position
	minpos   = _unwrap_minpos(minpos1d, ypos.size, xpos.size)
	# Copy these into the output point_pix
	opoint_pix = point_pix.copy()
	opoint_pix[:,bad] = minpos
	return opoint_pix

def _unwrap_minpos(minpos1d, ny, nx):
	minpos   = np.zeros([2,len(minpos1d)], minpos1d.dtype)
	mask = minpos1d < ny;
	minpos[0,mask], minpos[1,mask] = minpos1d[mask], 0
	mask = (minpos1d >= ny) & (minpos1d < 2*ny)
	minpos[0,mask], minpos[1,mask] = minpos1d[mask]-ny, nx-1
	mask = (minpos1d >= 2*ny) & (minpos1d < 2*ny+nx)
	minpos[0,mask], minpos[1,mask] = 0, minpos1d[mask]-2*ny
	mask = minpos1d >= 2*ny+nx
	minpos[0,mask], minpos[1,mask] = ny-1, minpos1d[mask]-(2*ny+nx)
	return minpos

def distance_from_points_bubble(posmap, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False):
	"""distance_from_points_buble(posmap, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False)"""
	# Check that our inputs make sense
	posmap    = np.asanyarray(posmap).astype(float, order="C", copy=False)
	point_pos = np.asanyarray(point_pos).astype(float,    order="C", copy=False)
	point_pix = np.asanyarray(point_pix).astype(np.int32, order="C", copy=False)
	assert posmap.ndim == 3 and len(posmap) == 2, "posmap must be [{dec,ra},ny,nx]"
	assert point_pos.ndim == 2 and len(point_pos) == 2, "point_pos must be [{dec,ra},npoint]"
	assert point_pix.ndim == 2 and len(point_pix) == 2, "point_pix must be [{y,x},npoint]"
	if omap is None: omap = np.empty_like(posmap[0], dtype=np.float64)
	assert omap.ndim == 2 and omap.shape == posmap.shape[-2:] and omap.dtype==np.float64, "omap must be [ny,nx] float64"
	if odomains is None: odomains = np.empty_like(posmap[0], dtype=np.int32)
	assert odomains.ndim == 2 and odomains.shape == posmap.shape[-2:] and odomains.dtype==np.int32, "odomains must be [ny,nx] int32"
	# Make point_pix safe
	point_pix = fix_point_pix(posmap, point_pos, point_pix)
	# Prepare to call C
	cdef int ny = posmap.shape[1]
	cdef int nx = posmap.shape[2]
	cdef inum npoint = point_pos.shape[1]
	cdef double      rmax_      = (0 if rmax is None else rmax)
	cdef double[::1] posmap_    = posmap.reshape(-1)
	cdef double[::1] point_pos_ = point_pos.reshape(-1)
	cdef int[::1]    point_pix_ = point_pix.reshape(-1)
	cdef double[::1] omap_   = omap.reshape(-1)
	cdef int[::1]    odomains_ = odomains.reshape(-1)
	c.distance_from_points_bubble(ny, nx, &posmap_[0], npoint, &point_pos_[0], &point_pix_[0], rmax_, &omap_[0], &odomains_[0])
	if domains: return omap, odomains
	else:       return omap

def distance_from_points_bubble_separable(ypos, xpos, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False):
	"""distance_from_points_bubble_separable(ypos, xpos, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False)

	Like distance_from_points_bubble, but optimized for the case where the coordinate system
	is separable, as is typically the case for cylindrical projections. Instead of a full
	posmap[{dec,ra},ny,nx] it takes ypos[ny] which gives the dec of each point along the y axis
	and xpos[nx] which gives the ra of each point along the x axis. The main advantage of this
	is that one can avoid the somewhat heavy computation of the full posmap. If rmax is specified,
	the calculation will stop at the distance rmax. Beyond that distance domain will be set to -1 and
	distance will be set to rmax. This can be used to speed up the calculation if one only cares about
	distances up to a certain point."""
	# Check that our inputs make sense
	ypos   = np.asanyarray(ypos).astype(float, order="C", copy=False)
	xpos   = np.asanyarray(xpos).astype(float, order="C", copy=False)
	point_pos = np.asanyarray(point_pos).astype(float,    order="C", copy=False)
	point_pix = np.asanyarray(point_pix).astype(np.int32, order="C", copy=False)
	assert ypos.ndim == 1, "ypos must be [ny]"
	assert xpos.ndim == 1, "xpos must be [nx]"
	assert point_pos.ndim == 2 and len(point_pos) == 2, "point_pos must be [{dec,ra},npoint]"
	assert point_pix.ndim == 2 and len(point_pix) == 2 and point_pix.shape[1] == point_pos.shape[1], "point_pos must be [{y,x},npoint]"
	if omap is None: omap = np.empty_like(ypos, dtype=np.float64, shape=(len(ypos),len(xpos)))
	assert omap.ndim == 2 and omap.shape[-2:] == (len(ypos),len(xpos)) and omap.dtype==np.float64, "omap must be [ny,nx] float64"
	if odomains is None: odomains = np.empty_like(ypos, dtype=np.int32, shape=(len(ypos),len(xpos)))
	assert odomains.ndim == 2 and odomains.shape[-2:] == (len(ypos),len(xpos)) and odomains.dtype==np.int32, "odomains must be [ny,nx] int32"
	# Make point_pix safe
	point_pix = fix_point_pix_separable(ypos, xpos, point_pos, point_pix)
	# Prepare to call C
	cdef int ny = len(ypos)
	cdef int nx = len(xpos)
	cdef inum npoint = point_pos.shape[1]
	cdef double      rmax_   = (0 if rmax is None else rmax)
	cdef double[::1] ypos_   = ypos.reshape(-1)
	cdef double[::1] xpos_   = xpos.reshape(-1)
	cdef double[::1] point_pos_ = point_pos.reshape(-1)
	cdef int[::1]    point_pix_ = point_pix.reshape(-1)
	cdef double[::1] omap_   = omap.reshape(-1)
	cdef int[::1]    odomains_ = odomains.reshape(-1)
	c.distance_from_points_bubble_separable(ny, nx, &ypos_[0], &xpos_[0], npoint, &point_pos_[0], &point_pix_[0], rmax_, &omap_[0], &odomains_[0])
	if domains: return omap, odomains
	else:       return omap

def distance_from_points_cellgrid(ypos, xpos, point_pos, point_pix, dr=np.inf, rmax=None, omap=None, odomains=None, domains=False, bsize=32):
	"""distance_from_points_cellgrid(ypos, xpos, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False)

	dr = np.inf and bsize=32 are based on benchmarking on my laptop. Reducing dr was never
	beneficial and could damage performance significantly if set too low. bsize in the range
	10-50 are OK, but worst-case performance rapidly gets worse after that. The best bsize was
	slightly dependent on the size of the map
	"""
	ypos   = np.asanyarray(ypos).astype(float, order="C", copy=False)
	xpos   = np.asanyarray(xpos).astype(float, order="C", copy=False)
	point_pos = np.asanyarray(point_pos).astype(float,    order="C", copy=False)
	point_pix = np.asanyarray(point_pix).astype(np.int32, order="C", copy=False)
	cdef int ny
	cdef int nx
	cdef int separable
	posmsg = "ypos must be [ny] or [ny,nx] and xpos must be [nx] or [ny,nx]"
	if ypos.ndim == 1:
		assert xpos.ndim == 1, posmsg
		ny, nx= len(ypos), len(xpos)
		separable = 1
	else:
		assert ypos.ndim == 2, posmsg
		assert xpos.shape == ypos.shape, posmsg
		ny, nx = ypos.shape
		separable = 0
	assert point_pos.ndim == 2 and len(point_pos) == 2, "point_pos must be [{dec,ra},npoint]"
	assert point_pix.ndim == 2 and len(point_pix) == 2 and point_pix.shape[1] == point_pos.shape[1], "point_pos must be [{y,x},npoint]"
	if omap is None: omap = np.empty_like(ypos, dtype=np.float64, shape=(ny,nx))
	assert omap.ndim == 2 and omap.shape[-2:] == (ny,nx) and omap.dtype==np.float64, "omap must be [ny,nx] float64"
	if odomains is None: odomains = np.empty_like(ypos, dtype=np.int32, shape=(ny,nx))
	assert odomains.ndim == 2 and odomains.shape[-2:] == (ny,nx) and odomains.dtype==np.int32, "odomains must be [ny,nx] int32"
	bsize = np.zeros(2,np.int32)+bsize
	# Make point_pix safe
	if separable:
		point_pix = fix_point_pix_separable(ypos, xpos, point_pos, point_pix)
	else:
		point_pix = fix_point_pix([ypos, xpos], point_pos, point_pix)
	# Prepare to call C
	cdef inum npoint = point_pos.shape[1]
	cdef double      rmax_   = (0 if rmax is None else rmax)
	cdef double[::1] ypos_   = ypos.reshape(-1)
	cdef double[::1] xpos_   = xpos.reshape(-1)
	cdef double[::1] point_pos_ = point_pos.reshape(-1)
	cdef int[::1]    point_pix_ = point_pix.reshape(-1)
	cdef double[::1] omap_   = omap.reshape(-1)
	cdef int[::1]    odomains_ = odomains.reshape(-1)
	c.distance_from_points_cellgrid(ny, nx, &ypos_[0], &xpos_[0], npoint, &point_pos_[0], &point_pix_[0], bsize[0], bsize[1], rmax_, dr, separable, &omap_[0], &odomains_[0])
	if domains: return omap, odomains
	else:       return omap


def find_edges(mask, flat=False):
	"""find_edges(mask, flat=False)

	Given a 2d numpy array mask[ny,nx], returns an list of indices (y[:],x[:]) for each
	pixel at the edge of a zero region in the mask. If flat==True then it will instead
	return a list of indicees [:] into the flattened mask."""
	# Ensure that we have the right data type and contiguity. Try extra hard to avoid copies
	# when the input is known to have a compatible data type.
	mask = np.asanyarray(mask)
	if mask.dtype in [np.uint8, np.int8, np.bool]:
		mask = mask.astype(mask.dtype, order="C", copy=False).view(np.uint8)
	else:
		mask = mask.astype(np.uint8, order="C", copy=False)
	assert mask.ndim == 2, "mask must be 2D"
	cdef inum n = 0
	cdef inum * edges_raw = NULL;
	cdef uint8_t[::1] mask_ = mask.reshape(-1)
	n = c.find_edges(mask.shape[0], mask.shape[1], &mask_[0], &edges_raw)
	# Copy into numpy array
	cdef inum[::1] view = <inum[:n]> edges_raw;
	edges = np.array(view)
	free(edges_raw)
	if not flat:
		edges = np.unravel_index(edges, mask.shape)
	return edges

def find_edges_labeled(labels, flat=False):
	"""find_edges_labeled(labels, flat=False)

	Given a 2d numpy array labels[ny,nx], returns an list of indices (y[:],x[:]) for each
	pixel at the edge of a region with constant, nonzero value in labels. If flat==True then it will instead
	return a list of indicees [:] into the flattened labels."""
	# Ensure that we have the right data type and contiguity. Try extra hard to avoid copies
	# when the input is known to have a compatible data type.
	labels = np.asanyarray(labels)
	if labels.dtype in [np.uint32, np.int32]:
		labels = labels.astype(labels.dtype, order="C", copy=False).view(np.int32)
	else:
		labels = labels.astype(np.int32, order="C", copy=False)
	assert labels.ndim == 2, "labels must be 2D"
	cdef inum n = 0
	cdef inum * edges_raw = NULL;
	cdef int[::1] labels_ = labels.reshape(-1)
	n = c.find_edges_labeled(labels.shape[0], labels.shape[1], &labels_[0], &edges_raw)
	# Copy into numpy array
	cdef inum[::1] view = <inum[:n]> edges_raw;
	edges = np.array(view)
	free(edges_raw)
	if not flat:
		edges = np.unravel_index(edges, labels.shape)
	return edges

#### healpix stuff below here ####

# Why is this so verbose?
cdef class healpix_info:
	cdef c.healpix_info * info
	def __init__(self, int nside):
		self.info = c.build_healpix_info(nside)
	def __dealloc__(self):
		c.free_healpix_info(self.info)
	@property
	def nside(self):   return self.info.nside
	@property
	def npix(self):   return self.info.npix
	@property
	def ny(self):      return self.info.ny
	@property
	def nx(self):      return np.asarray(<int[:self.info.ny]> self.info.nx)
	@property
	def off(self):     return np.asarray(<inum[:self.info.ny]> self.info.off)
	@property
	def shift(self):   return np.asarray(<int[:self.info.ny]> self.info.shift)
	@property
	def ra0(self):     return np.asarray(<double[:self.info.ny]> self.info.ra0)
	@property
	def dec(self):     return np.asarray(<double[:self.info.ny]> self.info.dec)
	@property
	def cos_dec(self): return np.asarray(<double[:self.info.ny]> self.info.cos_dec)
	@property
	def sin_dec(self): return np.asarray(<double[:self.info.ny]> self.info.sin_dec)

def find_edges_healpix(healpix_info info, mask, flat=True):
	"""find_edges_healpix(healpix_info info, mask, flat=True)

	Given a healpix_info info and a boolean mask[:], returns
	the pixel indices of all the pixels at the edge of the zero
	region of the mask. If flat=False, then the indices will be
	returned as [{row,col},:]. The default is to return them as
	plain 1d indices."""
	# Flat is the default for healpix because healpix is usually 1d
	cdef int * edges_raw
	cdef inum nedge = 0
	mask = np.asanyarray(mask)
	if mask.dtype in [np.uint8, np.int8, np.bool]:
		mask = mask.astype(mask.dtype, order="C", copy=False).view(np.uint8)
	else:
		mask = mask.astype(np.uint8, order="C", copy=False)
	assert mask.ndim == 1, "mask must be 1D"
	cdef uint8_t[::1] mask_ = mask
	nedge = c.find_edges_healpix(info.info, &mask_[0], &edges_raw)
	# Copy info numpy array
	cdef int[::1] view = <int[:2*nedge]> edges_raw
	edges = np.array(view).reshape(2,-1)
	free(edges_raw)
	if flat:
		edges = info.off[edges[0]]+edges[1]
	return edges

def find_edges_labeled_healpix(healpix_info info, labels, flat=True):
	"""find_edges_labeled_healpix(healpix_info info, labels, flat=True)

	Given a healpix_info info and integer labels[:], returns
	the pixel indices of all the pixels at the the edge of a
	region with the same, nonzero value in labels. If flat=False,
	then the indices will be returned as [{row,col},:]. The default
	is to return them as plain 1d indices."""
	cdef int * edges_raw
	cdef inum nedge = 0
	labels = np.asanyarray(labels)
	if labels.dtype in [np.uint32, np.int32]:
		labels = labels.astype(labels.dtype, order="C", copy=False).view(np.int32)
	else:
		labels = labels.astype(np.int32, order="C", copy=False)
	assert labels.ndim == 1, "labels must be 2D"
	cdef int[::1] labels_ = labels.reshape(-1)
	nedge = c.find_edges_labeled_healpix(info.info, &labels_[0], &edges_raw)
	# Copy info numpy array
	cdef int[::1] view = <int[:2*nedge]> edges_raw
	edges = np.array(view).reshape(2,-1)
	free(edges_raw)
	if flat:
		edges = info.off[edges[0]]+edges[1]
	return edges

def distance_from_points_healpix(healpix_info info, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False, method="bubble"):
	"""distance_from_points_healpix(healpix_info info, point_pos, point_pix, rmax=None, omap=None, odomains=None, domains=False)

	Computes the distance of each healpix pixel in the sky described by healpix_info info to
	the closest of the points with position point_pos[{dec,ra},npoint] and corresponding
	pixel position point_pix[{y,x},npoint]. If omap is specified, the result is written there.
	In any case the result is also returned. If domains=True then a domain map is also returned.
	In this map the value of each pixel is the index of the closest point in the point_pos array,
	or -1 if no valid point was found. The main way this could happen is if rmax is specified,
	which makes the search stop after a distance of rmax radians. Distances larger than this
	will be capped to this value."""
	# Check that our inputs make sense
	point_pos = np.asanyarray(point_pos).astype(float,    order="C", copy=False)
	assert point_pos.ndim == 2 and len(point_pos) == 2, "point_pos must be [{dec,ra},npoint]"
	# point_pix can be standard 1d or our internal 2d format
	point_pix = np.asanyarray(point_pix)
	if point_pix.ndim == 1:
		point_pix = unravel_healpix(info, point_pix)
	point_pix = point_pix.astype(np.int32, order="C", copy=False)
	assert point_pix.ndim == 2 and len(point_pix) == 2 and point_pix.shape[1] == point_pos.shape[1], "point_pos must be [npoint] or [{y,x},npoint]"
	if omap is None: omap = np.empty(info.npix, dtype=np.float64)
	assert omap.ndim == 1 and omap.size == info.npix and omap.dtype==np.float64, "omap must be [npix] float64"
	if odomains is None: odomains = np.empty(info.npix, dtype=np.int32)
	assert odomains.ndim == 1 and odomains.size == info.npix and odomains.dtype==np.int32, "odomains must be [npix] int32"
	# Prepare to call C
	cdef inum npoint = point_pos.shape[1]
	cdef double      rmax_   = (0 if rmax is None else rmax)
	cdef double[::1] point_pos_ = point_pos.reshape(-1)
	cdef int[::1]    point_pix_ = point_pix.reshape(-1)
	cdef double[::1] omap_   = omap.reshape(-1)
	cdef int[::1]    odomains_ = odomains.reshape(-1)
	if method == "bubble":
		c.distance_from_points_bubble_healpix(info.info, npoint, &point_pos_[0], &point_pix_[0], rmax_, &omap_[0], &odomains_[0])
	elif method == "heap":
		c.distance_from_points_heap_healpix(info.info, npoint, &point_pos_[0], &point_pix_[0], rmax_, &omap_[0], &odomains_[0])
	else: raise ValueError("Unrecognized method '%s'" % str(method))
	if domains: return omap, odomains
	else:       return omap

def unravel_healpix(healpix_info info, pix1d):
	pix1d = np.asanyarray(pix1d).astype(np.int64, order="C", copy=False)
	pix2d = np.empty((2,)+pix1d.shape, np.int32)
	cdef npoint = pix1d.size
	cdef inum[::1] pix1d_ = pix1d.reshape(-1)
	cdef int[::1]  pix2d_ = pix2d.reshape(-1)
	c.unravel_healpix(info.info, npoint, &pix1d_[0], &pix2d_[0])
	return pix2d

def ravel_healpix(healpix_info info, pix2d):
	pix2d = np.asanyarray(pix2d).astype(np.int32, order="C", copy=False)
	pix1d = np.empty(pix2d.shape[1:], np.int64)
	cdef npoint = pix1d.size
	cdef int[::1]  pix2d_ = pix2d.reshape(-1)
	cdef inum[::1] pix1d_ = pix1d.reshape(-1)
	c.ravel_healpix(info.info, npoint, &pix2d_[0], &pix1d_[0])
	return pix1d

def get_healpix_neighs(healpix_info info, int y, int x):
	res = np.zeros([2,4],np.int32)
	cdef int[::1] yneigh = res[0]
	cdef int[::1] xneigh = res[1]
	c.get_healpix_neighs(info.info, y, x, &yneigh[0], &xneigh[0])
	return res
