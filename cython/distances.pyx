import cython, numpy as np
cimport numpy as np
from libc.stdlib cimport free
from distances cimport inum
from distances cimport find_edges as find_edges_c
from distances cimport find_edges_labeled as find_edges_labeled_c
from distances cimport distance_from_points as distance_from_points_c
from distances cimport distance_from_points_separable as distance_from_points_separable_c
from distances cimport distance_from_points_treerings_separable as distance_from_points_treerings_separable_c

__version__ = 1.0

def distance_from_points(posmap, points, omap=None, odomains=None, domains=False):
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
		distance_from_points_c(npix, &posmap_[0], npoint, &points_[0], &omap_[0], NULL)
		return omap
	else:
		odomains_ = odomains.reshape(-1)
		distance_from_points_c(npix, &posmap_[0], npoint, &points_[0], &omap_[0], &odomains_[0])
		return omap, odomains

def distance_from_points_separable(ypos, xpos, points, omap=None, odomains=None, domains=False):
	"""distance_from_points_separable(ypos, xpos, points, omap=None, odomains=None, domains=False)

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
		distance_from_points_separable_c(ny, nx, &ypos_[0], &xpos_[0], npoint, &points_[0], &omap_[0], NULL)
		return omap
	else:
		odomains_ = odomains.reshape(-1)
		distance_from_points_separable_c(ny, nx, &ypos_[0], &xpos_[0], npoint, &points_[0], &omap_[0], &odomains_[0])
		return omap, odomains

def distance_from_points_treerings_separable(ypos, xpos, points, point_y, point_x, omap=None, odomains=None, domains=False):
	"""distance_from_points_separable(ypos, xpos, points, omap=None, odomains=None, domains=False)

	Like distance_from_points, but optimized for the case where the coordinate system
	is separable, as is typically the case for cylindrical projections. Instead of a full
	posmap[{dec,ra},ny,nx] it takes ypos[ny] which gives the dec of each point along the y axis
	and xpos[nx] which gives the ra of each point along the x axis. The main advantage of this
	is that one can avoid the somewhat heavy computation of the full posmap."""
	# Check that our inputs make sense
	ypos   = np.asanyarray(ypos).astype(float, order="C", copy=False)
	xpos   = np.asanyarray(xpos).astype(float, order="C", copy=False)
	points = np.asanyarray(points).astype(float, order="C", copy=False)
	point_y= np.asanyarray(point_y).astype(np.int32, order="C", copy=False)
	point_x= np.asanyarray(point_x).astype(np.int32, order="C", copy=False)
	assert ypos.ndim == 1, "ypos must be [ny]"
	assert xpos.ndim == 1, "xpos must be [nx]"
	assert points.ndim == 2 and len(points) == 2, "points must be [{dec,ra},npoint]"
	assert point_y.ndim == 1 and point_y.size == points.shape[1], "point_y must be [npoint]"
	assert point_x.ndim == 1 and point_x.size == points.shape[1], "point_x must be [npoint]"
	if omap is None: omap = np.empty_like(ypos, dtype=np.float64, shape=(len(ypos),len(xpos)))
	assert omap.ndim == 2 and omap.shape[-2:] == (len(ypos),len(xpos)) and omap.dtype==np.float64, "omap must be [ny,nx] float64"
	if odomains is None: odomains = np.empty_like(ypos, dtype=np.int32, shape=(len(ypos),len(xpos)))
	assert odomains.ndim == 2 and odomains.shape[-2:] == (len(ypos),len(xpos)) and odomains.dtype==np.int32, "odomains must be [ny,nx] int32"
	# Prepare to call C
	cdef int ny = len(ypos)
	cdef int nx = len(xpos)
	cdef inum npoint = points.shape[1]
	cdef double[::1] ypos_   = ypos.reshape(-1)
	cdef double[::1] xpos_   = xpos.reshape(-1)
	cdef double[::1] points_ = points.reshape(-1)
	cdef int[::1]    point_y_= point_y
	cdef int[::1]    point_x_= point_x
	cdef double[::1] omap_   = omap.reshape(-1)
	cdef int[::1]    odomains_ = odomains.reshape(-1)
	distance_from_points_treerings_separable_c(ny, nx, &ypos_[0], &xpos_[0], npoint, &points_[0], &point_y_[0], &point_x_[0], &omap_[0], &odomains_[0])
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
	n = find_edges_c(mask.shape[0], mask.shape[1], &mask_[0], &edges_raw)
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
	n = find_edges_labeled_c(labels.shape[0], labels.shape[1], &labels_[0], &edges_raw)
	# Copy into numpy array
	cdef inum[::1] view = <inum[:n]> edges_raw;
	edges = np.array(view)
	free(edges_raw)
	if not flat:
		edges = np.unravel_index(edges, labels.shape)
	return edges
