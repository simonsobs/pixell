import numpy as np, os, warnings
from . import enmap, utils

def zeros(tile_geom, dtype=np.float64):
	"""Construct a zero-initialized TileMap with the given TileGeometry and data type"""
	flat = np.zeros(tile_geom.pre + (np.sum(tile_geom.npixs[tile_geom.active]),), dtype)
	return TileMap(flat, tile_geom.copy())

def empty(tile_geom, dtype=np.float64):
	"""Construct a zero-initialized TileMap with the given TileGeometry and data type"""
	flat = np.empty(tile_geom.pre + (np.sum(tile_geom.npixs[tile_geom.active]),), dtype)
	return TileMap(flat, tile_geom.copy())

def full(tile_geom, val, dtype=np.float64):
	"""Construct a zero-initialized TileMap with the given TileGeometry and data type"""
	flat = np.full(tile_geom.pre + (np.sum(tile_geom.npixs[tile_geom.active]),), val, dtype)
	return TileMap(flat, tile_geom.copy())

def from_tiles(tiles, tile_geom):
	"""Construct a TileMap from a set of a full list of tiles, both active
	and inactive. Inactive tiles are indicated with None entries. The active
	information in tile_geom is ignored, as is the non-pixel part of tile_geom.shape,
	which is instead inferred from the tiles."""
	active_tiles = []
	active       = []
	for gi, tile in enumerate(tiles):
		if tile is None: continue
		active_tiles.append(tile)
		active.append(gi)
	return from_active_tiles(active_tiles, tile_geom.copy(active=active))

def from_active_tiles(tiles, tile_geom):
	"""Construct a TileMap from a list of active tiles that should match the
	active list in the provided tile geometry. The non-pixel part of tile_geom
	is ignored, and is instead inferred from the tile shapes."""
	if len(tiles) != tile_geom.nactive:
		raise ValueError("Wrong number of tiles passed. Expected %d but got %d" % (tile_geom.nactive, len(tiles)))
	if len(tiles) == 0: return zeros(tile_geom)
	data = np.concatenate([tile.reshape(tile.shape[:-2]+(-1,)) for tile in tiles],-1)
	return TileMap(data, tile_geom.copy(pre=data.shape[:-1]))

class TileMap(np.ndarray):
	"""Implements a sparse tiled map, as described by a TileGeometry. This is effectively
	a large enmap that has been split into tiles, of which only a subset is stored. This
	is implemented as a subclass of ndarray instead of a list of tiles to allow us to
	transparently perform math operations on it. The maps are stored stored as a single
	array with all tiles concatenated along a flattened pixel dimension, in the same
	order as in tile_geom.active.

	Example: A TileMap covering a logical area with shape (3,100,100) with (10,10) tiles
	and active tiles [7,5] will have a shape of (3,200=10*10*2) when accessed directly.
	When accessed through the .tiles view, .tiles[5] will return a view of an (3,10,10) enmap,
	as will .tiles[7]. For all other indices, .tiles[x] will return None. The same
	two tiles can be accessed as .active_tiles[1] and .active_tiles[0] respecitvely.

	Slicing the TileMap using the [] operator works. For all but the last axis, this
	does what you would expect. E.g. for the above example, tile_map[0].tiles[5] would
	return a view of a (10,10) enmap (so the first axis is gone). If the last axis,
	which represents a flattened view of the pixels and tiles, is sliced, then the
	returned object will be a plain numpy array.
	"""
	def __new__(cls, arr, tile_geom):
		"""Constructs a TileMap object given a raw array arr[...,totpix] and
		a tuple of geometries."""
		obj = np.asarray(arr).view(cls)
		obj.geometry  = tile_geom
		return obj
	def __array_finalize__(self, obj):
		if obj is None: return
		self.geometry = getattr(obj, "geometry", None)
	def __repr__(self):
		return "TileMap(%s,%s)" % (np.asarray(self), str(self.geometry))
	def __str__(self): return repr(self)
	def __array_wrap__(self, arr, context=None):
		return TileMap(arr, self.geometry)
	def __getitem__(self, sel):
		# Split sel into normal and wcs parts.
		sel1, sel2 = utils.split_slice(sel, [self.ndim-1,1])
		if len(sel2) > 1:
			raise IndexError("too many indices")
		elif len(sel2) == 1:
			# Degrade to plain array if we index the last, special pixel/tile axis
			return np.ndarray.__getitem__(self, sel)
		else:
			res  = np.ndarray.__getitem__(self, sel)
			ogeo = self.geometry.copy()
			ogeo.shape = res.shape[:-1]+self.geometry.shape[-2:]
			return TileMap(res, ogeo)
	def __getslice__(self, a, b=None, c=None): return self[slice(a,b,c)]
	def contig(self): return TileMap(np.ascontiguousarray(self), self.geometry)
	@property
	def pre(self): return self.geometry.pre
	@property
	def ntile(self): return self.geometry.ntile
	@property
	def nactive(self): return self.geometry.nactive
	def copy(self, order='K'):
		return TileMap(np.copy(self,order), self.geometry.copy())
	@property
	def tiles(self):
		return _TileView(self, active=False)
	@property
	def active_tiles(self):
		return _TileView(self, active=True)
	def with_tiles(self, other, strict=False):
		"""If a and b are TileMaps with the same overall tiling but different
		active tile sets, then c = a.with_tiles(b) will make c a TileMap
		with the union of the active tiles of a and b and the data from a
		(new tiles are zero-initialized).

		If strict==True, then c will have exactly the active tiles of b,
		in exactly that order. Binary operations on strictly compatible arrays
		should be considerably faster."""
		try: active = other.geometry.active
		except AttributeError: active = _parse_active(other, self.ntile)
		if np.all(active == self.geometry.active):
			return self.copy()
		# Construct the new geometry
		if strict: new_geom = self.geometry.copy(active    =active)
		else:      new_geom = self.geometry.copy(add_active=active)
		# Construct the new array
		res = zeros(new_geom, dtype=self.dtype)
		# Copy over data. Not optimized
		for gi in res.geometry.active:
			ai = self.geometry.lookup[gi]
			if ai >= 0: res.tiles[gi] = self.active_tiles[ai]
		return res
	# Forward some properties of TileGeometry
	@property
	def active(self): return self.geometry.active
	@property
	def lookup(self): return self.geometry.lookup
	@property
	def nactive(self): return self.geometry.nactive
	@property
	def ntile(self): return self.geometry.ntile
	@property
	def tile_shape(self): return self.geometry.tile_shape

class _TileView:
	"""Helper class used to implement access to the individual tiles that make up a TileMap object"""
	def __init__(self, tile_map, active=True):
		self.tile_map = tile_map
		self.active   = active
		self.offs     = utils.cumsum(tile_map.geometry.npixs[tile_map.geometry.active], endpoint=True)
	def __len__(self):
		if self.active: return len(self.tile_map.geometry.active)
		else:           return self.tile_map.geometry.ntile
	def __getitem__(self, sel):
		"""Get a single tile or subset of a tile from the TileMap. The first
		entry in the slice must be an integer - general slicing in the tile axis is not
		supported, though it could be added. The rest of the indices can be anything an
		enmap will accept."""
		if isinstance(sel, int):
			# Optimize common use case by avoiding slice decoding.
			# Doesn't save much time, really. 0.1 ms when combined with the
			# sel2 check below
			i      = sel
			sel2   = ()
		else:
			sel1, sel2 = utils.split_slice(sel, [1,self.tile_map.ndim+2-1])
			if len(sel1) == 0: return self.tile_map
			i      = sel1[0]
		geo  = self.tile_map.geometry
		if self.active:
			ai, gi = i, geo.active[i]
		else:
			ai, gi = geo.lookup[i], i
			# Return None for inactive tiles since that's what so3g does.
			# But consider raising an exception instead.
			if ai < 0: return None
		if ai < 0 or ai >= self.tile_map.nactive:
			raise IndexError("Active tile index %d (global %d) is out of bounds for TileMap with %d active tiles" % (ai, gi, self.tile_map.nactive))
		tile_info = geo.tiles[gi]
		tile = enmap.ndmap(self.tile_map[...,self.offs[ai]:self.offs[ai+1]].reshape(self.tile_map.pre + tile_info.shape[-2:]), tile_info.wcs)
		# Apply any slicing of the tile itself
		if len(sel2) > 0: tile = tile[sel2]
		return tile
	def __setitem__(self, sel, val):
		"""Set a single tile or subset of a tile to the given value, which can be a number
		or a compatibly shaped array. The first entry in the slice must be an integer -
		general slicing in the tile axis is not supported, though it could be added.
		The rest of the indices can be anything an enmap will accept. This relies on
		slicing not producing copies, and may fail if the TileMap is not contiguous."""
		sel1, sel2 = utils.split_slice(sel, [1,self.tile_map.ndim+2-1])
		if len(sel1) == 0: return self.tile_map
		i    = sel1[0]
		geo  = self.tile_map.geometry
		if self.active:
			ai, gi = i, geo.active[i]
		else:
			ai, gi = geo.lookup[i], i
			# Return None for inactive tiles since that's what so3g does.
			# But consider raising an exception instead.
			if ai < 0: raise IndexError("Tile %d is not active" % gi)
		# This assumes that the slicing and reshaping won't cause copies. This should be the case if
		# self.tile_map is contiguous. To be robust I should detect if a copy would be made, and fall back
		# on something slower in that case
		tile_info = geo.tiles[gi]
		enmap.ndmap(self.tile_map[...,self.offs[ai]:self.offs[ai+1]].reshape(self.tile_map.pre + tile_info.shape[-2:]), tile_info.wcs)[sel2] = val
	def __iter__(self):
		"""Iterator access. Faster than __getitem__ due to not having to resolve
		the more complicated slicing it supports."""
		geo = self.tile_map.geometry
		if self.active: items = [(ai,geo.active[ai]) for ai in range(geo.nactive)]
		else: items = [(geo.lookup[gi],gi) for gi in range(geo.ntile) if geo.lookup[gi] >= 0]
		for ai, gi in items:
				tile_info = geo.tiles[geo.active[ai]]
				yield enmap.ndmap(self.tile_map[...,self.offs[ai]:self.offs[ai+1]].reshape(self.tile_map.pre + tile_info.shape[-2:]), tile_info.wcs)

# Math operations. These are automatically supported by virtue of being a numpy subclass,
# but the functions below add support for expanding the active tiles automatically when TileMaps
# with incompatible active tiles are combined.
def make_binop(op, is_inplace=False):
	def binop(self, other):
		opfun = getattr(np.ndarray, op)
		if isinstance(other, TileMap): # could be replaced with a try statement for duck typing
			comp = self.geometry.compatible(other.geometry)
			if comp == 0:
				# Not compatible
				raise ValueError("operands could not be broadcast together with geometries %s and %s" % (str(self.geometry), str(other.geometry)))
			elif comp == 1:
				# Loosely compatible. Requires manual work. Slow, both due to looping and
				# poor memory localization.
				if is_inplace:
					# We can't change the shape of the output array for an in-place operation
					if np.any(self.geometry.lookup[other.geometry.active]<0):
						raise ValueError("Cannot expand TileMap with geometry %s to include active tiles in geometry %s in in-place operation" % (str(self.geometry), str(other.geometry)))
					opre = utils.broadcast_shape(self.pre, other.pre)
					if opre != self.pre:
						raise ValueError("operands could not be broadcast together with geometries %s and %s" % (str(self.geometry), str(other.geometry)))
					for gi in other.geometry.active:
						self.tiles[gi] = opfun(self.tiles[gi], other.tiles[gi])
					return self
				else:
					# Not in-place, so we have more flexibility. First build the output map
					oact = np.unique(np.concatenate([self.geometry.active, other.geometry.active]))
					otype= np.result_type(self.dtype, other.dtype)
					oshape = utils.broadcast_shape(self.pre, other.pre) + self.geometry.shape[-2:]
					ogeo = geometry(oshape, self.geometry.wcs, tile_shape=self.geometry.tile_shape, active=oact)
					out  = zeros(ogeo, otype)
					# Copy over our old values
					for gi in self.geometry.active:
						out.tiles[gi] = self.tiles[gi]
					# Then update with valus from other
					for gi in other.geometry.active:
						out.tiles[gi] = opfun(out.tiles[gi], other.tiles[gi])
					return out
			else:
				# Fully compatible. Handle outside
				pass
		# Handle fully compatible or plain array. Fast.
		out =  opfun(self, other)
		out = TileMap(out, self.geometry.copy(pre=out.shape[:-1]))
		return out
	return binop

for op in ["__add__", "__sub__", "__mul__", "__pow__", "__truediv__", "__floordiv__",
		"__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__",
		"__and__", "__or__", "__xor__", "__lshift__", "__rshift__"]:
	setattr(TileMap, op, make_binop(op))
for op in ["__iadd__", "__isub__", "__imul__", "__ipow__", "__itruediv__", "__ifloordiv__",
		"__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__"]:
	setattr(TileMap, op, make_binop(op, is_inplace=True))

########################################
############ Geometry stuff ############
########################################

def geometry(shape, wcs, tile_shape=(500,500), active=[]):
	"""TileGeometry constructor.
	shape, wcs: The enmap geometry of the full space the tiling covers.
	tile_shape: The (ny,nx) vertical and horizontal tile shape in pixels.
	active:     The list of active tile indices."""
	shape      = tuple(shape)
	wcs        = wcs
	tile_shape = tuple(np.zeros(2,int)+tile_shape) # broadcast to len(2)
	# Get the grid shape by rounding up to include any partial tiles at the edge
	grid_shape = tuple([(s+ts-1)//ts for s,ts in zip(shape[-2:], tile_shape[-2:])])
	ntile      = grid_shape[0]*grid_shape[1]
	# Compute all the individual tile shapes. Should maybe be done on-the-fly
	tile_shapes= np.zeros(grid_shape + (2,), int)
	tile_shapes[:, :] = tile_shape
	tile_shapes[-1,:,0] = np.minimum(tile_shape[-2], (shape[-2]-1)%tile_shape[-2]+1)
	tile_shapes[:,-1,1] = np.minimum(tile_shape[-1], (shape[-1]-1)%tile_shape[-1]+1)
	tile_shapes = tile_shapes.reshape(-1,2)
	# Number of pixels per tile
	npixs       = tile_shapes[:,0]*tile_shapes[:,1]
	# And the active tile list and reverse lookup
	active      = np.array(active,int)
	lookup      = np.full(ntile,-1,int)
	lookup[active] = np.arange(len(active))
	# Call the raw constructor
	return TileGeometry(shape, wcs, tile_shape, grid_shape, tile_shapes, npixs, active, lookup)

class TileGeometry:
	def __init__(self, shape, wcs, tile_shape, grid_shape, tile_shapes, npixs, active, lookup):
		"""Raw constructor for a TileGeometry. You normally don't want to use this. Use
		tilemap.geometry() instead."""
		self.shape      = shape
		self.wcs        = wcs
		self.tile_shape = tile_shape
		self.grid_shape = grid_shape
		self.ntile      = self.grid_shape[0]*self.grid_shape[1]
		self.tile_shapes= tile_shapes
		self.npixs      = npixs
		self.active     = active
		self.lookup     = lookup
	def grid2ind(self, ty, tx):
		"""Get the index of the tile wiht grid coordinates ty,tx in the full tiling"""
		return ty*self.grid_shape[1]+tx
	def ind2grid(self, i):
		"""Get the tile grid coordinates ty, tx for tile #i in the full tiling"""
		nx = self.grid_shape[-1]
		return i//nx, i%nx
	def copy(self, pre=None, active=None, add_active=None):
		if pre is not None: shape = tuple(pre) + self.shape[-2:]
		else:               shape = self.shape
		_active = self.active.copy()
		lookup  = self.lookup.copy()
		if active is not None or add_active is not None:
			# Allow us to override these, which will require recalculation of lookup
			if active is not None:
				_active = _parse_active(active, self.ntile)
				lookup  = np.full(self.ntile,-1,int)
				lookup[_active] = np.arange(len(_active))
			if add_active is not None:
				add_active = _parse_active(add_active, self.ntile)
				_active = np.concatenate([_active, add_active[lookup[add_active]<0]])
				lookup[_active] = np.arange(len(_active))
		return TileGeometry(shape, self.wcs, self.tile_shape, self.grid_shape, self.tile_shapes.copy(), self.npixs.copy(), _active, lookup)
	@property
	def pre(self): return self.shape[:-2]
	@property
	def nactive(self): return len(self.active)
	@property
	def tiles(self):
		"""Allow us to get the enmap geometry of tile #i by writing
		tile_geom.tiles[i]"""
		return _TileGeomHelper(self)
	def __repr__(self): return "TileGeometry(%s, %s, tile_shape=%s, active=%s)" % (str(self.shape), str(self.wcs), str(self.tile_shape), str(self.active))
	def compatible(self, other):
		"""Return our compatibility with binary operations with other.
		The return value can be 2, 1 or 0:
		2: Strictly compatible. Both the logical geometry (shape, wcs), tile shape
		   and active tiles match. This allows for direct numpy operations without
		   any manual looping over tiles.
		1: Loosely compatible. The logical and tile geometry match, but not the active tiles.
		0. Not compatible."""
		if tuple(self.shape[-2:]) != tuple(other.shape[-2:]): return 0
		if tuple(self.tile_shape) != tuple(other.tile_shape): return 0
		if self.nactive == other.nactive and np.all(self.active == other.active): return 2
		else: return 1

class _TileGeomHelper:
	def __init__(self, tile_geom):
		self.tile_geom = tile_geom
	def __getitem__(self, i):
		"""Return the geometry of index #i in the full tiling"""
		g  = self.tile_geom
		ty, tx = g.ind2grid(i)
		y1 = ty*g.tile_shape[-2]
		x1 = tx*g.tile_shape[-1]
		y2 = min(y1+g.tile_shape[-2], g.shape[-2])
		x2 = min(x1+g.tile_shape[-1], g.shape[-1])
		return enmap.Geometry(g.shape, g.wcs)[...,y1:y2,x1:x2]

def _parse_active(active, ntile):
	if utils.streq(active, "all"): return np.arange(ntile,dtype=int)
	else: return np.asarray(active,int)
