# Tiled map support. Will be moved to pixell or sotodlib later. Very similar to multimap
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

class TileMap(np.ndarray):
	"""Implements a sparse tiled map, as described by a TileGeometry. This is effectively
	a large enmap that has been split into tiles, of which only a subset is stored. This
	is implemented as a subclass of ndarray instead of a list of tiles to allow us to
	transparently perform math operations on it. The maps are stored stored as a single
	array with all tiles concatenated along a flattened pixel dimension, in the same
	order as in tile_geom.active.

	Example: A TileMap covering a logical area with shape (3,100,100) with (10,10) tiles
	and active tiles [7,5] will have a shape of (3,200=10*10*2) when accessed directly.
	When accessed through the .tile view, .tile[5] will return a view of an (3,10,10) enmap,
	as will .tile[7]. For all other indices, .tile[x] will return None. The same
	two tiles can be accessed as .active_tile[1] and .active_tile[0] respecitvely.

	Slicing the TileMap using the [] operator works. For all but the last axis, this
	does what you would expect. E.g. for the above example, tile_map[0].tile[5] would
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
	def tile(self):
		return _TileView(self, active=False)
	@property
	def active_tile(self):
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
		except AttributeError: active = np.asarray(other, int)
		if np.all(active == self.geometry.active):
			return self.copy()
		# Construct the new geometry
		new_geom = self.geometry.copy()
		if strict: new_geom.set_active(active)
		else:      new_geom.add_active(active)
		# Construct the new array
		res = zeros(new_geom, dtype=self.dtype)
		# Copy over data. Not optimized
		for gi in res.geometry.active:
			ai = self.geometry.lookup[gi]
			if ai >= 0: res.tile[gi] = self.active_tile[ai]
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
			if ai < 0: return None
		if ai < 0 or ai >= self.tile_map.nactive:
			raise IndexError("Active tile index %d (global %d) is out of bounds for TileMap with %d active tiles" % (ai, gi, self.tile_map.nactive))
		return enmap.ndmap(self.tile_map[...,self.offs[ai]:self.offs[ai+1]].reshape(self.tile_map.pre + self.tile_map.geometry.tile[gi].shape[-2:]), self.tile_map.geometry.tile[gi].wcs)[sel2]
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
		enmap.ndmap(self.tile_map[...,self.offs[ai]:self.offs[ai+1]].reshape(self.tile_map.pre + self.tile_map.geometry.tile[gi].shape[-2:]), self.tile_map.geometry.tile[gi].wcs)[sel2] = val

# Math operations. These are automatically supported by virtue of being a numpy subclass,
# but the functions below add support for expanding the active tiles automatically when TileMaps
# with incompatible active tiles are combined.
def make_binop(op, is_inplace=False):
	def binop(self, other):
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
						self.tile[gi] = getattr(np.ndarray, op)(self.tile[gi], other.tile[gi])
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
						out.tile[gi] = self.tile[gi]
					# Then update with valus from other
					for gi in other.geometry.active:
						out.tile[gi] = getattr(np.ndarray, op)(out.tile[gi], other.tile[gi])
					return out
			else:
				# Fully compatible. Handle outside
				pass
		# Handle fully compatible or plain array. Fast.
		out =  getattr(np.ndarray, op)(self, other)
		out = TileMap(out, self.geometry.with_pre(out.shape[:-1]))
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
	"""TileGeometry constructor. Equivalent to TileGeometry.__init__"""
	return TileGeometry(shape, wcs, tile_shape=tile_shape, active=active)

class TileGeometry:
	def __init__(self, shape, wcs, tile_shape=(500,500), active=[]):
		"""Initialize a TileGeometry that covers a logical enmap geometry given by
		shape, wcs, which is tiled by tiles of shape tile_shape in pixels, with
		only the tiles whose 1d index is listed in active actually stored."""
		self.shape      = tuple(shape)
		self.wcs        = wcs
		self.tile_shape = tuple(np.zeros(2,int)+tile_shape) # broadcast to len(2)
		# Derived properties. First the grid dimensions
		self.grid_shape = tuple([(s+ts-1)//ts for s,ts in zip(self.shape[-2:], self.tile_shape[-2:])])
		self.ntile      = self.grid_shape[0]*self.grid_shape[1]
		# The actual shape of each tile. Can be smaller than tile_shape at the edge
		self.tile_shapes= np.zeros(self.grid_shape + (2,), int)
		self.tile_shapes[:, :] = self.tile_shape
		self.tile_shapes[-1,:,0] = np.minimum(self.tile_shape[-2], (self.shape[-2]-1)%self.tile_shape[-2]+1)
		self.tile_shapes[:,-1,1] = np.minimum(self.tile_shape[-1], (self.shape[-1]-1)%self.tile_shape[-1]+1)
		self.tile_shapes = self.tile_shapes.reshape(-1,2)
		self.npixs       = self.tile_shapes[:,0]*self.tile_shapes[:,1]
		# Set the active tiles
		self.set_active(active)
	# Mutating methods
	def set_active(self, active):
		"""Replace our the list of active indices with those provided active. Changes the
		current object."""
		self.active   = np.array(active,int)
		self.lookup   = np.full(self.ntile,-1,int)
		self.lookup[active] = np.arange(len(active))
		return self
	def add_active(self, active):
		"""Add the list of active indices active to our current list. Entries already in
		active are ignored. Changes the current object."""
		active = np.asarray(active,int)
		new    = active[self.lookup[active]<0]
		if len(new) > 0:
			self.set_active(np.concatenate([self.active, new]))
		return self
	# Non-mutating methods. It seems a bit excessive to have all of set/add/with/plus...
	def with_active(self, active):
		"""As set_active, but returns a new object, leaving the current one unchanged."""
		return self.copy().set_active(active)
	def plus_active(self, active):
		"""As add_active, but returns a new object, leaving the current one unchanged."""
		return self.copy().add_active(active)
	def with_pre(self, pre):
		res = self.copy()
		res.shape = tuple(pre) + self.shape[-2:]
		return res
	def grid2ind(self, ty, tx):
		"""Get the index of the tile wiht grid coordinates ty,tx in the full tiling"""
		return ty*self.grid_shape[1]+tx
	def ind2grid(self, i):
		"""Get the tile grid coordinates ty, tx for tile #i in the full tiling"""
		nx = self.grid_shape[-1]
		return i//nx, i%nx
	def copy(self): return TileGeometry(self.shape, self.wcs.deepcopy(), self.tile_shape, self.active)
	@property
	def pre(self): return self.shape[:-2]
	@property
	def nactive(self): return len(self.active)
	@property
	def tile(self):
		"""Allow us to get the enmap geometry of tile #i by writing
		tile_geom.tile[i]"""
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
