import numpy as np
from pixell import bunch

class Device:
	def __init__(self):
		self.pools = None # Memory pools
		self.np    = None # numpy or equivalent
		self.lib   = bunch.Bunch() # place to store library functions
	def get(self, arr): raise NotImplementedError # copy device array to cpu

class DeviceCpu(Device):
	def __init__(self, align=16, alloc_factory=None):
		super().__init__()
		if alloc_factory is None:
			def alloc_factory(name):
				return ArrayPoolCpu(AllocAligned(AllocCpu(), align=align), name=name)
		self.pools = ArrayMultipool(alloc_factory)
		self.np    = np
	def get(self, arr): return arr.copy()

class DeviceGpu:
	def __init__(self, align=512, alloc_factory=None):
		import cupy
		if alloc_factory is None:
			def alloc_factory(name):
				return ArrayPoolGpu(AllocAligned(AllocGpu(), align=align), name=name)
		self.pools = ArrayMultipool(alloc_factory)
		self.np    = cupy
	def get(self, arr): return arr.get()

class AllocCpu:
	def alloc(self, n): return np.empty(n, dtype=np.uint8)

class AllocGpu:
	def __init__(self):
		import cupy
		self.allocator = cupy.cuda.get_allocator()
	def alloc(self, n):
		import cupy
		memptr = self.allocator(n)
		return cupy.ndarray(n, np.uint8, memptr=self.allocator(n))

class AllocAligned:
	"""Wraps an allocator to make it aligned. Should work for both cpu and gpu.
	A bit inefficient if the underlying allocator is already aligned, which it
	probably already is."""
	def __init__(self, allocator, align=16):
		self.allocator = allocator
		self.align     = align
	def alloc(self, n):
		buf = self.allocator.alloc(n+self.align-1)
		off = (-getptr(buf))%self.align
		return buf[off:off+n]

class Mempool:
	def __init__(self, aligned_alloc, name="[unnamed]"):
		self.allocator = aligned_alloc
		self.name      = name
		self.free()
	def alloc(self, n):
		effsize = round_up(n, self.allocator.align)
		if len(self.arenas) == 0 or self.arenas[-1].size < self.pos + n:
			# We don't have room. Make more
			self.arenas.append(self.allocator.alloc(n))
			buf = self.arenas[-1][0:n]
			self.pos   = effsize
			self.size += effsize
		else:
			# We have room. Parsel out some more
			buf       = self.arenas[-1][0:n]
			self.pos += effsize
		return buf
	def free(self):
		self.arenas    = []
		self.pos       = 0
		self.size      = 0
	def reset(self):
		"""Invalidate the memory we point to, potentially cleaning it up to a single
		fixed area we can allocate from. New allocations will reuse this memory
		as long as they don't exceed its capacity. If the capacity is exceeded, then
		it will start requesting new memory again."""
		self.pos = 0
		if   self.size == 0: self.arenas = []
		elif len(self.arenas) != 1 or self.arenas[0].size != self.size:
			# Free up our old arenas, and make our hopefully final one
			self.arenas = [self.allocator.alloc(self.size)]
	def __repr__(self): return "%s(name='%s', size=%d, free=%d, align=%d)" % (self.__class__.__name__, self.name, self.size, self.size-self.pos, self.allocator.align)

class ArrayPoolCpu(Mempool):
	def array(self, arr):
		arr  = np.asarray(arr)
		oarr = self.empty(arr.shape, dtype=arr.dtype)
		oarr[:] = arr
		return oarr
	def empty(self, shape, dtype=np.float32):
		return self.alloc(np.prod(shape)*np.dtype(dtype).itemsize).view(dtype).reshape(shape)
	def full(self, shape, val, dtype=np.float32):
		arr = self.empty(shape, dtype=dtype)
		arr[:] = val
		return arr
	def zeros(self, shape, dtype=np.float32):
		return self.full(shape, 0, dtype=dtype)

class ArrayPoolGpu(Mempool):
	def array(self, arr):
		# Make sure the array is contiguous, which our memcpy needs
		import cupy
		ap   = cupy if isinstance(arr, cupy.ndarray) else np
		arr  = ap.ascontiguousarray(arr)
		oarr = self.empty(arr.shape, dtype=arr.dtype)
		cuda_memcpy(arr, oarr)
		return oarr
	def empty(self, shape, dtype=np.float32):
		return self.alloc(np.prod(shape)*np.dtype(dtype).itemsize).view(dtype).reshape(shape)
	def full(self, shape, val, dtype=np.float32):
		arr = self.empty(shape, dtype=dtype)
		arr[:] = val
		return arr
	def zeros(self, shape, dtype=np.float32):
		return self.full(shape, 0, dtype=dtype)

class ArrayMultipool:
	def __init__(self, factory):
		self.factory = factory
		self.pools   = {}
	def want(self, *names):
		for name in names:
			if name not in self.pools:
				self.pools[name] = self.factory(name=name)
	def free(self):
		for name in self.pools:
			self.pools[name].free()
	def reset(self):
		for name in self.pools:
			self.pools[name].reset()
	def __getattr__(self, name):
		return self.pools[name]
	def __delattr__(self, name):
		del self.pools[name]
	def __repr__(self):
		msg = "ArrayMultipool("
		names = sorted(list(self.pools.keys()))
		for name in names:
			msg += "\n  %s" % repr(self.pools[name])
		if len(names) > 0:
			msg += "\n"
		msg += ")"
		return msg

def round_up(a,b): return (a+b-1)//b*b

def cuda_memcpy(afrom,ato):
	import cupy
	assert afrom.flags["C_CONTIGUOUS"] and ato.flags["C_CONTIGUOUS"]
	cupy.cuda.runtime.memcpy(getptr(ato), getptr(afrom), ato.nbytes, cupy.cuda.runtime.memcpyDefault)

def getptr(arr):
	try: return arr.data.ptr
	except AttributeError: return arr.ctypes.data
