import numpy as np, time, contextlib
from . import bunch

class Device:
	def __init__(self):
		self.pools = None # Memory pools
		self.np    = None # numpy or equivalent
		self.lib   = bunch.Bunch() # place to store library functions
	def get(self, arr): raise NotImplementedError # copy device array to cpu
	def ptr(self, arr): return getptr(arr)
	def synchronize(self): raise NotImplementedError
	def garbage_collect(self): raise NotImplementedError
	def memuse(self, type="total"): raise NotImplementedError
	def copy(self, afrom, ato): raise NotImplementedError
	def time(self):
		self.synchronize()
		return time.time()

class DeviceCpu(Device):
	def __init__(self, align=None, alloc_factory=None, verbose=False):
		super().__init__()
		if align is None: align = 16
		if alloc_factory is None:
			def alloc_factory(name):
				return ArrayPoolCpu(AllocAligned(AllocCpu(), align=align), name=name, verbose=verbose)
		self.pools = ArrayMultipool(alloc_factory)
		self.np    = np
	def get(self, arr): return arr.copy()
	def synchronize(self): pass
	def garbage_collect(self):
		import gc
		gc.collect()
	def memuse(self, type="total"):
		if type == "total":
			from . import memory
			return memory.current()
		elif type == "pools":
			return self.pools.capacity()
		elif type == "np":
			return 0
		else: raise ValueError("Unknown memuse type: '%s'" % str(type))
	def copy(self, afrom, ato):
		"""Copy (cpu/dev) → (cpu/dev)"""
		ato[:] = afrom

class DeviceGpu(Device):
	def __init__(self, align=None, alloc_factory=None, verbose=False):
		super().__init__()
		if align is None: align = 512
		import cupy
		if alloc_factory is None:
			def alloc_factory(name):
				return ArrayPoolGpu(AllocAligned(AllocGpu(), align=align), name=name, verbose=verbose)
		self.pools = ArrayMultipool(alloc_factory)
		self.np    = cupy
		self.heap  = cupy.get_default_memory_pool()
		self.nvhandle = None
	def get(self, arr): return arr.get()
	def synchronize(self):
		import cupy
		cupy.cuda.runtime.deviceSynchronize()
	def garbage_collect(self):
		self.heap.free_all_blocks()
	def memuse(self, type="total"):
		if type == "total":
			import nvidia_smi
			if self.nvhandle is None:
				nvidia_smi.nvmlInit()
				self.nvhandle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
			info   = nvidia_smi.nvmlDeviceGetMemoryInfo(self.nvhandle)
			return info.used
		elif type == "pools":
			return self.pools.capacity()
		elif type == "np":
			return self.heap.used_bytes()
		else: raise ValueError("Unknown memuse type: '%s'" % str(type))
	def copy(self, afrom, ato):
		"""Copy (cpu/dev) → (cpu/dev)"""
		cuda_memcpy(afrom,ato)

# I'm having trouble using these allocators in place of
# the built-in ones. I think wrapping everything in arrays
# from the beginning is messing things up. The basic allocators
# should simply deal with MemoryPointers, and arrays should only
# enter at the ArrayPool level. To make this work, I need a version
# of MemoryPointer for the cpu too. The gpu version has:
#  .mem: PooledMemory, which has .size
#  .ptr: int, pointer to our part
#  etc.
# My code uses this to construct cupy arrays with
# cupy.ndarray(shape, dtype, memptr=...)
#
# Numpy is missing:
# * allocator support
# * suport for building array from pointer
# Maybe it's not worth it to try to make these look the same
# at this low level. If I get rid of AllocAligned, then
# all the code is CPU/GPU-specific

class AllocCpu:
	def alloc(self, n): return np.empty(n, dtype=np.uint8)

class AllocGpu:
	def __init__(self):
		import cupy
		self.allocator = cupy.cuda.get_allocator()
	def alloc(self, n):
		import cupy
		n      = int(n)
		memptr = self.allocator(n)
		return cupy.ndarray(n, np.uint8, memptr=memptr)

class AllocAligned:
	"""Wraps an allocator to make it aligned. Should work for both cpu and gpu.
	A bit inefficient if the underlying allocator is already aligned, which it
	probably already is."""
	def __init__(self, allocator, align=16):
		self.allocator = allocator
		self.align     = align
	def alloc(self, n):
		n   = int(n)
		buf = self.allocator.alloc(n+self.align-1)
		off = (-getptr(buf))%self.align
		return buf[off:off+n]

class Mempool:
	def __init__(self, aligned_alloc, name="[unnamed]", verbose=False):
		self.allocator = aligned_alloc
		self.name      = name
		self.verbose   = verbose
		self.free()
	def alloc(self, n):
		n       = int(n)
		effsize = round_up(n, self.allocator.align)
		# We have two modes:
		# 1. hand out memory from a single existing allocation.
		#    This is only active when len(arenas) == 1
		# 2. append new arenas
		if len(self.arenas) != 1 or self.arenas[0].size < self.used + n:
			if self.verbose:
				print("Growing pool %s to size %d by %d. %s" % (self.name, self.used + n, n, str([a.size for a in self.arenas])))
			# We're in mode 1
			self.arenas.append(self.allocator.alloc(n))
			buf = self.arenas[-1][0:n]
			self.used += effsize
		else:
			# We're in mode 2. Hand out some more memory
			buf        = self.arenas[-1][self.used:self.used+n]
			self.used += effsize
		return buf
	@property
	def capacity(self): return self.arenas[0].size if len(self.arenas) == 1 else self.used
	def free(self):
		self.arenas = []
		self.used   = 0
	def reset(self):
		"""Invalidate the memory we point to, potentially cleaning it up to a single
		fixed area we can allocate from. New allocations will reuse this memory
		as long as they don't exceed its capacity. If the capacity is exceeded, then
		it will start requesting new memory again."""
		if len(self.arenas) != 1:
			# Go to from mode 1 to mode 2
			self.arenas = [self.allocator.alloc(self.capacity)]
		self.used = 0
		return self
	def reserve(self, n):
		"""Reserve space for at least n bytes without reallocation"""
		self.reset()  # Reset position
		self.alloc(n) # Perform an allocation
		self.reset()  # Mark it as unused
		return self
	def __repr__(self):
		arenas = "["+",".join(["%.3fG" % (arena.size/1024**3) for arena in self.arenas])+"]"
		return "%s(name='%s', capacity=%.3fG, used=%.3fG, align=%d, arenas=%s)" % (self.__class__.__name__, self.name, self.capacity/1024**3, self.used/1024**3, self.allocator.align, arenas)

class ArrayPoolCpu(Mempool):
	def array(self, arr, reset=True, verbose=False):
		self.verbose = verbose
		arr  = np.asarray(arr)
		oarr = self.empty(arr.shape, dtype=arr.dtype, reset=True)
		oarr[:] = arr
		return oarr
	def empty(self, shape, dtype=np.float32, reset=True):
		if reset: self.reset()
		return self.alloc(np.prod(shape)*np.dtype(dtype).itemsize).view(dtype).reshape(shape)
	def full(self, shape, val, dtype=np.float32, reset=True):
		arr = self.empty(shape, dtype=dtype, reset=reset)
		arr[:] = val
		return arr
	def zeros(self, shape, dtype=np.float32, reset=True):
		return self.full(shape, 0, dtype=dtype, reset=reset)
	def ones(self, shape, dtype=np.float32, reset=True):
		return self.full(shape, 1, dtype=dtype, reset=reset)
	def copy(self, other, reset=True):
		arr = self.empty(other.shape, other.dtype, reset=reset)
		arr[:] = other
		return arr
	# No allocator support in numpy, so undefined what this
	# should return. Just return a numpy array of bytes for now
	def alloc_raw(self, n): return self.alloc(n)
	@contextlib.contextmanager
	def as_allocator(self, reset=True):
		if reset: self.reset()
		# No-op
		try: yield
		finally: pass

class ArrayPoolGpu(Mempool):
	def array(self, arr, reset=True, verbose=False):
		# Make sure the array is contiguous, which our memcpy needs
		import cupy
		self.verbose = verbose
		ap   = cupy if isinstance(arr, cupy.ndarray) else np
		arr  = ap.ascontiguousarray(arr)
		oarr = self.empty(arr.shape, dtype=arr.dtype, reset=True)
		cuda_memcpy(arr, oarr)
		return oarr
	def empty(self, shape, dtype=np.float32, reset=True):
		if reset: self.reset()
		return self.alloc(np.prod(shape)*np.dtype(dtype).itemsize).view(dtype).reshape(shape)
	def full(self, shape, val, dtype=np.float32, reset=True):
		arr = self.empty(shape, dtype=dtype, reset=reset)
		arr[:] = val
		return arr
	def zeros(self, shape, dtype=np.float32, reset=True):
		return self.full(shape, 0, dtype=dtype, reset=reset)
	def ones(self, shape, dtype=np.float32, reset=True):
		return self.full(shape, 1, dtype=dtype, reset=reset)
	def copy(self, other, reset=True):
		arr = self.empty(other.shape, other.dtype, reset=reset)
		arr[:] = other
		return arr
	def alloc_raw(self, n): return self.alloc(n).data
	@contextlib.contextmanager
	def as_allocator(self, reset=True):
		import cupy
		if reset: self.reset()
		old_allocator = cupy.cuda.get_allocator()
		try:
			# This causes a crash. I think it happens because
			# reset() can end up freeing the memory we handed
			# out here. In the old design, things were done a bit
			# differently. The memory needed was pre-allocated
			# and parseled out. However, if a later to was bigger than
			# an earlier, then memory would still be reallocated, so I
			# don't see why that worked.
			# TODO: Make a small test case that demonstrates the problem
			# without all this abstraction.
			cupy.cuda.set_allocator(self.alloc_raw)
			yield
		finally:
			cupy.cuda.set_allocator(old_allocator)

class ArrayMultipool:
	def __init__(self, factory):
		self.factory = factory
		self.pools   = {}
	def want(self, *names):
		pools = []
		for name in names:
			if name not in self.pools:
				self.pools[name] = self.factory(name=name)
			pools.append(self.pools[name])
		return pools
	def used(self): return sum([pool.used for name, pool in self.pools.items()])
	def capacity(self): return sum([pool.capacity for name, pool in self.pools.items()])
	def free(self):
		for name in self.pools:
			self.pools[name].free()
	def reset(self):
		for name in self.pools:
			self.pools[name].reset()
	def swap(self, name1, name2):
		self.pools[name1], self.pools[name2] = self.pools[name2], self.pools[name1]
	def __getitem__(self, name):
		"""Returns the memory pool with the given name, creating it if it doesn't exist"""
		if name not in self.pools:
			self.pools[name] = self.factory(name=name)
		return self.pools[name]
	def __getattr__(self, name):
		"""Returns the memory pool with the given name, if it exists"""
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
	"""Returns a pointer to arr's data whether it's a numpy or cupy array"""
	try: return arr.data.ptr
	except AttributeError: return arr.ctypes.data

def anypy(arr):
	"""Returns the numpy or cupy modules depending on whether arr is
	a numpy or cupy array, or can be converted to these. Raises a
	ValueError if neither is the case. Still works if cupy can't be
	imported, as long as arr isn't a cupy array"""
	try: import cupy
	except ModuleNotFoundError: return np
	try:
		np.asanyarray(arr)
		return np
	except TypeError: pass
	try:
		cupy.asanyarray(arr)
		return cupy
	except TypeError: pass
	raise ValueError("Neither numpy or cupy array, and can't be converted to them either")
