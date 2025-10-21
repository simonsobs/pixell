import numpy as np
import operator

class WatchArray(np.ndarray):
	"""Subclass that behaves like a numpy array except that
	it calls a user-provided callback when modifying
	operations occur. This subcless is non-contagious:
	The result of non-inplace operations will be a standard
	numpy array. Can be used as a building block in e.g.
	cache invalidation."""
	def __new__(cls, arr, callback=None):
		obj = np.asarray(arr).view(cls)
		obj.callback = callback
		return obj
	def __array_finalize__(self, obj):
		if obj is None: return
		self.callback = getattr(obj, "callback", None)
	def copy(self):
		return np.array(self)

# Add operators. In-place operators should call callback.
# Other operators should result in a plain numpy array,
# so we don't end up calling the callback much later
# when working on the result of some computation.

for opname in dir(operator):
	if not (opname.startswith("__") and opname.endswith("__") and hasattr(np.ndarray, opname)
			and opname != "__name__"):
		continue
	def make_op_callback(opname):
		def op(self, *args, **kwargs):
			val = getattr(np.ndarray, opname)(self, *args, **kwargs)
			self.callback()
			return val
		return op
	def make_op_decay(opname):
		def op(self, *args, **kwargs):
			val = getattr(np.ndarray, opname)(self, *args, **kwargs)
			if isinstance(val, self.__class__): val = np.asarray(val)
			return val
		return op
	if opname.startswith("__i") or opname == "__setitem__":
		# In-place operation. Should call our callback
		setattr(WatchArray, opname, make_op_callback(opname))
	else:
		# Other operation. Should decay to standard numpy array
		setattr(WatchArray, opname, make_op_decay(opname))
