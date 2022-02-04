"""My own version of bunch, since the standard one lacks tab completion
and has trouble printing sometimes."""
class Bunch:
	def __init__(self, *args, **kwargs):
		self._dict = {}
		for args in args:
			self._dict.update(args)
		self._dict.update(kwargs)
	def __getattr__(self, name):
		# Members with names like __getattr__ could override class behavior, which
		# is confusing. This should also resolve an issue with copy.copy, where getattr
		# is called before __init__
		if name.startswith("__"): raise AttributeError(name)
		if name in self.__dict__["_dict"].keys():
			return self.__dict__["_dict"][name]
		raise AttributeError(name)
	def __setattr__(self, name, value):
		if "_dict" in self.__dict__:
			self._dict[name] = value
		else:
			self.__dict__[name] = value
	def __getitem__(self, name):
		return self._dict[name]
	def __setitem__(self, name, value):
		self._dict[name] = value
	def __delattr__(self, name):
		if name in self._dict:
			del self._dict[name]
		else:
			del self.__dict__[name]
	def __delitem__(self, name):
		self.__delattr__(name)
	def __contains__(self, name):
		return name in self._dict
	def __dir__(self):
		return sorted(list(self.__dict__.keys()) + list(self._dict.keys()))
	def __iter__(self):
		for key in self._dict:
			yield key
	def __len__(self):
		return len(self._dict)
	def keys(self): return self._dict.keys()
	def items(self): return self._dict.items()
	def iteritems(self): return self._dict.iteritems()
	def copy(self): return Bunch(self._dict.copy())
	def update(self, val): return self._dict.update(val)
	def __repr__(self):
		keys = sorted(self._dict.keys())
		return "Bunch(" + ", ".join([
			"%s=%s" % (str(key),repr(self._dict[key])) for key in keys
			])+")"

# Some simple I/O routines. These can't handle everything that could
# be in a bunch, but they cover all my most common use cases.

def read(fname, fmt="auto", group=None):
	if fmt == "auto":
		if is_hdf_path(fname): fmt = "hdf"
		else: raise ValueError("Could not infer format for '%s'" % fname)
	if fmt == "hdf": return read_hdf(fname, group=group)
	else: raise ValueError("Unrecognized format '%s'" % fmt)

def write(fname, bunch, fmt="auto", group=None):
	if fmt == "auto":
		if is_hdf_path(fname): fmt = "hdf"
		else: raise ValueError("Could not infer format for '%s'" % fname)
	if fmt == "hdf": write_hdf(fname, bunch, group=group)
	else: raise ValueError("Unrecognized format '%s'" % fmt)

def write_hdf(fname, bunch, group=None):
	import h5py
	fname, group = split_hdf_path(fname, group)
	with h5py.File(fname, "w") as hfile:
		if group: hfile = hfile.create_group(group)
		for key in bunch:
			hfile[key] = bunch[key]

def read_hdf(fname, group=None):
	import h5py
	bunch = Bunch()
	fname, group = split_hdf_path(fname, group)
	with h5py.File(fname, "r") as hfile:
		if group: hfile = hfile[group]
		for key in hfile:
			bunch[key] = hfile[key][()]
	return bunch

def is_hdf_path(fname):
	"""Returns true if the fname would be recognized by split_hdf_path"""
	for suf in [".hdf", ".h5"]:
		name, _, group = fname.rpartition(suf)
		if name and (not group or group[0] == "/"): return True
	return False

def split_hdf_path(fname, subgroup=None):
	"""Split an hdf path of the form path.hdf/group, where the group part is
	optional, into the path and the group parts. If subgroup is specified, then
	it will be appended to the group informaiton. returns fname, group. The
	fname will be a string, and the group will be a string or None. Raises
	a ValueError if the fname is not recognized as a hdf file."""
	for suf in [".hdf", ".h5"]:
		name, _, group = fname.rpartition(suf)
		if not name: continue
		name += suf
		if not group: return name, subgroup
		elif group[0] == "/":
			group = group[1:]
			if subgroup: group += "/" + subgroup
			return name, group
	raise ValueError("Not an hdf path")

def concatenate(bunches):
	"""Go from a list of bunches to a bunch of lists."""
	import numpy as np
	keys = bunches[0].keys()
	res  = Bunch()
	for key in keys: res[key] = []
	for bunch in bunches:
		for key in keys:
			res[key].append(bunch[key])
	for key in keys:
		# Try to build arrays while keeping type
		if len(res[key]) > 0:
			first = res[key][0]
			tmp   = np.array(res[key])
			new   = np.empty_like(first, shape=tmp.shape)
			new[:] = tmp
			res[key] = new
			del first, tmp, new
	return res
