"""My own version of bunch, since the standard one lacks tab completion
and has trouble printing sometimes."""
import os
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

def read(fname, fmt="auto", group=None, gmode="dot"):
	if fmt == "auto": fmt="hdf"
	if fmt == "hdf": return read_hdf(fname, group=group, gmode=gmode)
	else: raise ValueError("Unrecognized format '%s'" % fmt)

def write(fname, bunch, fmt="auto", group=None, gmode="dot"):
	if fmt == "auto": fmt = "hdf"
	if fmt == "hdf": write_hdf(fname, bunch, group=group, gmode=gmode)
	else: raise ValueError("Unrecognized format '%s'" % fmt)

def read_hdf(fname, group=None, gmode="dot"):
	import h5py
	if isinstance(fname, h5py.Group):
		# Already open hdf file
		if group is not None: fname = fname[group]
		return read_hdf_recursive(fname)
	else:
		# File name. Optionally extract group
		if group is None:
			fname, group = split_hdf_path(fname, group, mode=gmode)
		with h5py.File(fname, "r") as hfile:
			if group: hfile = hfile[group]
			return read_hdf_recursive(hfile)

def read_hdf_recursive(hfile):
	import h5py
	if isinstance(hfile, h5py.Dataset):
		return decode(hfile[()])
	else:
		bunch = Bunch()
		for key in hfile:
			bunch[key] = read_hdf_recursive(hfile[key])
		return bunch

def write_hdf(fname, bunch, group=None, gmode="dot"):
	import h5py
	if group is None:
		fname, group = split_hdf_path(fname, group, mode=gmode)
	with h5py.File(fname, "w") as hfile:
		if group: hfile = hfile.create_group(group)
		write_hdf_recursive(hfile, bunch)

def write_hdf_recursive(hfile, bunch):
	for key in bunch:
		if isinstance(bunch[key],Bunch):
			hfile.create_group(key)
			write_hdf_recursive(hfile[key], bunch[key])
		else:
			hfile[key] = encode(bunch[key])

def encode(val):
	import numpy as np
	if isinstance(val, np.ndarray):
		try: return np.char.encode(val)
		except (TypeError,AttributeError): return val
	elif isinstance(val, str):
		return val.encode()
	elif val is None:
		return "__None__".encode()
	else:
		return val

def decode(val):
	import numpy as np
	if isinstance(val, np.ndarray):
		try: return np.char.decode(val)
		except (TypeError,AttributeError): return val
	elif isinstance(val, bytes):
		val = val.decode()
		if val == "__None__": return None
		else: return val
	else:
		return val

def is_hdf_path(fname):
	"""Returns true if the fname would be recognized by split_hdf_path"""
	return True

def split_hdf_path(fname, subgroup=None, mode="dot"):
	"""Split an hdf path of the form path.hdf/group, where the group part is
	optional, into the path and the group parts. If subgroup is specified, then
	it will be appended to the group informaiton. returns fname, group. The
	fname will be a string, and the group will be a string or None. Raises
	a ValueError if unsuccessful.
	
	mode controles how the split is done:
	* "none": Don't split. fname is returned unmodified
	* "dot": The last entry in the path given by filename
	    containing a "." will be taken to be the real
	    file name, the rest till be the hdf group path.
	    For example, with a/b/c.d/e/f, a/b/c.d would be returned
	    as the file name and e/f as the hdf group
	* "exists": As dot, but based on whether a file with that
	    name can be found on disk. Seemed like a good idea,
	    except it doesn't work when writing a new file.
	"""
	toks = fname.split("/")
	if mode == "dot":
		# Find last entry with a dot i in it
		for i, tok in reversed(list(enumerate(toks))):
			if "." in tok: break
		else: raise ValueError("Could not split hdf path using 'dot' method: no . found")
	elif mode == "exists":
		for i in reversed(list(range(len(toks)))):
			cand = "/".join(toks[:i+1])
			if os.path.isfile(cand): break
		else: raise ValueError("Could not split hdf path using 'exists' method: no file found")
	elif mode == "none":
		i = len(toks)
	else: raise ValueError("Unrecognized split mode '%s'" % (str(mode)))
	# Return the result
	fname = "/".join(toks[:i+1])
	gtoks = toks[i+1:]
	if subgroup: gtoks.append(subgroup)
	group = "/".join(gtoks) if len(gtoks)>0 else None
	return fname, group

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
