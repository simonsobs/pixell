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
