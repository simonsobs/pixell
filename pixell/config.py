"""This module implements a simple system for passing configuration information
to functions etc. deep in various modules. Each parameter has a default given
where where it is used in the code, which is overridable in the configuration
file, which is overridable in the command line arguments, which is
(potentially) overridable in the function arguments.

Usage: Every parameter must be declared using config.default in global scope
before it can be used, and before it may appear in a config file. A natural
place to do this is at the top of the module where it is used, or just above
the function where it is used. For example:

 config.default("pixel_size", 0.5, "The map pixel size in arcminutes")

Since all these are in global scope, they will all have been executed by
the time the program starts. At this point they can be queried using
config.get, but in order to get the benefit of an actual configuration file
(which is the main point of this module), one needs to either call

 config.init(name=...) or config.init(fname=...)

which reads configuration from the specified filename or a location
inferred from the given name, see the init docstring for details.
By default, it is not an error for this file not to exist.

Alternatively, you can use config.ArgumentParser(name=..., fname=....),
which works just like argparse.ArgumentParser, except that it takes a configuration
file as an argument, and automatically calls init as well as registering
all configuration parameters as long-form arguments. This makes it easy
to override configuration parameters on the command line.

 parser = config.ArgumentParser(name)
 parser.add_argument("myarg")
 args = parser.parse_args()

When using this form, make sure none of your arguments clash with configuration
parameter names. The configuration parameters found on the command line will
be stripped before being returned to your program, so args will only contain
"myarg" in the case above.

With the configuration set up, you can get values via config.get:

 foo = config.get("foo") or foo = config.get("foo",foo)

The last form is useful for providing configurable default values for
function arguments:

 config.default("holler_volume", 9000, "Volume at which to holler")
 def holler(string, volume=None):
   volume = config.get("holler_volume", volume)

When volume is not specified (when it is None), it will get its value
from the configuration, but otherwise the specified value will be used.

One can also manually set configuration parameters using config.set,
though this usually shouldn't be necessary.

Finally, you can manually save and load parameters using

 config.save(filename)
 config.load(filename)

config.save is particularly useful for recording the exact parmaeters
your program was run with together with its output, so that the output
can be easily interpreted later. The optional 3rd argument of
config.default gets output as comments in the config file saved by
config.save."""

import argparse, os, textwrap, ast
from collections import OrderedDict

# Preserve order in order to give reasonable grouping of variable in the
# configuration file
parameters = OrderedDict()

# priorities
#  0: defaults and parameter file
#  1: command line arguments and similar
#  2: not used, but would override function arguments

class ArgumentParser(argparse.ArgumentParser):
	def __init__(self, name=None, fname=None, must_exist=False, **kwargs):
		"""A replacement for argparse.ArgumentParser that transparently handles
		configuration parameters, both reading them from file and registering and
		extracting them from command line arguments. It's usage is exactly the same
		as the normal ArgumentParser, except it takes a filename as an optional
		first argument, which specifies which configuration file to use. Se
		init() for details."""
		argparse.ArgumentParser.__init__(self, **kwargs)
		init(name=name, fname=fname, must_exist=must_exist)
		for name in parameters:
			typ = type(parameters[name]["value"])
			self.add_argument("--"+name, type=str if typ is bool else typ)
	def parse_args(self, argv=None):
		args = argparse.ArgumentParser.parse_args(self, argv)
		for name in parameters:
			if name in args and getattr(args,name) != None:
				typ = type(parameters[name]["value"])
				val = getattr(args, name)
				set(name, val=="True" if typ is bool else val)
				delattr(args, name)
		return args

def init(name=None, fname=None, must_exist=False):
	"""Initialize the module.

	If fname is specified, then settings are loaded from this file.

	Alternatively, if name is specified then the config file location
	is inferred from os.environ[name.upper()+"RC"] if present, defaulting
	to os.environ["HOME"]+"."+name+"rc" otherwise. For example, if name="foo",
	then it would look for $FOORC and default to $HOME/.foorc.

	If neither are specified, then no settings are loaded, leaving them
	at the default values set with config.default in the code."""
	if fname is None:
		if name is None:
			return
		else:
			envname = name.upper()+"RC"
			if envname in os.environ:
				fname = os.environ[envname]
			else:
				fname = os.path.expandvars("$HOME/.%src" % name)
	try:
		load(fname)
	except FileNotFoundError:
		if must_exist: raise

def to_str():
	"""Return a string representation of the configuration in a format that can
	be loaded by from_str."""
	res = ""
	for name in parameters:
		res += "\n".join(["# " + line for line in textwrap.wrap(parameters[name]["desc"])]) + "\n"
		res += "%s = %s\n" % (name, repr(parameters[name]["value"]))
		res += "\n"
	return res

def from_str(string):
	"""Update the configuration based on the specified string. The format is a simple
	key = value format, with valid values being integers, floats, booleans and quoted
	strings. Empty lines are ignored, and comments start with #. Comments come before the
	parameter they describe. The comments are mostly preserved when the file is auto-updated."""
	comment = []
	for line in string.split("\n"):
		if len(line) == 0 or line[0] == "#":
			if len(line) > 0: comment.append(line[1:].strip())
			continue
		toks = line.split("=")
		if len(toks) != 2:
			raise ValueError("Invalid format in config: %s" % line)
		name  = toks[0].strip()
		value = toks[1].strip()
		def deduce_ptype(name, value):
			try:
				return type(parameters[name]["value"])
			except KeyError:
				return type(ast.literal_eval(value))
		ptype = deduce_ptype(name, value)
		if ptype in [int,float]:
			value = ptype(value)
		elif ptype is bool:
			value = value == "True"
		elif ptype is str:
			if len(value) < 2 or value[0] != value[-1] or value[0] != "'" and value[0] != '"':
				raise ValueError("Invalid string in config: %s" % line)
			value = value[1:-1]
		else:
			raise ValueError("Unsupported config type: %s", repr(ptype))
		set(name, value, " ".join(comment), priority=0)
		comment = []

def save(config_file):
	"""Save our configuration parameters to the specified file. We support
	strings and numbers."""
	with open(config_file,"w") as f:
		f.write(to_str())

def load(config_file):
	"""Load configuration parameters. We support strings and numbers."""
	with open(config_file,"r") as f:
		from_str(f.read())

def set(name, value, desc=None, priority=1):
	if name in parameters and parameters[name]["priority"] > priority: return
	if name in parameters and desc is None: desc = parameters[name]["desc"]
	parameters[name] = {"value": value, "priority": priority, "desc": desc}

def default(name, value, desc=None):
	"""Declare a new configuration parameter, specifying its name, default value and
	description. Only variables that have been declared this way will be recognized in
	configuration files."""
	set(name, value, desc, priority=0)

def get(name, override=None):
	"""Get the value of the named parameter. If override is specified and is non-Null, then
	that will be returned instead. This lets the configured value work as a default value
	for a variable in the code."""
	return parameters[name]["value"] if override is None or parameters[name]["priority"] > 1 else override

class override:
	"""Use in with block to suppress warnings inside that block."""
	def __init__(self, name, value):
		self.name  = name
		self.value = value
	def __enter__(self):
		self.old = parameters[self.name]
		set(self.name, self.value)
		return self
	def __exit__(self, type, value, traceback):
		parameters[self.name] = self.old
