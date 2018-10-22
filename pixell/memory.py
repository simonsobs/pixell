# Get information about the process's memory usage. Ideally this would just
# be some simple calls to resource, but that library is almost useless. The
# python standard library is usually good, but here it's terrible.
# If we wanted to introduce an external dependency we could try psutil,
# though I would avoid it if possible.
# It looks like ctypes would work for this. It has some overhead, but this
# doesn't need to be that fast (and proc reading has overhead too)
import resource, os, ctypes

def current():  return fallback([(linux_current,  IOError), (mac_current,  AttributeError)])
def resident(): return fallback([(linux_resident, IOError), (mac_resident, AttributeError)])
def max():      return fallback([(linux_max,      IOError), (mac_max,      AttributeError)])

def fallback(things, default=lambda:0):
	for func, exceptions in things:
		try: return func()
		except exceptions: pass
	return default()

#### Linux stuff ####

def linux_current():
	with open("/proc/self/statm","r") as f:
		return int(f.readline().split()[0])*resource.getpagesize()

def linux_resident():
	with open("/proc/self/status","r") as f:
		for line in f:
			toks = line.split()
			if toks[0] == "VmRSS:":
				return int(toks[1])*1024

def linux_max():
	with open("/proc/self/status","r") as f:
		for line in f:
			toks = line.split()
			if toks[0] == "VmPeak:":
				return int(toks[1])*1024

##### MacOs stuff #####

def mac_current():  return get_mac_taskinfo().virtual_size
def mac_resident(): return get_mac_taskinfo().resident_size
def mac_max():      return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # bytes on mac

# Taskinfo stuff for memory lookups on macs
_libc = None
def get_mac_taskinfo():
	# Cache libc so we avoid a file system search every time this is called
	global _libc
	if _libc is None: _libc = ctypes.cdll.LoadLibrary(None)
	# Fail immediately if task_info doesn't exist
	_libc.task_info
	# Define data structurs
	from ctypes import c_int, c_uint, c_ulong
	class time_value_t(ctypes.Structure):
		_fields_ = [("seconds", c_int), ("microseconds", c_int)]
	class task_basic_info(ctypes.Structure):
		_pack_ = 4
		_fields_ = [("suspend_count", c_int), ("virtual_size", c_ulong), ("resident_size", c_ulong),
				("user_time", time_value_t), ("system_time", time_value_t), ("policy", c_int)]
	count = c_uint(ctypes.sizeof(task_basic_info)//ctypes.sizeof(c_uint))
	# Define function interfaces
	task_self = _libc.mach_task_self
	task_self.restype = c_uint
	task_self.argtypes = []
	me        = task_self()
	task_info = _libc.task_info
	task_info.restype = c_int
	task_info.argtypes = [c_uint, c_uint, ctypes.POINTER(task_basic_info), ctypes.POINTER(c_uint)]
	info    = task_basic_info()
	status  = _libc.task_info(me, 5, ctypes.byref(info), ctypes.byref(count))
	return info if status == 0 else None
