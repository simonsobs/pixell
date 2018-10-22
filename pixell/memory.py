# Get information about the process's memory usage. Ideally this would just
# be some simple calls to resource, but that library is almost useless. The
# python standard library is usually good, but here it's terrible.
# If we wanted to introduce an external dependency we could try psutil,
# though I would avoid it if possible.
# It looks like ctypes would work for this. It has some overhead, but this
# doesn't need to be that fast (and proc reading has overhead too)
import resource, os

def current():
	try:
		with open("/proc/self/statm","r") as f:
			return int(f.readline().split()[0])*resource.getpagesize()
	except IOError:
		# There doesn't seem to be any standard way to get this on a mac.
		return 0

def resident():
	try:
		with open("/proc/self/status","r") as f:
			for line in f:
				toks = line.split()
				if toks[0] == "VmRSS:":
					return int(toks[1])*1024
	except IOError:
		# No simple way to get this either
		return 0

def max():
	try:
		with open("/proc/self/status","r") as f:
			for line in f:
				toks = line.split()
				if toks[0] == "VmPeak:":
					return int(toks[1])*1024
	except IOError:
		# Assume we're on Mac here, in which case this is in bytes.
		# Not sure it's exactly the same thing, though - one is VM
		# and the other is resident.
		return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
