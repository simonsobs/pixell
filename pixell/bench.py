import time as _time
from contextlib import contextmanager
from . import bunch

"""bench: Simple timing of python code blocks.

Example usage:

	1. Manual printing

	from pixell import bench
	for i in range(nfile):
		with bench.mark("all"):
			with bench.mark("read"):
				a = np.loadtxt(afiles[i])
				b = np.loadtxt(bfiles[i])
			with bench.mark("sum"):
				a += b
			with bench.mark("write"):
				np.savetxt(ofiles[i], a)
		print("Processed case %d in %7.4f s. read %7.4f sum %7.4f write %7.4f" % (i, bench.t.all, bench.t.read, bench.t.sum, bench.t.write))
	print("Total %7.4f s. read %7.4f sum %7.4f write %7.4f" % (i, bench.t_tot.all, bench.t_tot.read, bench.t_tot.sum, bench.t_tot.write))

	2. Quick-and-dirty printing

	from pixell import bench
	for i in range(nfile):
		with bench.show("read"):
			a = np.loadtxt(afiles[i])
			b = np.loadtxt(bfiles[i])
		with bench.show("sum"):
			a += b
		with bench.show("write"):
			np.savetxt(ofiles[i], a)

	bench.show is equivalent to bench.mark, just with an extra print.
	This means that bench.show updates .ttot and .n just like bench.mark
	does.

The examples above collect statistics globally. You can create local
benchmark objects with bench.Bench(). Example:

	from pixell import bench
	mybench = bench.Bench()
	with mybench.mark("example"):
		do_something()

The overhead of bench.mark is around 3 µs.
"""

# Just wall times for now, but could be extended to measure
# cpu time or leaked memory
class Bench:
	def __init__(self):
		self.t_tot  = bunch.Bunch()
		self.t      = bunch.Bunch()
		self.n      = bunch.Bunch()
	@contextmanager
	def mark(self, name):
		if name not in self.n:
			self.t_tot[name] = 0
			self.n[name]    = 0
		t1 = _time.time()
		try:
			yield
		finally:
			t2 = _time.time()
			self.n[name] += 1
			self.t    [name] = t2-t1
			self.t_tot[name] += t2-t1
	@contextmanager
	def show(self, name):
		try:
			with self.mark(name):
				yield
		finally:
			print("%7.4f s (last) %7.4f s (mean) %4d (n) %s" % (self.t[name], self.t_tot[name]/self.n[name], self.n[name], name))

# Global interface
_default = Bench()
mark  = _default.mark
show  = _default.show
t_tot = _default.t_tot
t     = _default.t
n     = _default.n
