"""This is a convenience wrapper of pyfftw."""
from __future__ import print_function, division
import numpy as np, multiprocessing, os
from . import utils
engines = {}

# Define our engines. First a baseline numpy-based engine
class numpy_FFTW:
	"""Minimal wrapper of numpy in order to be able to provide it as an engine.
	Not a full-blown interface."""
	def __init__(self, a, b, axes=(-1), direction='FFTW_FORWARD', *args, **kwargs):
		self.a, self.b = a, b
		self.axes = axes
		self.direction = direction
	def __call__(self, normalise_idft=False):
		if self.direction == 'FFTW_FORWARD':
			if self.a.shape == self.b.shape:
				# Complex to complex
				self.b[:] = np.fft.fftn(self.a, axes=self.axes)
			else:
				# Real to complex
				self.b[:] = np.fft.rfftn(self.a, axes=self.axes)
		else:
			if self.a.shape == self.b.shape:
				# Complex to complex
				self.b[:] = np.fft.ifftn(self.a, axes=self.axes)
			else:
				self.b[:] = np.fft.irfftn(self.a, s=[self.b.shape[i] for i in self.axes], axes=self.axes)
			# Numpy already normalizes, so undo this if necessary
			if not normalise_idft:
				self.b *= np.product([self.b.shape[i] for i in self.axes])

def numpy_n_byte_align_empty(shape, alignment, dtype):
	"""This dummy function just skips the alignment, since numpy
	doesn't provide an easy way to get it."""
	return np.empty(shape, dtype)

class NumpyEngine: pass
numpy_engine = NumpyEngine()
numpy_engine.FFTW = numpy_FFTW
numpy_engine.n_byte_align_empty = numpy_n_byte_align_empty
engines["numpy"] = numpy_engine
engine = "numpy"

# Then optional, faster engines
try:
	import pyfftw
	engines["fftw"] = pyfftw
	engine = "fftw"
except ImportError: pass
try:
	import pyfftw_intel as intel
	engines["intel"] = intel
	engine = "intel"
except ImportError: pass
if len(engines) == 0:
	# This should not happen due to the numpy fallback
	raise ImportError("Could not find any fftw implementations!")
try:
	nthread_fft = int(os.environ['OMP_NUM_THREADS'])
except (KeyError, ValueError):
	nthread_fft=multiprocessing.cpu_count()
nthread_ifft=nthread_fft
default_flags=['FFTW_ESTIMATE']
alignment = 32

def set_engine(eng):
	global engine
	engine = eng

def fft(tod, ft=None, nthread=0, axes=[-1], flags=None):
	"""Compute discrete fourier transform of tod, and store it in ft. What
	transform to do (real or complex, number of dimension etc.) is determined
	from the size and type of tod and ft. The optional nthread argument specifies
	the number of theads to use in the fft. The default (0) uses the value specified
	by the OMP_NUM_THREAD environment varible if that is specified, or the total
	number of cores on the computer otherwise. If ft is left out, a complex
	transform is assumed."""
	tod = asfcarray(tod)
	if tod.size == 0: return
	nt = nthread or nthread_fft
	if flags is None: flags = default_flags
	if ft is None:
		otype = np.result_type(tod.dtype,0j)
		ft  = empty(tod.shape, otype)
		tod = tod.astype(otype, copy=False)
	plan = engines[engine].FFTW(tod, ft, flags=flags, threads=nt, axes=axes)
	plan()
	return ft

def ifft(ft, tod=None, nthread=0, normalize=False, axes=[-1],flags=None):
	"""Compute inverse discrete fourier transform of ft, and store it in tod. What
	transform to do (real or complex, number of dimension etc.) is determined
	from the size and type of tod and ft. The optional nthread argument specifies
	the number of theads to use in the fft. The default (0) uses the value specified
	by the OMP_NUM_THREAD environment varible if that is specified, or the total
	number of cores on the computer otherwise. By default this is not nrmalized,
	meaning that fft followed by ifft will multiply the data by the length of the
	transform. By specifying the normalize argument, you can turn normalization
	on, though the normalization step will not use paralellization."""
	ft = asfcarray(ft)
	if ft.size == 0: return
	nt = nthread or nthread_ifft
	if flags is None: flags = default_flags
	if tod is None: tod = empty(ft.shape, ft.dtype)
	plan = engines[engine].FFTW(ft, tod, flags=flags, direction='FFTW_BACKWARD', threads=nt, axes=axes)
	# I get a small, cumulative loss in amplitude when using
	# pyfftw's normalize function.. So normalize manually instead
	#plan(normalise_idft=normalize)
	plan(normalise_idft=False)
	if normalize: tod /= np.product([tod.shape[i] for i in axes])
	return tod

def rfft(tod, ft=None, nthread=0, axes=[-1], flags=None):
	"""Equivalent to fft, except that if ft is not passed, it is allocated with
	appropriate shape and data type for a real-to-complex transform."""
	tod = asfcarray(tod)
	if ft is None:
		oshape = list(tod.shape)
		oshape[axes[-1]] = oshape[axes[-1]]//2+1
		dtype = np.result_type(tod.dtype,0j)
		ft = empty(oshape, dtype)
	return fft(tod, ft, nthread, axes, flags=flags)

def irfft(ft, tod=None, n=None, nthread=0, normalize=False, axes=[-1], flags=None):
	"""Equivalent to ifft, except that if tod is not passed, it is allocated with
	appropriate shape and data type for a complex-to-real transform. If n
	is specified, that is used as the length of the last transform axis
	of the output array. Otherwise, the length of this axis is computed
	assuming an even original array."""
	ft = asfcarray(ft)
	if tod is None:
		oshape = list(ft.shape)
		oshape[axes[-1]] = n or (oshape[axes[-1]]-1)*2
		dtype = np.zeros([],ft.dtype).real.dtype
		tod = empty(oshape, dtype)
	return ifft(ft, tod, nthread, normalize, axes, flags=flags)

def redft00(a, b=None, nthread=0, normalize=False, flags=None):
	"""pyFFTW does not support the DCT yet, so this is a work-around.
	It's not very fast, sadly - about 5 times slower than an rfft.
	Transforms along the last axis."""
	a = asfcarray(a)
	if b is None: b = empty(a.shape, a.dtype)
	n = a.shape[-1]
	tshape = a.shape[:-1] + (2*(n-1),)
	itmp = empty(tshape, a.dtype)
	itmp[...,:n] = a[...,:n]
	itmp[...,n:] = a[...,-2:0:-1]
	otmp = rfft(itmp, axes=[-1], nthread=nthread, flags=flags)
	del itmp
	b[...] = otmp[...,:n].real
	if normalize: b /= 2*(n-1)
	return b

def chebt(a, b=None, nthread=0, flags=None):
	"""The chebyshev transform of a, along its last dimension."""
	b = redft00(a, b, nthread, normalize=True, flags=flags)
	b[1:-1] *= 2
	return b

def ichebt(a, b=None, nthread=0):
	"""The inverse chebyshev transform of a, along its last dimension."""
	a = asfcarray(a).copy()
	a[1:-1] *= 0.5
	return redft00(a, b, nthread)

def fft_len(n, direction="below", factors=None):
	if factors is None: factors = [2,3,5,7,11,13]
	return utils.nearest_product(n, factors, direction)

def asfcarray(a):
	a = np.asarray(a)
	return np.asarray(a, np.result_type(a,0.0))

def empty(shape, dtype):
	return engines[engine].n_byte_align_empty(shape, alignment, dtype)

def fftfreq(n, d=1.0): return np.fft.fftfreq(n, d=d)
def rfftfreq(n, d=1.0): return np.arange(n//2+1)/(1.0*n*d)

def shift(a, shift, axes=None, nofft=False, deriv=None):
	"""Shift the array a by a (possibly fractional) number of samples "shift"
	to the right, along the specified axis, which defaults to the last one.
	shift can also be an array, in which case multiple axes are shifted together."""
	a      = np.asanyarray(a)
	ca     = a+0j
	shift  = np.atleast_1d(shift)
	if axes is None: axes = range(-len(shift),0)
	fa = fft(ca, axes=axes) if not nofft else ca
	for i, ax in enumerate(axes):
		ax   %= ca.ndim
		freqs = fftfreq(ca.shape[ax])
		phase = np.exp(-2j*np.pi*freqs*shift[i])
		if deriv == i:
			phase *= -2j*np.pi*freqs
		fa   *= phase[(None,)*ax + (slice(None),) + (None,)*(a.ndim-ax-1)]
	if not nofft: ifft(fa, ca, axes=axes, normalize=True)
	else:         ca = fa
	return ca if np.iscomplexobj(a) else ca.real
