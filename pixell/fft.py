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
				self.b *= np.prod([self.b.shape[i] for i in self.axes])

class ducc_FFTW:
	"""Minimal wrapper of ducc in order to be able to provide it as an engine.
	Not a full-blown interface."""
	def __init__(self, a, b, axes=(-1,), direction='FFTW_FORWARD', threads=1, *args, **kwargs):
		self.a, self.b = np.asarray(a), np.asarray(b)
		self.axes = tuple(axes)
		self.direction = direction
		self.threads   = threads
	def do_dct(self, kind, *args, **kwargs):
		# Expect format of type FFTW_REDFT00
		name = {"REDFT":"DCT","RODFT":"DST"}[kind[5:10]]
		num  = {"00":1, "10": 2, "01": 3, "11":4}[kind[10:12]]
		if   name == "DCT": return ducc0.fft.dct(*args, type=num, **kwargs)
		elif name == "DST": return ducc0.fft.dst(*args, type=num, **kwargs)
	def __call__(self, normalise_idft=False):
		if self.direction == 'FFTW_FORWARD':
			if self.a.shape == self.b.shape:
				# Complex to complex
				ducc0.fft.c2c(self.a, axes=self.axes, out=self.b, nthreads=self.threads)
			else:
				# Real to complex
				ducc0.fft.r2c(self.a, axes=self.axes, out=self.b, nthreads=self.threads)
		elif self.direction == "FFTW_BACKWARD":
			if self.a.shape == self.b.shape:
				# Complex to complex
				ducc0.fft.c2c(a=self.a, axes=self.axes, out=self.b, forward=False, inorm=2 if normalise_idft else 0, nthreads=self.threads)
			else:
				ducc0.fft.c2r(a=self.a, axes=self.axes, out=self.b, forward=False, lastsize=self.b.shape[self.axes[-1]], inorm=2 if normalise_idft else 0, nthreads=self.threads)
		elif _check_ducc_r2r(self.direction):
			# dct and dst are passed with a list with one entry per dimension of the transform,
			# but ducc doesn't support heterogeneous transforms like this, so just use the first element
			self.do_dct(self.direction[0], self.a, axes=self.axes, out=self.b, nthreads=self.threads)

def _check_ducc_r2r(direction):
	if isinstance(direction, str): return False
	for d in direction:
		if d != direction[0]:
			raise ValueError("ducc only supports homogeneous r2r transforms")
	return direction[0].startswith("FFTW_REDFT") or direction[0].startswith("FFTW_RODFT")

def numpy_empty_aligned(shape, dtype, n=None):
	"""This dummy function just skips the alignment, since numpy
	doesn't provide an easy way to get it."""
	return np.empty(shape, dtype)

class NumpyEngine: pass
numpy_engine = NumpyEngine()
numpy_engine.FFTW = numpy_FFTW
numpy_engine.empty_aligned = numpy_empty_aligned
engines["numpy"] = numpy_engine
engine = "numpy"

# Then optional, faster engines
try:
	import pyfftw
	engines["fftw"] = pyfftw
	engine = "fftw"
except ImportError: pass
else:
	# Is FFTW actually using intel MKL as a backend? Check if 1D FT crashes for 3D input.
	try:
		engines['fftw'].FFTW(np.zeros((1,1,1)), np.zeros((1,1,1), dtype=np.complex128),
			flags=['FFTW_ESTIMATE'], threads=1, axes=[-1])
	except RuntimeError as e:
		engines['intel'] = engines.pop('fftw')
		engine = 'intel'
try:
	import pyfftw_intel as intel
	engines["intel"] = intel
	engine = "intel"
except ImportError: pass
# ducc is slower than intel, but can be faster than pyfftw
try:
	import ducc0
	class DuccEngine: pass
	ducc_engine = DuccEngine()
	ducc_engine.FFTW = ducc_FFTW
	ducc_engine.empty_aligned = numpy_empty_aligned
	engines["ducc"] = ducc_engine
	if engine != "intel": engine = "ducc"
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

def get_engine(eng):
	return engine if eng == "auto" else eng

def fft(tod, ft=None, nthread=0, axes=[-1], flags=None, _direction="FFTW_FORWARD", engine="auto"):
	"""Compute discrete fourier transform of tod, and store it in ft. What
	transform to do (real or complex, number of dimension etc.) is determined
	from the size and type of tod and ft. If ft is left out, a complex transform
	is assumed. The optional nthread argument specifies the number of theads to
	use in the fft. The default (0) uses the value specified by the
	OMP_NUM_THREAD environment varible if that is specified, or the total number
	of cores on the computer otherwise."""
	tod  = asfcarray(tod)
	axes = utils.astuple(-1 if axes is None else axes)
	if tod.size == 0: return
	nt = nthread or nthread_fft
	if flags is None: flags = default_flags
	if ft is None:
		otype = np.result_type(tod.dtype,0j)
		ft  = empty(tod.shape, otype)
		tod = tod.astype(otype, copy=False)
	engine = get_engine(engine)
	if engine == 'intel':
		ft[:] = fft_flat(tod, ft, axes=axes, nthread=nt, flags=flags, _direction=_direction)
	else:
		plan = engines[engine].FFTW(tod, ft, flags=flags, threads=nt, axes=axes, direction=_direction)
		plan()
	return ft

def ifft(ft, tod=None, nthread=0, normalize=False, axes=[-1],flags=None, engine="auto"):
	"""Compute inverse discrete fourier transform of ft, and store it in tod. What
	transform to do (real or complex, number of dimension etc.) is determined
	from the size and type of tod and ft. The optional nthread argument specifies
	the number of theads to use in the fft. The default (0) uses the value specified
	by the OMP_NUM_THREAD environment varible if that is specified, or the total
	number of cores on the computer otherwise. By default this is not normalized,
	meaning that fft followed by ifft will multiply the data by the length of the
	transform. By specifying the normalize argument, you can turn normalization
	on, though the normalization step will not use paralellization."""
	ft   = asfcarray(ft)
	axes = utils.astuple(-1 if axes is None else axes)
	if ft.size == 0: return
	nt = nthread or nthread_ifft
	if flags is None: flags = default_flags
	if tod is None:	tod = empty(ft.shape, ft.dtype)
	engine = get_engine(engine)
	if engine == 'intel':
		tod[:] = ifft_flat(ft, tod, axes=axes, nthread=nt, flags=flags)
	else:
		plan = engines[engine].FFTW(ft, tod, flags=flags, direction='FFTW_BACKWARD',
			threads=nt, axes=axes)
		plan(normalise_idft=False)
	# I get a small, cumulative loss in amplitude when using
	# pyfftw's normalize function.. So normalize manually instead	
	if normalize: tod /= np.prod([tod.shape[i] for i in axes])
	return tod

def rfft(tod, ft=None, nthread=0, axes=[-1], flags=None, engine="auto"):
	"""Equivalent to fft, except that if ft is not passed, it is allocated with
	appropriate shape and data type for a real-to-complex transform."""
	tod  = asfcarray(tod)
	axes = utils.astuple(-1 if axes is None else axes)
	if ft is None:
		oshape = list(tod.shape)
		oshape[axes[-1]] = oshape[axes[-1]]//2+1
		dtype = np.result_type(tod.dtype,0j)
		ft = empty(oshape, dtype)
	return fft(tod, ft, nthread, axes, flags=flags, engine=engine)

def irfft(ft, tod=None, n=None, nthread=0, normalize=False, axes=[-1], flags=None, engine="auto"):
	"""Equivalent to ifft, except that if tod is not passed, it is allocated with
	appropriate shape and data type for a complex-to-real transform. If n
	is specified, that is used as the length of the last transform axis
	of the output array. Otherwise, the length of this axis is computed
	assuming an even original array."""
	ft   = asfcarray(ft)
	axes = utils.astuple(-1 if axes is None else axes)
	if tod is None:
		oshape = list(ft.shape)
		oshape[axes[-1]] = n or (oshape[axes[-1]]-1)*2
		dtype = np.zeros([],ft.dtype).real.dtype
		tod = empty(oshape, dtype)
	return ifft(ft, tod, nthread, normalize, axes, flags=flags, engine=engine)

def dct(tod, dt=None, nthread=0, normalize=False, axes=[-1], flags=None, type="DCT-I", engine="auto"):
	"""Compute discrete cosine transform of tod, and store it in dt. By
	default it will do a DCT-I trasnform, but this can be controlled with the type argument.
	Even the much less common discrete sine transforms are avialble by passing e.g. type="DST-I".
	Valid values are DCT-I, DCT-II, DCT-III, DCT-IV, DST-I, DST-II, DST-III and DST-IV,
	or the raw FFTW names the correspond to (e.g. FFTW_REDFT00). If dt is not passed, it
	will be allocated with the same shape and data type as tod.

	The optional nthread argument specifies the number of theads to use in the fft. The
	default (0) uses the value specified by the OMP_NUM_THREAD environment varible if that
	is specified, or the total number of cores on the computer otherwise.

	Note that DCTs and DSTs were only added to pyfftw in version 13.0. The function will
	fail with an Invalid scheme error for older versions.
	"""
	tod  = asfcarray(tod)
	type = _dct_names[type]
	axes = utils.astuple(-1 if axes is None else axes)
	if dt is None:
		dt = empty(tod.shape, tod.dtype)
	return fft(tod, dt, nthread=nthread, axes=axes, flags=flags, _direction=[type]*len(axes), engine=engine)

def idct(dt, tod=None, nthread=0, normalize=False, axes=[-1], flags=None, type="DCT-I", engine="auto"):
	"""Compute the inverse discrete cosine transform of dt, and store it in tod. By
	default it will do the inverse of a DCT-I trasnform, but this can be controlled with the type argument.
	Even the much less common discrete sine transforms are avialble by passing e.g. type="DST-I".
	Valid values are DCT-I, DCT-II, DCT-III, DCT-IV, DST-I, DST-II, DST-III and DST-IV,
	or the raw FFTW names the correspond to (e.g. FFTW_REDFT00). If tod is not passed, it
	will be allocated with the same shape and data type as tod.

	By the default an unnormalized transform is performed. Pass normalize=True to get an
	actual inverse transform. This divides by a factor of 2*N+d for each axis the transform
	is performed along, where N is the length of the axis and d is -1 for DCT-1, +1 for
	DST-I and 0 for all the others. Usually it's faster to compute this factor once and
	combine it with other scalar factors in your math than to let this function do it,
	which is why it's turned off by default.

	Note that this function already takes care of figuring out which transform is the
	appropriate inverse. E.g. the inverse of b = dct(a, type="DCT-II") is
	idct(b, type="DCT-II", normalize=True), not idct(b, type="DCT-III", normalize=True)
	even though DCT-III is the inverse of DCT-II.

	The optional nthread argument specifies the number of theads to use in the fft. The
	default (0) uses the value specified by the OMP_NUM_THREAD environment varible if that
	is specified, or the total number of cores on the computer otherwise.

	Note that DCTs and DSTs were only added to pyfftw in version 13.0. The function will
	fail with an Invalid scheme error for older versions."""
	dt   = asfcarray(dt)
	type = _dct_inverses[_dct_names[type]]
	off  = _dct_sizes[type]
	axes = utils.astuple(-1 if axes is None else axes)
	if tod is None:
		tod = empty(dt.shape, dt.dtype)
	fft(dt, tod, nthread=nthread, axes=axes, flags=flags, _direction=[type]*len(axes), engine=engine)
	if normalize: tod /= np.prod([2*(tod.shape[i]+off) for i in axes])
	return tod

_dct_names = {
		"DCT-I":   "FFTW_REDFT00",  "FFTW_REDFT00":"FFTW_REDFT00",
		"DCT-II":  "FFTW_REDFT10",  "FFTW_REDFT10":"FFTW_REDFT10",
		"DCT-III": "FFTW_REDFT01",  "FFTW_REDFT01":"FFTW_REDFT01",
		"DCT-IV":  "FFTW_REDFT11",  "FFTW_REDFT11":"FFTW_REDFT11",
		"DST-I":   "FFTW_RODFT00",  "FFTW_RODFT00":"FFTW_RODFT00",
		"DST-II":  "FFTW_RODFT10",  "FFTW_RODFT10":"FFTW_RODFT10",
		"DST-III": "FFTW_RODFT01",  "FFTW_RODFT01":"FFTW_RODFT01",
		"DST-IV":  "FFTW_RODFT11",  "FFTW_RODFT11":"FFTW_RODFT11",
}
_dct_inverses = {
		"FFTW_REDFT00":"FFTW_REDFT00", "FFTW_REDFT10":"FFTW_REDFT01",
		"FFTW_REDFT01":"FFTW_REDFT10", "FFTW_REDFT11":"FFTW_REDFT11",
		"FFTW_RODFT00":"FFTW_RODFT00", "FFTW_RODFT10":"FFTW_RODFT01",
		"FFTW_RODFT01":"FFTW_RODFT10", "FFTW_RODFT11":"FFTW_RODFT11",
}
_dct_sizes = {
		"FFTW_REDFT00":-1, "FFTW_REDFT10":0, "FFTW_REDFT01":0, "FFTW_REDFT11":0,
		"FFTW_RODFT00":+1, "FFTW_RODFT10":0, "FFTW_RODFT01":0, "FFTW_RODFT11":0,
}

def redft00(a, b=None, nthread=0, normalize=False, flags=None, engine="auto"):
	"""Old brute-force work-around for missing dcts in pyfftw. Can be
	removed when newer versions of pyfftw become common. It's not very
	fast, sadly - about 5 times slower than an rfft. Transforms along the last axis."""
	a = asfcarray(a)
	if b is None: b = empty(a.shape, a.dtype)
	n = a.shape[-1]
	tshape = a.shape[:-1] + (2*(n-1),)
	itmp = empty(tshape, a.dtype)
	itmp[...,:n] = a[...,:n]
	itmp[...,n:] = a[...,-2:0:-1]
	otmp = rfft(itmp, axes=[-1], nthread=nthread, flags=flags, engine=engine)
	del itmp
	b[...] = otmp[...,:n].real
	if normalize: b /= 2*(n-1)
	return b

def chebt(a, b=None, nthread=0, flags=None, engine="auto"):
	"""The chebyshev transform of a, along its last dimension."""
	b = redft00(a, b, nthread, normalize=True, flags=flags, engine=engine)
	b[1:-1] *= 2
	return b

def ichebt(a, b=None, nthread=0, engine="auto"):
	"""The inverse chebyshev transform of a, along its last dimension."""
	a = asfcarray(a).copy()
	a[1:-1] *= 0.5
	return redft00(a, b, nthread, engine=engine)

def fft_len(n, direction="below", factors=None):
	if factors is None: factors = [2,3,5,7,11,13]
	return utils.nearest_product(n, factors, direction)

def asfcarray(a):
	a = np.asarray(a)
	return np.asarray(a, np.result_type(a,0.0))

def empty(shape, dtype):
	return engines[engine].empty_aligned(shape, dtype=dtype, n=alignment)

def fftfreq(n, d=1.0, dtype=np.float64): return np.fft.fftfreq(n, d=d).astype(dtype, copy=False)
def rfftfreq(n, d=1.0, dtype=np.float64): return np.arange(n//2+1, dtype=dtype)/(n*d)

def ind2freq (n, i, d=1.0): return np.where(i < n/2, i, -n+i)/(d*n)
def int2rfreq(n, i, d=1.0): return i/(n*d)
def freq2ind(n, f, d=1.0):
	j = f*(d*n)
	return np.where(j >= 0, j, n+j)
def rfreq2ind(n, f, d=1.0): return f*(n*d)

def shift(a, shift, axes=None, nofft=False, deriv=None, engine="auto"):
	"""Shift the array a by a (possibly fractional) number of samples "shift"
	to the right, along the specified axis, which defaults to the last one.
	shift can also be an array, in which case multiple axes are shifted together."""
	a      = np.asanyarray(a)
	ca     = a+0j
	shift  = np.atleast_1d(shift)
	if axes is None: axes = range(-len(shift),0)
	axes   = utils.astuple(axes)
	fa = fft(ca, axes=axes, engine=engine) if not nofft else ca
	for i, ax in enumerate(axes):
		ax   %= ca.ndim
		freqs = fftfreq(ca.shape[ax])
		phase = np.exp(-2j*np.pi*freqs*shift[i])
		if deriv == i:
			phase *= -2j*np.pi*freqs
		fa   *= phase[(None,)*ax + (slice(None),) + (None,)*(a.ndim-ax-1)]
	if not nofft: ifft(fa, ca, axes=axes, normalize=True, engine=engine)
	else:	      ca = fa
	return ca if np.iscomplexobj(a) else ca.real

def resample(a, n, axes=None, nthread=0, engine="auto"):
	"""Given an array a, resize the given axes (defaulting to the last ones) to
	length n (tuple or int) using Fourier resampling. For example, if a has shape
	(2,3,4), then resample(a, 10, -1) has shape (2,3,10), and resample(a, (20,10), (0,2))
	has shape (20,3,10)."""
	a    = np.asarray(a)
	n    = utils.astuple(n)
	if axes is None:
		axes = [-len(n)+i for i in range(len(n))]
	if len(n) != len(axes):
		raise ValueError("Resize size n = %s does not match axes = %s" % (str(n),str(axes)))
	fa   = fft(a, axes=axes, nthread=nthread, engine=engine)
	norm = 1/np.prod([a.shape[ax] for ax in axes])
	fa   = resample_fft(fa, n, axes=axes, norm=norm)
	out  = ifft(fa, axes=axes, normalize=False, nthread=nthread, engine=engine)
	if not np.iscomplexobj(a): out = out.real
	return out

def resample_fft(fa, n, out=None, axes=-1, norm=1, op=lambda a,b:b):
	"""Given array fa[{dims}] which is the fourier transform of some array a,
	transform it so that that it corresponds to the fourier transform of
	a version of a with a different number of samples by padding or truncating
	the fourier space. The argument n controls the new number of samples. By
	default this is for the last axis, but this can be changed using the axes
	argument. Multiple axes can be resampled at once by specifying a tuple for
	axes and n.

	The resulting array is multiplied by the argument norm. This can be used
	for normalization purposes. If norm is 1, then the multiplication is skipped.

	The argument out can be used to specify an already allocated output array.
	If it is None (the default), then an array will be allocated automatically.
	Normally the output array is overwritten, but this can be controlled using
	the op argument, which should be a function (out,fa)->out"""
	fa = np.asanyarray(fa)
	# Support n and axes being either tuples or a single number,
	# and broadcast n to match axes
	axes = utils.astuple(axes)
	n    = np.zeros(len(axes),int)+n
	# Determine the shape of the output array
	oshape = list(fa.shape)
	for i, ax in enumerate(axes):
		oshape[ax] = n[i]
	oshape = tuple(oshape)
	# Check or allocate output array
	if out is None:
		out = np.zeros(oshape, fa.dtype)
	else:
		if out.shape != oshape:
			raise ValueError("out argument has wrong shape in resample. Expected %s but got %s" % (str(oshape), str(out.shape)))
	# This function is used to avoid paying the cost of multiplying by norm when it's one
	def transfer(dest, source, norm, op):
		if norm != 1: source = source*norm
		dest[:] = op(dest, source)
	# Loop over start and end blocks for all dimensions
	for bi, I in enumerate(utils.nditer([2]*len(axes))):
		sel = [slice(None) for n in oshape]
		for ai, ax in enumerate(axes):
			c = min(fa.shape[ax], oshape[ax])
			if I[ai] == 0: sel[ax] = slice(0,c//2)
			else:          sel[ax] = slice(-(c-c//2),None)
		sel = tuple(sel)
		transfer(out[sel], fa[sel], norm, op)
	return out

def interpol_nufft(a, inds, out=None, axes=None, normalize=True,
		periodicity=None, epsilon=None, nthread=None, nofft=False, complex=False):
	"""Given some array a[{pre},{dims}] interpolate it at the given
	inds[len(dims),{post}], resulting in an output with shape [{pre},{post}].
	The signal is assumed to be periodic with the size of a unless this is overridden
	with the periodicity argument, which should have an integer for each axis being
	transformed. Normally the last ndim = len(inds) axes of a are interpolated.
	This can be overridden with the axes argument.

	By default the interpolation is properly normalized. This can be turned off
	with the normalization argument, in which case the output will be too high
	by a factor of np.prod([a.shape[ax] for ax in axes]). If all axes are used,
	this simplifies to a.size"""
	# This function could be implemented as simply u2nu(fft(a),inds). The problem
	# with this is that a full fourier-array needs to be allocated. I can save
	# some memory by instead doing the fft per field, at the cost of it being
	# a bit hacky
	op = None if nofft else lambda a, h: fft(a, nthread=h.nthread, axes=h.axall)
	return u2nu(a, inds, out=out, axes=axes, periodicity=periodicity,
		epsilon=epsilon, nthread=nthread, normalize=normalize, complex=complex, op=op)

def u2nu(fa, inds, out=None, axes=None, periodicity=None, epsilon=None, nthread=None,
			normalize=False, forward=False, complex=True, op=None):
	"""Given complex fourier coefficients fa[{pre},{dims}] corresponding to
	some real-space array a, evaluate the real-space signal at the given
	inds[len(dims),{post}], resulting in a output with shape [{pre},{post}].

	Arguments:
	* fa: Array of equi-spaced fourier coefficients. Complex with shape [{pre},{dims}]
	* inds: Array of positions at which to evaluate the inverse Fourier transform
	    of fa. Real with shape [len(dims),{post}]
	* out: Array to write result to. Real or complex with shape [{pre},{post}].
	    Optional. Allocated if missing.
	* axes: Tuple of axes to perform the transform along. len(axes)=len(dims).
	    Optional. Defaults to the last len(dims) axes.
	* periodicity: Periodicity assumed in the Fourier transform. Tuple with length
	   len(dims). Defaults to the shape of the axes being transformed.
	* epsilon: The target relative accuracy of the non-uniform FFT. Defaults
	   to 1e-5 for single precision and 1e-12 for double precision. See the
	   ducc0.nufft documentation for details.
	* normalize: If True, the output is divided by prod([fa.shape[ax] for ax in axes]),
	   that is, the total number of elements in the transform. This normalization is
	   equivalent to that of ifft. Defaults to False.
	* forward: Controls the sign of the exponent in the Fourier transform. By default
	   a backwards transform (fourier to real) is performed. By passing forward=True,
	   you can instead regard fa as a real-sapce array and out as a non-equispaced
	   Fourier array.
	* complex: Only relevant if out=None. Controls whether out is allocated as a
	   real or complex array. Defaults to complex.
	"""
	h = _nufft_helper(fa, out, inds, axes=axes, nuout=True, periodicity=periodicity,
		epsilon=epsilon, nthread=nthread, normalize=normalize, complex=complex)
	if op is None: op = lambda fa, h: fa
	for uI, nuI in zip(h.uiter, h.nuiter):
		grid = op(h.u[uI],h).astype(h.ctype, copy=False)
		res  = ducc0.nufft.u2nu(grid=grid, coord=h.iflat, forward=forward,
			epsilon=h.epsilon, nthreads=h.nthread, periodicity=h.periodicity,
			fft_order=True)
		if not np.iscomplexobj(h.nu):
			res = res.real
		h.nu[nuI] = res.reshape(h.inds.shape[1:])
	if h.normalize:
		h.nu /= h.norm
	return h.nu

# FIXME: Check normalization
def nu2u(a, inds, out=None, oshape=None, axes=None, periodicity=None, epsilon=None, nthread=None,
			normalize=False, forward=False):
	h = _nufft_helper(out, a, inds, axes=axes, nuout=False, periodicity=periodicity, ushape=oshape,
		epsilon=epsilon, nthread=nthread, normalize=normalize, complex=complex)
	work = np.zeros(h.tshape, h.ctype)
	for uI, nuI in zip(h.uiter, h.nuiter):
		res = ducc0.nufft.nu2u(points=h.nu[nuI], coord=h.iflat, out=work,
			forward=forward, epsilon=h.epsilon, nthreads=h.nthread,
			periodicity=h.periodicity, fft_order=True)
		if not np.iscomplexobj(h.u):
			res = res.real
		h.u[uI] = res
	if h.normalize:
		h.u /= h.norm
	return h.u

def iu2nu(a, inds, out=None, oshape=None, axes=None, periodicity=None, epsilon=None, nthread=None,
			normalize=False, forward=False):
	"""The inverse of nufft/u2nu. Given non-equispaced samples a[{pre},{post}] and
	their coordinates inds[len(dims),{post}], calculates the equispaced
	Fourier coefficients out[{pre},{dims}] of a.

	Arguments:
	* a: Array of of non-equispaced values. Real or complex with shape [{pre},{post}]
	* inds: Coordinates of samples in a. Real with shape [len(dims),{post}].
	* out: Equispaced Fourier coefficients of a. Complex with shape [{pre},{dims}].
	    Optional, but if missing, the shape of the out array to allocate must be
	    specified using the oshape argument
	* oshape: Tuple giving the shape to use when allocating out (if it's not passed in).
	See u2nu for the meaning of the other arguments.
	"""
	h = _nufft_helper(out, a, inds, axes=axes, nuout=False, ushape=oshape,
		periodicity=periodicity, epsilon=epsilon, nthread=nthread,
		normalize=normalize, complex=complex)
	work   = np.zeros(h.tshape, h.ctype)
	def wzip(u): return u.reshape(-1).view(h.rtype)
	def wunzip(x): return x.view(h.ctype).reshape(h.tshape)
	def P(u): return ducc0.nufft.u2nu(grid=u, coord=h.iflat, forward=forward,
		epsilon=h.epsilon, nthreads=h.nthread, periodicity=h.periodicity, fft_order=True)
	def PT(nu):
		return ducc0.nufft.nu2u(points=nu, coord=h.iflat,
		out=work, forward=not forward,
		epsilon=h.epsilon, nthreads=h.nthread, periodicity=h.periodicity, fft_order=True)
	for uI, nuI in zip(h.uiter, h.nuiter):
		# Invert u2nu by finding the least-squares solution to
		#  a = u2nu(out, inds). Written linearly this is a = P out
		# with solution out = (P'P)"P'a. The CG solver wants real numbers, though,
		# so we hack around that with view
		# Set up the equation system. Our degrees of freedom are flattened real u
		b = wzip(PT(h.nu[nuI].reshape(-1)))
		def A(x): return wzip(PT(P(wunzip(x))))
		solver = utils.CG(A, b)
		while solver.err > h.epsilon:
			solver.step()
		res = wunzip(solver.x)
		if not np.iscomplexobj(h.u):
			res = res.real
		h.u[uI] = res
	if h.normalize:
		h.u *= h.norm
	return h.u

# FIXME: Check normalization
def inu2u(fa, inds, out=None, axes=None, periodicity=None, epsilon=None, nthread=None,
			normalize=False, forward=False, complex=True):
	h = _nufft_helper(fa, out, inds, axes=axes, nuout=True,
		periodicity=periodicity, epsilon=epsilon, nthread=nthread,
		normalize=normalize, complex=complex)
	work = np.zeros(h.tshape, h.ctype)
	def wzip(nu): return nu.view(h.rtype)
	def wunzip(x): return x.view(h.ctype)
	def P(nu): return ducc0.nufft.nu2u(points=nu, coord=h.iflat, out=work, forward=forward,
		epsilon=h.epsilon, nthreads=h.nthread, periodicity=h.periodicity, fft_order=True)
	def PT(u): return ducc0.nufft.u2nu(grid=u, coord=h.iflat, forward=not forward,
		epsilon=h.epsilon, nthreads=h.nthread, periodicity=h.periodicity, fft_order=True)
	for uI, nuI in zip(h.uiter, h.nuiter):
		# Invert nu2u by finding the least-squares solution to
		#  fa = nu2u(out, inds). Written linearly this is fa = P out
		# with solution out = (P'P)"P'fa
		b = wzip(PT(h.u[uI]))
		def A(x): return wzip(PT(P(wunzip(x))))
		solver = utils.CG(A, b)
		while solver.err > h.epsilon:
			solver.step()
		res = wunzip(solver.x)
		if not np.iscomplexobj(h.nu):
			res = res.real
		h.nu[nuI] = res.reshape(h.inds.shape[1:])
	if h.normalize:
		h.nu *= h.norm
	return h.nu

# Alternative nufft interface more in line with fft and curvedsky.
# TODO: Add proper docstrings here. Can I avoid lots of repetition?

def nufft(a, inds, out=None, oshape=None, axes=None, periodicity=None, epsilon=None, nthread=None, normalize=False, flip=False):
	"""Alias of iu2nu(..., forward=flip). This involves inverting a system with conjugate gradients"""
	return iu2nu(a, inds, out=out, oshape=oshape, axes=axes, periodicity=periodicity, epsilon=epsilon, nthread=nthread, normalize=normalize, forward=flip)

def inufft(fa, inds, out=None, axes=None, periodicity=None, epsilon=None, nthread=None, normalize=False, flip=False, complex=True, op=None):
	"""Alias of u2nu(..., forward=flip)"""
	return u2nu(fa, inds, out=out, axes=axes, periodicity=periodicity, epsilon=epsilon, nthread=nthread, normalize=normalize, forward=flip, complex=complex, op=op)

def nufft_adjoint(a, inds, out=None, oshape=None, axes=None, periodicity=None, epsilon=None, nthread=None, normalize=False, flip=False):
	"""Alias of nu2u(..., forward=not flip)"""
	return nu2u(a, inds, out=out, oshape=oshape, axes=axes, periodicity=periodicity, epsilon=epsilon, nthread=nthread, normalize=normalize, forward=not flip)

def inufft_adjoint(fa, inds, out=None, axes=None, periodicity=None, epsilon=None, nthread=None, normalize=False, flip=False, complex=True):
	"""Alias of inu2u(..., forward=not flip). This involves inverting a system with conjugate gradients"""
	return inu2u(fa, inds, out=out, axes=axes, periodicity=periodicity, epsilon=epsilon, nthread=nthread, normalize=normalize, forward=not flip)

# Incremental nufft. This is useful for low-latency interpolation, or interpolation
# where you don't know how many points there will be

# Must figure out how much can be factorized out here. Sadly it wasn't
# practicaly to use _nufft_helper here. Will wait with factorization until
# I implement nu2u_plan
class u2nu_plan:
	def __init__(self, fa, axes, periodicity=None, epsilon=None, nthread=None,
			normalize=False, forward=False, complex=True, op=None):
		# Will set up one plan per fft-field
		fa           = np.asarray(fa)
		self.axes    = utils.astuple(axes)
		self.shape   = fa.shape
		self.gshape  = tuple([self.shape[ax] for ax in self.axes])
		self.paxes   = tuple(utils.complement_inds(self.axes, fa.ndim))
		self.pshape  = tuple([self.shape[ax] for ax in self.paxes])
		if op is None: op = lambda fa: fa
		if periodicity is None: periodicity = self.gshape
		else: periodicity = np.zeros(len(self.axes),int)+periodicity
		self.nthread = nthread or nthread_fft
		self.plans = []
		for I in utils.nditer(self.shape, axes=self.paxes):
			faI        = op(fa[I])
			self.ctype = faI.dtype
			# Target accuracy
			if epsilon is None:
				epsilon = 1e-5 if self.ctype == np.complex64 else 1e-12
			plan       = ducc0.nufft.experimental.incremental_u2nu(
				grid=faI, epsilon=epsilon, nthreads=self.nthread, forward=forward,
				periodicity=periodicity, fft_order=True)
			self.plans.append(plan)
		self.epsilon = epsilon
		self.forward = forward
		self.dtype   = utils.real_dtype(self.ctype)
		self.ndim    = len(self.axes)
		self.complex = complex
		self.norm    = np.prod([fa.shape[ax] for ax in axes])
		self.normalize = normalize
	def eval(self, inds, out=None):
		inds  = np.asarray(inds, dtype=self.dtype)
		iflat = inds.reshape(self.ndim,-1).T
		if out is None:
			out = np.zeros(self.pshape+inds.shape[1:], self.ctype if self.complex else self.dtype)
		oflat = out.reshape(len(self.plans),iflat.shape[0])
		for i, plan in enumerate(self.plans):
			vals = self.plans[i].get_points(coord=iflat)
			if not self.complex: vals = vals.real
			oflat[i] = vals
			del vals
		if self.normalize:
			out /= self.norm
		return out

########### Helper functions ##############


def fft_flat(tod, ft, nthread=1, axes=[-1], flags=None, _direction="FFTW_FORWARD"):
	"""Workaround for intel FFTW wrapper. Flattens appropriate dimensions of
	intput and output arrays to avoid crash that otherwise happens for arrays with
	ndim > N + 1, where N is the dimension of the transform. If 'axes' correspond
	to the last dimensions of the arrays, the workaround is essentially free. If
	`axes` correspond to other axes, copies are made when reshaping the arrays."""
	shape_ft = ft.shape
	naxes = np.atleast_1d(axes).size
	axes_new = list(range(-1, -1 - naxes, -1))
	ft = utils.partial_flatten(ft, axes=axes, pos=0)
	tod = utils.partial_flatten(tod, axes=axes, pos=0)
	plan = engines[engine].FFTW(tod, ft, flags=flags, threads=nthread, axes=axes_new, direction=_direction)
	plan()
	ft = utils.partial_expand(ft, shape_ft, axes=axes, pos=0)
	return ft

def ifft_flat(ft, tod, nthread=1, axes=[-1], flags=None):
	"""Same workaround as fft_flat but now for the inverse transform."""
	shape_tod = tod.shape
	naxes = np.atleast_1d(axes).size
	axes_new = list(range(-1, -1 - naxes, -1))
	tod = utils.partial_flatten(tod, axes=axes, pos=0)
	ft = utils.partial_flatten(ft, axes=axes, pos=0)
	plan = engines[engine].FFTW(ft, tod, flags=flags, direction='FFTW_BACKWARD',
		threads=nthread, axes=axes_new)
	plan(normalise_idft=False)
	tod = utils.partial_expand(tod, shape_tod, axes=axes, pos=0)
	return tod

def _nufft_helper(u, nu, inds, axes=None, periodicity=None, epsilon=None,
		nuout=False, nthread=None, complex=True, normalize=False, ushape=None):
	"""Do the type checking etc. needed to prepare for our nufft operations.
	This is a lot of code, but the overhead is around 300 µs, plus any time
	needed to allocate the output array. So there's a bit of overhead, but
	not anything we can't live with, and the raw ducc interface is available
	for when this overhead is too much."""
	from . import bunch
	# Prepare arguments for nufft operations. Must ensure that
	# * inds → iflat[ndim,npoint] f32 or f64
	# * ctype = c64 or c128 based u or nu, priority to which one is output which must be specified
	u     = np.asarray(u)  if u  is not None else None
	nu    = np.asarray(nu) if nu is not None else None
	inds  = np.asarray(inds)
	# Are we single or double precision? This set of statements sets up a priority
	# order for which array to get the dtype from
	dtypes = [nu if nuout else u, u if not nuout else nu, inds, np.float64()]
	rtypes = [utils.real_dtype(d.dtype) for d in dtypes if d is not None]
	rtype  = [d for d in rtypes if d in [np.float32, np.float64]][0]
	ctype  = utils.complex_dtype(rtype)
	if ctype not in [np.complex64, np.complex128]:
		raise ValueError("only single and double precision supported")
	# Convert inds to the right dtype only if it has an invalid dtype
	if inds.dtype not in [np.float32,np.float64]:
		inds = inds.astype(rtype, copy=False)
	ndim  = inds.shape[0]
	# By default the last ndim dimensions are transformed
	if axes is None: axes = tuple(range(-ndim,0))
	axes = utils.astuple(axes)
	if len(axes) != ndim: raise ValueError("Number of axes to transform does not match len(inds)!")
	# Set up output array. This depends on which direction we're going
	odtype = ctype if complex else rtype
	if nuout:
		npre   = u.ndim-ndim
		if npre < 0:
			raise ValueError("uniform array must has at least as many dimensions as indexed by the first axis of inds!")
		pshape = utils.without_inds(u.shape, axes)
		if nu is None:
			# Output array. Allocating it like this lets it inherit any subclass of
			# inds, which is useful when interpolating an enmap with another enmap
			nu = np.zeros_like(inds, shape=pshape+inds.shape[1:], dtype=odtype)
		if nu.shape != pshape+inds.shape[1:]:
			raise ValueError("nu must have shape pshape+inds.shape[1:]")
	else:
		if u is None:
			if ushape is None: raise ValueError("Either the output uniformly sampled array or its shape must be provided")
			u  = np.zeros(ushape, dtype=odtype)
		npre   = u.ndim-ndim
		pshape = utils.without_inds(u.shape, axes)
		# Hard to do any more sanity checks here
	tshape = utils.only_inds(u.shape, axes)
	npoint = np.prod(tshape)
	# Periodicity of the full space. Allows us to support arrays that represent
	# a subset of a bigger, periodic array
	if periodicity is None: periodicity = tshape
	else: periodicity = np.zeros(ndim,int)+periodicity
	nthread = nthread or nthread_fft
	# Target accuracy
	if epsilon is None:
		epsilon = 1e-5 if ctype == np.complex64 else 1e-12
	# ducc wants just a single pre-dimension for inds, so flatten it.
	iflat = inds.reshape(ndim,-1).T
	# Do the actual looping
	other_axes = tuple(utils.complement_inds(axes, u.ndim))
	axall = tuple(range(ndim))
	norm  = np.prod([u.shape[ax] for ax in axes])
	uiter  = utils.nditer(u.shape, axes=other_axes)
	nuiter = utils.nditer(nu.shape[:npre])
	return bunch.Bunch(u=u, nu=nu, inds=inds, iflat=iflat,
		epsilon=epsilon, nthread=nthread, normalize=normalize, norm=norm,
		periodicity=periodicity, pshape=pshape, tshape=tshape, npoint=npoint,
		other_axes=other_axes, axall=axall, complex=complex, rtype=rtype,
		ctype=ctype, npre=npre, uiter=uiter, nuiter=nuiter)
