import numpy as np
from . import utils, _array_ops_32, _array_ops_64

# This is a reduced version of enlib.array_ops. Everything that depends
# on lapack has been removed, as has complex number support.

def get_core(dtype):
	if   dtype == np.float32:    return _array_ops_32.array_ops
	elif dtype == np.float64:    return _array_ops_64.array_ops
	raise ValueError("Unsupported data type: %s" % str(dtype))

def find_contours(imap, vals, omap=None):
	core = get_core(imap.dtype)
	if omap is None:
		omap = imap.astype(np.int32)*0
	core.find_contours(imap.T, vals, omap.T)
	return omap

def ang2rect(a):
	core = get_core(a.dtype)
	res = np.zeros([len(a),3],dtype=a.dtype)
	core.ang2rect(a.T,res.T)
	return res

def matmul(A, B, axes=[-2,-1]):
	# Massage input arrays. This should be factored out,
	# as it is common for many functions
	axes = [i if i >= 0 else A.ndim+i for i in axes]
	bax  = axes[:len(axes)-(A.ndim-B.ndim)]
	Af = utils.partial_flatten(A,axes)
	Bf = utils.partial_flatten(B,bax)
	mustadd = Bf.ndim == 2
	if mustadd: Bf = Bf[:,None,:]
	Bf = np.ascontiguousarray(Bf)
	if A.dtype != B.dtype:
		dtype = np.result_type(A.dtype,B.dtype)
		Af = Af.astype(dtype,copy=False)
		Bf = Bf.astype(dtype,copy=False)
	# Compute the shape of the output array
	Xf = np.empty((Bf.shape[0],Bf.shape[1],Af.shape[1]),dtype=Bf.dtype)
	# Actually perform the operation
	core = get_core(Bf.dtype)
	core.matmul_multi(Af.T, Bf.T, Xf.T)
	# Unwrangle
	if mustadd: Xf = Xf[:,0,:]
	X = utils.partial_expand(Xf, B.shape, bax)
	return X

def wrap_mm_m(name, vec2mat=False):
	"""Wrap a fortran subroutine which takes (n,n,m),(n,k,m) and overwrites
	its second argument to a python function where the "n" axes can be
	at arbitrary locations, specified by the axes argument, and where
	the arrays can be arbitrary-dimensional. These are all flattened
	internally. If vec2mat is specified, the second argument will have
	a dummy axis added internally if needed."""
	def f(A,B,axes=[-2,-1]):
		axes = [i if i >= 0 else A.ndim+i for i in axes]
		bax  = axes[:len(axes)-(A.ndim-B.ndim)]
		B  = B.copy()
		Af = utils.partial_flatten(A,axes)
		Bf = utils.partial_flatten(B,bax)
		mustadd = vec2mat and Bf.ndim == 2
		if mustadd: Bf = Bf[:,None,:]
		Bf = np.ascontiguousarray(Bf)
		assert A.dtype == B.dtype
		core = get_core(A.dtype)
		fun  = getattr(core, name)
		fun(Af.T, Bf.T)
		if mustadd: Bf = Bf[:,0,:]
		B[...] = utils.partial_expand(Bf, B.shape, bax)
		return B
	return f

matmul_sym   = wrap_mm_m("matmul_multi_sym", vec2mat=True)
