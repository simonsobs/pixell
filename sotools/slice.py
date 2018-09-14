"""This module is intended to make it easier to implement slicing."""
from __future__ import print_function
import numpy as np
from .utils import cumsplit, listsplit, moveaxis

def expand_slice(sel, n, nowrap=False):
	"""Expands defaults and negatives in a slice to their implied values.
	After this, all entries of the slice are guaranteed to be present in their final form.
	Note, doing this twice may result in odd results, so don't send the result of this
	into functions that expect an unexpanded slice. Might be replacable with slice.indices()."""
	step = sel.step or 1
	def cycle(i,n):
		if nowrap: return i
		else: return min(i,n) if i >= 0 else n+i
	if step == 0: raise ValueError("slice step cannot be zero")
	if step > 0: return slice(cycle(sel.start or 0,n),cycle(sel.stop or n,n),step)
	else: return slice(cycle(sel.start or n-1, n), cycle(sel.stop,n) if sel.stop else -1, step)

def split_slice(sel, ndims):
	"""Splits a numpy-compatible slice "sel" into sub-slices sub[:], such that
	a[sel] = s[sub[0]][:,sub[1]][:,:,sub[2]][...], This is useful when
	implementing arrays with heterogeneous indices. Ndims indicates the number of
	indices to allocate to each split, starting from the left. Also expands all
	ellipsis."""
	if not isinstance(sel,tuple): sel = (sel,)
	# We know the total number of dimensions involved, so we can expand ellipis
	# What the heck? "in" operator is apparently broken for lists that
	# contain numpy arrays.
	parts = listsplit(sel, Ellipsis)
	if len(parts) > 1:
		# Only the rightmost ellipsis has any effect
		left, right = sum(parts[:-1],()), parts[-1]
		nfree = sum(ndims) - sum([i is not None for i in (left+right)])
		sel = left + tuple([slice(None) for i in range(nfree)]) + right
	return split_slice_simple(sel, ndims)

def split_slice_simple(sel, ndims):
	"""Helper function for split_slice. Splits a slice
	in the absence of ellipsis."""
	res = [[] for n in ndims]
	notNone = [v is not None for v in sel]
	subs = np.concatenate([[0],cumsplit(notNone, ndims)])
	for i, r in enumerate(res):
		r += sel[subs[i]:subs[i+1]]
	if subs[i+1] < len(sel):
		raise IndexError("Too many indices")
	return [tuple(v) for v in res]

def parse_slice(desc):
	class Foo:
		def __getitem__(self, p): return p
	foo = Foo()
	return eval("foo"+desc)

def slice_downgrade(d, s, axis=-1):
	"""Slice array d along the specified axis using the Slice s,
	but interpret the step part of the slice as downgrading rather
	than skipping."""
	a = moveaxis(d, axis, 0)
	step = s.step or 1
	a = a[s.start:s.stop:-1 if step < 0 else 1]
	step = abs(step)
	# Handle the whole blocks first
	a2 = a[:len(a)/step*step]
	a2 = np.mean(a2.reshape((len(a2)/step,step)+a2.shape[1:]),1)
	# Then append the incomplete block
	if len(a2)*step != len(a):
		rest = a[len(a2)*step:]
		a2 = np.concatenate([a2,[np.mean(rest,0)]],0)
	return moveaxis(a2, 0, axis)

def slice_union(a,b,n):
	"""Compute the effective slice from slicing first with a, then with b.
	Both must be slice objects. Simple for positive, explicit slices:
	res = a + b*a.step.
	But any of these may be None or negative, and we shouldn't need to know
	the length of the target array at this point. That makes it surprisingly
	complicated!
	"""
	astep = a.step or 1
	bstep = b.step or 1
	cstep = astep*bstep

	astart = (0 if astep > 0 else -1) if a.start is None else a.start
	bstart = (0 if bstep > 0 else -1) if b.start is None else b.start

	raise NotImplementedError
