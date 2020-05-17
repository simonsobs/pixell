"""This module handles resampling of time-series and similar arrays."""
import numpy as np
from . import utils, fft

def resample(d, factors=[0.5], axes=None, method="fft"):
	factors = np.atleast_1d(factors)
	if np.allclose(factors,1): return d
	if method == "fft":
		if axes is None: axes = range(-len(factors),0)
		lens = [int(d.shape[ax]*fact+0.5) for ax, fact in zip(axes, factors)]
		return resample_fft(d, lens, axes)
	elif method == "bin":
		return resample_bin(d, factors, axes)
	else:
		raise NotImplementedError("Resampling method '%s' is not implemented" % method)

def resample_bin(d, factors=[0.5], axes=None):
	if np.allclose(factors,1): return d
	down = [max(1,int(round(1/f))) for f in factors]
	up   = [max(1,int(round(f)))   for f in factors]
	d    = downsample_bin(d, down, axes)
	return upsample_bin  (d, up, axes)

def downsample_bin(d, steps=[2], axes=None):
	assert len(steps) <= d.ndim
	if axes is None: axes =range(-len(steps),0)
	assert len(axes) == len(steps)
	# Expand steps to cover every axis in order
	fullsteps = np.zeros(d.ndim,dtype=int)+1
	for ax, step in zip(axes, steps): fullsteps[ax]=step
	# Make each axis an even number of steps to prepare for reshape
	s = tuple([slice(0,L//step*step) for L,step in zip(d.shape,fullsteps)])
	d = d[s]
	# Reshape each axis to L/step,step to prepare for mean
	newshape = np.concatenate([[L//step,step] for L,step in zip(d.shape,fullsteps)])
	d = np.reshape(d, newshape)
	# And finally take the mean over all the extra axes
	return np.mean(d, tuple(range(1,d.ndim,2)))

def upsample_bin(d, steps=[2], axes=None):
	shape = d.shape
	assert len(steps) <= d.ndim
	if axes is None: axes = np.arange(-1,-len(steps)-1,-1)
	assert len(axes) == len(steps)
	# Expand steps to cover every axis in order
	fullsteps = np.zeros(d.ndim,dtype=int)+1
	for ax, step in zip(axes, steps): fullsteps[ax]=step
	# Reshape each axis to (n,1) to prepare for tiling
	newshape = np.concatenate([[L,1] for L in shape])
	d = np.reshape(d, newshape)
	# And tile
	d = np.tile(d, np.concatenate([[1,s] for s in fullsteps]))
	# Finally reshape back to proper dimensionality
	return np.reshape(d, np.array(shape)*np.array(fullsteps))

def resample_fft(d, n, axes=None):
	"""Resample numpy array d via fourier-reshaping. Requires periodic data.
	n indicates the desired output lengths of the axes that are to be
	resampled. By default the last len(n) axes are resampled, but this
	can be controlled via the axes argument."""
	d = np.asanyarray(d)
	# Compute output lengths from factors if necessary
	n = np.atleast_1d(n)
	if axes is None: axes = np.arange(-len(n),0)
	else: axes = np.atleast_1d(axes)
	if len(n) == 1: n = np.repeat(n, len(axes))
	else: assert len(n) == len(axes)
	assert len(n) <= d.ndim
	# Nothing to do?
	if np.all(d.shape[-len(n):] == n): return d
	# Use the simple version if we can. It has lower memory overhead
	if d.ndim == 2 and len(n) == 1 and (axes[0] == 1 or axes[0] == -1):
		return resample_fft_simple(d, n[0])
	# Perform the fourier transform
	fd = fft.fft(d, axes=axes)
	# Frequencies are 0 1 2 ... N/2 (-N)/2 (-N)/2+1 .. -1
	# Ex 0* 1 2* -1 for n=4 and 0* 1 2 -2 -1 for n=5
	# To upgrade,   insert (n_new-n_old) zeros after n_old/2
	# To downgrade, remove (n_old-n_new) values after n_new/2
	# The idea is simple, but arbitrary dimensionality makes it
	# complicated.
	norm = 1.0
	for ax, nnew in zip(axes, n):
		ax %= d.ndim
		nold = d.shape[ax]
		dn   = nnew-nold
		if dn > 0:
			padvals = np.zeros(fd.shape[:ax]+(dn,)+fd.shape[ax+1:],fd.dtype)
			spre  = tuple([slice(None)]*ax+[slice(0,nold//2)]+[slice(None)]*(fd.ndim-ax-1))
			spost = tuple([slice(None)]*ax+[slice(nold//2,None)]+[slice(None)]*(fd.ndim-ax-1))
			fd = np.concatenate([fd[spre],padvals,fd[spost]],axis=ax)
		elif dn < 0:
			spre  = tuple([slice(None)]*ax+[slice(0,nnew//2)]+[slice(None)]*(fd.ndim-ax-1))
			spost = tuple([slice(None)]*ax+[slice(nnew//2-dn,None)]+[slice(None)]*(fd.ndim-ax-1))
			fd = np.concatenate([fd[spre],fd[spost]],axis=ax)
		norm *= float(nnew)/nold
	# And transform back
	res  = fft.ifft(fd, axes=axes, normalize=True)
	del fd
	res *= norm
	return res if np.issubdtype(d.dtype, np.complexfloating) else res.real

def resample_fft_simple(d, n, ngroup=100):
	"""Resample 2d numpy array d via fourier-reshaping along
	last axis."""
	nold = d.shape[1]
	if n == nold: return d
	res  = np.zeros([d.shape[0],n],dtype=d.dtype)
	dn   = n-nold
	for di in range(0, d.shape[0], ngroup):
		fd = fft.fft(d[di:di+ngroup])
		if n < nold:
			fd = np.concatenate([fd[:,:n//2],fd[:,n//2-dn:]],1)
		else:
			fd = np.concatenate([fd[:,:nold//2],np.zeros([len(fd),n-nold],fd.dtype),fd[:,nold//2:]],-1)
		res[di:di+ngroup] = fft.ifft(fd, normalize=True).real
	del fd
	res *= float(n)/nold
	return res

def make_equispaced(d, t, quantile=0.1, order=3, mask_nan=False):
	"""Given an array d[...,nt] of data that has been sampled at times t[nt],
	return an array that has been resampled to have a constant sampling rate."""
	# Find the typical sampling rate of the input. We will lose information if
	# we don't use a sampling rate that's higher than the highest rate in the
	# input. But we also don't want to exaggerate the number of samples. Use a
	# configurable quantile as a compromise.
	dt   = np.percentile(np.abs(t[1:]-t[:-1]), quantile*100)
	# Modify so we get a whole number of samples
	nout = utils.nint(np.abs(t[-1]-t[0])/dt)+1
	dt   = (t[-1]-t[0])/(nout-1)
	# Construct our output time steps
	tout = np.arange(nout)*dt + t[0]
	# To interpolate, we need the input sample number as a function of time
	samples = np.interp(tout, t, np.arange(len(t)))
	# Now that we have the samples we can finally evaluate the function
	dout = utils.interpol(d, samples[None], mode="nearest", order=order, mask_nan=mask_nan)
	return dout, tout
