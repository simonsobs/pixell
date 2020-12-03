import numpy as np, scipy.ndimage, os, errno, scipy.optimize, time, datetime, warnings, re, sys
try: xrange
except: xrange = range
try: basestring
except: basestring = str

degree = np.pi/180
arcmin = degree/60
arcsec = arcmin/60
fwhm   = 1.0/(8*np.log(2))**0.5
T_cmb = 2.72548 # +/- 0.00057
c  = 299792458.0
h  = 6.62606957e-34
k  = 1.3806488e-23
AU = 149597870700.0
day2sec = 86400.
yr2days = 365.2422

# These are like degree, arcmin and arcsec, but turn any lists
# they touch into arrays.
a    = np.array(1.0)
adeg = np.array(degree)
amin = np.array(arcmin)
asec = np.array(arcsec)


def lines(file_or_fname):
	"""Iterates over lines in a file, which can be specified
	either as a filename or as a file object."""
	if isinstance(file_or_fname, basestring):
		with open(file_or_fname,"r") as file:
			for line in file: yield line
	else:
		for line in file_or_fname: yield line

def listsplit(seq, elem):
	"""Analogue of str.split for lists.
	listsplit([1,2,3,4,5,6,7],4) -> [[1,2],[3,4,5,6]]."""
	# Sadly, numpy arrays misbehave, and must be treated specially
	def iseq(e1, e2): return np.all(e1==e2)
	inds = [i for i,v in enumerate(seq) if iseq(v,elem)]
	ranges = zip([0]+[i+1 for i in inds],inds+[len(seq)])
	return [seq[a:b] for a,b in ranges]

def streq(x, s):
	"""Check if x is the string s. This used to be simply "x is s",
	but that now causes a warning. One can't just do "x == s", as
	that causes a numpy warning and will fail in the future."""
	return isinstance(x, basestring) and x == s

def find(array, vals, default=None):
	"""Return the indices of each value of vals in the given array."""
	if len(vals) == 0: return []
	array   = np.asarray(array)
	order   = np.argsort(array)
	cands   = np.minimum(np.searchsorted(array, vals, sorter=order),len(array)-1)
	res     = order[cands]
	bad     = array[res] != vals
	if np.any(bad):
		if default is None: raise ValueError("Value not found in array")
		else: res[bad] = default
	return res

def contains(array, vals):
	"""Given an array[n], returns a boolean res[n], which is True
	for any element in array that is also in vals, and False otherwise."""
	array = np.asarray(array)
	vals  = np.sort(vals)
	inds  = np.searchsorted(vals, array)
	# If a value would be inserted after the end, it wasn't
	# present in the original array.
	inds[inds>=len(vals)] = 0
	return vals[inds] == array

def common_vals(arrs):
	"""Given a list of arrays, returns their intersection.
	For example
	  common_vals([[1,2,3,4,5],[2,4,6,8]]) -> [2,4]"""
	res = arrs[0]
	for arr in arrs[1:]:
		res = np.intersect1d(res,arr)
	return res

def common_inds(arrs):
	"""Given a list of arrays, returns the indices into each of them of
	their common elements. For example
	  common_inds([[1,2,3,4,5],[2,4,6,8]]) -> [[1,3],[0,1]]"""
	vals = common_vals(arrs)
	return [find(arr, vals) for arr in arrs]

def union(arrs):
	"""Given a list of arrays, returns their union."""
	res = arrs[0]
	for arr in arrs[1:]:
		res = np.union1d(res,arr)
	return res

def dict_apply_listfun(dict, function):
	"""Applies a function that transforms one list to another
	with the same number of elements to the values in a dictionary,
	returning a new dictionary with the same keys as the input
	dictionary, but the values given by the results of the function
	acting on the input dictionary's values. I.e.
	if f(x) = x[::-1], then dict_apply_listfun({"a":1,"b":2},f) = {"a":2,"b":1}."""
	keys = dict.keys()
	vals = [dict[key] for key in keys]
	res  = function(vals)
	return {key: res[i] for i, key in enumerate(keys)}

def unwind(a, period=2*np.pi, axes=[-1], ref=0):
	"""Given a list of angles or other cyclic coordinates
	where a and a+period have the same physical meaning,
	make a continuous by removing any sudden jumps due to
	period-wrapping. I.e. [0.07,0.02,6.25,6.20] would
	become [0.07,0.02,-0.03,-0.08] with the default period
	of 2*pi."""
	res = rewind(a, period=period, ref=ref)
	for axis in axes:
		with flatview(res, axes=[axis]) as flat:
			# Avoid trying to sum nans
			mask = ~np.isfinite(flat)
			bad = flat[mask]
			flat[mask] = 0
			flat[:,1:]-= np.cumsum(np.round((flat[:,1:]-flat[:,:-1])/period),-1)*period
			# Restore any nans
			flat[mask] = bad
	return res

def rewind(a, ref=0, period=2*np.pi):
	"""Given a list of angles or other cyclic corodinates,
	add or subtract multiples of the period in order to ensure
	that they all lie within the same period. The ref argument
	specifies the angle furthest away from the cut, i.e. the
	period cut will be at ref+period/2."""
	a = np.asanyarray(a)
	if streq(ref, "auto"): ref = np.sort(a.reshape(-1))[a.size//2]
	return ref + (a-ref+period/2.)%period - period/2.

def cumsplit(sizes, capacities):
	"""Given a set of sizes (of files for example) and a set of capacities
	(of disks for example), returns the index of the sizes for which
	each new capacity becomes necessary, assuming sizes can be split
	across boundaries.
	For example cumsplit([1,1,2,0,1,3,1],[3,2,5]) -> [2,5]"""
	return np.searchsorted(np.cumsum(sizes),np.cumsum(capacities),side="right")

def mask2range(mask):
	"""Convert a binary mask [True,True,False,True,...] into
	a set of ranges [:,{start,stop}]."""
	# We consider the outside of the array to be False
	mask  = np.concatenate([[False],mask,[False]]).astype(np.int8)
	# Find where we enter and exit ranges with true mask
	dmask = mask[1:]-mask[:-1]
	start = np.where(dmask>0)[0]
	stop  = np.where(dmask<0)[0]
	return np.array([start,stop]).T

def repeat_filler(d, n):
	"""Form an array n elements long by repeatedly concatenating
	d and d[::-1]."""
	d = np.concatenate([d,d[::-1]])
	nmul = (n+d.size-1)//d.size
	dtot = np.concatenate([d]*nmul)
	return dtot[:n]

def deslope(d, w=1, inplace=False, axis=-1, avg=np.mean):
	"""Remove a slope and mean from d, matching up the beginning
	and end of d. The w parameter controls the number of samples
	from each end of d that is used to determine the value to
	match up."""
	if not inplace: d = np.array(d)
	with flatview(d, axes=[axis]) as dflat:
		for di in dflat:
			di -= np.arange(di.size)*(avg(di[-w:])-avg(di[:w]))/(di.size-1)+avg(di[:w])
	return d

def ctime2mjd(ctime):
	"""Converts from unix time to modified julian date."""
	return np.asarray(ctime)/86400. + 40587.0
def mjd2ctime(mjd):
	"""Converts from modified julian date to unix time."""
	return (np.asarray(mjd)-40587.0)*86400
def mjd2djd(mjd): return np.asarray(mjd) + 2400000.5 - 2415020
def djd2mjd(djd): return np.asarray(djd) - 2400000.5 + 2415020
def mjd2jd(mjd): return np.asarray(mjd) + 2400000.5
def jd2mjd(jd): return np.asarray(jd) - 2400000.5
def ctime2djd(ctime): return mjd2djd(ctime2mjd(ctime))
def djd2ctime(djd):    return mjd2ctime(djd2mjd(djd))

def mjd2ctime(mjd):
	"""Converts from modified julian date to unix time"""
	return (np.asarray(mjd)-40587.0)*86400

def medmean(x, frac=0.5):
	x = np.sort(x.reshape(-1))
	i = int(x.size*frac)//2
	return np.mean(x[i:-i])

def moveaxis(a, o, n):
	if o < 0: o = o+a.ndim
	if n < 0: n = n+a.ndim
	if n <= o: return np.rollaxis(a, o, n)
	else: return np.rollaxis(a, o, n+1)

def moveaxes(a, old, new):
	"""Move the axes listed in old to the positions given
	by new. This is like repeated calls to numpy rollaxis
	while taking into account the effect of previous rolls.

	This version is slow but simple and safe. It moves
	all axes to be moved to the end, and then moves them
	one by one to the target location."""
	# The final moves will happen in left-to-right order.
	# Hence, the first moves must be in the reverse of
	# this order.
	n = len(old)
	old   = np.asarray(old)
	order = np.argsort(new)
	rold  = old[order[::-1]]
	for i in range(n):
		a = moveaxis(a, rold[i], -1)
		# This may have moved some of the olds we're going to
		# move next, so update these
		for j in range(i+1,n):
			if rold[j] > rold[i]: rold[j] -= 1
	# Then do the final moves
	for i in range(n):
		a = moveaxis(a, -1, new[order[i]])
	return a

def partial_flatten(a, axes=[-1], pos=0):
	"""Flatten all dimensions of a except those mentioned
	in axes, and put the flattened one at the given position.

	Example: if a.shape is [1,2,3,4],
	then partial_flatten(a,[-1],0).shape is [6,4]."""
	# Move the selected axes first
	a = moveaxes(a, axes, range(len(axes)))
	# Flatten all the other axes
	a = a.reshape(a.shape[:len(axes)]+(-1,))
	# Move flattened axis to the target position
	return moveaxis(a, -1, pos)

def partial_expand(a, shape, axes=[-1], pos=0):
	"""Undo a partial flatten. Shape is the shape of the
	original array before flattening, and axes and pos should be
	the same as those passed to the flatten operation."""
	a = moveaxis(a, pos, -1)
	axes = np.array(axes)%len(shape)
	rest = list(np.delete(shape, axes))
	a = np.reshape(a, list(a.shape[:len(axes)])+rest)
	return moveaxes(a, range(len(axes)), axes)

def addaxes(a, axes):
	axes = np.array(axes,int)
	axes[axes<0] += a.ndim
	axes = np.sort(axes)[::-1]
	inds = [slice(None) for i in a.shape]
	for ax in axes: inds.insert(ax, None)
	return a[inds]

def delaxes(a, axes):
	axes = np.array(axes,int)
	axes[axes<0] += a.ndim
	axes = np.sort(axes)[::-1]
	inds = [slice(None) for i in a.shape]
	for ax in axes: inds[ax] = 0
	return a[inds]

class flatview:
	"""Produce a read/writable flattened view of the given array,
	via with flatview(arr) as farr:
		do stuff with farr
	Changes to farr are propagated into the original array.
	Flattens all dimensions of a except those mentioned
	in axes, and put the flattened one at the given position."""
	def __init__(self, array, axes=[], mode="rwc", pos=0):
		self.array = array
		self.axes  = axes
		self.flat  = None
		self.mode  = mode
		self.pos   = pos
	def __enter__(self):
		self.flat = partial_flatten(self.array, self.axes, pos=self.pos)
		if "c" in self.mode:
			self.flat = np.ascontiguousarray(self.flat)
		return self.flat
	def __exit__(self, type, value, traceback):
		# Copy back out from flat into the original array, if necessary
		if "w" not in self.mode: return
		if np.shares_memory(self.array, self.flat): return
		# We need to copy back out
		self.array[:] = partial_expand(self.flat, self.array.shape, self.axes, pos=self.pos)

class nowarn:
	"""Use in with block to suppress warnings inside that block."""
	def __enter__(self):
		self.filters = list(warnings.filters)
		warnings.filterwarnings("ignore")
		return self
	def __exit__(self, type, value, traceback):
		warnings.filters = self.filters

def dedup(a):
	"""Removes consecutive equal values from a 1d array, returning the result.
	The original is not modified."""
	return a[np.concatenate([[True],a[1:]!=a[:-1]])]

def interpol(a, inds, order=3, mode="nearest", mask_nan=False, cval=0.0, prefilter=True):
	"""Given an array a[{x},{y}] and a list of float indices into a,
	inds[len(y),{z}], returns interpolated values at these positions as [{x},{z}]."""
	a    = np.asanyarray(a)
	inds = np.asanyarray(inds)
	inds_orig_nd = inds.ndim
	if inds.ndim == 1: inds = inds[:,None]

	npre = a.ndim - inds.shape[0]
	res = np.empty(a.shape[:npre]+inds.shape[1:],dtype=a.dtype)
	fa, fr = partial_flatten(a, range(npre,a.ndim)), partial_flatten(res, range(npre, res.ndim))
	if mask_nan:
		mask = ~np.isfinite(fa)
		fa[mask] = 0
	for i in range(fa.shape[0]):
		fr[i].real = scipy.ndimage.map_coordinates(fa[i].real, inds, order=order, mode=mode, cval=cval, prefilter=prefilter)
		if np.iscomplexobj(fa[i]):
			fr[i].imag = scipy.ndimage.map_coordinates(fa[i].imag, inds, order=order, mode=mode, cval=cval, prefilter=prefilter)
	if mask_nan and np.sum(mask) > 0:
		fmask = np.empty(fr.shape,dtype=bool)
		for i in range(mask.shape[0]):
			fmask[i] = scipy.ndimage.map_coordinates(mask[i], inds, order=0, mode=mode, cval=cval, prefilter=prefilter)
		fr[fmask] = np.nan
	if inds_orig_nd == 1: res = res[...,0]
	return res

def interpol_prefilter(a, npre=None, order=3, inplace=False):
	a = np.asanyarray(a)
	if not inplace: a = a.copy()
	if npre is None: npre = max(0,a.ndim - 2)
	with flatview(a, range(npre, a.ndim), "rw") as aflat:
		for i in range(len(aflat)):
			aflat[i] = scipy.ndimage.spline_filter(aflat[i], order=order)
	return a

def bin_multi(pix, shape, weights=None):
	"""Simple multidimensional binning. Not very fast.
	Given pix[{coords},:] where coords are indices into an array
	with shape shape, count the number of hits in each pixel,
	returning map[shape]."""
	pix  = np.maximum(np.minimum(pix, (np.array(shape)-1)[:,None]),0)
	inds = np.ravel_multi_index(tuple(pix), tuple(shape))
	size = np.product(shape)
	if weights is not None: weights = inds*0+weights
	return np.bincount(inds, weights=weights, minlength=size).reshape(shape)

def grid(box, shape, endpoint=True, axis=0, flat=False):
	"""Given a bounding box[{from,to},ndim] and shape[ndim] in each
	direction, returns an array [ndim,shape[0],shape[1],...] array
	of evenly spaced numbers. If endpoint is True (default), then
	the end point is included. Otherwise, the last sample is one
	step away from the end of the box. For one dimension, this is
	similar to linspace:
		linspace(0,1,4)     =>  [0.0000, 0.3333, 0.6667, 1.0000]
		grid([[0],[1]],[4]) => [[0,0000, 0.3333, 0.6667, 1.0000]]
	"""
	n    = np.asarray(shape)
	box  = np.asfarray(box)
	off  = -1 if endpoint else 0
	inds = np.rollaxis(np.indices(n),0,len(n)+1) # (d1,d2,d3,...,indim)
	res  = inds * (box[1]-box[0])/(n+off) + box[0]
	if flat: res = res.reshape(-1, res.shape[-1])
	return np.rollaxis(res, -1, axis)

def cumsum(a, endpoint=False):
	"""As numpy.cumsum for a 1d array a, but starts from 0. If endpoint is True, the result
	will have one more element than the input, and the last element will be the sum of the
	array. Otherwise (the default), it will have the same length as the array, and the last
	element will be the sum of the first n-1 elements."""
	res = np.concatenate([[0],np.cumsum(a)])
	return res if endpoint else res[:-1]

def nearest_product(n, factors, direction="below"):
	"""Compute the highest product of positive integer powers of the specified
	factors that is lower than or equal to n. This is done using a simple,
	O(n) brute-force algorithm."""
	if 1 in factors: return n
	below = direction=="below"
	nmax = n+1 if below else n*min(factors)+1
	# a keeps track of all the visited multiples
	a = np.zeros(nmax+1,dtype=bool)
	a[1] = True
	best = None
	for i in range(n+1):
		if not a[i]: continue
		for f in factors:
			m = i*f
			if below:
				if m > n: continue
				else: best = m
			else:
				if m >= n and (best is None or best > m):
					best = m
			if m < a.size:
				a[m] = True
	return best

def mkdir(path):
	# It's useful to be able to do mkdir(os.path.dirname(fname)) to create the directory
	# a file should be in if it's missing. If fname has no directory component dirname
	# returns "". This check prevents this from causing an error.
	if path == "": return
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

def decomp_basis(basis, vec):
	return np.linalg.solve(basis.dot(basis.T),basis.dot(vec.T)).T

def find_period(d, axis=-1):
	dwork = partial_flatten(d, [axis])
	guess = find_period_fourier(dwork)
	res = np.empty([3,len(dwork)])
	for i, (d1, g1) in enumerate(zip(dwork, guess)):
		res[:,i] = find_period_exact(d1, g1)
	periods = res[0].reshape(d.shape[:axis]+d.shape[axis:][1:])
	phases  = res[1].reshape(d.shape[:axis]+d.shape[axis:][1:])
	chisqs  = res[2].reshape(d.shape[:axis]+d.shape[axis:][1:])
	return periods, phases, chisqs

def find_period_fourier(d, axis=-1):
	"""This is a simple second-order estimate of the period of the
	assumed-periodic signal d. It finds the frequency with the highest
	power using an fft, and partially compensates for nonperiodicity
	by taking a weighted mean of the position of the top."""
	d2 = partial_flatten(d, [axis])
	fd  = np.fft.rfft(d2)
	ps = np.abs(fd)**2
	ps[:,0] = 0
	periods = []
	for p in ps:
		n = np.argmax(p)
		r = [int(n*0.5),int(n*1.5)+1]
		denom = np.sum(p[r[0]:r[1]])
		if denom <= 0: denom = 1
		n2 = np.sum(np.arange(r[0],r[1])*p[r[0]:r[1]])/denom
		periods.append(float(d.shape[axis])/n2)
	return np.array(periods).reshape(d.shape[:axis]+d.shape[axis:][1:])

def find_period_exact(d, guess):
	n = d.size
	# Restrict to at most 10 fiducial periods
	n = int(min(10,n/float(guess))*guess)
	off = (d.size-n)//2
	d = d[off:off+n]
	def chisq(x):
		w,phase = x
		model = interpol(d, np.arange(n)[None]%w+phase, order=1)
		return np.var(d-model)
	period,phase = scipy.optimize.fmin_powell(chisq, [guess,guess], xtol=1, disp=False)
	return period, phase+off, chisq([period,phase])/np.var(d**2)

def equal_split(weights, nbin):
	"""Split weights into nbin bins such that the total
	weight in each bin is as close to equal as possible.
	Returns a list of indices for each bin."""
	inds = np.argsort(weights)[::-1]
	bins = [[] for b in xrange(nbin)]
	bw   = np.zeros([nbin])
	for i in inds:
		j = np.argmin(bw)
		bins[j].append(i)
		bw[j] += weights[i]
	return bins

def range_sub(a,b, mapping=False):
	"""Given a set of ranges a[:,{from,to}] and b[:,{from,to}],
	return a new set of ranges c[:,{from,to}] which corresponds to
	the ranges in a with those in b removed. This might split individual
	ranges into multiple ones. If mapping=True, two extra objects are
	returned. The first is a mapping from each output range to the
	position in a it comes from. The second is a corresponding mapping
	from the set of cut a and b range to indices into a and b, with
	b indices being encoded as -i-1. a and b are assumed
	to be internally non-overlapping.

	Example: utils.range_sub([[0,100],[200,1000]], [[1,2],[3,4],[8,999]], mapping=True)
	(array([[   0,    1],
	        [   2,    3],
	        [   4,    8],
	        [ 999, 1000]]),
	array([0, 0, 0, 1]),
	array([ 0, -1,  1, -2,  2, -3,  3]))

	The last array can be interpreted as: Moving along the number line,
	we first encounter [0,1], which is a part of range 0 in c. We then
	encounter range 0 in b ([1,2]), before we hit [2,3] which is
	part of range 1 in c. Then comes range 1 in b ([3,4]) followed by
	[4,8] which is part of range 2 in c, followed by range 2 in b
	([8,999]) and finally [999,1000] which is part of range 3 in c.

	The same call without mapping: utils.range_sub([[0,100],[200,1000]], [[1,2],[3,4],[8,999]])
	array([[   0,    1],
	       [   2,    3],
	       [   4,    8],
	       [ 999, 1000]])
	"""
	def fixshape(a):
		a = np.asarray(a)
		if a.size == 0: a = np.zeros([0,2],dtype=int)
		return a
	a     = fixshape(a)
	b     = fixshape(b)
	ainds = np.argsort(a[:,0])
	binds = np.argsort(b[:,0])
	rainds= np.arange(len(a))[ainds]
	rbinds= np.arange(len(b))[binds]
	a = a[ainds]
	b = b[binds]
	ai,bi = 0,0
	c = []
	abmap = []
	rmap  = []
	while ai < len(a):
		# Iterate b until it ends past the start of a
		while bi < len(b) and b[bi,1] <= a[ai,0]:
			abmap.append(-rbinds[bi]-1)
			bi += 1
		# Now handle each b until the start of b is past the end of a
		pstart = a[ai,0]
		while bi < len(b) and b[bi,0] <= a[ai,1]:
			r=(pstart,min(a[ai,1],b[bi,0]))
			if r[1]-r[0] > 0:
				abmap.append(len(c))
				rmap.append(rainds[ai])
				c.append(r)
			abmap.append(-rbinds[bi]-1)
			pstart = b[bi,1]
			bi += 1
		# Then append what remains
		r=(pstart,a[ai,1])
		if r[1]>r[0]:
			abmap.append(len(c))
			rmap.append(rainds[ai])
			c.append(r)
		else:
			# If b extended beyond the end of a, then
			# we need to consider it again for the next a,
			# so undo the previous increment. This may lead to
			# the same b being added twice. We will handle that
			# by removing duplicates at the end.
			bi -= 1
		# And advance to the next range in a
		ai += 1
	c = np.array(c)
	# Remove duplicates if necessary
	abmap=dedup(np.array(abmap))
	rmap = np.array(rmap)
	return (c, rmap, abmap) if mapping else c

def range_union(a, mapping=False):
	"""Given a set of ranges a[:,{from,to}], return a new set where all
	overlapping ranges have been merged, where to >= from. If mapping=True,
	then the mapping from old to new ranges is also returned."""
	# We will make a single pass through a in sorted order
	a    = np.asarray(a)
	n    = len(a)
	inds = np.argsort(a[:,0])
	rmap = np.zeros(n,dtype=int)-1
	b    = []
	# i will point at the first unprocessed range
	for i in xrange(n):
		if rmap[inds[i]] >= 0: continue
		rmap[inds[i]] = len(b)
		start, end = a[inds[i]]
		# loop through every unprocessed range in range
		for j in xrange(i+1,n):
			if rmap[inds[j]] >= 0: continue
			if a[inds[j],0] > end: break
			# This range overlaps, so register it and merge
			rmap[inds[j]] = len(b)
			end = max(end, a[inds[j],1])
		b.append([start,end])
	b = np.array(b)
	if b.size == 0: b = b.reshape(0,2)
	return (b,rmap) if mapping else b

def range_normalize(a):
	"""Given a set of ranges a[:,{from,to}], normalize the ranges
	such that no ranges are empty, and all ranges go in increasing
	order. Decreasing ranges are interpreted the same way as in a slice,
	e.g. empty."""
	a = np.asarray(a)
	n1 = len(a)
	a = a[a[:,1]!=a[:,0]]
	reverse = a[:,1]<a[:,0]
	a = a[~reverse]
	n2 = len(a)
	return a

def range_cut(a, c):
	"""Cut range list a at positions given by c. For example
	range_cut([[0,10],[20,100]],[0,2,7,30,200]) -> [[0,2],[2,7],[7,10],[20,30],[30,100]]."""
	return range_sub(a,np.dstack([c,c])[0])

def compress_beam(sigma, phi):
	sigma = np.asarray(sigma,dtype=float)
	c,s=np.cos(phi),np.sin(phi)
	R = np.array([[c,-s],[s,c]])
	C = np.diag(sigma**-2)
	C = R.dot(C).dot(R.T)
	return np.array([C[0,0],C[1,1],C[0,1]])

def expand_beam(irads, return_V=False):
	C = np.array([[irads[0],irads[2]],[irads[2],irads[1]]])
	E, V = np.linalg.eigh(C)
	phi = np.arctan2(V[1,0],V[0,0])
	sigma = E**-0.5
	if sigma[1] > sigma[0]:
		sigma = sigma[::-1]
		phi += np.pi/2
	phi %= np.pi
	if return_V: return sigma, phi, V
	else: return sigma, phi

def combine_beams(irads_array):
	Cs = np.array([[[ir[0],ir[2]],[ir[2],ir[1]]] for ir in irads_array])
	Ctot = np.eye(2)
	for C in Cs:
		E, V = np.linalg.eigh(C)
		B = (V*E[None]**0.5).dot(V.T)
		Ctot = B.dot(Ctot).dot(B.T)
	return np.array([Ctot[0,0],Ctot[1,1],Ctot[0,1]])

def regularize_beam(beam, cutoff=1e-2, nl=None):
	"""Given a beam transfer function beam[...,nl], replace
	small values with an extrapolation that has the property
	that the ratio of any pair of such regularized beams is
	constant in the extrapolated region."""
	beam  = np.asarray(beam)
	# Get the length of the output beam, and the l to which both exist
	if nl is None: nl = beam.shape[-1]
	nl_both = min(nl, beam.shape[-1])
	# Build the extrapolation for the full range. We will overwrite the part
	# we want to keep unextrapolated later.
	l     = np.maximum(1,np.arange(nl))
	vcut  = np.max(beam,-1)*cutoff
	above = beam > vcut
	lcut  = np.argmin(above, -1)
	if lcut > nl or lcut == 0: return beam[:nl]
	obeam = vcut * (l/lcut)**(2*np.log(cutoff))
	# Get the mask for what we want to keep. This looks complicated, but that's
	# just to support arbitrary-dimensionality (maybe that wasn't really necessary).
	mask  = np.zeros(obeam.shape, int)
	iflat = lcut.reshape(-1) + np.arange(lcut.size)*nl
	mask.reshape(-1)[iflat] = 1
	mask  = np.cumsum(mask,-1) < 0.5
	obeam[:nl_both] = np.where(mask[:nl_both], beam[:nl_both], obeam[:nl_both])
	return obeam

def read_lines(fname, col=0):
	"""Read lines from file fname, returning them as a list of strings.
	If fname ends with :slice, then the specified slice will be applied
	to the list before returning."""
	toks = fname.split(":")
	fname, fslice = toks[0], ":".join(toks[1:])
	lines = [line.split()[col] for line in open(fname,"r") if line[0] != "#"]
	n = len(lines)
	return eval("lines"+fslice)

def loadtxt(fname):
	"""As numpy.loadtxt, but allows slice syntax."""
	toks = fname.split(":")
	fname, fslice = toks[0], ":".join(toks[1:])
	a = np.loadtxt(fname)
	return eval("a"+fslice)

def atleast_3d(a):
	a = np.asanyarray(a)
	if a.ndim == 0: return a.reshape(1,1,1)
	elif a.ndim == 1: return a.reshape(1,1,-1)
	elif a.ndim == 2: return a.reshape((1,)+a.shape)
	else: return a

def to_Nd(a, n, return_inverse=False):
	a = np.asanyarray(a)
	if n >= a.ndim:
		res = a.reshape((1,)*(n-a.ndim)+a.shape)
	else:
		res = a.reshape((-1,)+a.shape[1:])
	return (res, a.shape) if return_inverse else res

def between_angles(a, range, period=2*np.pi):
	a = rewind(a, np.mean(range), period=period)
	return (a>=range[0])&(a<range[1])

def greedy_split(data, n=2, costfun=max, workfun=lambda w,x: x if w is None else x+w):
	"""Given a list of elements data, return indices that would
	split them it into n subsets such that cost is approximately
	minimized. costfun specifies which cost to minimize, with
	the default being the value of the data themselves. workfun
	specifies how to combine multiple values. workfun(datum,workval)
	=> workval. scorefun then operates on a list of the total workval
	for each group score = scorefun([workval,workval,....]).

	Example: greedy_split(range(10)) => [[9,6,5,2,1,0],[8,7,4,3]]
	         greedy_split([1,10,100]) => [[2],[1,0]]
	         greedy_split("012345",costfun=lambda x:sum([xi**2 for xi in x]),
	          workfun=lambda w,x:0 if x is None else int(x)+w)
	          => [[5,2,1,0],[4,3]]
	"""
	# Sort data based on standalone costs
	costs = []
	nowork = workfun(None,None)
	work = [nowork for i in xrange(n)]
	for d in data:
		work[0] = workfun(nowork,d)
		costs.append(costfun(work))
	order = np.argsort(costs)[::-1]
	# Build groups using greedy algorithm
	groups = [[] for i in xrange(n)]
	work   = [nowork for i in xrange(n)]
	cost   = costfun(work)
	for di in order:
		d = data[di]
		# Try adding to each group
		for i in xrange(n):
			iwork = workfun(work[i],d)
			icost = costfun(work[:i]+[iwork]+work[i+1:])
			if i == 0 or icost < best[2]: best = (i,iwork,icost)
		# Add it to the best group
		i, iwork, icost = best
		groups[i].append(di)
		work[i] = iwork
		cost = icost
	return groups, cost, work

def greedy_split_simple(data, n=2):
	"""Split array "data" into n lists such that each list has approximately the same
	sum, using a greedy algorithm."""
	inds = np.argsort(data)[::-1]
	rn   = range(n)
	sums = [0  for i in rn]
	res  = [[] for i in rn]
	for i in inds:
		small = 0
		for j in rn:
			if sums[j] < sums[small]: small = j
		sums[small] += data[i]
		res[small].append(i)
	return res

def cov2corr(C):
	"""Scale rows and columns of C such that its diagonal becomes one.
	This produces a correlation matrix from a covariance matrix. Returns
	the scaled matrix and the square root of the original diagonal."""
	std  = np.diag(C)**0.5
	istd = 1/std
	return np.einsum("ij,i,j->ij",C,istd,istd), std
def corr2cov(corr,std):
	"""Given a matrix "corr" and an array "std", return a version
	of corr with each row and column scaled by the corresponding entry
	in std. This is the reverse of cov2corr."""
	return np.einsum("ij,i,j->ij",corr,std,std)

def eigsort(A, nmax=None, merged=False):
	"""Return the eigenvalue decomposition of the real, symmetric matrix A.
	The eigenvalues will be sorted from largest to smallest. If nmax is
	specified, only the nmax largest eigenvalues (and corresponding vectors)
	will be returned. If merged is specified, E and V will not be returned
	separately. Instead, Q=VE**0.5 will be returned, such that QQ' = VEV'."""
	E,V  = np.linalg.eigh(A)
	inds = np.argsort(E)[::-1][:nmax]
	if merged: return V[:,inds]*E[inds][None]**0.5
	else:      return E[inds],V[:,inds]

def nodiag(A):
	"""Returns matrix A with its diagonal set to zero."""
	A = np.array(A)
	np.fill_diagonal(A,0)
	return A

def date2ctime(dstr):
	import dateutil.parser
	d = dateutil.parser.parse(dstr, ignoretz=True, tzinfos=0)
	return time.mktime(d.timetuple())

def bounding_box(boxes):
	"""Compute bounding box for a set of boxes [:,2,:], or a
	set of points [:,2]"""
	boxes = np.asarray(boxes)
	if boxes.ndim == 2:
		return np.array([np.min(boxes,0),np.max(boxes,0)])
	else:
		return np.array([np.min(boxes[:,0,:],0),np.max(boxes[:,1,:],0)])

def unpackbits(a): return np.unpackbits(np.atleast_1d(a).view(np.uint8)[::-1])[::-1]

def box2corners(box):
	"""Given a [{from,to},:] bounding box, returns [ncorner,:] coordinates
	of of all its corners."""
	box = np.asarray(box)
	ndim= box.shape[1]
	return np.array([[box[b,bi] for bi,b in enumerate(unpackbits(i)[:ndim])] for i in range(2**ndim)])

def box2contour(box, nperedge=5):
	"""Given a [{from,to},:] bounding box, returns [npoint,:] coordinates
	definiting its edges. Nperedge is the number of samples per edge of
	the box to use. For nperedge=2 this is equal to box2corners. Nperegege
	can be a list, in which case the number indicates the number to use in
	each dimension."""
	box      = np.asarray(box)
	ndim     = box.shape[1]
	nperedge = np.zeros(ndim,int)+nperedge
	# Generate the range of each coordinate
	points = []
	for i in range(ndim):
		x = np.linspace(box[0,i],box[1,i],nperedge[i])
		for j in range(2**ndim):
			bits = unpackbits(j)[:ndim]
			if bits[i]: continue
			y = np.zeros((len(x),ndim))
			y[:] = box[bits,np.arange(ndim)]; y[:,i] = x
			points.append(y)
	return np.concatenate(points,0)

def box_slice(a, b):
	"""Given two boxes/boxarrays of shape [{from,to},dims] or [:,{from,to},dims],
	compute the bounds of the part of each b that overlaps with each a, relative
	to the corner of a. For example box_slice([[2,5],[10,10]],[[0,0],[5,7]]) ->
	[[0,0],[3,2]]."""
	a  = np.asarray(a)
	b  = np.asarray(b)
	fa = a.reshape(-1,2,a.shape[-1])
	fb = b.reshape(-1,2,b.shape[-1])
	s  = np.minimum(np.maximum(0,fb[None,:]-fa[:,None,0,None]),fa[:,None,1,None]-fa[:,None,0,None])
	return s.reshape(a.shape[:-2]+b.shape[:-2]+(2,2))

def box_area(a):
	"""Compute the area of a [{from,to},ndim] box, or an array of such boxes."""
	return np.abs(np.product(a[...,1,:]-a[...,0,:],-1))

def box_overlap(a, b):
	"""Given two boxes/boxarrays, compute the overlap of each box with each other
	box, returning the area of the overlaps. If a is [2,ndim] and b is [2,ndim], the
	result will be a single number. if a is [n,2,ndim] and b is [2,ndim], the result
	will be a shape [n] array. If a is [n,2,ndim] and b is [m,2,ndim], the result will'
	be [n,m] areas."""
	return box_area(box_slice(a,b))

def widen_box(box, margin=1e-3, relative=True):
	box = np.asarray(box)
	margin = np.zeros(box.shape[1:])+margin
	if relative: margin = (box[1]-box[0])*margin
	margin = np.asarray(margin) # Support 1d case
	margin[box[0]>box[1]] *= -1
	return np.array([box[0]-margin/2, box[1]+margin/2])

def unwrap_range(range, nwrap=2*np.pi):
	"""Given a logically ordered range[{from,to},...] that
	may have been exposed to wrapping with period nwrap,
	undo the wrapping so that range[1] > range[0]
	but range[1]-range[0] is as small as possible.
	Also makes the range straddle 0 if possible.

	Unlike unwind and rewind, this function will not
	turn a very wide range into a small one because it
	doesn't assume that ranges are shorter than half the
	sky. But it still shortens ranges that are longer than
	a whole wrapping period."""
	range = np.asanyarray(range)
	range[1] -= np.floor((range[1]-range[0])/nwrap)*nwrap
	range    -= np.floor(range[1,None]/nwrap)*nwrap
	return range

def sum_by_id(a, ids, axis=0):
	ra = moveaxis(a, axis, 0)
	fa = ra.reshape(ra.shape[0],-1)
	fb = np.zeros((np.max(ids)+1,fa.shape[1]),fa.dtype)
	for i,id in enumerate(ids):
		fb[id] += fa[i]
	rb = fb.reshape((fb.shape[0],)+ra.shape[1:])
	return moveaxis(rb, 0, axis)

def pole_wrap(pos):
	"""Given pos[{lat,lon},...], normalize coordinates so that
	lat is always between -pi/2 and pi/2. Coordinates outside this
	range are mirrored around the poles, and for each mirroring a phase
	of pi is added to lon."""
	pos = pos.copy()
	lat, lon  = pos # references to columns of pos
	halforbit = np.floor((lat+np.pi/2)/np.pi).astype(int)
	front     = halforbit % 2 == 0
	back      = ~front
	# Get rid of most of the looping
	lat -= np.pi*halforbit
	# Then handle the "backside" of the sky, where lat is between pi/2 and 3pi/2
	lat[back] = -lat[back]
	lon[back]+= np.pi
	return pos

def parse_box(desc):
	"""Given a string of the form from:to,from:to,from:to,... returns
	an array [{from,to},:]"""
	return np.array([[float(word) for word in pair.split(":")] for pair in desc.split(",")]).T

def allreduce(a, comm, op=None):
	"""Convenience wrapper for Allreduce that returns the result
	rather than needing an output argument."""
	res = a.copy()
	if op is None: comm.Allreduce(a, res)
	else:          comm.Allreduce(a, res, op)
	return res

def reduce(a, comm, root=0, op=None):
	res = a.copy() if comm.rank == root else None
	if op is None: comm.Reduce(a, res, root=root)
	else:          comm.Reduce(a, res, op, root=root)
	return res

def allgather(a, comm):
	"""Convenience wrapper for Allgather that returns the result
	rather than needing an output argument."""
	a   = np.asarray(a)
	res = np.zeros((comm.size,)+a.shape,dtype=a.dtype)
	if np.issubdtype(a.dtype, np.string_):
		comm.Allgather(a.view(dtype=np.uint8), res.view(dtype=np.uint8))
	else:
		comm.Allgather(a, res)
	return res

def allgatherv(a, comm, axis=0):
	"""Perform an mpi allgatherv along the specified axis of the array
	a, returning an array with the individual process arrays concatenated
	along that dimension. For example gatherv([[1,2]],comm) on one task
	and gatherv([[3,4],[5,6]],comm) on another task results in
	[[1,2],[3,4],[5,6]] for both tasks."""
	a  = np.asarray(a)
	fa = moveaxis(a, axis, 0)
	# mpi4py doesn't handle all types. But why not just do this
	# for everything?
	must_fix = np.issubdtype(a.dtype, np.str_) or a.dtype == bool
	if must_fix:
		fa = fa.view(dtype=np.uint8)
	ra = fa.reshape(fa.shape[0],-1) if fa.size > 0 else fa.reshape(0,np.product(fa.shape[1:],dtype=int))
	N  = ra.shape[1]
	n  = allgather([len(ra)],comm)
	o  = cumsum(n)
	rb = np.zeros((np.sum(n),N),dtype=ra.dtype)
	comm.Allgatherv(ra, (rb, (n*N,o*N)))
	fb = rb.reshape((rb.shape[0],)+fa.shape[1:])
	# Restore original data type
	if must_fix:
		fb = fb.view(dtype=a.dtype)
	return moveaxis(fb, 0, axis)

def send(a, comm, dest=0, tag=0):
	"""Faster version of comm.send for numpy arrays.
	Avoids slow pickling. Used with recv below."""
	a = np.asanyarray(a)
	comm.send((a.shape,a.dtype), dest=dest, tag=tag)
	comm.Send(a, dest=dest, tag=tag)

def recv(comm, source=0, tag=0):
	"""Faster version of comm.recv for numpy arrays.
	Avoids slow pickling. Used with send above."""
	shape, dtype = comm.recv(source=source, tag=tag)
	res = np.empty(shape, dtype)
	comm.Recv(res, source=source, tag=tag)
	return res

def tuplify(a):
	try: return tuple(a)
	except TypeError: return (a,)

def resize_array(arr, size, axis=None, val=0):
	"""Return a new array equal to arr but with the given
	axis reshaped to the given sizes. Inserted elements will
	be set to val."""
	arr    = np.asarray(arr)
	size   = tuplify(size)
	axis   = range(len(size)) if axis is None else tuplify(axis)
	axis   = [a%arr.ndim for a in axis]
	oshape = np.array(arr.shape)
	oshape[np.array(axis)] = size
	res    = np.full(oshape, val, arr.dtype)
	slices = tuple([slice(0,min(s1,s2)) for s1,s2 in zip(arr.shape,res.shape)])
	res[slices] = arr[slices]
	return res

# This function does a lot of slice logic, and that actually makes it pretty
# slow when dealing with large numbers of small boxes. It's tempting to move
# the slow parts into fortran... But the way I've set things up there's no
# natural module to put that in. The slowest part is sbox_intersect, which
# deals with variable-length lists, which is also bad for fortran.
def redistribute(iarrs, iboxes, oboxes, comm, wrap=0):
	"""Given the array iarrs[[{pre},{dims}]] which represents slices
	garr[...,narr,ibox[0,0]:ibox[0,1]:ibox[0,2],ibox[1,0]:ibox[1,1]:ibox[1,2],etc]
	of some larger, distributed array garr, returns a different
	slice of the global array given by obox."""
	iarrs  = [np.asanyarray(iarr) for iarr in iarrs]
	iboxes = sbox_fix(iboxes)
	oboxes = sbox_fix(oboxes)
	ndim   = iboxes[0].shape[-2]
	# mpi4py can't handle all ways of expressing the same dtype.
	# this attempts to force the dtype into an equivalent form it can handle
	dtype  = np.dtype(np.dtype(iarrs[0].dtype).char)
	preshape = iarrs[0].shape[:-2]
	oshapes= [tuple(sbox_size(b)) for b in oboxes]
	oarrs  = [np.zeros(preshape+oshape,dtype) for oshape in oshapes]
	presize= np.product(preshape,dtype=int)
	# Find out what we must send to and receive from each other task.
	# rboxes will contain slices into oarr and sboxes into iarr.
	# Due to wrapping, a single pair of boxes can have multiple intersections,
	# so we may need to send multiple arrays to each other task.
	# We handle this by flattening and concatenating into a single buffer.
	# sbox_intersect must return a list of lists of boxes
	niarrs = allgather(len(iboxes), comm)
	nimap  = [i for i,a in enumerate(niarrs) for j in range(a)]
	noarrs = allgather(len(oboxes), comm)
	nomap  = [i for i,a in enumerate(noarrs) for j in range(a)]
	def safe_div(a,b,wrap=0):
		return sbox_div(a,b,wrap=wrap) if len(a) > 0 else [np.array([[0,0,1]]*ndim)]

	# Set up receive buffer
	nrecv = np.zeros(len(niarrs), int)
	all_iboxes = allgatherv(iboxes, comm)
	rboxes     = sbox_intersect(all_iboxes, oboxes, wrap=wrap)
	for i1 in range(rboxes.shape[0]):
		count = 0
		for i2 in range(rboxes.shape[1]):
			rboxes[i1,i2] = safe_div(rboxes[i1,i2], oboxes[i2])
			for box in rboxes[i1,i2]:
				count += np.product(sbox_size(box))
		nrecv[nimap[i1]] += count*presize
	recvbuf = np.empty(np.sum(nrecv), dtype)

	# Set up send buffer
	nsend   = np.zeros(len(noarrs), int)
	sendbuf = []
	all_oboxes = allgatherv(oboxes, comm)
	sboxes     = sbox_intersect(all_oboxes, iboxes, wrap=wrap)
	for i1 in range(sboxes.shape[0]):
		count = 0
		for i2 in range(sboxes.shape[1]):
			sboxes[i1,i2] = safe_div(sboxes[i1,i2], iboxes[i2])
			for box in sboxes[i1,i2]:
				count += np.product(sbox_size(box))
				sendbuf.append(iarrs[i2][sbox2slice(box)].reshape(-1))
		nsend[nomap[i1]] += count*presize
	sendbuf = np.concatenate(sendbuf) if len(sendbuf) > 0 else np.zeros(0,dtype)

	# Perform the actual all-to-all send
	sbufinfo = (nsend,cumsum(nsend))
	rbufinfo = (nrecv,cumsum(nrecv))

	comm.Alltoallv((sendbuf, sbufinfo), (recvbuf,rbufinfo))

	# Copy out the result
	off = 0
	for i1 in range(rboxes.shape[0]):
		for i2 in range(rboxes.shape[1]):
			for rbox in rboxes[i1,i2]:
				rshape = tuple(sbox_size(rbox))
				data   = recvbuf[off:off+np.product(rshape)*presize]
				oarrs[i2][sbox2slice(rbox)] = data.reshape(preshape + rshape)
				off += data.size
	return oarrs

def sbox_intersect(a,b,wrap=0):
	"""Given two Nd sboxes a,b [...,ndim,{start,end,step}] into the
	same array, compute an sbox representing
	their intersection. The resulting sbox will have positive step size.
	The result is a possibly empty list of sboxes - it is empty if there is
	no overlap. If wrap is specified, then it should be a list of length ndim
	of pixel wraps, each of which can be zero to disable wrapping in
	that direction."""
	# First get intersection along each axis
	a = sbox_fix(a)
	b = sbox_fix(b)
	fa = a.reshape((-1,)+a.shape[-2:])
	fb = b.reshape((-1,)+b.shape[-2:])
	ndim = a.shape[-2]
	wrap = np.zeros(ndim,int)+wrap
	# Loop over all combinations
	res = np.empty((fa.shape[0],fb.shape[0]),dtype=np.object)
	for ai, a1 in enumerate(fa):
		for bi, b1 in enumerate(fb):
			peraxis = [sbox_intersect_1d(a1[d],b1[d],wrap=wrap[d]) for d in range(ndim)]
			# Get the outer product of these
			nper    = tuple([len(p) for p in peraxis])
			iflat   = np.arange(np.product(nper))
			ifull   = np.array(np.unravel_index(iflat, nper)).T
			subres  = [[p[i] for i,p in zip(inds,peraxis)] for inds in ifull]
			res[ai,bi] = subres
	res = res.reshape(a.shape[:-2]+b.shape[:-2])
	if res.ndim == 0:
		res = res.reshape(-1)[0]
	return res

def sbox_intersect_1d(a,b,wrap=0):
	"""Given two 1d sboxes into the same array, compute an sbox representing
	their intersecting area. The resulting sbox will have positive step size. The result
	is a list of intersection sboxes. This can be empty if there is no intersection,
	such as between [0,n,2] and [1,n,2]. If wrap is not 0, then it
	should be an integer at which pixels repeat, so i and i+wrap would be
	equivalent. This can lead to more intersections than one would usually get.
	"""
	a = sbox_fix(a)
	b = sbox_fix(b)
	if a[2] < 0: a = sbox_flip(a)
	if b[2] < 0: b = sbox_flip(b)
	segs = [(a,b)]
	if wrap:
		a, b = np.array(a), np.array(b)
		a[:2]  -= a[0]//wrap*wrap
		b[:2]  -= b[0]//wrap*wrap
		segs[0] = (a,b)
		if a[1] > wrap: segs.append((a-[wrap,wrap,0],b))
		if b[1] > wrap: segs.append((a,b-[wrap,wrap,0]))
	res = []
	for a,b in segs:
		if b[0] < a[0]: a,b = b,a
		step  = lcm(abs(a[2]),abs(b[2]))
		# Find the first point in the intersection
		rel_inds = np.arange(b[0]-a[0],b[0]-a[0]+step,b[2])
		match = np.where(rel_inds % a[2] == 0)[0]
		if len(match) == 0: continue
		start = rel_inds[match[0]]+a[0]
		# Find the last point in the intersection
		end   =(min(a[1]-a[2],b[1]-b[2])-start)//step*step+start+step
		if end <= start: continue
		res.append([start,end,step])
	return res

def sbox_div(a,b,wrap=0):
	"""Find c such that arr[a] = arr[b][c]."""
	a = sbox_fix(a)
	b = sbox_fix(b)
	step  = a[...,2]//b[...,2]
	num   = (a[...,1]-a[...,0])//a[...,2]
	start = (a[...,0]-b[...,0])//b[...,2]
	end   = start + step*num
	res   = np.stack([start,end,step],-1)
	if wrap:
		wrap  = np.asarray(wrap,int)[...,None]
		swrap = wrap.copy()
		swrap[wrap==0] = 1
		res[...,:2] -= res[...,0,None]//swrap*wrap
	return res

def sbox_mul(a,b):
	"""Find c such that arr[c] = arr[a][b]"""
	a = sbox_fix(a).copy()
	b = sbox_fix(b).copy()
	# It's easiest to implement this for sboxes in normal ordering.
	# So we will flip a and b as needed, and then flip back the result
	# if necessary. First compute which  result entries must be flipped.
	flip = (a[...,2] < 0)^(b[...,2] < 0)
	# Then flip
	a[a[...,2]<0] = sbox_flip(a[a[...,2]<0])
	b[b[...,2]<0] = sbox_flip(b[b[...,2]<0])
	# Now that everything is in the right order, we combine
	c0 = a[...,0] + b[...,0]*a[...,2]
	c1 = np.minimum(a[...,0] + b[...,1]*a[...,2],a[...,1])
	c2 = a[...,2]*b[...,2]
	res = sbox_fix(np.stack([c0,c1,c2],-1))
	# Flip back where necessary
	res[flip] = sbox_flip(res[flip])
	return res

def sbox_flip(sbox):
	sbox = sbox_fix0(sbox)
	return np.stack([sbox[...,1]-sbox[...,2],sbox[...,0]-sbox[...,2],-sbox[...,2]],-1)

def sbox2slice(sbox):
	sbox = sbox_fix0(sbox)
	return (Ellipsis,)+tuple([slice(s[0],s[1] if s[1]>=0 else None,s[2]) for s in sbox])

def sbox_size(sbox):
	"""Return the size [...,n] of an sbox [...,{start,end,step}].
	The end must be a whole multiple of step away from start, like
	as with the other sbox functions."""
	sbox = sbox_fix0(sbox)
	sbox = sbox*np.sign(sbox[...,2,None])
	return (((sbox[...,1]-sbox[...,0])-1)//sbox[...,2]).astype(int)+1

def sbox_fix0(sbox):
	try: sbox.ndim
	except AttributeError: sbox = np.asarray(sbox)
	if sbox.shape[-1] == 2:
		tmp = np.full(sbox.shape[:-1]+(3,),1,int)
		tmp[...,:2] = sbox
		sbox = tmp
	if sbox.dtype != np.int:
		sbox = sbox.astype(int)
	return sbox

def sbox_fix(sbox):
	# Ensure that we have a step, setting it to 1 if missing
	sbox = sbox_fix0(sbox)
	# Make sure our end point is a whole multiple of the step
	# from the start
	sbox[...,1] = sbox[...,0] + sbox_size(sbox)*sbox[...,2]
	return sbox

def sbox_wrap(sbox, wrap=0, cap=0):
	""""Given a single sbox [...,{from,to,step?}] representing a slice of an N-dim array,
	wraps and caps the sbox, returning a list of sboxes for each
	contiguous section of the slice.

	The wrap argument, which can be scalar or a length N array-like,
	indicates the wrapping length along each dimension. Boxes that
	extend beyond the wrapping length will be split into two at the
	wrapping position, with the overshooting part wrapping around
	to the beginning of the array. The speical value 0 disables wrapping
	for that dimension.

	The cap argument, which can also be a scalar or length N array-like,
	indicates the physical length of each array dimension. The sboxes will
	be truncated to avoid accessing any data beyond this length, after wrapping
	has been taken into account.

	The function returns a list of the form [(ibox1,obox1),(ibox2,obox2)...],
	where the iboxes are sboxes representing slices into the input array
	(the array the original sbox refers to), while the oboxes represent slices
	into the output array. These sboxes can be turned into actual slices using
	sbox2slice.

	A typical example of the use of this function would be a sky map that wraps
	horizontally after 360 degrees, where one wants to support extracting subsets
	that straddle the wrapping point."""
	# This function was surprisingly complicated. If I had known it would be
	# this long I would have built it from the sbox-stuff above. But at least
	# this one should have lower overhead than that would have had.
	sbox = sbox_fix(sbox)
	ndim = sbox.shape[0]
	wrap = np.zeros(ndim,int)+wrap
	cap  = np.zeros(ndim,int)+cap
	dim_boxes = []
	for d, w in enumerate(wrap):
		ibox = sbox[d]
		ilen = sbox_size(ibox)
		c = cap[d]
		# Things will be less redundant if we ensure that a has a positive stride
		flip = ibox[2] < 0
		if flip:
			ibox = sbox_flip(ibox)
		if w:
			# move starting point to first positive loop
			ibox[:2] -= ibox[0]//w*w
			boxes_1d = []
			i = 0
			while ibox[1] > 0:
				npre = max((-ibox[0])//ibox[2],0)
				# ibox slice assuming all of the logical ibox is available
				isub = sbox_fix([ibox[0]+npre*ibox[2],min(ibox[1],w),ibox[2]])
				nsub = sbox_size(isub)
				# the physical array may be smaller than the logical one.
				if c:
					isub = sbox_fix([ibox[0]+npre*ibox[2],min(ibox[1],c),ibox[2]])
					ncap = sbox_size(isub)
				else: ncap = nsub
				if not flip: osub = [i, i+ncap, 1]
				else:        osub = [ilen-1-i, ilen-1-(i+ncap), -1]
				# accept this slice unless it doesn't overlap with anything
				if ncap > 0:
					boxes_1d.append((list(isub),osub))
				i += nsub
				ibox[:2] -= w
		else:
			# No wrapping, but may want to cap. In this case we will have both
			# upwards and downwards capping. Find the number of samples to
			# crop on each side
			if c:
				npre  = max((-ibox[0])//ibox[2],0)
				npost = max((ibox[1]-ibox[2]-(c-1))//ibox[2],0)
			else: npre, npost = 0, 0
			# Don't produce a slice if it would be empty
			if npre + npost < ilen:
				isub = [ibox[0]+npre*ibox[2], ibox[1]-npost*ibox[2], ibox[2]]
				if not flip: osub = [npre, ilen-npost, 1]
				else:        osub = [ilen-1-npre, npost-1, -1]
				boxes_1d = [(isub,osub)]
			else:
				boxes_1d = []
		dim_boxes.append(boxes_1d)
	# Now create the outer product of all the individual dimensions' box sets
	nper    = tuple([len(p) for p in dim_boxes])
	iflat   = np.arange(np.product(nper))
	ifull   = np.array(np.unravel_index(iflat, nper)).T
	res     = [[[p[i][io] for i,p in zip(inds,dim_boxes)] for io in [0,1]] for inds in ifull]
	return res

def gcd(a, b):
	"""Greatest common divisor of a and b"""
	return gcd(b, a % b) if b else a
def lcm(a, b):
	"""Least common multiple of a and b"""
	return a*b//gcd(a,b)

def uncat(a, lens):
	"""Undo a concatenation operation. If a = np.concatenate(b)
	and lens = [len(x) for x in b], then uncat(a,lens) returns
	b."""
	cum = cumsum(lens, endpoint=True)
	return [a[cum[i]:cum[i+1]] for i in xrange(len(lens))]

def ang2rect(angs, zenith=False, axis=0):
	"""Convert a set of angles [{phi,theta},...] to cartesian
	coordinates [{x,y,z},...]. If zenith is True,
	the theta angle will be taken to go from 0 to pi, and measure
	the angle from the z axis. If zenith is False, then theta
	goes from -pi/2 to pi/2, and measures the angle up from the xy plane."""
	angs       = np.asanyarray(angs)
	phi, theta = moveaxis(angs, axis, 0)
	ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
	if zenith: res = np.array([st*cp,st*sp,ct])
	else:      res = np.array([ct*cp,ct*sp,st])
	return moveaxis(res, 0, axis)

def rect2ang(rect, zenith=False, axis=0):
	"""The inverse of ang2rect."""
	x,y,z = moveaxis(rect, axis, 0)
	r     = (x**2+y**2)**0.5
	phi   = np.arctan2(y,x)
	if zenith: theta = np.arctan2(r,z)
	else:      theta = np.arctan2(z,r)
	return moveaxis(np.array([phi,theta]), 0, axis)

def angdist(a, b, zenith=False, axis=0):
	"""Compute the angular distance between a[{ra,dec},...]
	and b[{ra,dec},...] using a Vincenty formula that's stable
	both for small and large angular separations. a and b must
	broadcast correctly."""
	a = moveaxis(np.asarray(a), axis, 0)
	b = moveaxis(np.asarray(b), axis, 0)
	dra = a[0]-b[0]
	sin_dra = np.sin(dra)
	cos_dra = np.cos(dra)
	del dra
	cos, sin = (np.cos,np.sin) if not zenith else (np.sin, np.cos)
	a_sin_dec = sin(a[1])
	a_cos_dec = cos(a[1])
	b_sin_dec = sin(b[1])
	b_cos_dec = cos(b[1])
	y = ((b_cos_dec*sin_dra)**2 + (a_cos_dec*b_sin_dec-a_sin_dec*b_cos_dec*cos_dra)**2)**0.5
	del sin_dra
	x = a_sin_dec*b_sin_dec + a_cos_dec*b_cos_dec*cos_dra
	del a_sin_dec, a_cos_dec, b_sin_dec, b_cos_dec, a, b
	return np.arctan2(y,x)

def vec_angdist(v1, v2, axis=0):
	"""Use Kahan's version of Heron's formula to compute a stable angular
	distance between to vectors v1 and v2, which don't have to be unit vectors.
	See https://scicomp.stackexchange.com/a/27694"""
	v1 = np.asanyarray(v1)
	v2 = np.asanyarray(v2)
	a  = np.sum(v1**2,axis)**0.5
	b  = np.sum(v2**2,axis)**0.5
	c  = np.sum((v1-v2)**2,axis)**0.5
	mu = np.where(b>=c, c-(a-b), b-(a-c))
	ang= 2*np.arctan(((((a-b)+c)*mu)/((a+(b+c))*((a-c)+b)))**0.5)
	return ang

def rotmatrix(ang, raxis, axis=0):
	"""Construct a 3d rotation matrix representing a rotation of
	ang degrees around the specified rotation axis raxis, which can be "x", "y", "z"
	or 0, 1, 2. If ang is a scalar, the result will be [3,3]. Otherwise,
	it will be ang.shape + (3,3)."""
	ang  = np.asarray(ang)
	raxis = raxis.lower()
	c, s = np.cos(ang), np.sin(ang)
	R = np.zeros(ang.shape + (3,3))
	if   raxis == 0 or raxis == "x": R[...,0,0]=1;R[...,1,1]= c;R[...,1,2]=-s;R[...,2,1]= s;R[...,2,2]=c
	elif raxis == 1 or raxis == "y": R[...,0,0]=c;R[...,0,2]= s;R[...,1,1]= 1;R[...,2,0]=-s;R[...,2,2]=c
	elif raxis == 2 or raxis == "z": R[...,0,0]=c;R[...,0,1]=-s;R[...,1,0]= s;R[...,1,1]= c;R[...,2,2]=1
	else: raise ValueError("Rotation axis %s not recognized" % raxis)
	return moveaxis(R, 0, axis)

def label_unique(a, axes=(), rtol=1e-5, atol=1e-8):
	"""Given an array of values, return an array of
	labels such that all entries in the array with the
	same label will have approximately the same value.
	Labels count contiguously from 0 and up.
	axes specifies which axes make up the subarray that
	should be compared for equality. For scalars,
	use axes=()."""
	a = np.asarray(a)
	axes = [i % a.ndim for i in axes]
	rest = [s for i,s in enumerate(a.shape) if i not in axes]

	# First reshape into a doubly-flattened 2d array [nelem,ndim]
	fa = partial_flatten(a, axes, 0)
	fa = fa.reshape(np.product(rest),-1)
	# Can't use lexsort, as it has no tolerance. This
	# is O(N^2) instead of O(NlogN)
	id = 0
	ids = np.zeros(len(fa),dtype=int)-1
	for i,v in enumerate(fa):
		if ids[i] >= 0: continue
		match = np.all(np.isclose(v,fa,rtol=rtol,atol=atol),-1)
		ids[match] = id
		id += 1
	return ids.reshape(rest)

def transpose_inds(inds, nrow, ncol):
	"""Given a set of flattened indices into an array of shape (nrow,ncol),
	return the indices of the corresponding elemens in a transposed array."""
	row_major = inds
	row, col = row_major//ncol, row_major%ncol
	return col*nrow + row

def rescale(a, range=[0,1]):
	"""Rescale a such that min(a),max(a) -> range[0],range[1]"""
	mi, ma = np.min(a), np.max(a)
	return (a-mi)/(ma-mi)*(range[1]-range[0])+range[0]

def split_by_group(a, start, end):
	"""Split string a into non-group and group sections,
	where a group is defined as a set of characters from
	a start character to a corresponding end character."""
	res, ind, n = [], 0, 0
	new = True
	for c in a:
		if new:
			res.append("")
			new = False
		i = start.find(c)
		if n == 0:
			if i >= 0:
				# Start of new group
				res.append("")
				ind = i
				n += 1
		else:
			if start[ind] == c:
				n += 1
			elif end[ind] == c:
				n-= 1
				if n == 0: new = True
		res[-1] += c
	return res

def split_outside(a, sep, start="([{", end=")]}"):
	"""Split string a at occurences of separator sep, except when
	it occurs inside matching groups of start and end characters."""
	segments = split_by_group(a, start, end)
	res = [""]
	for seg in segments:
		if len(seg) == 0: continue
		if seg[0] in start:
			res[-1] += seg
		else:
			toks = seg.split(sep)
			res[-1] += toks[0]
			res += toks[1:]
	return res

def find_equal_groups(a, tol=0):
	"""Given a[nsamp,ndim], return groups[ngroup][{ind,ind,ind,...}]
	of indices into a for which all the values in the second index
	of a is the same. find_equal_groups([[0,1],[1,2],[0,1]]) -> [[0,2],[1]]."""
	def calc_diff(a1,a2):
		if a1.dtype.char in 'SU': return a1 != a2
		else: return a1-a2
	a = np.asarray(a)
	if a.ndim == 1: a = a[:,None]
	n = len(a)
	inds = np.argsort(a[:,0])
	done = np.full(n, False, dtype=bool)
	res = []
	for i in xrange(n):
		if done[i]: continue
		xi = inds[i]
		res.append([xi])
		done[i] = True
		for j in xrange(i+1,n):
			if done[j]: continue
			xj = inds[j]
			if calc_diff(a[xj,0], a[xi,0]) > tol:
				# Current group is done
				break
			if np.sum(calc_diff(a[xj],a[xi])**2) <= tol**2:
				# Close enough
				res[-1].append(xj)
				done[j] = True
	return res

def minmax(a, axis=None):
	"""Shortcut for np.array([np.min(a),np.max(a)]), since I do this
	a lot."""
	return np.array([np.min(a, axis=axis),np.max(a, axis=axis)])

def point_in_polygon(points, polys):
	"""Given a points[...,2] and a set of polys[...,nvertex,2], return
	inside[...]. points[...,0] and polys[...,0,0] must broadcast correctly.

	Examples:
	utils.point_in_polygon([0.5,0.5],[[0,0],[0,1],[1,1],[1,0]]) -> True
	utils.point_in_polygon([[0.5,0.5],[2,1]],[[0,0],[0,1],[1,1],[1,0]]) -> [True, False]
	"""
	# Make sure we have arrays, and that they have a floating point data type
	points = np.asarray(points)+0.0
	polys  = np.asarray(polys) +0.0
	verts  = polys - points[...,None,:]
	ncross = np.zeros(verts.shape[:-2], dtype=np.int32)
	# For each vertex, check if it crosses y=0 by computing the x
	# position of that crossing, and seeing if that x is within the
	# poly's bounds.
	for i in range(verts.shape[-2]):
		x1, y1 = verts[...,i-1,:].T
		x2, y2 = verts[...,i,:].T
		with nowarn():
			x = -y1*(x2-x1)/(y2-y1) + x1
		ncross += ((y1*y2 < 0) & (x > 0)).T
	return ncross.T % 2 == 1

def poly_edge_dist(points, polygons):
	"""Given points [...,2] and a set of polygons [...,nvertex,2], return
	dists[...], which represents the distance of the points from the edges
	of the corresponding polygons. This means that the interior of the
	polygon will not be 0. points[...,0] and polys[...,0,0] must broadcast
	correctly."""
	points   = np.asarray(points)
	polygons = np.asarray(polygons)
	nvert    = polygons.shape[-2]
	p        = ang2rect(points,axis=-1)
	vertices = ang2rect(polygons,axis=-1)
	del points, polygons
	dists = []
	for i in range(nvert):
		v1   = vertices[...,i,:]
		v2   = vertices[...,(i+1)%nvert,:]
		vz   = np.cross(v1,v2)
		vz  /= np.sum(vz**2,-1)[...,None]**0.5
		# Find out if the point is inside our line segment or not
		vx   = v1
		vy   = np.cross(vz,vx)
		vy  /= np.sum(vy**2,-1)[...,None]**0.5
		pang = np.arctan2(np.sum( p*vy,-1),np.sum( p*vx,-1))
		ang2 = np.arctan2(np.sum(v2*vy,-1),np.sum(v2*vx,-1))
		between = (pang >= 0) & (pang < ang2)
		# If we are inside, the distance will simply be the distance
		# from the line segment, which is the distance from vz minus pi/2.
		# If we are outside, then use the distance from the edge of the line segment.
		dist_between = np.abs(np.arccos(np.clip(np.sum(p*vz,-1),-1,1))-np.pi/2)
		dist_outside = np.minimum(
			np.arccos(np.clip(np.sum(p*v1,-1),-1,1)),
			np.arccos(np.clip(np.sum(p*v2,-1),-1,1))
		)
		dist = np.where(between, dist_between, dist_outside)
		del v1, v2, vx, vy, vz, pang, ang2, between
		dists.append(dist)
	dists = np.min(dists,0)
	return dists

def block_mean_filter(a, width):
	"""Perform a binwise smoothing of a, where all samples
	in each bin of the given width are replaced by the mean
	of the samples in that bin."""
	a = np.array(a)
	if a.shape[-1] < width:
		a[:] = np.mean(a,-1)[...,None]
	else:
		width  = int(width)
		nblock = (a.shape[-1]+width-1)//width
		apad   = np.concatenate([a,a[...,-2::-1]],-1)
		work   = apad[...,:width*nblock]
		work   = work.reshape(work.shape[:-1]+(nblock,width))
		work[:]= np.mean(work,-1)[...,None]
		work   = work.reshape(work.shape[:-2]+(-1,))
		a[:]   = work[...,:a.shape[-1]]
	return a

def ctime2date(timestamp, tzone=0, fmt="%Y-%m-%d"):
	return datetime.datetime.utcfromtimestamp(timestamp+tzone*3600).strftime(fmt)

def tofinite(arr, val=0):
	"""Return arr with all non-finite values replaced with val."""
	arr = np.asanyarray(arr).copy()
	if arr.ndim == 0:
		if ~np.isfinite(arr): arr = val
	else:
		arr[~np.isfinite(arr)] = val
	return arr

def parse_ints(s): return parse_numbers(s, int)
def parse_floats(s): return parse_numbers(s, float)
def parse_numbers(s, dtype=None):
	res = []
	for word in s.split(","):
		toks = [float(w) for w in word.split(":")]
		if ":" not in word:
			res.append(toks[:1])
		else:
			start, stop = toks[:2]
			step = toks[2] if len(toks) > 2 else 1
			res.append(np.arange(start,stop,step))
	res = np.concatenate(res)
	if dtype is not None:
		res = res.astype(dtype)
	return res

def triangle_wave(x, period=1):
	"""Return a triangle wave with amplitude 1 and the given period."""
	# This order (rather than x/period%1) gave smaller errors
	x = x % period / period * 4
	m1 = x < 1
	m2 = (x < 3) ^ m1
	m3 = x >= 3
	res = x.copy()
	res[m1] = x[m1]
	res[m2] = 2-x[m2]
	res[m3] = x[m3]-4
	return res

def flux_factor(beam_area, freq, T0=T_cmb):
	"""Compute the factor A that when multiplied with a linearized
	temperature increment dT around T0 (in K) at the given frequency freq
	in Hz and integrated over the given beam_area in steradians, produces
	the corresponding flux = A*dT. This is useful for converting between
	point source amplitudes and point source fluxes.

	For uK to mJy use flux_factor(beam_area, freq)/1e3
	"""
	# A blackbody has intensity I = 2hf**3/c**2/(exp(hf/kT)-1) = V/(exp(x)-1)
	# with V = 2hf**3/c**2, x = hf/kT.
	# dI/dx = -V/(exp(x)-1)**2 * exp(x)
	# dI/dT = dI/dx * dx/dT
	#       = 2hf**3/c**2/(exp(x)-1)**2*exp(x) * hf/k / T**2
	#       = 2*h**2*f**4/c**2/k/T**2 * exp(x)/(exp(x)-1)**2
	#       = 2*x**4 * k**3*T**2/(h**2*c**2) * exp(x)/(exp(x)-1)**2
	#       = .... /(4*sinh(x/2)**2)
	x     = h*freq/(k*T0)
	dIdT  = 2*x**4 * k**3*T0**2/(h**2*c**2) / (4*np.sinh(x/2)**2)
	dJydK = dIdT * 1e26 * beam_area
	return dJydK

def noise_flux_factor(beam_area, freq, T0=T_cmb):
	"""Compute the factor A that converts from white noise level in K sqrt(steradian)
	to uncertainty in Jy for the given beam area in steradians and frequency in Hz.
	This assumes white noise and a gaussian beam, so that the area of the real-space squared beam is
	just half that of the normal beam area.

	For uK arcmin to mJy, use noise_flux_factor(beam_area, freq)*arcmin/1e3
	"""
	squared_beam_area = beam_area/2
	return flux_factor(beam_area/squared_beam_area**0.5, freq, T0=T0)

def planck(f, T):
	"""Return the Planck spectrum at the frequency f and temperature T in Jy/sr"""
	return 2*h*f**3/c**2/(np.exp(h*f/(k*T))-1) * 1e26
blackbody = planck

def graybody(f, T, beta=1):
	"""Return a graybody spectrum at the frequency f and temperature T in Jy/sr"""
	return  2*h*f**(3+beta)/c**2/(np.exp(h*f/(k*T))-1) * 1e26

def tsz_spectrum(f, T=T_cmb):
	"""The increase in flux due to tsz in Jy/sr per unit of y. This is
	just the first order approximation, but it's good enough for realistic
	values of y, i.e. y << 1"""
	x  = h*f/(k*T)
	ex = np.exp(x)
	return 2*h*f**3/c**2 * (x*ex)/(ex-1)**2 * (x*(ex+1)/(ex-1)-4) * 1e26

### Binning ####

def edges2bins(edges):
	res = np.zeros((edges.size-1,2),int)
	res[:,0] = edges[:-1]
	res[:,1] = edges[1:]
	return res

def bins2edges(bins):
	return np.concatenate([bins[:,0],bins[1,-1:]])

def linbin(n, nbin=None, nmin=None):
	"""Given a number of points to bin and the number of approximately
	equal-sized bins to generate, returns [nbin_out,{from,to}].
	nbin_out may be smaller than nbin. The nmin argument specifies
	the minimum number of points per bin, but it is not implemented yet.
	nbin defaults to the square root of n if not specified."""
	if not nbin: nbin = int(np.round(n**0.5))
	tmp  = np.arange(nbin+1)*n//nbin
	return np.vstack((tmp[:-1],tmp[1:])).T

def expbin(n, nbin=None, nmin=8, nmax=0):
	"""Given a number of points to bin and the number of exponentially spaced
	bins to generate, returns [nbin_out,{from,to}].
	nbin_out may be smaller than nbin. The nmin argument specifies
	the minimum number of points per bin. nbin defaults to n**0.5"""
	if not nbin: nbin = int(np.round(n**0.5))
	tmp  = np.array(np.exp(np.arange(nbin+1)*np.log(n+1)/nbin)-1,dtype=int)
	fixed = [tmp[0]]
	i = 0
	while i < nbin:
		for j in range(i+1,nbin+1):
			if tmp[j]-tmp[i] >= nmin:
				fixed.append(tmp[j])
				i = j
	# Optionally split too large bins
	if nmax:
		tmp = [fixed[0]]
		for v in fixed[1:]:
			dv = v-tmp[-1]
			nsplit = (dv+nmax-1)//nmax
			tmp += [tmp[-1]+dv*(i+1)//nsplit for i in range(nsplit)]
		fixed = tmp
	tmp = np.array(fixed)
	tmp[-1] = n
	return np.vstack((tmp[:-1],tmp[1:])).T

def bin_data(bins, d, op=np.mean):
	"""Bin the data d into the specified bins along the last dimension. The result has
	shape d.shape[:-1] + (nbin,)."""
	nbin  = bins.shape[0]
	dflat = d.reshape(-1,d.shape[-1])
	dbin  = np.zeros([dflat.shape[0], nbin])
	for bi, b in enumerate(bins):
		dbin[:,bi] = op(dflat[:,b[0]:b[1]],1)
	return dbin.reshape(d.shape[:-1]+(nbin,))

def bin_expand(bins, bdata):
	res = np.zeros(bdata.shape[:-1]+(bins[-1,1],),bdata.dtype)
	for bi, b in enumerate(bins):
		res[...,b[0]:b[1]] = bdata[...,bi]
	return res

def is_int_valued(a): return a%1 == 0

#### Matrix operations that don't need fortran ####

# Don't do matmul - it's better expressed with einsum

def solve(A, b, axes=[-2,-1], masked=False):
	"""Solve the linear system Ax=b along the specified axes
	for A, and axes[0] for b. If masked is True, then entries
	where A00 along the given axes is zero will be skipped."""
	A,b = np.asanyarray(A), np.asanyarray(b)
	baxes = axes if A.ndim == b.ndim else [axes[0]%A.ndim]
	fA = partial_flatten(A, axes)
	fb = partial_flatten(b, baxes)
	if masked:
		mask = fA[...,0,0] != 0
		fb[~mask] = 0
		fb[mask]  = np.linalg.solve(fA[mask],fb[mask])
	else:
		fb = np.linalg.solve(fA,fb)
	return partial_expand(fb, b.shape, baxes)

def eigpow(A, e, axes=[-2,-1], rlim=None, alim=None):
	"""Compute the e'th power of the matrix A (or the last
	two axes of A for higher-dimensional A) by exponentiating
	the eigenvalues. A should be real and symmetric.

	When e is not a positive integer, negative eigenvalues
	could result in a complex result. To avoid this, negative
	eigenvalues are set to zero in this case.

	Also, when e is not positive, tiny eigenvalues dominated by
	numerical errors can be blown up enough to drown out the
	well-measured ones. To avoid this, eigenvalues
	smaller than 1e-13 for float64 or 1e-4 for float32 of the
	largest one (rlim), or with an absolute value less than 2e-304 for float64 or
	1e-34 for float32 (alim) are set to zero for negative e. Set alim
	and rlim to 0 to disable this behavior.
	"""
	# This function basically does
	# E,V = np.linalg.eigh(A)
	# E **= e
	# return (V*E).dot(V.T)
	# All the complicated stuff is there to support axes and tolerances.
	if axes[0]%A.ndim != A.ndim-2 or axes[1]%A.ndim != A.ndim-1:
		fa = partial_flatten(A, axes)
		fa = eigpow(fa, e, rlim=rlim, alim=alim)
		return partial_expand(fa, A.shape, axes)
	else:
		E, V = np.linalg.eigh(A)
		if rlim is None: rlim = np.finfo(E.dtype).resolution*100
		if alim is None: alim = np.finfo(E.dtype).tiny*1e4
		mask = np.full(E.shape, False, np.bool)
		if not is_int_valued(e):
			mask |= E < 0
		if e < 0:
			aE = np.abs(E)
			if A.ndim > 2:
				mask |= (aE < np.max(aE,1)[:,None]*rlim) | (aE < alim)
			else:
				mask |= (aE < np.max(aE)*rlim) | (aE < alim)
		E[~mask] **= e
		E[mask]    = 0
		if A.ndim > 2:
			res = np.einsum("...ij,...kj->...ik",V*E[...,None,:],V)
		else:
			res = V.dot(E[:,None]*V.T)
		return res

def build_conditional(ps, inds, axes=[0,1]):
	"""Given some covariance matrix ps[n,n] describing a
	set of n Gaussian distributed variables, and a set of
	indices inds[m] specifying which of these variables are already
	known, return matrices A[n-m,m], cov[m,m] such that the
	conditional distribution for the unknown variables is
	x_unknown ~ normal(A x_known, cov). If ps has more than
	2 dimensions, then the axes argument indicates which
	dimensions contain the matrix.

	Example:

	C  = np.array([[10,2,1],[2,8,1],[1,1,5]])
	vknown = np.linalg.cholesky(C[:1,:1]).dot(np.random.standard_normal(1))
	A, cov = lensing.build_conditional(C, v0)
	vrest  = A.dot(vknown) + np.linalg.cholesky(cov).dot(np.random_standard_normal(2))

	vtot = np.concatenate([vknown,vrest]) should have the same distribution
	as a sample drawn directly from the full C.
	"""
	ps   = np.asarray(ps)
	# Make the matrix parts the last axes, so we have [:,ncomp,ncomp]
	C = partial_flatten(ps,   axes)
	# Define masks for what is known and unknown
	known       = np.full(C.shape[1],False,bool)
	known[inds] = True
	unknown     = ~known
	# Hack: Handle masked arrays where some elements are all-zero.
	def inv(A):
		good = ~np.all(np.einsum("aii->ai", A)==0,-1)
		res  = A*0
		res[good] = np.linalg.inv(A[good])
		return res
	# Build the parameters for the conditional distribution
	Ci     = inv(C)
	Ciuk   = Ci[:,unknown,:][:,:,known]
	Ciuu   = Ci[:,unknown,:][:,:,unknown]
	Ciuui  = inv(Ciuu)
	A      = -np.matmul(Ciuui, Ciuk)
	# Expand back to original shape
	A      = partial_expand(A,     ps.shape, axes)
	cov    = partial_expand(Ciuui, ps.shape, axes)
	return A, cov

def nint(a):
	"""Return a rounded to the nearest integer, as an integer."""
	return np.round(a).astype(int)

format_regex = r"%(\([a-zA-Z]\w*\)|\(\d+)\)?([ +0#-]*)(\d*|\*)(\.\d+|\.\*)?(ll|[lhqL])?(.)"
def format_to_glob(format):
	"""Given a printf format, construct a glob pattern that will match
	its outputs. However, since globs are not very powerful, the resulting
	glob will be much more premissive than the input format, and you will
	probably want to filter the results further."""
	# This matches a pretty general printf format
	def subfun(m):
		name, flags, width, prec, size, type = m.groups()
		if type == '%': return '%'
		else: return '*'
	return re.sub(format_regex, subfun, format)

def format_to_regex(format):
	"""Given a printf format, construct a regex that will match its outputs."""
	ireg = r"([^%]*)"+format_regex+r"([^%]*)"
	def subfun(m):
		pre, name, flags, width, prec, size, type, post = m.groups()
		opre  = re.escape(pre)
		opost = re.escape(post)
		open  = r"(?P<"+name[1:-1]+">" if name is not None else "("
		# Expand variable widths
		iwidth = 0 if width is None or width == '*' or width == '' else int(width)
		iprec  = 0 if prec  is None or prec  == '*' else int(prec[1:])
		if type == '%': return opre + '%' + opost
		if type == 's':
			if "-" in flags: return opre + open + ".*) *" + opost
			else:            return opre + r" *" + open + ".*)" + opost
		else:
			# Numeric type
			if   "+" in flags: omid = r"[+-]"
			elif " " in flags: omid = r"[ -]"
			else: omid = r"-?"
			if "-" in flags:
				prepad  = ""
				postpad = " *"
			else:
				prepad  = r"0*" if "0" in flags else r" *"
				postpad = ""
			if type in ['d','i','u'] or type in ['f','F'] and prec == '0':
				num = r"\d+"
			elif type == 'o': num = r"[0-7]+"
			elif type == 'x': num = r"[0-9a-f]+"
			elif type == 'X': num = r"[0-9A-F]+"
			elif type == 'f': num = r"\d+\.\d*"
			elif type == 'e': num = r"\d+\.\d*e[+-]\d+"
			elif type == 'E': num = r"\d+\.\d*E[+-]\d+"
			elif type == 'g': num = r"(\d+(\.\d*)?)|(\d+\.\d*e[+-]\d+)"
			elif type == 'G': num = r"(\d+(\.\d*)?)|(\d+\.\d*E[+-]\d+)"
			else: return NotImplementedError("Format character '%s'" % type)
			omid = prepad + open + omid + num + r")" + postpad
			return opre + omid + opost
	return re.sub(ireg, subfun, format)

# Not sure if this belongs here...
class Printer:
	def __init__(self, level=1, prefix=""):
		self.level  = level
		self.prefix = prefix
	def write(self, desc, level, exact=False, newline=True, prepend=""):
		if level == self.level or not exact and level <= self.level:
			sys.stderr.write(prepend + self.prefix + desc + ("\n" if newline else ""))
	def push(self, desc):
		return Printer(self.level, self.prefix + desc)
	def time(self, desc, level, exact=False, newline=True):
		class PrintTimer:
			def __init__(self, printer): self.printer = printer
			def __enter__(self): self.time = time.time()
			def __exit__(self, type, value, traceback):
				self.printer.write(desc, level, exact=exact, newline=newline, prepend="%6.2f " % (time.time()-self.time))
		return PrintTimer(self)

def ndigit(num):
	"""Returns the number of digits in non-negative number num"""
	with nowarn(): return np.int32(np.floor(np.maximum(1,np.log10(num))))+1

def contains_any(a, bs):
	"""Returns true if any of the strings in list bs are found
	in the string a"""
	for b in bs:
		if b in a: return True
	return False

def build_legendre(x, nmax):
	x   = np.asarray(x)
	vmin, vmax = minmax(x)
	x   = (x-vmin)*(2.0/(vmax-vmin))-1
	res = np.zeros((nmax,)+x.shape)
	if nmax > 0: res[0] = 1
	if nmax > 1: res[1] = x
	for i in range(1, nmax-1):
		res[i+1] = ((2*i+1)*x*res[i] - i*res[i-1])/(i+1)
	return res

def build_cossin(x, nmax):
	x   = np.asarray(x)
	res = np.zeros((nmax,)+x.shape, x.dtype)
	if nmax > 0: res[0] = np.sin(x)
	if nmax > 1: res[1] = np.cos(x)
	if nmax > 2: res[2] = 2*res[0]*res[1]
	if nmax > 3: res[3] = res[1]**2-res[0]**2
	for i in range(3,nmax):
		if i % 2 == 0: res[i] = res[i-2]*res[1] + res[i-1]*res[0]
		if i % 2 == 1: res[i] = res[i-2]*res[1] - res[i-3]*res[0]
	return res

def load_ascii_table(fname, desc, sep=None, dsep=None):
	"""Load an ascii table with heterogeneous columns.
	fname: Path to file
	desc: whitespace-separated list of name:typechar pairs, or | for columns that are to be ignored.
	desc must cover every column present in the file"""
	dtype = []
	j = 0
	for i, tok in enumerate(desc.split(dsep)):
		if ":" not in tok:
			j += 1
			dtype.append(("sep%d"%j,"U%d"%len(tok)))
		else:
			name, typ = tok.split(":")
			dtype.append((name,typ))
	return np.loadtxt(fname, dtype=dtype, delimiter=sep)

def count_variable_basis(bases):
	"""Counts from 0 and up through a variable-basis number,
	where each digit has a different basis. For example,
	count_variable_basis([2,3]) would yield [0,0], [0,1], [0,2],
	[1,0], [1,1], [1,2]."""
	N = bases
	n = len(bases)
	I = [0 for i in range(n)]
	yield I
	while True:
		for i in range(n-1,-1,-1):
			I[i] += 1
			if I[i] < N[i]: break
			else:
				for j in range(i,n): I[j] = 0
		else: break
		yield I

def list_combination_iter(ilist):
	"""Given a list of lists of values, yields every combination of
	one value from each list."""
	for I in count_variable_basis([len(v) for v in ilist]):
		yield [v[i] for v,i in zip(ilist,I)]

class _SliceEval:
	def __getitem__(self, sel):
		if isinstance(sel, slice): return (sel,)
		else: return sel
sliceeval = _SliceEval()

def expand_slice(sel, n, nowrap=False):
	"""Expands defaults and negatives in a slice to their implied values.
	After this, all entries of the slice are guaranteed to be present in their final form.
	Note, doing this twice may result in odd results, so don't send the result of this
	into functions that expect an unexpanded slice. Might be replacable with slice.indices()."""
	step = sel.step or 1
	def cycle(i,n):
		if nowrap: return i
		else: return min(i,n) if i >= 0 else n+i
	def default(a, val): return a if a is not None else val
	if step == 0: raise ValueError("slice step cannot be zero")
	if step > 0: return slice(cycle(default(sel.start,0),n),cycle(default(sel.stop,n),n),step)
	else: return slice(cycle(default(sel.start,n-1), n), cycle(sel.stop,n) if sel.stop is not None else -1, step)

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
	a2 = a[:len(a)//step*step]
	a2 = np.mean(a2.reshape((len(a2)//step,step)+a2.shape[1:]),1)
	# Then append the incomplete block
	if len(a2)*step != len(a):
		rest = a[len(a2)*step:]
		a2 = np.concatenate([a2,[np.mean(rest,0)]],0)
	return moveaxis(a2, 0, axis)

def outer_stack(arrays):
	"""Example. outer_stack([[1,2,3],[10,20]]) -> [[[1,1],[2,2],[3,3]],[[10,20],[10,20],[10,2]]]"""
	res = np.empty([len(arrays)]+[len(a) for a in arrays], arrays[0].dtype)
	for i, array in enumerate(arrays):
		res[i] = array[(None,)*i + (slice(None),) + (None,)*(len(arrays)-i-1)]
	return res

def beam_transform_to_profile(bl, theta, normalize=False):
	"""Given the transform b(l) of a beam, evaluate its real space angular profile
	at the given radii theta."""
	bl = np.asarray(bl)
	l  = np.arange(bl.size)
	x  = np.cos(theta)
	a  = bl*(2*l+1)/(4*np.pi)
	profile = np.polynomial.legendre.legval(x,a)
	if normalize: profile /= np.sum(a)
	return profile

def fix_dtype_mpi4py(dtype):
	"""Work around mpi4py bug, where it refuses to accept dtypes with endian info"""
	return np.dtype(np.dtype(dtype).char)

def decode_array_if_necessary(arr):
	"""Given an arbitrary numpy array arr, decode it if it is of type S and we're in a
	version of python that doesn't like that"""
	try:
		# Check if we're in python 3
		np.array(["a"],"S")[0] in "a"
		return arr
	except TypeError:
		# Yes, we need to decode
		if arr.dtype.type is np.bytes_:
			return np.char.decode(arr)
		else:
			return arr

def encode_array_if_necessary(arr):
	"""Given an arbitrary numpy array arr, encode it if it is of type S and we're in a
	version of python that doesn't like that"""
	arr = np.asarray(arr)
	try:
		# Check if we're in python 3
		np.array(["a"],"S")[0] in "a"
		return arr
	except TypeError:
		# Yes, we need to encode
		if arr.dtype.type is np.str_:
			return np.char.encode(arr)
		else:
			return arr
