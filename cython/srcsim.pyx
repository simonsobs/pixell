import cython, numpy as np
cimport numpy as np
from libc.stdlib cimport free, calloc
# I had to call this distances_wrap instead of just distances because
# .pdx files with the same name get automatically imported into the current
# namespace, which means I would have to give stupid names to the functions here
cimport srcsim_wrap as c

__version__ = 1.0

def sim_objects(float[:,:,::1] map, float[::1] obj_decs, float[::1] obj_ras, int[::1] obj_ys, int[::1] obj_xs, float[:,::1] amps, profs, int[::1] prof_ids, posmap, float vmin, float rmax=0, separable=False, transpose=False, prof_equi=False, op="add", int csize=8, return_times=False):
	"""
	map: array[ncomp,ny,nx]. Caller should make sure it has exactly 3 dims
	poss:   [{ra,dec},nobj] float
	pixs:   [{x,y},nobj] float
	amps:   [ncomp,nobj] float
	profs:  [nprof][{r,val}] float, can have different length for each profile
	prof_ids: [nprof] int
	posmap: [dec[ny],ra[nx]] if separable, otherwise [{dec,ra}][ny,nx]
	vmin:   The lowest value to evaluate profiles to. If this is set too low combined
	        with profiles that never reach zero, then things will get very slow.

	op:     How to combine contributions from multiple simulated objects and the input map.
	        add: Things add together linearly
	        max: Each pix will be the max of all contributions to that pixel
	        min: Each pix will be the min of all contributions to that pixel
	csize:   Size of cells used internally when determining which parts of the sky to
	         consider for each source, in pixels.
	
	Returns the resulting map. If inplace, then this will be the same object as the
	map that was passed in, otherwise it will be a new map.
	"""
	# Prepare our inputs. First the map
	cdef int ncomp = map.shape[0]
	if ncomp == 0: return map # nothing to do
	cdef int ny = map.shape[1]
	cdef int nx = map.shape[2]
	# I can't figure out how to make a memoryview of an array of pointers, so let's just do
	# things brute force
	cdef float ** map_ = <float**>calloc(ncomp, sizeof(float*));
	for i in range(ncomp):
		map_[i] = &map[i,0,0]
	# Then the positions
	cdef int nobj = len(obj_ras)
	assert len(obj_decs) == len(obj_xs) == len(obj_ys) == nobj
	# and amplitudes
	assert amps.shape[0] == ncomp and amps.shape[1] == nobj, "amps [%d,%d] must be [ncomp=%d,nobj=%d]" % (amps.shape[0], amps.shape[1], ncomp, nobj)
	cdef float ** amps_ = <float**>calloc(ncomp, sizeof(float*))
	for i in range(ncomp):
		amps_[i] = &amps[i,0]
	# The profiles are variable-length
	cdef int nprof = len(profs)
	cdef int[::1] prof_ns = np.zeros(nprof,dtype=np.int32)
	cdef float ** prof_rs = <float**>calloc(nprof, sizeof(float*))
	cdef float ** prof_vs = <float**>calloc(nprof, sizeof(float*))
	cdef float[:,::1] prof
	for i in range(nprof):
		prof = np.asanyarray(profs[i], dtype=np.float32, order="C")
		assert prof.ndim == 2 and len(prof) == 2, ("prof must be [{r,val},:], but #%d is not" % i)
		prof_ns[i] = prof.shape[1]
		prof_rs[i] = &prof[0,0]
		prof_vs[i] = &prof[1,0]
	cdef int prof_equi_ = prof_equi
	# Profile ids
	assert len(prof_ids) == nobj, "prof_ids (%d) must be [nobj=%d]" % (len(prof_ids), nobj)
	# pixel positions
	cdef int separable_ = separable
	assert len(posmap) == 2, "posmap must be [dec[ny],ra[nx]] if separable, else [{dec,ra},ny,nx]"
	pix_decs = np.asanyarray(posmap[0], dtype=np.float32, order="C")
	pix_ras  = np.asanyarray(posmap[1], dtype=np.float32, order="C")
	if separable:
		assert pix_decs.shape == (ny,) and pix_ras.shape == (nx,), "posmap ([%d],[%d]) must be [dec[ny=%d],ra[nx=%d]] if separable" % (len(pix_decs), len(pix_ras), ny, nx)
	else:
		assert pix_decs.shape == (ny,nx) and pix_ras.shape == (ny,nx), "posmap must be [{dec,ra},ny,nx] if not separable"
	cdef float [::1] pix_decs_ = pix_decs.reshape(-1)
	cdef float [::1] pix_ras_  = pix_ras.reshape(-1)
	# The rest are simple
	cdef int op_ = {"add":0, "max":1, "min":2}[op]
	times = np.zeros(3)
	cdef double [::1] times_ = times
	# Phew! That's a lot of wrapping!
	c.sim_objects(nobj, &obj_decs[0], &obj_ras[0], &obj_ys[0], &obj_xs[0], &amps_[0], nprof, &prof_ids[0], &prof_ns[0], &prof_rs[0], &prof_vs[0], prof_equi, vmin, rmax, op_, ncomp, ny, nx, separable, transpose, &pix_decs_[0], &pix_ras_[0], &map_[0], &map_[0], csize, &times_[0])
	free(map_)
	free(amps_)
	free(prof_rs)
	free(prof_vs)
	if return_times: return map, times
	else:            return map

def radial_sum(float[:,:,::1] map, float[::1] obj_decs, float[::1] obj_ras, int[::1] obj_ys, int[::1] obj_xs, float[::1] rs, posmap, profs=None, separable=False, prof_equi=False, return_times=False):
	"""
	map: array[ncomp,ny,nx]. Caller should make sure it has exactly 3 dims
	poss:   [{ra,dec},nobj] float
	pixs:   [{x,y},nobj] float
	rs:     [nbin+1] float, bin edges (ascending). Faster if equi-spaced starting at 0
	posmap: [dec[ny],ra[nx]] if separable, otherwise [{dec,ra}][ny,nx]

	Returns the resulting profiles. If inplace, then this will be the same object as the
	map that was passed in, otherwise it will be a new map.
	"""
	# Prepare our inputs. First the map
	cdef int ncomp = map.shape[0]
	if ncomp == 0: return map # nothing to do
	cdef int ny = map.shape[1]
	cdef int nx = map.shape[2]
	# I can't figure out how to make a memoryview of an array of pointers, so let's just do
	# things brute force
	cdef float ** map_ = <float**>calloc(ncomp, sizeof(float*));
	for i in range(ncomp):
		map_[i] = &map[i,0,0]
	# Then the positions
	cdef int nobj = len(obj_ras)
	assert len(obj_decs) == len(obj_xs) == len(obj_ys) == nobj
	# Bin definitions
	cdef int nbin = len(rs)-1
	assert nbin >= 2, "rs must be [nbin] where nbin >= 2"
	# Output profiles
	if profs is None: profs = np.zeros((nobj,ncomp,nbin),np.float32)
	cdef float[:,:,::1] profs_buf = profs
	cdef float *** profs_ = <float***>calloc(nobj, sizeof(float**));
	for oi in range(nobj):
		profs_[oi] = <float**>calloc(ncomp, sizeof(float*));
		for ci in range(ncomp):
			profs_[oi][ci] = &profs_buf[oi,ci,0]
	# pixel positions
	cdef int separable_ = separable
	cdef int prof_equi_ = prof_equi
	assert len(posmap) == 2, "posmap must be [dec[ny],ra[nx]] if separable, else [{dec,ra},ny,nx]"
	pix_decs = np.asanyarray(posmap[0], dtype=np.float32, order="C")
	pix_ras  = np.asanyarray(posmap[1], dtype=np.float32, order="C")
	if separable:
		assert pix_decs.shape == (ny,) and pix_ras.shape == (nx,), "posmap ([%d],[%d]) must be [dec[ny=%d],ra[nx=%d]] if separable" % (len(pix_decs), len(pix_ras), ny, nx)
	else:
		assert pix_decs.shape == (ny,nx) and pix_ras.shape == (ny,nx), "posmap must be [{dec,ra},ny,nx] if not separable"
	cdef float [::1] pix_decs_ = pix_decs.reshape(-1)
	cdef float [::1] pix_ras_  = pix_ras.reshape(-1)
	times = np.zeros(1)
	cdef double [::1] times_ = times
	# Phew! That's a lot of wrapping!
	c.radial_sum(nobj, &obj_decs[0], &obj_ras[0], &obj_ys[0], &obj_xs[0], nbin, &rs[0], prof_equi_, ncomp, ny, nx, separable_, &pix_decs_[0], &pix_ras_[0], map_, profs_, &times_[0])
	free(map_)
	for oi in range(nobj): free(profs_[oi])
	free(profs_)
	if return_times: return profs, times
	else:            return profs
