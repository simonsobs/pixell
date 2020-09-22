from __future__ import print_function
import numpy as np
from . import wcsutils, utils, enmap, coordinates, curvedsky, fft
try: from . import sharp
except ImportError: pass

# Python 2/3 compatibility
try: basestring
except NameError: basestring = str

# ----------------------------

# Analyst-facing functions

def postage_stamp(inmap, ra_deg, dec_deg, width_arcmin,
                  res_arcmin, proj='gnomonic', return_cutout=False,
                  npad=3, rotate_pol=True, **kwargs):
    """Extract a postage stamp from a larger map by reprojecting
    to a coordinate system centered on the given position.

    Args:
        imap: (ncomp,Ny,Nx) or (Ny,Nx) enmap array from which to 
        extract stamps or filename or list of filenames for map
        ra_deg: right ascension in degrees
        dec_deg: declination in degrees
        width_arcmin: stamp dimension in arcminutes
        res_arcmin: width of pixel in arcminutes
        proj: coordinate system for postage stamp; default is 'gnomonic';
        can also specify 'cea' or 'car'
        return_cutout: return the pre-reprojection cutout as well
        npad: integer specifying number of extra pixels in pre-reprojection
        cutout
        **kwargs: additional parameters passed to interpolation enmap.at
        function

    Returns:
        rots: ndmap containing reprojected maps
        If return_cutout is True,
        cutout: pre-reprojection cutout as ndmap

    """
    proj = proj.strip().lower()
    assert proj in ['gnomonic', 'car', 'cea']
    dec = np.deg2rad(dec_deg)
    ra = np.deg2rad(ra_deg)
    width = np.deg2rad(width_arcmin / 60.)
    res = np.deg2rad(res_arcmin / 60.)
    
    if not(type(inmap) is list or type(inmap) is tuple):
        imaps = [inmap]
    else:
        imaps = inmap

    rots = []
    stamps = []
    for imap in imaps:
        if isinstance(imap,basestring):
            ishape,iwcs = enmap.read_map_geometry(imap)
        else:
            ishape,iwcs = imap.shape,imap.wcs
        mapres = np.min(np.abs(enmap.extent(ishape,iwcs))/ishape[-2:])
        # cut out a stamp assuming CAR ; TODO: generalize?
        stamp = cutout(imap, width=npad*mapres+np.deg2rad(width_arcmin / 60.) /
                       np.cos(dec), ra=ra, dec=dec,
                       return_slice=isinstance(imap,basestring))
        if isinstance(imap,basestring):
            stamp = enmap.read_map(imap, sel=stamp)
        if stamp is None:
            return (None,None) if return_cutout else None
        if stamp.ndim==2: stamp = stamp[None,:]
        ncomp = stamp.shape[0]
        assert ncomp==1 or ncomp==3, \
            "Only leading dimensions of 1 (intensity) or 3 (I,Q,U) are supported."
        sshape, swcs = stamp.shape, stamp.wcs
        if proj == 'car' or proj == 'cea':
            tshape, twcs = rect_geometry(width=width, res=res, proj=proj)
        elif proj == 'gnomonic':
            tshape, twcs = gnomonic_pole_geometry(width, res)
        rpix = get_rotated_pixels(sshape, swcs, tshape, twcs, inverse=False,
                                  pos_target=None, center_target=(0., 0.),
                                  center_source=(dec, ra))
        rot = enmap.enmap(rotate_map(stamp, pix_target=rpix[:2], **kwargs), twcs)
        if ncomp==3 and rotate_pol:
            rot[1:3] = enmap.rotate_pol(rot[1:3], -rpix[2]) # for polarization rotation if enough components
        rots.append(rot.copy())
        if return_cutout: stamps.append(enmap.enmap(stamp.copy(),swcs))
    rots = enmap.enmap(np.stack(rots),twcs)
    if len(imaps)==1: rots = rots[0]
    if return_cutout:
        return rots,stamps[0] if len(imaps)==1 else stamps
    return rots


def centered_map(imap, res, box=None, pixbox=None, proj='car', rpix=None,
                 width=None, height=None, width_multiplier=1.,
                 rotate_pol=True, **kwargs):
    """Reproject a map such that its central pixel is at the origin of a
    given projection system (default: CAR).

    imap -- (Ny,Nx) enmap array from which to extract stamps
    TODO: support leading dimensions
    res -- width of pixel in radians
    box -- optional bounding box of submap in radians
    pixbox -- optional bounding box of submap in pixel numbers
    proj -- coordinate system for target map; default is 'car';
    can also specify 'cea' or 'gnomonic'
    rpix -- optional pre-calculated pixel positions from get_rotated_pixels()
    """
    if imap.ndim==2: imap = imap[None,:]
    ncomp = imap.shape[0]
    proj = proj.strip().lower()
    assert proj in ['car', 'cea']
    # cut out a stamp assuming CAR ; TODO: generalize?
    if box is not None:
        pixbox = enmap.skybox2pixbox(imap.shape, imap.wcs, box)
    if pixbox is not None:
        omap = enmap.extract_pixbox(imap, pixbox)
    else:
        omap = imap
    sshape, swcs = omap.shape, omap.wcs
    # central pixel of source geometry
    dec, ra = enmap.pix2sky(sshape, swcs, (sshape[0] / 2., sshape[1] / 2.))
    dims = enmap.extent(sshape, swcs)
    dheight, dwidth = dims
    if height is None:
        height = dheight
    if width is None:
        width = dwidth
    width *= width_multiplier
    tshape, twcs = rect_geometry(
        width=width, res=res, proj=proj, height=height)
    if rpix is None:
        rpix = get_rotated_pixels(sshape, swcs, tshape, twcs, inverse=False,
                                  pos_target=None, center_target=(0., 0.),
                                  center_source=(dec, ra))
    rot = enmap.enmap(rotate_map(omap, pix_target=rpix[:2], **kwargs), twcs)
    if ncomp==3 and rotate_pol:
        rot[1:3] = enmap.rotate_pol(rot[1:3], -rpix[2]) # for polarization rotation if enough components
    return rot, rpix
    
def healpix_from_enmap_interp(imap, **kwargs):
    return imap.to_healpix(**kwargs)


def healpix_from_enmap(imap, lmax, nside):
    """Convert an ndmap to a healpix map such that the healpix map is
    band-limited up to lmax. Only supports single component (intensity)
    currently. The resulting map will be band-limited. Bright sources and 
    sharp edges could cause ringing. Use healpix_from_enmap_interp if you 
    are worried about this (e.g. for a mask), but that routine will not ensure 
    power to be correct to some lmax.


    Args:
        imap: ndmap of shape (Ny,Nx)
        lmax: integer specifying maximum multipole of map
        nside: integer specifying nside of healpix map

    Returns:
        retmap: (Npix,) healpix map as array

    """
    import healpy as hp
    alm = curvedsky.map2alm(imap, lmax=lmax, spin=0)
    if alm.ndim > 1:
        assert alm.shape[0] == 1
        alm = alm[0]
    retmap = hp.alm2map(alm.astype(np.complex128), nside, lmax=lmax)
    return retmap


def enmap_from_healpix(hp_map, shape, wcs, ncomp=1, unit=1, lmax=0,
                       rot="gal,equ", first=0, is_alm=False, return_alm=False, f_ell=None):
    """Convert a healpix map to an ndmap using harmonic space reprojection.
    The resulting map will be band-limited. Bright sources and sharp edges
    could cause ringing. Use enmap_from_healpix_interp if you are worried
    about this (e.g. for a mask), but that routine will not ensure power to 
    be correct to some lmax.

    Args:
        hp_map: an (Npix,) or (ncomp,Npix,) healpix map, or alms,  or a string containing
        the path to a healpix map on disk
        shape: the shape of the ndmap geometry to project to
        wcs: the wcs object of the ndmap geometry to project to
        ncomp: the number of components in the healpix map (either 1 or 3)
        unit: a unit conversion factor to divide the map by
        lmax: the maximum multipole to include in the reprojection
        rot: comma separated string that specify a coordinate rotation to
        perform. Use None to perform no rotation. e.g. default "gal,equ"
        to rotate a Planck map in galactic coordinates to the equatorial
        coordinates used in ndmaps.
        first: if a filename is provided for the healpix map, this specifies
        the index of the first FITS field
        is_alm: if True, interprets hp_map as alms
        return_alm: if True, returns alms also
        f_ell: optionally apply a transfer function f_ell(ell) -- this should be 
        a function of a single variable ell. e.g., lambda x: exp(-x**2/2/sigma**2)

    Returns:
        res: the reprojected ndmap or the a tuple (ndmap,alms) if return_alm
        is True

    """
    import healpy as hp

    dtype = np.float64
    if not(is_alm):
        assert ncomp == 1 or ncomp == 3, "Only 1 or 3 components supported"
        ctype = np.result_type(dtype, 0j)
        # Read the input maps
        if type(hp_map) == str:
            m = np.atleast_2d(hp.read_map(hp_map, field=tuple(
                range(first, first + ncomp)))).astype(dtype)
        else:
            m = np.atleast_2d(hp_map).astype(dtype)
        if unit != 1:
            m /= unit
        # Prepare the transformation
        print("Preparing SHT")
        nside = hp.npix2nside(m.shape[1])
        lmax = lmax or 3 * nside
        minfo = sharp.map_info_healpix(nside)
        ainfo = sharp.alm_info(lmax)
        sht = sharp.sht(minfo, ainfo)
        alm = np.zeros((ncomp, ainfo.nelem), dtype=ctype)
        # Perform the actual transform
        print("T -> alm")
        print(m.dtype, alm.dtype)
        sht.map2alm(m[0], alm[0])
        if ncomp == 3:
            print("P -> alm")
            sht.map2alm(m[1:3], alm[1:3], spin=2)
        del m
    else:
        alm = hp_map

    if f_ell is not None: alm = curvedsky.almxfl(alm,f_ell)

    if rot is not None:
        # Rotate by displacing coordinates and then fixing the polarization
        print("Computing pixel positions")
        pmap = enmap.posmap(shape, wcs)
        if rot:
            print("Computing rotated positions")
            s1, s2 = rot.split(",")
            opos = coordinates.transform(s2, s1, pmap[::-1], pol=ncomp == 3)
            pmap[...] = opos[1::-1]
            if len(opos) == 3:
                psi = -opos[2].copy()
            del opos
        print("Projecting")
        res = curvedsky.alm2map_pos(alm, pmap)
        if rot and ncomp == 3:
            print("Rotating polarization vectors")
            res[1:3] = enmap.rotate_pol(res[1:3], psi)
    else:
        print("Projecting")
        res = enmap.zeros((len(alm),) + shape[-2:], wcs, dtype)
        res = curvedsky.alm2map(alm, res)
    if return_alm: return res,alm
    return res


def enmap_from_healpix_interp(hp_map, shape, wcs , rot="gal,equ",
                              interpolate=False):
    """Project a healpix map to an enmap of chosen shape and wcs. The wcs
    is assumed to be in equatorial (ra/dec) coordinates. No coordinate systems 
    other than equatorial or galactic are currently supported. Only intensity 
    maps are supported.
    
    Args:
        hp_map: an (Npix,) healpix map
        shape: the shape of the ndmap geometry to project to
        wcs: the wcs object of the ndmap geometry to project to
        rot: comma separated string that specify a coordinate rotation to
        perform. Use None to perform no rotation. e.g. default "gal,equ"
        to rotate a Planck map in galactic coordinates to the equatorial
        coordinates used in ndmaps.
        interpolate: if True, bilinear interpolation using 4 nearest neighbours
        is done.

    """
    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    eq_coords = ['fk5', 'j2000', 'equatorial']
    gal_coords = ['galactic']
    imap = enmap.zeros(shape, wcs)
    Ny, Nx = shape
    pixmap = enmap.pixmap(shape, wcs)
    y = pixmap[0, ...].T.ravel()
    x = pixmap[1, ...].T.ravel()
    del pixmap
    posmap = enmap.posmap(shape, wcs)
    if rot is not None:
        s1, s2 = rot.split(",")
        opos = coordinates.transform(s2,s1, posmap[::-1], pol=None)
        posmap[...] = opos[1::-1]
    th = np.rad2deg(posmap[1, ...].T.ravel())
    ph = np.rad2deg(posmap[0, ...].T.ravel())
    del posmap
    if interpolate:
        imap[y, x] = hp.get_interp_val(
            hp_map, th, ph, lonlat=True)
    else:
        ind = hp.ang2pix(hp.get_nside(hp_map),
                         th, ph, lonlat=True)
        del th
        del ph
        imap[:] = 0.
        imap[(y, x)] = hp_map[ind]
        del y
        del x
    return enmap.ndmap(imap, wcs)



def ivar_hp_to_cyl(hmap, shape, wcs, rot=False,do_mask=True,extensive=True):
    from . import mpi, utils
    import healpy as hp
    comm = mpi.COMM_WORLD
    rstep = 100
    dtype = np.float32
    nside = hp.npix2nside(hmap.size)
    dec, ra = enmap.posaxes(shape, wcs)
    pix = np.zeros(shape, np.int32)
    # Get the pixel area. We assume a rectangular pixelization, so this is just
    # a function of y
    ipixsize = 4 * np.pi / (12 * nside ** 2)
    opixsize = get_pixsize_rect(shape, wcs)
    nblock = (shape[-2] + rstep - 1) // rstep
    for bi in range(comm.rank, nblock, comm.size):
        if bi % comm.size != comm.rank:
            continue
        i = bi * rstep
        rdec = dec[i : i + rstep]
        opos = np.zeros((2, len(rdec), len(ra)))
        opos[0] = rdec[:, None]
        opos[1] = ra[None, :]
        if rot:
            # This is unreasonably slow
            ipos = coordinates.transform("equ", "gal", opos[::-1], pol=True)
        else:
            ipos = opos[::-1]
        pix[i : i + rstep, :] = hp.ang2pix(nside, np.pi / 2 - ipos[1], ipos[0])
        del ipos, opos
    for i in range(0, shape[-2], rstep):
        pix[i : i + rstep] = utils.allreduce(pix[i : i + rstep], comm)
    omap = enmap.zeros((1,) + shape, wcs, dtype)
    imap = np.array(hmap).astype(dtype)
    imap = imap[None]
    if do_mask:
        bad = hp.mask_bad(imap)
        bad |= imap <= 0
        imap[bad] = 0
        del bad
    # Read off the nearest neighbor values
    omap[:] = imap[:, pix]
    if extensive: omap *= opixsize[:, None] / ipixsize
    # We ignore QU mixing during rotation for the noise level, so
    # it makes no sense to maintain distinct levels for them
    if do_mask:
        mask = omap[1:] > 0
        omap[1:] = np.mean(omap[1:], 0)
        omap[1:] *= mask
        del mask
    return omap

# Helper functions


def gnomonic_pole_wcs(shape, res):
    Ny, Nx = shape[-2:]
    wcs = wcsutils.WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [0., 0.]
    wcs.wcs.cdelt[:] = np.rad2deg(res)
    wcs.wcs.crpix = [Ny / 2. + 0.5, Nx / 2. + 0.5]
    return wcs


def gnomonic_pole_geometry(width, res, height=None):
    if height is None:
        height = width
    Ny = int(height / res)
    Nx = int(width / res)
    return (Ny, Nx), gnomonic_pole_wcs((Ny, Nx), res)


def rotate_map(imap, shape_target=None, wcs_target=None, shape_source=None,
               wcs_source=None, pix_target=None, **kwargs):
    if pix_target is None:
        pix_target = get_rotated_pixels(
            shape_source, wcs_source, shape_target, wcs_target)
    else:
        assert (shape_target is None) and (
            wcs_target is None), "Both pix_target and shape_target, \
            wcs_target must not be specified."
    rotmap = enmap.at(imap, pix_target[:2], unit="pix", **kwargs)
    return rotmap


def get_rotated_pixels(shape_source, wcs_source, shape_target, wcs_target,
                       inverse=False, pos_target=None,
                       center_target=None, center_source=None):
    """ Given a source geometry (shape_source,wcs_source)
    return the pixel positions in the target geometry (shape_target,wcs_target)
    if the source geometry were rotated such that its center lies on the center
    of the target geometry.

    WARNING: Only currently tested for a rotation along declination
    from one CAR geometry to another CAR geometry.
    """
    # what are the center coordinates of each geometries
    if center_source is None:
        center_source = enmap.pix2sky(
            shape_source, wcs_source,
            (shape_source[0] / 2., shape_source[1] / 2.))
    if center_target is None:
        center_target = enmap.pix2sky(
            shape_target, wcs_target,
            (shape_target[0] / 2., shape_target[1] / 2.))
    decs, ras = center_source
    dect, rat = center_target
    # what are the angle coordinates of each pixel in the target geometry
    if pos_target is None:
        pos_target = enmap.posmap(shape_target, wcs_target)
    #del pos_target
    # recenter the angle coordinates of the target from the target center
    # to the source center
    if inverse:
        transfun = lambda x: coordinates.decenter(x, (rat, dect, ras, decs))
    else:
        transfun = lambda x: coordinates.recenter(x, (rat, dect, ras, decs))
    res = coordinates.transform_meta(transfun, pos_target[1::-1], fields=["ang"])
    pix_new = enmap.sky2pix(shape_source, wcs_source, res.ocoord[1::-1])
    pix_new = np.concatenate((pix_new,res.ang[None]))
    return pix_new


def cutout(imap, width=None, ra=None, dec=None, pad=1, corner=False,
           res=None, npix=None, return_slice=False,sindex=None):
    if type(imap) == str:
        shape, wcs = enmap.read_map_geometry(imap)
    else:
        shape, wcs = imap.shape, imap.wcs
    Ny, Nx = shape[-2:]
    def fround(x):
        return int(np.round(x))
    iy, ix = enmap.sky2pix(shape, wcs, coords=(dec, ra), corner=corner)
    if res is None:
        res = np.min(enmap.extent(shape, wcs) / shape[-2:])
    if npix is None:
        npix = int(width / res)
    if fround(iy - npix / 2) < pad or fround(ix - npix / 2) < pad or \
       fround(iy + npix / 2) > (Ny - pad) or \
       fround(ix + npix / 2) > (Nx - pad):
        return None
    if sindex is None:
        s = np.s_[...,fround(iy - npix / 2. + 0.5):fround(iy + npix / 2. + 0.5),
                  fround(ix - npix / 2. + 0.5):fround(ix + npix / 2. + 0.5)]
    else:
        s = np.s_[sindex,fround(iy - npix / 2. + 0.5):fround(iy + npix / 2. + 0.5),
                  fround(ix - npix / 2. + 0.5):fround(ix + npix / 2. + 0.5)]

    if return_slice:
        return s
    cutout = imap[s]
    return cutout


def rect_box(width, center=(0., 0.), height=None):
    if height is None:
        height = width
    ycen, xcen = center
    box = np.array([[-height / 2. + ycen, -width / 2. + xcen],
                    [height / 2. + ycen, width / 2. + xcen]])
    return box


def get_pixsize_rect(shape, wcs):
    """Return the exact pixel size in steradians for the rectangular cylindrical
    projection given by shape, wcs. Returns area[ny], where ny = shape[-2] is the
    number of rows in the image. All pixels on the same row have the same area."""
    ymin = enmap.sky2pix(shape, wcs, [-np.pi / 2, 0])[0]
    ymax = enmap.sky2pix(shape, wcs, [np.pi / 2, 0])[0]
    y = np.arange(shape[-2])
    x = y * 0
    dec1 = enmap.pix2sky(shape, wcs, [np.maximum(ymin, y - 0.5), x])[0]
    dec2 = enmap.pix2sky(shape, wcs, [np.minimum(ymax, y + 0.5), x])[0]
    area = np.abs((np.sin(dec2) - np.sin(dec1)) * wcs.wcs.cdelt[0] * np.pi / 180)
    return area

def rect_geometry(width, res, height=None, center=(0., 0.), proj="car"):
    shape, wcs = enmap.geometry(pos=rect_box(
        width, center=center, height=height), res=res, proj=proj)
    return shape, wcs


def distribute(N,nmax):
    """
    Distribute N things into cells as equally as possible such that 
    no cell has more than nmax things.
    """
    actual_max = int(2.*(nmax+1)/3.)
    numcells = int(round(N*1./actual_max))
    each_cell = [actual_max]*(numcells-1)
    rem = N-sum(each_cell)
    if rem>0: each_cell.append(rem)
    assert sum(each_cell)==N
    return each_cell

def populate(shape,wcs,ofunc,maxpixy = 400,maxpixx = 400):
    """
    Loop through tiles in a new map of geometry (shape,wcs)
    with tiles that have maximum allowed shape (maxpixy,maxpixx)
    such that each tile is populated with the result of
    ofunc(oshape,owcs) where oshape,owcs is the geometry of each
    tile.
    """
    omap = enmap.zeros(shape,wcs)
    Ny,Nx = shape[-2:]
    tNys = distribute(Ny,maxpixy)
    tNxs = distribute(Nx,maxpixx)
    numy = len(tNys)
    numx = len(tNxs)
    sny = 0
    ntiles = numy*numx
    print("Number of tiles = ",ntiles)
    done = 0
    for i in range(numy):
        eny = sny+tNys[i]
        snx = 0
        for j in range(len(tNxs)):
            enx = snx+tNxs[j]
            sel = np.s_[...,sny:eny,snx:enx]
            oshape,owcs = enmap.slice_geometry(shape,wcs,sel)
            omap[sel] = ofunc(oshape,owcs)
            snx += tNxs[j]
            done += 1
        sny += tNys[i]
        print(done , " / ", ntiles, " tiles done...")
    return omap

# --------- Candidate replacement functions below. Please comment. ----------------

def thumbnails(imap, coords, r=5*utils.arcmin, res=None, proj="tan", apod=2*utils.arcmin,
		order=3, oversample=4, pol=None, oshape=None, owcs=None, extensive=False, verbose=False,
		filter=None):
	"""Given an enmap [...,ny,nx] and a set of coords [n,{dec,ra}], extract a set
	of thumbnail images [n,...,thumby,thumbx] centered on each set of
	coordinates. Each of these thumbnail images is projected onto a local tangent
	plane, removing the effect of size and shape distortions in the input map.

	If oshape, owcs are specified, then the thumbnails will have this geometry,
	which should be centered on [0,0]. Otherwise, a geometry with the given
	projection (defaults to "tan" = gnomonic projection) will be constructed,
	going up to a maximum radius of r.

	The reprojection involved in this operation implies interpolation. The default
	is to use fft rescaling to oversample the input pixels by the given pixel, and
	then use bicubic spline interpolation to read off the values at the output
	pixel centers. The fft oversampling can be controlled with the oversample argument.
	Values <= 1 turns this off. The other interpolation step is controlled using the
	"order" argument. 0/1/3 corresponds to nearest neighbor, bilinear and bicubic spline
	interpolation respectively.

	If pol == True, then Q,U will be rotated to take into account the change in
	the local northward direction impled in the reprojection. The default is to
	do polarization rotation automatically if the input map has a compatible shape,
	e.g. at least 3 axes and a length of 3 for the 3rd last one. TODO: I haven't
	tested this yet.

	If extensive == True (not the default), then the map is assumed to contain an
	extensive field rather than an intensive one. An extensive field is one where
	the values in the pixels depend on the size of the pixel. For example, if the
	inverse variance in the map is given per pixel, then this ivar map will be
	extensive, but if it's given in units of inverse variance per square arcmin
	then it's intensive.

	For reprojecting inverse variance maps, consider using the wrapper thumbnails_ivar,
	which makes it easier to avoid common pitfalls."""
	# FIXME: Specifying a geometry manually is broken - see usage of r in neighborhood_pixboxes below
	# Handle arbitrary coords shape
	coords = np.asarray(coords)
	ishape = coords.shape[:-1]
	coords = coords.reshape(-1, coords.shape[-1])
	# If the output geometry was not given explicitly, then build one
	if oshape is None:
		if res is None: res = min(np.abs(imap.wcs.wcs.cdelt))*utils.degree/2
		oshape, owcs = enmap.thumbnail_geometry(r=r, res=res, proj=proj)
	# Check if we should be doing polarization rotation
	pol_compat = imap.ndim >= 3 and imap.shape[-3] == 3
	if pol is None: pol = pol_compat
	if pol and not pol_compat: raise ValueError("Polarization rotation requested, but can't interpret map shape %s as IQU map" % (str(imap.shape)))
	nsrc = len(coords)
	if verbose: print("Extracting %d %dx%d thumbnails from %s map" % (nsrc, oshape[-2], oshape[-1], str(imap.shape)))
	opos = enmap.posmap(oshape, owcs)
	# Get the pixel area around each of the coordinates
	rtot     = r + apod
	apod_pix = utils.nint(apod/(np.min(np.abs(imap.wcs.wcs.cdelt))*utils.degree))
	pixboxes = enmap.neighborhood_pixboxes(imap.shape, imap.wcs, coords, rtot)
	# Define our output maps, which we will fill below
	omaps = enmap.zeros((nsrc,)+imap.shape[:-2]+oshape, owcs, imap.dtype)
	for si, pixbox in enumerate(pixboxes):
		if oversample > 1:
			# Make the pixbox fft-friendly
			for i in range(2):
				pixbox[1,i] = pixbox[0,i] + fft.fft_len(pixbox[1,i]-pixbox[0,i], direction="above", factors=[2,3,5])
		ithumb = imap.extract_pixbox(pixbox)
		if extensive: ithumb /= ithumb.pixsizemap()
		ithumb = ithumb.apod(apod_pix, fill="median")
		if filter is not None: ithumb = filter(ithumb)
		if verbose:
			print("%4d/%d %6.2f %6.2f %8.2f %dx%d" % (si+1, nsrc, coords[si,0]/utils.degree, coords[si,1]/utils.degree, np.max(ithumb), ithumb.shape[-2], ithumb.shape[-1]))
		# Oversample using fourier if requested. We do this because fourier
		# interpolation is better than spline interpolation overall
		if oversample > 1:
			fshape = utils.nint(np.array(oshape[-2:])*oversample)
			ithumb = ithumb.resample(fshape, method="fft")
		# I apologize for the syntax. There should be a better way of doing this
		ipos = coordinates.transform("cel", ["cel",[[0,0,coords[si,1],coords[si,0]],False]], opos[::-1], pol=pol)
		ipos, rest = ipos[1::-1], ipos[2:]
		omaps[si] = ithumb.at(ipos, order=order)
		# Apply the polarization rotation. The sign is flipped because we computed the
		# rotation from the output to the input
		if pol: omaps[si] = enmap.rotate_pol(omaps[si], -rest[0])
	if extensive: omaps *= omaps.pixsizemap()
	# Restore original dimension
	omaps = omaps.reshape(ishape + omaps.shape[1:])
	return omaps

def thumbnails_ivar(imap, coords, r=5*utils.arcmin, res=None, proj="tan",
		oshape=None, owcs=None, extensive=True, verbose=False):
	"""Like thumbnails, but for hitcounts, ivars, masks, and other quantities that
	should stay positive and local. Remember to set extensive to True if you have an
	extensive quantity, i.e. if the values in each pixel would go up if multiple pixels
	combined. An example of this is a hitcount map or ivar per pixel. Conversely, if
	you have an intensive quantity like ivar per arcmin you should set extensive=False."""
	return thumbnails(imap, coords, r=r, res=res, proj=proj, oshape=oshape, owcs=owcs,
			order=1, oversample=1, pol=False, extensive=extensive, verbose=verbose)
