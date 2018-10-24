from __future__ import print_function
import numpy as np
from . import wcsutils, enmap, coordinates, sharp, curvedsky


# Analyst-facing functions

def postage_stamp(imap, ra_deg, dec_deg, width_arcmin,
                  res_arcmin, proj='gnomonic', **kwargs):
    """Extract a postage stamp from a larger map by reprojecting
    to a coordinate system centered on the given position.

    imap -- (Ny,Nx) enmap array from which to extract stamps or
    filename for map TODO: support leading dimensions
    ra_deg -- right ascension in degrees
    dec_deg -- declination in degrees
    width_arcmin -- stamp dimension in arcminutes
    res_arcmin -- width of pixel in arcminutes
    proj -- coordinate system for postage stamp; default is 'gnomonic';
    can also specify 'cea' or 'car'
    """
    proj = proj.strip().lower()
    assert proj in ['gnomonic', 'car', 'cea']
    dec = np.deg2rad(dec_deg)
    ra = np.deg2rad(ra_deg)
    width = np.deg2rad(width_arcmin / 60.)
    res = np.deg2rad(res_arcmin / 60.)
    # cut out a stamp assuming CAR ; TODO: generalize?
    stamp = cutout(imap, width=np.deg2rad(width_arcmin / 60.) /
                   np.cos(dec), ra=ra, dec=dec,
                   return_slice=(type(imap) == str))
    if (type(imap) == str):
        stamp = enmap.read_map(imap, sel=stamp)
    if stamp is None:
        return None
    sshape, swcs = stamp.shape, stamp.wcs
    if proj == 'car' or proj == 'cea':
        tshape, twcs = rect_geometry(width=width, res=res, proj=proj)
    elif proj == 'gnomonic':
        tshape, twcs = gnomonic_pole_geometry(width, res)
    rpix = get_rotated_pixels(sshape, swcs, tshape, twcs, inverse=False,
                              pos_target=None, center_target=(0., 0.),
                              center_source=(dec, ra))
    return enmap.enmap(rotate_map(stamp, pix_target=rpix, **kwargs), twcs)


def centered_map(imap, res, box=None, pixbox=None, proj='car', rpix=None,
                 width=None, height=None, width_multiplier=1., **kwargs):
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
    return enmap.enmap(rotate_map(omap, pix_target=rpix, **kwargs), twcs), rpix


def healpix_from_enmap_interp(imap, **kwargs):
    return imap.to_healpix(**kwargs)


def healpix_from_enmap(imap, lmax, nside):
    """Convert an ndmap to a healpix map such that the healpix map is
    band-limited up to lmax.

    Args:

    """
    import healpy as hp
    alm = curvedsky.map2alm(imap, lmax=lmax)
    if alm.ndim > 1:
        assert alm.shape[0] == 1
        alm = alm[0]
    retmap = hp.alm2map(alm, nside, lmax=lmax)
    return retmap


def enmap_from_healpix(hp_map, shape, wcs, ncomp=1, unit=1, lmax=0,
                       rot="gal,equ", first=0):
    """Convert a healpix map to an ndmap using harmonic space reprojection.
    The resulting map will be band-limited.

    Args:
        hp_map: an (Npix,) or (ncomp,Npix,) healpix map or a string containing
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

    Returns:
        res: the reprojected ndmap

    """
    import healpy as hp

    assert ncomp == 1 or ncomp == 3, "Only 1 or 3 components supported"
    dtype = np.float64
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
    return res


def enmap_from_healpix_interp(shape, wcs, hp_map, hp_coords="galactic",
                              interpolate=True):
    """Project a healpix map to an enmap of chosen shape and wcs. The wcs
    is assumed to be in equatorial (ra/dec) coordinates. If the healpix map
    is in galactic coordinates, this can be specified by hp_coords, and a
    slow conversion is done. No coordinate systems other than equatorial
    or galactic are currently supported. Only intensity maps are supported.
    If interpolate is True, bilinear interpolation using 4 nearest neighbours
    is done.

    shape -- 2-tuple (Ny,Nx)
    wcs -- enmap wcs object in equatorial coordinates
    hp_map -- array-like healpix map
    hp_coords -- "galactic" to perform a coordinate transform,
    "fk5","j2000" or "equatorial" otherwise
    interpolate -- boolean

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
    posmap = enmap.posmap(shape, wcs)

    ph = posmap[1, ...].T.ravel()
    th = posmap[0, ...].T.ravel()

    if hp_coords.lower() not in eq_coords:
        # This is still the slowest part. If there are faster coord transform
        # libraries, let me know!
        assert hp_coords.lower() in gal_coords
        gc = SkyCoord(ra=ph * u.degree, dec=th * u.degree, frame='fk5')
        gc = gc.transform_to('galactic')
        phOut = gc.l.deg * np.pi / 180.
        thOut = gc.b.deg * np.pi / 180.
    else:
        thOut = th
        phOut = ph

    thOut = np.pi / 2. - thOut  # polar angle is 0 at north pole

    # Not as slow as you'd expect
    if interpolate:
        imap[y, x] = hp.get_interp_val(
            hp_map, np.rad2deg(thOut), np.rad2deg(phOut))
    else:
        ind = hp.ang2pix(hp.get_nside(hp_map),
                         np.rad2deg(thOut), np.rad2deg(phOut))
        imap[:] = 0.
        imap[[y, x]] = hp_map[ind]

    return enmap.ndmap(imap, wcs)

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
    rotmap = enmap.at(imap, pix_target, unit="pix", **kwargs)
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
    lra = pos_target[1, :, :].ravel()
    ldec = pos_target[0, :, :].ravel()
    del pos_target
    # recenter the angle coordinates of the target from the target center
    # to the source center
    if inverse:
        newcoord = coordinates.decenter((lra, ldec), (rat, dect, ras, decs))
    else:
        newcoord = coordinates.recenter((lra, ldec), (rat, dect, ras, decs))
    del lra
    del ldec
    # reshape these new coordinates into enmap-friendly form
    new_pos = np.empty((2, shape_target[0], shape_target[1]))
    new_pos[0, :, :] = newcoord[1, :].reshape(shape_target)
    new_pos[1, :, :] = newcoord[0, :].reshape(shape_target)
    del newcoord
    # translate these new coordinates to pixel positions in the target geometry
    # based on the source's wcs
    pix_new = enmap.sky2pix(shape_source, wcs_source, new_pos)
    return pix_new


def cutout(imap, width=None, ra=None, dec=None, pad=1, corner=False,
           preserve_wcs=False, res=None, npix=None, return_slice=False):
    if type(imap) == str:
        shape, wcs = enmap.read_map_geometry(imap)
    else:
        shape, wcs = imap.shape, imap.wcs
    shape = shape[-2:]
    Ny, Nx = shape

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
    s = np.s_[fround(iy - npix / 2. + 0.5):fround(iy + npix / 2. + 0.5),
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


def rect_geometry(width, res, height=None, center=(0., 0.), proj="car"):
    shape, wcs = enmap.geometry(pos=rect_box(
        width, center=center, height=height), res=res, proj=proj)
    return shape, wcs
