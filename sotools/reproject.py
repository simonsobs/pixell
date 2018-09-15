from __future__ import print_function
import numpy as np
from . import wcs as enwcs, enmap, coordinates


## Analyst-facing functions

def postage_stamp(imap,ra_deg,dec_deg,width_arcmin,res_arcmin,proj='gnomonic',**kwargs):
    """Extract a postage stamp from a larger map by reprojecting to a coordinate system centered on the given position.
    
    imap -- (Ny,Nx) enmap array from which to extract stamps TODO: support leading dimensions
    ra_deg -- right ascension in degrees
    dec_deg -- declination in degrees
    width_arcmin -- stamp dimension in arcminutes
    res_arcmin -- width of pixel in arcminutes
    proj -- coordinate system for postage stamp; default is 'gnomonic'; can also specify 'cea' or 'car'
    """
    proj = proj.strip().lower() ; assert proj in ['gnomonic','car','cea']
    dec = np.deg2rad(dec_deg) ; ra = np.deg2rad(ra_deg)
    width = np.deg2rad(width_arcmin/60.)
    res = np.deg2rad(res_arcmin/60.)
    # cut out a stamp assuming CAR ; TODO: generalize?
    stamp = cutout(imap,width=np.deg2rad(width_arcmin/60.)/np.cos(dec),ra=ra,dec=dec)
    if stamp is None: return None
    sshape,swcs = stamp.shape,stamp.wcs
    if proj=='car' or proj=='cea':
        tshape,twcs = rect_geometry(width=width,res=res,proj=proj)
    elif proj=='gnomonic':
        tshape,twcs = gnomonic_pole_geometry(width,res)
    rpix = get_rotated_pixels(sshape,swcs,tshape,twcs,inverse=False,pos_target=None,center_target=(0.,0.),center_source=(dec,ra))
    return rotate_map(stamp,pix_target=rpix,**kwargs)

def healpix_from_enmap(imap,**kwargs):
    return imap.to_healpix(**kwargs)

def enmap_from_healpix(hp_map,shape,wcs,ncomp=1,unit=1,lmax=0,rot_method="not-alm",rot=None,first=0):
    # TODO: Implement
    pass


## Helper functions

def gnomonic_pole_wcs(shape,res):
    Ny,Nx = shape[-2:]
    wcs = enwcs.WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN','DEC--TAN']
    wcs.wcs.crval = [0.,0.]
    wcs.wcs.cdelt[:] = np.rad2deg(res)
    #wcs.wcs.crpix=[Ny/2+1,Nx/2+1]
    wcs.wcs.crpix=[Ny/2.+0.5,Nx/2.+0.5]
    return wcs

def gnomonic_pole_geometry(width,res,height=None):
    if height is None: height = width
    Ny = int(height/res)
    Nx = int(width/res)
    return (Ny,Nx),gnomonic_pole_wcs((Ny,Nx),res)


def rotate_map(imap,shape_target=None,wcs_target=None,pix_target=None,**kwargs):
    if pix_target is None:
        pix_target = get_rotated_pixels(shape_source,wcs_source,shape_target,wcs_target)
    else:
        assert (shape_target is None) and (wcs_target is None), "Both pix_target and shape_target,wcs_target must not be specified."
    rotmap = enmap.at(imap,pix_target,unit="pix",**kwargs)
    return rotmap

def get_rotated_pixels(shape_source,wcs_source,shape_target,wcs_target,inverse=False,pos_target=None,center_target=None,center_source=None):
    """ Given a source geometry (shape_source,wcs_source)
    return the pixel positions in the target geometry (shape_target,wcs_target)
    if the source geometry were rotated such that its center lies on the center
    of the target geometry.

    WARNING: Only currently tested for a rotation along declination from one CAR
    geometry to another CAR geometry.
    """
    # what are the center coordinates of each geometries
    if center_source is None: center_source = enmap.pix2sky(shape_source,wcs_source,(shape_source[0]/2.,shape_source[1]/2.))
    if center_target is None: center_target = enmap.pix2sky(shape_target,wcs_target,(shape_target[0]/2.,shape_target[1]/2.))
    decs,ras = center_source
    dect,rat = center_target
    # what are the angle coordinates of each pixel in the target geometry
    if pos_target is None: pos_target = enmap.posmap(shape_target,wcs_target)
    lra = pos_target[1,:,:].ravel()
    ldec = pos_target[0,:,:].ravel()
    del pos_target
    # recenter the angle coordinates of the target from the target center to the source center
    if inverse:
        newcoord = coordinates.decenter((lra,ldec),(rat,dect,ras,decs))
    else:
        newcoord = coordinates.recenter((lra,ldec),(rat,dect,ras,decs))
    del lra
    del ldec
    # reshape these new coordinates into enmap-friendly form
    new_pos = np.empty((2,shape_target[0],shape_target[1]))
    new_pos[0,:,:] = newcoord[1,:].reshape(shape_target)
    new_pos[1,:,:] = newcoord[0,:].reshape(shape_target)
    del newcoord
    # translate these new coordinates to pixel positions in the target geometry based on the source's wcs
    pix_new = enmap.sky2pix(shape_source,wcs_source,new_pos)
    return pix_new


def cutout(imap,width=None,ra=None,dec=None,pad=1,corner=False,preserve_wcs=False,res=None,npix=None):
    shape = imap.shape[-2:]
    wcs = imap.wcs
    Ny,Nx = shape
    fround = lambda x : int(np.round(x))
    iy,ix = enmap.sky2pix(shape,wcs,coords=(dec,ra),corner=corner)
    if res is None: res = np.min(enmap.extent(shape,wcs)/shape[-2:])
    if npix is None: npix = int(width/res)
    if fround(iy-npix/2)<pad or fround(ix-npix/2)<pad or fround(iy+npix/2)>(Ny-pad) or fround(ix+npix/2)>(Nx-pad): return None
    s = np.s_[fround(iy-npix/2.+0.5):fround(iy+npix/2.+0.5),fround(ix-npix/2.+0.5):fround(ix+npix/2.+0.5)]
    cutout = imap[s]
    return cutout

def rect_geometry(width,res,height=None,proj="car"):
    if height is None: height = width
    shape, wcs = enmap.geometry(pos=[[-height/2.,-width/2.],[height/2.,width/2.]], res=res, proj=proj)
    return shape,wcs

