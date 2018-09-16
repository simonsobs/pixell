from __future__ import print_function
import numpy as np
from . import wcs as enwcs, enmap, coordinates


## Analyst-facing functions

def postage_stamp(imap,ra_deg,dec_deg,width_arcmin,res_arcmin,proj='gnomonic',**kwargs):
    """Extract a postage stamp from a larger map by reprojecting to a coordinate system centered on the given position.
    
    imap -- (Ny,Nx) enmap array from which to extract stamps or filename for map TODO: support leading dimensions 
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
    stamp = cutout(imap,width=np.deg2rad(width_arcmin/60.)/np.cos(dec),ra=ra,dec=dec,return_slice=(type(imap)==str))
    if (type(imap)==str): stamp = enmap.read_map(imap,sel=stamp)
    if stamp is None: return None
    sshape,swcs = stamp.shape,stamp.wcs
    if proj=='car' or proj=='cea':
        tshape,twcs = rect_geometry(width=width,res=res,proj=proj)
    elif proj=='gnomonic':
        tshape,twcs = gnomonic_pole_geometry(width,res)
    rpix = get_rotated_pixels(sshape,swcs,tshape,twcs,inverse=False,pos_target=None,center_target=(0.,0.),center_source=(dec,ra))
    return enmap.enmap(rotate_map(stamp,pix_target=rpix,**kwargs),twcs)


def centered_map(imap,res,box=None,pixbox=None,proj='car',rpix=None,width=None,height=None,**kwargs):
    """Reproject a map such that its central pixel is at the origin of a given projection system (default: CAR).
    
    imap -- (Ny,Nx) enmap array from which to extract stamps TODO: support leading dimensions 
    res -- width of pixel in radians
    box -- optional bounding box of submap in radians
    pixbox -- optional bounding box of submap in pixel numbers
    proj -- coordinate system for target map; default is 'car'; can also specify 'cea' or 'gnomonic'
    rpix -- optional pre-calculated pixel positions from get_rotated_pixels()
    """
    proj = proj.strip().lower() ; assert proj in ['gnomonic','car','cea']
    # cut out a stamp assuming CAR ; TODO: generalize?
    if box is not None: pixbox = enmap.skybox2pixbox(imap.shape, imap.wcs, box)
    if pixbox is not None:
        omap = enmap.extract_pixbox(imap, pixbox)
    else:
        omap = imap
    sshape,swcs = omap.shape,omap.wcs
    dec,ra = enmap.pix2sky(sshape,swcs,(sshape[0]/2.,sshape[1]/2.)) # central pixel of source geometry
    height,width = enmap.extent(sshape,swcs)
    #box = enmap.box(sshape,swcs)
    #height = np.abs(box[1,0]-box[0,0])
    #width = np.abs(box[1,1]-box[0,1])
    if proj=='car' or proj=='cea':
        tshape,twcs = rect_geometry(width=width,res=res,proj=proj,height=height)
    elif proj=='gnomonic':
        tshape,twcs = gnomonic_pole_geometry(width,res,height=height)
    print(sshape,swcs,tshape,twcs,dec,ra)
    if rpix is None: rpix = get_rotated_pixels(sshape,swcs,tshape,twcs,inverse=False,pos_target=None,center_target=(0.,0.),center_source=(dec,ra))
    return enmap.enmap(rotate_map(omap,pix_target=rpix,**kwargs),twcs),rpix


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


def cutout(imap,width=None,ra=None,dec=None,pad=1,corner=False,preserve_wcs=False,res=None,npix=None,return_slice=False):
    if type(imap)==str:
        shape,wcs = enmap.read_map_geometry(imap)
    else:
        shape,wcs = imap.shape,imap.wcs
    shape = shape[-2:]
    Ny,Nx = shape
    fround = lambda x : int(np.round(x))
    iy,ix = enmap.sky2pix(shape,wcs,coords=(dec,ra),corner=corner)
    if res is None: res = np.min(enmap.extent(shape,wcs)/shape[-2:])
    if npix is None: npix = int(width/res)
    if fround(iy-npix/2)<pad or fround(ix-npix/2)<pad or fround(iy+npix/2)>(Ny-pad) or fround(ix+npix/2)>(Nx-pad): return None
    s = np.s_[fround(iy-npix/2.+0.5):fround(iy+npix/2.+0.5),fround(ix-npix/2.+0.5):fround(ix+npix/2.+0.5)]
    if return_slice: return s
    cutout = imap[s]
    return cutout

def rect_box(width,center=(0.,0.),height=None):
    if height is None: height = width
    ycen,xcen = center
    box = np.array([[-height/2.+ycen,-width/2.+xcen],[height/2.+ycen,width/2.+xcen]])
    return box 

def rect_geometry(width,res,height=None,center=(0.,0.),proj="car"):
    shape, wcs = enmap.geometry(pos=rect_box(width,center=center,height=height), res=res, proj=proj)
    return shape,wcs

