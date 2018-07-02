import numpy as np

## Analyst-facing functions

def rotate_patch(imap,pos=None):
    """Reproject a patch to a geometry centered on a new location.
    imap -- (...,Ny,Nx) enmap array from which to extract stamps
    pos -- array-like (dec,ra) in radians. If None, rotates patch along declination only to equator.
    """
    pass


def postage_stamp(imap,pos,width,height=None):
    """Extract a postage stamp from a larger map by reprojecting to a coordinate system centered on the given position.
    
    imap -- (...,Ny,Nx) enmap array from which to extract stamps
    pos -- array-like [{dec},{ra}] in radians
    width -- stamp dimension along the ra direction in radians
    height -- stamp dimension along the dec direction
    """
    
    height = width if height is None else height
    pass

def healpix_from_enmap(imap,**kwargs):
    return imap.to_healpix(**kwargs)

def enmap_from_healpix(hp_map,shape,wcs,ncomp=1,unit=1,lmax=0,rot_method="not-alm",rot=None,first=0):
    pass


## Helper function


