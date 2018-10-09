from __future__ import print_function
import numpy as np
from . import utils

def euler_mat(euler_angles, kind="zyz"):
    """Defines the rotation matrix M for a ABC euler rotation,
    such that M = A(alpha)B(beta)C(gamma), where euler_angles =
    [alpha,beta,gamma]. The default kind is ABC=ZYZ."""
    alpha, beta, gamma = euler_angles
    R1 = utils.rotmatrix(gamma, kind[2])
    R2 = utils.rotmatrix(beta,  kind[1])
    R3 = utils.rotmatrix(alpha, kind[0])
    return np.einsum("...ij,...jk->...ik",np.einsum("...ij,...jk->...ik",R3,R2),R1)

def euler_rot(euler_angles, coords, kind="zyz"):
    coords = np.asarray(coords)
    co     = coords.reshape(2,-1)
    M      = euler_mat(euler_angles, kind)
    rect   = utils.ang2rect(co, False)
    rect   = np.einsum("...ij,j...->i...",M,rect)
    co     = utils.rect2ang(rect, False)
    return co.reshape(coords.shape)

def recenter(angs, center, restore=False):
    """Recenter coordinates "angs" (as ra,dec) on the location given by "center",
    such that center moves to the north pole."""
    # Performs the rotation E(0,-theta,-phi). Originally did
    # E(phi,-theta,-phi), but that is wrong (at least for our
    # purposes), as it does not preserve the relative orientation
    # between the boresight and the sun. For example, if the boresight
    # is at the same elevation as the sun but 10 degrees higher in az,
    # then it shouldn't matter what az actually is, but with the previous
    # method it would.
    #
    # Now supports specifying where to recenter by specifying center as
    # lon_from,lat_from,lon_to,lat_to
    if len(center) == 4: ra0, dec0, ra1, dec1 = center
    elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], center[0]*0, center[1]*0+np.pi/2
    if restore: ra1 += ra0
    return euler_rot([ra1,dec0-dec1,-ra0], angs, kind="zyz")

def decenter(angs, center, restore=False):
    """Inverse operation of recenter."""
    if len(center) == 4: ra0, dec0, ra1, dec1 = center
    elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], center[0]*0, center[1]*0+np.pi/2
    if restore: ra1 += ra0
    return euler_rot([ra0,dec1-dec0,-ra1],  angs, kind="zyz")
