from __future__ import print_function
from pixell import sharp
import matplotlib
matplotlib.use('Agg')
from pixell import enmap,curvedsky,wcsutils,reproject
import numpy as np
import itertools,yaml,pickle,os,sys
import matplotlib.pyplot as plt

TEST_DIR = os.path.dirname(__file__)
DATA_PREFIX = os.path.join(TEST_DIR, 'data/')

"""
This script generates a set of reference values against which
pixell tests will be done.

"""


def get_reference_pixels(shape):
    """For a given 2D array, return a list of pixel indices
    corresponding to locations of a pre-determined and fixed
    pattern of reference pixels.

    e.g even x even
    1100110011
    1100110011
    0000000000
    0000000000
    1100110011
    1100110011
    0000000000
    0000000000
    1100110011
    1100110011

    e,g. odd x odd
    110010011
    110010011
    000000000
    000000000
    110010011
    000000000
    000000000
    110010011
    110010011

    e.g even x odd
    110010011
    110010011
    000000000
    000000000
    110010011
    110010011
    000000000
    000000000
    110010011
    110010011

    requires N>=5 in each axis
    """
    Ny,Nx = shape[-2:]
    assert (Ny>=5) and (Nx>=5), "Tests are not implemented for arrays with a dimension<5."
    """Given N, return 0,1,{x},N-2,N-1, where {x} is N//2-1,N//2 if N is even
    and {x} is N//2 if N is odd.
    """
    midextremes = lambda N: [0,1,N//2-1,N//2,N-2,N-1] if N%2==0 else [0,1,N//2,N-2,N-1]
    ys = midextremes(Ny)
    xs = midextremes(Nx)
    pixels = np.array(list(itertools.product(ys,xs)))
    return pixels

def mask(arr,pixels,val=0):
    """Mask an array arr based on array pixels of (y,x) pixel coordinates of (Npix,2)"""
    arr[...,pixels[:,0],pixels[:,1]] = val
    return arr

def get_pixel_values(arr,pixels):
    """Get values of arr at pixels specified in pixels (Npix,2)"""
    return arr[...,pixels[:,0],pixels[:,1]]

def get_meansquare(arr):
    return np.mean(arr*2.)

def save_mask_image(filename,shape):
    """Save a minimal plot of an array masked by the currently implemented reference
    pixel geometry

    e.g.
    > shape = (11,12)
    > save_mask_image("test_mask.png",shape)
    """
    arr = np.zeros(shape)
    pixels = get_reference_pixels(shape)
    masked = mask(arr,pixels,val=1)
    fig = plt.figure()
    im = plt.imshow(masked,cmap='rainbow')
    ax = plt.gca()
    ax.set_xticks(np.arange(0,shape[1])+0.5);
    ax.set_yticks(np.arange(0,shape[0])+0.5);
    ax.grid(which='major',color='w', linestyle='-', linewidth=5)
    ax.tick_params(axis='x', colors=(0,0,0,0))
    ax.tick_params(axis='y', colors=(0,0,0,0))
    for spine in im.axes.spines.values():
        spine.set_edgecolor((0,0,0,0))
    plt.savefig(filename, bbox_inches='tight')

def get_spectrum(ntype,noise,lmax,lmax_pad):
    ells = np.arange(0,lmax+lmax_pad)
    if ntype=="white": return np.ones(shape=(ells.size,))*(noise**2.)*((np.pi/180./60.)**2.)
    if ntype=="white_dl":
        spec = np.zeros(shape=(ells.size,))
        spec[2:] = (noise**2.)*((np.pi/180./60.)**2.)*2.*np.pi/ells[2:]/(ells+1.)[2:]
        return spec
    raise NotImplementedError

def get_spectra(yml_section,lmax,lmax_pad):
    spectra = {}
    for s in yml_section:
        spectra[s['name']] = get_spectrum(s['type'],s['noise'],lmax,lmax_pad)
    return spectra

def get_geometries(yml_section):
    geos = {}
    for g in yml_section:
        if g['type']=='fullsky':
            geos[g['name']] = enmap.fullsky_geometry(res=np.deg2rad(g['res_arcmin']/60.),proj=g['proj'])
        elif g['type']=='pickle':
            geos[g['name']] = pickle.load(open(DATA_PREFIX+"%s"%g['filename'],'rb'))
        else:
            raise NotImplementedError
    return geos

def generate_map(shape,wcs,powspec,lmax,seed):
    return curvedsky.rand_map(shape, wcs, powspec, lmax=lmax, dtype=np.float64, seed=seed, oversample=2.0, spin=[0,2], method="auto", direct=False, verbose=False)

def check_equality(imap1,imap2):
    assert np.all(imap1.shape==imap2.shape)
    assert wcsutils.equal(imap1.wcs,imap2.wcs)
    try:
        assert np.all(np.isclose(imap1,imap2))
    except:
        from orphics import io
        io.plot_img(imap1,"i1.png",lim=[-1.5,2])
        io.plot_img(imap2,"i2.png",lim=[-1.5,2])
        io.plot_img((imap1-imap2)/imap1,"ip.png",lim=[-0.1,0.1])
        assert 1==0

    
def get_extraction_test_results(yaml_file):
    print("Starting tests from ",yaml_file)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    geos = get_geometries(config['geometries'])
    lmax = config['lmax'] ; lmax_pad = config['lmax_pad']
    spectra = get_spectra(config['spectra'],lmax,lmax_pad)
    seed = config['seed']

    results = {}
    for g in geos.keys():
        results[g] = {}
        for s in spectra.keys():
            results[g][s] = {}
            imap = generate_map(geos[g][0][-2:],geos[g][1],spectra[s],lmax,seed)

            # Do write and read test
            filename = "temporary_map.fits" # NOT THREAD SAFE
            enmap.write_map(filename,imap)
            imap_in = enmap.read_map(filename)
            check_equality(imap,imap_in)
            for e in config['extracts']:
                print("Doing test for extract ",e['name']," with geometry ",g," and spectrum ",s,"...")
                if e['type']=='slice':
                    box = np.deg2rad(np.array(e['box_deg']))
                    cutout = enmap.read_map(filename,box=box)
                    cutout_internal = imap.submap(box=box)
                elif e['type']=='postage':
                    dec_deg,ra_deg = e['center_deg']
                    width_arcmin = e['width_arcmin']
                    res_arcmin = e['res_arcmin']
                    cutout = reproject.postage_stamp(filename,ra_deg,dec_deg,width_arcmin,res_arcmin,proj='gnomonic')
                    cutout_internal = reproject.postage_stamp(imap,ra_deg,dec_deg,width_arcmin,res_arcmin,proj='gnomonic')
                check_equality(cutout,cutout_internal)
                pixels = get_reference_pixels(cutout.shape)
                results[g][s]['refpixels'] = get_pixel_values(cutout,pixels)
                results[g][s]['meansquare'] = get_meansquare(cutout)

    os.remove(filename)
    return results,config['result_name']
