"""Tests for `pixell` package."""

import unittest

from pixell import enmap
from pixell import curvedsky
from pixell import lensing
from pixell import array_ops
from pixell import enplot
from pixell import powspec
from pixell import reproject
from pixell import pointsrcs
from pixell import wcsutils
from pixell import utils as u
from pixell import colors
from pixell import fft
from pixell import tilemap
from pixell import utils
import numpy as np
import pickle
import os,sys,time

import matplotlib
matplotlib.use('Agg')
import numpy as np
import itertools,yaml,pickle,os,sys
import matplotlib.pyplot as plt

TEST_DIR = os.path.dirname(__file__)
DATA_PREFIX = os.path.join(TEST_DIR, 'data/')

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
            geos[g['name']] = enmap.fullsky_geometry(res=np.deg2rad(g['res_arcmin']/60.),proj=g['proj'],variant="CC")
        elif g['type']=='pickle':
            geos[g['name']] = pickle.load(open(DATA_PREFIX+"%s"%g['filename'],'rb'))
        else:
            raise NotImplementedError
    return geos

def generate_map(shape,wcs,powspec,lmax,seed):
    return curvedsky.rand_map(shape, wcs, powspec, lmax=lmax, dtype=np.float64, seed=seed, spin=[0,2], method="auto", verbose=False)

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
                check_equality(cutout,cutout_internal)
                pixels = get_reference_pixels(cutout.shape)
                results[g][s]['refpixels'] = get_pixel_values(cutout,pixels)
                results[g][s]['meansquare'] = get_meansquare(cutout)

    os.remove(filename)
    return results,config['result_name']

lens_version = '071123'

def get_offset_result(res=1.,dtype=np.float64,seed=1):
    shape,wcs  = enmap.fullsky_geometry(res=np.deg2rad(res), variant="CC")
    shape = (3,) + shape
    obs_pos = enmap.posmap(shape, wcs)
    np.random.seed(seed)
    grad = enmap.enmap(np.random.random(shape),wcs)*1e-3
    raw_pos = enmap.samewcs(lensing.offset_by_grad(obs_pos, grad, pol=shape[-3]>1, geodesic=True), obs_pos)
    return obs_pos,grad,raw_pos

def get_lens_result(res=1.,lmax=400,dtype=np.float64,seed=1):
    shape,wcs  = enmap.fullsky_geometry(res=np.deg2rad(res), variant="CC")
    shape = (3,) + shape
    # ells = np.arange(lmax)
    ps_cmb,ps_lens = powspec.read_camb_scalar(DATA_PREFIX+"test_scalCls.dat")
    ps_lensinput = np.zeros((4,4,ps_cmb.shape[-1]))
    ps_lensinput[0,0] = ps_lens
    ps_lensinput[1:,1:] = ps_cmb
    lensed = lensing.rand_map(shape, wcs, ps_lensinput, lmax=lmax, maplmax=None, dtype=dtype, seed=seed, phi_seed=None, spin=[0,2], output="lu", geodesic=True, verbose=False, delta_theta=None)
    return lensed

# Helper functions for adjointness tests
def zip_alm(alm, ainfo):
    n = ainfo.lm2ind(1,1)
    first  = alm[...,:n].real
    second = alm[...,n:].view(utils.real_dtype(alm.dtype))*2**0.5
    return np.concatenate([first, second],-1)

def unzip_alm(zalm, ainfo):
    n = ainfo.lm2ind(1,1)
    oalm = np.zeros(zalm.shape[:-1] + (ainfo.nelem,), utils.complex_dtype(zalm.dtype))
    oalm[...,:n] = zalm[...,:n]
    oalm[...,n:] = zalm[...,n:].view(oalm.dtype)/2**0.5
    return oalm

def zalm_len(ainfo): return (2*ainfo.nelem-ainfo.lm2ind(1,1)).astype(int)
def zip_mat(mat):
    # Mat is ncomp_alm,nzalm,ncomp_map,ny,nx.
    # Want (ncomp*ncomp*nzalm,ny,nx)
    mat = np.moveaxis(mat, 2, 1)
    mat = mat.reshape((-2,)+mat.shape[-2:])
    return mat

def map_bash(fun, shape, wcs, ncomp, lmax, dtype=np.float64):
    ctype = utils.complex_dtype(dtype)
    ainfo = curvedsky.alm_info(lmax)
    nzalm = zalm_len(ainfo)
    umap  = enmap.zeros((ncomp,)+shape, wcs, dtype=dtype)
    oalm  = np.zeros((ncomp,ainfo.nelem), dtype=ctype)
    mat   = np.zeros((ncomp,nzalm,ncomp)+shape, dtype=dtype)
    for I in utils.nditer((ncomp,)+shape):
        umap[I] = 1
        oalm[:] = 0
        fun(map=umap, alm=oalm, ainfo=ainfo)
        mat[(slice(None),slice(None))+I] = zip_alm(oalm, ainfo)
        umap[I] = 0
    return zip_mat(mat)

def alm_bash(fun, shape, wcs, ncomp, lmax, dtype=np.float64):
    ctype = utils.complex_dtype(dtype)
    ainfo = curvedsky.alm_info(lmax)
    nzalm = zalm_len(ainfo)
    zalm  = np.zeros((ncomp,nzalm), dtype)
    omap  = enmap.zeros((ncomp,)+shape, wcs, dtype)
    mat   = np.zeros((ncomp,nzalm,ncomp)+shape, dtype)
    for ci in range(ncomp):
        for i in range(nzalm):
            # Why is this 0.5 needed?
            zalm[ci,i] = 1 #if i < ainfo.lm2ind(1,1) else 0.5
            omap[:] = 0
            fun(alm=unzip_alm(zalm,ainfo), map=omap, ainfo=ainfo)
            mat[ci,i] = omap
            zalm[ci,i] = 0
    return zip_mat(mat)

# End of adjointness helpers

class PixelTests(unittest.TestCase):


    def test_almxfl(self):
        import healpy as hp

        for lmax in [100,400,500,1000]:
            ainfo = curvedsky.alm_info(lmax)
            alms = hp.synalm(np.ones(lmax+1),lmax = lmax, new=True)
            filtering = np.ones(lmax+1)
            alms0 = ainfo.lmul(alms.copy(),filtering)
            assert np.all(np.isclose(alms0,alms))

        for lmax in [100,400,500,1000]:
            ainfo = curvedsky.alm_info(lmax)
            alms = hp.synalm(np.ones(lmax+1),lmax = lmax, new=True)
            alms0 = curvedsky.almxfl(alms.copy(),lambda x: np.ones(x.shape))
            assert np.all(np.isclose(alms0,alms))
            
        

    def test_rand_alm(self):
        def nalm(lmax):
            return (lmax + 1) * (lmax + 2) / 2

        lmaxes = [50, 100, 150, 300]
        
        mypower = np.ones(50)
        for lmax in lmaxes:
            palm = curvedsky.rand_alm(mypower, lmax = lmax)
            halm = curvedsky.rand_alm_healpy(  mypower, lmax = lmax)
            
            print("nalm(%i) = %i, curvedsky.rand_alm gives %s, curvedsky.rand_alm_healpy gives %s "\
	              % (lmax, \
                     nalm(lmax),\
                     palm.shape, \
                     halm.shape)        )
            assert np.all(np.isclose(np.asarray(palm.shape),np.asarray(halm.shape)))
            
    
    def test_offset(self):
        obs_pos,grad,raw_pos = get_offset_result(1.)
        obs_pos0 = enmap.read_map(DATA_PREFIX+"MM_offset_obs_pos_%s.fits" % lens_version)
        grad0 = enmap.read_map(DATA_PREFIX+"MM_offset_grad_%s.fits"  % lens_version)
        raw_pos0 = enmap.read_map(DATA_PREFIX+"MM_offset_raw_pos_%s.fits"  % lens_version)
        assert np.all(np.isclose(obs_pos,obs_pos0))
        assert np.all(np.isclose(raw_pos,raw_pos0))
        assert np.all(np.isclose(grad,grad0))
        assert wcsutils.equal(grad.wcs,grad0.wcs)
        assert wcsutils.equal(obs_pos.wcs,obs_pos0.wcs)
        assert wcsutils.equal(raw_pos.wcs,raw_pos0.wcs)

    def test_lensing(self):
        lensed,unlensed = get_lens_result(1.,400,np.float64)
        lensed0 = enmap.read_map(DATA_PREFIX+"MM_lensed_%s.fits"  % lens_version)
        unlensed0 = enmap.read_map(DATA_PREFIX+"MM_unlensed_%s.fits"  % lens_version)
        y,x = lensed0.posmap()
        assert np.all(np.isclose(lensed,lensed0))
        assert np.all(np.isclose(unlensed,unlensed0))
        assert wcsutils.equal(lensed.wcs,lensed0.wcs)
        assert wcsutils.equal(unlensed.wcs,unlensed0.wcs)
        assert wcsutils.equal(unlensed.wcs,lensed.wcs)
    
    def test_enplot(self):
        print("Testing enplot...")
        shape,wcs = enmap.geometry(pos=(0,0),shape=(3,100,100),res=0.01)
        a = enmap.ones(shape,wcs)
        # basic
        p = enplot.plot(a)
        # colorbar
        p = enplot.plot(a, colorbar=True)
        # annotation
        p = enplot.plot(a, annotate=DATA_PREFIX+"annot.txt")

    def test_fft(self):
        # Tests that ifft(ifft(imap))==imap, i.e. default normalizations are consistent
        shape,wcs = enmap.geometry(pos=(0,0),shape=(3,100,100),res=0.01)
        imap = enmap.enmap(np.random.random(shape),wcs)
        assert np.all(np.isclose(imap,enmap.ifft(enmap.fft(imap,normalize='phy'),normalize='phy').real))
        assert np.all(np.isclose(imap,enmap.ifft(enmap.fft(imap)).real))

    def test_fft_input_shape(self):
        # Tests fft for various shapes and choices of axes.
        # 1D FFT over last axis for 3d array.
        signal = np.ones((1, 2, 5))
        signal[0,1,:] = 10.
        out_exp = np.zeros((1, 2, 5), dtype=np.complex128)
        out_exp[0,0,0] = 5
        out_exp[0,1,0] = 50
        out = fft.fft(signal)
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 1D FFT over middle axis for 3d array.
        signal = np.ones((1, 5, 2))
        signal[0,:,1] = 10.
        out_exp = np.zeros((1, 5, 2), dtype=np.complex128)
        out_exp[0,0,0] = 5
        out_exp[0,0,1] = 50
        out = fft.fft(signal, axes=[-2])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 2D FFT over last 2 axes of 4d array.
        signal = np.ones((1, 2, 5, 10))
        signal[0,1,:] = 10.
        out_exp = np.zeros((1, 2, 5, 10), dtype=np.complex128)
        out_exp[0,0,0,0] = 50
        out_exp[0,1,0,0] = 500
        out = fft.fft(signal, axes=[-2, -1])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 2D FFT over last 2 axes of 4d non-contiguous array.
        signal = np.ones((1, 2, 5, 10), dtype=np.complex128)
        signal[0,1,:] = 10
        ft = np.zeros((5, 10, 1, 2), dtype=np.complex128).transpose(2, 3, 0, 1)
        out_exp = np.zeros_like(ft)
        out_exp[0,0,0,0] = 50
        out_exp[0,1,0,0] = 500
        out = fft.fft(signal, ft=ft, axes=[-2, -1])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(np.shares_memory(ft, out))
        self.assertFalse(out.flags['C_CONTIGUOUS'])

        # 2D FFT over middle 2 axes of 4d array.
        signal = np.ones((1, 5, 10, 2))
        signal[0,:,:,1] = 10.
        out_exp = np.zeros((1, 5, 10, 2), dtype=np.complex128)
        out_exp[0,0,0,0] = 50
        out_exp[0,0,0,1] = 500
        out = fft.fft(signal, axes=[-3, -2])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

    def test_ifft_input_shape(self):
        # Tests ifft for various shapes and choices of axes.
        # 1D IFFT over last axis for 3d array.
        fsignal = np.ones((1, 2, 5), dtype=np.complex128)
        fsignal[0,1,:] = 10.
        out_exp = np.zeros((1, 2, 5))
        out_exp[0,0,0] = 5
        out_exp[0,1,0] = 50
        out = fft.ifft(fsignal)
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 1D IFFT over middle axis for 3d array.
        fsignal = np.ones((1, 5, 2), dtype=np.complex128)
        fsignal[0,:,1] = 10.
        out_exp = np.zeros((1, 5, 2))
        out_exp[0,0,0] = 5
        out_exp[0,0,1] = 50
        out = fft.ifft(fsignal, axes=[-2])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 2D IFFT over last 2 axes of 4d array.
        fsignal = np.ones((1, 2, 5, 10), dtype=np.complex128)
        fsignal[0,1,:] = 10.
        out_exp = np.zeros((1, 2, 5, 10))
        out_exp[0,0,0,0] = 50
        out_exp[0,1,0,0] = 500
        out = fft.ifft(fsignal, axes=[-2, -1])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

        # 2D IFFT over last 2 axes of 4d non-contiguous array.
        fsignal = np.ones((1, 2, 5, 10), dtype=np.complex128)
        fsignal[0,1,:] = 10.
        tod = np.zeros((5, 10, 1, 2), dtype=np.complex128).transpose(2, 3, 0, 1)
        out_exp = np.zeros_like(tod)
        out_exp[0,0,0,0] = 50
        out_exp[0,1,0,0] = 500
        out = fft.ifft(fsignal, tod=tod, axes=[-2, -1])
        self.assertTrue(np.shares_memory(tod, out))
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertFalse(out.flags['C_CONTIGUOUS'])

        # 2D IFFT over middle 2 axes of 4d array.
        fsignal = np.ones((1, 5, 10, 2), dtype=np.complex128)
        fsignal[0,:,:,1] = 10.
        out_exp = np.zeros((1, 5, 10, 2))
        out_exp[0,0,0,0] = 50
        out_exp[0,0,0,1] = 500
        out = fft.ifft(fsignal, axes=[-3, -2])
        np.testing.assert_allclose(out, out_exp, atol=1e-12)
        self.assertTrue(out.flags['C_CONTIGUOUS'])

    def test_queb_rotmat_complex(self):
        # Tests that the rotmat respects fft symmetry constraints
        # for real maps -- ie, map2harm, harm2map produces
        # sensible round-trips. This version tests a map that's
        # complex in real-space
        ishapes = [(10, 10), (10, 11), (11, 10), (11, 11)]
        dtypes = [np.complex64, np.complex128]
        comps = [1, 2] # this is the comp we will set to non-zero

        for ishape, dtype, comp in itertools.product(ishapes, dtypes, comps):
            shape, wcs = enmap.geometry(pos=(0, 0), shape=ishape, res=0.01)

            # define some easy test input to evaluate for mixing
            input_complex_hmap = enmap.zeros((3, *shape), wcs, dtype=dtype)
            input_complex_hmap[comp] += 1. + 1.j
            output_complex_hmap = enmap.zeros((3, *shape), wcs, dtype=dtype)
            output_complex_hmap[comp] += 1. + 1.j

            # do a round-trip and check that we get what we expect.
            # the queb operations mix E and B and could lead to artifacts if
            # we're not careful
            if input_complex_hmap.real.dtype.itemsize == 8:
                atol = 1e-10
            elif input_complex_hmap.real.dtype.itemsize == 4:
                atol = 1e-5
            test_output_complex_hmap = enmap.map2harm(
                enmap.harm2map(input_complex_hmap, keep_imag=True)
                )
            assert np.allclose(
                test_output_complex_hmap, output_complex_hmap, rtol=0, atol=atol
                ), f'{ishape=}, {dtype=}, {comp=}'

    def test_queb_rotmat_real(self):
        # Tests that the rotmat respects fft symmetry constraints
        # for real maps -- ie, map2harm, harm2map produces
        # sensible round-trips. This version tests a map that's
        # real in real-space. This means that its DC must be real
        # and along even dimensions the nyquist freq must also be real
        ishapes = [(10, 10), (10, 11), (11, 10), (11, 11)]
        dtypes = [np.complex64, np.complex128]
        comps = [1, 2] # this is the comp we will set to non-zero
        np.random.seed(0)

        for ishape, dtype, comp in itertools.product(ishapes, dtypes, comps):
            shape, wcs = enmap.geometry(pos=(0, 0), shape=ishape, res=0.01)

            # define some easy test input to evaluate for mixing
            input_hmap = enmap.map2harm(enmap.rand_gauss((3,)+shape, wcs, dtype=dtype))
            # do a round-trip and check that we get what we expect.
            atol = 1e-10 if dtype == np.complex128 else 1e-5
            output_hmap = enmap.map2harm(enmap.harm2map(input_hmap))
            assert np.allclose(
                input_hmap, output_hmap, rtol=0, atol=atol), f'{ishape=}, {dtype=}, {comp=}'

    def test_extract(self):
        # Tests that extraction is sensible
        shape,wcs = enmap.geometry(pos=(0,0),shape=(500,500),res=0.01)
        imap = enmap.enmap(np.random.random(shape),wcs)
        smap = imap[200:300,200:300]
        sshape,swcs = smap.shape,smap.wcs
        smap2 = enmap.extract(imap,sshape,swcs)
        pixbox = enmap.pixbox_of(imap.wcs,sshape,swcs)
        # Do write and read test
        filename = "temporary_extract_map.fits" # NOT THREAD SAFE
        enmap.write_map(filename,imap)
        smap3 = enmap.read_map(filename,pixbox=pixbox)
        os.remove(filename)
        assert np.all(np.isclose(smap,smap2))
        assert np.all(np.isclose(smap,smap3))
        assert wcsutils.equal(smap.wcs,smap2.wcs)
        assert wcsutils.equal(smap.wcs,smap3.wcs)

    def test_fullsky_geometry(self):
        # Tests whether number of pixels and area of a full-sky 0.5 arcminute resolution map are correct
        print("Testing full sky geometry...")
        test_res_arcmin = 0.5
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(test_res_arcmin/60.),proj='car')
        assert shape[0]==21600 and shape[1]==43200
        assert abs(enmap.area(shape,wcs) - 4*np.pi) < 1e-6

    def test_pixels(self):
        """Runs reference pixel and mean-square comparisons on extracts from randomly generated
        maps"""
        print("Testing reference pixels...")
        results,rname = get_extraction_test_results(TEST_DIR+"/tests.yml")
        cresults = pickle.load(open(DATA_PREFIX+"%s.pkl" % rname,'rb'))
        assert sorted(results.keys())==sorted(cresults.keys())
        for g in results.keys():
            assert sorted(results[g].keys())==sorted(cresults[g].keys())
            for s in results[g].keys():
                assert sorted(results[g][s].keys())==sorted(cresults[g][s].keys())
                for e in results[g][s].keys():
                    assert np.all(np.isclose(results[g][s][e],cresults[g][s][e]))


    def test_sim_slice(self):
        ps = powspec.read_spectrum(DATA_PREFIX+"test_scalCls.dat")[:1,:1]
        test_res_arcmin = 10.0
        lmax = 2000
        fact = 2.
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(test_res_arcmin/60.),proj='car')
        omap = curvedsky.rand_map(shape, wcs, ps,lmax=lmax)
        ofunc = lambda ishape,iwcs: fact*enmap.extract(omap,ishape,iwcs)
        nmap = reproject.populate(shape,wcs,ofunc,maxpixy = 400,maxpixx = 400)
        assert np.all(np.isclose(nmap/omap,2.))


    # This is currently broken, but it's always been broken. For doubly-even
    # dimensions, the double-nyquist frequency entry has inconsistent sign.
    # Need to find a way to fix this.
    #def test_b_sign(self):
    #    """
    #    We generate a random IQU map with geometry such that cdelt[0]<0 and
    #    cdelt[1]>0.
    #    We transform this to TEB with map2harm and map2alm followed by 
    #    scalar harm2map and alm2map and use these as reference T,E,B maps.
    #    We flip the original map along the RA direction, Dec direction, or both.
    #    We transform this to TEB with map2harm and map2alm followed by 
    #    scalar harm2map and alm2map and use these as comparison T,E,B maps.
    #    We compare these maps.
    #    """
    #    cltt,clee,clbb,clte = powspec.read_spectrum(DATA_PREFIX+"cosmo2017_10K_acc3_lensedCls.dat",expand=None)

    #    ps_cmb = np.zeros((3,3,cltt.size))
    #    ps_cmb[0,0] = cltt
    #    ps_cmb[1,1] = clee
    #    ps_cmb[2,2] = clbb
    #    ps_cmb[1,0] = clte
    #    ps_cmb[0,1] = clte
    #    seed = 100

    #    # test all possible cdelt flips and shapes
    #    sels = (np.s_[...], np.s_[...,::-1], np.s_[...,::-1,:], np.s_[...,::-1,::-1])

    #    Ny, Nx = 1080, 2160 
    #    shapes = ((Ny, Nx), (Ny-1, Nx), (Ny, Nx-1), (Ny-1, Nx-1))
    #    for ishape, sel in itertools.product(shapes, sels):
    #        # Curved-sky
    #        lmax = 1000
    #        alm = curvedsky.rand_alm_healpy(ps_cmb,lmax=lmax,seed=seed)
    #        shape,iwcs = enmap.fullsky_geometry(shape=ishape)
    #        wcs = enmap.empty(shape,iwcs)[sel].wcs
    #        shape = (3,) + shape
    #        imap = curvedsky.alm2map(alm,enmap.empty(shape,wcs))
    #        oalm = curvedsky.map2alm(imap.copy(),lmax=lmax)
    #        rmap = curvedsky.alm2map(oalm,enmap.empty(shape,wcs),spin=0) # reference map

    #        imap2 = imap.copy()[sel]
    #        oalm = curvedsky.map2alm(imap2.copy(),lmax=lmax)
    #        rmap2 = curvedsky.alm2map(oalm,enmap.empty(shape,wcs),spin=0) # comparison map

    #        assert np.allclose(rmap[0],rmap2[0],atol=0,rtol=1e-7)
    #        assert np.allclose(rmap[1],rmap2[1],atol=0,rtol=1e-7)
    #        assert np.allclose(rmap[2],rmap2[2],atol=0,rtol=1e-7)
    #        
    #    Ny, Nx = 300, 300 
    #    shapes = ((Ny, Nx), (Ny-1, Nx), (Ny, Nx-1), (Ny-1, Nx-1))
    #    for ishape, sel in itertools.product(shapes, sels):
    #        # Flat-sky
    #        px = 2.0
    #        shape,iwcs = enmap.geometry(pos=(0,0),res=np.deg2rad(px/60.),shape=ishape)
    #        shape = (3,) + shape
    #        a = enmap.zeros(shape,iwcs)
    #        a = a[sel]
    #        wcs = a.wcs

    #        imap = enmap.rand_map(shape,wcs,ps_cmb,seed=seed)
    #        kmap = enmap.map2harm(imap.copy())
    #        rmap = enmap.harm2map(kmap,spin=0) # reference map

    #        imap = imap[sel]
    #        kmap = enmap.map2harm(imap.copy())
    #        rmap2 = enmap.harm2map(kmap,spin=0)[sel] # comparison map

    #        assert np.allclose(rmap[0],rmap2[0],atol=0,rtol=1e-7), f'{ishape=}, {sel=}'
    #        assert np.allclose(rmap[1],rmap2[1],atol=0,rtol=1e-7), f'{ishape=}, {sel=}'
    #        assert np.allclose(rmap[2],rmap2[2],atol=0,rtol=1e-7), f'{ishape=}, {sel=}'

    def test_plain_wcs(self):
        # Test area and box for a small Cartesian geometry
        shape,wcs = enmap.geometry(res=np.deg2rad(1./60.),shape=(600,600),pos=(0,0),proj='plain')
        box = np.rad2deg(enmap.box(shape,wcs))
        area = np.rad2deg(np.rad2deg(enmap.area(shape,wcs)))
        assert np.all(np.isclose(box,np.array([[-5,-5],[5,5]])))
        assert np.isclose(area,100.)

        # and for an artifical Cartesian geometry with area>4pi
        shape,wcs = enmap.geometry(res=np.deg2rad(10),shape=(100,100),pos=(0,0),proj='plain')
        box = np.rad2deg(enmap.box(shape,wcs))
        area = np.rad2deg(np.rad2deg(enmap.area(shape,wcs)))
        assert np.all(np.isclose(box,np.array([[-500,-500],[500,500]])))
        assert np.isclose(area,1000000)


    def test_pospix(self):
        # Posmap separable and non-separable on CAR
        for res in [6,12,24]:
            shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(res/60.),proj='car')
            posmap1 = enmap.posmap(shape,wcs)
            posmap2 = enmap.posmap(shape,wcs,separable=True)
            assert np.all(np.isclose(posmap1,posmap2))

        # Pixmap plain
        pres = 0.5
        shape,wcs = enmap.geometry(pos=(0,0),shape=(30,30),res=pres*u.degree,proj='plain')
        yp,xp = enmap.pixshapemap(shape,wcs)
        assert np.all(np.isclose(yp,pres*u.degree))
        assert np.all(np.isclose(xp,pres*u.degree))
        yp,xp = enmap.pixshape(shape,wcs)
        parea = enmap.pixsize(shape,wcs)
        assert np.isclose(parea,(pres*u.degree)**2)
        assert np.isclose(yp,pres*u.degree)
        assert np.isclose(xp,pres*u.degree)
        pmap = enmap.pixsizemap(shape,wcs)
        assert np.all(np.isclose(pmap,(pres*u.degree)**2))

        # Pixmap CAR
        pres = 0.1
        dec_cut = 89.5 # pixsizemap is not accurate near the poles currently
        shape,wcs = enmap.band_geometry(dec_cut=dec_cut*u.degree,res=pres*u.degree,proj='car')
        # Current slow and general but inaccurate near the poles implementation
        pmap = enmap.pixsizemap(shape,wcs)
        # Fast CAR-specific pixsizemap implementation
        dra, ddec = wcs.wcs.cdelt*u.degree
        dec = enmap.posmap([shape[-2],1],wcs)[0,:,0]
        area = np.abs(dra*(np.sin(np.minimum(np.pi/2.,dec+ddec/2))-np.sin(np.maximum(-np.pi/2.,dec-ddec/2))))
        Nx = shape[-1]
        pmap2 = enmap.ndmap(area[...,None].repeat(Nx,axis=-1),wcs)
        assert np.all(np.isclose(pmap,pmap2))
        

    def test_project_nn(self):
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(12/60.),proj='car')
        shape2,wcs2 = enmap.fullsky_geometry(res=np.deg2rad(6/60.),proj='car')
        shape3,wcs3 = enmap.fullsky_geometry(res=np.deg2rad(24/60.),proj='car')
        imap = enmap.ones(shape,wcs)
        omap2 = enmap.project(imap,shape2,wcs2,order=0,border='wrap')
        omap3 = enmap.project(imap,shape3,wcs3,order=0,border='wrap')
        assert np.all(np.isclose(omap2,1))
        assert np.all(np.isclose(omap3,1))

    def test_wcsunequal(self):
        shape1,wcs1 = enmap.geometry(pos=(0,0),shape=(100,100),res=1*u.arcmin,proj='car')
        shape1,wcs2 = enmap.geometry(pos=(0,0),shape=(100,100),res=1*u.arcmin,proj='cea')
        shape1,wcs3 = enmap.geometry(pos=(10,10),shape=(100,100),res=1*u.arcmin,proj='car')
        shape1,wcs4 = enmap.geometry(pos=(0,0),shape=(100,100),res=2*u.arcmin,proj='car')
        assert not(wcsutils.equal(wcs1,wcs2))
        assert not(wcsutils.equal(wcs1,wcs3))
        assert not(wcsutils.equal(wcs1,wcs4))
        
        
    def test_scale(self):
        # Test (with a plain geometry) that scale_geometry
        # will result in geometries with the same bounding box
        # but different area pixel
        pres = 0.5
        ufact = 2
        dfact = 0.5
        shape,wcs = enmap.geometry(pos=(0,0),shape=(30,30),res=pres*u.arcmin,proj='plain')
        ushape,uwcs = enmap.scale_geometry(shape,wcs,ufact)
        dshape,dwcs = enmap.scale_geometry(shape,wcs,dfact)
        box = enmap.box(shape,wcs)
        ubox = enmap.box(ushape,uwcs)
        dbox = enmap.box(dshape,dwcs)
        parea = enmap.pixsize(shape,wcs)
        uparea = enmap.pixsize(ushape,uwcs)
        dparea = enmap.pixsize(dshape,dwcs)
        assert np.all(np.isclose(box,ubox))
        assert np.all(np.isclose(box,dbox))
        assert np.isclose(parea/(ufact**2),uparea)
        assert np.isclose(parea/(dfact**2),dparea)

    def test_prepare_alm_mmax(self):
        # Check if mmax is correctly handled by prepare_alm.

        # Create lmax=mmax=3 alm array and corresponding alm_info.
        lmax = 3
        nalm = 10  # Triangular alm array of lmax=3 has 10 elements.
        alm_in = np.arange(nalm, dtype=np.complex128)
        ainfo_in = curvedsky.alm_info(
            lmax=3, mmax=3, nalm=nalm, stride=1, layout="triangular")

        # Case 1: provide only alm.
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=alm_in, ainfo=None)

        np.testing.assert_array_almost_equal(alm_out, alm_in)
        self.assertEqual(ainfo_out.lmax, ainfo_in.lmax)
        self.assertEqual(ainfo_out.mmax, ainfo_in.mmax)
        self.assertEqual(ainfo_out.nelem, ainfo_in.nelem)

        # Case 2: provide only alm_info.
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=None, ainfo=ainfo_in)
        # Expect zero array.
        np.testing.assert_array_almost_equal(alm_out, alm_in * 0)
        self.assertEqual(ainfo_out.lmax, ainfo_in.lmax)
        self.assertEqual(ainfo_out.mmax, ainfo_in.mmax)
        self.assertEqual(ainfo_out.nelem, ainfo_in.nelem)

        # Case 3: provide alm and alm_info
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=alm_in, ainfo=ainfo_in)

        np.testing.assert_array_almost_equal(alm_out, alm_in)
        self.assertEqual(ainfo_out.lmax, ainfo_in.lmax)
        self.assertEqual(ainfo_out.mmax, ainfo_in.mmax)
        self.assertEqual(ainfo_out.nelem, ainfo_in.nelem)

        # Case 4: provide only alm with lmax=3 and mmax=1.
        # This should currently fail.
        nalm = 7
        alm_in = np.arange(7, dtype=np.complex128)
        self.assertRaises(AssertionError, curvedsky.prepare_alm,
                          **dict(alm=alm_in, ainfo=None, lmax=lmax))

        # Case 5: provide only alm_info with lmax=3 and mmax=1.
        nalm = 7
        ainfo_in = curvedsky.alm_info(
            lmax=3, mmax=1, nalm=nalm, stride=1, layout="triangular")
        alm_exp = np.zeros(7, dtype=np.complex128)
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=None, ainfo=ainfo_in)

        np.testing.assert_array_almost_equal(alm_out, alm_exp)
        self.assertEqual(ainfo_out.lmax, ainfo_in.lmax)
        self.assertEqual(ainfo_out.mmax, ainfo_in.mmax)
        self.assertEqual(ainfo_out.nelem, ainfo_in.nelem)

        # Case 6: provide both alm and alm_info with lmax=3 and mmax=1.
        # This should be allowed.
        nalm = 7
        ainfo_in = curvedsky.alm_info(
            lmax=3, mmax=1, nalm=nalm, stride=1, layout="triangular")
        alm_in = np.arange(7, dtype=np.complex128)
        alm_out, ainfo_out = curvedsky.prepare_alm(alm=alm_in, ainfo=ainfo_in)

        np.testing.assert_array_almost_equal(alm_out, alm_in)
        self.assertEqual(ainfo_out.lmax, ainfo_in.lmax)
        self.assertEqual(ainfo_out.mmax, ainfo_in.mmax)
        self.assertEqual(ainfo_out.nelem, ainfo_in.nelem)


    def test_lens_alms(self):
        # We generate phi alms and convert them to kappa and back
        lmax = 100
        ps = np.zeros(lmax+1)
        ls = np.arange(lmax+1)
        ps[ls>=2] = 1./ls[ls>=2]
        phi_alm = curvedsky.rand_alm(ps,lmax=lmax)
        kappa_alm = lensing.phi_to_kappa(phi_alm)
        phi_alm2 = lensing.kappa_to_phi(kappa_alm)
        np.testing.assert_array_almost_equal(phi_alm, phi_alm2)

    def test_downgrade(self):
        shape,wcs = enmap.geometry(pos=(0,0),shape=(100,100),res=0.01)
        imap = enmap.ones(shape,wcs)
        for dfact in [None,1]:
            omap = enmap.downgrade(imap,dfact)
            np.testing.assert_equal(imap,omap)

        dfact = 2
        omap = enmap.downgrade(imap,dfact,op=np.sum)
        np.testing.assert_equal(omap,np.ones(enmap.scale_geometry(shape,wcs,1./dfact)[0])*4)

        
    def test_almxfl(self):
        # We try to filter alms of shape (nalms,) and (ncomp,nalms) with
        # a filter of shape (nells,)
        lmax = 30
        ells = np.arange(lmax+1)
        nells = ells.size
        for ncomp in range(4):
            if ncomp==0:
                fl = np.ones((nells,))
                ps = np.zeros((nells,))
                ps[ells>1] = 1./ells[ells>1]
            else:
                fl = np.ones((nells))
                ps = np.zeros((ncomp,ncomp,nells))
                for i in range(ncomp):
                    ps[i,i][ells>1] = 1./ells[ells>1]
            ialm = curvedsky.rand_alm(ps,lmax=lmax)
            oalm = curvedsky.almxfl(ialm,fl)
            np.testing.assert_array_almost_equal(ialm, oalm)

    def test_alm2map_2d_roundtrip(self):
        # Test curvedsky's alm2map/map2alm.
        lmax = 30
        ainfo = curvedsky.alm_info(lmax)

        nrings = lmax + 2
        nphi = 2 * lmax + 1
        shape, wcs = enmap.fullsky_geometry(shape=(nrings,nphi))

        # Test different input shapes and dtypes.
        # Case 1a: 1d double precision.
        spin = 0
        alm = np.zeros((ainfo.nelem), dtype=np.complex128)
        i   = ainfo.lm2ind(lmax,lmax)
        alm[i] = 1. + 1.j

        omap = enmap.zeros(shape, wcs, np.float64)
        curvedsky.alm2map(alm, omap, spin=spin)
        alm_out = curvedsky.map2alm(omap, spin=spin, ainfo=ainfo)

        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 1b: 1d single precision.
        spin = 0
        alm = np.zeros((ainfo.nelem), dtype=np.complex64)
        alm[i] = 1. + 1.j

        omap = enmap.zeros(shape, wcs, np.float32)
        curvedsky.alm2map(alm, omap, spin=spin)
        alm_out = curvedsky.map2alm(omap, spin=spin, ainfo=ainfo)
        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 2a: 2d double precision.
        spin = 1
        nspin = 2
        alm = np.zeros((nspin, ainfo.nelem), dtype=np.complex128)
        alm[0,i] = 1. + 1.j
        alm[1,i] = 2. - 2.j

        omap = enmap.zeros((nspin,)+shape, wcs, np.float64)
        curvedsky.alm2map(alm, omap, spin=spin)
        alm_out = curvedsky.map2alm(omap, spin=spin, ainfo=ainfo)
        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 2b: 2d single precision.
        spin = 1
        nspin = 2
        alm = np.zeros((nspin, ainfo.nelem), dtype=np.complex64)
        alm[0,i] = 1. + 1.j
        alm[1,i] = 2. - 2.j

        omap = enmap.zeros((nspin,)+shape, wcs, np.float32)
        curvedsky.alm2map(alm, omap, spin=spin)
        alm_out = curvedsky.map2alm(omap, spin=spin, ainfo=ainfo)
        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 3a: 3d double precision.
        spin = 1
        nspin = 2
        ntrans = 3
        alm = np.zeros((ntrans, nspin, ainfo.nelem), dtype=np.complex128)
        alm[0,0,i] = 1. + 1.j
        alm[0,1,i] = 2. - 2.j
        alm[1,0,i] = 3. + 3.j
        alm[1,1,i] = 4. - 4.j
        alm[2,0,i] = 5. + 5.j
        alm[2,1,i] = 6. - 6.j

        omap = enmap.zeros((ntrans,nspin)+shape, wcs, np.float64)
        curvedsky.alm2map(alm, omap, spin=spin)
        alm_out = curvedsky.map2alm(omap, spin=spin, ainfo=ainfo)
        np.testing.assert_array_almost_equal(alm_out, alm)

        # Case 3b: 3d single precision.
        spin = 1
        nspin = 2
        ntrans = 3
        alm = np.zeros((ntrans, nspin, ainfo.nelem), dtype=np.complex64)
        alm[0,0,i] = 1. + 1.j
        alm[0,1,i] = 2. - 2.j
        alm[1,0,i] = 3. + 3.j
        alm[1,1,i] = 4. - 4.j
        alm[2,0,i] = 5. + 5.j
        alm[2,1,i] = 6. - 6.j

        omap = enmap.zeros((ntrans,nspin)+shape, wcs, np.float32)
        curvedsky.alm2map(alm, omap, spin=spin)
        alm_out = curvedsky.map2alm(omap, spin=spin, ainfo=ainfo)
        np.testing.assert_array_almost_equal(alm_out, alm)

    def test_alm2map_healpix_roundtrip(self):
        # Test curvedsky's alm2map/map2alm.
        nside = 2
        lmax  = nside*2
        nside = lmax//2
        ainfo = curvedsky.alm_info(lmax)
        npix  = 12*nside**2
        # 7 iterations needed to reach 6 digits of
        # precision. This is more than the 3 default
        # in healpy and the 0 default in pixell
        niter = 7

        for dtype in [np.float64, np.float32]:
            ctype = utils.complex_dtype(dtype)

            # Case 1: 1d
            spin = 0
            alm = np.zeros((ainfo.nelem), dtype=ctype)
            i   = ainfo.lm2ind(lmax,lmax)
            alm[i] = 1. + 1.j

            omap = np.zeros(npix, dtype)
            curvedsky.alm2map_healpix(alm, omap, spin=spin)
            alm_out = curvedsky.map2alm_healpix(omap, spin=spin, ainfo=ainfo, niter=niter)

            np.testing.assert_array_almost_equal(alm_out, alm)

            # Case 2: 2d
            spin = 1
            nspin = 2
            alm = np.zeros((nspin, ainfo.nelem), dtype=ctype)
            alm[0,i] = 1. + 1.j
            alm[1,i] = 2. - 2.j

            omap = np.zeros((nspin,npix), dtype)
            curvedsky.alm2map_healpix(alm, omap, spin=spin)
            alm_out = curvedsky.map2alm_healpix(omap, spin=spin, ainfo=ainfo, niter=niter)
            np.testing.assert_array_almost_equal(alm_out, alm)

            # Case 3: 3d
            spin = 1
            nspin = 2
            ntrans = 3
            alm = np.zeros((ntrans, nspin, ainfo.nelem), dtype=ctype)
            alm[0,0,i] = 1. + 1.j
            alm[0,1,i] = 2. - 2.j
            alm[1,0,i] = 3. + 3.j
            alm[1,1,i] = 4. - 4.j
            alm[2,0,i] = 5. + 5.j
            alm[2,1,i] = 6. - 6.j

            omap = np.zeros((ntrans,nspin,npix), dtype)
            curvedsky.alm2map_healpix(alm, omap, spin=spin)
            alm_out = curvedsky.map2alm_healpix(omap, spin=spin, ainfo=ainfo, niter=niter)
            np.testing.assert_array_almost_equal(alm_out, alm)

    # MM: Re-enabled 09/17/2024
    # --Disabled for now because the version of ducc currently on pypi
    # has an adjointness bug. It's fixed in the ducc git repo.--
    def test_adjointness(self):
       # This tests if alm2map_adjoint is the adjoint of alm2map,
       # and if map2alm_adjoint is the adjoint of map2alm.
       # (This doesn't test if they're correct, just that they're
       # consistent with each other). This test is a bit slow, taking
       # 5 s or so. It would be much faster if we dropped the ncomp=3 case.
       for dtype in [np.float32, np.float64]:
           for ncomp in [1,3]:
               # Define our geometries
               geos = []
               res  = 30*utils.degree
               shape, wcs = enmap.fullsky_geometry(res=res, variant="fejer1")
               geos.append(("fullsky_fejer1", shape, wcs))

               shape, wcs = enmap.fullsky_geometry(res=res, variant="cc")
               geos.append(("fullsky_cc", shape, wcs))
               lmax = shape[-2]-2

               shape, wcs = enmap.Geometry(shape, wcs)[3:-3,3:-3]
               geos.append(("patch_cc", shape, wcs))

               wcs = wcs.deepcopy()
               wcs.wcs.crpix += 0.123
               geos.append(("patch_gen_cyl", shape, wcs))

               shape, wcs = enmap.geometry(np.array([[-45,45],[45,-45]])*utils.degree, res=res, proj="tan")
               geos.append(("patch_tan", shape, wcs))

               for gi, (name, shape, wcs) in enumerate(geos):
                   mat1  = alm_bash(curvedsky.alm2map,         shape, wcs, ncomp, lmax, dtype)
                   mat2  = map_bash(curvedsky.alm2map_adjoint, shape, wcs, ncomp, lmax, dtype)
                   np.testing.assert_array_almost_equal(mat1, mat2)
                   mat1 = map_bash(curvedsky.map2alm,         shape, wcs, ncomp, lmax, dtype)
                   mat2 = alm_bash(curvedsky.map2alm_adjoint, shape, wcs, ncomp, lmax, dtype)
                   np.testing.assert_array_almost_equal(mat1, mat2)


    #def test_sharp_alm2map_der1(self):
    #            
    #    # Test the wrapper around libsharps alm2map_der1.
    #    lmax = 3
    #    ainfo = sharp.alm_info(lmax)

    #    nrings = lmax + 1
    #    nphi = 2 * lmax + 1
    #    minfo = sharp.map_info_gauss_legendre(nrings, nphi)

    #    sht = sharp.sht(minfo, ainfo)

    #    # Test different input shapes and dtypes.
    #    # Case 1a: 1d double precision.
    #    alm = np.zeros((ainfo.nelem), dtype=np.complex128)
    #    alm[4] = 1. + 1.j

    #    omap = sht.alm2map_der1(alm)
    #    # Compare to expected value by doing spin 1 transform
    #    # on sqrt(ell (ell + 1)) alm.
    #    alm_spin = np.zeros((2, ainfo.nelem), dtype=np.complex128)
    #    alm_spin[0] = alm * np.sqrt(2)
    #    omap_exp = sht.alm2map(alm_spin, spin=1)

    #    np.testing.assert_array_almost_equal(omap, omap_exp)

    #    # Case 1b: 1d single precision.
    #    alm = np.zeros((ainfo.nelem), dtype=np.complex64)
    #    alm[4] = 1. + 1.j

    #    omap = sht.alm2map_der1(alm)
    #    # Compare to expected value by doing spin 1 transform
    #    # on sqrt(ell (ell + 1)) alm.
    #    alm_spin = np.zeros((2, ainfo.nelem), dtype=np.complex64)
    #    alm_spin[0] = alm * np.sqrt(2)
    #    omap_exp = sht.alm2map(alm_spin, spin=1)

    #    np.testing.assert_array_almost_equal(omap, omap_exp)

    #    # Case 2a: 2d double precision.
    #    ntrans = 3
    #    alm = np.zeros((ntrans, ainfo.nelem), dtype=np.complex128)
    #    alm[0,4] = 1. + 1.j
    #    alm[1,4] = 2. + 2.j
    #    alm[2,4] = 3. + 3.j

    #    omap = sht.alm2map_der1(alm)
    #    # Compare to expected value by doing spin 1 transform
    #    # on sqrt(ell (ell + 1)) alm.
    #    alm_spin = np.zeros((ntrans, 2, ainfo.nelem), dtype=np.complex128)
    #    alm_spin[0,0] = alm[0] * np.sqrt(2)
    #    alm_spin[1,0] = alm[1] * np.sqrt(2)
    #    alm_spin[2,0] = alm[2] * np.sqrt(2)
    #    omap_exp = sht.alm2map(alm_spin, spin=1)

    #    np.testing.assert_array_almost_equal(omap, omap_exp)

    #    # Case 2b: 2d single precision.
    #    ntrans = 3
    #    alm = np.zeros((ntrans, ainfo.nelem), dtype=np.complex64)
    #    alm[0,4] = 1. + 1.j
    #    alm[1,4] = 2. + 2.j
    #    alm[2,4] = 3. + 3.j

    #    omap = sht.alm2map_der1(alm)
    #    # Compare to expected value by doing spin 1 transform
    #    # on sqrt(ell (ell + 1)) alm.
    #    alm_spin = np.zeros((ntrans, 2, ainfo.nelem), dtype=np.complex64)
    #    alm_spin[0,0] = alm[0] * np.sqrt(2)
    #    alm_spin[1,0] = alm[1] * np.sqrt(2)
    #    alm_spin[2,0] = alm[2] * np.sqrt(2)
    #    omap_exp = sht.alm2map(alm_spin, spin=1)

    #    np.testing.assert_array_almost_equal(omap, omap_exp)

    def test_thumbnails(self):
        print("Testing thumbnails...")

        # Make a geometry far away from the equator
        dec_min = 70 * u.degree
        dec_max = 80 * u.degree
        res = 0.5 * u.arcmin
        shape,wcs = enmap.band_geometry((dec_min,dec_max),res=res)

        # Create a set of point source positions separated by
        # 2 degrees but with 1 column wrapping around the RA
        # direction
        width = 120 * u.arcmin
        Ny = int((dec_max-dec_min)/(width))
        Nx = int((2*np.pi/(width)))
        pys = np.linspace(0,shape[0],Ny)[1:-1]
        pxs = np.linspace(0,shape[1],Nx)[:-1]
        Ny = len(pys)
        Nx = len(pxs)
        xx,yy = np.meshgrid(pxs,pys)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        ps = np.vstack((yy,xx))
        decs,ras = enmap.pix2sky(shape,wcs,ps)
        
        # Simulate these sources with unit peak value and 2.5 arcmin FWHM
        N = ps.shape[1]
        srcs = np.zeros((N,3))
        srcs[:,0] = decs
        srcs[:,1] = ras
        srcs[:,2] = ras*0 + 1
        sigma = 2.5 * u.fwhm * u.arcmin
        omap = pointsrcs.sim_srcs(shape,wcs,srcs,beam=sigma)

        # Reproject thumbnails centered on the sources
        # with gnomonic/tangent projection
        proj = "tan"
        r = 10*u.arcmin
        ret = reproject.thumbnails(omap, srcs[:,:2], r=r, res=res, proj=proj,
            apod=2*u.arcmin, order=3, oversample=4, pixwin=False)

        # Create a reference source at the equator to compare this against
        ishape,iwcs = enmap.geometry(shape=ret.shape,res=res,pos=(0,0),proj=proj)
        imodrmap = enmap.modrmap(ishape,iwcs)
        model = np.exp(-imodrmap**2./2./sigma**2.)

        # Make sure all thumbnails agree with the reference at the
        # sub-percent level
        for i in range(ret.shape[0]):
            diff = ret[i] - model
            assert np.all(np.isclose(diff,0,atol=1e-3))

    def test_tilemap(self):
        shape, wcs = enmap.fullsky_geometry(30*utils.degree, variant="CC")
        assert shape == (7,12)
        geo  = tilemap.geometry((3,)+shape, wcs, tile_shape=(2,2))
        assert len(geo.active) == 0
        assert np.all(geo.lookup<0)
        assert geo.ntile   == 24
        assert geo.nactive == 0
        assert geo.tile_shape == (2,2)
        assert geo.grid_shape == (4,6)
        assert tuple(geo.tile_shapes[ 0]) == (2,2)
        assert tuple(geo.tile_shapes[ 5]) == (2,2)
        assert tuple(geo.tile_shapes[18]) == (1,2)
        assert tuple(geo.tile_shapes[23]) == (1,2)
        assert geo.ind2grid(7) == (1,1)
        assert geo.grid2ind(1,1) == 7
        geo = geo.copy(active=[1])
        assert geo.nactive == 1
        assert np.sum(geo.lookup>=0) == 1
        assert geo.active[0] == 1
        assert geo.lookup[1] == 0
        geo2 = geo.copy(active=[0,1,2])
        assert geo.nactive == 1
        assert geo2.nactive == 3
        assert geo.compatible(geo) == 2
        assert geo.compatible(geo2) == 1
        geo3 = tilemap.geometry((3,)+shape, wcs, tile_shape=(2,3))
        assert geo.compatible(geo3) == 0
        del geo2, geo3
        m1  = tilemap.zeros(geo.copy(active=[1,2]))
        m2  = tilemap.zeros(geo.copy(active=[2,3,4]))
        m3  = tilemap.zeros(geo.copy(active=[2]))
        for a, i in enumerate(m1.geometry.active): m1.active_tiles[a] = i
        for a, i in enumerate(m2.geometry.active): m2.active_tiles[a] = i*10
        for a, i in enumerate(m3.geometry.active): m3.active_tiles[a] = i*100
        assert m1[0,0] == 1
        assert np.all(m1.tiles[1] == m1.active_tiles[0])
        m12 = m1+m2
        m21 = m2+m1
        assert(m12.nactive == 4)
        assert(m21.nactive == 4)
        assert(np.all(m12.tiles[1] == 1))
        assert(np.all(m21.tiles[1] == 1))
        assert(np.all(m12.tiles[2] == 22))
        assert(np.all(m21.tiles[2] == 22))
        assert(sorted(m12.geometry.active)==sorted(m21.geometry.active))
        m1 += m3
        assert np.all(m1.tiles[2] == 202)
        with self.assertRaises(ValueError): m3 += m1
        m1[:] = 0
        m1c   = np.cos(m1)
        assert m1c.geometry.nactive == 2
        assert np.allclose(m1c, 1)

    def test_interpol_1d(self):
        dtype = np.float64
        n     = 10
        data  = np.sin(2*np.pi*3*np.arange(n)/n)
        inds  = np.array([0.0,0.1,0.51,0.9,1.0])
        # Nearest neighbor interpolation
        vals  = utils.interpol(data, inds[None], mode="spline", order=0)
        assert np.allclose(vals, data[utils.nint(inds)])
        # Linear interpolation. This check assumes that inds is [0:1]!
        vals  = utils.interpol(data, inds[None], mode="spline", order=1)
        assert np.allclose(vals, data[0]*(1-inds)+data[1]*inds)
        # Cubic spline interpolation. No simple formula here, so hardcode
        vals  = utils.interpol(data, inds[None], mode="spline", order=3)
        assert np.allclose(vals, [-1.39824833112681842e-12, 1.16478779103952171e-01, 6.66141591529904153e-01, 9.62884886419913988e-01, 9.51056516295153753e-01])
        # Nufft interpolation
        vals  = utils.interpol(data, inds[None], mode="fourier")
        targ  = np.sin(2*np.pi*3*inds/n)
        assert np.allclose(vals, targ)

    def test_interpol_map(self):
        # Make a small map to interpolate. We will simply use the angular distance
        # from the center, which avoids any non-periodicity issues
        shape, wcs = enmap.geometry([[-1,1],[1,-1]], res=0.1, deg=True, proj="car")
        # Make test cases where the exact answer is known. Bilinear is easy for a linear field,
        # and NN is easy anywayre
        dtype = np.float64
        ypix  = np.arange(shape[-2], dtype=dtype)
        xpix  = np.arange(shape[-1], dtype=dtype)
        opix  = np.array([[10,10.2],[10,12.7]])
        # NN and bilinear
        def f(y,x): return 2*y-x
        data  = enmap.enmap(f(ypix[:,None],xpix[None,:]),wcs)
        vals  = data.at(opix, unit="pix", mode="spline", order=0)
        targ  = data[tuple(utils.nint(opix))]
        assert np.allclose(vals, targ)
        vals  = data.at(opix, unit="pix", mode="spline", order=1)
        targ  = f(*opix)
        assert np.allclose(vals, targ)
        # Bicubic is exact up to 3rd order, but only if we ignore
        # boundary conditions. I'm not sure how to calculate the
        # expected answer here, so just hardcode it
        def f(y,x): return y**3+2*y**2+3*y-2*x**3-3*x**2-4*x+x*y
        data  = enmap.enmap(f(ypix[:,None],xpix[None,:]),wcs)
        vals  = data.at(opix, unit="pix", mode="spline", order=3)
        targ  = np.array([-1.01000000000000000e+03, -3.20214384455599429e+03])
        assert np.allclose(vals, targ)
        # Fourier is simple. All fourier modes up to Nyquist are exact.
        # Nufft adds some inaccuracy, but it's tiny
        def f(y,x): return np.sin(2*np.pi*3*y/shape[-2])+np.cos(2*np.pi*5*x/shape[-1])
        data  = enmap.enmap(f(ypix[:,None],xpix[None,:]),wcs)
        vals  = data.at(opix, unit="pix", mode="fourier")
        targ  = f(*opix)
        assert np.allclose(vals, targ)

    def test_interpol_map_Nd(self):
        # Test that multidimensional interpolation works
        shape, wcs = enmap.geometry([[-1,1],[1,-1]], res=0.1, deg=True, proj="car")
        dtype = np.float64
        ypix  = np.arange(shape[-2], dtype=dtype)
        xpix  = np.arange(shape[-1], dtype=dtype)
        def f(y,x): return np.sin(2*np.pi*3*y/shape[-2])+np.cos(2*np.pi*5*x/shape[-1])
        data  = (1+np.arange(12)).reshape(3,4)[:,:,None,None]*enmap.enmap(f(ypix[:,None],xpix[None,:]),wcs)
        opix  = np.array([[10,10.2],[10,12.7]])
        vals  = data.at(opix, unit="pix")
        assert vals.shape == (3,4)+opix.shape[1:]
        targ  = vals*0
        for I in utils.nditer(data.shape[:-2]):
            targ[I] = data[I].at(opix, unit="pix")
        assert np.allclose(vals, targ)

    def test_interpolator(self):
        # Test that lookups using an interpolator work, and are fast.
        # Use a decently large array to be sure we can measure the
        # speed difference.
        shape = (1000,1000)
        dtype = np.float64
        ypix  = np.arange(shape[-2], dtype=dtype)
        xpix  = np.arange(shape[-1], dtype=dtype)
        opix  = np.array([[10,10.2],[10,12.7]])
        def f(y,x): return np.sin(2*np.pi*3*y/shape[-2])+np.cos(2*np.pi*5*x/shape[-1])
        data  = f(ypix[:,None],xpix)
        cases = [("spline",3),("fourier",None)]
        for ci, (mode,order) in enumerate(cases):
            t1 = time.time()
            vals_raw = utils.interpol(data, opix, mode=mode, order=order)
            t2 = time.time()
            ip = utils.interpolator(data, mode=mode, order=order)
            t3 = time.time()
            vals_ip  = ip(opix)
            t4 = time.time()
            assert np.allclose(vals_ip, vals_raw)
            time_full  = t2-t1
            time_eval  = t4-t3
            # Should be much faster than 10%, but timing is unreliable, especially in github actions
            assert time_eval < time_full / 10
            # Also test that we can pass the interpolator as an argument
            vals_ip2 = utils.interpol(data, opix, mode=mode, order=order, ip=ip)
            assert np.allclose(vals_ip2, vals_raw)
