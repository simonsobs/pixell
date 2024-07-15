import unittest
import tempfile
import os

import numpy as np
import h5py
from pixell import enmap, utils

class IOTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _tempfile(self, basename):
        return os.path.join(self.tempdir.name, basename)

    def test_100_hdf(self):
        box = np.array([[-5,10],[5,-10]]) * utils.degree
        shape, wcs = enmap.geometry(
            pos=box, res=0.5 * utils.degree, proj='car')
        c_shape = (100, 200)

        super_filename = self._tempfile('test-super.h5')
        with h5py.File(super_filename, 'w') as super_out:
            filenames = []
            for name, super_shape in [
                    ('flat', ()),
                    ('classic', (3,)),
                    ('monster', (10, 23)),
            ]:           
                my_map = enmap.zeros(super_shape + c_shape,
                                     wcs=wcs)
                filenames.append(self._tempfile(f'test-{name}.h5'))
                enmap.write_hdf(filenames[-1], my_map)
                enmap.write_hdf(super_out, my_map, address=f'/{name}')

        # Single file read back...
        for f in filenames:
            map_in = enmap.read_hdf(f)
            print(f'File {f} contains a map with shape={map_in.shape}, wcs={map_in.wcs}')

        # Super file, with filename
        map_in = enmap.read_hdf(super_filename, address='monster')

        # Super file, with contextmanager
        with h5py.File(super_filename, 'r') as fin:
            for k in fin.keys():
                print(f'Found group {k} ...')
                map_in = enmap.read_hdf(fin, address=k)
                print(f'   {map_in.shape}, {map_in.wcs}')

if __name__ == '__main__':
    unittest.main()
