"""
Core benchmark script to be evaluated with various threading
configurations. See benchmark_pixell.py for the actual benchmark
that varies the number of threads.
"""

from pixell import curvedsky, enmap, utils
import time
import numpy as np


def main():
    np.random.seed(100)
    shape, wcs = enmap.fullsky_geometry(res=12.0 * utils.arcmin)
    imap = enmap.enmap(np.random.random(shape), wcs)

    nsims = 40
    lmax = int(6000 * (2.0 / 16.0))

    t0 = time.time()
    for i in range(nsims):
        alm = curvedsky.map2alm(imap, lmax=lmax)
        curvedsky.alm2map(alm, enmap.empty(shape, wcs))
    t1 = time.time()

    total = t1 - t0
    print(f"{total:.4f} seconds.")


if __name__ == "__main__":
    main()
