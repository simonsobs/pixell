"""Utilities for making mpi use safer and easier."""
from __future__ import print_function
import sys, os, traceback, copy, numpy as np
from .mpiutils import FakeCommunicator, FAKE_WORLD

COMM_WORLD = FAKE_WORLD
COMM_SELF  = FAKE_WORLD
disabled   = True

# Allow us to disable real mpi, creating only a simple placeholder object
# that will work for trivially parallelizable code run with only 1 task
try:
    if not("DISABLE_MPI" in os.environ and os.environ["DISABLE_MPI"].lower() in ["true","1"]):
        # We transparently pass through all the mpi4py.MPI stuff, but also add a cleanup
        # hook. On scinet I found that uncaught exceptions did not cause mpi to abort,
        # leading to thousands of wasted CPU hours. That may have been system-specific,
        # though. Perhaps this isn't necessary in general.
        from mpi4py.MPI import *
        #def cleanup(type, value, traceback):
        #    sys.__excepthook__(type, value, traceback)
        #    COMM_WORLD.Abort(1)
        #sys.excepthook = cleanup
        disabled = False
except:
    pass

class itemhack:
    @staticmethod
    def Alltoallv(sendbuf, sendn, sendoff, recvbuf, recvn, recvoff, comm, bsize=1):
        """Version of Alltoallv that does the transfer in terms of items
        with size bsize of the original data type (or bsize*dtype.itemsize
        in terms of bytes). The only reason to do this is to work around
        the signed 32-bit limit on counts and offsets in MPI before
        version 4."""
        # Calculate blocked counts and offsets
        bsendn   = sendn   // bsize
        brecvn   = recvn   // bsize
        bsendoff = sendoff // bsize
        brecvoff = recvoff // bsize
        assert np.all(bsendn*bsize==sendn), "sendn must be a multiple of bsize"
        assert np.all(brecvn*bsize==recvn), "recvn must be a multiple of bsize"
        assert np.all(bsendoff*bsize==sendoff), "sendoff must be a multiple of bsize"
        assert np.all(brecvoff*bsize==recvoff), "recvoff must be a multiple of bsize"
        # Define new mpi data type for the transfer
        mtype = BYTE.Create_contiguous(sendbuf.itemsize*bsize)
        mtype.Commit()
        comm.Alltoallv(
            (sendbuf, (bsendn,bsendoff), mtype),
            (recvbuf, (brecvn,brecvoff), mtype),
        )
        mtype.Free()
