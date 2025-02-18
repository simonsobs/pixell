import numpy as np, copy

def _unbuf(bufspec):
    return bufspec[0] if isinstance(bufspec, tuple) else np.asarray(bufspec)

class FakeCommunicator:
    def __init__(self):
        self.size = 1
        self.rank = 0
    def Allreduce(self, sendbuf, recvbuf, op=lambda a,b:a+b):
        _unbuf(recvbuf)[()] = _unbuf(sendbuf)
    def Allgather(self, sendbuf, recvbuf):
        _unbuf(recvbuf)[0] = _unbuf(sendbuf)
    def Allgatherv(self, sendbuf, redvbuf):
        _unbuf(recvbuf)[()] = _unbuf(sendbuf)
    def Alltoallv(self, sendbuf, recvbuf):
        _unbuf(recvbuf)[()] = _unbuf(sendbuf)
    def Barrier(self): pass
    def allreduce(self, sendobj, op=lambda a,b:a+b):
        return copy.deepcopy(sendobj)
    def allgather(self, sendobj, op=lambda a,b:a+b):
        return [copy.deepcopy(sendobj)]
    def allgatherv(self, sendobj, op=lambda a,b:a+b):
        return [copy.deepcopy(sendobj)]

FAKE_WORLD = FakeCommunicator()
