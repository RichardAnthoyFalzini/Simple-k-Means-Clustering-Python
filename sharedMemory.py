from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

size = 1000
mpi_worker = comm.Get_size()

itemsize = MPI.DOUBLE.Get_size()
if comm.Get_rank() == 0:
    nbytes = size * itemsize
else:
    nbytes = 0
win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

buf, itemsize = win.Shared_query(0)
assert itemsize == MPI.DOUBLE.Get_size()
buf = np.array(buf, dtype='B', copy=False)
ary = np.ndarray(buffer=buf, dtype='d', shape=(size,))

chunk_size = size/mpi_worker

ary[comm.rank*chunk_size:comm.rank*(chunk_size+1)] = comm.rank

comm.Barrier()
if comm.rank == 0:
    print(ary[:])
