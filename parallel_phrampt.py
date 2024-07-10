import multiprocessing as mp
import os
from lammps import lammps

#----------------------------------------------------------------------------------------------------------------#

# method that does the the calculating in parallel
def parallel_Calc(in_file, proc_num):

    lmp = lammps()
    lmp.file(in_file)
    natoms = lmp.get_natoms()

#----------------------------------------------------------------------------------------------------------------#

# method for creating parallel pools

def make_parallel(in_file, natoms):

    # get number of available processors
    procs = len(os.sched_getaffinity(0)) - 1

    # if using more processors than atoms, change processor count to be number of atoms
    if procs > natoms:
        import warnings
        warnings.warn('''There is no benefit to using more processors than number of atoms.\n
                        Number of parallel jobs will be set to number of atoms.
                        ''')
        
        procs = natoms

    # create pool
    pool = mp.Pool(processes=procs)

    # setup arguments
    arg1 = [in_file for i in range(procs)]
    arg2 = [*range(procs)]
    args = [*zip(arg1, arg2)]

    # submit arguments to pool jobs
    calc_output = pool.starmap(parallel_Calc, iterable = args)