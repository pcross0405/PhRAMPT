import multiprocessing as mp
import os    

#----------------------------------------------------------------------------------------------------------------#

# method that does the the calculating in parallel
def parallel_Calc(self):

    # get number of available processors
    procs = len(os.sched_getaffinity(0)) - 1
    if procs > self._natoms:
        import warnings
        warnings.warn('''There is no benefit to using more processors than number of atoms.\n
                        Number of parallel jobs will be set to number of atoms.
                        ''')
        
        procs = self._natoms

    

    # create pool
    pool = mp.Pool(processes=procs)