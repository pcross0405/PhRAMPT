import multiprocessing as mp
from functools import partial
import os
from . import phrampt_tools as pt

#----------------------------------------------------------------------------------------------------------------#

# method for deciding how to distribute atoms to processors
def assign_atoms(natoms, procs):

    # find number of atoms per processor, always rounding up
    atoms_per_proc = round(natoms/procs + 0.5)

    # create list of atoms per processor
    proc_list = [atoms_per_proc for _ in range(procs)]

    # since we round up, likely to have more atoms per processor than atoms in simulation
    # remove 1 atom from each processor necessary so that sum(proc_list) == natoms
    for i in range(sum(proc_list) - natoms):
        proc_list[i] = proc_list[i] - 1

    return proc_list

#----------------------------------------------------------------------------------------------------------------#

# method that does the the calculating in parallel
def parallel_Calc(in_file, natoms, procs, make_supercell, methodname, proc_num):

    # figure out which atoms belong to which processor
    proc_list = assign_atoms(natoms, procs)

    # identify which displacement method is calling this function
    if methodname == 'pairwise':
        new_calc = pt.Pairwise(in_file)
    
    elif methodname == 'general':
        new_calc = pt.General(in_file)

    else:
        raise SystemExit('Parallel calculation is unable to identify displacement method, please report this!')
    
    # choose to whether or not a supercell is made
    if make_supercell == True:
        new_calc.MakeSupercell()

    else:
        new_calc.MakeCell()

    # find total number of atoms already assigned to processors
    total_atoms = 0
    for i in range(proc_num):
        total_atoms += proc_list[i]

    # temporary dictionary for reassigning center_info dictionary
    center_info_temp = {}
    for i, j in enumerate(new_calc._center_info):
        if i in range(total_atoms, total_atoms + proc_list[proc_num]):
            center_info_temp[j] = new_calc._center_info[j]
    
    # reassign center_info dictionary
    new_calc._center_info = center_info_temp

    # calculate force constants for assigned atoms
    new_calc.CFCM()

    # close lammps after calculation finishes
    new_calc._lmp.close()

    return [new_calc._force_constants, new_calc._inter_dists]

#----------------------------------------------------------------------------------------------------------------#

# method for creating parallel pools

def make_parallel(in_file, natoms, make_supercell, methodname):

    # get number of available processors
    # subtract 1 since 1 processor runs main python script
    procs = len(os.sched_getaffinity(0)) - 1

    # exit if using more processors than atoms
    if procs > natoms:
        raise Warning('''Using more processors than atoms does not increase speed of calculation.\n
                    Rerun with number of processors = number of atoms + 1.''')
    
    # if processor count is 1 or less then parallel calc will not work
    elif procs == 0 or procs == 1:
        raise SystemExit('''Number of available must be greater than one to benefit from parallelization.\n
                        Rerun with more processors or run as a serial calculation.''')

    # create pool
    pool = mp.get_context('spawn').Pool(processes=procs)

    # setup arguments
    iterable = [*range(procs)]
    func = partial(parallel_Calc, in_file, natoms, procs, make_supercell, methodname)

    # submit arguments to pool jobs
    pool_output = pool.imap(func, iterable)

    # close pool
    pool.close()
    pool.join()

    # pool will return list of force constants (fc) and list of interatomic distances (iad)
    pool_output = list(pool_output)
    fc_output = [fc_dict[0] for fc_dict in pool_output]
    iad_output = [iad_dict[1] for iad_dict in pool_output]

    # concatenate ouputs into single force constants and interatomic distances dictionaries
    force_constants = {k: v for fc in fc_output for k, v in fc.items()}
    inter_dists = {k: v for iad in iad_output for k, v in iad.items()}

    # free up some memory
    del pool_output
    del fc_output
    del iad_output

    return force_constants, inter_dists