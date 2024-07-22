import multiprocessing as mp
from collections import Counter
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
def parallel_Calc(in_file, proc_num, proc_list, methodname, klist, hkl, resolution):

    # identify which displacement method is calling this function
    if methodname == 'pairwise':
        new_calc = pt.Pairwise(in_file)
    
    elif methodname == 'general':
        new_calc = pt.General(in_file)

    else:
        raise SystemExit('Parallel calculation is unable to identify displacement method, please report this!')
    
    # reassign class attributes to new class
    new_calc.klist = klist
    new_calc.hkl = hkl
    new_calc.resolution = resolution
    
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

    # calculate dynamical matrices for assigned atoms
    new_calc.CDM()

    # close lammps after calculation finishes
    new_calc._lmp.close()

    return new_calc.d_matrices

#----------------------------------------------------------------------------------------------------------------#

# method for creating parallel pools

def make_parallel(in_file, natoms, methodname, klist, hkl, resolution):

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
    arg1 = [in_file for _ in range(procs)]
    arg2 = [*range(procs)]
    proc_list = assign_atoms(natoms, procs)
    arg3 = [proc_list for _ in range(procs)]
    arg4 = [methodname for _ in range(procs)]
    arg5 = [klist for _ in range(procs)]
    arg6 = [hkl for _ in range(procs)]
    arg7 = [resolution for _ in range(procs)]
    args = [*zip(arg1, arg2, arg3, arg4, arg5, arg6, arg7)]

    # submit arguments to pool jobs
    calc_output = pool.starmap(parallel_Calc, iterable = args)

    # sum output from parallel calc with collections counter
    total_output = Counter()
    for d_matrix_dict in calc_output:
        total_output.update(d_matrix_dict)

    # close pool before returning parallel calc
    pool.close()
    return dict(total_output)