#----------------------------------------------------------------------------------------------------------------#
#---------------- Tool for comparing machine-learned potential phonon plots to DFT phonon plots -----------------#
#----------------------------------------------------------------------------------------------------------------#

import numpy as np
from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, LMP_STYLE_LOCAL

#----------------------------------------------------------------------------------------------------------------#

# class for managing phonon calculation within LAMMPS
class phonon_manager:

    def __init__(self, in_file):

        # create LAMMPS object
        self.lmp = lammps()

        # load LAMMPS in_file
        self.lmp.file(in_file)

        # use LAMMPS run command to initialize energy
        self.lmp.command('run 0')

        # replicate cell into 3x3x3 supercell
        # identify centeral atoms that make up cell of interest
        # get box parameters for finding center atoms
        self.lat_params = self.lmp.extract_box()
        self.natoms = self.lmp.get_natoms()
        self.CenterAtomCheck(1.99999)

        # create dictionaries for tracking atom information
        self.all_info = {}
        self.center_info = {}
        self.d_matrices = {}
        self.frequencies = {}

        # list of reciprocal space points to sample
        # can use phonon_manager.klist = 'all' to sample entire brillouin zone
        # h, k, and l define how finely brillouin zone is sampled when using 'all'
        # if h, k, or l remain None the fineness of that dimesion is automatically determined
        # see KPath method for more details on fineness determination
        self.klist = 'all'
        self.h = None
        self.k = None
        self.l = None

        # initialize variables needed for phonon calc
        # displacement is amount atoms are displaced in angstroms
        # resolution is the number of points interpolated between reciprocal space points
        # conversion is a conversion factor for plotting frequencies in units of THz
        self.resolution = 50

        units = self.lmp.extract_global('units')
        if units == 'metal':
            self.displacement = 0.015
            self.conversion = 1.6021773/6.022142*10**3

        elif units == 'real':
            self.displacement = 0.015
            self.conversion = 1

        elif units == 'si':
            self.displacement = 1.5*10**(-12)
            self.conversion = 1

        else:
            raise SystemExit(f'The {units} units style is not supported. Please change to metal, real, or si units.')

        # fetch information from all atoms and centeral atoms
        self.lmp.commands_string('''
            compute CenterInfo CenterAtoms property/atom id type mass
            compute AllInfo all property/atom id type mass
            ''')
        
        # extract compute information from LAMMPS into numpy arrays
        center_array = self.lmp.numpy.extract_compute('CenterInfo', LMP_STYLE_ATOM, LMP_TYPE_ARRAY).astype(np.float64)
        all_array = self.lmp.numpy.extract_compute('AllInfo', LMP_STYLE_ATOM, LMP_TYPE_ARRAY).astype(np.float64)

        # clean up before proceeding
        self.lmp.commands_string('''
            uncompute CenterInfo
            uncompute AllInfo
            group CenterAtoms delete
            region CellCenter delete
            ''')

        # organize information into dictionaries
        for i in center_array:

            # lammps outputs [0.0, 0.0, 0.0] for atoms not in this compute
            # ignore those atoms by skipping when atom id == 0
            # subtract one since LAMMPS indexes IDs from 1 
            if int(i[0]) != 0:
                self.center_info[f'{(int(i[0]) - 1)}'] = [i[1], i[2]]

        for i in all_array:
            self.all_info[f'{int(i[0]) - 1}'] = [i[1], i[2]]

        # there are 3 times the number of atoms of branches in phonon dispersion
        for branch in range(0, 3*self.natoms):
            self.frequencies[f'{branch}'] = []

    #----------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------- METHODS --------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------#

    # method for getting info of atoms from cell of interest
    # this method will only be used during initialization
    def CenterAtomCheck(self, multiplier):

        # create region in center of cell
        # add all atoms in region to group
        if multiplier != 2.0:
            self.lmp.command('replicate 3 3 3')
        
        self.lmp.commands_string(f'''
            region CellCenter prism {self.lat_params[1][0]} {multiplier*self.lat_params[1][0]} &
                                    {self.lat_params[1][1]} {multiplier*self.lat_params[1][1]} &
                                    {self.lat_params[1][2]} {multiplier*self.lat_params[1][2]} &
                                    {self.lat_params[2]} {self.lat_params[4]} {self.lat_params[3]}
            group CenterAtoms region CellCenter
            compute CountAtoms CenterAtoms count/type atom
            ''')
        
        # check if number of atoms in group matches natoms
        num_atoms = self.lmp.numpy.extract_compute('CountAtoms', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR).astype(np.float64)

        # if numbers don't match and multiplier has been changed,
        # then some unexpected behavior has occurred and program exits
        if int(np.sum(num_atoms)) != int(self.natoms) and multiplier == 2.0:
            raise SystemExit('Number of centeral atoms cannot be identified. Please report this!')
        
        # if numbers don't match, retry with new multiplier
        elif int(np.sum(num_atoms)) != int(self.natoms):
            self.lmp.commands_string('''
                uncompute CountAtoms
                group CenterAtoms delete
                region CellCenter delete
                ''')
            return self.CenterAtomCheck(2.0)

        # clean up before proceeding
        self.lmp.command('uncompute CountAtoms')
    
    #----------------------------------------------------------------------------------------------------------------#

    # method for displacing second atom and getting energy response
    def DispAtom2(self, atom2_id):

        # create LAMMPS group for atom2
        # add back one since it was removed earlier to match 0 based indexing of python
        self.lmp.command(f'group Atom2 id {int(atom2_id) + 1}')

        # create list for atomic displacements
        disp2 = [0, 0, self.displacement]

        # numpy array for tracking forces
        positive_disp_energies = np.zeros((1,3))
        negative_disp_energies = np.zeros((1,3))

        # loop for permuting second atom's displacements
        for j in range(0,3):

            # cyclic permutation of displacemnt for second atom
            disp2[j] = disp2[(j + 2) % 3]
            disp2[(j + 2) % 3] = disp2[(j + 1) % 3]

            # displace atom2 in positive direction
            self.lmp.command(f'displace_atoms Atom2 move {disp2[0]} {disp2[1]} {disp2[2]}')
            self.lmp.command('run 0')

            # get potential energy
            positive_disp_energies[0][j] = self.lmp.get_thermo('pe')

            # displace atom2 in negative direction
            self.lmp.command(f'displace_atoms Atom2 move {-2*disp2[0]} {-2*disp2[1]} {-2*disp2[2]}')
            self.lmp.command('run 0')

            # get potential energy
            negative_disp_energies[0][j] = self.lmp.get_thermo('pe')

            # reset atom2 back to equilibrium position
            self.lmp.command(f'displace_atoms Atom2 move {disp2[0]} {disp2[1]} {disp2[2]}')

        # clean up before exiting method
        self.lmp.command('group Atom2 delete')
        
        return -(1/(2*self.displacement))*(positive_disp_energies - negative_disp_energies)
    
    #----------------------------------------------------------------------------------------------------------------#

    # method for displacing atoms and fetching resulting potential energy
    def DispAtoms(self, atom1_id, atom2_id):

        # create LAMMPS group for atom1
        # add back 1 since it was removed earlier to match 0 based indexing of python
        self.lmp.command(f'group Atom1 id {int(atom1_id) + 1}')

        # create list for atom1 displacements
        disp1 = [0, 0, self.displacement]

        # create force constant matrix
        fcm = np.zeros((3,3))

        # loop for permuting first atom's displacements
        for i in range(0,3):

            # cyclic permurtation of displacement for first atom
            disp1[i] = disp1[(i + 2) % 3]
            disp1[(i + 2) % 3] = disp1[(i + 1) % 3]

            # displace atom1 in positive direction
            self.lmp.commands_string(f'displace_atoms Atom1 move {disp1[0]} {disp1[1]} {disp1[2]}')
            
            # displace atom2
            positive_disp_forces = self.DispAtom2(atom2_id)

            # displace atom1 in negative direction
            self.lmp.command(f'displace_atoms Atom1 move {-2*disp1[0]} {-2*disp1[1]} {-2*disp1[2]}')

            # displace atom2
            negative_disp_forces = self.DispAtom2(atom2_id)

            # update force constant matrix
            fcm[i,:] = -(1/(2*self.displacement))*(positive_disp_forces - negative_disp_forces)

            # move atom1 back to equilibrium position
            self.lmp.command(f'displace_atoms Atom1 move {disp1[0]} {disp1[1]} {disp1[2]}')

        # clean up before proceeding
        self.lmp.command('group Atom1 delete')

        return fcm
    
    #----------------------------------------------------------------------------------------------------------------#

    # method for calculating interatomic distance
    def PairDist(self, other_atom, center_atom, interatomic_dists):

        # use modulo to append distance to correct list in dictionary
        atom1 = int(center_atom) % self.natoms
        atom2 = int(other_atom) % self.natoms

        # find interatomic distances which are required later for calculating phonon frequencies
        # get list of all atom ids, subtract one since LAMMPS indexes IDs from 1
        ids = list(self.lmp.numpy.extract_atom('id').astype(np.float64) - 1)

        # find index of each both atoms in the id list
        index1 = ids.index(int(center_atom))
        index2 = ids.index(int(other_atom))

        # get list of all atom coordinates
        positions = list(self.lmp.numpy.extract_atom('x').astype(np.float64))

        # get coordinates of both atoms using indices from id list
        atom1_coord = positions[index1]
        atom2_coord = positions[index2]
        
        # find distance components
        dist_comps = atom2_coord - atom1_coord

        # update interatomic distances dictionary
        interatomic_dists[f'{atom2}_{atom1}'].append(dist_comps)
        
        return interatomic_dists
    
    #----------------------------------------------------------------------------------------------------------------#

    # method for displacing atoms and Constructing Force Constant Matrix (CFCM)
    def CFCM(self):

        # define dictionaries for storing necessary info used later
        force_constants = {}
        interatomic_dists = {}

        # create empty lists for appending to later
        # range counts from 1 since LAMMPS ids count from 1
        for atom1 in range(0, self.natoms):
            for atom2 in range(0, self.natoms):
                force_constants[f'{atom2}_{atom1}'] = []
                interatomic_dists[f'{atom2}_{atom1}'] = []

        # modulos are used to append force constant matrices to correct list in dictionary
        # loop for displacing center atoms
        for center_atom in self.center_info:
            atom1 = int(center_atom) % self.natoms

            # loop for displacing all atoms
            for other_atom in self.all_info:
                atom2 = int(other_atom) % self.natoms

                # find interatomic distance for calculating 
                # phonon frequencies later on
                interatomic_dists = self.PairDist(other_atom, center_atom, interatomic_dists)
                fcm = self.DispAtoms(other_atom, center_atom)
                
                # store force constant matrix (fcm) in force constants dictionary
                force_constants[f'{atom2}_{atom1}'].append(fcm)

        return force_constants, interatomic_dists

    #----------------------------------------------------------------------------------------------------------------#

    # method for interpolating reciprocal space path
    def KPath(self):

        # create variables for real space lattice vectors
        a = np.array([self.lat_params[1][0] - self.lat_params[0][0], 0.0, 0.0])
        b = np.array([self.lat_params[2], self.lat_params[1][1] - self.lat_params[0][1], 0.0])
        c = np.array([self.lat_params[4], self.lat_params[3], self.lat_params[1][2] - self.lat_params[0][2]])

        # find reciprocal lattice vectors
        b1 = 2*np.pi*(np.cross(b,c))/np.dot(a,np.cross(b,c))
        b2 = 2*np.pi*(np.cross(c,a))/np.dot(a,np.cross(b,c))
        b3 = 2*np.pi*(np.cross(a,b))/np.dot(a,np.cross(b,c))

        # check reciprocal space path
        # use keyword 'all' to sample over entire brillouin zone, this is the default
        # if h, k, and l are not specified (i.e. remain as None)
        # then auto assign h, k, and l based off of lattice parameters
        if self.klist == 'all':
            self.klist = []
            if self.h == None:
                self.h = int(np.ceil(1/(np.linalg.norm(a))*50))
            if self.k == None:
                self.k = int(np.ceil(1/(np.linalg.norm(b))*50))
            if self.l == None:
                self.l = int(np.ceil(1/(np.linalg.norm(c))*50))
            for h in range(0, self.h):
                for k in range(0, self.k):
                    for l in range(0, self.l):
                        k_vec = np.array([h/self.h, k/self.k, l/self.l])
                        self.klist.append(k_vec)

            # convert from reduced coordinates to cartesian coordinates
            for i, q in enumerate(self.klist):
                self.klist[i] = q[0]*b1 + q[1]*b2 + q[2]*b3

            self.klist = np.concatenate(self.klist)
        
        # if something other than 'all' is provided, then interpolate given path
        else:

            # convert from reduced coordinates to cartesian coordinates
            for i, q in enumerate(self.klist):
                self.klist[i] = q[0]*b1 + q[1]*b2 + q[2]*b3

            # interpolate reciprocal space path
            # list is for adding interpolated points to
            q_points = []
            for q in range(0, len(self.klist) - 1):
                diff = self.klist[q+1] - self.klist[q]
                scale = np.ceil(np.linalg.norm(diff))
                line_values = [self.klist[q] + t*diff for t in np.linspace(0, 1, int(scale*self.resolution))]
                q_points.append(line_values)

            # convert q_points from list of numpy arrays to one numpy array of all interpolated points
            cat_points = q_points[0]
            for i in range(1,len(q_points)):
                cat_points = np.concatenate((cat_points, q_points[i]))

            # some points are generated twice so second instance is deleted
            to_be_del = np.ones(len(cat_points), dtype = bool)
            for i in range(len(cat_points)):
                if i != 0 and i % self.resolution == 0:
                    to_be_del[i] == False

            self.klist = cat_points[to_be_del]

    #----------------------------------------------------------------------------------------------------------------#

    # method for Constructing Dynamical Matrix (CDM)
    def CDM(self):

        # fetch force constant matrices and interatomic distances
        force_constants, interatomic_dists = self.CFCM()

        # generate points along reciprocal space path
        self.KPath()

        # loop over all points along recipocal space path
        for q_val in self.klist:

            # construct a dynamical matrix for each q point
            self.d_matrices[f'{q_val}'] = np.zeros((3*self.natoms, 3*self.natoms), dtype = complex)

            # loop over force constant dictionary
            for i in force_constants:

                # force constant matrices of the same atom in different unit cells with a central atom are summed
                summation_list = []

                # in 3x3x3 supercell there are 27 unit cells, loop over all 27
                for j in range(0, 27):

                    # find the force constant fourier transform (fcft)
                    fcft = force_constants[i][j]*np.exp(-1j*np.dot(q_val, interatomic_dists[i][j]))
                    summation_list.append(fcft)

                # indices represent which atoms contribute to the force matrix
                index1 = int(i.split('_')[0]) 
                index2 = int(i.split('_')[1]) 

                # grab masses of each atom
                mass1 = self.all_info[f'{index1}'][1]
                mass2 = self.all_info[f'{index2}'][1]

                # compute elements of the dynamical matrix
                d_elements = (1/np.sqrt(mass1*mass2))*sum(summation_list)

                # update dynamical matrix with elements
                i1 = index1 % self.natoms
                i2 = index2 % self.natoms
                self.d_matrices[f'{q_val}'][3*(i1):3*(i1)+3, 3*(i2):3*(i2)+3] = d_elements

    #----------------------------------------------------------------------------------------------------------------#
    
    # method for Finding Frequencies From Path (F3P), this is used when entire brillouin zone is not sampled
    def F3P(self):

        # diagonalize matrices along reciprocal space path
        for q_val in self.klist:
            freq_vals = np.linalg.eigvalsh(self.d_matrices[f'{q_val}'])

            # check sign of frequencies and convert to THz
            # imaginary frequencies are made negative for plotting purposes
            # organize frequencies into respective branch
            for branch, freq in enumerate(freq_vals):
                if freq < 0:
                    freq_vals[branch] = -1*np.sqrt(-1*freq*self.conversion)
                    self.frequencies[f'{branch}'].append(freq)
                else:
                    freq_vals[branch] = np.sqrt(freq*self.conversion)
                    self.frequencies[f'{branch}'].append(freq)

    #----------------------------------------------------------------------------------------------------------------#

    # method that serves as shortcut for calling methods needed for calculating frequencies
    def Calc(self):

        self.CDM()
        self.F3P()

    #----------------------------------------------------------------------------------------------------------------#

    # method for interpolating frequencies after sampling entire brillouin zone
    # use this after calling obj.Calc() with default self.klist = 'all' to interpolate frequencies along any path
    def Interpolate(self):

        pass

    #----------------------------------------------------------------------------------------------------------------#
    
    # method for ploting with matplotlib
    def MPLplot(self):

        import matplotlib.pyplot as plt

        x = np.linspace(0, 1, len(self.klist))

        fig, ax = plt.subplots()

        for branch in self.frequencies:
            y = self.frequencies[branch]
            ax.plot(x, y, color = 'black')

        plt.savefig('phonon_dispersion.png')