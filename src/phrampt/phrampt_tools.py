#---------------------------------------------------------------------------------------------------------------------#
#----------------- Tool for comparing machine-learned potential phonon plots to DFT phonon plots ---------------------#
#---------------------------------------------------------------------------------------------------------------------#

import numpy as np
from lammps import lammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR

#---------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- CLASS START ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

# class for managing phonon calculation within LAMMPS
class PhononManager:
    '''
    Class used to setup functionality for calculating phonons in LAMMPS
    '''

    def __init__(self, in_file):
        '''
        Parameters
        ----------
        in_file : str
            Name of the LAMMPS input file
        klist : array
            List of points that define reciprocal space path
            Example for FCC: [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.375, 0.375, 0.75], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0], 'L', 'G', 'K', 'X', 'G']
            Leave blank to sample all of brillouin zone
        hkl : array
            Determines how finely brillouin zone is sampled if klist is left blank
        resolution : int
            Determines how many points are calculated between reciprocal space path points
        symmetry : bool
            Determines whether calculation considers symmetry or not
        '''
        self._infile = in_file

        # create LAMMPS object
        self._lmp = lammps()

        # load LAMMPS in_file
        self._lmp.file(in_file)

        # use LAMMPS run command to initialize energy
        self._lmp.command('run 0')

        # get box parameters for finding center atoms
        # get total number of atoms
        self._lat_params = self._lmp.extract_box()
        self._natoms = self._lmp.get_natoms()

        # create dictionaries for tracking atom information
        self._all_info = {}
        self._center_info = {}
        self._force_constants = {}
        self._inter_dists = {}
        self.d_matrices = {}
        self.frequencies = {}
        self.normal_modes = {}
        self.dft_frequencies = None

        # list of reciprocal space points to sample
        # can use klist = 'all' to sample entire brillouin zone
        # h, k, and l define how finely brillouin zone is sampled when using 'all'
        # if h, k, or l remain None the fineness of that dimesion is automatically determined
        # see _KPath method for more details on fineness determination
        self.klist = 'all'
        self.hkl = None
        self._hi_sym_pts = None
        self._knames = []

        # initialize variables needed for phonon calc
        # displacement is amount atoms are displaced in angstroms
        # resolution is the number of points interpolated between reciprocal space points
        # parallel determines whether the calculation is done in parallel or serial
        # _conversion is a _conversion factor for plotting frequencies in units of THz
        # symmetry determines whether calculation is made using symmetry or not
        # make_supercell determines whether simulation makes supercell or not, set to False to turn off
        self.symmetry = True
        self.make_supercell = [3,3,3]
        self.resolution = 100
        self.parallel = False

        units = self._lmp.extract_global('units')
        if units == 'metal':
            self.displacement = 0.015
            self._conversion = 1.6021773/6.022142*10**3

        elif units == 'real':
            self.displacement = 0.015
            self._conversion = None

        elif units == 'si':
            self.displacement = 1.5*10**(-12)
            self._conversion = None

        else:
            raise SystemExit(f'The {units} units style is not supported. Please change to metal, real, or si units.')

    #-----------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------- METHODS ---------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------#    

    # method for displacing atoms and fetching resulting potential energy
    def _DispAtoms(self, atom1_id, atom2_id):

        # this method will be overwritten by the subclasses classes
        # see subclasses at end of script
        pass
    
    #-----------------------------------------------------------------------------------------------------------------#

    # method for calculating interatomic distance
    def _PairDist(self, other_atom, center_atom):

        # use modulo to append distance to correct list in dictionary
        atom1 = int(center_atom) % self._natoms
        atom2 = int(other_atom) % self._natoms

        # find interatomic distances which are required later for calculating phonon frequencies
        # get list of all atom ids, subtract one since LAMMPS indexes IDs from 1
        ids = list(self._lmp.numpy.extract_atom('id').astype(np.float64) - 1)

        # find index of each both atoms in the id list
        index1 = ids.index(int(center_atom))
        index2 = ids.index(int(other_atom))

        # get list of all atom coordinates
        positions = list(self._lmp.numpy.extract_atom('x').astype(np.float64))

        # get coordinates of both atoms using indices from id list
        atom1_coord = positions[index1]
        atom2_coord = positions[index2]
        
        # find distance components
        dist_comps = atom2_coord - atom1_coord

        # update interatomic distances dictionary
        self._inter_dists[f'{atom2}_{atom1}'].append(dist_comps)
    
    #-----------------------------------------------------------------------------------------------------------------#

    # method for displacing atoms and Constructing Force Constant Matrix (CFCM)
    def _CFCM(self):

        # create empty lists for appending to later
        for center_atom1 in self._center_info:
            for atom2 in range(self._natoms):
                atom1 = int(center_atom1) % self._natoms
                self._force_constants[f'{atom2}_{atom1}'] = []
                self._inter_dists[f'{atom2}_{atom1}'] = []

        # modulos are used to append force constant matrices to correct list in dictionary
        # loop for displacing center atoms
        for center_atom in self._center_info:
            atom1 = int(center_atom) % self._natoms

            # loop for displacing all atoms
            for other_atom in self._all_info:
                atom2 = int(other_atom) % self._natoms

                # if atoms are the same, skip
                # self interaction is evaluated later
                if other_atom == center_atom:
                    continue

                # find interatomic distance for calculating phonon frequencies later on
                self._PairDist(other_atom, center_atom)
                fcm = self._DispAtoms(other_atom, center_atom)
                
                # store force constant matrix (fcm) in force constants dictionary
                self._force_constants[f'{atom2}_{atom1}'].append(fcm)

    #-----------------------------------------------------------------------------------------------------------------#

    # method for computing force constants from self interactions
    def _AcousticSum(self):

        # compute self interaction according to acoustic sum rule
        # force matrices between the same atoms are compute as the negative sum of the other force matrices
        force_on_self = 0
        count = 0
        for force_matrix in self._force_constants:
            force_on_self -= sum(self._force_constants[force_matrix])
            count += 1

            # once all force matrices between 1 atom and all other atoms are summed, append and redo for next atom
            if count == self._natoms:
                atom_num = force_matrix.split('_')[1]
                self._force_constants[f'{atom_num}_{atom_num}'].append(force_on_self)
                self._inter_dists[f'{atom_num}_{atom_num}'].append(np.zeros(3))
                force_on_self = 0
                count = 0 

    #-----------------------------------------------------------------------------------------------------------------#

    # method for interpolating reciprocal space path
    def _KPath(self):

        # create variables for real space lattice vectors
        a = np.array([self._lat_params[1][0] - self._lat_params[0][0], 0.0, 0.0])
        b = np.array([self._lat_params[2], self._lat_params[1][1] - self._lat_params[0][1], 0.0])
        c = np.array([self._lat_params[4], self._lat_params[3], self._lat_params[1][2] - self._lat_params[0][2]])

        # find reciprocal lattice vectors
        b1 = 2*np.pi*(np.cross(b,c))/np.dot(a,np.cross(b,c))
        b2 = 2*np.pi*(np.cross(c,a))/np.dot(a,np.cross(b,c))
        b3 = 2*np.pi*(np.cross(a,b))/np.dot(a,np.cross(b,c))
        rec_lat = np.array([b1,b2,b3]).reshape((3,3))

        # check reciprocal space path
        # use keyword 'all' to sample over entire brillouin zone, this is the default
        # if h, k, and l are not specified (i.e. remain as None)
        # then auto assign h, k, and l based off of lattice parameters
        if self.klist == 'all':
            self.klist = []
            if self.hkl == None:
                self.hkl = np.array([0.0, 0.0, 0.0])
                self.hkl[0] = np.ceil(1/(np.linalg.norm(a))*50)
                self.hkl[1] = np.ceil(1/(np.linalg.norm(b))*50)
                self.hkl[2] = np.ceil(1/(np.linalg.norm(c))*50)
            for h in range(0, int(self.hkl[0])):
                for k in range(0, int(self.hkl[1])):
                    for l in range(0, int(self.hkl[2])):
                        k_vec = np.array([h/self.hkl[0], k/self.hkl[1], l/self.hkl[2]])
                        self.klist.append(k_vec)

            # convert from reduced coordinates to cartesian coordinates
            for i, q in enumerate(self.klist):
                self.klist[i] = q[0]*b1 + q[1]*b2 + q[2]*b3

            # save high symmetry points for plotting later
            self._hi_sym_pts = self.klist
            self.klist = np.concatenate(self.klist)
        
        # if something other than 'all' is provided, then interpolate given path
        else:

            # get kpoint labels from klist
            for i, k_label in enumerate(self.klist):
                if i >= len(self.klist)/2:
                    self._knames.append(k_label)

            # clean up klist by removing labels
            del self.klist[int(len(self.klist)/2):]

            # convert from reduced coordinates to cartesian coordinates
            for i, q in enumerate(self.klist):
                self.klist[i] = np.matmul(q, rec_lat)
            
            # save high symmetry points for plotting later
            self._hi_sym_pts = self.klist

            # interpolate reciprocal space path
            # list is for adding interpolated points to
            q_pts = []
            for q in range(0, len(self.klist) - 1):
                diff = self.klist[q+1] - self.klist[q]
                scale = np.linalg.norm(diff)
                line_values = [self.klist[q] + t*diff for t in np.linspace(0, 1, int(scale*self.resolution))]
                q_pts.append(line_values)

            # convert q_points from list of numpy arrays to one numpy array of all interpolated points
            cat_pts = q_pts[0]
            for i in range(1,len(q_pts)):
                cat_pts = np.concatenate((cat_pts, q_pts[i]))

            # some points are generated twice so second instance is deleted
            to_be_del = np.ones(len(cat_pts), dtype = bool)
            for i, q_vec in enumerate(cat_pts):
                if q_vec[0] == cat_pts[i-1][0] and q_vec[1] == cat_pts[i-1][1] and q_vec[2] == cat_pts[i-1][2]:
                    to_be_del[i] = False

            self.klist = cat_pts[to_be_del]

    #-----------------------------------------------------------------------------------------------------------------#

    # method for Constructing Dynamical Matrix (CDM)
    def _CDM(self):

        # generate points along reciprocal space path
        self._KPath()

        # compute how many unit cells there are
        cell_range = self.make_supercell[0]*self.make_supercell[1]*self.make_supercell[2]

        # loop over all points along recipocal space path
        for ind, q_val in enumerate(self.klist):

            # construct a dynamical matrix for each q point
            self.d_matrices[f'{q_val}_{ind}'] = np.zeros((3*self._natoms, 3*self._natoms), dtype = complex)

            # loop over force constant dictionary
            for i in self._force_constants:

                # force constant matrices of the same atom in different unit cells with a central atom are summed
                summation_list = []

                # in XxYxZ supercell there are cell_range unit cells, loop over all cell_range
                for j in range(0, cell_range):

                    # find the force constant fourier transform (fcft)
                    fcft = self._force_constants[i][j]*np.exp(-1j*np.dot(q_val, self._inter_dists[i][j]))
                    summation_list.append(fcft)

                # indices represent which atoms contribute to the force matrix
                index1 = int(i.split('_')[0]) 
                index2 = int(i.split('_')[1]) 

                # grab masses of each atom
                mass1 = self._all_info[f'{index1}'][1]
                mass2 = self._all_info[f'{index2}'][1]

                # compute elements of the dynamical matrix
                d_elements = (1/np.sqrt(mass1*mass2))*sum(summation_list)

                # update dynamical matrix with elements
                i1 = index1 % self._natoms
                i2 = index2 % self._natoms
                self.d_matrices[f'{q_val}_{ind}'][3*(i1):3*(i1)+3, 3*(i2):3*(i2)+3] = d_elements

    #-----------------------------------------------------------------------------------------------------------------#
    
    # method for Finding Frequencies From Path (F3P)
    def _F3P(self):

        # there are 3 times the number of atoms of branches in phonon dispersion
        for branch in range(3*self._natoms):
            self.frequencies[f'{branch}'] = []

        # diagonalize matrices along reciprocal space path
        for ind, q_val in enumerate(self.klist):
            freq_vals= np.linalg.eigvalsh(self.d_matrices[f'{q_val}_{ind}'])

            # check sign of frequencies and convert to THz
            # imaginary frequencies are made negative for plotting purposes
            # organize frequencies into respective branch
            for branch, freq in enumerate(freq_vals):
                freq_vals[branch] = np.sqrt(abs(freq)*self._conversion)*np.sign(freq)
                self.frequencies[f'{branch}'].append(freq_vals[branch])
    
    #-----------------------------------------------------------------------------------------------------------------#

    # method that serves as shortcut for calling methods needed for calculating frequencies
    def Calc(self):

        # this method will be over written by subclasses
        # see subclasses at end of script
        pass

    #-----------------------------------------------------------------------------------------------------------------#

    # method for finding normal modes from eigenvectors of dynamical matrices
    def NormalModes(self):

        # find atomic ids and inverse root of masses
        atom_ids = np.zeros(self._natoms)
        masses = []
        for i, atom in enumerate(self._center_info):
            atom_ids[i] = int(atom)
            masses.append([np.reciprocal(np.sqrt(self._center_info[atom][1]))])

        masses = np.array(masses)

        # loop over reciprocal space points since there is a normal mode for each point
        for ind, q_val in enumerate(self.klist):

            # update dictionary
            self.normal_modes[f'{q_val}_{ind}'] = [[], atom_ids.tolist(), []]

            # compute eigenvectors
            eig_vals, eig_vecs = np.linalg.eigh(self.d_matrices[f'{q_val}_{ind}'])

            # reorganize eigenvectors according to numpy documentation
            eig_vecs = [eig_vecs[:,i] for i in range(len(eig_vecs))]

            # tuple of vals and vecs
            eig_vals_and_vecs = list(zip(eig_vals, eig_vecs))

            # loop through tuple to convert vals and vecs to correct units
            for val_and_vec in eig_vals_and_vecs:

                # convert frequencies to THz, aka eig_vals
                freq = val_and_vec[0]
                freq = np.sqrt(abs(freq)*self._conversion)*np.sign(freq)

                # split up eigenvector into smaller vectors of length 3 for each atom
                # scale eigenvectors by masses
                vecs = np.split(val_and_vec[1], self._natoms)
                vecs = vecs*masses*2*np.pi

                # append to normal modes dictionary
                self.normal_modes[f'{q_val}_{ind}'][0].append(freq)
                self.normal_modes[f'{q_val}_{ind}'][2].append(vecs)

    #-----------------------------------------------------------------------------------------------------------------#

    # method for constructing supercell in LAMMPS
    def _MakeSupercell(self):

        # use LAMMPS command to create supercell
        self._lmp.command(f'replicate {self.make_supercell[0]} {self.make_supercell[1]} {self.make_supercell[2]}')
        
        # find center unit cell based off of make_supercell dimensions
        bottom_cells = self.make_supercell[0]*self.make_supercell[1]*round(self.make_supercell[2]/2 - 0.5)
        center_cells = self.make_supercell[0]*round(self.make_supercell[1]/2 - 0.5)
        remaining_cells = round(self.make_supercell[0]/2 - 0.5)
        center_number = bottom_cells + center_cells + remaining_cells
        
        # find atoms that are in center unit cell of the XxYxZ supercell
        for i in range(center_number*self._natoms + 1, (center_number + 1)*self._natoms + 1):
            self._lmp.command(f'group CenterAtoms id {i}')

        # fetch information from all atoms and centeral atoms
        self._lmp.commands_string('''
            compute CenterInfo CenterAtoms property/atom id type mass
            compute AllInfo all property/atom id type mass
            run 0
            ''')
        
        # extract compute information from LAMMPS into numpy arrays
        center_array = self._lmp.numpy.extract_compute('CenterInfo', LMP_STYLE_ATOM, LMP_TYPE_ARRAY).astype(np.float64)
        all_array = self._lmp.numpy.extract_compute('AllInfo', LMP_STYLE_ATOM, LMP_TYPE_ARRAY).astype(np.float64)

        # clean up before proceeding
        self._lmp.commands_string('''
            uncompute CenterInfo
            uncompute AllInfo
            group CenterAtoms delete
            ''')

        # organize information into dictionaries
        for i in center_array:

            # lammps outputs [0.0, 0.0, 0.0] for atoms not in this compute
            # ignore those atoms by skipping when atom id == 0
            # subtract one since LAMMPS indexes IDs from 1 
            if int(i[0]) != 0:
                self._center_info[f'{(int(i[0]) - 1)}'] = [i[1], i[2]]

        for i in all_array:
            self._all_info[f'{int(i[0]) - 1}'] = [i[1], i[2]]

    #-----------------------------------------------------------------------------------------------------------------#

    # method for interpolating frequencies after sampling entire brillouin zone
    # use this after calling obj.Calc() with default self.klist = 'all' to interpolate frequencies along any path
    def Interpolate(self, _kpath=None):
        pass

    #-----------------------------------------------------------------------------------------------------------------#

    # method for extracting phonon frequencies from vaspout.h5 file
    def ReadVasp_h5(self, path='.', h5_file=None):

        import py4vasp as p4v

        # create p4v object from h5 file
        if h5_file == None:
            dft_calc = p4v.Calculation.from_path(path)
        else:
            dft_calc = p4v.Calculation.from_file(h5_file)

        # extract phonon frequencies from calculation
        phonon_dict = dft_calc.phonon_band.to_dict()

        # loop through dictionary to organize frequencies by branch
        for i in range(0, len(phonon_dict['bands'][0])):
            self.dft_frequencies[f'{i}'] = []
            for branch in phonon_dict['bands']:
                self.dft_frequencies.append(branch[i])

    #-----------------------------------------------------------------------------------------------------------------#
    
    # method for plotting with matplotlib
    def Plot_mpl(
        self, rgb=[0,0,0], title=None, xaxis=None, yaxis='Frequency (THz)', zeroline=False, zeroline_rgb=[1,0,0],
        file_name='phonon_dispersion.png', vgrid=True, vgrid_rgb=[0,0,0], hgrid=False, hgrid_rgb=[0,0,0], dpi=1000
    ):
        '''
        Method for plotting phonon frequencies with the matplotlib library

        Parameters
        ----------
        rgb : array
            Set color of phonon bands with rgb format
        title : str
            Set title of plot
        xaxis : str
            Set title of xaxis
        yaxis : str
            Set title of yaxis
        zeroline : bool
            Plots a line at y = 0
        zeroline_rgb : array
            Set color of zeroline with rgb format
        file_name : str
            Set name of output plot
        vgrid : bool
            Plots vertical lines at high symmetry points along reciprocal space path
        vgrid_rgb : array
            Sets color of vgrid lines with rgb format
        hgrid : bool
            Plots horizontal grid lines
        hgrid_rgb : array
            Sets color of hgrid lines with rgb format
        dpi : int
            Determines image resolution
        '''

        import matplotlib.pyplot as plt

        # create x axis
        x = np.linspace(0, 1, len(self.klist))

        # create subplots, each phonon branch will have its own y values
        _, ax = plt.subplots()

        # add high symmetry point labels to plot        
        x_vals = []
        kpoints = self.klist.tolist()
        for q_point in self._hi_sym_pts:
            ind = kpoints.index(q_point.tolist())
            x_vals.append(ind/len(self.klist))
            kpoints[ind] = None
        plt.xticks(ticks=x_vals, labels=self._knames)
        
        # functionality for plotting vertical lines at all high symmetry points
        if vgrid == True:

            # last high symmetry point plots awkwardly on top of plot boundary, so remove it
            del x_vals[-1]
            ax.vlines(x_vals, 0, 1, transform=ax.get_xaxis_transform(), color=vgrid_rgb, alpha=0.5)

        # functionality for plotting horizontal grid lines
        if hgrid == True:
            ax.grid(axis='y', color=hgrid_rgb, alpha=0.5)

        # functionality for plotting horizontal line at zero
        if zeroline == True:
            ax.plot(x, np.zeros((1,len(x)))[0], color=zeroline_rgb)

        # loop through branches plotting the frequencies
        for branch in self.frequencies:
            y = self.frequencies[branch]
            ax.plot(x, y, color=rgb)

        # user can choose how to visualize plot by updating these arguments
        ax.set_title(title)
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)

        # fix margins and save plot to png 
        plt.margins(x=0)
        plt.savefig(file_name, dpi=dpi)

    #-----------------------------------------------------------------------------------------------------------------#

    # method for plotting with plotly
    def Plot_plotly(self):
        pass

    #-----------------------------------------------------------------------------------------------------------------#

    # method for writing normal modes to JSON file to be visualized at https://interactivephonon.materialscloud.io/
    def Write_json(self, file_name='phononvis.json'):
        '''
        Method for writing normal modes to JSON file for visualization.
        JSON file can be visualized @ https://interactivephonon.materialscloud.io/
        
        Parameters
        ----------
        file_name : str
            Sets name of JSON file
        '''

        import json

        # speed of light constant in cm/s for converting THz -> cm^-1
        SOL = 29979245800

        # see https://interactivephonon.materialscloud.io/compute/input_help/#phononvis-json for JSON format
        json_dict = {
            'name':self._infile,
            'natoms':self._natoms,
            'lattice':None,
            'atom_types':None,
            'atom_numbers':None,
            'formula':None,
            'repetitions':self.make_supercell,
            'atom_pos_car':None,
            'atom_pos_red':None,
            'highsym_qpts':None,
            'qpoints':None,
            'distances':None,
            'eigenvalues':[[a*10**12/SOL for a in self.normal_modes[k][0]] for k in self.normal_modes],
            'vectors':[[[[list(c) for c in zip(b.real.tolist(), b.imag.tolist())] for b in a] for a in self.normal_modes[k][2]] for k in self.normal_modes]
        }

        # create variables for real space lattice vectors since many of following functions will use these
        a = np.array([self._lat_params[1][0] - self._lat_params[0][0], 0.0, 0.0])
        b = np.array([self._lat_params[2], self._lat_params[1][1] - self._lat_params[0][1], 0.0])
        c = np.array([self._lat_params[4], self._lat_params[3], self._lat_params[1][2] - self._lat_params[0][2]])

        # function to update lattice key in json_dict
        def _update_lattice():
            return [a.tolist(), b.tolist(), c.tolist()]
        
        # function to update atom_types and atom_numbers in json_dict
        # attempts to do so based off of provided masses since LAMMPS does not really track atoms by elemental label
        def _update_atoms():
            atom_labels = ['H', 'He', 'Li', 'Be', 'B',
                           'C', 'N', 'O', 'F', 'Ne',
                           'Na', 'Mg', 'Al', 'Si', 'P',
                           'S', 'Cl', 'Ar', 'K', 'Ca',
                           'Sc', 'Ti', 'V', 'Cr', 'Mn',
                           'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                           'Ga', 'Ge', 'As', 'Se', 'Br',
                           'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                           'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
                           'Pd', 'Ag', 'Cd', 'In', 'Sn',
                           'Sb', 'Te', 'I', 'Xe', 'Cs',
                           'Ba', 'La', 'Ce', 'Pr', 'Nd',
                           'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                           'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                           'Lu', 'Hf', 'Ta', 'W', 'Re',
                           'Os', 'Ir', 'Pt', 'Au', 'Hg',
                           'Tl', 'Pb', 'Bi', 'Po', 'At',
                           'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                           'Pa', 'U']
            atom_masses = np.array([1.008, 4.003, 6.941, 9.0122, 10.811,
                           12.011, 14.007, 15.999, 18.998, 20.18,
                           22.99, 24.305, 26.982, 28.086, 30.974,
                           32.065, 35.453, 39.948, 39.098, 40.076,
                           44.956, 47.867, 50.943, 51.996, 54.938,
                           55.845, 58.993, 58.693, 63.546, 65.409,
                           69.723, 72.64, 74.922, 78.96, 79.904,
                           83.798, 85.468, 87.62, 88.906, 91.224,
                           92.906, 95.94, 98, 101.07, 102.906,
                           106.42, 107.868, 112.411, 114.818, 118.71,
                           121.76, 127.6, 126.904, 131.293, 132.905,
                           137.327, 138.906, 140.116, 140.908, 144.24,
                           145, 150.36, 151.964, 157.25, 158.925,
                           162.5, 164.93, 167.259, 168.934, 173.04,
                           174.967, 178.49, 180.948, 183.84, 186.207,
                           190.23, 192.217, 195.078, 196.967, 200.59,
                           204.383, 207.2, 208.98, 209, 210,
                           222, 223, 226, 227, 232.038,
                           231.036, 238.029])
            
            # loop through all atoms
            # subtract mass from array of masses, find index of what value is closest to zero
            elements = []
            atomic_num = []
            for atom in self._all_info:
                if int(atom) < self._natoms:
                    mass = self._all_info[atom][1]
                    i = np.argmin(abs(atom_masses - mass))
                    elements.append(atom_labels[i])
                    atomic_num.append(i+1)

            # find formula based off of recurrence of elements
            formula = ''
            for label in list(set(elements)):
                count = len([_ for _ in elements if _ == label])
                formula += label + f'{count}'

            return elements, atomic_num, formula

        # fetch atomic positions and update json_dict
        def _update_pos():

            # get list of all atom ids, subtract one since LAMMPS indexes IDs from 1
            ids = list(self._lmp.numpy.extract_atom('id').astype(np.float64) - 1)

            # get list of all atom coordinates
            positions = list(self._lmp.numpy.extract_atom('x').astype(np.float64))

            # loop through atoms
            pos_car = []
            for atom in self._all_info:
                if int(atom) < self._natoms:
                    id_index = ids.index(int(atom))
                    pos_car.append(positions[id_index].tolist())

            # find reduced positions
            lat_mat = np.array([a,b,c])
            inv_mat = np.linalg.inv(lat_mat)
            pos_red = []
            for pos in pos_car:
                red = np.matmul(pos, inv_mat)
                pos_red.append(red.tolist())

            return pos_car, pos_red
        
        # function for assigning qpts, highsym_qpts, and distances between qpts in json_dict
        def _update_qpts():
        
            # find reciprocal lattice vectors
            b1 = 2*np.pi*(np.cross(b,c))/np.dot(a,np.cross(b,c))
            b2 = 2*np.pi*(np.cross(c,a))/np.dot(a,np.cross(b,c))
            b3 = 2*np.pi*(np.cross(a,b))/np.dot(a,np.cross(b,c))
            rec_mat = np.array([b1,b2,b3])
            inv_mat = np.linalg.inv(rec_mat)

            # find reduced q points
            qpts = []
            for point in self.klist:
                qpts_red = np.matmul(point, inv_mat)
                qpts.append(qpts_red.tolist())
                
            # find index and label of high symmetry q points in klist
            # as points are identified, delete them to prevent repeats from getting wrong index value
            # add back the number of points deleted to keep index value accurate
            highsym_qpts = []
            kpoints = self.klist.tolist()
            add_back = 0
            for i, point in enumerate(self._hi_sym_pts):
                ind = kpoints.index(point.tolist())
                label = self._knames[i]
                highsym_qpts.append([ind + add_back, label])
                del kpoints[ind]
                add_back += 1

            # find distances between qpts
            # distances are reduced by 2pi for some reason unknown to me
            # distances are also calculated as distance between high symmetry point and all 
            # points between until the next high symmetry point
            # the next distances add on to the distances of the previous high symmetry point
            # idk why this is how "distances" are specified, but that's what is needed to work with this json
            distances = []
            count = 1
            last_val = 0
            ind = 0
            for i, point in enumerate(self.klist):
                if i - 1 == highsym_qpts[count][0]:
                    count += 1
                    ind = i - 1
                    last_val = cur_val
                p1 = self.klist[ind]
                p2 = self.klist[i]
                cur_val = np.linalg.norm(p1 - p2)/(2*np.pi) + last_val
                distances.append(cur_val)
            
            return highsym_qpts, qpts, distances
        
        # encoder for printing numpy objects to json
        # credit for encoder:
        # https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        # use functions to assign values to json_dict
        json_dict['lattice'] = _update_lattice()
        json_dict['atom_types'], json_dict['atom_numbers'], json_dict['formula'] = _update_atoms()
        json_dict['atom_pos_car'], json_dict['atom_pos_red'] = _update_pos()
        json_dict['highsym_qpts'], json_dict['qpoints'], json_dict['distances'] = _update_qpts()

        # dump json_dict to json file
        with open(file_name, 'w') as json_file:
            json.dump(json_dict, json_file, cls=NpEncoder)

    #-----------------------------------------------------------------------------------------------------------------#

    # method for saving calculation as binary
    def SaveCalc(self, file_name='SaveFile.pkl'):
        '''
        Method for saving frequencies and dynamical matrices of calculation to binary

        Parameters
        ----------
        file_name : str
            Sets name of binary to save
        '''

        import pickle

        # print calculation info to file that can be loaded later
        with open(file_name, 'wb') as f:
            pickle.dump(self.klist, f)
            pickle.dump(self._hi_sym_pts, f)
            pickle.dump(self._knames, f) 
            pickle.dump(self.d_matrices, f)
            pickle.dump(self.frequencies, f)

    #-----------------------------------------------------------------------------------------------------------------#

    # method for loading dictionaries from previous calculation
    def LoadCalc(self, file_name='SaveFile.pkl'):
        '''
        Method for loading frequencies and dynamical matrices of calculation from binary

        Parameters
        ----------
        file_name : str
            Name of binary to load
        '''

        import pickle

        # read in binary and reset previous attributes to current ones
        with open(file_name, 'rb') as f:
            self.klist = pickle.load(f)
            self._hi_sym_pts = pickle.load(f)
            self._knames = pickle.load(f)
            self.d_matrices = pickle.load(f)
            self.frequencies = pickle.load(f)

#---------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- CLASS END -------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- CLASS START ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

# this class will compute phonon frequencies by finding pairwise forces between displaced atoms
# this method is faster, but requires a pairwise potential and will not work for multi-body potentials
class Pairwise(PhononManager):
    '''
    This class will compute phonon frequencies by finding pairwise forces between displaced atoms.
    This method is faster than the General class, but requires a pairwise potential and will not work for multi-body potentials.

    Attributes
    ----------
    in_file : str
        Name of the LAMMPS input file
    klist : array
        List of points that define reciprocal space path
        Example for FCC: [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.375, 0.375, 0.75], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0], 'L', 'G', 'K', 'X', 'G']
        Leave blank to sample all of brillouin zone
    hkl : array
        Determines how finely brillouin zone is sampled if klist is left blank
    resolution : int
        Determines how many points are calculated between reciprocal space path points

    Methods
    -------
    Calc()
        Calculates phonon frequencies after reading in LAMMPS input file
    Plot_mpl(rgb=[0,0,0], title=None, xaxis=None, yaxis='Frequency (THz)', zeroline=False, zeroline_rgb=[1,0,0],
    file_name='phonon_dispersion.png', vgrid=True, vgrid_rgb=[0,0,0], hgrid=False, hgrid_rgb=[0,0,0], dpi=1000)
        Plots phonon frequencies along selected reciprocal space path
    SaveCalc('SaveFile.pkl')
        Saves frequencies and dynamical matrices to binary
    LoadCalc('SaveFile.pkl)
        Loads freqencies and dynamical matrices from binary
    Interpolate(_kpath=None)
        Interpolates frequencies along any reciprocal space path if entire brillouin zone was sampled
    '''

    #-----------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------- METHODS ---------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------#  

    # method that serves as shortcut for calling methods needed for calculating frequencies
    def Calc(self):

        # make super cell based off of specification
        self._MakeSupercell()

        # check if parallel calc 
        if self.parallel == True:
            from . import parallel_phrampt as pp
            
            # get force constants and interatomic distances
            self._force_constants, self._inter_dists = pp.make_parallel(self._infile, 
                                                                        self._natoms, 
                                                                        self.make_supercell,
                                                                        'pairwise')
            
            # find self interaction force constants, construct dynamical matrices, and compute frequencies
            self._AcousticSum()
            self._CDM()
            self._F3P()

        # serial calc
        else:

            # get force contants and interatomic distances, construct dynamical matrices, compute frequencies
            self._CFCM()
            self._AcousticSum()
            self._CDM()
            self._F3P()

    #-----------------------------------------------------------------------------------------------------------------# 

    # method for displacing atoms and fetching resulting potential energy
    def _DispAtoms(self, atom1_id, atom2_id):

        # create LAMMPS group for atoms 1 and 2
        # add back 1 since it was removed earlier to match 0 based indexing of python
        # setup compute that reports the interatomic forces
        self._lmp.commands_string(f'''
            group Atom1 id {int(atom1_id) + 1}
            group Atom2 id {int(atom2_id) + 1}
            compute Forces Atom2 group/group Atom1
            run 0 
            ''')

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
            self._lmp.commands_string(f'''
                displace_atoms Atom1 move {disp1[0]} {disp1[1]} {disp1[2]}
                run 0
                ''')
            
            # find interatomic forces after displacement
            positive_disp_forces = self._lmp.numpy.extract_compute('Forces', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR).astype(np.float64)

            # displace atom1 in negative direction
            self._lmp.commands_string(f'''
                displace_atoms Atom1 move {-2*disp1[0]} {-2*disp1[1]} {-2*disp1[2]}
                run 0 
                ''')

            # find interatomic forces after displacement
            negative_disp_forces = self._lmp.numpy.extract_compute('Forces', LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR).astype(np.float64)

            # update force constant matrix
            fcm[i,:] = -(1/(2*self.displacement))*(positive_disp_forces - negative_disp_forces)

            # move atom1 back to equilibrium position
            self._lmp.command(f'displace_atoms Atom1 move {disp1[0]} {disp1[1]} {disp1[2]}')

        # clean up before proceeding
        self._lmp.commands_string('''
            uncompute Forces
            group Atom1 delete                     
            group Atom2 delete
            ''')

        return fcm
    
#---------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- CLASS END -------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- CLASS START ------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#

class General(PhononManager):
    '''
    This class will compute phonon frequencies by finding the change in potential energy after displacing atoms.
    This method is slower than the Pairwise method but will work with any potential, including multi-body potentials.

    Attributes
    ----------
    in_file : str
        Name of the LAMMPS input file
    klist : array
        List of points that define reciprocal space path
        Example for FCC: [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.375, 0.375, 0.75], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0], 'L', 'G', 'K', 'X', 'G']
        Leave blank to sample all of brillouin zone
    hkl : array
        Determines how finely brillouin zone is sampled if klist is left blank
    resolution : int
        Determines how many points are calculated between reciprocal space path points

    Methods
    -------
    Calc()
        Calculates phonon frequencies after reading in LAMMPS input file
    Plot_mpl(rgb=[0,0,0], title=None, xaxis=None, yaxis='Frequency (THz)', zeroline=False, zeroline_rgb=[1,0,0],
    file_name='phonon_dispersion.png', vgrid=True, vgrid_rgb=[0,0,0], hgrid=False, hgrid_rgb=[0,0,0], dpi=1000)
        Plots phonon frequencies along selected reciprocal space path
    SaveCalc('SaveFile.pkl')
        Saves frequencies and dynamical matrices to binary
    LoadCalc('SaveFile.pkl)
        Loads freqencies and dynamical matrices from binary
    Interpolate(_kpath=None)
        Interpolates frequencies along any reciprocal space path if entire brillouin zone was sampled
    '''

    #-----------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------- METHODS ---------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------# 

    # method that serves as shortcut for calling methods needed for calculating frequencies
    def Calc(self):

        # make super cell based off of specification
        self._MakeSupercell()

        # check if parallel calc 
        if self.parallel == True:
            from . import parallel_phrampt as pp

            # get force constants and interatomic distances
            self._force_constants, self._inter_dists = pp.make_parallel(self._infile, 
                                                                        self._natoms, 
                                                                        self.make_supercell, 
                                                                        'general')
            
            # find self interaction force constants, construct dynamical matrices, and compute frequencies
            self._AcousticSum()
            self._CDM()
            self._F3P()

        # serial calc
        else:

            # get force contants and interatomic distances, construct dynamical matrices, compute frequencies
            self._CFCM()
            self._AcousticSum()
            self._CDM()
            self._F3P()

    #-----------------------------------------------------------------------------------------------------------------#

    # method for displacing atoms and fetching resulting potential energy
    def _DispAtom2(self):

        # create list for atom1 displacements
        disp = [0, 0, self.displacement]

        # create force constant matrix
        force_vec = np.zeros(3)

        # loop for permuting first atom's displacements
        for i in range(0,3):

            # cyclic permurtation of displacement for second atom
            disp[i] = disp[(i + 2) % 3]
            disp[(i + 2) % 3] = disp[(i + 1) % 3]

            # displace atom2 in positive direction
            self._lmp.commands_string(f'''
                displace_atoms Atom2 move {disp[0]} {disp[1]} {disp[2]}
                run 0
                ''')
            
            # find potential energy after displacement
            positive_disp_energy = self._lmp.get_thermo('pe')

            # displace atom2 in negative direction
            self._lmp.commands_string(f'''
                displace_atoms Atom2 move {-2*disp[0]} {-2*disp[1]} {-2*disp[2]}
                run 0 
                ''')

            # find potential energy after displacement
            negative_disp_energy = self._lmp.get_thermo('pe')

            # update force constant matrix
            force_vec[i] = -(1/(2*self.displacement))*(positive_disp_energy - negative_disp_energy)

            # move atom2 back to equilibrium position
            self._lmp.command(f'displace_atoms Atom2 move {disp[0]} {disp[1]} {disp[2]}')

        return force_vec
    
    #-----------------------------------------------------------------------------------------------------------------#

    # method for displacing atoms and fetching resulting potential energy
    def _DispAtoms(self, atom1_id, atom2_id):

        # create LAMMPS group for atoms 1 and 2
        # add back 1 since it was removed earlier to match 0 based indexing of python
        # setup compute that reports the interatomic forces
        self._lmp.commands_string(f'''
            group Atom1 id {int(atom1_id) + 1}
            group Atom2 id {int(atom2_id) + 1}
            run 0 
            ''')

        # create list for atom1 displacements
        disp = [0, 0, self.displacement]

        # create force constant matrix
        fcm = np.zeros((3,3))

        # loop for permuting first atom's displacements
        for i in range(0,3):

            # cyclic permurtation of displacement for first atom
            disp[i] = disp[(i + 2) % 3]
            disp[(i + 2) % 3] = disp[(i + 1) % 3]

            # displace atom1 in positive direction
            self._lmp.commands_string(f'''
                displace_atoms Atom1 move {disp[0]} {disp[1]} {disp[2]}
                run 0
                ''')
            
            # find interatomic forces after displacement
            positive_disp_forces = self._DispAtom2()

            # displace atom1 in negative direction
            self._lmp.commands_string(f'''
                displace_atoms Atom1 move {-2*disp[0]} {-2*disp[1]} {-2*disp[2]}
                run 0 
                ''')

            # find interatomic forces after displacement
            negative_disp_forces = self._DispAtom2()

            # update force constant matrix
            fcm[i,:] = -(1/(2*self.displacement))*(positive_disp_forces - negative_disp_forces)

            # move atom1 back to equilibrium position
            self._lmp.command(f'displace_atoms Atom1 move {disp[0]} {disp[1]} {disp[2]}')

        # clean up before proceeding
        self._lmp.commands_string('''
            group Atom1 delete                     
            group Atom2 delete
            ''')

        return fcm
    
#---------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- CLASS END -------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------#