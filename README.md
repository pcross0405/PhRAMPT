-------------------------------------------------------------------------------------------------------  
<h1><p align="center">PhRAMPT OVERVIEW</p></h1>

-------------------------------------------------------------------------------------------------------  

<p align="justify">The Phonon Replication And Modeling/Plotting Tool (PhRAMPT) is a tool used assess the 
accuracy of  interatomic potentials by comparing phonon dispersions generated in LAMMPS to DFT generated data.
</p>

<p align="justify">While there are existing packages that compute phonons within LAMMPS, such as Phonopy 
and the PHONON package in LAMMPS, they offer much more functionality which naturally increases the complexity when 
using these packages.</p>

<p align="justify">This package is designed to be very straightforward in its use, offering phonon dispersion 
comparisons in as little as 5 lines of Python script. What this package lacks in functionality it makes up for in
ease of use. Further, extracting phonon frequencies for user customized post processing is made easy 
allowing this package to be extended by users in nearly any way they desire.</p>

-------------------------------------------------------------------------------------------------------  
<h1><p align="center">INSTALLATION INSTRUCTIONS</p></h1>

-------------------------------------------------------------------------------------------------------  

1) Inside that directory type on the command line  
   "git clone https://github.com/pcross0405/PhRAMPT.git"

2) Type "cd PhRAMPT"

3) Make sure you have python's build tool up to date with  
   "python3 -m pip install --upgrade build"

4) Once up to date type  
   "python3 -m build"

5) This should create a "dist" directory with a .whl file inside

6) On the command line type  
   "pip install dist/'*.whl'" 

-------------------------------------------------------------------------------------------------------  
<h1><p align="center">DEPENDENCIES</p></h1>

-------------------------------------------------------------------------------------------------------  

REQUIRED FOR CALCULATING PHONON FREQUENCIES

   - [lammps](https://docs.lammps.org/Python_module.html) (the python module for LAMMPS)

   - [numpy](https://numpy.org/)

REQUIRED FOR PLOTTING

   One of the following two can be used currently.

   - [matplotlib](https://matplotlib.org/)

   - [plotly.express](https://plotly.com/python/plotly-express/)

REQUIRED FOR DFT COMPARISON

   Currently only comparsion with VASP is supported, further only VASP compiled with HDF5 support.
   Comparisons using the VASP OUTCAR file and other DFT packages will come in future update.

   - [py4vasp](https://www.vasp.at/py4vasp/latest/)

---------------------------------------------------------------------------------------------------------  
<h1><p align="center">REPORTING ISSUES</p></h1>

---------------------------------------------------------------------------------------------------------  

Please report any issues to "https://github.com/pcross0405/PhRAMPT/issues"  

-------------------------------------------------------------------------------------------------------------------------  
<h1><p align="center">TUTORIAL</p></h1>