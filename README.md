-------------------------------------------------------------------------------------------------------  
<span style="font-size:0.5em;"><p align="center">PhRAMPT OVERVIEW</p></span>
-------------------------------------------------------------------------------------------------------  

<span><p align="justify">The Phonon Replication And Modeling/Plotting Tool (PhRAMPT) is a tool used assess the accuracy of 
interatomic potentials by comparing phonon dispersions generated in LAMMPS to DFT generated data.</p></span>

<span><p align="center">While there are existing packages that compute phonons within LAMMPS, such as Phonopy and the PHONON
package in LAMMPS, they offer much more functionality which naturally increases the complexity when 
using these packages.</p></span>

<span><p align="center">This package is designed to be very straightforward in its use, offering phonon dispersion comparisons
in as little as 5 lines of Python script. What this package lacks in functionality it makes up for in
ease of use. Further, extracting phonon frequencies for user customized post processing is made easy 
allowing this package to be extended by users in nearly any way they desire.</p></span>

-------------------------------------------------------------------------------------------------------  
<span style="font-size:0.5em;"><p align="center">INSTALLATION INSTRUCTIONS</p></span>
-------------------------------------------------------------------------------------------------------  

1) Make a directory to build the package in

2) Inside that directory type on the command line  
   "git clone https://github.com/pcross0405/PhRAMPT.git"

3) Make sure you have python's build tool up to date with  
   "python3 -m pip install --upgrade build"

4) Once up to date type  
   "python3 -m build"

5) This should create a "dist" directory with a .whl file inside

6) On the command line type  
   "pip install dist/'*.whl'" 

-------------------------------------------------------------------------------------------------------  
<span style="font-size:0.5em;"><p align="center">INSTRUCTIONS FOR USE</p></span>
-------------------------------------------------------------------------------------------------------  

TBA

---------------------------------------------------------------------------------------------------------  
<span style="font-size:0.5em;"><p align="center">REPORTING ISSUES</p></span>
---------------------------------------------------------------------------------------------------------  

Please report any issues to "https://github.com/pcross0405/PhRAMPT/issues"  

-------------------------------------------------------------------------------------------------------------------------  
<span style="font-size:0.5em;"><p align="center">SEE SAMPLES DIRECTORY FOR AN EXAMPLE OF HOW TO RUN THE PACKAGE FROM A PYTHON SCRIPT</p></span>
