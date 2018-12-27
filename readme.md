2D Uniform Directional FMM for e^{ikr}/r
========================

To run the code, you would first need to make the Makefile.

	make -f Makefile2D.mk

Then we need to run the executable, i.e.,

	./testFMM2D nLevels nCones_in_MFR nNodes L Wave_no DFMM/FMM

Some notations used:
level_LFR: The level where the low frequency regime starts
level_FarField: The level where Interaction lists start existing
DFMM/FMM: 1 for using Directional FMM; 0 for traditional FMM




For instance, below is a sample execution of the code and its output.

It is always good to clean using make clean before running the code, i.e.,
	
	make -f Makefile2D.mk clean

Then make the file

	make -f Makefile2D.mk

Run the generated executable as, for instance,

	./testFMM2D 5 16 14 1 16 1

The generated output is as follows:

level_LFR: 4
level_FarField: 2

Number of particles is: 200704

Time taken to create the tree is: 0.0419702

Time taken to assemble the operators is: 0.195972

Time taken to assemble the charges is: 0.0451229

Time taken for multipole to multipole is: 23.778

Time taken for multipole to local is: 8.08685

Time taken for local to local is: 41.8821

Total time taken is: 74.2261

Apply time taken is: 73.747

Total Speed in particles per second is: 2703.96

Apply Speed in particles per second is: 2721.52

Number of particles is: 200704

Performing Error check...

Box number is: 38

Box center is: (-0.53125, -0.65625);

Error is: 0.000510984

Time taken to compute error is: 5.04843

