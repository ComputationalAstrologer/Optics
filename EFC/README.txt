This is in support of the EFC module.  It is designed to help people play with the
codes used to create the article I sent to the Journal of Astronomical Telescopes,
Instruments and Systems (JATIS) on Oct. 3, 2024, entitled: A Laboratory Method for 
Measuring the Cross-Polarizaion in High-Contrast Imaging.
This article is also publicly available at http://arxiv.org/abs/2410.03579
Feel free to email me:   rfrazin@umich.edu


1- Go to:
https://drive.google.com/drive/folders/1wFmUJgcfBwJdrzOX7WS-GC6W2AWn9xw2?usp=sharing
GitHub is not a good place for large binary files, and god help you if get such a file
stuck in your commit history. (Note that if this happens to you and prayer is 
unsuccessful in resolving the issue, you can use the BFG Repo-Cleaner to do the job.)

2- From the Google drive link, download the 4 files:
SpeckleFieldReducedFrom33x33PhaseScreen_Ex.npy
SpeckleFieldReducedFrom33x33PhaseScreen_Ey.npy
ThreeOAP20mmSquareApCgkn33x33_SystemMatrixReducedContrUnits_Ex.npy
ThreeOAP20mmSquareApCgkn33x33_SystemMatrixReducedContrUnits_Ey.npy
SplCoPerturbs4SpeckleFieldReducedFrom33x33PhaseScreen.npy
The quantities in these files are complex-valued.
The first two are x and y components of a speckle field resulting from a random amplitude
and phase screen in the input.  They correxpond to f_x(0) and f_y(0) in Eqs. 31 and 32 of
the paper.  The second two are the system matrices in contrast units (explained in the
paper).  They correspond to \tilde{D}_x and \tilde{D}_y in Eqs. 19 and 20.
The last one is set of 1089 complex-valued b-spline coefficients that creates a phase and
amplitude screen that mimics low-frquency and mid-frequency optical aberrations.  It
was used to create the speckle fields in the paper.

3a: clone this repo.  Note that "EFC" is subdirectory of the "Optics" repo, which git
will clone by default.  You need only two things from the Optics repo, which are the
the program Bspline3.py and the EFC subdirectory, which has the rest of the needed
Python codes.  

3b: Create a new branch on your computer for you to edit.  This will allow you to pull changes from the repo on github (master branch) without merge conflicts.  Use: "git checkout -b MyBranch" .
To pull updates from the master branch use the commands: "git checkout master" and then "git pull origin master".  Go back to your own branch with "git checkout MyBranch" and then you can use "git merge master" at your own risk.    


4: Go to the EFC directory.  You will do everything there.

5: Open EFC.py.  This is the file with all of the tools, except for Bspline3.py (see Step 3 above).

6: The first section of EFC.py imports standard Python stuff.   Just after that, you will see
lines that specify the variable MySplineToolsLocation.  That needs to be set to the path
specifying the location of the Bspline3.py file, so that the line "import Bspline3 as BS" works.
Similarly, the variable PropMatLoc needs to be set to the location of the .npy files  you
dowloaded in Step 2 above.
	
7: The other key files are DarkHoleAnalysis.py, which contains the codes I used to create figures 
for the paper, but it's not user friendly, and CrossPolGames.py, which I made just for you to
have example to see how my tools can be used.   Open CrossPolGames.py, which is set up in a 
tutorial fashion.  Now you are ready to party!
