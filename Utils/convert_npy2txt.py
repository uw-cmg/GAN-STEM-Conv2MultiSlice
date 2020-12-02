#!/usr/bin/env python
"""
Convert the npy file to the needed txt file

Convert the npy files in one simulation data folder to txt which makes later processing easy. Now all same type data are in the same format.

YOU ONLY NEED TO RUN THIS SCRIPT ONCE!!!

"""

# import libraries
import os
import numpy as np

"""
Project Information modify if needed
"""

__author__ = "Mingren Shen"
__copyright__ = "Copyright 2018, The GAN for STEM Project"
__credits__ = [""]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Mingren Shen"
__email__ = "mshen32@wisc.edu"
__status__ = "Development"

"""
End of Project information
"""

workingDir = "../Data/Pt-Mo50_input_images/Pt281-Mo280-convolution/"
os.chdir(workingDir)
#print(os.getcwd())
#print()

for npyFile in os.listdir(os.getcwd()):
    npyArr = np.load(npyFile)
    txtFileName = npyFile.strip(".npy") + ".txt"
    np.savetxt(txtFileName,npyArr)




