#!/usr/bin/env python
"""
Calculate normalized root_mean_square_error of two images

The equation is

    E = [Sum(g-f)^2] / [Sum g^2]

Where f is the generated images pixel and g is the multislice generated images as ground truth. Both f and g are pixel value of a (x,y) point in the image.

The average errors are

the average error over 19 images is
0.42098796707235264

the average error over 18 images is
0.49464164650354925

the average error over 19 images is
0.5052395499863189
"""


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

# import libraries
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# global path and prefix or suffix

"""
Pt-Mo5_input_images

Notice ID is from 0 to 19
"""

convImgDir = "./Data/Pt-Mo5_input_images/Pt-Mo_convolution"
convPrefix = "Pt_Mo5_convolution_"
convSuffix = ".txt"
# then convPrefix + ID + convSuffix is the data file name for convolution images

multiImgDir = "./Data/Pt-Mo5_input_images/Pt-Mo_multislice"
multiPrefix = "pt_mo5_"
multiSuffix = "_cl160mm_ss.tif"
# then multiPrefix + ID + multiSuffix is the data file name for multislice images

"""
Pt-Mo50_input_images

Notice ID is from 2 to 19
"""

# # You only need to uncomment the section below to see the data of Pt-Mo50_input_images
#
# convImgDir = "./Data/Pt-Mo50_input_images/Pt281-Mo280-convolution"
# convPrefix = "Pt_Mo50_convolution_"
# convSuffix = ".txt"
# # then convPrefix + ID + convSuffix is the data file name for convolution images
#
# multiImgDir = "./Data/Pt-Mo50_input_images/Pt281-Mo280-multislice"
# multiPrefix = "pt_mo50_"
# multiSuffix = "_cl160mm_ss.tif"
# # then multiPrefix + ID + multiSuffix is the data file name for multislice images

"""
Pt_input_images

Notice ID is from 0 to 19
"""
#
# # You only need to uncomment the section below to see the data of Pt_input_images
#
# convImgDir = "./Data/Pt_input_images/Pt_convolution"
# convPrefix = "Pt_convolution_"
# convSuffix = ".txt"
# # then convPrefix + ID + convSuffix is the data file name for convolution images
#
# multiImgDir = "./Data/Pt_input_images/Pt_multislice_16_phonons"
# multiPrefix = "Pt_"
# multiSuffix = "_cl160mm_ss.tif"
# # then multiPrefix + ID + multiSuffix is the data file name for multislice images


def calculate_NRMSE_List(convImgDir,multiImgDir):
    """
    Control the calculation of NRMSE over certain directory

    Parameters
    ----------
    convImgDir : the directory stores image data of convolution images
    multiImgDir : the directory stores image data of multislice images

    Returns
    -------
    NRMSESummaryList : list that contains NRMSE of each images
    """
    numImages = len(os.listdir(convImgDir))
    NRMSESummaryList = list()
    # check starting ID for the edge case starting from 2
    if "Pt-Mo50_input_images" in convImgDir.split("/"):
        for id in range(2,numImages+1):
            NRMSESummaryList.append(NRMSE_two_images(convPrefix,convSuffix,convImgDir,multiPrefix,multiSuffix,multiImgDir,id))
    else:
        for id in range(1,numImages):
            NRMSESummaryList.append(NRMSE_two_images(convPrefix,convSuffix,convImgDir,multiPrefix,multiSuffix,multiImgDir,id))
    return NRMSESummaryList

def NRMSE_two_images(convPrefix,convSuffix,convImgDir,multiPrefix,multiSuffix,multiImgDir,id):
    """
    Calculate NRMSE of two images

    Parameters
    ----------
    convPrefix : the prefix of convolution image name
    convSuffix : the Suffix of convolution image name
    convImgDir : the directory stores image data of convolution images
    multiPrefix : the prefix of multislice image name
    multiSuffix : the Suffix of multislice image name
    multiImgDir : the directory stores image data of multislice images
    id : the current image id to compare

    Returns
    -------
    nrMSE : the NRMSE of current two images
    """
    convArr = np.loadtxt(convImgDir + "/" + convPrefix + str(id) + convSuffix)
    multiArr = io.imread(multiImgDir + "/" + multiPrefix + str(id) + multiSuffix)
    # check if the two array has the same shape
    # if not, something wrong with the data
    assert convArr.shape == multiArr.shape
    mseArr = ((normalize(convArr) - normalize(multiArr))**2) / (normalize(multiArr)**2)
    return np.sqrt(np.sum(mseArr) / (convArr.shape[0] * convArr.shape[1]))

def normalize(v):
    """
    Normalize the image 2D array of the image

    Parameters
    ----------
    v : input image 2D array that needs to be normalized

    Returns
    -------
    normailzed 2D array
    """
    return (v / np.sqrt((np.sum(v ** 2))))

if __name__ == '__main__':
    print("This is the program to calculate the normalized RMSE of two images")
    totalNRMSEList = calculate_NRMSE_List(convImgDir, multiImgDir)
    print("the average error over %d images is" % (len(totalNRMSEList) ))
    print( 1 -  (sum(totalNRMSEList) / len(totalNRMSEList) ) )


