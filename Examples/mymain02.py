# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:43:26 2019

@author: Alexandre JANIN
__All rights reserved__
"""

# pypStag import
import sys
sys.path.append('../')
from pypStag.stagData import StagData, StagCloudData
from pypStag.stagVTK import stagCloud2timeVTU




# ==========================================
# Creation of a stagCloudData for 3D grid and generate time .vtu 
# ==========================================

# The following example shows you how to build a stagClouData object
# and how to generate with this object a time series of meshed files
# for paraview




# ------------------
# 1. Give file path:

directory = '../docs/test_data/'  # directory where all files are stored
gfname    = "SRW46mpi3D_eta%s"    # generic name of all files
indices   = list([520,523])         # list of indices that pypStag have to consider with gfname


# ------------------
# 2. Build our stagCloudData object:

scd = StagCloudData(geometry='cart3D')
scd.build(directory,gfname,indices=indices)


# ------------------
# 3. Creat your export

fname     = 'my_test'   # file name of the final export
path      = './'        # path where all data will be exported
timepar   = 0           # time values will be file index, see function doc
multifile = False       # if you want to split data in several .h5 files

stagCloud2timeVTU(fname,scd,multifile=False,timepar=0,path='./',verbose=True)

