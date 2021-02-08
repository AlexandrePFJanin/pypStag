# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:43:26 2019

@author: Alexandre JANIN
__All rights reserved__
"""

# pypStag import
import sys
sys.path.append('../')
from pypStag.stagData import StagData
from pypStag.stagInterpolator import sliceInterpolator
from pypStag.stagData import SliceData
from pypStag.stagViewer import stagCartesian3DMap, sliceMap
from pypStag.stagViewerMod import PlotParam




# ==========================================
# Slicing and mapping YinYang StagYY outputs
# ==========================================

# The following example shows you how to build a slice object
# (derived from pypStag.MainSliceData) from a Yin Yang stagData




# ------------------
# 1. Give file path:

directory = '../docs/test_data/'
fname = 'lowRh_t01300'


# ------------------
# 2. Build our stagData object:

sd = StagData(geometry='yy')
sd.verbose = True
sd.stagImport(directory, fname, resampling=[1,1,10])
sd.stagProcessing() 


# ------------------
# 2.bis If you want also a velocity field:
"""
sdv = StagData(geometry='yy')
sdv.stagImport(directory, fname.split('_')[0]+'_vp'+fname[-5:], resampling=[2,2,10])
sdv.stagProcessing()
"""

# ------------------
# 3. Creat a SliceData object with the same geometry as the geometry of your StagData object
sld = SliceData(geometry=sd.geometry)


# ------------------
# 4. Extract a slice on your StagData

layer = -1  # -1 will extract the surface, 0 the CMB ...
sld.sliceExtractor(sd,layer)


# ------------------
# 4.bis Extract a slice on your Velocity
"""
sldv = SliceData(geometry=sdv.geometry)
sldv.sliceExtractor(sdv,layer)
"""

#______________
# General Note
# 
# Slicing operation is done!
# Now your slice object contain buy computation and inheritance
# all information on your slice.
# The following code show you how to use stagViewer package to 
# plot your slice.


# ------------------
# 5. Build an InterpolatedSliceData object from your SliceData

isd = sliceInterpolator(sld,deg=True,spacing=0.05)


# ------------------
# 5.bis The same for your velocity (if you want)
"""
isdv = sliceInterpolator(sldv,deg=True,spacing=1)
"""

# ------------------
# 6. Build or plot parameters from the pypStag.stagViewerMod <optional>

ppar_t   = PlotParam(log10=False,cmap='vik',reverseCMAP=False,minVal=0,maxVal=1,show=True,figsize=(10,10))


# ------------------
# 7. Call sliceMap function from stagViewer

# Projection package
import cartopy.crs as ccrs

sliceMap(isd,sliceVelocity=None,vresampling=[5,5],veloWidth=8e-4,plotparam=ppar_t,aspect_ratio=1,Qscale=1000,projection=ccrs.Robinson())


# ------------------
# 7.bis Call sliceMap function from stagViewer with a slice of the corresponding velocity field
"""
# Projection package
import cartopy.crs as ccrs

ppar_eta = PlotParam(log10=True,cmap='vik',reverseCMAP=True,minVal=0,maxVal=4,show=True,figsize=(10,10))
ppar_t   = PlotParam(log10=False,cmap='vik',reverseCMAP=False,minVal=0,maxVal=1,show=True,figsize=(10,10))

sliceMap(isd,sliceVelocity=isdv,vresampling=[5,5],veloWidth=8e-4,plotparam=ppar_t,aspect_ratio=1,Qscale=1000,projection=ccrs.Robinson())
"""

