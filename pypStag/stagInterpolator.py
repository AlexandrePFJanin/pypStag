# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:31:48 2019

@author: Alexandre JANIN
"""

import numpy as np
from .stagData import StagData, StagCartesianGeometry, StagYinYangGeometry
from .stagData import SliceData, CartesianSliceData, YinYangSliceData
from .stagData import InterpolatedSliceData

from .stagError import GridInterpolationError

from scipy.interpolate import griddata
from time import time



def im(textMessage,pName,verbose):
    """Print verbose internal message. This function depends on the
    argument of self.verbose. If self.verbose == True then the message
    will be displayed on the terminal.
    <i> : textMessage = str, message to display
          pName = str, name of the subprogram
          verbose = bool, condition for the verbose output
    """
    if verbose == True:
        print('>> '+pName+'| '+textMessage)
    


def regularSphericalGrid(radius,spacing=1):
    """Return a regular spherical grid for a single depth. The regularity is 
    guaranted on longitude and latitude. e.g: For a spacing parameter in
    input of 1, the regular grid produced in return will have 1 point per
    deg in lon and in lat and the same radius to the center of the sphere.
    <i> : spacing = int, number of degree in longitude and latitude between
                  each point of the grid
    <o> : ((x,y,z),(R,Lon,Lat)) where (x,y,z) car the cartesian coordinates
          of points in the new grid and (R,Lon,Lat) the spherical coordinates
          of points in the new grid."""
    #First, creation of point in spherical coordinates
    nbinLon = int(361/spacing)
    nbinLat = int((181)/spacing)
    nbinR   = 1
    lon = np.linspace(0,360,nbinLon)*np.pi/180
    lat = np.linspace(-90,90,nbinLat)*np.pi/180
    r   = [radius]*nbinR
    #2. Mesh grid
    (Lon,Lat,R) = np.meshgrid(lon,lat,r,indexing='ij')
    #3. Projection on cartesian coordinates
    x = R*np.cos(Lat)*np.cos(Lon)
    y = R*np.cos(Lat)*np.sin(Lon)
    z = R*np.sin(Lat)
    return ((x,y,z),(R,Lon,Lat))



def sliceInterpolator(sliceData,interpGeom='rgS',spacing=1,interpMethod='nearest',deg=False,verbose=True):
    """
    Interpolates a stagData.YinYangSliceData object in an other grid.
    <i> : sliceData  = stagData.YinYangSliceDat object
          interpGeom = str, indicates the type of geometry used for the new
                       grid. (in {rgS})
          spacing    = int, parameter of the interpGeom
          interpMethod = str, method used for the interpolation. In ('nearest' 
                         'linear', 'cubic'). Default = 'nearest'
          deg = bool, for interpGeom == 'rgS' only ! if deg is True, then the
                x,y,z on output will be lon,lat,r repsectivelly
          verbose = bool, controls inhibition of the internal message
    <o> : Return a stagData.InterpolatedSliceData objet
    """
    time0 = time() #Init time for log message
    if verbose:
        print()
    pName = 'sliceInterpolator'
    #1. Creation of the new grid
    im('Creation of interpGeom Grid',pName,verbose)
    if interpGeom == 'rgS':
        #Regular Grid - Spherical 1:
        ((x,y,z),(r,lon,lat)) = regularSphericalGrid(radius=sliceData.r[0],spacing=spacing)
        npx, npy, npz = x.shape[0], x.shape[1], x.shape[2]
        Xrg = x.reshape(x.shape[0]*x.shape[1]*x.shape[2])
        Yrg = y.reshape(y.shape[0]*y.shape[1]*y.shape[2])
        Zrg = z.reshape(z.shape[0]*z.shape[1]*z.shape[2])
    else:
        raise GridInterpolationError(interpGeom)
    im('    - Spacing for grid : '+str(spacing),pName,verbose)
    im('    - Number of Points : '+str(len(Xrg)),pName,verbose)
    #2. Preparation of the interpolation:
    im('Interpolation of the slice:',pName,verbose)
    X = sliceData.x
    Y = sliceData.y
    Z = sliceData.z
    im('    - Slice layer index            : '+str(sliceData.layer),pName,verbose)
    im('    - Corresponding depth          : '+str(sliceData.depth),pName,verbose)
    im('    - Number of Points in the slice: '+str(len(X)),pName,verbose)
    points = np.array([(X[i],Y[i],Z[i]) for i in range(len(X))])
    # Stores all in an stagData.InterpolatedSliceData object
    isd = InterpolatedSliceData()
    isd.sliceInheritance(sliceData)
    isd.nxi, isd.nyi, isd.nzi = npx, npy, npz
    isd.interpGeom   = interpGeom
    isd.spacing      = spacing
    isd.interpMethod = interpMethod
    isd.deg          = deg
    # Scalar or Vectorial
    if sliceData.fieldNature == 'Scalar':
        im('    - Interpolation of a Sclar field',pName,verbose)
        values = np.array(sliceData.v)
        isd.v = griddata(points, values, (Xrg, Yrg, Zrg), method=interpMethod)
        im('Interpolation done for the slice !',pName,verbose)
        im('    - Duration of interpolation: '+str(time()-time0)[0:5]+' s',pName,verbose)
        if deg:
            y = lat*180/np.pi
            x = lon*180/np.pi
            z = r
            Xrg = x.reshape(x.shape[0]*x.shape[1]*x.shape[2])
            Yrg = y.reshape(y.shape[0]*y.shape[1]*y.shape[2])
            Zrg = z.reshape(z.shape[0]*z.shape[1]*z.shape[2])
    else: #Vectorial field
        im('    - Interpolation of a Vectorial field: can take time',pName,verbose)
        values_vx = np.array(sliceData.vx)
        values_vy = np.array(sliceData.vy)
        values_vz = np.array(sliceData.vz)
        values_P = np.array(sliceData.P)
        values_vtheta = np.array(sliceData.vtheta)
        values_vphi = np.array(sliceData.vphi)
        values_vr = np.array(sliceData.vr)
        values_v  = np.array(sliceData.v)
        isd.vx = griddata(points, values_vx, (Xrg, Yrg, Zrg), method=interpMethod)
        isd.vy = griddata(points, values_vy, (Xrg, Yrg, Zrg), method=interpMethod)
        isd.vz = griddata(points, values_vz, (Xrg, Yrg, Zrg), method=interpMethod)
        isd.P = griddata(points, values_P, (Xrg, Yrg, Zrg), method=interpMethod)
        isd.vtheta = griddata(points, values_vtheta, (Xrg, Yrg, Zrg), method=interpMethod)
        isd.vphi = griddata(points, values_vphi, (Xrg, Yrg, Zrg), method=interpMethod)
        isd.vr = griddata(points, values_vr, (Xrg, Yrg, Zrg), method=interpMethod)
        isd.v = griddata(points, values_v, (Xrg, Yrg, Zrg), method=interpMethod)
        im('Interpolation done for the slice !',pName,verbose)
        im('    - Duration of interpolation: '+str(time()-time0)[0:5]+' s',pName,verbose)
        if deg:
            im('Requested: Conversion of the grid into degree',pName,verbose)
            y = lat*180/np.pi
            x = lon*180/np.pi
            z = r
            Xrg = x.reshape(x.shape[0]*x.shape[1]*x.shape[2])
            Yrg = y.reshape(y.shape[0]*y.shape[1]*y.shape[2])
            Zrg = z.reshape(z.shape[0]*z.shape[1]*z.shape[2])
    isd.x = Xrg
    isd.y = Yrg
    isd.z = Zrg
    return isd
