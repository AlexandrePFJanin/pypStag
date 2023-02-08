# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:31:48 2019

@author: Alexandre JANIN
"""

import numpy as np
from .stagData import StagData, StagCartesianGeometry, StagYinYangGeometry, StagSphericalGeometry
from .stagData import SliceData, CartesianSliceData, YinYangSliceData
from .stagData import InterpolatedSliceData

from .stagError import GridInterpolationError,StagMapUnknownFieldError,StagMapFieldError


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
    nbinLon = int(360/spacing)
    nbinLat = int((180)/spacing)
    nbinR   = 1
    lon = np.linspace(-180,180,nbinLon)*np.pi/180
    lon = lon[0:-1]  # remove the redondant point at 360 deg
    lat = np.linspace(-90,90,nbinLat)*np.pi/180
    r   = [radius]*nbinR
    #2. Mesh grid
    (Lon,Lat) = np.meshgrid(lon,lat,indexing='ij')
    R = np.zeros(Lon.shape)
    #3. Projection on cartesian coordinates
    x = R*np.cos(Lat)*np.cos(Lon)
    y = R*np.cos(Lat)*np.sin(Lon)
    z = R*np.sin(Lat)
    return ((x,y,z),(R,Lon,Lat))


def regularAnnulusGrid(innerradius=1.2,outerradius=2.2,ntheta=128,nz=64):
    """Return a regular annulus grid for a single depth. The annulus will
    extend from the innerradius to the outerradius and will have ntheta points
    for each of the nz depth layers.
    <i> : innerradius = int/float, depth of our CMB (default is set like StagYY to 1.19)
          outerradius = int/float, depth of the surface (default is set like StagYY to 2.19)
          ntheta      = int, number of points in the theta direction
          nz          = int, number of points in the z (r) direction
    <o> : ((x,y),(r,theta))
          x,y = the 2D cartesian coordinates of points in the new grid.
          theta,r = the 2D spherical coordinates of the points in the new grid."""
    #define base rectangle (r,theta) = (u,v)
    theta = np.linspace(0, 2*np.pi, ntheta)
    r     = np.linspace(innerradius, outerradius, nz)
    theta,r = np.meshgrid(theta,r)
    #evaluate the parameterization at the flattened u and v
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return (x,y),(r,theta)



def sliceInterpolator(sliceData,interpGeom='rgS',\
    spacing=1,innerradius=1.19,outerradius=2.19,ntheta=128,nz=64,\
    interpMethod='nearest',deg=False,verbose=True):
    """
    Interpolates a stagData.YinYangSliceData object in an other grid.
    <i> : sliceData  = stagData.YinYangSliceDat object
          interpGeom = str, indicates the type of geometry used for the new
                       grid. (in {rgS,rgA})
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
        npx, npy, npz = x.shape[0], x.shape[1], 0
        Xrg = x.reshape(x.shape[0]*x.shape[1])
        Yrg = y.reshape(y.shape[0]*y.shape[1])
        Zrg = z.reshape(z.shape[0]*z.shape[1])
        im('    - Spacing for grid : '+str(spacing),pName,verbose)
        im('    - Number of Points : '+str(len(Xrg)),pName,verbose)
    elif interpGeom == 'rgA':
        #Regular annulus grid
        (x,y),(rg,thetag) = regularAnnulusGrid(innerradius=innerradius,outerradius=outerradius,ntheta=ntheta,nz=nz)
        z   = np.ones(x.shape)*sliceData.z[0]
        npx, npy, npz = x.shape[0], x.shape[1], 1
        Xrg = x.reshape(npx*npy)
        Yrg = y.reshape(npx*npy)
        Zrg = np.ones((npx*npy))*sliceData.z[0]
        im('    - ntheta,nz        : '+str(ntheta)+','+str(nz),pName,verbose)
        im('    - Number of Points : '+str(len(Xrg)),pName,verbose)
    else:
        raise GridInterpolationError(interpGeom)
    
    #2. Preparation of the interpolation:
    if interpGeom == 'rgS':
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
        isd.geom         = interpGeom
        isd.interpGeom   = interpGeom
        isd.spacing      = spacing
        isd.innerradius  = None
        isd.outerradius  = None
        isd.ntheta       = None
        isd.nz           = None
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
        # Add coordinates:
        isd.x = Xrg
        isd.y = Yrg
        isd.z = Zrg
        isd.lon = lon*180/np.pi
        isd.lat = lat*180/np.pi
        isd.r   = r
    
    elif interpGeom == 'rgA':
        im('Interpolation of the slice:',pName,verbose)
        X = sliceData.x
        Y = sliceData.y
        im('    - Normal vector                : '+str(sliceData.normal),pName,verbose)
        im('    - Number of Points in the slice: '+str(len(X)),pName,verbose)
        points = np.array([(X[i],Y[i]) for i in range(len(X))])
        # Stores all in an stagData.InterpolatedSliceData object
        isd = InterpolatedSliceData()
        isd.sliceInheritance(sliceData)
        isd.nxi, isd.nyi, isd.nzi = npx, npy, npz
        isd.interpGeom   = interpGeom
        isd.spacing      = None
        isd.innerradius  = innerradius
        isd.outerradius  = outerradius
        isd.ntheta       = ntheta
        isd.nz           = nz
        isd.interpMethod = interpMethod
        isd.deg          = deg
        # Scalar or Vectorial
        if sliceData.fieldNature == 'Scalar':
            im('    - Interpolation of a Sclar field',pName,verbose)
            values = np.array(sliceData.v)
            isd.v = griddata(points, values, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            im('Interpolation done for the slice !',pName,verbose)
            im('    - Duration of interpolation: '+str(time()-time0)[0:5]+' s',pName,verbose)
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
            isd.vx = griddata(points, values_vx, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            isd.vy = griddata(points, values_vy, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            isd.vz = griddata(points, values_vz, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            isd.P = griddata(points, values_P, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            isd.vtheta = griddata(points, values_vtheta, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            isd.vphi = griddata(points, values_vphi, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            isd.vr = griddata(points, values_vr, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            isd.v = griddata(points, values_v, (Xrg, Yrg), method=interpMethod).reshape(npx,npy)
            im('Interpolation done for the slice !',pName,verbose)
            im('    - Duration of interpolation: '+str(time()-time0)[0:5]+' s',pName,verbose)
        # Add coordinates:
        isd.x = Xrg.reshape(npx,npy)
        isd.y = Yrg.reshape(npx,npy)
        isd.z = Zrg.reshape(npx,npy)
        isd.r = rg.reshape(npx,npy)
        isd.theta = thetag.reshape(npx,npy)
        isd.phi = np.zeros((npx,npy))
    return isd







def sliceYYInterpolator_mapping(sliceData,field,spacing=1,\
    interpMethod='nearest',verbose=True,log10=False,deg=True):
    """
    Interpolates a stagData.YinYangSliceData object in an other grid.
    Function dedicated to be call in pypStag.stagViewer
    <i> : sliceData  = stagData.YinYangSliceDat object
          spacing    = int, parameter of the interpGeom
          interpMethod = str, method used for the interpolation. In ('nearest' 
                         'linear', 'cubic'). Default = 'nearest'
          verbose = bool, controls inhibition of the internal message
    <o> : Return the regular lon/lat grid in deg and the interpolated field
    """
    time0 = time() #Init time for log message
    pName = 'sliceYYInterpolator_mapping'
    #1. Creation of the new grid
    im('Creation of interpGeom Grid',pName,verbose)
    ((x,y,z),(r,lon,lat)) = regularSphericalGrid(radius=sliceData.r[0],spacing=spacing)
    im('    - Spacing for grid : '+str(spacing),pName,verbose)
    im('    - Number of Points : '+str(lon.shape[0]*lon.shape[1]),pName,verbose)

    im('Interpolation of the slice:',pName,verbose)
    sldLon = sliceData.phi# + np.pi
    sldLat = -(sliceData.theta - np.pi/2)
    im('    - Slice layer index            : '+str(sliceData.layer),pName,verbose)
    im('    - Corresponding depth          : '+str(sliceData.depth),pName,verbose)
    im('    - Number of Points in the slice: '+str(len(sldLon)),pName,verbose)
    points = np.zeros((sliceData.phi.shape[0],2))
    points[:,0] = sldLon
    points[:,1] = sldLat
    
    # --- Test the field
    if field == 'scalar' or field == 'v':
        stagfield = sliceData.v
    elif field == 'vx':
        if sliceData.fieldNature == 'Vectorial':
            stagfield = sliceData.vx
        else:
            raise StagMapFieldError(field,sliceData.geometry,sliceData.fieldNature)
    elif field == 'vy':
        if sliceData.fieldNature == 'Vectorial':
            stagfield = sliceData.vy
        else:
            raise StagMapFieldError(field,sliceData.geometry,sliceData.fieldNature)
    elif field == 'vz':
        if sliceData.fieldNature == 'Vectorial':
            stagfield = sliceData.vz
        else:
            raise StagMapFieldError(field,sliceData.geometry,sliceData.fieldNature)
    elif field == 'vphi':
        if sliceData.fieldNature == 'Vectorial':
            stagfield = sliceData.vphi
        else:
            raise StagMapFieldError(field,sliceData.geometry,sliceData.fieldNature)
    elif field == 'vtheta':
        if sliceData.fieldNature == 'Vectorial':
            stagfield = sliceData.vtheta
        else:
            raise StagMapFieldError(field,sliceData.geometry,sliceData.fieldNature)
    elif field == 'vr':
        if sliceData.fieldNature == 'Vectorial':
            stagfield = sliceData.vr
        else:
            raise StagMapFieldError(field,sliceData.geometry,sliceData.fieldNature)
    elif field == 'p' or field == 'P':
        if sliceData.fieldNature == 'Vectorial':
            stagfield = sliceData.P
        else:
            raise StagMapFieldError(field,sliceData.geometry,sliceData.fieldNature)
    else:
        raise StagMapUnknownFieldError(field)
    # --- Log10:
    if log10:
        stagfield = np.log10(stagfield)
        
    # Scalar or Vectorial
    v = griddata(points, stagfield, (lon, lat), method=interpMethod)
    im('Interpolation done for the slice !',pName,verbose)
    im('    - Duration of interpolation: '+str(time()-time0)[0:5]+' s',pName,verbose)
    
    # conversion to deg
    if deg:
        lon = lon * 180/np.pi
        lat = lat * 180/np.pi
    
    return lon,lat,v
    