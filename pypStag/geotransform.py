# -*- coding: utf-8 -*-
"""
@author: Alexandre Janin
@aim:    geographic transformation module
"""

# External dependencies:
import numpy as np

# Internal dependencies:
from .generics import im
from .errors import StagTypeError


# ----------------- FUNCTIONS -----------------


def latlon2xyz(lat,lon,R):
    """
    Returns the X,Y,Z cartesian coordinates in the ECEF reference frame
    of a point described by a geodetic coordinates (lat,lon) and the radius
    of the sphere R

    lat,lon in RADIANS
    """
    x = R*np.cos(lat)*np.cos(lon)
    y = R*np.cos(lat)*np.sin(lon)
    z = R*np.sin(lat)
    return x,y,z


def xyz2latlon(x,y,z):
    """
    Transforms x,y,z cartesian coordinates into geographical (colat,lon,r)
    coordinates (in radians).
    """
    r     = np.sqrt(x**2+y**2+z**2)
    colat = np.arctan2(np.sqrt(x**2+y**2),z)
    lon   = np.arctan2(y,x)
    return colat,lon,r


def bend_rectangular2Spherical_YY(x,y,z,rcmb):
    """
    Returns the coordinates of points of the Yin
    and Yang grids after bending the original
    cartesian boxes.
    """
    #Spherical coordinates:
    R = z+rcmb
    lat = np.pi/4 - x
    lon = y - 3*np.pi/4
    #Yin grid
    x1,y1,z1 = latlon2xyz(lat,lon,R)
    #Yang grid
    x2 = -x1
    y2 = z1
    z2 = y1
    return ((x1,y1,z1),(x2,y2,z2))


def bend_rectangular2Spherical(x,y,z,rcmb,shell='yin'):
    """
    Returns the coordinates of points of the
    spherical grid after bending the cartesian box"""
    #Spherical coordinates:
    R = z+rcmb
    lat = np.pi/4 - x
    lon = y - 3*np.pi/4
    #Spherical grid
    x1,y1,z1 = latlon2xyz(lat,lon,R)
    # which shell
    if shell == 'yin':
        x = x1
        y = y1
        z = z1
    elif shell == 'yang':
        # Rotate
        x = -x1
        y = z1
        z = y1
    else:
        print('Unknown shell')
        raise ValueError()
    return (x,y,z)


def cartfipole(x,y,z,wx,wy,wz):
    """
    wx,wy,wz in rad/Myrs
    returns vx,vy,vz
    """
    vx =  0*wx +  z*wy + -y*wz
    vy = -z*wx +  0*wy +  x*wz
    vz =  y*wx + -x*wy +  0*wz
    return vx,vy,vz


def Rgt(lat,lon):
    """
    Rgt =  geocentric to topocentric rotation matrix

    For Venu = (Ve,Vn,Vz) and Vxyz = (Vx,Vy,Vz)

    Venu = np.dot(Rgt,Vxyz)
    Vxyz = np.dot(Rgt_inv,Venu)

    ** Lat, Lon coordinates in RADIANS **
    """
    return np.array([[-np.sin(lon),np.cos(lon),0],\
                     [-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],\
                     [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])


def rotation_matrix_3D(axis,theta):
    """
    Computes the rotation matrix R in cartesian 3D geometry for a rotation
    on x, y or z axis and a rotation angle theta
    <i> axis = str, 'x' for a rotation around the x axis
                    'y' for a rotation around the y axis
                    'z' for a rotation around the z axis
        theta = int/float, rotation angle in *RADIANS*
    NOTE: Application:
    If you have a vector A that you want to rotate aroud the X axis
    with a rotation angle theta = 45 deg, write:

    >> R     = rotation_matrix_3D('x',45*np.pi/180)
    >> A_rot = np.dot(R,A)
    """
    if axis == 'x':
        R = np.array([[             1,             0,            0],\
                      [             0, np.cos(theta),-np.sin(theta)],\
                      [             0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        R = np.array([[ np.cos(theta),             0, np.sin(theta)],\
                      [             0,             1,            0],\
                      [-np.sin(theta),             0, np.cos(theta)]])
    elif axis == 'z':
        R = np.array([[ np.cos(theta),-np.sin(theta),            0],\
                      [ np.sin(theta), np.cos(theta),            0],\
                      [             0,             0,            1]])
    return R


def ecef2enu_stagYY(x,y,z,vx,vy,vz):
    """
    Transform the ECEF vectors (vx,vy,vz) into StagYY ENU vectors (Vphi,vtheta,vr)
    Here, vtheta is -vlat (as computed in stagData)
    """
    lat = np.arctan2(np.sqrt(x**2+y**2),z) # in reality, it is the colatitude: so it is why, the following formula
                                           # is not exactly the same as in textbooks!! sin(lat) -> -cos(lat) and cos(lat) -> -sin(lat)
    lon = np.arctan2(y,x)
    vtheta =  vx*(np.cos(lon)*np.cos(lat)) + vy*(np.sin(lon)*np.cos(lat)) - vz*(np.sin(lat))
    vphi   = -vx*(np.sin(lon))             + vy*(np.cos(lon))
    vr     = -vx*(np.cos(lon)*np.sin(lat)) - vy*(np.sin(lon)*np.sin(lat)) - vz*(np.cos(lat))
    vr = -vr
    return vphi,vtheta,vr


def velocity_pole_projecton(x,y,z,vx,vy,vz,wx,wy,wz):
    """
    Substract the angular rotation defined by (wx,wy,wz) to the velocity field
    in input. wx,wy,wz in RAD/MYR
    """
    vxf,vyf,vzf = cartfipole(x,y,z,wx,wy,wz)
    vxo = vx-vxf
    vyo = vy-vyf
    vzo = vz-vzf
    vphio,vthetao,vro = ecef2enu_stagYY(x,y,z,vxo,vyo,vzo)
    return vxo, vyo, vzo, vphio, vthetao, vro



def get_xzy_scoords(stagData,lon,lat,verbose=True):
    """
    Function that returns the cartesian x,y and z coordinates of the point
    located at the surface of a Yin Yang model that is the closest from
    the input lon, lat coordinates.
    
    <i> : stagData (pypStag.stagData.StagYinYangGeometry): input data
          lon,lat (float,float): coordinates [in DEGREES] of the point
    """
    if stagData.geometry != 'yy':
        raise StagTypeError(str(type(stagData)),'pypStag.stagData.StagYinYangGeometry')
    else:
        from .stagData  import SliceData
        pName = 'get_xzy_scoords'
        im('Search the closest point to:',pName,verbose=verbose)
        im('lon = '+str(lon)+', lat = '+str(lat),pName,verbose=verbose)
        sld = SliceData(geometry=stagData.geometry)
        sld.verbose = False
        sld.slicing(stagData,axis='layer',normal=None,layer=-1)
        long = sld.phi*180/np.pi
        latg = -(sld.theta*180/np.pi-90)
        dist = np.sqrt((long-lon)**2+(latg-lat)**2)
        gind = np.where(dist == np.amin(dist))[0][0]
        x,y,z = sld.x[gind],sld.y[gind],sld.z[gind]
        im('   -> Point index: '+str(gind),pName,verbose=verbose)
        im('   -> Found lon/lat: '+str(int(long[gind]*100)/100)+', '+str(int(latg[gind]*100)/100),pName,verbose=verbose)
        return x,y,z



def get_normal_2points(stagData,point1,point2,verbose=True):
    """
    Computes the normal vector of the plan passing by (lon1,lat1), (lon2,lat2) and
    by the origin (the center of the Earth: 0,0,0). The norm of this vector is set to 1.
    This function have been made to be used with the slicing package to define slicing plan.
    
    <i> : stagData (pypStag.stagData.StagYinYangGeometry): input data
          point1 = [lon1,lat1] = [float,float] coordinates [in DEGREES] of the point 1
          point2 = [lon2,lat2] = [float,float] coordinates [in DEGREES] of the point 2
    """
    pName = 'get_normal_2points'
    lon1,lat1 = point1
    lon2,lat2 = point2
    im('Compute the normal vector of the plan passing by:',pName,verbose=verbose)
    im('   -> lon1,lat1 = '+str(lon1)+', '+str(lat1),pName,verbose=verbose)
    im('   -> lon2,lat2 = '+str(lon2)+', '+str(lat2),pName,verbose=verbose)
    im('   -> x0,y0,z0  = 0, 0, 0',pName,verbose=verbose)
    # --- Compute the xyz coordinates of the two points
    x1,y1,z1 = get_xzy_scoords(stagData,lon1,lat1,verbose=verbose)
    x2,y2,z2 = get_xzy_scoords(stagData,lon2,lat2,verbose=verbose)
    # --- Formulation of the problem under the form AX = B
    A = np.array([[x1, y1, z1],\
                  [x2, y2, z2],\
                  [1,  1,  1]]).T
    normal = np.dot(np.array([0,0,1]),np.linalg.inv(A))
    im('Done',pName,verbose=verbose)
    return normal
