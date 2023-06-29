import numpy as np


def xyz2latlon(x,y,z):
    """
    Transforms x,y,z cartesian coordinates into geographical lon,lat,r
    coordinates (in radians)
    """
    r     = np.sqrt(x**2+y**2+z**2)
    lat   = np.arctan2(np.sqrt(x**2+y**2),z)
    lon   = np.arctan2(y,x)
    return lat,lon,r


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
    Compute the rotation matrix R in cartesian 3D geometry for a rotation
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


def ecef2enu(lat,lon,vx,vy,vz):
    """
    Transform velocities from ECEF to ENU.
    ** Lat, Lon coordinates in RADIANS **
    """
    nod  = len(vx)
    vlon = np.zeros(nod)
    vlat = np.zeros(nod)
    vr   = np.zeros(nod)
    for i in range(nod):
        v_i = np.array([vx[i],vy[i],vz[i]])
        Rmatrix = Rgt(lat[i],lon[i])
        vlon[i],vlat[i],vr[i] = np.dot(Rmatrix,v_i)
    return vlon,-vlat,vr


def ecef2enu_stagYY(x,y,z,vx,vy,vz):
    """
    Transform the ECEF vectors (vx,vy,vz) into ENU vectors (Vphi,vtheta,vr)
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
    Substract the rotation defined by (wx,wy,wz) to the velocity field
    in input.
    """
    vxf,vyf,vzf = cartfipole(x,y,z,wx,wy,wz)
    vxo = vx-vxf
    vyo = vy-vyf
    vzo = vz-vzf
    vphio,vthetao,vro = ecef2enu_stagYY(x,y,z,vxo,vyo,vzo)
    return vxo, vyo, vzo, vphio, vthetao, vro






