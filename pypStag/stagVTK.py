# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:43:26 2019

@author: Alexandre Janin
"""

"""
This script contains routines composing the pypStag Visualisation ToolKit
"""

from .stagData import StagData
from .stagError import *
import numpy as np
import h5py
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay




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
   




def scalar2VTU(fname,x,y,z,v,Nz,fieldName,path='./',ASCII=False,return_only=False,creat_pointID=False,verbose=True):
    """ This function creat '.vtu' file readable with Paraview to efficiently 
    visualize 3D data simply defined with grid matrix (x,y,z) and a field v 
    as well as a number of layer Nz.
    <i> : fname     = str, name of the exported file without any extention
          x         = list/numpy.ndarray, matrix of the cartesian x coordinates (flatten)
          y         = list/numpy.ndarray, matrix of the cartesian y coordinates (flatten)
          z         = list/numpy.ndarray, matrix of the cartesian z coordinates (flatten)
          v         = list/numpy.ndarray, matrix of the field (flatten)
          Nz        = int, dimension of the grid in the z direction, aka number of z layers
          fieldName = str, name of the field for Paraview
          path      = str, path where you want to export your new .vtu file.
                      [Default: path='./']
          ASCII     = bool, if True, the .vtu file will be write in ASCII mode
                      if not, in binary mode. [Default, ASCII=False]
          verbose   = bool, if True activates the verbose output
    """
    pName = 'scalar2VTU'
    if verbose:
        print()
    im('Scalar field Visualization ToolKit',pName,verbose)
    if ASCII:
        im('Requested: stagData -> .vtu',pName,verbose)
    else:
        im('Requested: stagData -> .xdmf + .h5',pName,verbose)
    im('    - Grid preparation',pName,verbose)
    # =======================================
    #Re-formating initial grids data
    X_yy  = np.array(x)
    Y_yy  = np.array(y)
    Z_yy  = np.array(z)
    #reformating
    Nz   = Nz                         #Number of depth layers
    NxNy = int(len(x)/Nz)   #Number of points for each layers
    x1    = X_yy.reshape(NxNy,Nz)
    X_yy  = x1.tolist()
    y1    = Y_yy.reshape(NxNy,Nz)
    Y_yy  = y1.tolist()
    z1    = Z_yy.reshape(NxNy,Nz)
    Z_yy  = z1.tolist()
    #Re-organisation of data to have X,Y and Z grid matrices organized by depths:
    X = []
    Y = []
    Z = []
    for j in range(Nz):
        x1t = [x1[i][j] for i in range(NxNy)]
        X.append(x1t)
        y1t = [y1[i][j] for i in range(NxNy)]
        Y.append(y1t)
        z1t = [z1[i][j] for i in range(NxNy)]
        Z.append(z1t)    
    # =========================================================================
    # 1) Take the surface of the 2 grids, patch together and triangulate
    # =========================================================================
    im('    - Triangulation on convex hull',pName,verbose)
    # NotaBene: _s for the surface layer
    X_s    = X[Nz-1]
    Y_s    = Y[Nz-1]
    Z_s    = Z[Nz-1]
    # =======================================
    # Triangulation of the surface using a convex hull algorithm
    points      = [[X_s[i],Y_s[i],Z_s[i]] for i in range(len(X_s))]
    triYingYang = ConvexHull(points).simplices # simple way to grid it
    # =========================================================================
    # 2) Create a 3D grid with tetrahedron elements
    # =========================================================================
    # Number all gridpoints we have
    NUM         = np.array(range(0,NxNy*Nz))
    NUMBER      = NUM.reshape((NxNy,Nz), order='F')
    # Make a loop over all levels
    ElementNumbers      = []
    for iz in range(Nz-1):
        num_upper      = NUMBER[:,iz+1]
        num_lower      = NUMBER[:,iz]
        num_tri = [[num_upper[t[0]], \
                    num_upper[t[1]], \
                    num_upper[t[2]], \
                    num_lower[t[0]], \
                    num_lower[t[1]], \
                    num_lower[t[2]]] for t in triYingYang]
        ElementNumbers.extend(num_tri)
    # =======================================
    # Convert data into correct vector format
    im('    - Convert data into correct vector format',pName,verbose)
    Points = [list(np.array(x1).reshape((NxNy*Nz), order='F')), \
              list(np.array(y1).reshape((NxNy*Nz), order='F')), \
              list(np.array(z1).reshape((NxNy*Nz), order='F'))]
    Points = np.array(Points).transpose()
    # ===================
    V_yy  = np.array(v).reshape(NxNy,Nz)
    vstack = list(V_yy.reshape((NxNy*Nz), order='F'))
    # =========================================================================
    mysd = StagData(geometry='cart3D')
    mysd.fieldNature = 'Scalar'
    mysd.fieldType   = fieldName
    if creat_pointID:
        im('      - Creat pointID',pName,verbose)
        pointID = np.array(range(NxNy*Nz),dtype=np.int32).reshape(NxNy*Nz, order='F')        
        pointID = pointID.reshape((NxNy*Nz), order='F')
    else:
        pointID = None
    if not return_only:
        # Exportation under VTKUnstructuredGrid format
        if ASCII:
            im('    - Writing under .vtu format',pName,verbose)
        else:
            im('    - Writing under .xdmf + .h5 formats',pName,verbose)
        __writeVKTStag(fname,mysd,Points,ElementNumbers,vstack,ASCII=ASCII,path=path,pointID=pointID)
        im('Exportation done!',pName,verbose)
        if ASCII:
            im('File: '+fname+'.vtu',pName,verbose)
            im('Path: '+path,pName,verbose)
        else:
            im('Files: '+fname+'.xdmf + '+fname+'.h5',pName,verbose)
            im('Path : '+path,pName,verbose)
    else:
        # Return all grid/mesh elements
        return Points,ElementNumbers,vstack,pointID





def surface2VTK(x,y,z,v,fname,fieldName,Nz=1,path='./',verbose=True,simplex_threshold=0.02):
    """ Function writting a Paraview VTK (XDMF+H5) file for a surface described by
    3D (flatten) cartesian coordinates.

    Args:
        x (ndarray): x coordinates. Shape: flatten
        y (ndarray): y coordinates. Shape: flatten
        z (ndarray): z coordinates. Shape: flatten
        v (ndarray): Field data. Shape: flatten
        fname (str): Output filename without formati extension
        fieldName (str): Name of the field (v) that will be displayed with Paraview
        path (str, optional): Path to the output directory. Defaults to './'.
        verbose (str, optional): Verbose option. Defaults to True.
    """
    pName = 'surface2VTK'
    im('Exportation of 3D surface data points to meshed data for Paraview',pName,verbose)
    
    if path[-1] != '/':
        path += '/'
    
    xdmf_file = fname + '.xdmf'
    h5_file   = fname + '.h5'

    nod = len(x)
    points = np.zeros((nod,3))
    points[:,0] = x
    points[:,1] = y
    points[:,2] = z

    im('  -> Delaunay Triangulation',pName,verbose)
    tri = Delaunay(points[:,0:2])
    
    # Simplification of the Delaunay triangulation: Remove long connections
    simplices = tri.simplices
    dist = np.zeros(simplices.shape)
    mask = np.ones(simplices.shape[0],dtype=bool)
    for i in range(simplices.shape[0]):
        a,b,c = simplices[i,:]
        dist[i,0] = np.sqrt((x[a]-x[b])**2+(y[a]-y[b])**2+(z[a]-z[b])**2)
        dist[i,1] = np.sqrt((x[b]-x[c])**2+(y[b]-y[c])**2+(z[b]-z[c])**2)
        dist[i,2] = np.sqrt((x[c]-x[a])**2+(y[c]-y[a])**2+(z[c]-z[a])**2)
        if np.count_nonzero(dist[i,:]>simplex_threshold) > 0:
            mask[i] = False

    if np.unique(simplices[mask,:]).shape[0] == nod:
        im('No point missed',pName,verbose)
        simplices = simplices[mask,:]
    else:
        im('   * ERROR *',pName,verbose)
        im('Process aborded!',pName,verbose)
        im('Point(s) missed in the simplification of the Delaunay triangulation',pName,verbose)
        im('  -> Reduce the argument simplex_threshold',pName,verbose)
        return 0

    im('  -> Writing the instruction file (XDMF)',pName,verbose)
    fid = open(path+xdmf_file,'w')
    fid.write('<?xml version="1.0" ?>'+'\n')
    fid.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'+'\n')
    fid.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">'+'\n')
    fid.write('  <Domain>'+'\n')
    fid.write('    <Grid GridType="Uniform">'+'\n')
    fid.write('      <Topology TopologyType="Triangle" Dimensions="%s">'%int(simplices.shape[0])+'\n')
    fid.write('        <DataItem Dimensions="%s'%int(simplices.shape[0])+' 3" NumberType="Int" Precision="8" Format="HDF">'+h5_file+':/Topology</DataItem>'+'\n')
    fid.write('      </Topology>'+'\n')
    fid.write('      <Geometry GeometryType="XYZ">'+'\n')
    fid.write('        <DataItem Dimensions="%s 3" NumberType="Float" Precision="4" Format="HDF">'%x.shape[0]+h5_file+':/Geometry/Points</DataItem>'+'\n')
    fid.write('      </Geometry>'+'\n')
    fid.write('      <Attribute Name="'+fieldName+'" Active="1" AttributeType="None" Center="Node">'+'\n')
    fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">'%x.shape[0]+h5_file+':/Node/field</DataItem>'+'\n')
    fid.write('      </Attribute>'+'\n')
    fid.write('    </Grid>'+'\n')
    fid.write('  </Domain>'+'\n')
    fid.write('</Xdmf>')
    fid.close()

    im('  -> Writing the data file (HDF5)',pName,verbose)
    fid = h5py.File(path+h5_file,'w')
    dset = fid.create_dataset('Topology', data = simplices)
    dset = fid.create_dataset('/Geometry/Points', data = points)
    dset = fid.create_dataset('/Node/field', data = v)
    fid.close()

    im('Process complete',pName,verbose)
    im('Files generated:',pName,verbose)
    im('  -> '+path+xdmf_file,pName,verbose)
    im('  -> '+path+h5_file,pName,verbose)





def vectorial2VTU(fname,x,y,z,v,Nz,path='./',ASCII=False,return_only=False,creat_pointID=False,verbose=True,\
                  vect_topo_name='Cartesian Velocity',vect_geo_name='Spherical Velocity'):
    """ This function creat '.vtu' file readable with Paraview to efficiently 
    visualize 3D data simply defined with grid matrix (x,y,z) and a field v 
    as well as a number of layer Nz.
    <i> : fname     = str, name of the exported file without any extention
          x         = list/numpy.ndarray, matrix of the cartesian x coordinates (flatten)
          y         = list/numpy.ndarray, matrix of the cartesian y coordinates (flatten)
          z         = list/numpy.ndarray, matrix of the cartesian z coordinates (flatten)
          v         = list of numpy.ndarray representing the different component of your
                      vectorial fields. The length of v can be either 3 or 6 
                      according if you just enter the topocentric field (vx,vy,vz) or if you
                      also entered the geodetic field (vr,vtheta,vphi).
                            v = (vx,vy,vz)
                        or  v = (vr,vtheta,vphi)
                        or  v = (vx,vy,vz,vr,vtheta,vphi)
                      numpy.ndarray, matrix of the field (flatten)
          Nz        = int, dimension of the grid in the z direction, aka number of z layers
          fieldName = str, name of the field for Paraview
          path      = str, path where you want to export your new .vtu file.
                      [Default: path='./']
          ASCII     = bool, if True, the .vtu file will be write in ASCII mode
                      if not, in binary mode. [Default, ASCII=False]
          verbose   = bool, if True activates the verbose output
    """
    pName = 'scalar2VTU'
    if verbose:
        print()
    im('Scalar field Visualization ToolKit',pName,verbose)
    if ASCII:
        im('Requested: stagData -> .vtu',pName,verbose)
    else:
        im('Requested: stagData -> .xdmf + .h5',pName,verbose)
    im('    - Grid preparation',pName,verbose)
    #
    if len(v) == 3:
        vx,vy,vz = v
        im('    - 3 components detected',pName,verbose)
    elif len(v) == 6:
        vx,vy,vz,vr,vt,vp = v
        im('    - 6 components detected',pName,verbose)
    else:
        print('The input list of velocity components have not the good shape. Have a look on the help of the function.')
        return 0
    # =======================================
    #Re-formating initial grids data
    X_yy  = np.array(x)
    Y_yy  = np.array(y)
    Z_yy  = np.array(z)
    #reformating
    Nz   = Nz                         #Number of depth layers
    NxNy = int(len(x)/Nz)   #Number of points for each layers
    x1    = X_yy.reshape(NxNy,Nz)
    X_yy  = x1.tolist()
    y1    = Y_yy.reshape(NxNy,Nz)
    Y_yy  = y1.tolist()
    z1    = Z_yy.reshape(NxNy,Nz)
    Z_yy  = z1.tolist()
    #Re-organisation of data to have X,Y and Z grid matrices organized by depths:
    X = []
    Y = []
    Z = []
    for j in range(Nz):
        x1t = [x1[i][j] for i in range(NxNy)]
        X.append(x1t)
        y1t = [y1[i][j] for i in range(NxNy)]
        Y.append(y1t)
        z1t = [z1[i][j] for i in range(NxNy)]
        Z.append(z1t)    
    # =========================================================================
    # 1) Take the surface of the 2 grids, patch together and triangulate
    # =========================================================================
    im('    - Triangulation on convex hull',pName,verbose)
    # NotaBene: _s for the surface layer
    X_s    = X[Nz-1]
    Y_s    = Y[Nz-1]
    Z_s    = Z[Nz-1]
    # =======================================
    # Triangulation of the surface using a convex hull algorithm
    points      = [[X_s[i],Y_s[i],Z_s[i]] for i in range(len(X_s))]
    triYingYang = ConvexHull(points).simplices # simple way to grid it
    # =========================================================================
    # 2) Create a 3D grid with tetrahedron elements
    # =========================================================================
    # Number all gridpoints we have
    NUM         = np.array(range(0,NxNy*Nz))
    NUMBER      = NUM.reshape((NxNy,Nz), order='F')
    # Make a loop over all levels
    ElementNumbers      = []
    for iz in range(Nz-1):
        num_upper      = NUMBER[:,iz+1]
        num_lower      = NUMBER[:,iz]
        num_tri = [[num_upper[t[0]], \
                    num_upper[t[1]], \
                    num_upper[t[2]], \
                    num_lower[t[0]], \
                    num_lower[t[1]], \
                    num_lower[t[2]]] for t in triYingYang]
        ElementNumbers.extend(num_tri)
    # =======================================
    # Convert data into correct vector format
    im('    - Convert data into correct vector format',pName,verbose)
    Points = [list(np.array(x1).reshape((NxNy*Nz), order='F')), \
              list(np.array(y1).reshape((NxNy*Nz), order='F')), \
              list(np.array(z1).reshape((NxNy*Nz), order='F'))]
    Points = np.array(Points).transpose()
    # ===================
    VX = np.array(vx).reshape(NxNy,Nz)
    VY = np.array(vy).reshape(NxNy,Nz)
    VZ = np.array(vz).reshape(NxNy,Nz)
    Vm = np.sqrt(VX**2+VY**2+VZ**2)    # magnitude will be stored in the pressure field
    if len(v) == 3:
        mysd = StagData(geometry='cart3D')
        mysd.fieldNature = 'Vectorial'
        mysd.fieldType   = 'bin'
        vstack = (list(VX.reshape((NxNy*Nz), order='F')),\
                list(VY.reshape((NxNy*Nz), order='F')),\
                list(VZ.reshape((NxNy*Nz), order='F')),\
                list(Vm.reshape((NxNy*Nz), order='F')))
    elif len(v) == 6:
        mysd = StagData(geometry='yy')
        mysd.fieldNature = 'Vectorial'
        mysd.fieldType   = 'bin'
        VR = np.array(vr).reshape(NxNy,Nz)
        VT = np.array(vt).reshape(NxNy,Nz)
        VP = np.array(vp).reshape(NxNy,Nz)
        vstack = (list(VX.reshape((NxNy*Nz), order='F')),\
                list(VY.reshape((NxNy*Nz), order='F')),\
                list(VZ.reshape((NxNy*Nz), order='F')),\
                list(VR.reshape((NxNy*Nz), order='F')),\
                list(VT.reshape((NxNy*Nz), order='F')),\
                list(VP.reshape((NxNy*Nz), order='F')),\
                list(Vm.reshape((NxNy*Nz), order='F')))
        
    # =========================================================================
    if creat_pointID:
        im('      - Creat pointID',pName,verbose)
        pointID = np.array(range(NxNy*Nz),dtype=np.int32).reshape(NxNy,Nz)        
        pointID = pointID.reshape((NxNy*Nz), order='F')
    else:
        pointID = None
    if not return_only:
        # Exportation under VTKUnstructuredGrid format
        if ASCII:
            im('    - Writing under .vtu format',pName,verbose)
        else:
            im('    - Writing under .xdmf + .h5 formats',pName,verbose)
        __writeVKTStag(fname,mysd,Points,ElementNumbers,vstack,ASCII=ASCII,path=path,pointID=pointID,\
                       vect_topo_name=vect_topo_name,vect_geo_name=vect_geo_name)
        im('Exportation done!',pName,verbose)
        if ASCII:
            im('File: '+fname+'.vtu',pName,verbose)
            im('Path: '+path,pName,verbose)
        else:
            im('Files: '+fname+'.xdmf + '+fname+'.h5',pName,verbose)
            im('Path : '+path,pName,verbose)
    else:
        # Return all grid/mesh elements
        return Points,ElementNumbers,vstack,pointID





def cart2VTU(fname,x,y,z,v,Nx,Ny,Nz,fieldName,path='./',ASCII=False,verbose=True):
    """ This function creat '.vtu' file readable with Paraview to efficiently 
    visualize 3D data simply defined with grid matrix (x,y,z) and a field v 
    as well as a the dimension in each direction Nx, Ny and Nz.
    <i> : fname     = str, name of the exported file without any extention
          x         = list/numpy.ndarray, matrix of the cartesian x coordinates (flatten)
          y         = list/numpy.ndarray, matrix of the cartesian y coordinates (flatten)
          z         = list/numpy.ndarray, matrix of the cartesian z coordinates (flatten)
          v         = list/numpy.ndarray, matrix of the field (flatten)
          Nx        = int, dimension of the grid in the x direction
          Ny        = int, dimension of the grid in the y direction
          Nz        = int, dimension of the grid in the z direction aka number of z layers
          fieldName = str, name of the field for Paraview, e.g. 'Temperature'
          path      = str, path where you want to export your new .vtu file.
                      [Default: path='./']
          ASCII     = bool, if True, the .vtu file will be write in ASCII mode
                      if not, in binary mode. [Default, ASCII=True]
          verbose   = bool, if True activates the verbose output
    """
    pName = 'car2VTU'
    if verbose:
        print()
    im('PyStag Visualization ToolKit',pName,verbose)
    im('Requested: flatten 3D matrix -> .vtu',pName,verbose)
    im('    - Grid preparation',pName,verbose)
    # =======================================
    #Re-formating initial grids data
    NxNy = Nx*Ny
    # =========================================================================
    # 1) Take the surface of the 2 grids, patch together and triangulate
    # =========================================================================
    im('    - Planar triangulation',pName,verbose)
    # =======================================
    #Computation of the triangulation of just a level of depth
    triPlanar_simplices = triangulationPlanar(Nx,Ny,ordering='yx')
    # =========================================================================
    # 2) Create a 3D grid with tetrahedron elements
    # =========================================================================
    # Number all gridpoints we have
    NUM         = np.array(range(0,NxNy*Nz))
    NUMBER      = NUM.reshape((NxNy,Nz), order='F')
    # Make a loop over all levels
    ElementNumbers      = []
    for iz in range(Nz-1):
        num_upper      = NUMBER[:,iz+1]
        num_lower      = NUMBER[:,iz]
        num_tri = [[num_upper[t[0]], \
                    num_upper[t[1]], \
                    num_upper[t[2]], \
                    num_lower[t[0]], \
                    num_lower[t[1]], \
                    num_lower[t[2]]] for t in triPlanar_simplices]
        ElementNumbers.extend(num_tri)
    # =======================================
    # Convert data into correct vector format
    im('    - Convert data into correct vector format',pName,verbose)
    Points = [list(np.array(x).reshape((NxNy*Nz), order='F')), \
              list(np.array(y).reshape((NxNy*Nz), order='F')), \
              list(np.array(z).reshape((NxNy*Nz), order='F'))]
    Points = np.array(Points).transpose()
    # ===================
    V_yy  = np.array(v).reshape(NxNy,Nz, order='F')
    vstack = list(V_yy.reshape((NxNy*Nz), order='F'))
    im('    - Writing under .vtu format',pName,verbose)
    __WriteVTU(fname,Points,ElementNumbers,vstack,fieldName,ASCII=ASCII,path=path)
    im('Exportation done!',pName,verbose)
    im('File: '+fname+'.vtu',pName,verbose)





def stag2VTU(fname,stagData,path='./',ASCII=False,verbose=True,return_only=False,creat_pointID=False,\
             vect_topo_name='Velocity Cartesian',vect_geo_name='Velocity Spherical'):
    """ -- Geometry Adaptative Toolkit transforming stagData into VTU --
    This function creats readable file for Paraview for an efficient 
    3D visualization of data contain in an input stagData instance.
    This function works directly on a stagData input object and adapts the
    constuction of the triangulation according to this geometry. Furthermore
    stag2VTU is able to deal with vectorial and scalar fields. Outputs can be
    written under two format: 1. the explicit ascii .vtu format or 2. the
    coupled .h5 and .xdmf format (more efficient)
    Concerning geometies:
      - 'yy' deal with non overlapping stagData object so with stagData.x1,
        stagData.x2, stagData.y1 ... fields
      - 'cart3D', read stagData.x, .y, .z and .v fields for a scalar field as
        example
      - 'spherical' deal with the spherical grid so stagData.x, .y, .z fields
        and ignore .xc, .yc and .zc fields.
    Note that the internal field stagData.slayers of the stagData object
    must be filled!
    --> stag2vtu is main function of the pypStag Visualization ToolKit

    <i> : fname = str, name of the exported file without any extention
          stagData = stagData object, stagData object that will be transform
                     into meshed file
          path = str, path where you want to export your new meshed file.
                 [Default: path='./']
          ASCII = bool, if True, the export paraview file will be write
                  in ASCII .vtu format (take lots of memory space),
                  if not ASCII, the export will be partitioned in
                  a .xdmf and .h5 file (very efficient in writing,
                  reading and save lots of memory).
                  [Default, ASCII=False]
          verbose = bool, if True, then generate a verbose output
                    [Default, verbose=True]
          return_only = bool, if True, stag2VTU() will just manage the
                        3D tetrahedralization of the grid, and returns:
                        (Points,ElementNumbers,vstack) corresponding to 
                        points matrix, meshing doc and corresponding field.
                        Do not call __writeVKTStag() if True
                        If False, generate meshed file for paraview, call
                        __writeVTKStag()
                        [Default, return_only=False]
          creat_pointID = bool, if True then creat the list of points ID sorted as
                          it is in the field stagData.x (Yin and Yang together).
                          This field will then transfert to the writing function and
                          the .h5/.xdmf will have an extra field corresponding to 
                          these points ID (e.g. very usefull if post processing with TTK)
                          WARNING: This option is only available if ASCII = False
                                   (ie .h5/.xdmf output)
    """
    pName = 'stag2VTU'
    im('pypStag Visualization ToolKit',pName,verbose)
    if ASCII:
        im('Requested: stagData -> .vtu',pName,verbose)
    else:
        im('Requested: stagData -> .xdmf + .h5',pName,verbose)

    #--------------
    if path[-1] != '/':
        path += '/'
    #--------------
    if stagData.slayers.shape[0] > 1:
        im('Volumetric data detected (multiple depths): -> tetrahedralization',pName,verbose)
        # --------------------
        # Here below the routine to transform a multi-layers (multi-depths)
        # stagData object into a VTU file readable with the sorftware Paraview.
        #   -> N.B. We really make a distinction between creat a multi-layers
        #           and a single (depth) layer file. Indeed, a multi-layers obj 
        #           contains *volume* data that need to be tetrahedralized
        #           whereas a single-layer obj containing 'only' surface data
        #           that thus need 'only' to be triangulized.
        # -------------------- 
        if stagData.geometry == 'cart3D' or stagData.geometry == 'spherical':
            """
            Adaptations for the exportation of the complete
            stagData object with a 3D cartesian or spherical
            geometries.
            """
            im('    - Grid preparation',pName,verbose)
            # =======================================
            #Re-formating initial grids data
            Nx = len(stagData.x_coords)
            Ny = len(stagData.y_coords)
            Nz = len(stagData.slayers)
            NxNy = Nx*Ny
            x = np.array(stagData.x).reshape(NxNy*Nz, order='F')
            y = np.array(stagData.y).reshape(NxNy*Nz, order='F')
            z = np.array(stagData.z).reshape(NxNy*Nz, order='F')
            v = np.array(stagData.v).reshape(NxNy*Nz, order='F')
            # =========================================================================
            # 1) Take the surface of the 2 grids, patch together and triangulate
            # =========================================================================
            im('    - Planar triangulation',pName,verbose)
            # =======================================
            #Computation of the triangulation of just a level of depth
            triPlanar_simplices = triangulationPlanar(Nx,Ny,ordering='yx')
            nod = len(triPlanar_simplices)
            # =========================================================================
            # 2) Create a 3D grid with tetrahedron elements
            # =========================================================================
            # Number all gridpoints we have
            NUM         = np.array(range(0,NxNy*Nz))
            NUMBER      = NUM.reshape((NxNy,Nz), order='F')
            # Make a loop over all levels
            ElementNumbers = np.zeros(((Nz-1)*nod,6),dtype=np.int32)
            num_upper      = np.zeros(NxNy)
            num_lower      = np.zeros(NxNy)
            num_tri = np.zeros((nod,6))
            for iz in range(Nz-1):
                num_upper      = NUMBER[:,iz+1]
                num_lower      = NUMBER[:,iz]
                num_tri[:,0] = num_upper[triPlanar_simplices[:,0]]
                num_tri[:,1] = num_upper[triPlanar_simplices[:,1]]
                num_tri[:,2] = num_upper[triPlanar_simplices[:,2]]
                num_tri[:,3] = num_lower[triPlanar_simplices[:,0]]
                num_tri[:,4] = num_lower[triPlanar_simplices[:,1]]
                num_tri[:,5] = num_lower[triPlanar_simplices[:,2]]
                ElementNumbers[nod*iz:nod*(iz+1),:] = num_tri
            # =======================================
            # Convert data into correct vector format
            im('    - Convert data into correct vector format',pName,verbose)
            im('      - Grid',pName,verbose)
            Points = np.zeros((NxNy*Nz,3))
            Points[:,0] = np.array(x).reshape((NxNy*Nz), order='F')
            Points[:,1] = np.array(y).reshape((NxNy*Nz), order='F')
            Points[:,2] = np.array(z).reshape((NxNy*Nz), order='F')
            # ===================
            im('      - Field',pName,verbose)
            if stagData.fieldNature == 'Scalar' or stagData.fieldNature == '':
                V_yy  = np.array(v).reshape(NxNy,Nz, order='F')
                vstack = V_yy.reshape((NxNy*Nz), order='F')
            # ===================
            if stagData.fieldNature == 'Vectorial':
                if stagData.geometry == 'cart3D':
                    # ------ Vx ------
                    V_vx  = np.array(stagData.vx).reshape(NxNy,Nz, order='F')
                    vstackx = V_vx.reshape((NxNy*Nz), order='F')
                    # ------ Vy ------
                    V_vy  = np.array(stagData.vy).reshape(NxNy,Nz, order='F')
                    vstacky = V_vy.reshape((NxNy*Nz), order='F')
                    # ------ Vz ------
                    V_vz  = np.array(stagData.vz).reshape(NxNy,Nz, order='F')
                    vstackz = V_vz.reshape((NxNy*Nz), order='F')
                    # ------ P ------
                    V_vp  = np.array(stagData.P).reshape(NxNy,Nz, order='F')
                    vstackp = V_vp.reshape((NxNy*Nz), order='F')
                    # ------ stack ------
                    vstack = (vstackx,vstacky,vstackz,vstackp)
                elif stagData.geometry == 'spherical':
                    # ------ Vx ------
                    V_vx  = np.array(stagData.vx).reshape(NxNy,Nz, order='F')
                    vstackx = V_vx.reshape((NxNy*Nz), order='F')
                    # ------ Vy ------
                    V_vy  = np.array(stagData.vy).reshape(NxNy,Nz, order='F')
                    vstacky = V_vy.reshape((NxNy*Nz), order='F')
                    # ------ Vz ------
                    V_vz  = np.array(stagData.vz).reshape(NxNy,Nz, order='F')
                    vstackz = V_vz.reshape((NxNy*Nz), order='F')
                    # ------ Vr ------
                    V_vr  = np.array(stagData.vr).reshape(NxNy,Nz, order='F')
                    vstackr = V_vr.reshape((NxNy*Nz), order='F')
                    # ------ Vtheta ------
                    V_theta  = np.array(stagData.vtheta).reshape(NxNy,Nz, order='F')
                    vstacktheta = V_theta.reshape((NxNy*Nz), order='F')
                    # ------ Vphi ------
                    V_phi  = np.array(stagData.vphi).reshape(NxNy,Nz, order='F')
                    vstackphi = V_phi.reshape((NxNy*Nz), order='F')
                    # ------ P ------
                    V_vp  = np.array(stagData.P).reshape(NxNy,Nz, order='F')
                    vstackp = V_vp.reshape((NxNy*Nz), order='F')
                    # ------ stack ------
                    vstack = (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)

        elif stagData.geometry == 'yy':
            """
            Adaptations for the exportation of the complete
            stagData object with a Yin-Yang geoemtry.
            """
            im('    - Grid preparation',pName,verbose)
            # =======================================
            #Re-formating initial grids data
            Nz   = len(stagData.slayers)      #Number of depth layers
            NxNy = int(len(stagData.x1)/Nz)   #Number of points for each layers
            x1     = stagData.x1.reshape(NxNy,Nz)
            x2     = stagData.x2.reshape(NxNy,Nz)
            y1     = stagData.y1.reshape(NxNy,Nz)
            y2     = stagData.y2.reshape(NxNy,Nz)
            z1     = stagData.z1.reshape(NxNy,Nz)
            z2     = stagData.z2.reshape(NxNy,Nz)
            #Re-organisation of data to have X,Y and Z grid matrices organized by depths:
            X = np.zeros((Nz,2*NxNy))
            Y = np.zeros((Nz,2*NxNy))
            Z = np.zeros((Nz,2*NxNy))
            X[:,0:NxNy]      = x1.T
            X[:,NxNy:2*NxNy] = x2.T
            Y[:,0:NxNy]      = y1.T
            Y[:,NxNy:2*NxNy] = y2.T
            Z[:,0:NxNy]      = z1.T
            Z[:,NxNy:2*NxNy] = z2.T
            # =========================================================================
            # 1) Take the surface of the 2 grids, patch together and triangulate
            # =========================================================================
            im('    - Triangulation on convex hull',pName,verbose)
            # NotaBene: _s for the surface layer
            X_s    = X[Nz-1]
            Y_s    = Y[Nz-1]
            Z_s    = Z[Nz-1]
            # =======================================
            # Triangulation of the surface using a convex hull algorithm
            points = np.array([X_s,Y_s,Z_s]).T
            triYingYang = ConvexHull(points).simplices # simple way to grid it
            nod = triYingYang.shape[0]
            # =========================================================================
            # 2) Create a 3D grid with tetrahedron elements
            # =========================================================================
            # Number all gridpoints we have
            NUM_1       = np.array(range(0,NxNy*Nz))
            NUMBER_1    = NUM_1.reshape((NxNy,Nz), order='F')
            NUMBER_2    = NUMBER_1 + NxNy*Nz
            # -- Make a loop over all levels
            # init all arrays
            ElementNumbers = np.zeros(((Nz-1)*nod,6),dtype=np.int32)
            num_upper      = np.zeros(NxNy*2)
            num_lower      = np.zeros(NxNy*2)
            num_tri = np.zeros((nod,6))
            for iz in range(Nz-1):
                num_upper[0:NxNy]      = NUMBER_1[:,iz+1]#np.array(list(num_upper1) + list(num_upper2))
                num_upper[NxNy:NxNy*2] = NUMBER_2[:,iz+1]
                num_lower[0:NxNy]      = NUMBER_1[:,iz]#np.array(list(num_upper1) + list(num_upper2))
                num_lower[NxNy:NxNy*2] = NUMBER_2[:,iz]
                num_tri[:,0] = num_upper[triYingYang[:,0]]
                num_tri[:,1] = num_upper[triYingYang[:,1]]
                num_tri[:,2] = num_upper[triYingYang[:,2]]
                num_tri[:,3] = num_lower[triYingYang[:,0]]
                num_tri[:,4] = num_lower[triYingYang[:,1]]
                num_tri[:,5] = num_lower[triYingYang[:,2]]
                ElementNumbers[nod*iz:nod*(iz+1),:] = num_tri 

            # =======================================
            # Convert data into correct vector format
            im('    - Convert data into correct vector format:',pName,verbose)
            im('      - Grid',pName,verbose)
            Points = np.zeros((2*NxNy*Nz,3))
            Points[0:NxNy*Nz,0]         = np.array(x1).reshape((NxNy*Nz), order='F')
            Points[NxNy*Nz:2*NxNy*Nz,0] = np.array(x2).reshape((NxNy*Nz), order='F')
            Points[0:NxNy*Nz,1]         = np.array(y1).reshape((NxNy*Nz), order='F')
            Points[NxNy*Nz:2*NxNy*Nz,1] = np.array(y2).reshape((NxNy*Nz), order='F')
            Points[0:NxNy*Nz,2]         = np.array(z1).reshape((NxNy*Nz), order='F')
            Points[NxNy*Nz:2*NxNy*Nz,2] = np.array(z2).reshape((NxNy*Nz), order='F')
            # ===================
            im('      - Field',pName,verbose)
            if stagData.fieldNature == 'Scalar' or stagData.fieldNature == '':
                V_yin  = np.array(stagData.v1).reshape(NxNy,Nz)
                V_yang = np.array(stagData.v2).reshape(NxNy,Nz)
                vstack = np.zeros(2*NxNy*Nz)
                vstack[0:NxNy*Nz]         = V_yin.reshape((NxNy*Nz), order='F')
                vstack[NxNy*Nz:2*NxNy*Nz] = V_yang.reshape((NxNy*Nz),order='F')
            # ===================
            if stagData.fieldNature == 'Vectorial':
                # ------ Vx ------
                V_yinx  = np.array(stagData.vx1).reshape(NxNy,Nz)
                V_yangx = np.array(stagData.vx2).reshape(NxNy,Nz)
                vstackx = np.zeros(2*NxNy*Nz)
                vstackx[0:NxNy*Nz]         = V_yinx.reshape((NxNy*Nz), order='F')
                vstackx[NxNy*Nz:2*NxNy*Nz] = V_yangx.reshape((NxNy*Nz),order='F')
                # ------ Vy ------
                V_yiny  = np.array(stagData.vy1).reshape(NxNy,Nz)
                V_yangy = np.array(stagData.vy2).reshape(NxNy,Nz)
                vstacky = np.zeros(2*NxNy*Nz)
                vstacky[0:NxNy*Nz]         = V_yiny.reshape((NxNy*Nz), order='F')
                vstacky[NxNy*Nz:2*NxNy*Nz] = V_yangy.reshape((NxNy*Nz),order='F')
                # ------ Vz ------
                V_yinz  = np.array(stagData.vz1).reshape(NxNy,Nz)
                V_yangz = np.array(stagData.vz2).reshape(NxNy,Nz)
                vstackz = np.zeros(2*NxNy*Nz)
                vstackz[0:NxNy*Nz]         = V_yinz.reshape((NxNy*Nz), order='F')
                vstackz[NxNy*Nz:2*NxNy*Nz] = V_yangz.reshape((NxNy*Nz),order='F')
                # ------ Vr ------
                V_yinr  = np.array(stagData.vr1).reshape(NxNy,Nz)
                V_yangr = np.array(stagData.vr2).reshape(NxNy,Nz)
                vstackr = np.zeros(2*NxNy*Nz)
                vstackr[0:NxNy*Nz]         = V_yinr.reshape((NxNy*Nz), order='F')
                vstackr[NxNy*Nz:2*NxNy*Nz] = V_yangr.reshape((NxNy*Nz),order='F')
                # ------ Vtheta ------
                V_yintheta  = np.array(stagData.vtheta1).reshape(NxNy,Nz)
                V_yangtheta = np.array(stagData.vtheta2).reshape(NxNy,Nz)
                vstacktheta = np.zeros(2*NxNy*Nz)
                vstacktheta[0:NxNy*Nz]         = V_yintheta.reshape((NxNy*Nz), order='F')
                vstacktheta[NxNy*Nz:2*NxNy*Nz] = V_yangtheta.reshape((NxNy*Nz),order='F')
                # ------ Vphi ------
                V_yinphi  = np.array(stagData.vphi1).reshape(NxNy,Nz)
                V_yangphi = np.array(stagData.vphi2).reshape(NxNy,Nz)
                vstackphi = np.zeros(2*NxNy*Nz)
                vstackphi[0:NxNy*Nz]         = V_yinphi.reshape((NxNy*Nz), order='F')
                vstackphi[NxNy*Nz:2*NxNy*Nz] = V_yangphi.reshape((NxNy*Nz),order='F')
                # ------ P ------
                V_yinp  = np.array(stagData.P1).reshape(NxNy,Nz)
                V_yangp = np.array(stagData.P2).reshape(NxNy,Nz)
                vstackp = np.zeros(2*NxNy*Nz)
                vstackp[0:NxNy*Nz]         = V_yinp.reshape((NxNy*Nz), order='F')
                vstackp[NxNy*Nz:2*NxNy*Nz] = V_yangp.reshape((NxNy*Nz),order='F')
                vstack = (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)
        # ===================
        if creat_pointID:
            im('      - Creat pointID',pName,verbose)
            if stagData.geometry == 'yy':
                IDyin   = np.array(range(NxNy*Nz),dtype=np.int32).reshape(NxNy,Nz)
                IDyang  = np.array(range(NxNy*Nz,2*NxNy*Nz),dtype=np.int32).reshape(NxNy,Nz)
                pointID = np.zeros(2*NxNy*Nz)
                pointID[0:NxNy*Nz]         = IDyin.reshape((NxNy*Nz), order='F')
                pointID[NxNy*Nz:2*NxNy*Nz] = IDyang.reshape((NxNy*Nz),order='F')
            else:
                pointID = np.array(range(NxNy*Nz),dtype=np.int32).reshape(NxNy*Nz, order='F')        
                pointID = pointID.reshape((NxNy*Nz), order='F')
        else:
            pointID = None
        # =========================================================================
        if not return_only:
            # Exportation under VTKUnstructuredGrid format
            if ASCII:
                im('    - Writing under .vtu format',pName,verbose)
            else:
                im('    - Writing under .xdmf + .h5 formats',pName,verbose)
            __writeVKTStag(fname,stagData,Points,ElementNumbers,vstack,ASCII=ASCII,path=path,pointID=pointID)
            im('Exportation done!',pName,verbose)
            if ASCII:
                im('File: '+fname+'.vtu',pName,verbose)
                im('Path: '+path,pName,verbose)
            else:
                im('Files: '+fname+'.xdmf + '+fname+'.h5',pName,verbose)
                im('Path : '+path,pName,verbose)
        else:
            # Return all grid/mesh elements
            return Points,ElementNumbers,vstack,pointID
    else:
        im('Areal data detected (single depth): -> triangulation',pName,verbose)
        # --- File names
        xdmf_file = fname + '.xdmf'
        h5_file   = fname + '.h5'
        write_slice = False
        # =======================================
        if stagData.geometry == 'cart3D' or stagData.geometry == 'spherical':
            write_slice = True
            X = stagData.x.flatten()
            Y = stagData.y.flatten()
            Z = stagData.z.flatten()
            V = stagData.v.flatten()
            VX = stagData.vx.flatten()
            VY = stagData.vy.flatten()
            VZ = stagData.vz.flatten()
            PRESSURE = stagData.P.flatten()
        # =======================================
        elif stagData.geometry == 'yy':
            write_slice = True
            X = stagData.x
            Y = stagData.y
            Z = stagData.z
            V = stagData.v
            VX = stagData.vx
            VY = stagData.vy
            VZ = stagData.vz
            VR = stagData.vr
            VTHETA = stagData.vtheta
            VPHI = stagData.vphi
            PRESSURE = stagData.P
        # =======================================
        if write_slice:
            nod = len(X)
            # ---
            if creat_pointID:
                im('  -> Creat pointID',pName,verbose)
                pointID = np.arange(nod)
            # ---
            points = np.zeros((nod,3))
            points[:,0] = X
            points[:,1] = Y
            points[:,2] = Z
            # ----
            if stagData.geometry == 'cart3D' or stagData.geometry == 'spherical':
                im('  -> Delaunay Triangulation',pName,verbose)
                tri = Delaunay(points[:,0:2])
            elif stagData.geometry == 'yy':
                im('  -> Convex Hull Triangulation',pName,verbose)    
                tri = ConvexHull(points)
            simplices = tri.simplices
            # --- Write the XDMF file
            im('  -> Writing the instruction file (XDMF)',pName,verbose)
            fid = open(path+xdmf_file,'w')
            fid.write('<?xml version="1.0" ?>'+'\n')
            fid.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'+'\n')
            fid.write('<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">'+'\n')
            fid.write('  <Domain>'+'\n')
            fid.write('    <Grid GridType="Uniform">'+'\n')
            fid.write('      <Topology TopologyType="Triangle" Dimensions="%s">'%int(simplices.shape[0])+'\n')
            fid.write('        <DataItem Dimensions="%s'%int(simplices.shape[0])+' 3" NumberType="Int" Precision="8" Format="HDF">'+h5_file+':/Topology</DataItem>'+'\n')
            fid.write('      </Topology>'+'\n')
            fid.write('      <Geometry GeometryType="XYZ">'+'\n')
            fid.write('        <DataItem Dimensions="%s 3" NumberType="Float" Precision="4" Format="HDF">'%nod+h5_file+':/Geometry/Points</DataItem>'+'\n')
            fid.write('      </Geometry>'+'\n')
            # --- Field(s)
            if stagData.fieldNature == 'Scalar':
                fid.write('      <Attribute Name="'+stagData.fieldType+'" Active="1" AttributeType="Scalar" Center="Node">'+'\n')
                fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">'%nod+h5_file+':/Node/field_scalar</DataItem>'+'\n')
                fid.write('      </Attribute>'+'\n')
            elif stagData.fieldNature == 'Vectorial':
                # ---- Cartesian Velocities ----
                fid.write('      <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % vect_topo_name)
                fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%(nod, 3))
                fid.write('            '+h5_file+':/Node/field_cart\n')
                fid.write('        </DataItem>\n')
                fid.write('      </Attribute>\n\n')
                if stagData.geometry == 'yy':
                    # ---- Spherical Velocities ----
                    fid.write('      <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % vect_geo_name)
                    fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%(nod, 3))
                    fid.write('            '+h5_file+':/Node/field_sphe\n')
                    fid.write('        </DataItem>\n')
                    fid.write('      </Attribute>\n\n')
                # ---- Pressure ----
                fid.write('      <Attribute Name="'+'Pressure'+'" Active="1" AttributeType="Scalar" Center="Node">'+'\n')
                fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">'%nod+h5_file+':/Node/field_pressure</DataItem>'+'\n')
                fid.write('      </Attribute>'+'\n')
            # --- pointID
            if creat_pointID:
                fid.write('      <Attribute AttributeType="Scalar" Center="Node" Name="PointID">\n')
                fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="8">\n'%pointID.shape[0])
                fid.write('            '+h5_file+':/Node/pointID\n')
                fid.write('        </DataItem>\n')
                fid.write('      </Attribute>\n\n')
            # --- close
            fid.write('    </Grid>'+'\n')
            fid.write('  </Domain>'+'\n')
            fid.write('</Xdmf>')
            fid.close()
            # --- write the H5 file
            im('  -> Writing the data file (HDF5)',pName,verbose)
            fid = h5py.File(path+h5_file,'w')
            dset = fid.create_dataset('Topology', data = simplices)
            dset = fid.create_dataset('/Geometry/Points', data = points)
            if stagData.fieldNature == 'Scalar':
                dset = fid.create_dataset('/Node/field_scalar', data = V)
            elif stagData.fieldNature == 'Vectorial':
                dset = fid.create_dataset('/Node/field_cart', data = np.array([VX,VY,VZ]).T)
                if stagData.geometry == 'yy':
                    dset = fid.create_dataset('/Node/field_sphe', data = np.array([VR,VTHETA,VPHI]).T)
                dset = fid.create_dataset('/Node/field_pressure', data = PRESSURE)
            if creat_pointID:
                dset = fid.create_dataset('/Node/pointID', data = pointID)
            fid.close()
            # --- Finish!
            im('Process complete',pName,verbose)
            im('Files generated:',pName,verbose)
            im('  -> '+path+xdmf_file,pName,verbose)
            im('  -> '+path+h5_file,pName,verbose)





def __writeVKTStag(fname,stagData,Points,ElementNumbers,vstack,ASCII=False,path='./',pointID=None,\
                   vect_topo_name='Velocity Cartesian',vect_geo_name='Velocity Spherical'):
    """ This function creats and exports trianguled geometry and field(s) into
    a .vtu/.h5 file under the name fname. This function was built to be used by
    stag2VTU() and stag2VTU_For_overlapping() functions.
    This function export the geometry into a VTKUnstructuredGrid format.
    <i> : fname = str, name of the exported file without any extention
          stagData = stagData object, stagData object that will be transform
                     into meshed file
          Points = np.ndarray, matrices of points as defined in stag2VTU()
          ElementNumbers = list, list of how points have to be organized to
                           form 3D-'roofs' in space
          vstack = list OR tuple of lists, if stagData.fieldNature == Scalar,
                   vstack must be a list and if  == Vectorial, must be a 
                   tuple as vstack=(vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)
          ASCII = bool, if True, the export paraview file will be write
                  in ASCII .vtu format (take lots of memory space),
                  if not ASCII, the export will be partitioned in
                  a .xdmf and .h5 file (very efficient in writing,
                  reading and save lots of memory).
                  [Default, ASCII=False]
          path = str, path where you want to export your new .vtu file.
                 [Default: path='./']
          pointID = np.ndarray (or None), contains the list of points ID sorted as
                    it is in the field stagData.x (Yin and Yang together). If pointID != None,
                    then, the .h5/.xdmf will have an extra field corresponding to 
                    these points ID (e.g. very usefull if post processing with TTK)
                    WARNING: This option is only available if ASCII = False
                             (ie .h5/.xdmf output)
    """
    # If ASCII then, write an unique file for paraview: ASCII .vtu file
    if ASCII:
        # =========================================================================
        # Write VTK file (unstructured mesh)
        # =========================================================================
        # Definitions and initialization
        sizeof_Float32  =   4
        sizeof_Float64  =   4
        sizeof_UInt32   =   4
        sizeof_UInt8    =   1
        Offset          =   0      # Initial offset
        # =======================================
        # Write the header for a structured grid:
        fname_vtk = fname+'.vtu'
        fid       = open(path+fname_vtk,'w')
        fid.write('<?xml version="1.0"?> \n')
        fid.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian" >\n')
        fid.write('  <UnstructuredGrid>\n')
        fid.write('    <Piece NumberOfPoints="%s"  NumberOfCells="%s">\n' % \
                (str(np.int32(Points.shape[0])), str(len(np.int32(ElementNumbers)))))
        # =======================================
        # Add point-wise data
        #Adapt the exporting file according to the nature of the field (Scalar/Vectorial)
        fid.write('    <PointData Scalars="T" Vectors="Velocity"  >\n')
        # ===================
        if stagData.fieldNature == 'Scalar' or stagData.fieldNature == '':
            fid.write('      <DataArray type="Float32" Name="%s" format="ascii">\n' % stagData.fieldType)
            for i in range(len(vstack)):
                fid.write('        %s \n' % vstack[i])
            fid.write('      </DataArray>\n')
        # ===================
        elif stagData.fieldNature == 'Vectorial':
            # ===================
            #condition on stagData.geometry: the vstack input is different
            if stagData.geometry == 'cart3D':
                (vstackx,vstacky,vstackz,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3])
                # ------ Vx, Vy, Vz ------
                fid.write('      <DataArray type="Float32" Name="'+vect_topo_name+'" NumberOfComponents="3" format="ascii">\n')
                for i in range(len(vstackx)):
                    fid.write('   %g %g %g \n' % (vstackx[i],vstacky[i],vstackz[i]))
                fid.write('      </DataArray>\n')
                # ------ P ------
                fid.write('      <DataArray type="Float32" Name="%s" format="ascii">\n' % 'Pressure')
                for i in range(len(vstackp)):
                    fid.write('        %s \n' % vstackp[i])
                fid.write('      </DataArray>\n')
            # ===================
            if stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3],vstack[4],vstack[5],vstack[6])
                # ------ Vx, Vy, Vz ------
                fid.write('      <DataArray type="Float32" Name="'+vect_topo_name+'" NumberOfComponents="3" format="ascii">\n')
                for i in range(len(vstackx)):
                    fid.write('   %g %g %g \n' % (vstackx[i],vstacky[i],vstackz[i]))
                fid.write('      </DataArray>\n')
                # ------ Vr, Vtheta, Vphi ------
                fid.write('      <DataArray type="Float32" Name="'+vect_geo_name+'" NumberOfComponents="3" format="ascii">\n')
                for i in range(len(vstackx)):
                    fid.write('   %g %g %g \n' % (vstackr[i],vstacktheta[i],vstackphi[i]))
                fid.write('      </DataArray>\n')
                # ------ P ------
                fid.write('      <DataArray type="Float32" Name="%s" format="ascii">\n' % 'Pressure')
                for i in range(len(vstackp)):
                    fid.write('        %s \n' % vstackp[i])
        # =======================================
        # Add coordinates of structured grid
        fid.write('    </PointData>\n')
        fid.write('    <Points>\n')
        fid.write('      <DataArray type="Float32" Name="Array" NumberOfComponents="3" format="ascii">\n')
        for i in range(len(Points)):
            fid.write('         %s %s %s \n' % (Points[i][0],Points[i][1],Points[i][2]))
        fid.write('        </DataArray>\n')
        fid.write('    </Points>\n')
        # =======================================
        # Add CELLS data
        fid.write('    <Cells>\n')    
        # - Connectivity -----------
        fid.write('      <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i in range(len(ElementNumbers)):
            fid.write('        %s %s %s %s %s %s \n' % (np.int32(ElementNumbers[i][0]), \
                                                        np.int32(ElementNumbers[i][1]), \
                                                        np.int32(ElementNumbers[i][2]), \
                                                        np.int32(ElementNumbers[i][3]), \
                                                        np.int32(ElementNumbers[i][4]), \
                                                        np.int32(ElementNumbers[i][5])))
        fid.write('      </DataArray>\n')
        # - Offsets -----------
        offsets = np.cumsum(np.ones(len(ElementNumbers))*6)
        fid.write('  <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(len(ElementNumbers)):
            fid.write('        %s\n' % np.int32(offsets[i]))
        fid.write('      </DataArray>\n')
        # - Types -----------
        types = np.ones(len(ElementNumbers))*13
        fid.write('      <DataArray type="UInt8" Name="types" format="ascii">\n')
        for i in range(len(ElementNumbers)):
            fid.write('        %s\n' % np.uint8(13))
        fid.write('      </DataArray>\n')
        # =======================================
        # Close .vtu
        fid.write('    </Cells>\n')
        fid.write('    </Piece>\n')
        fid.write('  </UnstructuredGrid>\n')
        fid.write('</VTKFile>\n')
        fid.close()
    
    # If not ASCII then creat 2 files: a .h5 file for grid elements and a .xdmf for all
    # the grid format instructions
    else:
        # =========================================================================
        # Write XDMF file
        # =========================================================================
        Points         = np.asarray(Points)
        ElementNumbers = np.asarray(ElementNumbers)
        if stagData.fieldNature == 'Scalar':
            Data       = np.asarray(vstack)
        else:
            if stagData.geometry == 'cart3D':
                (vstackx,vstacky,vstackz,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3])
                Datax  = np.asarray(vstackx)
                Datay  = np.asarray(vstacky)
                Dataz  = np.asarray(vstackz)
                Datap  = np.asarray(vstackp)
            elif stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3],vstack[4],vstack[5],vstack[6])
                Datax  = np.asarray(vstackx)
                Datay  = np.asarray(vstacky)
                Dataz  = np.asarray(vstackz)
                Datar  = np.asarray(vstackr)
                Datath = np.asarray(vstacktheta)
                Dataph = np.asarray(vstackphi)
                Datap  = np.asarray(vstackp)

        # =======================================
        # Write the header for a structured grid:
        fname_vtk = fname+'.xdmf'
        fname_h5  = fname+'.h5'
        fid       = open(path+fname_vtk,'w')
        fid.write('<Xdmf Version="3.0">\n')
        fid.write('<Domain>\n')
        fid.write('<Grid Name="Grid">\n\n')
        fid.write('    <Geometry GeometryType="XYZ">\n')
        # =======================================
        # Write Points
        fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n' %\
                   (Points.shape[0],Points.shape[1]))
        fid.write('            '+fname_h5+':/Points\n')
        fid.write('        </DataItem>\n')
        fid.write('    </Geometry>\n\n')
        # =======================================
        # Write NumberOfElements
        fid.write('    <Topology NodesPerElement="%s" NumberOfElements="%s" TopologyType="Wedge">\n' %\
                  (ElementNumbers.shape[1], ElementNumbers.shape[0]))
        fid.write('        <DataItem DataType="Int" Dimensions="%s %s" Format="HDF" Precision="8">\n' %\
                  ((ElementNumbers.shape[0], ElementNumbers.shape[1])))
        fid.write('            '+fname_h5+':/NumberOfElements\n')
        fid.write('        </DataItem>\n')
        fid.write('    </Topology>\n\n')
        # =======================================
        # Write field
        if stagData.fieldNature == 'Scalar':
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % stagData.fieldType)
            fid.write('        <DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    Data.shape[0])
            fid.write('            '+fname_h5+':/Data0\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
        else:
            # ---- Cartesian Velocities ----
            fid.write('    <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % vect_topo_name)
            fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%\
                    (Datax.shape[0], 3))
            fid.write('            '+fname_h5+':/Data1\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
            if stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                # ----  Shperical Velocities ----
                fid.write('    <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % vect_geo_name)
                fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%\
                        (Datar.shape[0], 3))
                fid.write('            '+fname_h5+':/Data2\n')
                fid.write('        </DataItem>\n')
                fid.write('    </Attribute>\n\n')
            # ---- Pressure ----
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % 'Pressure')
            fid.write('        <DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    Datap.shape[0])
            fid.write('            '+fname_h5+':/Data3\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
        # =======================================
        # PointID
        if pointID is not None:
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="PointID">\n')
            fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    pointID.shape[0])
            fid.write('            '+fname_h5+':/pointID\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')

        # =======================================
        # Ending
        fid.write('</Grid>\n')
        fid.write('</Domain>\n')
        fid.write('</Xdmf>')
        fid.close()
        # =========================================================================
        # Write HDF5 file
        # =========================================================================
        #Code:
        #  Data0 = scalar field
        #  Data1 = topocentric (cartesian) velocities
        #  Data2 = geodetic (spherical) velocities
        #  Data3 = Pressure
        import h5py
        fid = h5py.File(path+fname_h5, 'w')
        dset = fid.create_dataset("Points", data=Points, dtype=np.float32)
        dset = fid.create_dataset("NumberOfElements", data=ElementNumbers, dtype=np.int32)
        if stagData.fieldNature == 'Scalar':
            dset = fid.create_dataset("Data0", data=Data, dtype=np.float32)
        else:
            if stagData.geometry == 'cart3D':
                Data = np.array([Datax,Datay,Dataz]).T
                dset = fid.create_dataset("Data1", data=Data, dtype=np.float32)
            elif stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                Data = np.array([Datax,Datay,Dataz]).T
                dset = fid.create_dataset("Data1", data=Data, dtype=np.float32)
                Data = np.array([Datar,Datath,Dataph]).T
                dset = fid.create_dataset("Data2", data=Data, dtype=np.float32)
            Data = Datap
            dset = fid.create_dataset("Data3", data=Data, dtype=np.float32)
        if pointID is not None:
            dset = fid.create_dataset("pointID", data=pointID, dtype=np.int32)
        fid.close()
        




def stagCloud2timeVTU(fname,stagCloudData,multifile=True,timeAsF=False,timepar=1,path='./',\
                      creat_pointID=False,verbose=True,extended_verbose=False):
    """ -- Geometry Adaptative Toolkit transforming stagCloudData into VTU --
    This function creats readable file for Paraview for an efficient 
    3D visualization of data contain in an input stagCloudData instance.
    This function works directly on a stagCloudData input object and adapts the
    constuction of the triangulation according to this geometry and iterate
    automatically in the cloud. Due to large quantities of data created, outputs
    are necessarily written under the coupled .h5 and .xdmf format (more efficient)
    The 'multifile' option of this function allow you to split data in as many .h5
    files as iterations possible for the input stagCloudData.
    Note that in order to save space, all grid information are not repeated in .h5
    files exported. Thus, if you decide to activate the multifile option, the first
    .h5 generated will be bigger than the one generated after (have a look on the
    .xfmd file, it is explicitelly explained inside).
    Concerning geometies:
      - 'yy' deal with non overlapping stagData object so with stagData.x1,
        stagData.x2, stagData.y1 ... fields
      - 'cart3D', read stagData.x, .y, .z and .v fields for a scalar field as
        example
      - 'spherical' deal with the spherical grid so stagData.x, .y, .z fields
        and ignore .xc, .yc and .zc fields.
    Triangulation are computed using the function stag2VTU.

    <i> : fname = str, name of the exported file without any extention
          stagCloudData = stagCloudData object, stagCloudData object that will
                          be transform into meshed file
          multifile = bool, if multifile is True then, data generated will
                      split in 1 .xdmf file + as many .h5 files as iterations
                      possible for the input stagCloudData, else all data will
                      be stored in a uniq .h5 file
          timeAsF   = bool, [only if multifile is False] is timeAsF is True then,
                      the time will not be implemented as time for VTK files
                      but will appears as a fields. Put another way, to navigate on
                      time on paraview, you will change the display field and not the
                      the time value (that will be unique).
                      if timeAsF is False, you will ends up with a classical time VTK
                      where you can press 'play' to see the evolution of your
                      fields through time.
          timepar = int, define your option for the time of each field:
                            timepar = 0    -> time in the .xdmf will be the file
                                              index of the stagCloudData
                            timepar = 1    -> time in the .xdmf will be the internal
                                              (adim) simulation ages (.simuAge)
                            timepar = 2    -> time in the .xdmf will be the internal
                                              simulation steps (.ti_step)
                    [Default, timepar=1]
          creat_pointID = bool, if True then creat the list of points ID sorted as
                          it is in the field stagData.x (Yin and Yang together).
                          This field will then transfert to the writing function and
                          the .h5/.xdmf will have an extra field corresponding to 
                          these points ID (e.g. very usefull if post processing with TTK)
          verbose = bool, if True, then generate a verbose output
                    [Default, verbose=True]
          extended_verbose = bool, if True, then generate an extended verbose output
                    conditioning the verbose output of subfunction called by
                    stagCloud2timeVTU
                    [Default, verbose=True]
    """
    pName = 'stagCloud2timeVTU'
    im("Creation of a time VTU from a stagCloudData",pName,verbose)
    # test the geometry of clouddata: have to be 3D 
    if stagCloudData.geometry not in ['cart3D','spherical','yy']:
        raise VisuGridGeometryError(stagCloudData.geometry,"'cart3D' or 'spherical' or 'yy'")
    #--------------
    if path[-1] != '/':
        path += '/'
    #--------------
    fname_vtk = fname+'.xdmf'
    if multifile:
        if timeAsF:
            im("Non fatal WARNING: you ask for the option 'timeAsF', however, this function is not compatible with the multifile=True condition.\nSee the documentation for more details",pName,verbose)
            im("Export mesh data in several files",pName,verbose)
            im("Iteration on the cloud:",pName,verbose)
        for i in stagCloudData.indices:
            stagCloudData.iterate()
            ifname = stagCloudData.drop.fname
            im("    Current cloud drop: "+ifname,pName,verbose)
            if timepar == 0:
                timex = stagCloudData.indices[stagCloudData.ci]
            elif timepar == 1:
                timex = stagCloudData.drop.simuAge
            elif timepar == 2:
                timex = stagCloudData.drop.ti_step
            # ----------
            fname_h5 = ifname+'.h5'
            if i == stagCloudData.indices[0]:
                fname_h5_geom = fname_h5
            Points,ElementNumbers,vstack,pointID = stag2VTU(ifname,stagCloudData.drop,verbose=extended_verbose,return_only=True,creat_pointID=creat_pointID)
            # ----------
            if i == stagCloudData.indices[0]:
                step = 'init'
            elif i == stagCloudData.indices[-1]:
                step = 'end'
            else:
                step = 'add'
            # ----------
            __write_time_H5(step,fname_h5,stagCloudData.drop,Points,ElementNumbers,vstack,path=path,pointID=pointID)
            __write_time_xdmf(step,fname_vtk,fname_h5_geom,fname_h5,stagCloudData.drop,Points.shape[0],Points.shape[1],\
                        ElementNumbers.shape[0],ElementNumbers.shape[1],vstack,path=path,timex=timex,pointID=pointID,\
                        timeAsF=timeAsF)
    
    else:
        im("Export mesh data in a unique file",pName,verbose)
        fname_h5      = fname+'.h5'
        fname_h5_geom = fname+'.h5'
        im("Iteration on the cloud:",pName,verbose)
        for i in stagCloudData.indices:
            stagCloudData.iterate()
            ifname = stagCloudData.drop.fname
            im("    Current cloud drop: "+ifname,pName,verbose)
            if timepar == 0:
                timex = stagCloudData.indices[stagCloudData.ci]
            elif timepar == 1:
                timex = stagCloudData.drop.simuAge
            elif timepar == 2:
                timex = stagCloudData.drop.ti_step
            # ----------
            Points,ElementNumbers,vstack,pointID = stag2VTU(ifname,stagCloudData.drop,verbose=extended_verbose,return_only=True,creat_pointID=creat_pointID)
            # ----------
            if i == stagCloudData.indices[0]:
                step = 'init'
            elif i == stagCloudData.indices[-1]:
                step = 'end'
            else:
                step = 'add'
            # ----------
            __write_time_H5(step,fname_h5,stagCloudData.drop,Points,ElementNumbers,vstack,path=path,multifile=multifile,field_preposition=ifname,pointID=pointID)
            __write_time_xdmf(step,fname_vtk,fname_h5_geom,fname_h5,stagCloudData.drop,Points.shape[0],Points.shape[1],\
                        ElementNumbers.shape[0],ElementNumbers.shape[1],vstack,path=path,multifile=multifile,timex=timex,field_preposition=ifname,pointID=pointID,\
                        timeAsF=timeAsF)
    # --- ending
    im("Creation of time VTU: done!",pName,verbose)
    im("File log:",pName,verbose)
    im("    - .xdmf file:\t"+path+fname_vtk,pName,verbose)
    if multifile:
        im("    - .h5 files:",pName,verbose)
        im("       -> "+str(len(stagCloudData.indices))+' files',pName,verbose)
        im("       -> stored in: "+path,pName,verbose)
    else:
        im("    - .h5 file:\t"+path+fname_h5,pName,verbose)








def stagCloud2timeVTU_TTK(fname,stagCloudData,cloudPlateGather,multifile=True,field='topo',timeAsF=False,timepar=1,path='./',\
                      creat_pointID=False,verbose=True,extended_verbose=False):
    """ -- Geometry Adaptative Toolkit transforming stagCloudData into VTU --
    This function creats readable file for Paraview for an efficient 
    3D visualization of data contained in an input stagCloudData instance.
    This function works directly on a stagCloudData input object and adapts the
    constuction of the triangulation according to this geometry and iterate
    automatically in the cloud. Due to large quantities of data created, outputs
    are necessarily written under the coupled .h5 and .xdmf format (more efficient)
    The 'multifile' option of this function allow you to split data in as many .h5
    files as iterations possible for the input stagCloudData.
    Note that in order to save space, all grid information are not repeated in .h5
    files exported. Thus, if you decide to activate the multifile option, the first
    .h5 generated will be bigger than the one generated after (have a look on the
    .xfmd file, it is explicitelly explained inside).
    Concerning geometies:
      - 'yy' deal with non overlapping stagData object so with stagData.x1,
        stagData.x2, stagData.y1 ... fields
      - 'cart3D', read stagData.x, .y, .z and .v fields for a scalar field as
        example
      - 'spherical' deal with the spherical grid so stagData.x, .y, .z fields
        and ignore .xc, .yc and .zc fields.
    Triangulation are computed using the function stag2VTU.

    <i> : fname = str, name of the exported file without any extention
          stagCloudData = stagCloudData object, stagCloudData object that will
                          be transform into meshed file
          multifile = bool, if multifile is True then, data generated will
                      split in 1 .xdmf file + as many .h5 files as iterations
                      possible for the input stagCloudData, else all data will
                      be stored in a uniq .h5 file
          timeAsF   = bool, [only if multifile is False] is timeAsF is True then,
                      the time will not be implemented as time for VTK files
                      but will appears as a fields. Put another way, to navigate on
                      time on paraview, you will change the display field and not the
                      the time value (that will be unique).
                      if timeAsF is False, you will ends up with a classical time VTK
                      where you can press 'play' to see the evolution of your
                      fields through time.
          timepar = int, define your option for the time of each field:
                            timepar = 0    -> time in the .xdmf will be the file
                                              index of the stagCloudData
                            timepar = 1    -> time in the .xdmf will be the internal
                                              (adim) simulation ages (.simuAge)
                            timepar = 2    -> time in the .xdmf will be the internal
                                              simulation steps (.ti_step)
                    [Default, timepar=1]
          creat_pointID = bool, if True then creat the list of points ID sorted as
                          it is in the field stagData.x (Yin and Yang together).
                          This field will then transfert to the writing function and
                          the .h5/.xdmf will have an extra field corresponding to 
                          these points ID (e.g. very usefull if post processing with TTK)
          verbose = bool, if True, then generate a verbose output
                    [Default, verbose=True]
          extended_verbose = bool, if True, then generate an extended verbose output
                    conditioning the verbose output of subfunction called by
                    stagCloud2timeVTU
                    [Default, verbose=True]
    """
    pName = 'stagCloud2timeVTU'
    im("Creation of a time VTU from a stagCloudData",pName,verbose)
    # test the geometry of clouddata: have to be 3D 
    if stagCloudData.geometry not in ['cart3D','spherical','yy']:
        raise VisuGridGeometryError(stagCloudData.geometry,"'cart3D' or 'spherical' or 'yy'")
    #--------------
    if path[-1] != '/':
        path += '/'
    #--------------
    fname_vtk = fname+'.xdmf'
    if multifile:
        if timeAsF:
            im("Non fatal WARNING: you ask for the option 'timeAsF', however, this function is not compatible with the multifile=True condition.\nSee the documentation for more details",pName,verbose)
            im("Export mesh data in several files",pName,verbose)
            im("Iteration on the cloud:",pName,verbose)
        for i in stagCloudData.indices:
            stagCloudData.iterate()
            print()
            cloudPlateGather.iterate()
            if field == 'topo':
                cloudPlateGather.drop.topologicalSimplification()
                sd = cloudPlateGather.drop.export_topo(stagCloudData.drop)
            elif field == 'magGSV':
                sd = cloudPlateGather.drop.export_magGSV(stagCloudData.drop)
            print()
            
            ifname = stagCloudData.drop.fname
            im("    Current cloud drop: "+ifname,pName,verbose)
            if timepar == 0:
                timex = stagCloudData.indices[stagCloudData.ci]
            elif timepar == 1:
                timex = stagCloudData.drop.simuAge
            elif timepar == 2:
                timex = stagCloudData.drop.ti_step
            # ----------
            fname_h5 = ifname+'.h5'
            if i == stagCloudData.indices[0]:
                fname_h5_geom = fname_h5
            Points,ElementNumbers,vstack,pointID = stag2VTU(ifname,stagCloudData.drop,verbose=extended_verbose,return_only=True,creat_pointID=creat_pointID)
            # ----------
            if i == stagCloudData.indices[0]:
                step = 'init'
            elif i == stagCloudData.indices[-1]:
                step = 'end'
            else:
                step = 'add'
            # ----------
            __write_time_H5(step,fname_h5,stagCloudData.drop,Points,ElementNumbers,vstack,path=path,pointID=pointID)
            __write_time_xdmf(step,fname_vtk,fname_h5_geom,fname_h5,stagCloudData.drop,Points.shape[0],Points.shape[1],\
                        ElementNumbers.shape[0],ElementNumbers.shape[1],vstack,path=path,timex=timex,pointID=pointID,\
                        timeAsF=timeAsF)
    
    else:
        im("Export mesh data in a unique file",pName,verbose)
        fname_h5      = fname+'.h5'
        fname_h5_geom = fname+'.h5'
        im("Iteration on the cloud:",pName,verbose)
        for i in stagCloudData.indices:
            stagCloudData.iterate()
            print()
            cloudPlateGather.iterate()
            if field == 'topo':
                cloudPlateGather.drop.topologicalSimplification()
                sd = cloudPlateGather.drop.export_topo(stagCloudData.drop)
            elif field == 'magGSV':
                sd = cloudPlateGather.drop.export_magGSV(stagCloudData.drop)
            print()
            
            ifname = stagCloudData.drop.fname
            im("    Current cloud drop: "+ifname,pName,verbose)
            if timepar == 0:
                timex = stagCloudData.indices[stagCloudData.ci]
            elif timepar == 1:
                timex = stagCloudData.drop.simuAge
            elif timepar == 2:
                timex = stagCloudData.drop.ti_step
            # ----------
            Points,ElementNumbers,vstack,pointID = stag2VTU(ifname,stagCloudData.drop,verbose=extended_verbose,return_only=True,creat_pointID=creat_pointID)
            # ----------
            if i == stagCloudData.indices[0]:
                step = 'init'
            elif i == stagCloudData.indices[-1]:
                step = 'end'
            else:
                step = 'add'
            # ----------
            __write_time_H5(step,fname_h5,stagCloudData.drop,Points,ElementNumbers,vstack,path=path,multifile=multifile,field_preposition=ifname,pointID=pointID)
            __write_time_xdmf(step,fname_vtk,fname_h5_geom,fname_h5,stagCloudData.drop,Points.shape[0],Points.shape[1],\
                        ElementNumbers.shape[0],ElementNumbers.shape[1],vstack,path=path,multifile=multifile,timex=timex,field_preposition=ifname,pointID=pointID,\
                        timeAsF=timeAsF)
    # --- ending
    im("Creation of time VTU: done!",pName,verbose)
    im("File log:",pName,verbose)
    im("    - .xdmf file:\t"+path+fname_vtk,pName,verbose)
    if multifile:
        im("    - .h5 files:",pName,verbose)
        im("       -> "+str(len(stagCloudData.indices))+' files',pName,verbose)
        im("       -> stored in: "+path,pName,verbose)
    else:
        im("    - .h5 file:\t"+path+fname_h5,pName,verbose)







def __write_time_H5(step,fname_h5,stagData,Points,ElementNumbers,vstack,path='./',\
                    multifile=True,field_preposition='',pointID=None):
    """
    This function creats and exports trianguled geometry and field(s) into
    a .h5 files under the name fname. This function was built to be used by
    stagCloud2timeVTU() and __write_time_xdmf(). 
    <i> : step = str, describe the state of the work.
                 step = 'init'    -> initialization
                 step = 'add'     -> add data to files
                 step = 'end'     -> end files
          fname_h5 = str, name of the exported .h5 file without .h5 extention
          stagData = stagData object, stagData object that will be transform
                     into meshed file (from a stagCloudData.drop)
          Points = np.ndarray, matrices of points as defined in stag2VTU()
          ElementNumbers = list, list of how points have to be organized to
                           form 3D-'roofs' in space
          vstack = list OR tuple of lists, if stagData.fieldNature == Scalar,
                   vstack must be a list and if  == Vectorial, must be a 
                   tuple as vstack=(vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)
          path = str, path where you want to export your new .vtu file.
                 [Default: path='./']
          multifile = bool, if multifile is True then, data generated will
                      split in as many .h5 files as iterations possible
                      for the input stagCloudData, else all data will be
                      stored in a uniq .h5 file
          field_preposition = str, header of .h5 field if multifile = False
          pointID = np.ndarray (or None), contains the list of points ID sorted as
                    it is in the field stagData.x (Yin and Yang together). If pointID != None,
                    then, the .h5/.xdmf will have an extra field corresponding to 
                    these points ID (e.g. very usefull if post processing with TTK)
    """
    Points         = np.asarray(Points)
    ElementNumbers = np.asarray(ElementNumbers)
    fname = fname_h5.split('.')[0]

    if stagData.fieldNature == 'Scalar':
        Data       = np.asarray(vstack)
    else:
        if stagData.geometry == 'cart3D':
            (vstackx,vstacky,vstackz,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3])
            Datax  = np.asarray(vstackx)
            Datay  = np.asarray(vstacky)
            Dataz  = np.asarray(vstackz)
            Datap  = np.asarray(vstackp)
        elif stagData.geometry == 'yy' or stagData.geometry == 'spherical':
            (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3],vstack[4],vstack[5],vstack[6])
            Datax  = np.asarray(vstackx)
            Datay  = np.asarray(vstacky)
            Dataz  = np.asarray(vstackz)
            Datar  = np.asarray(vstackr)
            Datath = np.asarray(vstacktheta)
            Dataph = np.asarray(vstackphi)
            Datap  = np.asarray(vstackp)
    # =======================================
    # Write HDF5 file
    #Code:
    #  Data0 = scalar field
    #  Data1 = cartesian velocities
    #  Data2 = spherical velocities
    #  Data3 = Pressure
    # -----
    if step == 'init':
        fid = h5py.File(path+fname_h5, 'w')
        dset = fid.create_dataset("Points", data=Points, dtype=np.float32)
        dset = fid.create_dataset("NumberOfElements", data=ElementNumbers, dtype=np.int32)    
        if multifile:
            if stagData.fieldNature == 'Scalar':
                dset = fid.create_dataset("Data0", data=Data, dtype=np.float32)
            else:
                if stagData.geometry == 'cart3D':
                    Data = np.array([Datax,Datay,Dataz]).T
                    dset = fid.create_dataset("Data1", data=Data, dtype=np.float32)
                elif stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                    Data = np.array([Datax,Datay,Dataz]).T
                    dset = fid.create_dataset("Data1", data=Data, dtype=np.float32)
                    Data = np.array([Datar,Datath,Dataph]).T
                    dset = fid.create_dataset("Data2", data=Data, dtype=np.float32)
                Data = Datap
                dset = fid.create_dataset("Data3", data=Data, dtype=np.float32)
            if pointID is not None:
                dset = fid.create_dataset("pointID", data=pointID, dtype=np.int32)
        else:
            if stagData.fieldNature == 'Scalar':
                dset = fid.create_dataset(field_preposition+"_Data0", data=Data, dtype=np.float32)
            else:
                if stagData.geometry == 'cart3D':
                    Data = np.array([Datax,Datay,Dataz]).T
                    dset = fid.create_dataset(field_preposition+"_Data1", data=Data, dtype=np.float32)
                elif stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                    Data = np.array([Datax,Datay,Dataz]).T
                    dset = fid.create_dataset(field_preposition+"_Data1", data=Data, dtype=np.float32)
                    Data = np.array([Datar,Datath,Dataph]).T
                    dset = fid.create_dataset(field_preposition+"_Data2", data=Data, dtype=np.float32)
                Data = Datap
                dset = fid.create_dataset(field_preposition+"_Data3", data=Data, dtype=np.float32)
            if pointID is not None:
                dset = fid.create_dataset(field_preposition+"_pointID", data=pointID, dtype=np.int32)
        fid.close()

    else:
        if multifile:
            fid = h5py.File(path+fname_h5, 'w')
            if stagData.fieldNature == 'Scalar':
                dset = fid.create_dataset("Data0", data=Data, dtype=np.float32)
            else:
                if stagData.geometry == 'cart3D':
                    Data = np.array([Datax,Datay,Dataz]).T
                    dset = fid.create_dataset("Data1", data=Data, dtype=np.float32)
                elif stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                    Data = np.array([Datax,Datay,Dataz]).T
                    dset = fid.create_dataset("Data1", data=Data, dtype=np.float32)
                    Data = np.array([Datar,Datath,Dataph]).T
                    dset = fid.create_dataset("Data2", data=Data, dtype=np.float32)
                Data = Datap
                dset = fid.create_dataset("Data3", data=Data, dtype=np.float32)
            if pointID is not None:
                dset = fid.create_dataset("pointID", data=pointID, dtype=np.int32)
            fid.close()
        else:
            fid = h5py.File(path+fname_h5, 'a')
            if stagData.fieldNature == 'Scalar':
                dset = fid.create_dataset(field_preposition+"_Data0", data=Data, dtype=np.float32)
            else:
                if stagData.geometry == 'cart3D':
                    Data = np.array([Datax,Datay,Dataz]).T
                    dset = fid.create_dataset(field_preposition+"_Data1", data=Data, dtype=np.float32)
                elif stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                    Data = np.array([Datax,Datay,Dataz]).T
                    dset = fid.create_dataset(field_preposition+"_Data1", data=Data, dtype=np.float32)
                    Data = np.array([Datar,Datath,Dataph]).T
                    dset = fid.create_dataset(field_preposition+"_Data2", data=Data, dtype=np.float32)
                Data = Datap
                dset = fid.create_dataset(field_preposition+"_Data3", data=Data, dtype=np.float32)
            if pointID is not None:
                dset = fid.create_dataset(field_preposition+"_pointID", data=pointID, dtype=np.int32)
            fid.close()





def __write_time_xdmf(step,fname_vtk,fname_h5_geom,fname_h5,stagData,Points_shape0,Points_shape1,\
                      ElementNumbers_shape0,ElementNumbers_shape1,vstack,path='./',multifile=True,\
                      timeAsF=False,timex=None,field_preposition='',pointID=None):
    """
    This function creats and exports the geometry description file .xdmf
    under the name fname. This function was built to be used by
    stagCloud2timeVTU() and __write_time_H5(). 
    <i> : step = str, describe the state of the work.
                 step = 'init'    -> initialization
                 step = 'add'     -> add data to files
                 step = 'end'     -> end files
          fname_vtk     = str, name of the exported .xdmf file without .xdmf extention
          fname_h5_geom = str, name of the .h5 file containing the grid.
          fname_h5      = str, name of the .h5 file corresponding to the current step.
          stagData = stagData object, stagData object that will be transform
                     into meshed file (from a stagCloudData.drop)
          Points_shape0 = element 0 of the Points matrix shape
          Points_shape1 = element 1 of the Points matrix shape
          ElementNumbers_shape0 = element 0 of the ElementNumbers matrix shape
          ElementNumbers_shape1 = element 1 of the ElementNumbers matrix shape
          vstack = list OR tuple of lists, if stagData.fieldNature == Scalar,
                   vstack must be a list and if  == Vectorial, must be a 
                   tuple as vstack=(vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)
          path = str, path where you want to export your new .vtu file.
                 [Default: path='./']
          multifile = bool, if multifile is True then, data generated will
                      split in as many .h5 files as iterations possible
                      for the input stagCloudData, else all data will be
                      stored in a uniq .h5 file
          timeAsF   = bool, [only if multifile is False] is timeAsF is True then,
                      the time will not be implemented as time for VTK files
                      but will appears as a fields. Put another way, to navigate on
                      time on paraview, you will change the display field and not the
                      the time value (that will be unique).
                      if timeAsF is False, you will ends up with a classical time VTK
                      where you can press 'play' to see the evolution of your
                      fields through time.
          field_preposition = str, header of .h5 field if multifile = False
          timex = int/float, value of the time that have to be entered in the
                  .xdmf file.
    """
    if step == 'init':
        if not timeAsF:
            # Write the header for a structured grid:
            fid       = open(path+fname_vtk,'w')  # Open the file for the first time
            fid.write('<Xdmf Version="3.0">\n')
            fid.write('<Domain>\n')
            fid.write('<Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">\n\n')
            fid.close()
        else:
            fid       = open(path+fname_vtk,'w')
            fid.write('<Xdmf Version="3.0">\n')
            fid.write('<Domain>\n')
            fid.write('<Grid Name="Grid">\n\n')
            fid.write('    <Geometry GeometryType="XYZ">\n')
            fid.close()
    # Prepare the data set
    if stagData.fieldNature == 'Scalar':
        Data       = np.asarray(vstack)
    else:
        if stagData.geometry == 'cart3D':
            (vstackx,vstacky,vstackz,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3])
            Datax  = np.asarray(vstackx)
            Datay  = np.asarray(vstacky)
            Dataz  = np.asarray(vstackz)
            Datap  = np.asarray(vstackp)
        elif stagData.geometry == 'yy' or stagData.geometry == 'spherical':
            (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3],vstack[4],vstack[5],vstack[6])
            Datax  = np.asarray(vstackx)
            Datay  = np.asarray(vstacky)
            Dataz  = np.asarray(vstackz)
            Datar  = np.asarray(vstackr)
            Datath = np.asarray(vstacktheta)
            Dataph = np.asarray(vstackphi)
            Datap  = np.asarray(vstackp)
    
    # Write the geometry
    fid       = open(path+fname_vtk,'a')
    if not timeAsF or np.logical_and(timeAsF,step == 'init'):
        # Re-open the existing the header for a structured grid:
        if not timeAsF:
            fid.write('<Grid Name="Grid">\n\n')
            fid.write('    <Time Value="%s" />\n'%timex)
            fid.write('    <Geometry GeometryType="XYZ">\n')
        # =======================================
        # Write Points
        fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n' %\
                    (Points_shape0,Points_shape1))
        fid.write('            '+fname_h5_geom+':/Points\n')
        fid.write('        </DataItem>\n')
        fid.write('    </Geometry>\n\n')
        # =======================================
        # Write NumberOfElements
        fid.write('    <Topology NodesPerElement="%s" NumberOfElements="%s" TopologyType="Wedge">\n' %\
                    (ElementNumbers_shape1, ElementNumbers_shape0))
        fid.write('        <DataItem DataType="Int" Dimensions="%s %s" Format="HDF" Precision="8">\n' %\
                    ((ElementNumbers_shape0, ElementNumbers_shape1)))
        fid.write('            '+fname_h5_geom+':/NumberOfElements\n')
        fid.write('        </DataItem>\n')
        fid.write('    </Topology>\n\n')

    # =======================================
    # Write field
    if multifile:
        if stagData.fieldNature == 'Scalar':
            if timeAsF:
                Attname = ''.join(stagData.fieldType.strip().split(' ')) + '_%s' %timex
            else:
                Attname = stagData.fieldType
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % Attname)
            fid.write('        <DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    Data.shape[0])
            fid.write('            '+fname_h5+':/Data0\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
        else:
            # ---- Cartesian Velocities ----
            if timeAsF:
                Attname = ''.join('Cartesian Velo'.strip().split(' ')) + '_%s' %timex
            else:
                Attname = 'Cartesian Velocity'
            fid.write('    <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % Attname)
            fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%\
                    (Datax.shape[0], 3))
            fid.write('            '+fname_h5+':/Data1\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
            if stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                if timeAsF:
                    Attname = ''.join('Spherical Velo'.strip().split(' ')) + '_%s' %timex
                else:
                    Attname = 'Spherical Velocity'
                # ----  Shperical Velocities ----
                fid.write('    <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % Attname)
                fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%\
                        (Datar.shape[0], 3))
                fid.write('            '+fname_h5+':/Data2\n')
                fid.write('        </DataItem>\n')
                fid.write('    </Attribute>\n\n')
            # ---- Pressure ----
            if timeAsF:
                Attname = 'Pressure' + '_%s' %timex
            else:
                Attname = 'Pressure'
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % Attname)
            fid.write('        <DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    Datap.shape[0])
            fid.write('            '+fname_h5+':/Data3\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
        # ---- PointID ----
        if pointID is not None:
            if timeAsF:
                Attname = 'PointID' + '_%s' %timex
            else:
                Attname = 'PointID'
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % Attname)
            fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    pointID.shape[0])
            fid.write('            '+fname_h5+':/pointID\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
        # Ending
        fid.write('</Grid>\n\n')
        # close the file
        fid.close()
    else:
        if stagData.fieldNature == 'Scalar':
            if timeAsF:
                Attname = ''.join(stagData.fieldType.strip().split(' ')) + '_%s' %timex
            else:
                Attname = stagData.fieldType
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % Attname)
            fid.write('        <DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    Data.shape[0])
            fid.write('            '+fname_h5+':/'+field_preposition+'_Data0\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
        else:
            # ---- Cartesian Velocities ----
            if timeAsF:
                Attname = ''.join('Cartesian Velo'.strip().split(' ')) + '_%s' %timex
            else:
                Attname = 'Cartesian Velocity'
            fid.write('    <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % Attname)
            fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%\
                    (Datax.shape[0], 3))
            fid.write('            '+fname_h5+':/'+field_preposition+'_Data1\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
            if stagData.geometry == 'yy' or stagData.geometry == 'spherical':
                # ----  Shperical Velocities ----
                if timeAsF:
                    Attname = ''.join('Spherical Velo'.strip().split(' ')) + '_%s' %timex
                else:
                    Attname = 'Spherical Velocity'
                fid.write('    <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % Attname)
                fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%\
                        (Datar.shape[0], 3))
                fid.write('            '+fname_h5+':/'+field_preposition+'_Data2\n')
                fid.write('        </DataItem>\n')
                fid.write('    </Attribute>\n\n')
            # ---- Pressure ----
            if timeAsF:
                Attname = 'Pressure' + '_%s' %timex
            else:
                Attname = 'Pressure'
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % Attname)
            fid.write('        <DataItem DataType="Float" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    Datap.shape[0])
            fid.write('            '+fname_h5+':/'+field_preposition+'_Data3\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
        # ---- PointID ----
        if pointID is not None:
            if timeAsF:
                Attname = 'PointID' + '_%s' %timex
            else:
                Attname = 'PointID'
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % Attname)
            fid.write('        <DataItem DataType="Int" Dimensions="%s" Format="HDF" Precision="8">\n'%\
                    pointID.shape[0])
            fid.write('            '+fname_h5+':/'+field_preposition+'_pointID\n')
            fid.write('        </DataItem>\n')
            fid.write('    </Attribute>\n\n')
        # Ending
        if not timeAsF:
            fid.write('</Grid>\n\n')
        # close the file
        fid.close()

    if step == 'end':
        # Ending, re-open the existing the header for a structured grid:
        fid       = open(path+fname_vtk,'a')
        fid.write('</Grid>\n')
        fid.write('</Domain>\n')
        fid.write('</Xdmf>')
        fid.close()







# ============================================================
# 
# Ci dessous book2VTU: pour le moment que la copy de stag2VTU
# 
# ============================================================

def book2VTU(fname,stagData,path='./',ASCII=False,verbose=True,return_only=False,creat_pointID=False):
    """ -- Geometry Adaptative Toolkit transforming stagData into VTU --
    This function creats readable file for Paraview for an efficient 
    3D visualization of data contain in an input stagData instance.
    This function works directly on a stagData input object and adapts the
    constuction of the triangulation according to this geometry. Furthermore
    stag2VTU is able to deal with vectorial and scalar fields. Outputs can be
    written under two format: 1. the explicit ascii .vtu format or 2. the
    coupled .h5 and .xdmf format (more efficient)
    Concerning geometies:
      - 'yy' deal with non overlapping stagData object so with stagData.x1,
        stagData.x2, stagData.y1 ... fields
      - 'cart3D', read stagData.x, .y, .z and .v fields for a scalar field as
        example
      - 'spherical' deal with the spherical grid so stagData.x, .y, .z fields
        and ignore .xc, .yc and .zc fields.
    Note that the internal field stagData.slayers of the stagData object
    must be filled!
    --> stag2vtu is main function of the pypStag Visualization ToolKit

    <i> : fname = str, name of the exported file without any extention
          stagData = stagData object, stagData object that will be transform
                     into meshed file
          path = str, path where you want to export your new meshed file.
                 [Default: path='./']
          ASCII = bool, if True, the export paraview file will be write
                  in ASCII .vtu format (take lots of memory space),
                  if not ASCII, the export will be partitioned in
                  a .xdmf and .h5 file (very efficient in writing,
                  reading and save lots of memory).
                  [Default, ASCII=False]
          verbose = bool, if True, then generate a verbose output
                    [Default, verbose=True]
          return_only = bool, if True, stag2VTU() will just manage the
                        3D tetrahedralization of the grid, and returns:
                        (Points,ElementNumbers,vstack) corresponding to 
                        points matrix, meshing doc and corresponding field.
                        Do not call __writeVKTStag() if True
                        If False, generate meshed file for paraview, call
                        __writeVTKStag()
                        [Default, return_only=False]
          creat_pointID = bool, if True then creat the list of points ID sorted as
                          it is in the field stagData.x (Yin and Yang together).
                          This field will then transfert to the writing function and
                          the .h5/.xdmf will have an extra field corresponding to 
                          these points ID (e.g. very usefull if post processing with TTK)
                          WARNING: This option is only available if ASCII = False
                                   (ie .h5/.xdmf output)
    """
    pName = 'stag2VTU'
    im('pypStag Visualization ToolKit',pName,verbose)
    if ASCII:
        im('Requested: stagData -> .vtu',pName,verbose)
    else:
        im('Requested: stagData -> .xdmf + .h5',pName,verbose)

    #--------------
    if path[-1] != '/':
        path += '/'
    #--------------
    if stagData.geometry == 'cart3D' or stagData.geometry == 'spherical':
        """
        Adaptations for the exportation of the complete
        stagData object with a 3D cartesian or spherical
        geometries.
        """
        im('    - Grid preparation',pName,verbose)
        # =======================================
        #Re-formating initial grids data
        Nx = len(stagData.x_coords)
        Ny = len(stagData.y_coords)
        Nz = len(stagData.slayers)
        NxNy = Nx*Ny
        x = np.array(stagData.x).reshape(NxNy*Nz, order='F')
        y = np.array(stagData.y).reshape(NxNy*Nz, order='F')
        z = np.array(stagData.z).reshape(NxNy*Nz, order='F')
        v = np.array(stagData.v).reshape(NxNy*Nz, order='F')
        # =========================================================================
        # 1) Take the surface of the 2 grids, patch together and triangulate
        # =========================================================================
        im('    - Planar triangulation',pName,verbose)
        # =======================================
        #Computation of the triangulation of just a level of depth
        triPlanar_simplices = triangulationPlanar(Nx,Ny,ordering='yx')
        nod = len(triPlanar_simplices)
        # =========================================================================
        # 2) Create a 3D grid with tetrahedron elements
        # =========================================================================
        # Number all gridpoints we have
        NUM         = np.array(range(0,NxNy*Nz))
        NUMBER      = NUM.reshape((NxNy,Nz), order='F')
        # Make a loop over all levels
        ElementNumbers = np.zeros(((Nz-1)*nod,6),dtype=np.int32)
        num_upper      = np.zeros(NxNy)
        num_lower      = np.zeros(NxNy)
        num_tri = np.zeros((nod,6))
        for iz in range(Nz-1):
            num_upper      = NUMBER[:,iz+1]
            num_lower      = NUMBER[:,iz]
            num_tri[:,0] = num_upper[triPlanar_simplices[:,0]]
            num_tri[:,1] = num_upper[triPlanar_simplices[:,1]]
            num_tri[:,2] = num_upper[triPlanar_simplices[:,2]]
            num_tri[:,3] = num_lower[triPlanar_simplices[:,0]]
            num_tri[:,4] = num_lower[triPlanar_simplices[:,1]]
            num_tri[:,5] = num_lower[triPlanar_simplices[:,2]]
            ElementNumbers[nod*iz:nod*(iz+1),:] = num_tri
        # =======================================
        # Convert data into correct vector format
        im('    - Convert data into correct vector format',pName,verbose)
        im('      - Grid',pName,verbose)
        Points = np.zeros((NxNy*Nz,3))
        Points[:,0] = np.array(x).reshape((NxNy*Nz), order='F')
        Points[:,1] = np.array(y).reshape((NxNy*Nz), order='F')
        Points[:,2] = np.array(z).reshape((NxNy*Nz), order='F')
        # ===================
        im('      - Field',pName,verbose)
        if stagData.fieldNature == 'Scalar' or stagData.fieldNature == '':
            V_yy  = np.array(v).reshape(NxNy,Nz, order='F')
            vstack = V_yy.reshape((NxNy*Nz), order='F')
        # ===================
        if stagData.fieldNature == 'Vectorial':
            if stagData.geometry == 'cart3D':
                # ------ Vx ------
                V_vx  = np.array(stagData.vx).reshape(NxNy,Nz, order='F')
                vstackx = V_vx.reshape((NxNy*Nz), order='F')
                # ------ Vy ------
                V_vy  = np.array(stagData.vy).reshape(NxNy,Nz, order='F')
                vstacky = V_vy.reshape((NxNy*Nz), order='F')
                # ------ Vz ------
                V_vz  = np.array(stagData.vz).reshape(NxNy,Nz, order='F')
                vstackz = V_vz.reshape((NxNy*Nz), order='F')
                # ------ P ------
                V_vp  = np.array(stagData.P).reshape(NxNy,Nz, order='F')
                vstackp = V_vp.reshape((NxNy*Nz), order='F')
                # ------ stack ------
                vstack = (vstackx,vstacky,vstackz,vstackp)
            elif stagData.geometry == 'spherical':
                # ------ Vx ------
                V_vx  = np.array(stagData.vx).reshape(NxNy,Nz, order='F')
                vstackx = V_vx.reshape((NxNy*Nz), order='F')
                # ------ Vy ------
                V_vy  = np.array(stagData.vy).reshape(NxNy,Nz, order='F')
                vstacky = V_vy.reshape((NxNy*Nz), order='F')
                # ------ Vz ------
                V_vz  = np.array(stagData.vz).reshape(NxNy,Nz, order='F')
                vstackz = V_vz.reshape((NxNy*Nz), order='F')
                # ------ Vr ------
                V_vr  = np.array(stagData.vr).reshape(NxNy,Nz, order='F')
                vstackr = V_vr.reshape((NxNy*Nz), order='F')
                # ------ Vtheta ------
                V_theta  = np.array(stagData.vtheta).reshape(NxNy,Nz, order='F')
                vstacktheta = V_theta.reshape((NxNy*Nz), order='F')
                # ------ Vphi ------
                V_phi  = np.array(stagData.vphi).reshape(NxNy,Nz, order='F')
                vstackphi = V_phi.reshape((NxNy*Nz), order='F')
                # ------ P ------
                V_vp  = np.array(stagData.P).reshape(NxNy,Nz, order='F')
                vstackp = V_vp.reshape((NxNy*Nz), order='F')
                # ------ stack ------
                vstack = (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)

    elif stagData.geometry == 'yy':
        """
        Adaptations for the exportation of the complete
        stagData object with a Yin-Yang geoemtry.
        """
        im('    - Grid preparation',pName,verbose)
        # =======================================
        #Re-formating initial grids data
        Nz   = len(stagData.slayers)      #Number of depth layers
        NxNy = int(len(stagData.x1)/Nz)   #Number of points for each layers
        x1     = stagData.x1.reshape(NxNy,Nz)
        x2     = stagData.x2.reshape(NxNy,Nz)
        y1     = stagData.y1.reshape(NxNy,Nz)
        y2     = stagData.y2.reshape(NxNy,Nz)
        z1     = stagData.z1.reshape(NxNy,Nz)
        z2     = stagData.z2.reshape(NxNy,Nz)
        #Re-organisation of data to have X,Y and Z grid matrices organized by depths:
        X = np.zeros((Nz,2*NxNy))
        Y = np.zeros((Nz,2*NxNy))
        Z = np.zeros((Nz,2*NxNy))
        X[:,0:NxNy]      = x1.T
        X[:,NxNy:2*NxNy] = x2.T
        Y[:,0:NxNy]      = y1.T
        Y[:,NxNy:2*NxNy] = y2.T
        Z[:,0:NxNy]      = z1.T
        Z[:,NxNy:2*NxNy] = z2.T
        # =========================================================================
        # 1) Take the surface of the 2 grids, patch together and triangulate
        # =========================================================================
        im('    - Triangulation on convex hull',pName,verbose)
        # NotaBene: _s for the surface layer
        X_s    = X[Nz-1]
        Y_s    = Y[Nz-1]
        Z_s    = Z[Nz-1]
        # =======================================
        # Triangulation of the surface using a convex hull algorithm
        points = np.array([X_s,Y_s,Z_s]).T
        triYingYang = ConvexHull(points).simplices # simple way to grid it
        nod = triYingYang.shape[0]
        # =========================================================================
        # 2) Create a 3D grid with tetrahedron elements
        # =========================================================================
        # Number all gridpoints we have
        NUM_1       = np.array(range(0,NxNy*Nz))
        NUMBER_1    = NUM_1.reshape((NxNy,Nz), order='F')
        NUMBER_2    = NUMBER_1 + NxNy*Nz
        # -- Make a loop over all levels
        # init all arrays
        ElementNumbers = np.zeros(((Nz-1)*nod,6),dtype=np.int32)
        num_upper      = np.zeros(NxNy*2)
        num_lower      = np.zeros(NxNy*2)
        num_tri = np.zeros((nod,6))
        for iz in range(Nz-1):
            num_upper[0:NxNy]      = NUMBER_1[:,iz+1]#np.array(list(num_upper1) + list(num_upper2))
            num_upper[NxNy:NxNy*2] = NUMBER_2[:,iz+1]
            num_lower[0:NxNy]      = NUMBER_1[:,iz]#np.array(list(num_upper1) + list(num_upper2))
            num_lower[NxNy:NxNy*2] = NUMBER_2[:,iz]
            num_tri[:,0] = num_upper[triYingYang[:,0]]
            num_tri[:,1] = num_upper[triYingYang[:,1]]
            num_tri[:,2] = num_upper[triYingYang[:,2]]
            num_tri[:,3] = num_lower[triYingYang[:,0]]
            num_tri[:,4] = num_lower[triYingYang[:,1]]
            num_tri[:,5] = num_lower[triYingYang[:,2]]
            ElementNumbers[nod*iz:nod*(iz+1),:] = num_tri 

        # =======================================
        # Convert data into correct vector format
        im('    - Convert data into correct vector format:',pName,verbose)
        im('      - Grid',pName,verbose)
        Points = np.zeros((2*NxNy*Nz,3))
        Points[0:NxNy*Nz,0]         = np.array(x1).reshape((NxNy*Nz), order='F')
        Points[NxNy*Nz:2*NxNy*Nz,0] = np.array(x2).reshape((NxNy*Nz), order='F')
        Points[0:NxNy*Nz,1]         = np.array(y1).reshape((NxNy*Nz), order='F')
        Points[NxNy*Nz:2*NxNy*Nz,1] = np.array(y2).reshape((NxNy*Nz), order='F')
        Points[0:NxNy*Nz,2]         = np.array(z1).reshape((NxNy*Nz), order='F')
        Points[NxNy*Nz:2*NxNy*Nz,2] = np.array(z2).reshape((NxNy*Nz), order='F')
        # ===================
        im('      - Field',pName,verbose)
        if stagData.fieldNature == 'Scalar' or stagData.fieldNature == '':
            V_yin  = np.array(stagData.v1).reshape(NxNy,Nz)
            V_yang = np.array(stagData.v2).reshape(NxNy,Nz)
            vstack = np.zeros(2*NxNy*Nz)
            vstack[0:NxNy*Nz]         = V_yin.reshape((NxNy*Nz), order='F')
            vstack[NxNy*Nz:2*NxNy*Nz] = V_yang.reshape((NxNy*Nz),order='F')
        # ===================
        if stagData.fieldNature == 'Vectorial':
            # ------ Vx ------
            V_yinx  = np.array(stagData.vx1).reshape(NxNy,Nz)
            V_yangx = np.array(stagData.vx2).reshape(NxNy,Nz)
            vstackx = np.zeros(2*NxNy*Nz)
            vstackx[0:NxNy*Nz]         = V_yinx.reshape((NxNy*Nz), order='F')
            vstackx[NxNy*Nz:2*NxNy*Nz] = V_yangx.reshape((NxNy*Nz),order='F')
            # ------ Vy ------
            V_yiny  = np.array(stagData.vy1).reshape(NxNy,Nz)
            V_yangy = np.array(stagData.vy2).reshape(NxNy,Nz)
            vstacky = np.zeros(2*NxNy*Nz)
            vstacky[0:NxNy*Nz]         = V_yiny.reshape((NxNy*Nz), order='F')
            vstacky[NxNy*Nz:2*NxNy*Nz] = V_yangy.reshape((NxNy*Nz),order='F')
            # ------ Vz ------
            V_yinz  = np.array(stagData.vz1).reshape(NxNy,Nz)
            V_yangz = np.array(stagData.vz2).reshape(NxNy,Nz)
            vstackz = np.zeros(2*NxNy*Nz)
            vstackz[0:NxNy*Nz]         = V_yinz.reshape((NxNy*Nz), order='F')
            vstackz[NxNy*Nz:2*NxNy*Nz] = V_yangz.reshape((NxNy*Nz),order='F')
            # ------ Vr ------
            V_yinr  = np.array(stagData.vr1).reshape(NxNy,Nz)
            V_yangr = np.array(stagData.vr2).reshape(NxNy,Nz)
            vstackr = np.zeros(2*NxNy*Nz)
            vstackr[0:NxNy*Nz]         = V_yinr.reshape((NxNy*Nz), order='F')
            vstackr[NxNy*Nz:2*NxNy*Nz] = V_yangr.reshape((NxNy*Nz),order='F')
            # ------ Vtheta ------
            V_yintheta  = np.array(stagData.vtheta1).reshape(NxNy,Nz)
            V_yangtheta = np.array(stagData.vtheta2).reshape(NxNy,Nz)
            vstacktheta = np.zeros(2*NxNy*Nz)
            vstacktheta[0:NxNy*Nz]         = V_yintheta.reshape((NxNy*Nz), order='F')
            vstacktheta[NxNy*Nz:2*NxNy*Nz] = V_yangtheta.reshape((NxNy*Nz),order='F')
            # ------ Vphi ------
            V_yinphi  = np.array(stagData.vphi1).reshape(NxNy,Nz)
            V_yangphi = np.array(stagData.vphi2).reshape(NxNy,Nz)
            vstackphi = np.zeros(2*NxNy*Nz)
            vstackphi[0:NxNy*Nz]         = V_yinphi.reshape((NxNy*Nz), order='F')
            vstackphi[NxNy*Nz:2*NxNy*Nz] = V_yangphi.reshape((NxNy*Nz),order='F')
            # ------ P ------
            V_yinp  = np.array(stagData.P1).reshape(NxNy,Nz)
            V_yangp = np.array(stagData.P2).reshape(NxNy,Nz)
            vstackp = np.zeros(2*NxNy*Nz)
            vstackp[0:NxNy*Nz]         = V_yinp.reshape((NxNy*Nz), order='F')
            vstackp[NxNy*Nz:2*NxNy*Nz] = V_yangp.reshape((NxNy*Nz),order='F')
            vstack = (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)
    # ===================
    if creat_pointID:
        im('      - Creat pointID',pName,verbose)
        if stagData.geometry == 'yy':
            IDyin   = np.array(range(NxNy*Nz),dtype=np.int32).reshape(NxNy,Nz)
            IDyang  = np.array(range(NxNy*Nz,2*NxNy*Nz),dtype=np.int32).reshape(NxNy,Nz)
            pointID = np.zeros(2*NxNy*Nz)
            pointID[0:NxNy*Nz]         = IDyin.reshape((NxNy*Nz), order='F')
            pointID[NxNy*Nz:2*NxNy*Nz] = IDyang.reshape((NxNy*Nz),order='F')
        else:
            pointID = np.array(range(NxNy*Nz),dtype=np.int32).reshape(NxNy*Nz, order='F')        
            pointID = pointID.reshape((NxNy*Nz), order='F')
    else:
        pointID = None
    # =========================================================================
    if not return_only:
        # Exportation under VTKUnstructuredGrid format
        if ASCII:
            im('    - Writing under .vtu format',pName,verbose)
        else:
            im('    - Writing under .xdmf + .h5 formats',pName,verbose)
        __writeVKTStag(fname,stagData,Points,ElementNumbers,vstack,ASCII=ASCII,path=path,pointID=pointID)
        im('Exportation done!',pName,verbose)
        if ASCII:
            im('File: '+fname+'.vtu',pName,verbose)
            im('Path: '+path,pName,verbose)
        else:
            im('Files: '+fname+'.xdmf + '+fname+'.h5',pName,verbose)
            im('Path : '+path,pName,verbose)
    else:
        # Return all grid/mesh elements
        return Points,ElementNumbers,vstack,pointID

















def __WriteVTU(fname,Points,ElementNumbers,vstack,fieldName,ASCII=True,path='./'):
    """ This function creats and export trianguled geometry and field(s) into
    a .vtu file under the name fname. This function was built to be used by
    stag2VTU() and stag2VTU_For_overlapping() functions.
    This function export the geometry into a VTKUnstructuredGrid format.
    <i> : fname = str, name of the exported file without any extention
          stagData = stagData object, stagData object that will be transform
                     into .vtu file
          Points = np.ndarray, matrices of points as defined in stag2VTU()
          ElementNumbers = list, list of how points have to be organized to
                           form 3D-'roofs' in space
          vstack = list OR tuple of lists, if stagData.fieldNature == Scalar,
                   vstack must be a list and if  == Vectorial, must be a 
                   tuple as vstack=(vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)
          ASCII = bool, if True, the .vtu file will be write in ASCII mode
                  if not, in binary mode. [Default, ASCII=True]
          path = str, path where you want to export your new .vtu file.
                 [Default: path='./']
    """
    # =========================================================================
    #  Write VTK file (unstructured mesh)
    # =========================================================================
    # Definitions and initialization
    sizeof_Float32  =   4
    sizeof_Float64  =   4
    sizeof_UInt32   =   4
    sizeof_UInt8    =   1
    Offset          =   0      # Initial offset
    # =======================================
    # Write the header for a structured grid:
    fname_vtk = fname+'.vtu'
    fid       = open(path+fname_vtk,'w')
    
    fid.write('<?xml version="1.0"?> \n')
    fid.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian" >\n')
    fid.write('  <UnstructuredGrid>\n')
    fid.write('    <Piece NumberOfPoints="%s"  NumberOfCells="%s">\n' % \
              (str(np.int32(Points.shape[0])), str(len(np.int32(ElementNumbers)))))
    # =======================================
    # Add point-wise data
    #Adapt the exporting file according to the nature of the field (Scalar/Vectorial)
    fid.write('    <PointData Scalars="T" Vectors="Velocity"  >\n')
    # ===================
    if ASCII:
        fid.write('      <DataArray type="Float32" Name="%s" format="ascii">\n' % fieldName)
        for i in range(len(vstack)):
            fid.write('        %s \n' % vstack[i])
    else:
        fid.write('      <DataArray type="Float32" Name="%s" format="appended" offset="%s">\n' %\
                  (fieldName,np.int32(Offset)))
        Offset = Offset + len(vstack)*sizeof_Float32 + 1*sizeof_UInt32
    fid.write('      </DataArray>\n')
    # =======================================
    # Add coordinates of structured grid
    fid.write('    </PointData>\n')
    fid.write('    <Points>\n')
    if ASCII:
        fid.write('      <DataArray type="Float32" Name="Array" NumberOfComponents="3" format="ascii">\n')
        for i in range(len(Points)):
            fid.write('         %s %s %s \n' % (Points[i][0],Points[i][1],Points[i][2]))
    else:
         fid.write('      <DataArray type="Float32" Name="Array" NumberOfComponents="3" format="appended" offset="%s" >\n' % \
                   np.int32(Offset))
         Offset = Offset + len(Points)*sizeof_Float32 + 1*sizeof_UInt32
    fid.write('        </DataArray>\n')
    fid.write('    </Points>\n')
    # =======================================
    # Add CELLS data
    fid.write('    <Cells>\n')    
    # - Connectivity -----------
    if ASCII:
        fid.write('      <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i in range(len(ElementNumbers)):
            fid.write('        %s %s %s %s %s %s \n' % (np.int32(ElementNumbers[i][0]), \
                                                        np.int32(ElementNumbers[i][1]), \
                                                        np.int32(ElementNumbers[i][2]), \
                                                        np.int32(ElementNumbers[i][3]), \
                                                        np.int32(ElementNumbers[i][4]), \
                                                        np.int32(ElementNumbers[i][5])))     
    else:
       fid.write('      <DataArray type="Int32" Name="connectivity" format="appended" offset="%s">\n' % np.int32(Offset))
       Offset = Offset + len(ElementNumbers)*sizeof_UInt32 + 1*sizeof_UInt32
    fid.write('      </DataArray>\n')
    # - Offsets -----------
    offsets = np.cumsum(np.ones(len(ElementNumbers))*6)
    if ASCII:
        fid.write('  <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(len(ElementNumbers)):
            fid.write('        %s\n' % np.int32(offsets[i]))
    else:
        fid.write('      <DataArray type="Int32" Name="offsets" format="appended" offset="%s">\n' % np.int32(Offset))
        Offset = Offset + len(offsets)*sizeof_UInt32 + 1*sizeof_UInt32
    fid.write('      </DataArray>\n')
    # - Types -----------
    types = np.ones(len(ElementNumbers))*13
    if ASCII:
        fid.write('      <DataArray type="UInt8" Name="types" format="ascii">\n')
        for i in range(len(ElementNumbers)):
            fid.write('        %s\n' % np.uint8(13))
    else:
        fid.write('      <DataArray type="UInt8" Name="types" format="appended" offset="%s">\n' % np.int32(Offset))
        Offset = Offset + len(types)*sizeof_UInt8 + 1*sizeof_UInt32;
    fid.write('      </DataArray>\n')
    # =======================================
    # Close .vtu
    fid.write('    </Cells>\n')
    #   !!!! Include here routine for binary export !!!!
    fid.write('    </Piece>\n')
    fid.write('  </UnstructuredGrid>\n')
    fid.write('</VTKFile>\n')
    fid.close()





def stack_XDMF(directory,xdmf_fnames):
    """
    This function stacks a list a .xdmf files contained together
    in the same directory, which is also the directory where the
    corresponding .h5 is contained.
    Warning: The grid have to be the same between files.
    """
    nof = len(xdmf_fnames)
    # -------------------------
    # Read the first file to extract the header, the geometry
    # and the file's tail
    header = []     # will contain the header lines
    tail   = []     # will contain the tail of the file
    geom   = []     # will contain the lines corresponding to the geometry of the grid
    temp = []
    with open(directory+xdmf_fnames[0],'r') as data:
        for line in data:
            temp.append(line)
    return temp







def triangulationPlanar(Nx,Ny,ordering='xy'):
    """
    Function of planar triangulation based on indicies of point on the square grid.
    This function returns simplicies (i.e. list of tuples of three elements
    corresponding to index of point on the grid that must be gather to form
    a triangle).
    Example of the function on the following grid: Nx=4, Ny=3
        0---1---2---3             0---1---2---3
        |   |   |   |             | \ | \ | \ |
        4---5---6---7     ==>     4---5---6---7
        |   |   |   |             | \ | \ | \ |
        8---9---10--11            8---9---10--11
    
    triangulationPlanar(Nx,Ny) = [[0,4,5],     for an 'yx' ordering
                                    [0,1,5],
                                    [1,5,6],
                                    [1,2,6],
                                    [2,6,7],
                                    [2,3,7],
                                    [4,8,9],
                                    ...
                                    [6,11,7]]
    <i> : Nx = int, number of grid points on the x-axis
            Ny = int, number of grid points on the y-axis
            ordering = str, optional (by default 'xy'), value in ('xy','yx','ij')
                        corresponds to the ordering of how triangulation
                        is generated.
    <y> : hull = list, simplicies
    """
    hull = np.zeros(((Nx-1)*(Ny-1)*2,3),dtype=np.int32)
    if ordering == 'yx':
        for i in range(Ny-1):
            for j in range(Nx-1):
                hull[2*i*(Nx-1)+2*j]   = np.array([i*Nx+j,  (i+1)*Nx+j,  (i+1)*Nx+j+1])
                hull[2*i*(Nx-1)+2*j+1] = np.array([i*Nx+j,  i*Nx+j+1,    (i+1)*Nx+j+1])
    elif ordering == 'xy':
        for i in range(Nx-1):
            for j in range(Ny-1):
                hull[2*i*(Nx-1)+2*j]   = np.array([j*Nx+i,  j*Nx+i+1,   (j+1)*Nx+i+1])
                hull[2*i*(Nx-1)+2*j+1] = np.array([j*Nx+i,  (j+1)*Nx+i, (j+1)*Nx+i+1])
    elif ordering == 'ij':
        for i in range(Ny-1):
            hull = []
            #Depends on the parity of y
            if i%2 == 0:
                L1 = list(range(i*Nx,(i+1)*Nx))
                L2 = sorted(list(range((i+1)*Nx,(i+2)*Nx)),reverse=True)
            else:
                L1 = sorted(list(range(i*Nx,(i+1)*Nx)),reverse=True)
                L2 = list(range((i+1)*Nx,(i+2)*Nx))
            for j in range(Nx-1):
                hull.append([ L1[j], L1[j+1], L2[j+1] ])
                hull.append([ L1[j], L2[j],   L2[j+1] ])
    else:
        print('ERROR: error in the ordering of the triangulation')
        pass
    return np.array(hull)



















