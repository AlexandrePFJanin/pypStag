# -*- coding: utf-8 -*-
"""
@author: Alexandre Janin
@aim:    pypStag Visualisation ToolKit
"""

# External dependencies:
import numpy as np
import h5py
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

# Internal dependencies:
from .errors import *
from .generics import im


# ----------------- FUNCTIONS -----------------


def stag2VTU(fname,stagData,path='./',ASCII=False,verbose=True,return_only=False,creat_pointID=False,\
             vect_topo_name='Cartesian Velocity',vect_geo_name='Spherical Velocity'):
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
    is_possibly_3D = False  # is it possible that the mesh is 3D?
    if stagData.geometry in ['cart3D','spherical','yy']:
        is_possibly_3D = True
    #
    if stagData.slayers.shape[0] > 1 and is_possibly_3D:
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
            Nx = stagData.nx
            Ny = stagData.ny
            Nz = stagData.nz
            NxNy = Nx*Ny
            x = stagData.x.reshape(NxNy*Nz, order='F')
            y = stagData.y.reshape(NxNy*Nz, order='F')
            z = stagData.z.reshape(NxNy*Nz, order='F')
            v = stagData.v.reshape(NxNy*Nz, order='F')
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
            NUM         = np.arange(0,NxNy*Nz)
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
            im('    - Convert data into a correct matrix format',pName,verbose)
            im('      - Grid',pName,verbose)
            Points = np.zeros((NxNy*Nz,3))
            Points[:,0] = x#x.reshape((NxNy*Nz), order='F')
            Points[:,1] = y#y.reshape((NxNy*Nz), order='F')
            Points[:,2] = z#z.reshape((NxNy*Nz), order='F')
            # ===================
            im('      - Field',pName,verbose)
            if stagData.scalar:
                V_yy  = v.reshape(NxNy,Nz, order='F')
                vstack = V_yy.reshape((NxNy*Nz), order='F')
            # ===================
            else:
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
            NxNy = int(len(stagData.layers)/Nz)   #Number of points for each layers
            if not stagData.yin.allocated or not stagData.yang.allocated:
                # split both the mesh and the fields to rebuild the partial yy structures
                stagData.splitGrid()
                stagData.splitFields()
            x1     = stagData.yin.x.reshape(NxNy,Nz)
            x2     = stagData.yang.x.reshape(NxNy,Nz)
            y1     = stagData.yin.y.reshape(NxNy,Nz)
            y2     = stagData.yang.y.reshape(NxNy,Nz)
            z1     = stagData.yin.z.reshape(NxNy,Nz)
            z2     = stagData.yang.z.reshape(NxNy,Nz)
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
            if stagData.scalar:
                V_yin  = np.array(stagData.yin.v).reshape(NxNy,Nz)
                V_yang = np.array(stagData.yang.v).reshape(NxNy,Nz)
                vstack = np.zeros(2*NxNy*Nz)
                vstack[0:NxNy*Nz]         = V_yin.reshape((NxNy*Nz), order='F')
                vstack[NxNy*Nz:2*NxNy*Nz] = V_yang.reshape((NxNy*Nz),order='F')
            # ===================
            else:
                # ------ Vx ------
                V_yinx  = np.array(stagData.yin.vx).reshape(NxNy,Nz)
                V_yangx = np.array(stagData.yang.vx).reshape(NxNy,Nz)
                vstackx = np.zeros(2*NxNy*Nz)
                vstackx[0:NxNy*Nz]         = V_yinx.reshape((NxNy*Nz), order='F')
                vstackx[NxNy*Nz:2*NxNy*Nz] = V_yangx.reshape((NxNy*Nz),order='F')
                # ------ Vy ------
                V_yiny  = np.array(stagData.yin.vy).reshape(NxNy,Nz)
                V_yangy = np.array(stagData.yang.vy).reshape(NxNy,Nz)
                vstacky = np.zeros(2*NxNy*Nz)
                vstacky[0:NxNy*Nz]         = V_yiny.reshape((NxNy*Nz), order='F')
                vstacky[NxNy*Nz:2*NxNy*Nz] = V_yangy.reshape((NxNy*Nz),order='F')
                # ------ Vz ------
                V_yinz  = np.array(stagData.yin.vz).reshape(NxNy,Nz)
                V_yangz = np.array(stagData.yang.vz).reshape(NxNy,Nz)
                vstackz = np.zeros(2*NxNy*Nz)
                vstackz[0:NxNy*Nz]         = V_yinz.reshape((NxNy*Nz), order='F')
                vstackz[NxNy*Nz:2*NxNy*Nz] = V_yangz.reshape((NxNy*Nz),order='F')
                # ------ Vr ------
                V_yinr  = np.array(stagData.yin.vr).reshape(NxNy,Nz)
                V_yangr = np.array(stagData.yang.vr).reshape(NxNy,Nz)
                vstackr = np.zeros(2*NxNy*Nz)
                vstackr[0:NxNy*Nz]         = V_yinr.reshape((NxNy*Nz), order='F')
                vstackr[NxNy*Nz:2*NxNy*Nz] = V_yangr.reshape((NxNy*Nz),order='F')
                # ------ Vtheta ------
                V_yintheta  = np.array(stagData.yin.vtheta).reshape(NxNy,Nz)
                V_yangtheta = np.array(stagData.yang.vtheta).reshape(NxNy,Nz)
                vstacktheta = np.zeros(2*NxNy*Nz)
                vstacktheta[0:NxNy*Nz]         = V_yintheta.reshape((NxNy*Nz), order='F')
                vstacktheta[NxNy*Nz:2*NxNy*Nz] = V_yangtheta.reshape((NxNy*Nz),order='F')
                # ------ Vphi ------
                V_yinphi  = np.array(stagData.yin.vphi).reshape(NxNy,Nz)
                V_yangphi = np.array(stagData.yang.vphi).reshape(NxNy,Nz)
                vstackphi = np.zeros(2*NxNy*Nz)
                vstackphi[0:NxNy*Nz]         = V_yinphi.reshape((NxNy*Nz), order='F')
                vstackphi[NxNy*Nz:2*NxNy*Nz] = V_yangphi.reshape((NxNy*Nz),order='F')
                # ------ P ------
                V_yinp  = np.array(stagData.yin.P).reshape(NxNy,Nz)
                V_yangp = np.array(stagData.yang.P).reshape(NxNy,Nz)
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
            __writeVKTStag(fname,stagData,Points,ElementNumbers,vstack,ASCII=ASCII,path=path,pointID=pointID,\
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
            if not stagData.scalar:
                VX = stagData.vx.flatten()
                VY = stagData.vy.flatten()
                VZ = stagData.vz.flatten()
            if stagData.pressure:
                PRESSURE = stagData.P.flatten()
        # =======================================
        elif stagData.geometry == 'yy':
            write_slice = True
            X = stagData.x
            Y = stagData.y
            Z = stagData.z
            V = stagData.v
            if not stagData.scalar:
                VX = stagData.vx
                VY = stagData.vy
                VZ = stagData.vz
                VR = stagData.vr
                VTHETA = stagData.vtheta
                VPHI = stagData.vphi
            if stagData.pressure:
                PRESSURE = stagData.P
        # =======================================
        elif stagData.geometry == 'cart2D':
            write_slice = True
            if not stagData.reorganized:
                # verbose error
                if stagData.plan == 'xy':
                    txt = 'z'
                elif stagData.plan == 'xz':
                    txt = 'y'
                else:
                    txt = 'x'
                raise ExpectedReorganized2DError('Not supported by ConvexHull or Delaunay. QhullError: qhull input error: input is less than 3-dimensional since all points have the same %s coordinate'%txt)
            else:
                X = stagData.x.flatten()
                Y = stagData.y.flatten()
                Z = np.zeros(X.shape)
                V = stagData.v.flatten()
                if not stagData.scalar:
                    VX = stagData.vx.flatten()
                    VY = stagData.vy.flatten()
                    VZ = np.zeros(X.shape)
            if stagData.pressure:
                PRESSURE = stagData.P
        # =======================================
        elif stagData.geometry == 'annulus':
            write_slice = True
            if not stagData.reorganized:
                X = stagData.x.flatten()
                Y = stagData.y.flatten()
                Z = np.zeros(X.shape)
                V = stagData.v.flatten()
                if not stagData.scalar:
                    VX = stagData.vx.flatten()
                    VY = stagData.vy.flatten()
                    VZ = stagData.vz.flatten()
                    VR = stagData.vr.flatten()
                    VTHETA = stagData.vtheta.flatten()
                    VPHI = stagData.vphi.flatten()
            else:
                X = stagData.x.flatten()
                Y = stagData.y.flatten()
                Z = np.zeros(X.shape)
                V = stagData.v.flatten()
                if not stagData.scalar:
                    VX = stagData.vx.flatten()
                    VY = stagData.vy.flatten()
                    VZ = np.zeros(X.shape)
                    VR = stagData.vr.flatten()
                    VTHETA = np.zeros(X.shape)
                    VPHI = stagData.vphi.flatten()
            if stagData.pressure:
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
            if stagData.geometry in ['cart3D','spherical']:
                im('  -> Delaunay Triangulation',pName,verbose)
                tri = Delaunay(points[:,0:2])
                simplices = tri.simplices
            elif stagData.geometry in ['yy']:
                im('  -> Convex Hull Triangulation',pName,verbose)
                tri = ConvexHull(points)
                simplices = tri.simplices
            elif stagData.geometry in ['cart2D']:
                im('  -> Planar Triangulation',pName,verbose)
                simplices = triangulationPlanar(stagData.ny,stagData.nx,ordering='yx',cyclic=False)
            elif stagData.geometry in ['annulus']:
                if stagData.plan == 'yz':
                    im('  -> Planar Triangulation for an equatorial slice',pName,verbose)
                    simplices = triangulationPlanar(stagData.ny,stagData.nx,ordering='yx',cyclic=True)
                elif stagData.plan == 'xz':
                    im('  -> Planar Triangulation for a spherical axi-symmetric slice',pName,verbose)
                    simplices = triangulationPlanar(stagData.ny,stagData.nx,ordering='yx',cyclic=False)
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
            if stagData.scalar:
                fid.write('      <Attribute Name="'+stagData.fieldName+'" Active="1" AttributeType="Scalar" Center="Node">'+'\n')
                fid.write('        <DataItem Dimensions="%s" NumberType="Float" Precision="4" Format="HDF">'%nod+h5_file+':/Node/field_scalar</DataItem>'+'\n')
                fid.write('      </Attribute>'+'\n')
            else:
                # ---- Cartesian Velocities ----
                fid.write('      <Attribute AttributeType="Vector" Center="Node" Name="%s">\n' % vect_topo_name)
                fid.write('        <DataItem DataType="Float" Dimensions="%s %s" Format="HDF" Precision="8">\n'%(nod, 3))
                fid.write('            '+h5_file+':/Node/field_cart\n')
                fid.write('        </DataItem>\n')
                fid.write('      </Attribute>\n\n')
                if stagData.geometry in ['yy','annulus','spherical']:
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
            if stagData.scalar:
                dset = fid.create_dataset('/Node/field_scalar', data = V)
            else:
                dset = fid.create_dataset('/Node/field_cart', data = np.array([VX,VY,VZ]).T)
                if stagData.geometry in ['yy','annulus','spherical']:
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
          vstack = list OR tuple of lists, if stagData.scalar,
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
        if stagData.scalar:
            fid.write('      <DataArray type="Float32" Name="%s" format="ascii">\n' % stagData.fieldName)
            for i in range(len(vstack)):
                fid.write('        %s \n' % vstack[i])
            fid.write('      </DataArray>\n')
        # ===================
        else:
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
            if stagData.geometry in ['yy','annulus','spherical']:
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
        if stagData.scalar:
            Data       = np.asarray(vstack)
        else:
            if stagData.geometry == 'cart3D':
                (vstackx,vstacky,vstackz,vstackp) = (vstack[0],vstack[1],vstack[2],vstack[3])
                Datax  = np.asarray(vstackx)
                Datay  = np.asarray(vstacky)
                Dataz  = np.asarray(vstackz)
                Datap  = np.asarray(vstackp)
            elif stagData.geometry in ['yy','annulus','spherical']:
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
        if stagData.scalar:
            fid.write('    <Attribute AttributeType="Scalar" Center="Node" Name="%s">\n' % stagData.fieldName)
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
            if stagData.geometry in ['yy','annulus','spherical']:
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
        fid = h5py.File(path+fname_h5, 'w')
        dset = fid.create_dataset("Points", data=Points, dtype=np.float32)
        dset = fid.create_dataset("NumberOfElements", data=ElementNumbers, dtype=np.int32)
        if stagData.scalar:
            dset = fid.create_dataset("Data0", data=Data, dtype=np.float32)
        else:
            if stagData.geometry == 'cart3D':
                Data = np.array([Datax,Datay,Dataz]).T
                dset = fid.create_dataset("Data1", data=Data, dtype=np.float32)
            elif stagData.geometry in ['yy','annulus','spherical']:
                Data = np.array([Datax,Datay,Dataz]).T
                dset = fid.create_dataset("Data1", data=Data, dtype=np.float32)
                Data = np.array([Datar,Datath,Dataph]).T
                dset = fid.create_dataset("Data2", data=Data, dtype=np.float32)
            Data = Datap
            dset = fid.create_dataset("Data3", data=Data, dtype=np.float32)
        if pointID is not None:
            dset = fid.create_dataset("pointID", data=pointID, dtype=np.int32)
        fid.close()



def triangulationPlanar(Nx,Ny,ordering='xy',cyclic=False):
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
    Args
        Nx (int): number of grid points on the x-axis
        Ny (int): number of grid points on the y-axis
        ordering (str, optional): value in ('xy','yx','ij')
                corresponds to the ordering of how triangulation
                is generated. Defaults, ordering = 'xy'
        cyclic (bool, optional): is the mesh cyclic (annulus).
                Close the mesh on the annulus. Works only if the
                ordering method is set to 'yx'. Otherwise, raises
                an error.    
                Defaults, cyclic = False
    Return:
        hull (np.ndarray): simplicies
    """
    if not cyclic:
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
            hull = np.array(hull)
        else:
            print('ERROR: error in the ordering of the triangulation')
            raise AssertionError()
    else:
        if ordering == 'yx':
            hull = np.zeros(((Nx-1)*(Ny)*2,3),dtype=np.int32)
        if ordering == 'yx':
            k = 0 # number of closed circles
            for i in range(Ny-1):
                for j in range(Nx-1):
                    hull[2*i*(Nx-1)+2*j]   = np.array([i*Nx+j,  (i+1)*Nx+j,  (i+1)*Nx+j+1])
                    hull[2*i*(Nx-1)+2*j+1] = np.array([i*Nx+j,  i*Nx+j+1,    (i+1)*Nx+j+1])
            # close the annulus
            i += 1
            for j in range(Nx-1):
                hull[2*i*(Nx-1)+2*j]   = np.array([i*Nx+j,  0*Nx+j,    0*Nx+j+1])
                hull[2*i*(Nx-1)+2*j+1] = np.array([i*Nx+j,  i*Nx+j+1,  0*Nx+j+1])
        else:
            print('ERROR: ordering not compatible with the cyclic mode')
            raise AssertionError()
    return hull



















