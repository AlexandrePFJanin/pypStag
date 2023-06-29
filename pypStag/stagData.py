# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:46:12 2019
Last edition on Tue Nov 17 14:43:37 2020

@author: Alexandre Janin
@Aim: Universal objects for the treatment of Stag ouptuts
"""



from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .stagReader import fields, reader_time, reader_rprof, reader_plates_analyse
from .stagComputeMod import velocity_pole_projecton, ecef2enu_stagYY, rotation_matrix_3D, \
                            xyz2latlon
from .stagError import NoFileError, InputGridGeometryError, GridGeometryError, fieldTypeError, \
                       MetaCheckFieldUnknownError, MetaFileInappropriateError, FieldTypeInDevError, \
                       VisuGridGeometryError, StagTypeError, CloudBuildIndexError, SliceAxisError, \
                       IncoherentSliceAxisError, StagUnknownLayerError, StagComputationalError,\
                       GridGeometryIncompatibleError, StagBaseError, fieldNatureError





class MainStagObject:
    """
    Main class defining the highest level of inheritance
    for StagData derived object
    """
    def __init__(self):
        """
        Parent builder
        """
        # ----- Generic ----- #
        self.pName = 'stagData'
        self.verbose = True       #Condition on the verbose output
        self.fieldType = 'Temperature'  #Field contained in the current object
        self.fieldNature = 'Scalar'     #Nature of the field: Scalar or Vectorial
        self.path  = ''                 #The path to the stag file
        self.fname = ''                 #File name of the stag file
        self.resampling = []#Resampling Parameters
        self.header = []    #Raw header of stag file
        self.simuAge = 0    #Dimensionless age of the simulation
        self.ti_step = 0    #Inner step of the stag simualtion state
        self.flds = []      #Raw fields of stag file
        self.x_coords = []  #x matrix in the header (modified by the resampling)
        self.y_coords = []  #y matrix in the header (modified by the resampling)
        self.z_coords = []  #z matrix in the header (modified by the resampling)
        self.ntb = 0        #number of blocks, 2 for yinyang or cubed sphere, 1 for others
        self.slayers = []   #matrix of selected layers (same size as z_coord)
        self.depths = []    #matrix of depths in real Earth for each layers
        self.rcmb = 0       #Radius of the Core-Mantle Boundary
        self.xind = []      #Mask: List of index keep in x (follows good index)
        self.yind = []      #Mask: List of index keep in y (follows good index)
        self.zind = []      #Mask: List of index keep in z (follows good index)
        self.nx0 = 0        #Number of points in the x direction in the original input file
        self.ny0 = 0        #Number of points in the y direction in the original input file
        self.nz0 = 0        #Number of points in the z direction in the original input file
        self.nx  = 0        #Current number of points in the x direction (after resampling)
        self.ny  = 0        #Current number of points in the y direction (after resampling)
        self.nz  = 0        #Current number of points in the z direction (after resampling)
        # Other
        self.BIN = None
        self.bin = None
    

    def im(self,textMessage):
        """Print verbose internal message. This function depends on the
        argument of self.verbose. If self.verbose == True then the message
        will be displayed on the terminal.
        <i> : textMessage = str, message to display
        """
        if self.verbose == True:
            print('>> '+self.pName+'| '+textMessage)


    def stagImport(self, directory, fname, beginIndex=-1, endIndex=-1, resampling=[1,1,1]):
        """ This function reads a stag data file using the modul stagReader.fields
        and fill the appropriated fields of the current StagData object.
        <i> : directory = str, path to reach the data file
              fname = str, name of the data file
              beginIndex = int, deepest index for considered layers. If beginIndex=-1, the
                           deepest index is 0, the deepest produced by stag.
                           (Default: beginIndex = -1)
              endIndex = int, shallowest index for considered layers. If endIndex=-1, the
                         shallowest index is the shallowest index produced by stag.
                         (Default: endIndex = -1)
              resampling = list/array/tuple, matrix of dim 3 containing the 
                           resampling parameters (int) on X, Y and Z axis as:
                           resampling = [resampling_on_X,resampling_on_Y,resampling_on_Z]
                           (Default: resampling=[1,1,1], means no resampling)
              """
        self.im('Reading and resampling: '+fname)
        # - Autocompletion of the path
        if directory[-1] != '/':
            directory += '/'
        self.path  = Path(directory+fname) #creat a Path object
        self.fname = fname
        self.resampling = resampling
        # - First, test the geometry:
        if self.geometry not in ('cart2D','cart3D','yy','spherical','annulus'):
            raise InputGridGeometryError(self.geometry)
        # - Read Stag binary files:
        try:
            (self.header,self.flds) = fields(self.path)
        except:
            raise NoFileError(directory,fname)
        #Strcuture for 'flds' variables:
        #  [Var][x-direction][y_direction][z-direction][block_index]
        
        self.x_coords = self.header.get('e1_coord')
        self.y_coords = self.header.get('e2_coord')
        self.z_coords = self.header.get('e3_coord')

        # number of blocks, 2 for yinyang or cubed sphere
        self.ntb = self.header.get('ntb')

        #Conditioning the 2D/3D geometry problem:
        if type(self.x_coords) != np.ndarray:
            self.x_coords = np.array([self.x_coords])
            self.im('  - 2D data detected: plan yz')
            if self.geometry != 'cart2D' and self.geometry != 'annulus':
                raise GridGeometryError(self.geometry,'cart2D or annulus')
        elif type(self.y_coords) != np.ndarray:
            self.y_coords = np.array([self.y_coords])
            self.im('  - 2D data detected: plan xz')
            if self.geometry != 'cart2D':
                raise GridGeometryError(self.geometry,'cart2D')
        elif type(self.z_coords) != np.ndarray:
            self.z_coords = np.array([self.z_coords])
            self.im('  - 2D data detected: plan xy')
            if self.geometry != 'cart2D':
                raise GridGeometryError(self.geometry,'cart2D')
        else:
            self.im('  - 3D data detected')
            if self.geometry != 'cart3D' and self.geometry != 'yy' and self.geometry != 'spherical':
                raise GridGeometryError(self.geometry,'cart3D, yy or spherical')
            if self.ntb == 1:
                #Grid geometry compatible with cart3D or spherical
                self.im('    -> Grid geometry compatible with cart3D or spherical')
                if self.geometry == 'yy':
                    raise GridGeometryError(self.geometry,'cart3D or spherical')
            elif self.ntb == 2:
                self.im('    -> YinYang grid detected')
                if self.geometry == 'cart3D' or self.geometry == 'spherical':
                    raise GridGeometryError(self.geometry,'yy')

        self.nx0 = len(self.x_coords)
        self.ny0 = len(self.y_coords)
        self.nz0 = len(self.z_coords)
        self.nx  = len(self.x_coords)
        self.ny  = len(self.y_coords)
        self.nz  = len(self.z_coords)
        self.im("  - Original grid geometry:")
        self.im("    - Nx = "+str(self.nx0))
        self.im("    - Ny = "+str(self.ny0))
        self.im("    - Nz = "+str(self.nz0))

        Nz = len(self.header.get('e3_coord'))

        #attribution values of default parameters
        if beginIndex == -1:
            beginIndex = 0
        if endIndex == -1:
            endIndex = self.nz

        self.slayers = np.linspace(1,self.nz,self.nz)
        self.rcmb = self.header.get('rcmb')
        self.simuAge = self.header.get('ti_ad')
        self.ti_step = self.header.get('ti_step')
        
        def resampling_coord(coords,sampling):
            """This function resamples coords matrix according to the sampling rate.
            sampling must be an integer. If sampling ==1 the output is the input.
            If sampling == 2, the output is twince smaller than the input.
            Return the new matrix after the resampling and the matrix of elements
            (index) that have been keep (1) and remove (0)
            """
            n = 0
            new_coords = []
            ind = []
            while n < len(coords):
                new_coords.append(coords[n])
                ind.append(n)
                n += sampling
            if new_coords[len(new_coords)-1] != coords[len(coords)-1] and len(coords)>1:#+sampling and len(coords)>1:
                #garanty to have the firt AND the last value of coords: garanty to
                #conserve the input shape
                new_coords.append(coords[len(coords)-1])
                ind.append(len(coords)-1)
            index = []
            for i in range(len(coords)):
                if i in ind:
                    index.append(1)
                else:
                    index.append(0)
            return (np.array(new_coords), np.array(index)) #conserve the array-type
        
        (self.x_coords, self.xind) = resampling_coord(self.x_coords,resampling[0])
        (self.y_coords, self.yind) = resampling_coord(self.y_coords,resampling[1])
        (self.z_coords, self.zind) = resampling_coord(self.z_coords,resampling[2])
        #(self.slayers, self.BIN)  = resampling_coord(self.slayers, resampling[2])

        ## Re-mapping of the zind matrix according to the range of depth considered
        #  -1- Work on indexes:
        zindtemp = np.zeros(Nz)
        for i in range(Nz):
            if i>=beginIndex and i <endIndex:
                zindtemp[i] = 1
        multi = np.multiply(self.zind,zindtemp)
        if np.count_nonzero(multi) == 0:
            self.zind = zindtemp
        else:
            self.zind = multi
        # -2- Work on coordinates matrix
        indexNewZCoord = np.where(self.zind == 1)[0]
        ztemp = self.header.get('e3_coord')
        new_z_coords = []
        new_slayers   = []
        for ind in indexNewZCoord:
            new_z_coords.append(ztemp[ind])
            new_slayers.append(self.slayers[ind])    #Follows self.z_coord
        self.z_coords = new_z_coords
        self.slayers = np.array(new_slayers)
        
        #Update the geometrical variable defining the grid
        self.nx  = len(self.x_coords)
        self.ny  = len(self.y_coords)
        self.nz  = len(self.z_coords)
        self.im("  - New grid geometry:")
        self.im("    - Nx = "+str(self.nx))
        self.im("    - Ny = "+str(self.ny))
        self.im("    - Nz = "+str(self.nz))
        
        #Compute depths:
        dCMB = 2890 #depth CMB (km)
        self.depths = [(1-self.z_coords[i])*dCMB for i in range(self.nz)]
        self.depths = np.array(sorted(self.depths,reverse=True)) #sorted as self.z_coord
        
        #What type of data is reading ?
        fname = fname.split('_')[-1]
        n = [fname[i] for i in range(len(fname))]
        if ''.join(n[0:3]) == 'div':
            self.fieldType = 'Divergence'
        elif ''.join(n[0:3]) == 'eta':
            self.fieldType = 'Viscosity'
        elif n[0] == 't':
            self.fieldType = 'Temperature'
        elif ''.join(n[0:2]) == 'vp':
            self.fieldType = 'Velocity'
        elif ''.join(n[0:4]) == 'smax':
            self.fieldType = 'Sigma max'
        elif ''.join(n[0:3]) == 'dam':
            self.fieldType = 'Damage'
        elif ''.join(n[0:2]) == 'cs':
            self.fieldType = 'Topography'
        elif ''.join(n[0:3]) == 'rho':
            self.fieldType = 'Density'
        elif ''.join(n[0:2]) == 'ly':
            self.fieldType = 'Lyapunov'
        elif ''.join(n[0:3]) == 'div':
            self.fieldType = 'Divergence'
        elif ''.join(n[0:3]) == 'vor':
            self.fieldType = 'Vorticity'
        elif ''.join(n[0:3]) == 'str':
            self.fieldType = 'Stress'
        elif ''.join(n[0:2]) == 'po':
            self.fieldType = 'Poloidal'
        elif ''.join(n[0:2]) == 'to':
            self.fieldType = 'Toroidal'
        elif ''.join(n[0:2]) == 'ed':
            self.fieldType = 'Strain Rate'
        elif ''.join(n[0:1]) == 'c':
            self.fieldType = 'Composition'
        elif ''.join(n[0:1]) == 'f':
            self.fieldType = 'Melt Fraction'
        elif ''.join(n[0:2]) == 'vm':
            self.fieldType = 'Melt Velocity'
        elif ''.join(n[0:3]) == 'age':
            self.fieldType = 'Age'
        elif ''.join(n[0:3]) == 'nrc':
            self.fieldType = 'Continents'
        elif ''.join(n[0:1]) == 'w':
            self.fieldType = 'Vorticity'
        elif ''.join(n[0:4]) == 'prot':
            self.fieldType = 'Prot'
        elif ''.join(n[0:3]) == 'prm':
            self.fieldType = 'Prm'
        elif ''.join(n[0:4]) == 'defm':
            self.fieldType = 'Deformation Mode'
        elif ''.join(n[0:2]) == 'gs':
            self.fieldType = 'Grain Size'
        elif ''.join(n[0:1]) == 'h':    ############## TEMPORARY ADDED A.JANIN 01.06.21 #################
            self.fieldType = 'Healing'
        elif ''.join(n[0:2]) == 'sy':   ############## TEMPORARY ADDED A.JANIN 01.06.21 #################
            self.fieldType = 'Yield Stress'
        else:
            self.fieldType = 'Error: Undetermined'
            raise FieldTypeInDevError(fname)
        if self.flds.shape[0] == 1:
            self.im('  - Scalar field detected')
            self.fieldNature = 'Scalar'
        else:
            self.im('  - Vectorial field detected: '+str(self.flds.shape[0])+' fields')
            self.fieldNature  ='Vectorial'
        self.im('    -> '+self.fieldType)
        self.im('Reading and resampling operations done!')
    

    def stag2VTU(self,fname=None,path='./',ASCII=False,return_only=False,creat_pointID=False,verbose=True):
            """ Extension of the stagVTK package, directly available on stagData !
            This function creat '.vtu' or 'xdmf/h5' file readable with Paraview to efficiently 
            visualize 3D data contain in a stagData object. This function works directly
            on non overlapping stagData object.
            Note also that the internal field stagData.slayers of the stagData object
            must be filled.
            <i> : fname = str, name of the exported file without any extention
                path = str, path where you want to export your new .vtu file.
                        [Default: path='./']
                ASCII = bool, if True, the .vtu file will be write in ASCII mode
                        if not, in binary mode. [Default, ASCII=True]
                creat_pointID = bool, if True then creat the list of points ID sorted as
                          it is in the field stagData.x (Yin and Yang together).
                          This field will then transfert to the writing function and
                          the .h5/.xdmf will have an extra field corresponding to 
                          these points ID (e.g. very usefull if post processing with TTK)
                          WARNING: This option is only available if ASCII = False
                                   (ie .h5/.xdmf output)
            """
            self.im('Requested: Build VTU from StagData object')
            if self.geometry == 'cart2D' or self.geometry == 'annulus':
                raise VisuGridGeometryError(self.geometry,'cart3D or yy or spherical')
            if fname == None:
                import time
                (y,m,d,h,mins,secs,bin1,bin2,bin3) = time.localtime()
                fname = self.fname+'_'+str(d)+'-'+str(m)+'-'+str(y)+'_'+str(h)+'-'+str(mins)+'-'+str(secs)
                self.im('Automatic file name attribution: '+fname)
            #Importation of the stagVTK package
            from .stagVTK import stag2VTU
            if not return_only:
                stag2VTU(fname,self,path,ASCII=ASCII,creat_pointID=creat_pointID,return_only=return_only,verbose=verbose)
            else:
                Points,ElementNumbers,vstack,pointID = stag2VTU(fname,self,path,ASCII=ASCII,creat_pointID=creat_pointID,return_only=return_only,verbose=verbose)
                return Points,ElementNumbers,vstack,pointID
    
    







class StagCartesianGeometry(MainStagObject):
    """
    Defines the StagCartesianGeometry object, derived from MainStagObject
    This object is conditionally inherited in StagData.
    """
    def __init__(self,geometry):
        super().__init__()  # inherit all the methods and properties from MainStagObject
        self.geometry = geometry
        self.plan     = None
        # ----- Cartesian 2D and 3D geometries ----- #
        self.XYZind = []    #Matrix of good index after the mesh operation
        self.x = []         #Matrix of X coordinates meshed
        self.y = []         #Matrix of Y coordinates meshed
        self.z = []         #Matrix of Z coordinates meshed
        self.v = []         #Matrix of scalar field (or norm of velocity)
        self.vx = []        #Matrix of x-component of the velocity field for Cartesian grids
        self.vy = []        #Matrix of y-component of the velocity field for Cartesian grids
        self.vz = []        #Matrix of z-component of the velocity field for Cartesian grids
        self.P  = []        #Matrix of Pressure field for Cartesian grids
    
    def stagProcessing(self):
        """
        This function processes stag data according to a Cartesian geometry.
        """
        self.im('Processing stag Data:')
        self.im('  - Grid Geometry')
        # Meshing
        (self.x,self.y,self.z) = np.meshgrid(self.x_coords,self.y_coords,self.z_coords,indexing='ij')
        # Geometry
        if self.geometry == 'cart2D':
            self.im('      - 2D cartesian grid geometry')
            if  self.x.shape[0] == 1:
                self.im('      - data detected: plan yz')
                self.plan = 'yz'
            elif self.x.shape[1] == 1:
                self.im('      - data detected: plan xz')
                self.plan = 'xz'
            elif self.x.shape[2] == 1:
                self.im('      - data detected: plan xy')
                self.plan = 'xy'
        else:
            self.im('      - 3D cartesian grid geometry')
        # Same operation but on index matrix:
        (Xind,Yind,Zind) = np.meshgrid(self.xind,self.yind,self.zind, indexing='ij')
        Xind = Xind.reshape(Xind.shape[0]*Xind.shape[1]*Xind.shape[2])
        Yind = Yind.reshape(Yind.shape[0]*Yind.shape[1]*Yind.shape[2])
        Zind = Zind.reshape(Zind.shape[0]*Zind.shape[1]*Zind.shape[2])
        self.XYZind = np.multiply(np.multiply(Xind,Yind),Zind)
        # Application of redFlag on index matrix:
        goodIndex = np.array(range(self.nx0*self.ny0*self.nz0))
        goodIndex = goodIndex[np.array(self.XYZind,dtype=bool)]
        
        #Processing of the field according to its scalar or vectorial nature:
        if self.fieldNature == 'Scalar':
            self.im('      - Build data grid for scalar field')
            (Nx, Ny, Nz) = self.header.get('nts')
            V = self.flds[0,:,:,:,0].reshape(Nx*Ny*Nz)
            self.v = V[goodIndex].reshape(self.nx,self.ny,self.nz)
            #Creation of empty vectorial fields arrays:
            self.vx     = np.array(self.vx)
            self.vy     = np.array(self.vy)
            self.vz     = np.array(self.vz)
            self.P      = np.array(self.P)

        elif self.fieldNature == 'Vectorial':
            self.im('      - Build data grid for vectorial field')
            (Nx, Ny, Nz) = self.header.get('nts')
            temp_vx = self.flds[0][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_vy = self.flds[1][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_vz = self.flds[2][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_P  = self.flds[3][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            self.vx = temp_vx[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.vy = temp_vy[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.vz = temp_vz[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.P  = temp_P[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.v  = np.sqrt(self.vx**2+self.vy**2+self.vz**2) #the norm

        # == Processing Finish !
        self.im('Processing of stag data done!')
    
    







class StagYinYangGeometry(MainStagObject):
    """
    Secondary geom class
    """
    def __init__(self):
        super().__init__()  # inherit all the methods and properties from MainStagObject
        self.geometry = 'yy'
        # ----- Yin Yang geometry ----- #
        self.X = []         #Matrix of X coordinates meshed
        self.Y = []         #Matrix of Y coordinates meshed
        self.Z = []         #Matrix of Z coordinates meshed
        self.layers = []    #matrix of layer's index meshed
        self.XYZind = []    #Matrix of good index after the mesh operation
        self.x1_overlap = []#Yin grid x matrix - overlapping grids:
        self.y1_overlap = []#Yin grid y matrix
        self.z1_overlap = []#Yin grid z matrix
        self.x2_overlap = []#Yang grid x matrix
        self.y2_overlap = []#Yang grid y matrix
        self.z2_overlap = []#Yang grid z matrix
        self.x1 = []        #Yin grid x matrix - non-overlapping grids:
        self.y1 = []        #Yin grid y matrix
        self.z1 = []        #Yin grid z matrix
        self.x2 = []        #Yang grid x matrix
        self.y2 = []        #Yang grid y matrix
        self.z2 = []        #Yang grid z matrix
        self.r1     = []    #Matrice of spherical coordinates r for the Yin grid
        self.theta1 = []    #Matrice of spherical coordinates theta for the Yin grid
        self.phi1   = []    #Matrice of spherical coordinates phi for the Yin grid
        self.r2     = []    #Matrice of spherical coordinates r for the Yang grid
        self.theta2 = []    #Matrice of spherical coordinates theta for the Yang grid
        self.phi2   = []    #Matrice of spherical coordinates phi for the Yang grid
        self.redFlags = []  #Matrix of wrong index in YY (overlaping pbs)
        self.x1_redf = []   #Matrix of redflag x-coordinates for Yin grid
        self.y1_redf = []   #Matrix of redflag y-coordinates for Yin grid
        self.z1_redf = []   #Matrix of redflag z-coordinates for Yin grid
        self.x2_redf = []   #Matrix of redflag x-coordinates for Yang grid
        self.y2_redf = []   #Matrix of redflag y-coordinates for Yang grid
        self.z2_redf = []   #Matrix of redflag z-coordinates for Yang grid
        self.redFlags_layers = [] #Matrix of layer's index meshed for redFlags points
        #For scalar field only:
        self.v1_overlap = []#Complete Yin field, corresponding to over '_overlap' matrices
        self.v2_overlap = []#Complete Yang field, corresponding to over '_overlap' matrices
        self.v1 = []        #Matrix of scalar field for the Yin grid (or norm of vectorial on Yin)
        self.v2 = []        #Matrix of scalar field for the Yang grid (or norm of vectorial on Yang)
        #For vectorial field only:
        self.vx1_overlap= [] #Complete vx Yin field, corresponding to over '_overlap' matrices
        self.vx2_overlap= [] #Complete vx Yang field, corresponding to over '_overlap' matrices
        self.vy1_overlap= [] #Complete vy Yin field
        self.vy2_overlap= [] #Complete vy Yang field
        self.vz1_overlap= [] #Complete vz Yin field
        self.vz2_overlap= [] #Complete vz Yang field
        self.P1_overlap = [] #Complete P Yin field
        self.P2_overlap = [] #Complete P Yang field
        self.vx1 = []        #Matrix of x-component of the vectorial field for the Yin grid
        self.vx2 = []        #Matrix of x-component of the vectorial field for the Yang grid
        self.vy1 = []        #Matrix of y-component of the vectorial field for the Yin grid
        self.vy2 = []        #Matrix of y-component of the vectorial field for the Yang grid
        self.vz1 = []        #Matrix of z-component of the vectorial field for the Yin grid
        self.vz2 = []        #Matrix of z-component of the vectorial field for the Yang grid
        self.P1  = []        #Matrix of the Pressure field for the Yin grid
        self.P2 = []         #Matrix of the Pressure field for the Yang grid
        self.vr1     = []    #Matrix of radial component of the vectorial field for the Yin grid
        self.vtheta1 = []    #Matrix of theta component of the vectorial field for the Yin grid
        self.vphi1   = []    #Matrix of phi component of the vectorial field for the Yin grid
        self.vr2     = []    #Matrix of radial component of the vectorial field for the Yang grid
        self.vtheta2 = []    #Matrix of theta component of the vectorial field for the Yang grid
        self.vphi2   = []    #Matrix of phi component of the vectorial field for the Yang grid
        # Stacked Yin Yang grid
        self.x      = []    #stacked grid (cartesian)
        self.y      = []
        self.z      = []
        self.r      = []    #stacked grid (spherical)
        self.theta  = []
        self.phi    = []
        self.v      = []    #stacked scalar field
        self.P      = []    #stacked pressure
        self.vx     = []    #stacked vectorial fields
        self.vy     = []
        self.vz     = []
        self.vtheta = []
        self.vphi   = []
        self.vr     = []


    def stagProcessing(self, build_redflag_point=False, build_overlapping_field=False):
        """ This function process stag data according to a YinYang geometry.
        If build_redflag_point == True, build coordinates matrices of the 
           redflag points and fills fields x-y-z_redf
        If build_overlapping_field == True, build ghost points on YY corner"""
        
        self.im('Processing stag Data:')
        self.im('  - Grid Geometry')
        self.im('      - Yin-Yang grid geometry')
        self.im('      - Preprocessing of coordinates matrices')
        (self.X,self.Y,self.Z) = np.meshgrid(self.x_coords,self.y_coords,self.z_coords, indexing='ij')
        self.X = self.X.reshape(self.X.shape[0]*self.X.shape[1]*self.X.shape[2])
        self.Y = self.Y.reshape(self.Y.shape[0]*self.Y.shape[1]*self.Y.shape[2])
        self.Z = self.Z.reshape(self.Z.shape[0]*self.Z.shape[1]*self.Z.shape[2])
        #Same operation but on layers matrix:
        (self.bin, self.BIN, self.layers) = np.meshgrid(self.x_coords,self.y_coords,self.slayers, indexing='ij')
        self.layers = self.layers.reshape(self.layers.shape[0]*self.layers.shape[1]*self.layers.shape[2])
        self.bin = None
        self.BIN = None
        #Same operation but on index matrix:
        (Xind,Yind,Zind) = np.meshgrid(self.xind,self.yind,self.zind, indexing='ij')
        Xind = Xind.reshape(Xind.shape[0]*Xind.shape[1]*Xind.shape[2])
        Yind = Yind.reshape(Yind.shape[0]*Yind.shape[1]*Yind.shape[2])
        Zind = Zind.reshape(Zind.shape[0]*Zind.shape[1]*Zind.shape[2])
        self.XYZind = np.multiply(np.multiply(Xind,Yind),Zind)
        #self.XYZind is built during the construction of the YY grid and follows
        #the good index for the field read here (= readFlag in index)

        #Functions for the 3D spherical YY grids
        def rectangular2YY(x,y,z,rcmb):
            """Returns the geometry of the two cartesian blocks corresponding
            to the overlapping Yin (x1,y1,z1) and Yang (x2,y2,z2) grids
            from the single block contained in the StagYY binary outputs.
            after bending cartesian boxes"""
            #Spherical coordinates:
            R = z+rcmb
            lat = np.pi/4 - x
            lon = y - 3*np.pi/4
            #Yin grid
            x1 = np.multiply(np.multiply(R,np.cos(lat)),np.cos(lon))
            y1 = np.multiply(np.multiply(R,np.cos(lat)),np.sin(lon))
            z1 = np.multiply(R,np.sin(lat))
            #Yang grid
            x2 = -x1
            y2 = z1
            z2 = y1
            return ((x1,y1,z1),(x2,y2,z2))
        
        def cartesian2spherical(x1,y1,z1,x2,y2,z2):
            """Converts cartesian coordinates of YY grid into spherical coordinates"""
            #Yin grid
            r1     = np.sqrt(x1**2+y1**2+z1**2)
            theta1 = np.arctan2(np.sqrt(x1**2+y1**2),z1)
            phi1   = np.arctan2(y1,x1)
            #Yang grid
            r2     = np.sqrt(x2**2+y2**2+z2**2)
            theta2 = np.arctan2(np.sqrt(x2**2+y2**2),z2)
            phi2   = np.arctan2(y2,x2)
            return ((r1,theta1,phi1),(r2,theta2,phi2))
        
        #Creation of Yin-Yang grids:
        self.im('      - Creation of the Yin-Yang grids')
        ((self.x1_overlap,self.y1_overlap,self.z1_overlap),(self.x2_overlap,self.y2_overlap,self.z2_overlap)) = \
            rectangular2YY(self.X,self.Y,self.Z,self.rcmb)
        ((self.r1,self.theta1,self.phi1),(self.r2,self.theta2,self.phi2)) = \
            cartesian2spherical(self.x1_overlap,self.y1_overlap,self.z1_overlap,self.x2_overlap,self.y2_overlap,self.z2_overlap)

        ##Cut off the corners from grid #1, which seems to do #2:
        ##Build Redflags on wrong coordinates
        theta12 = np.arccos(np.multiply(np.sin(self.theta1),np.sin(self.phi1)))
        self.redFlags = np.where(np.logical_or(np.logical_and((theta12>np.pi/4),(self.phi1>np.pi/2)),\
                                                np.logical_and((theta12<3*np.pi/4),(self.phi1<-np.pi/2))))[0]

        if build_redflag_point == True:
            print('      - Building RedFlags Points...')
            ((self.x1_redf,self.y1_redf,self.z1_redf),(self.x2_redf,self.y2_redf,self.z2_redf)) = (([],[],[]),([],[],[]))
            self.redFlags_layers = []
            for ind in self.redFlags:
                self.x1_redf.append(self.x1_overlap[ind])
                self.y1_redf.append(self.y1_overlap[ind])
                self.z1_redf.append(self.z1_overlap[ind])
                self.x2_redf.append(self.x2_overlap[ind])
                self.y2_redf.append(self.y2_overlap[ind])
                self.z2_redf.append(self.z2_overlap[ind])
                self.redFlags_layers.append(self.layers[ind])
        
        #Assembly Yin and Yang grids
        self.im('      - Assembly Yin and Yang grids')
        goodIndex = np.ones(len(self.x1_overlap),dtype=bool)
        goodIndex[np.array(self.redFlags)] = False
        self.x1     = self.x1_overlap[goodIndex]
        self.y1     = self.y1_overlap[goodIndex]
        self.z1     = self.z1_overlap[goodIndex]
        self.x2     = self.x2_overlap[goodIndex]
        self.y2     = self.y2_overlap[goodIndex]
        self.z2     = self.z2_overlap[goodIndex]
        self.r1     = self.r1[goodIndex]
        self.r2     = self.r2[goodIndex]
        self.theta1 = self.theta1[goodIndex]
        self.theta2 = self.theta2[goodIndex]
        self.phi1   = self.phi1[goodIndex]
        self.phi2   = self.phi2[goodIndex]
        self.layers = self.layers[goodIndex]
        self.layers = self.layers.astype(np.int)
        
        # Extract the scalar or the vectorial field V: V1 on Yin, V2 on Yang
        self.im('  - Construction of the appropriated vectorial field:')
        ## Application of redFlag on index matrix:
        ## return good index for the vectorial field (goodIndex):
        goodIndex = np.array(range(self.nx0*self.ny0*self.nz0))
        goodIndex = goodIndex[np.array(self.XYZind,dtype=bool)]

        #Two different types of field: Scalar or Vectorial
        if self.fieldNature == 'Scalar':
            self.im('      - Build data for the entire grids')
            tempField = self.flds[0].reshape(self.flds.shape[1]*self.flds.shape[2]*self.flds.shape[3],2)
            V1 = tempField[:,0]
            V2 = tempField[:,1]
            if build_overlapping_field:
                self.im('         - Overlapping field requested')
                self.v1_overlap = V1[goodIndex] #Yin
                self.v2_overlap = V2[goodIndex] #Yang
                
            #Apply redFlags on goodindex:
            self.im('      - Processing of redFlags')
            mask = np.ones(len(goodIndex),dtype=bool) # all True
            mask[np.array(self.redFlags)] = False
            #Creation of non overlapping data matrices for Yin and Yang
            goodIndex = goodIndex[mask]
            self.v1 = np.array(V1)[goodIndex]
            self.v2 = np.array(V2)[goodIndex]
            #Creation of empty vectorial fields arrays:
            self.vx1     = np.array(self.vx1)
            self.vy1     = np.array(self.vy1)
            self.vz1     = np.array(self.vz1)
            self.P1      = np.array(self.P1)
            self.vr1     = np.array(self.vr1)
            self.vtheta1 = np.array(self.vtheta1)
            self.vphi1   = np.array(self.vphi1)
            self.vx2     = np.array(self.vx2)
            self.vy2     = np.array(self.vy2)
            self.vz2     = np.array(self.vz2)
            self.P2      = np.array(self.P2)
            self.vr2     = np.array(self.vr2)
            self.vtheta2 = np.array(self.vtheta2)
            self.vphi2   = np.array(self.vphi2)
            # Gather Yin and Yang
            self.im('  - Gather Yin and Yang: all Yin then all Yang')
            self.im('     - Stack the grids')
            self.x       = np.stack((self.x1,self.x2)).reshape(2*len(self.x1))
            self.y       = np.stack((self.y1,self.y2)).reshape(2*len(self.y1))
            self.z       = np.stack((self.z1,self.z2)).reshape(2*len(self.z1))
            self.r       = np.stack((self.r1,self.r2)).reshape(2*len(self.r1))
            self.theta   = np.stack((self.theta1,self.theta2)).reshape(2*len(self.theta1))
            self.phi     = np.stack((self.phi1,self.phi2)).reshape(2*len(self.phi1))
            self.im('     - Stack the fields')
            # Scalar
            self.v       = np.stack((self.v1,self.v2)).reshape(2*len(self.v1))
            # Vectorial (no stacked because empty)
            self.vx      = np.array(self.vx)
            self.vy      = np.array(self.vy)
            self.vz      = np.array(self.vz)
            self.P       = np.array(self.P)
            self.vtheta  = np.array(self.vtheta)
            self.vphi    = np.array(self.vphi)
            self.vr      = np.array(self.vr)

            
        elif self.fieldNature == 'Vectorial':
            self.im('      - Build data for the entire grids')
            (Nx, Ny, Nz) = self.header.get('nts')
            tempField_vx = self.flds[0][0:Nx,0:Ny,:,:].reshape(Nx*Ny*Nz,2)
            tempField_vy = self.flds[1][0:Nx,0:Ny,:,:].reshape(Nx*Ny*Nz,2)
            tempField_vz = self.flds[2][0:Nx,0:Ny,:,:].reshape(Nx*Ny*Nz,2)
            tempField_P  = self.flds[3][0:Nx,0:Ny,:,:].reshape(Nx*Ny*Nz,2)
            VX1 = tempField_vx[:,0]
            VX2 = tempField_vx[:,1]
            VY1 = tempField_vy[:,0]
            VY2 = tempField_vy[:,1]
            VZ1 = tempField_vz[:,0]
            VZ2 = tempField_vz[:,1]
            P1 = tempField_P[:,0]
            P2 = tempField_P[:,1]

            #Transform velocities from internal Yin or Yang coord -> Cartesian
            self.im('      - Merging of velocities: YY -> Cartesian')
            tx_coord = self.header.get('e1_coord') #temps, will be immediately deleted after use
            ty_coord = self.header.get('e2_coord')
            tz_coord = self.header.get('e3_coord')
            (tX,tY,tZ) = np.meshgrid(tx_coord,ty_coord,tz_coord, indexing='ij')
            tX = tX.reshape(Nx*Ny*Nz)
            tY = tY.reshape(Nx*Ny*Nz)
            tZ = tZ.reshape(Nx*Ny*Nz)
            #R = tZ + self.rcmb
            lat = np.pi/4 - tX
            lon = tY - 3*np.pi/4
            # --- on Yin grid ---
            Vtheta = VX1
            Vphi   = VY1
            Vr     = VZ1
            VX1    =    Vtheta*np.sin(lat)*np.cos(lon) - Vphi*np.sin(lon) + Vr*np.cos(lat)*np.cos(lon)
            VY1    =    Vtheta*np.sin(lat)*np.sin(lon) + Vphi*np.cos(lon) + Vr*np.cos(lat)*np.sin(lon)
            VZ1    = -1*Vtheta*np.cos(lat)                                + Vr*np.sin(lat)
            vr1 = Vr
            # --- on Yang grid ---
            Vtheta = VX2
            Vphi   = VY2
            Vr     = VZ2
            VX2    = -1*(Vtheta*np.sin(lat)*np.cos(lon) - Vphi*np.sin(lon) + Vr*np.cos(lat)*np.cos(lon))
            VZ2    =     Vtheta*np.sin(lat)*np.sin(lon) + Vphi*np.cos(lon) + Vr*np.cos(lat)*np.sin(lon)
            VY2    =  -1*Vtheta*np.cos(lat)                                + Vr*np.sin(lat)
            vr2 = Vr
            #Discharge of the memory
            (tX, tY, tZ)       = (None, None, None)
            (Vtheta, Vphi, Vr) = (None, None, None)
            if build_overlapping_field:
                self.im('         - Overlapping field requested')
                #Re-sampling
                self.vx1_overlap = VX1[goodIndex] #Yin
                self.vx2_overlap = VX2[goodIndex] #Yang
                self.vy1_overlap = VY1[goodIndex]
                self.vy2_overlap = VY2[goodIndex]
                self.vz1_overlap = VZ1[goodIndex]
                self.vz2_overlap = VZ2[goodIndex]
                self.P1_overlap  = P1[goodIndex]
                self.P2_overlap  = P2[goodIndex]
                            
            #Apply redFlags on goodindex:
            self.im('      - Processing of redFlags')
            mask = np.ones(len(goodIndex),dtype=bool) # all True
            mask[np.array(self.redFlags)] = False
            goodIndex = goodIndex[mask]
            self.vx1 = VX1[goodIndex]
            self.vy1 = VY1[goodIndex]
            self.vz1 = VZ1[goodIndex]
            self.vx2 = VX2[goodIndex]
            self.vy2 = VY2[goodIndex]
            self.vz2 = VZ2[goodIndex]
            self.P1  = P1[goodIndex]
            self.P2  = P2[goodIndex]
            #Radial velocities
            self.vr1 = vr1[goodIndex]
            self.vr2 = vr2[goodIndex]
            
            #Tranformation of velocities from cartesian to spherical:
            self.im('      - Conversion of Velocities: Cartesian -> Spherical')
            lat1 = np.arctan2(np.sqrt(self.x1**2+self.y1**2),self.z1)
            lon1 = np.arctan2(self.y1,self.x1)
            lat2 = np.arctan2(np.sqrt(self.x2**2+self.y2**2),self.z2)
            lon2 = np.arctan2(self.y2,self.x2)
            
            Vlat1 =  self.vx1*(np.cos(lon1)*np.cos(lat1)) + self.vy1*(np.sin(lon1)*np.cos(lat1)) - self.vz1*(np.sin(lat1))
            Vlon1 = -self.vx1*(np.sin(lon1))              + self.vy1*(np.cos(lon1))
            Vlat2 =  self.vx2*(np.cos(lon2)*np.cos(lat2)) + self.vy2*(np.sin(lon2)*np.cos(lat2)) - self.vz2*(np.sin(lat2))
            Vlon2 = -self.vx2*(np.sin(lon2))              + self.vy2*(np.cos(lon2))
            
            #Conservation of the ndarray-type:
            self.vr1     = np.array(self.vr1)
            self.vr2     = np.array(self.vr2)
            self.vtheta1 = Vlat1
            self.vtheta2 = Vlat2
            self.vphi1   = Vlon1
            self.vphi2   = Vlon2

            #fills the .v1 and .v2 by the norm of the velocity
            self.v1  = np.sqrt(self.vx1**2+self.vy1**2+self.vz1**2) #the norm
            self.v2  = np.sqrt(self.vx2**2+self.vy2**2+self.vz2**2) #the norm

            # Gather Yin and Yang
            self.im('  - Gather Yin and Yang: all Yin then all Yang')
            self.im('     - Stack the grids')
            self.x       = np.stack((self.x1,self.x2)).reshape(2*len(self.x1))
            self.y       = np.stack((self.y1,self.y2)).reshape(2*len(self.y1))
            self.z       = np.stack((self.z1,self.z2)).reshape(2*len(self.z1))
            self.r       = np.stack((self.r1,self.r2)).reshape(2*len(self.r1))
            self.theta   = np.stack((self.theta1,self.theta2)).reshape(2*len(self.theta1))
            self.phi     = np.stack((self.phi1,self.phi2)).reshape(2*len(self.phi1))
            self.im('     - Stack the fields')
            # Scalar
            self.v       = np.stack((self.v1,self.v2)).reshape(2*len(self.v1))
            # Vectorial (no stacked because empty)
            self.vx      = np.stack((self.vx1,self.vx2)).reshape(2*len(self.v1))
            self.vy      = np.stack((self.vy1,self.vy2)).reshape(2*len(self.v1))
            self.vz      = np.stack((self.vz1,self.vz2)).reshape(2*len(self.v1))
            self.P       = np.stack((self.P1,self.P2)).reshape(2*len(self.v1))
            self.vtheta  = np.stack((self.vtheta1,self.vtheta2)).reshape(2*len(self.v1))
            self.vphi    = np.stack((self.vphi1,self.vphi2)).reshape(2*len(self.v1))
            self.vr      = np.stack((self.vr1,self.vr2)).reshape(2*len(self.v1))
        
        # == Processing Finish !
        self.im('Processing of stag data done!')
    
    
    def splitGird(self):
        """ This function split the loaded grid (x->x1+x2,
        for instance and do the operation for x, y, z, r,
        theta and phi)"""
        nYin     = len(self.x1)
        nYinYang = len(self.x)
        self.x1 = self.x[0:nYin]
        self.x2 = self.x[nYin:nYinYang]
        self.y1 = self.y[0:nYin]
        self.y2 = self.y[nYin:nYinYang]
        self.z1 = self.z[0:nYin]
        self.z2 = self.z[nYin:nYinYang]
        self.r1 = self.r[0:nYin]
        self.r2 = self.r[nYin:nYinYang]
        self.theta1 = self.theta[0:nYin]
        self.theta2 = self.theta[nYin:nYinYang]
        self.phi1 = self.phi[0:nYin]
        self.phi2 = self.phi[nYin:nYinYang]

    
    def mergeGird(self):
        """ This function merge the loaded sub-grids (x1+x2->x,
        for instance and do the operation for x, y, z, r, theta
        and phi)"""
        self.x = np.stack((self.x1,self.x2)).reshape(2*len(self.x1))
        self.y = np.stack((self.y1,self.y2)).reshape(2*len(self.x1))
        self.z = np.stack((self.z1,self.z2)).reshape(2*len(self.x1))
        self.r = np.stack((self.r1,self.r2)).reshape(2*len(self.x1))
        self.theta = np.stack((self.theta1,self.theta2)).reshape(2*len(self.x1))
        self.phi = np.stack((self.phi1,self.phi2)).reshape(2*len(self.x1))
    
    
    def splitFields(self):
        """ This function split the loaded fields on the all mesh (v and if available, vx, vy, vz,
        vphi, vtheta, vr and P) into the Yin-Yang subgrid: v1, v2 (and vx1,vx2,vy1,vy2...)"""
        self.im('Split the Yin-Yang fields to a field on the Yin gird and a field on the Yang grid (v->v1+v2)')
        nYin     = len(self.v1)
        nYinYang = len(self.v)
        self.v1 = self.v[0:nYin]
        self.v2 = self.v[nYin:nYinYang]
        if self.fieldNature == 'Vectorial':
            self.P1 = self.P[0:nYin]
            self.P2 = self.P[nYin:nYinYang]
            self.vx1 = self.vx[0:nYin]
            self.vx2 = self.vx[nYin:nYinYang]
            self.vy1 = self.vy[0:nYin]
            self.vy2 = self.vy[nYin:nYinYang]
            self.vz1 = self.vz[0:nYin]
            self.vz2 = self.vz[nYin:nYinYang]
            self.vr1 = self.vr[0:nYin]
            self.vr2 = self.vr[nYin:nYinYang]
            self.vtheta1 = self.vtheta[0:nYin]
            self.vtheta2 = self.vtheta[nYin:nYinYang]
            self.vphi1 = self.vphi[0:nYin]
            self.vphi2 = self.vphi[nYin:nYinYang]
    
    
    def mergeFields(self):
        """ This function merge the loaded fields from the sub-meshes (Yin and Yang) to
        the entire YY (Yin+Yang). i.e. merge v1+v2 -> v (and vx1+vx2->vx, vy1+vy2->vy ...
        if vectorial)."""
        self.im('Merge Yin and Yang fields (v1+v2->v)')
        self.v       = np.stack((self.v1,self.v2)).reshape(2*len(self.v1))
        if self.fieldNature == 'Vectorial':
            self.vx      = np.stack((self.vx1,self.vx2)).reshape(2*len(self.v1))
            self.vy      = np.stack((self.vy1,self.vy2)).reshape(2*len(self.v1))
            self.vz      = np.stack((self.vz1,self.vz2)).reshape(2*len(self.v1))
            self.P       = np.stack((self.P1,self.P2)).reshape(2*len(self.v1))
            self.vtheta  = np.stack((self.vtheta1,self.vtheta2)).reshape(2*len(self.v1))
            self.vphi    = np.stack((self.vphi1,self.vphi2)).reshape(2*len(self.v1))
            self.vr      = np.stack((self.vr1,self.vr2)).reshape(2*len(self.v1))
    
    
    def get_vprofile(self,field='v',lon=None,lat=None,x=None,y=None,z=None,phi=None,theta=None):
        """ Extract a vertical profile in the loaded data according to the coordinates
        of the intersection between the profile and shallowest layers (e.g the surface).
        The coordinates of this point can be set in three different bases: (lon,lat), or
        (theta,phi,[r is imposed to the shallowest loaded layer]) or (x,y,z).
        Note. The returned profile is not interpolated. This function will search the
        nearest point on the loaded grid for the profile.
        
        N.B. All the arguments are optional. However, is you set a longitude, a latitude
            is expected and vice versa. The same for x where y and z are expected or phi
            where theta is expected (and vice versa).
            The priority order is (lon,lat)>(x,y,z)>(theta,phi) if several coordinates system
            is set in input.
        Args:
            field (str, optionnal): String defining the field of the current class instance on which the
                                    the user want to extract a vertical profile. This argument must be
                                    in ['v' for scalar fields, 'vx', 'vy', 'vz', 'vr', 'vtheta', 'vphi',
                                    or 'P' for vectorial fields]. Defaults to 'v'
            lon (int/float, optional): longitude (in *DREGREE*) of the intersection between the profile
                                    and the shallowest layer. Defaults to None.
            lat (int/float, optional): latitude (in *DREGREE*) of the intersection between the profile
                                    and the shallowest layer. Defaults to None.
            x (int/float, optional): x coordinate of the intersection between the profile
                                    and the shallowest layer. Defaults to None.
            y (int/float, optional): y coordinate of the intersection between the profile
                                    and the shallowest layer. Defaults to None.
            z (int/float, optional): z coordinate of the intersection between the profile
                                    and the shallowest layer. Defaults to None.
            phi (int/float, optional): phi coordinate (in *RADIANS*) of the intersection between the profile
                                    and the shallowest layer. Defaults to None.
            theta (int/float, optional): theta coordinate (in *RADIANS*) of the intersection between the profile
                                    and the shallowest layer. Defaults to None.
        Outputs:
            vprof (np.ndarray): Vector of size Nz containing the extracted vertical profile
            coordinates (np.ndarray): Matrix (Nz,3) containing the true coordinates of the
                                      profile points. If the user set a lon/lat coordinate in input,
                                      returns (lon,lat,depths[km]) of the points. If the user set a
                                      x,y,z coordinates, returns a coordinates matix alos in x,y,z.
                                      Finally, if the user set a phi/theta coordinates to define the
                                      profile, returns a coordinate matrix with r/theta/phi.
        """
        self.im('Vertical profile extraction')
        Nz = self.slayers.shape[0]
        NxNy = int(self.x.shape[0]/Nz)
        compute = True
        if lon is not None and lat is None:
            raise StagBaseError('Longitude detected but the latitude is missing')
        if lon is None and lat is not None:
            raise StagBaseError('Latitude detected but the longitude is missing')
        cartesianBase = 0
        if x is not None:
            cartesianBase += 1
        if y is not None:
            cartesianBase += 1
        if z is not None:
            cartesianBase += 1
        if cartesianBase == 0 or cartesianBase == 3:
            cartesianBaseComplete = True
        else:
            cartesianBaseComplete = False
        if lon is None and lat is None and not cartesianBaseComplete:
            raise StagBaseError('Uncomplete cartesian base detected: Please, set both x,y and z')
        TPBase = 0
        if theta is not None:
            TPBase += 1
        if phi is not None:
            TPBase += 1
        if TPBase == 0 or TPBase == 2:
            TPBaseComplete = True
        else:
            TPBaseComplete = False
        print(TPBase, TPBaseComplete)
        if lon is None and lat is None and x is None and y is None and z is None and not TPBaseComplete:
            raise StagBaseError('Uncomplete theta,phi base detected: Please, set both theta and phi (in radians)')
        if lon is None and lat is None and x is None and y is None and z is None and theta is None and phi is None:
            raise StagBaseError('No coordinate for the profile: Please set a surface coordinate for the profile!')
        # ---
        if lon is not None and lat is not None:
            self.im('  -> Profile below:')
            self.im('      - Longitude = '+str(lon)+' deg')
            self.im('      - Latitude  = '+str(lat)+' deg')
            LONg = self.phi.reshape((NxNy,Nz))*180/np.pi
            LATg = -(self.theta.reshape((NxNy,Nz))*180/np.pi-90)
            long = LONg[:,-1]
            latg = LATg[:,-1]
            dist = np.sqrt((long-lon)**2+(latg-lat)**2)
            ids = np.where(dist==np.amin(dist))[0][0]
            self.im('      - Nearest surface point index = '+str(ids))
            coordinates = np.zeros((Nz,3))
            coordinates[:,0] = LONg[ids,:]
            coordinates[:,1] = LATg[ids,:]
            coordinates[:,2] = self.depths
            compute = False
        elif compute and x is not None and y is not None and z is not None:
            self.im('  -> Profile below:')
            self.im('      - x = '+str(x))
            self.im('      - y = '+str(y))
            self.im('      - z = '+str(z))
            Xg = self.x.reshape((NxNy,Nz))
            Yg = self.y.reshape((NxNy,Nz))
            Zg = self.z.reshape((NxNy,Nz))
            xg = Xg[:,-1]
            yg = Yg[:,-1]
            zg = Zg[:,-1]
            dist = np.sqrt((xg-x)**2+(yg-y)**2+(zg-z)**2)
            ids = np.where(dist==np.amin(dist))[0][0]
            self.im('      - Nearest surface point index = '+str(ids))
            coordinates = np.zeros((Nz,3))
            coordinates[:,0] = Xg[ids,:]
            coordinates[:,1] = Yg[ids,:]
            coordinates[:,2] = Zg[ids,:]
            compute = False
        elif compute and theta is not None and phi is not None:
            self.im('  -> Profile below:')
            self.im('      - theta = '+str(theta)+' rad')
            self.im('      - phi   = '+str(phi)+' rad')
            THETAg = self.theta.reshape((NxNy,Nz))
            PHIg   = self.phi.reshape((NxNy,Nz))
            thetag = THETAg[:,-1]
            phig   = PHIg[:,-1]
            dist = np.sqrt((thetag-theta)**2+(phig-phi)**2)
            ids = np.where(dist==np.amin(dist))[0][0]
            self.im('      - Nearest surface point index = '+str(ids))
            coordinates = np.zeros((Nz,3))
            coordinates[:,0] = self.r.reshape((NxNy,Nz))[ids,:]
            coordinates[:,1] = THETAg[ids,:]
            coordinates[:,2] = PHIg[ids,:]
            compute = False
        if field == 'v':
            vprof = self.v.reshape((NxNy,Nz))[ids,:]
        elif field == 'vx':
            if self.fieldNature == 'Vectorial':
                vprof = self.vx.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vy':
            if self.fieldNature == 'Vectorial':
                vprof = self.vy.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vz':
            if self.fieldNature == 'Vectorial':
                vprof = self.vz.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vr':
            if self.fieldNature == 'Vectorial':
                vprof = self.vr.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vtheta':
            if self.fieldNature == 'Vectorial':
                vprof = self.vtheta.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vphi':
            if self.fieldNature == 'Vectorial':
                vprof = self.vphi.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'P':
            if self.fieldNature == 'Vectorial':
                vprof = self.P.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        return vprof,coordinates

    
    def set_pole_projection(self,rot,verbose=True):
        """
        Substract the rotation 'rot' defined by (wx,wy,wz) to the
        entire velocity field, layer by layer.
        """
        self.im('Substract a rotation to the whole mantle velocity field')
        if self.fieldType != 'Velocity':
            raise fieldTypeError('Velocity')
        else:
            wx,wy,wz = rot
            Nz = self.nz
            NxNy = int(self.x.shape[0]/Nz)
            # --- input data
            x = self.x.reshape((NxNy,Nz))
            y = self.y.reshape((NxNy,Nz))
            z = self.z.reshape((NxNy,Nz))
            vx = self.vx.reshape((NxNy,Nz))
            vy = self.vy.reshape((NxNy,Nz))
            vz = self.vz.reshape((NxNy,Nz))
            # --- output data
            vxo = np.zeros(x.shape)
            vyo = np.zeros(x.shape)
            vzo = np.zeros(x.shape)
            vphio   = np.zeros(x.shape)
            vthetao = np.zeros(x.shape)
            vro   = np.zeros(x.shape)
            # --- iteration on all layers
            self.im('  -> Iterative computation on the %s layers'%str(self.nz))
            for i in range(Nz):
                if verbose:
                    print('Works on the layer: '+str(i+1)+'/'+str(Nz))
                xi = x[:,i]
                yi = y[:,i]
                zi = z[:,i]
                vxi = vx[:,i]
                vyi = vy[:,i]
                vzi = vz[:,i]
                # 
                vxo[:,i], vyo[:,i], vzo[:,i], vphio[:,i], vthetao[:,i], vro[:,i] = velocity_pole_projecton(xi,yi,zi,vxi,vyi,vzi,wx,wy,wz)
            # --- rewrite the current class instance fields
            self.vx = vxo.reshape(NxNy*Nz)
            self.vy = vyo.reshape(NxNy*Nz)
            self.vz = vzo.reshape(NxNy*Nz)
            self.vphi   = vphio.reshape(NxNy*Nz)
            self.vtheta = vthetao.reshape(NxNy*Nz)
            self.vr     = vro.reshape(NxNy*Nz)
            self.splitFields()
            self.im('Velocity reprojection: Done')
    
    
    def grid_rotation(self,axis='x',theta='1*np.pi/180',R=None):
        """
        Function for the rotation of the grid defined as either (1) a rotation
        around a given cartesian axis (x,y,z) or (2) as 3D rotation matrix
        <i> : - axis, str,  in ['x','y','z']
              - theta, int/float,  in  *RADIANS*
              - R, np.ndarray, shape = 3x3
        N.B. The rotation is applied on both the Yin and the Yang grid,
             plus the merged grid and fields but not to the overlapping
             mesh and fields.
        """
        if self.fieldNature == 'Scalar':
            self.im('Rotation of the grid')
        else:
            self.im('Rotation of the grid and vectors')
        if R is None:
            self.im('  -> axis:  '+axis)
            self.im('  -> theta: '+str(theta))
            R = rotation_matrix_3D(axis,theta)
        else:
            self.im('  -> rotation matrix: '+str(R))
        # ---
        self.im('  -> Grid rotation:')
        x = R[0,0]*self.x+R[0,1]*self.y+R[0,2]*self.z
        y = R[1,0]*self.x+R[1,1]*self.y+R[1,2]*self.z
        z = R[2,0]*self.x+R[2,1]*self.y+R[2,2]*self.z
        self.x = x
        self.y = y
        self.z = z
        self.theta, self.phi, self.r = xyz2latlon(self.x,self.y,self.z)
        self.splitGird()
        if self.fieldType == 'Velocity':
            self.im('  -> Vectors rotation:')
            vx = R[0,0]*self.vx+R[0,1]*self.vy+R[0,2]*self.vz
            vy = R[1,0]*self.vx+R[1,1]*self.vy+R[1,2]*self.vz
            vz = R[2,0]*self.vx+R[2,1]*self.vy+R[2,2]*self.vz
            self.vx = vx
            self.vy = vy
            self.vz = vz
            self.vphi, self.vtheta, self.vr = ecef2enu_stagYY(self.x,self.y,self.z,self.vx,self.vy,self.vz)
            self.splitFields()
        

        
    

                        
            
            
        
            
        
        
   
    
    


class StagSphericalGeometry(MainStagObject):
    """
    Defines the StagSphericalGeometry object, derived from MainStagObject
    This object is conditionally inherited in StagData.
    """
    def __init__(self,geometry):
        super().__init__()  # inherit all the methods and properties from MainStagObject
        self.geometry = geometry
        self.plan     = None # stay None for 3D spherical and get a value for annulus
        self.x  = []        #Matrix of X coordinates meshed (in spherical shape)
        self.y  = []        #Matrix of Y coordinates meshed (in spherical shape)
        self.z  = []        #Matrix of Z coordinates meshed (in spherical shape)
        self.xc = []        #Matrice of cartesian x coordinates (in cartesian shape)
        self.yc = []        #Matrice of cartesian y coordinates (in cartesian shape)
        self.zc = []        #Matrice of cartesian z coordinates (in cartesian shape)
        self.r     = []     #Matrice of spherical coordinates r
        self.theta = []     #Matrice of spherical coordinates theta
        self.phi   = []     #Matrice of spherical coordinates phi
        self.v  = []        #Matrix of scalar field (or norm of vectorial)
        self.vx = []        #Matrix of x-component of the vectorial field for Cartesian grids
        self.vy = []        #Matrix of y-component of the vectorial field for Cartesian grids
        self.vz = []        #Matrix of z-component of the vectorial field for Cartesian grids
        self.vtheta = []    #Matrix of theta component of the vectorial field
        self.vphi   = []    #Matrix of phi component of the vectorial field
        self.vr     = []    #Matrix of radial component of the vectorial field
        self.P  = []        #Matrix of Pressure field for Cartesian grids
    
    def stagProcessing(self):
        """
        This function process stag data and returns the appropriated coords
        matrices (1 matrix Yin and 1 matrix for Yqng coords) as well as matrix
        of the reading field for Yin and for Yang.
        If build_redflag_point == True, build coordinates matrices of the 
           redflag points and fills fields x-y-z_redf
        If build_overlapping_field == True, build ghost points on YY corner
        """
        self.im('Processing stag Data:')
        self.im('  - Grid Geometry')
        # Meshing
        (self.x,self.y,self.z) = np.meshgrid(self.x_coords,self.y_coords,self.z_coords,indexing='ij')
        # Geometry
        if self.geometry == 'spherical':
            self.im('      - 3D cartesian grid geometry')
        elif self.geometry == 'annulus':
            self.im('      - 2D annulus grid geometry')
            if  self.x.shape[0] == 1:
                self.im('      - data detected: plan yz')
                self.plan = 'yz'
            elif self.x.shape[1] == 1:
                self.im('      - data detected: plan xz')
                self.plan = 'xz'
            elif self.x.shape[2] == 1:
                self.im('      - data detected: plan xy')
                self.plan = 'xy'
        #save cartesian grid geometry
        self.xc = self.x
        self.yc = self.y
        self.zc = self.z
        #Same operation but on index matrix:
        (Xind,Yind,Zind) = np.meshgrid(self.xind,self.yind,self.zind, indexing='ij')
        Xind = Xind.reshape(Xind.shape[0]*Xind.shape[1]*Xind.shape[2])
        Yind = Yind.reshape(Yind.shape[0]*Yind.shape[1]*Yind.shape[2])
        Zind = Zind.reshape(Zind.shape[0]*Zind.shape[1]*Zind.shape[2])
        self.XYZind = np.multiply(np.multiply(Xind,Yind),Zind)
        # Application of redFlag on index matrix:
        goodIndex = np.array(range(self.nx0*self.ny0*self.nz0))
        goodIndex = goodIndex[np.array(self.XYZind,dtype=bool)]

        #Function for 3D psherical YY grids
        def rectangular2Spherical(x,y,z,rcmb):
            """Returns the geometry of the spherical grid
            after bending the cartesian box"""
            #Spherical coordinates:
            R = z+rcmb
            lat = np.pi/4 - x
            lon = y - 3*np.pi/4
            #Spherical grid
            x = np.multiply(np.multiply(R,np.cos(lat)),np.cos(lon))
            y = np.multiply(np.multiply(R,np.cos(lat)),np.sin(lon))
            z = np.multiply(R,np.sin(lat))
            return (x,y,z)
        
        def cartesian2spherical(x,y,z):
            """Converts cartesian coordinates into spherical coordinates"""
            r     = np.sqrt(x**2+y**2+z**2)
            theta = np.arctan2(np.sqrt(x**2+y**2),z)
            phi   = np.arctan2(y,x)
            return (r,theta,phi)
        
        #Creation of the spherical grid:
        self.im('      - Creation of the spherical grids')
        (self.x,self.y,self.z) = rectangular2Spherical(self.x,self.y,self.z,self.rcmb)

        (self.r,self.theta,self.phi) = cartesian2spherical(self.x,self.y,self.z)
            #=============================================================

        #Processing of the field according to its scalar or vectorial nature:
        if self.fieldNature == 'Scalar':
            self.im('      - Build data grid for scalar field')
            (Nx, Ny, Nz) = self.header.get('nts')
            V = self.flds[0,:,:,:,0].reshape(Nx*Ny*Nz)
            self.v = V[goodIndex].reshape(self.nx,self.ny,self.nz)
            #Creation of empty vectorial fields arrays:
            self.vx     = np.array(self.vx)
            self.vy     = np.array(self.vy)
            self.vz     = np.array(self.vz)
            self.P      = np.array(self.P)
            self.vr     = np.array(self.vr)
            self.vtheta = np.array(self.vtheta)
            self.vphi   = np.array(self.vphi)

        elif self.fieldNature == 'Vectorial':
            self.im('      - Build data grid for vectorial field')
            (Nx, Ny, Nz) = self.header.get('nts')
            temp_vx = self.flds[0][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_vy = self.flds[1][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_vz = self.flds[2][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_P  = self.flds[3][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            self.vx = temp_vx[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.vy = temp_vy[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.vz = temp_vz[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.P  = temp_P[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.v  = np.sqrt(self.vx**2+self.vy**2+self.vz**2) #the norm

            # -- From now, like for YY grids
            #Transform velocities from internal Yin or Yang coord -> Cartesian
            self.im('      - Merging of velocities: YY -> Cartesian')
            tx_coord = self.header.get('e1_coord') #temps, will be immediately deleted after use
            ty_coord = self.header.get('e2_coord')
            tz_coord = self.header.get('e3_coord')
            (tX,tY,tZ) = np.meshgrid(self.x_coords,self.y_coords,self.z_coords,indexing='ij')
            tX = tX.flatten()
            tY = tY.flatten()
            tZ = tZ.flatten()
            #R = tZ + self.rcmb
            lat = np.pi/4 - tX
            lon = tY - 3*np.pi/4
            # --- on grid ---
            Vtheta = self.vx.flatten()
            Vphi   = self.vy.flatten()
            Vr     = self.vz.flatten()
            self.vx    =    Vtheta*np.sin(lat)*np.cos(lon) - Vphi*np.sin(lon) + Vr*np.cos(lat)*np.cos(lon)
            self.vy    =    Vtheta*np.sin(lat)*np.sin(lon) + Vphi*np.cos(lon) + Vr*np.cos(lat)*np.sin(lon)
            self.vz    = -1*Vtheta*np.cos(lat)                                + Vr*np.sin(lat)
            self.vr    = Vr
            #Discharge of the memory
            (tX, tY, tZ)       = (None, None, None)
            (Vtheta, Vphi, Vr) = (None, None, None)
            
            #Adjusting the type
            #self.vx = np.array(self.vx)
            #self.vy = np.array(self.vy)
            #self.vz = np.array(self.vz)
            #self.vr = np.array(self.vr)
            #self.P = np.array(self.P)
            
            #Tranformation of velocities from cartesian to spherical:
            self.im('      - Conversion of Velocities: Cartesian -> Spherical')
            lat1 = np.arctan2(np.sqrt(self.x.flatten()**2+self.y.flatten()**2),self.z.flatten())
            lon1 = np.arctan2(self.y.flatten(),self.x.flatten())
            
            Vlat1 =  self.vx*(np.cos(lon1)*np.cos(lat1)) + self.vy*(np.sin(lon1)*np.cos(lat1)) - self.vz*(np.sin(lat1))
            Vlon1 = -self.vx*(np.sin(lon1))              + self.vy*(np.cos(lon1))
            
            #Conservation of the ndarray-type:
            self.vtheta = Vlat1.reshape(self.nx,self.ny,self.nz)
            self.vphi   = Vlon1.reshape(self.nx,self.ny,self.nz)
            self.vx = self.vx.reshape(self.nx,self.ny,self.nz)
            self.vy = self.vy.reshape(self.nx,self.ny,self.nz)
            self.vz = self.vz.reshape(self.nx,self.ny,self.nz)
            self.vr = self.vr.reshape(self.nx,self.ny,self.nz)
            self.P  = self.P.reshape(self.nx,self.ny,self.nz)
        
            #fills the .v1 and .v2 by the norm of the velocity
            self.v  = np.sqrt(self.vx**2+self.vy**2+self.vz**2) #the norm
            self.v  = self.v.reshape(self.nx,self.ny,self.nz)
        
        # == Processing Finish !
        self.im('Processing of stag data done!')







class StagData():
    """
    Defines the StagData structure dynamically from geometry of the grid.
    """
    def __new__(cls,geometry='cart3D'):
        """
        Force to have more than just 'duck typing' in Python: 'dynamical typing'
        <i> : geometry = str, geometry of the grid. Must be in ('cart2D',
                         'cart3D','yy','annulus') for cartesian 2D, 3D,
                         Yin-Yang or annulus geometry, respectively. By
                         default, geometry = 'cart3D'
        """
        if geometry == 'yy':
            return StagYinYangGeometry()
        elif geometry == 'cart2D' or geometry == 'cart3D':
            return StagCartesianGeometry(geometry)
        elif geometry == 'spherical':
            return StagSphericalGeometry(geometry)
        elif geometry == 'annulus':
            return StagSphericalGeometry(geometry)
            #raise GridGeometryInDevError(geometry)
        else:
            raise InputGridGeometryError(geometry)









class MainSliceData:
    """
    Main class defining the highest level of inheritance
    for StagData derived object
    """
    def __init__(self):
        """
        Parent builder
        """
        # ----- Generic ----- #
        self.pName = 'sliceData'
        self.verbose = True       #Condition on the verbose output
        self.fieldType = 'Temperature'  #Field contained in the current object
        self.fieldNature = 'Scalar'     #Nature of the field: Scalar or Vectorial
        self.path  = ''                 #The path to the stag file
        self.fname = ''                 #File name of the stag file
        self.resampling = []#Resampling Parameters
        self.simuAge = 0    #Dimensionless age of the simulation
        self.ti_step = 0    #Inner step of the stag simualtion state
        self.layer = 0      #Selected value of the stagData.slayer for the current slice
        self.depth = 0      #Corresponding depth in km according to rcmb
        self.rcmb  = 0      #Radius of the Core-Mantle Boundary
        self.nx0 = 0        #Number of point in the x direction in the original input file
        self.ny0 = 0        #Number of point in the y direction in the original input file
        self.nz0 = 0        #Number of point in the z direction in the original input file
        self.nx  = 0        #Current number of point in the x direction (after resampling)
        self.ny  = 0        #Current number of point in the y direction (after resampling)
        self.nz  = 0        #Current number of point in the z direction (after resampling)
        # Slice parameters:
        self.axis  = None   #Axis of the slice
        self.layer = None   #Layer index of the slice
        # Other
        self.BIN = None
        self.bin = None
    

    def im(self,textMessage):
        """Print verbose internal message. This function depends on the
        argument of self.verbose. If self.verbose == True then the message
        will be displayed on the terminal.
        <i> : textMessage = str, message to display
        """
        if self.verbose == True:
            print('>> '+self.pName+'| '+textMessage)
    
    def sliceInheritance(self,stagData):
        """
        Manages all field inheritance. Notice that stagData can be here
        another sliceData (for instance in the particular case of a InterpolatedSliceData)
        """
        self.fieldType   = stagData.fieldType
        self.fieldNature = stagData.fieldNature
        self.path       = stagData.path
        self.fname      = stagData.fname
        self.resampling = stagData.resampling
        self.simuAge = stagData.simuAge
        self.ti_step = stagData.ti_step
        self.rcmb    = stagData.rcmb
        self.nx0 = stagData.nx0
        self.ny0 = stagData.ny0
        self.nz0 = stagData.nz0
        self.nx  = stagData.nx
        self.ny  = stagData.ny
        self.nz  = stagData.nz
        




class YinYangSliceData(MainSliceData):
    """
    Defines the structure of the YinYangSliceData object derived from MainSliceData type.
    This object corresponds to a simplified StagYinYangGeometry object.
    """
    def __init__(self):
        super().__init__()  # inherit all the methods and properties from MainSliceData
        self.geometry = 'yy'
        self.normal = None
        # ----- Yin Yang geometry ----- #
        self.r1 = []        #Matrix of the radius of points for Yin grid
        self.r2 = []        #Matrix of the radius of points for Yang grid
        self.x1 = []        #Yin grid x matrix - non-overlapping grids:
        self.y1 = []        #Yin grid y matrix
        self.z1 = []        #Yin grid z matrix
        self.x2 = []        #Yang grid x matrix
        self.y2 = []        #Yang grid y matrix
        self.z2 = []        #Yang grid z matrix
        self.theta1 = []    #Yin  grid theta matrix
        self.theta2 = []    #Yang grid theta matrix
        self.phi1 = []      #Yin  grid phi matric
        self.phi2 = []      #Yang grid phi matric
        #For scalar field only:
        self.v1 = []        #Matrix of scalar field for the Yin grid (or norm of velocity on Yin)
        self.v2 = []        #Matrix of scalar field for the Yang grid (or norm of velocity on Yang)
        #For vectorial field only:
        self.vx1 = []        #Matrix of x-component of the velocity field for the Yin grid
        self.vx2 = []        #Matrix of x-component of the velocity field for the Yang grid
        self.vy1 = []        #Matrix of y-component of the velocity field for the Yin grid
        self.vy2 = []        #Matrix of y-component of the velocity field for the Yang grid
        self.vz1 = []        #Matrix of z-component of the velocity field for the Yin grid
        self.vz2 = []        #Matrix of z-component of the velocity field for the Yang grid
        self.P1  = []        #Matrix of the Pressure field for the Yin grid
        self.P2 = []         #Matrix of the Pressure field for the Yang grid
        self.vr1     = []    #Matrix of radial component of the velocity field for the Yin grid
        self.vtheta1 = []    #Matrix of theta component of the velocity field for the Yin grid
        self.vphi1   = []    #Matrix of phi component of the velocity field for the Yin grid
        self.vr2     = []    #Matrix of radial component of the velocity field for the Yang grid
        self.vtheta2 = []    #Matrix of theta component of the velocity field for the Yang grid
        self.vphi2   = []    #Matrix of phi component of the velocity field for the Yang grid
        #Stacked geometry
        self.x = []    #np.ndarray for x1 and x2 stacked
        self.y = []    #np.ndarray for y1 and y2 stacked
        self.z = []    #np.ndarray for z1 and z2 stacked
        self.r = []
        self.theta = []
        self.phi = []
        #Stacked scalar fields
        self.v = []    #np.ndarray for v1 and v2 stacked
        #Staked vectorial fields
        self.vx = []
        self.vy = []
        self.vz = []
        self.P  = []
        self.vtheta = []
        self.vphi   = []
        self.vr     = []

        
    def stackyy(self,nodp_x1,nodp_x2,nod_v1,nod_v2):
        """
        Computes all stacked fields from YinYang grid
        -> Stack Yin and Yang grid

            nodp_x1 = self.x1.shape[0]
            nodp_x2 = self.x2.shape[0]
            nod_v1  = self.v1.shape[0]
            nod_v2  = self.v2.shape[0]
        """
        #Dynamic containers: Use CPU on each call
        self.im('Stack grid matrices')
        self.x              = np.zeros((nodp_x1+nodp_x2))
        self.x[0:nodp_x1]      = self.x1
        self.x[nodp_x1:nodp_x1+nodp_x2] = self.x2
        self.y              = np.zeros((nodp_x1+nodp_x2))
        self.y[0:nodp_x1]      = self.y1
        self.y[nodp_x1:nodp_x1+nodp_x2] = self.y2
        self.z              = np.zeros((nodp_x1+nodp_x2))
        self.z[0:nodp_x1]      = self.z1
        self.z[nodp_x1:nodp_x1+nodp_x2] = self.z2
        self.r              = np.zeros((nodp_x1+nodp_x2))
        self.r[0:nodp_x1]      = self.r1
        self.r[nodp_x1:nodp_x1+nodp_x2] = self.r2
        self.theta              = np.zeros((nodp_x1+nodp_x2))
        self.theta[0:nodp_x1]      = self.theta1
        self.theta[nodp_x1:nodp_x1+nodp_x2] = self.theta2
        self.phi                = np.zeros((nodp_x1+nodp_x2))
        self.phi[0:nodp_x1]        = self.phi1
        self.phi[nodp_x1:nodp_x1+nodp_x2]   = self.phi2
        self.im('Stack fields')
        if self.fieldNature == 'Scalar':
            self.v              = np.zeros((nod_v1+nod_v2))
            self.v[0:nod_v1]      = self.v1
            self.v[nod_v1:nod_v1+nod_v2] = self.v2
            # empty
            self.vx,self.vy,self.vz,self.vr = np.array([]),np.array([]),np.array([]),np.array([])
            self.vtheta,self.vphi,self.P = np.array([]),np.array([]),np.array([])
        else:
            self.v              = np.zeros((nod_v1+nod_v2)) # norm
            self.v[0:nod_v1]      = self.v1
            self.v[nod_v1:nod_v1+nod_v2] = self.v2
            self.vx              = np.zeros((nod_v1+nod_v2))
            self.vx[0:nod_v1]      = self.vx1
            self.vx[nod_v1:nod_v1+nod_v2] = self.vx2
            self.vy              = np.zeros((nod_v1+nod_v2))
            self.vy[0:nod_v1]      = self.vy1
            self.vy[nod_v1:nod_v1+nod_v2] = self.vy2
            self.vz              = np.zeros((nod_v1+nod_v2))
            self.vz[0:nod_v1]      = self.vz1
            self.vz[nod_v1:nod_v1+nod_v2] = self.vz2
            self.P               = np.zeros((nod_v1+nod_v2))
            self.P[0:nod_v1]       = self.P1
            self.P[nod_v1:nod_v1+nod_v2]  = self.P2
            self.vr              = np.zeros((nod_v1+nod_v2))
            self.vr[0:nod_v1]      = self.vr1
            self.vr[nod_v1:nod_v1+nod_v2] = self.vr2
            self.vtheta              = np.zeros((nod_v1+nod_v2))
            self.vtheta[0:nod_v1]      = self.vtheta1
            self.vtheta[nod_v1:nod_v1+nod_v2] = self.vtheta2
            self.vphi                = np.zeros((nod_v1+nod_v2)) # norm
            self.vphi[0:nod_v1]        = self.vphi1
            self.vphi[nod_v1:nod_v1+nod_v2]   = self.vphi2
        self.im('Stacking done successfully!')


    def slicing(self,stagData,axis=0,normal=[1,0,0],layer=-1,interp_method='nearest'):
        """
        Extract an annulus-slice or a depth-slice in a stagData.StagYinYangGeometry object.
        The annulus-slice is defined according to a normal vector, perpendicular to the slicing plan.
        The detph-slice of defined according to the depth layer.
        <i> : stagData = stagData.StagYinYangGeometry
              axis     = int, integer indicating the axis of the slice (default, axis=0).
                         axis = 0  or axis = 'annulus'
                                    -> an annulus-slice 
                                      The normal of the plan containing the annulus is 
                                      is given in the 'normal' input argument
                                      WARNING: With the annulus slicing, you will loose the dual Ying Yang description (x1,x2 -> x)
                         axis = 1  or axis = 'layer'
                                    -> a r-constant-slice (depth-slice, a.k.a iso r)
                                      The layer index for the slice is given in the 'layer' input argument
              layer    = int, (only of axis == 1), value of the stagData.slayers that will be extracted in
                         the new SliceData object.
                         e.g. if you chose layer = 109 then you will extract the layer of the stagData where
                              stagData.slayers == 109
                         Default: layer = -1
              normal   = list/array, (only if axis == 0), vector of coordinates corresponding
                         to the normal to the plan containing the annlus.
                         This definition is consistent with the normal of the slicing plan in the Paraview software!
                         normal = (nx,ny,nz)
                         Default: normal = (1,0,0)
        """
        self.im('Begin the slice extraction')
        #check the geometry:
        if not isinstance(stagData,StagYinYangGeometry):
            raise StagTypeError(str(type(stagData)),'stagData.StagYinYangGeometry')
        # field type:
        ftype = stagData.fieldNature
        if ftype == 'Scalar':
            self.im('  - Scalar field detected')
        else:
            self.im('  - Vectorial field detected: '+str(stagData.flds.shape[0])+' fields')
        #Parameters of the StagData object:
        Nz   = len(stagData.slayers)      #Number of depth layers
        NxNy = int(len(stagData.x1)/Nz)   #Number of points for each layers
        #StagInheritance
        self.sliceInheritance(stagData)

        if axis == 0 or axis == 'annulus':
            self.im('REMINDER:  With the annulus slicing, you will loose the dual Ying Yang description')
            self.im('Extraction of an annulus slice (i.e. axis=0')
            self.im('   Normal to the slicing plan: '+str(normal[0])+','+str(normal[1])+','+str(normal[2]))
            self.normal = normal
            small = 1e-10           # to avoid to divide by 0
            normal[0] += small      #
            normal[1] += small      #
            normal[2] += small      #
            a = normal[0]
            b = normal[1]
            c = normal[2]
            # Compute the thickness of the slice:
            R = np.sqrt((4*np.pi*2.19**2)/(self.nx*self.ny))/2
            # Plan equation:
            self.im('   Search on the plan')
            gind1,gind2 = [],[]
            self.im('    -> Optimization of the search')
            while len(gind1)+len(gind2) < 0.005*len(stagData.x):
                gind1 = np.where(abs(a*stagData.x1+b*stagData.y1+c*stagData.z1) <= R)[0]
                gind2 = np.where(abs(a*stagData.x2+b*stagData.y2+c*stagData.z2) <= R)[0]
                R = R*1.2
            self.Rfinal = R/1.2
            self.im('    -> Final thickness of the pre-slice: '+str(R))
            # Cut YY according to plan equation
            x1,y1,z1,self.r1 = stagData.x1[gind1],stagData.y1[gind1],stagData.z1[gind1],stagData.r1[gind1]
            self.theta1,self.phi1 = stagData.theta1[gind1],stagData.phi1[gind1]
            x2,y2,z2,self.r2 = stagData.x2[gind2],stagData.y2[gind2],stagData.z2[gind2],stagData.r2[gind2]
            self.theta2,self.phi2 = stagData.theta2[gind2],stagData.phi2[gind2]
            if ftype == 'Scalar':
                self.v1,self.v2 = stagData.v1[gind1],stagData.v2[gind2]
                # empty
                self.vx1,self.vy1, self.vz1, self.vr1 = stagData.vx1,stagData.vy1,stagData.vz1,stagData.vr1
                self.vx2,self.vy2, self.vz2, self.vr2 = stagData.vx2,stagData.vy2,stagData.vz2,stagData.vr2
                self.vtheta1, self.vphi1, self.P1 = stagData.vtheta1,stagData.vphi1,stagData.P1
                self.vtheta2, self.vphi2, self.P2 = stagData.vtheta2,stagData.vphi2,stagData.P2
            else:
                self.v1,self.v2 = stagData.v1[gind1],stagData.v2[gind2] # norm
                self.vx1,self.vy1,self.vz1,self.vr1 = stagData.vx1[gind1],stagData.vy1[gind1],stagData.vz1[gind1],stagData.vr1[gind1]
                self.vx2,self.vy2,self.vz2,self.vr2 = stagData.vx2[gind2],stagData.vy2[gind2],stagData.vz2[gind2],stagData.vr2[gind2]
                self.vtheta1,self.vphi1,self.P1 = stagData.vtheta1[gind1],stagData.vphi1[gind1],stagData.P1[gind1]
                self.vtheta2,self.vphi2,self.P2 = stagData.vtheta2[gind2],stagData.vphi2[gind2],stagData.P2[gind2]
            # Compute the normal vectors to the slicing plan:
            self.normalu = np.array([1,-a/b,0])
            self.normalv = np.array([a/b,1,-(a**2+b**2)/(c*b)])
            self.normalw = np.array([a,b,c])
            # Projection
            self.im('   Projection on the plan')
            self.x1 = np.dot(np.array([x1,y1,z1]).T,self.normalu)/np.linalg.norm(self.normalu)
            self.y1 = np.dot(np.array([x1,y1,z1]).T,self.normalv)/np.linalg.norm(self.normalv)
            self.z1 = np.dot(np.array([x1,y1,z1]).T,self.normalw)/np.linalg.norm(self.normalw)
            self.x2 = np.dot(np.array([x2,y2,z2]).T,self.normalu)/np.linalg.norm(self.normalu)
            self.y2 = np.dot(np.array([x2,y2,z2]).T,self.normalv)/np.linalg.norm(self.normalv)
            self.z2 = np.dot(np.array([x2,y2,z2]).T,self.normalw)/np.linalg.norm(self.normalw)
            #stacking
            self.stackyy(self.x1.shape[0],self.x2.shape[0],self.v1.shape[0],self.v2.shape[0])
            
            # Added by A.JANIN 24/03/22
            # ----------------------------------------------
            # --- Search the ideal number of points along the annulus
            self.im('   - build ideal annulus geometry')
            totSurf     = np.count_nonzero(stagData.layers == stagData.slayers[0])*2
            opti_radius = np.sqrt(totSurf/(4*np.pi))
            opti_peri   = int(2*np.pi*opti_radius)
            opti_peri   = opti_peri * 1 # optimization

            # --- Description of the new grid: the two axis
            lon = np.linspace(0,2*np.pi,opti_peri)
            R   = np.array(stagData.z_coords)+stagData.rcmb

            # --- meshed grid
            lon,R = np.meshgrid(lon,R)
            lat   = lon.copy()*0

            # --- Annulus grid: conversion to x,y (z = 0 here)
            self.im('   - Compute the annulus')
            xan = np.multiply(np.multiply(R,np.cos(lat)),np.cos(lon))
            yan = np.multiply(np.multiply(R,np.cos(lat)),np.sin(lon))
            zan = np.multiply(R,np.sin(lat))

            # --- interpolation
            self.im('   - interpolation on the annulus')
            from scipy.interpolate import griddata
            from time import time
            points      = np.zeros((self.x.shape[0],3))
            points[:,0] = self.x
            points[:,1] = self.y
            points[:,2] = self.z
            self.im('     -> Number of data points: '+str(self.x.shape[0]))
            self.im('     -> Number of new points:  '+str(xan.shape[0]*xan.shape[1]))
            time0 = time()
            self.v = griddata(points, self.v, (xan, yan, zan), method=interp_method)
            if ftype == 'Vectorial':
                self.vx = griddata(points, self.vx, (xan, yan, zan), method=interp_method)
                self.vy = griddata(points, self.vy, (xan, yan, zan), method=interp_method)
                self.vz = griddata(points, self.vz, (xan, yan, zan), method=interp_method)
                self.vtheta = griddata(points, self.vtheta, (xan, yan, zan), method=interp_method)
                self.vphi = griddata(points, self.vphi, (xan, yan, zan), method=interp_method)
                self.vr = griddata(points, self.vr, (xan, yan, zan), method=interp_method)
            time1 = time()
            self.x = xan ; self.phi   = lon - np.pi
            self.y = yan ; self.theta = lat
            self.z = zan ; self.r     = R
            #exit
            self.im('Slicing done successfully!')
            self.im('    -> Time for the interpolation: '+str(time1-time0))
            # ----------------------------------------------
        
        if axis == 1 or axis == 'layer':
            self.im('Extraction of a depth slice from a Yin Yang (i.e. axis=1)')
            txt_layer = str(layer)
            if layer not in stagData.slayers and layer not in list(range(-1,-len(stagData.slayers)-1,-1)):
                raise StagUnknownLayerError(layer)
            else:
                layer = stagData.slayers[layer]
                layer = np.where(np.array(stagData.slayers,dtype=np.int32)==layer)[0][0]
            #Specific inheritance
            self.r1 = stagData.r1.reshape(NxNy,Nz)[:,layer]
            self.r2 = stagData.r2.reshape(NxNy,Nz)[:,layer]
            #Creation of the SliceData object:
            nod     = len(stagData.slayers)
            NxNy    = int(len(stagData.r1)/nod)
            radtemp = stagData.r1.reshape(NxNy,nod)
            rad = [np.amax(radtemp[:,i]) for i in range(nod)]
            self.layer = stagData.slayers[layer]
            self.depth = stagData.depths[layer]
            self.x1 = stagData.x1.reshape(NxNy,Nz)[:,layer]
            self.y1 = stagData.y1.reshape(NxNy,Nz)[:,layer]
            self.z1 = stagData.z1.reshape(NxNy,Nz)[:,layer]
            self.r1 = stagData.r1.reshape(NxNy,Nz)[:,layer]
            self.theta1 = stagData.theta1.reshape(NxNy,Nz)[:,layer]
            self.phi1   = stagData.phi1.reshape(NxNy,Nz)[:,layer]
            self.x2 = stagData.x2.reshape(NxNy,Nz)[:,layer]
            self.y2 = stagData.y2.reshape(NxNy,Nz)[:,layer]
            self.z2 = stagData.z2.reshape(NxNy,Nz)[:,layer]
            self.r2 = stagData.r2.reshape(NxNy,Nz)[:,layer]
            self.theta2 = stagData.theta2.reshape(NxNy,Nz)[:,layer]
            self.phi2   = stagData.phi2.reshape(NxNy,Nz)[:,layer]
            if ftype == 'Scalar':
                self.v1   = stagData.v1.reshape(NxNy,Nz)[:,layer]
                self.v2   = stagData.v2.reshape(NxNy,Nz)[:,layer]
                # empty
                self.vx1,self.vy1, self.vz1, self.vr1 = stagData.vx1,stagData.vy1,stagData.vz1,stagData.vr1
                self.vx2,self.vy2, self.vz2, self.vr2 = stagData.vx2,stagData.vy2,stagData.vz2,stagData.vr2
                self.vtheta1, self.vphi1, self.P1 = stagData.vtheta1,stagData.vphi1,stagData.P1
                self.vtheta2, self.vphi2, self.P2 = stagData.vtheta2,stagData.vphi2,stagData.P2
            else:
                self.vx1   = stagData.vx1.reshape(NxNy,Nz)[:,layer]
                self.vx2   = stagData.vx2.reshape(NxNy,Nz)[:,layer]
                self.vy1   = stagData.vy1.reshape(NxNy,Nz)[:,layer]
                self.vy2   = stagData.vy2.reshape(NxNy,Nz)[:,layer]
                self.vz1   = stagData.vz1.reshape(NxNy,Nz)[:,layer]
                self.vz2   = stagData.vz2.reshape(NxNy,Nz)[:,layer]
                self.vr1   = stagData.vr1.reshape(NxNy,Nz)[:,layer]
                self.vr2   = stagData.vr2.reshape(NxNy,Nz)[:,layer]
                self.vtheta1   = stagData.vtheta1.reshape(NxNy,Nz)[:,layer]
                self.vtheta2   = stagData.vtheta2.reshape(NxNy,Nz)[:,layer]
                self.vphi1     = stagData.vphi1.reshape(NxNy,Nz)[:,layer]
                self.vphi2     = stagData.vphi2.reshape(NxNy,Nz)[:,layer]
                self.P1   = stagData.P1.reshape(NxNy,Nz)[:,layer]
                self.P2   = stagData.P2.reshape(NxNy,Nz)[:,layer]
                self.v1   = stagData.v1.reshape(NxNy,Nz)[:,layer]  #norm
                self.v2   = stagData.v2.reshape(NxNy,Nz)[:,layer]  #norm
            #stacking
            self.stackyy(self.x1.shape[0],self.x2.shape[0],self.v1.shape[0],self.v2.shape[0])
            #exit
            self.im('Extraction done successfully!')
            self.im('    - layer        = '+txt_layer)
            self.im('    - pts in slice = '+str(NxNy))
    
    
    def locate_on_annulus_slicing(self,stagData,point,normal):
        """
        Extract an annulus-slice or a depth-slice in a stagData.StagYinYangGeometry object.
        The annulus-slice is defined according to a normal vector, perpendicular to the slicing plan.
        The detph-slice of defined according to the depth layer.
        <i> : stagData = stagData.StagYinYangGeometry
              normal   = list/array, (only if axis == 0), vector of coordinates corresponding
                         to the normal to the plan containing the annlus.
                         This definition is consistent with the normal of the slicing plan in the Paraview software!
                         normal = (nx,ny,nz)
                         Default: normal = (1,0,0)
        """
        self.im('Search the location of a point on an annulus slice')
        lon,lat = point
        #check the geometry:
        if not isinstance(stagData,StagYinYangGeometry):
            raise StagTypeError(str(type(stagData)),'stagData.StagYinYangGeometry')
        # ---
        self.im('Normal to the slicing plan: '+str(normal[0])+','+str(normal[1])+','+str(normal[2]))
        self.normal = normal
        small = 1e-10           # to avoid to divide by 0
        normal[0] += small      #
        normal[1] += small      #
        normal[2] += small      #
        a = normal[0]
        b = normal[1]
        c = normal[2]
        # Plan equation:
        from .stagCompute import get_xzy_scoords
        xp,yp,zp = get_xzy_scoords(stagData,lon,lat,verbose=self.verbose)
        # test if on the plan:
        R = np.sqrt((4*np.pi*2.19**2)/(self.nx*self.ny))
        self.im('   Search on the plan')
        if abs(a*xp+b*yp+c*zp) > R:
            msg = 'Points not on the annulus plan: unable to locate '
            raise StagComputationalError(msg)
        
        # Compute the normal vectors to the slicing plan:
        self.normalu = np.array([1,-a/b,0])
        self.normalv = np.array([a/b,1,-(a**2+b**2)/(c*b)])
        self.normalw = np.array([a,b,c])
        # Projection
        self.im('1. Projection on the plan')
        x = np.dot(np.array([xp,yp,zp]).T,self.normalu)/np.linalg.norm(self.normalu)
        y = np.dot(np.array([xp,yp,zp]).T,self.normalv)/np.linalg.norm(self.normalv)
        z = np.dot(np.array([xp,yp,zp]).T,self.normalw)/np.linalg.norm(self.normalw)            
        # ---
        
        self.im('2. Find the point on the regular annulus grid')
        # --- Search the ideal number of points along the annulus
        self.im('   - build ideal annulus geometry')
        totSurf     = np.count_nonzero(stagData.layers == stagData.slayers[0])*2
        opti_radius = np.sqrt(totSurf/(4*np.pi))
        opti_peri   = int(2*np.pi*opti_radius)
        opti_peri   = opti_peri * 5 # optimization
        # --- Description of the new grid: the two axis
        lon = np.linspace(0,2*np.pi,opti_peri)
        R   = np.array(stagData.z_coords)+stagData.rcmb
        # --- meshed grid
        lon,R = np.meshgrid(lon,R)
        lat   = lon.copy()*0
        # --- Annulus grid: conversion to x,y (z = 0 here)
        self.im('   - Compute the annulus')
        xan = np.multiply(np.multiply(R,np.cos(lat)),np.cos(lon))
        yan = np.multiply(np.multiply(R,np.cos(lat)),np.sin(lon))
        zan = np.multiply(R,np.sin(lat))
        # --- Find the nearest point
        dist = np.sqrt((xan.flatten()-x)**2+(yan.flatten()-y)**2+(zan.flatten()-z)**2)
        gind = np.where(dist == np.amin(dist))[0][0]
        self.im('Point found!')
        xf   = xan.flatten()[gind]
        yf   = yan.flatten()[gind]
        zf   = zan.flatten()[gind]
        lonf = lon.flatten()[gind] - np.pi
        latf = lat.flatten()[gind]
        rf   = R.flatten()[gind]
        self.im('   x   = '+str(xf))
        self.im('   y   = '+str(yf))
        self.im('   z   = '+str(zf))
        self.im('   lon = '+str(lonf))
        self.im('   lat = '+str(latf))
        self.im('   r   = '+str(rf))
        return xf,yf,zf,lonf,latf,rf

    




class CartesianSliceData(MainSliceData):
    """
    Defines the structure of the CartesianSliceData object derived from MainSliceData type.
    This object corresponds to a simplified StagCartesianGeometry object.
    """
    def __init__(self,geometry):
        super().__init__()  # inherit all the methods and properties from MainSliceData
        self.geometry = geometry
        self.plan  = None  # If cartesian 2D, contains information of the plan as 'xy' or 'yz etc
                           # self.plan != self.axis, self.plan describe plan where data are contained
        # ----- Cartesian 2D and 3D geometries ----- #
        self.x = np.array([])         #Matrix of X coordinates meshed
        self.y = np.array([])         #Matrix of Y coordinates meshed
        self.z = np.array([])         #Matrix of Z coordinates meshed
        # Scalar field
        self.v = np.array([])         #Matrix of scalar field (or norm of velocity)
        # vectorial field
        self.vx = np.array([])        #Matrix of x-component of the velocity field for Cartesian grids
        self.vy = np.array([])        #Matrix of y-component of the velocity field for Cartesian grids
        self.vz = np.array([])        #Matrix of z-component of the velocity field for Cartesian grids
        self.P  = np.array([])        #Matrix of Pressure field for Cartesian grids
    

    def slicing(self,stagData,axis=1,normal=[1,0,0],layer=-1,interp_method='nearest'):
        """
        Extract a depth slice in a stagData.StagCartesianGeometry object.
        As stagData.StagCartesianGeometry can have two different geometry, the slicing
        will conserve the duality of this object. The slicing will reduce by 1 the dimension
        of the grid.
          - For a cart3D geometry, the slicing will produce an iso-depth plan at the depth 'layer'.
          - For a cart2D geometry, the slicing will produce an iso-depth line at the depth 'layer' 
            if the original geometry was 'xz' or 'yz'.In the case of an original 'xy' geometry,
            the slicing will extract a line according the xy_axis.
            e.g. sliceExtractor(stagData,3,xy_axis=1) will make a slice on the cart2D stagData
            on axis 'y' (xy_axis=1), i.e. slice it on y=layer.
        <i> : stagData = stagData.StagCartesianGeometry
              axis = int, slice direction (defaut axis=1) according to:
                    axis = 0    -> slice according to a plan          --| Spherical slicing
                                   perpendicular to the given normal
                    axis = 1 or axis = 'x'  -> slice on x     (x=layer,y=:,z=:)  --|
                    axis = 2 or axis = 'y'  -> slice on y     (x=:,y=layer,z=:)    |-  Cartesian slicing
                    axis = 3 or axis = 'z'  -> slice on z     (x=:,y=:,z=layer)  --|

              normal   = list/array, (only if axis == 0), vector of coordinates corresponding
                         to the normal to the plan containing the annlus (partial).
                         normal = (nx,ny,nz)
                         This definition is consistent with the normal of the slicing plan in the Paraview software!
                         Default: normal = [1,0,0]
              layer    = int, (only if axis >= 1), index of the stagData layer that will be extracted in
                         the new SliceData object.
                         Default: layer = -1
        """
        self.im('Begin the slice extraction')
        self.axis  = axis
        self.layer = layer
        # -------
        #check the type:
        if not isinstance(stagData,StagCartesianGeometry):
            raise StagTypeError(str(type(stagData)),'stagData.StagCartesianGeometry')
        # field type:
        ftype = stagData.fieldNature
        if ftype == 'Scalar':
            self.im('  - Scalar field detected')
        else:
            self.im('  - Vectorial field detected: '+str(stagData.flds.shape[0])+' fields')
        # -------
        #StagInheritance
        self.sliceInheritance(stagData)
        # -------
        #Slicing
        # ================== cart3D ==================
        if stagData.geometry == 'cart3D':
            if axis == 1 or axis == 'x':
                self.im('  - Slicing on axis 0, slice on x')
                self.x = stagData.x[layer,:,:]
                self.y = stagData.y[layer,:,:]
                self.z = stagData.z[layer,:,:]
                if ftype == 'Scalar':
                    self.v = stagData.v[layer,:,:]
                else:
                    self.vx = stagData.vx[layer,:,:]
                    self.vy = stagData.vy[layer,:,:]
                    self.vz = stagData.vz[layer,:,:]
                    self.v = stagData.v[layer,:,:]
                    self.P = stagData.P[layer,:,:]
            elif axis == 2 or axis == 'y':
                self.im('  - Slicing on axis 1, slice on y')
                self.x = stagData.x[:,layer,:]
                self.y = stagData.y[:,layer,:]
                self.z = stagData.z[:,layer,:]
                if ftype == 'Scalar':
                    self.v = stagData.v[:,layer,:]
                else:
                    self.vx = stagData.vx[:,layer,:]
                    self.vy = stagData.vy[:,layer,:]
                    self.vz = stagData.vz[:,layer,:]
                    self.v = stagData.v[:,layer,:]
                    self.P = stagData.P[:,layer,:]
            elif axis == 3 or axis == 'z':
                self.im('  - Slicing on axis 2, slice on z')
                self.x = stagData.x[:,:,layer]
                self.y = stagData.y[:,:,layer]
                self.z = stagData.z[:,:,layer]
                if ftype == 'Scalar':
                    self.v = stagData.v[:,:,layer]
                else:
                    self.vx = stagData.vx[:,:,layer]
                    self.vy = stagData.vy[:,:,layer]
                    self.vz = stagData.vz[:,:,layer]
                    self.v = stagData.v[:,:,layer]
                    self.P = stagData.P[:,:,layer]
            elif axis == 0:
                self.im('Extraction of a slice along a plan defined by its normal (i.e. axis=0)')
                self.im('   Normal to the slicing plan: '+str(normal[0])+','+str(normal[1])+','+str(normal[2]))
                small = 1e-10           # to avoid to divide by 0
                normal[0] += small      #
                normal[1] += small      #
                normal[2] += small      #
                a = normal[0]
                b = normal[1]
                c = normal[2]
                # Compute the thickness of the slice:
                R = np.sqrt((4*np.pi*2.19**2)/(self.nx*self.ny))
                # Plan equation:
                self.im('   Search on the plan')
                sdx = stagData.x.flatten()
                sdy = stagData.y.flatten()
                sdz = stagData.z.flatten()
                # center the grid on 0,0,0
                sdx = sdx - (np.amax(sdx)-np.amin(sdx))/2
                sdy = sdy - (np.amax(sdy)-np.amin(sdy))/2
                sdz = sdz - (np.amax(sdz)-np.amin(sdz))/2
                #
                gind = np.where(abs(a*sdx+b*sdy+c*sdz) <= R)[0]
                if len(gind) == 0:
                    self.im('******************************')
                    self.im('*** WARNING ***')
                    self.im('** -> No points crossing the slicing plan:')
                    self.im('**    Maybe you have to reconsider the input normal vector')
                    self.im('******************************')
                # Cut YY according to plan equation
                x,y,z = sdx[gind],sdy[gind],sdz[gind]
                #self.xc,self.yc,self.zc = stagData.xc.flatten()[gind],stagData.yc.flatten()[gind],stagData.zc.flatten()[gind]
                if ftype == 'Scalar':
                    self.v = stagData.v.flatten()[gind]
                    # empty
                    self.vx,self.vy, self.vz = np.array([]),np.array([]),np.array([])
                    self.P = np.array([])
                else:
                    self.v = stagData.v.flatten()[gind] # norm
                    self.vx,self.vy,self.vz = stagData.vx.flatten()[gind],stagData.vy.flatten()[gind],stagData.vz.flatten()[gind]
                    self.P    = stagData.P.flatten()[gind]
                # Compute the normal vectors to the slicing plan:
                self.normalu = np.array([1,-a/b,0])
                self.normalv = np.array([a/b,1,-(a**2+b**2)/(c*b)])
                self.normalw = np.array([a,b,c])
                # Projection
                self.im('   Projection on the plan')
                self.x = np.dot(np.array([x,y,z]).T,self.normalu)/np.linalg.norm(self.normalu)
                self.y = np.dot(np.array([x,y,z]).T,self.normalv)/np.linalg.norm(self.normalv)
                self.z = np.dot(np.array([x,y,z]).T,self.normalw)/np.linalg.norm(self.normalw)
                # Interpolate the slice on a new plan
                # ----------------------------------------------
                # --- Search the ideal number of points for the new grid
                self.im('   - build a new cart2D geometry')

                # --- Description of the new grid: the two axis
                xnew = np.linspace(np.amin(self.x),np.amax(self.x),stagData.nx)
                ynew = np.linspace(np.amin(self.y),np.amax(self.y),stagData.ny)

                # --- meshed grid
                xnew,ynew = np.meshgrid(xnew,ynew)
                znew      = xnew.copy()*0

                # --- interpolation
                self.im('   - interpolation on the grid')
                from scipy.interpolate import griddata
                from time import time
                points      = np.zeros((self.x.shape[0],3))
                points[:,0] = self.x
                points[:,1] = self.y
                points[:,2] = self.z
                self.im('     -> Number of data points: '+str(self.x.shape[0]))
                self.im('     -> Number of new points:  '+str(xnew.shape[0]*xnew.shape[1]))
                time0 = time()
                self.v = griddata(points, self.v, (xnew, ynew, znew), method=interp_method)
                if ftype == 'Vectorial':
                    # projection
                    self.vx   = np.dot(np.array([self.vx,self.vy,self.vz]).T,self.normalu)/np.linalg.norm(self.normalu)
                    self.vy   = -np.dot(np.array([self.vx,self.vy,self.vz]).T,self.normalv)/np.linalg.norm(self.normalv) # minus because, plot with ax.invert_yaxis()
                    self.vz   = np.dot(np.array([self.vx,self.vy,self.vz]).T,self.normalv)/np.linalg.norm(self.normalv)
                    # interpolation
                    self.vx = griddata(points, self.vx, (xnew, ynew, znew), method=interp_method)
                    self.vy = griddata(points, self.vy, (xnew, ynew, znew), method=interp_method)
                    self.vz = griddata(points, self.vz, (xnew, ynew, znew), method=interp_method)
                time1 = time()
                # Mask to remove the very distant points
                self.im('   - remove ghost points')
                distmin = np.sqrt((xnew[0][0]-xnew[1][1])**2+(ynew[0][0]-ynew[1][1])**2)
                mask    = np.zeros(self.v.shape,dtype=bool)
                for i in range(len(xnew)):
                    for j in range(len(xnew[i])):
                        dist = np.sqrt((xnew[i][j]-self.x)**2+(ynew[i][j]-self.y)**2)
                        # dist min
                        if np.amin(dist) >= distmin:
                            mask[i][j] = True
                # apply the mask:
                self.v[mask] = np.nan
                if ftype == 'Vectorial':
                    self.vx[mask] = np.nan
                    self.vy[mask] = np.nan
                    self.vz[mask] = np.nan
                self.x = xnew
                self.y = ynew
                self.z = znew
                #exit
                self.im('Slicing done successfully!')
                self.im('    -> Time for the interpolation: '+str(time1-time0))
                # ----------------------------------------------
                
            else:
                raise SliceAxisError(axis)
        # ================== cart2D ==================
        elif stagData.geometry == 'cart2D':
            #Conditioning the 2D/3D geometry problem:
            # --------------------
            if stagData.x.shape[0] == 1:
                self.im('  - 2D data detected: plan yz')
                self.plan = 'yz'
                if axis == 0:
                    raise IncoherentSliceAxisError(axis)
                elif axis == 1:
                    self.im('  - Slicing on axis 1, slice on y')
                    self.y = stagData.y[0,layer,:]
                    self.z = stagData.z[0,layer,:]
                    if ftype == 'Scalar':
                        self.v = stagData.v[0,layer,:]
                    else:
                        self.vy = stagData.vy[0,layer,:]
                        self.vz = stagData.vz[0,layer,:]
                        self.v = stagData.v[0,layer,:]
                        self.P = stagData.P[0,layer,:]
                elif axis == 2:
                    self.im('  - Slicing on axis 2, slice on z')
                    self.y = stagData.y[0,:,layer]
                    self.z = stagData.z[0,:,layer]
                    if ftype == 'Scalar':
                        self.v = stagData.v[0,:,layer]
                    else:
                        self.vy = stagData.vy[0,:,layer]
                        self.vz = stagData.vz[0,:,layer]
                        self.v = stagData.v[0,:,layer]
                        self.P = stagData.P[0,:,layer]
                else:
                    raise SliceAxisError(axis)
            # --------------------
            elif stagData.x.shape[1] == 1:
                self.im('  - 2D data detected: plan xz')
                self.plan = 'xz'
                if axis == 0:
                    self.im('  - Slicing on axis 0, slice on x')
                    self.x = stagData.x[layer,0,:]
                    self.z = stagData.z[layer,0,:]
                    if ftype == 'Scalar':
                        self.v = stagData.v[layer,0,:]
                    else:
                        self.vx = stagData.vx[layer,0,:]
                        self.vz = stagData.vz[layer,0,:]
                        self.v = stagData.v[layer,0,:]
                        self.P = stagData.P[layer,0,:]
                elif axis == 1:
                    raise IncoherentSliceAxisError(axis)
                elif axis == 2:
                    self.im('  - Slicing on axis 2, slice on z')
                    self.x = stagData.x[:,0,layer]
                    self.z = stagData.z[:,0,layer]
                    if ftype == 'Scalar':
                        self.v = stagData.v[:,0,layer]
                    else:
                        self.vx = stagData.vx[:,0,layer]
                        self.vz = stagData.vz[:,0,layer]
                        self.v = stagData.v[:,0,layer]
                        self.P = stagData.P[:,0,layer]
                else:
                    raise SliceAxisError(axis)
            # --------------------
            elif stagData.x.shape[2] == 1:
                self.im('  - 2D data detected: plan xy')
                self.plan = 'xy'
                if axis == 0:
                    self.im('  - Slicing on axis 0, slice on x')
                    self.x = stagData.x[layer,:,0]
                    self.y = stagData.y[layer,:,0]
                    if ftype == 'Scalar':
                        self.v = stagData.v[layer,:,0]
                    else:
                        self.vx = stagData.vx[layer,:,0]
                        self.vy = stagData.vy[layer,:,0]
                        self.v = stagData.v[layer,:,0]
                        self.P = stagData.P[layer,:,0]
                elif axis == 1:
                    self.im('  - Slicing on axis 1, slice on y')
                    self.x = stagData.x[:,layer,0]
                    self.y = stagData.y[:,layer,0]
                    if ftype == 'Scalar':
                        self.v = stagData.v[:,layer,0]
                    else:
                        self.vx = stagData.vx[:,layer,0]
                        self.vy = stagData.vy[:,layer,0]
                        self.v = stagData.v[:,layer,0]
                        self.P = stagData.P[:,layer,0]
                elif axis == 2:
                    IncoherentSliceAxisError(axis)
                else:
                    raise SliceAxisError(axis)
    
    def dimExpand(self,axis):
        """
        This function expands the dimensions of internal grid and fields
        matrices according to a given axis index.
        <i> : axis = int, index of dimension that have to be expanded
                        axis = 0    -> expand on x
                        axis = 1    -> expand on y
                        axis = 2    -> expand on z
        """
        self.x  = np.expand_dims(self.x,axis=axis)
        self.y  = np.expand_dims(self.y,axis=axis)
        self.z  = np.expand_dims(self.z,axis=axis)
        self.v  = np.expand_dims(self.v,axis=axis)
        if self.fieldNature == 'Vectorial':
            self.vx = np.expand_dims(self.vx,axis=axis)
            self.vy = np.expand_dims(self.vy,axis=axis)
            self.vz = np.expand_dims(self.vz,axis=axis)
            self.P  = np.expand_dims(self.P,axis=axis)





class SphericalSliceData(MainSliceData):
    """
    Defines the structure of the SphericalSliceData object derived from MainSliceData type.
    This object corresponds to a simplified StagSphericalGeometry object.
    """
    def __init__(self,geometry):
        super().__init__()  # inherit all the methods and properties from MainSliceData
        self.geometry = geometry
        self.plan  = None  # If cartesian 2D, contains information of the plan as 'xy' or 'yz etc
                           # self.plan != self.axis, self.plan describe plan where data are contained
        self.x  = np.array([])        #Matrix of X coordinates meshed (in spherical shape)
        self.y  = np.array([])        #Matrix of Y coordinates meshed (in spherical shape)
        self.z  = np.array([])        #Matrix of Z coordinates meshed (in spherical shape)
        self.xc = np.array([])        #Matrix of cartesian X coordinates meshed (in cartesian shape)
        self.yc = np.array([])        #Matrix of cartesian Y coordinates meshed (in cartesian shape)
        self.zc = np.array([])        #Matrix of cartesian Z coordinates meshed (in cartesian shape)
        self.r     = np.array([])     #Matrice of spherical coordinates r
        self.theta = np.array([])     #Matrice of spherical coordinates theta
        self.phi   = np.array([])     #Matrice of spherical coordinates phi
        # Scalar field
        self.v = np.array([])         #Matrix of scalar field (or norm of velocity)
        # vectorial field
        self.vx = np.array([])        #Matrix of x-component of the velocity field for Cartesian grids
        self.vy = np.array([])        #Matrix of y-component of the velocity field for Cartesian grids
        self.vz = np.array([])        #Matrix of z-component of the velocity field for Cartesian grids
        self.vtheta = np.array([])    #Matrix of theta component of the vectorial field
        self.vphi   = np.array([])    #Matrix of phi component of the vectorial field
        self.vr     = np.array([])    #Matrix of radial component of the vectorial field
        self.P  = np.array([])        #Matrix of Pressure field for Cartesian grids



    def slicing(self,stagData,axis=1,normal=[1,0,0],layer=-1):
        """
        Extract an annulus-slice or a depth-slice in a stagData.StagSphericalGeometry object.
        The annulus-slice is defined according to a normal vector, perpendicular to the slicing plan.
        The detph-slice of defined according to an axis (x,y or z) and the index of the layer.
        NOTE: Note that this slicing routine works for both 3D spherical and annulus geometry except
              for an annulus-slice! Indeed, an annulus-slice of an annulus will just produce
              itself (for the best) but generally 2 disconected lines.
              In the depth-slice case, a slice on an annulus will return a line.
        In the case of a depth-slice, the slicing will reduce by 1 the dimension of the grid.
        <i> : stagData = stagData.StagSphericalGeometry
              axis = int, slice direction (defaut axis=1) according to:
                    axis = 0 or axis = 'annulus -> slice according to a plan          --| Spherical slicing
                                                   perpendicular to the given normal
                    axis = 1 or axis = 'x'  -> slice on x     (x=layer,y=:,z=:)  --|
                    axis = 2 or axis = 'y'  -> slice on y     (x=:,y=layer,z=:)    |-  Cartesian slicing
                    axis = 3 or axis = 'z'  -> slice on z     (x=:,y=:,z=layer)  --|

              normal   = list/array, (only if axis == 0), vector of coordinates corresponding
                         to the normal to the plan containing the annlus (partial).
                         normal = (nx,ny,nz)
                         This definition is consistent with the normal of the slicing plan in the Paraview software!
                         Default: normal = [1,0,0]
              layer    = int, (only if axis >= 1), index of the stagData layer that will be extracted in
                         the new SliceData object.
        """
        self.im('Begin the slice extraction')
        self.axis  = axis
        self.layer = layer
        # -------
        #check the type:
        if not isinstance(stagData,StagSphericalGeometry):
            raise StagTypeError(str(type(stagData)),'stagData.StagSphericalGeometry')
        # field type:
        ftype = stagData.fieldNature
        if ftype == 'Scalar':
            self.im('  - Scalar field detected')
        else:
            self.im('  - Vectorial field detected: '+str(stagData.flds.shape[0])+' fields')
        # -------
        #StagInheritance
        self.sliceInheritance(stagData)
        # -------
        #Annulus plan
        if self.geometry == 'annulus':
            if stagData.x.shape[0] == 1:
                self.im('  - Annulus data detected: plan yz')
                self.plan = 'yz'
            elif stagData.x.shape[1] == 1:
                self.im('  - Annulus data detected: plan xz')
                self.plan = 'xz'
            elif stagData.x.shape[2] == 1:
                self.im('  - Annulus data detected: plan xy')
                self.plan = 'xy'
        # -------
        #Slicing
        if axis == 0 or axis == 'annulus':
            self.im('Extraction of an annulus slice (i.e. axis=0)')
            self.im('   Normal to the slicing plan: '+str(normal[0])+','+str(normal[1])+','+str(normal[2]))
            small = 1e-10           # to avoid to divide by 0
            normal[0] += small      #
            normal[1] += small      #
            normal[2] += small      #
            a = normal[0]
            b = normal[1]
            c = normal[2]
            # Compute the thickness of the slice:
            R = np.sqrt((4*np.pi*2.19**2)/(self.nx*self.ny))
            # Plan equation:
            self.im('   Search on the plan')
            sdx = stagData.x.flatten()
            sdy = stagData.y.flatten()
            sdz = stagData.z.flatten()
            gind = np.where(abs(a*sdx+b*sdy+c*sdz) <= R)[0]
            if len(gind) == 0:
                self.im('******************************')
                self.im('*** WARNING ***')
                self.im('** -> No points crossing the slicing plan:')
                self.im('**    Maybe you have to reconsider the input normal vector')
                self.im('******************************')
            # Cut YY according to plan equation
            x,y,z,self.r = sdx[gind],sdy[gind],sdz[gind],stagData.r.flatten()[gind]
            self.theta,self.phi = stagData.theta.flatten()[gind],stagData.phi.flatten()[gind]
            self.xc,self.yc,self.zc = stagData.xc   .flatten()[gind],stagData.yc.flatten()[gind],stagData.zc.flatten()[gind]
            if ftype == 'Scalar':
                self.v = stagData.v.flatten()[gind]
                # empty
                self.vx,self.vy, self.vz, self.vr = np.array([]),np.array([]),np.array([]),np.array([])
                self.vtheta, self.vphi, self.P = np.array([]),np.array([]),np.array([])
            else:
                self.v = stagData.v.flatten()[gind] # norm
                self.vx,self.vy,self.vz,self.vr = stagData.vx.flatten()[gind],stagData.vy.flatten()[gind],stagData.vz.flatten()[gind],stagData.vr.flatten()[gind]
                self.vtheta,self.vphi,self.P    = stagData.vtheta.flatten()[gind],stagData.vphi.flatten()[gind],stagData.P.flatten()[gind]
            # Compute the normal vectors to the slicing plan:
            self.normalu = np.array([1,-a/b,0])
            self.normalv = np.array([a/b,1,-(a**2+b**2)/(c*b)])
            self.normalw = np.array([a,b,c])
            # Projection
            self.im('   Projection on the plan')
            self.x = np.dot(np.array([x,y,z]).T,self.normalu)/np.linalg.norm(self.normalu)
            self.y = np.dot(np.array([x,y,z]).T,self.normalv)/np.linalg.norm(self.normalv)
            self.z = np.dot(np.array([x,y,z]).T,self.normalw)/np.linalg.norm(self.normalw)
    
        elif axis == 1 or axis == 'x':
            self.im('  - Slicing on axis 1, slice on x')
            self.x  = stagData.x[layer,:,:]
            self.y  = stagData.y[layer,:,:]
            self.z  = stagData.z[layer,:,:]
            self.xc = stagData.xc[layer,:,:]
            self.yc = stagData.yc[layer,:,:]
            self.zc = stagData.zc[layer,:,:]
            self.r     = stagData.r[layer,:,:]
            self.theta = stagData.theta[layer,:,:]
            self.phi   = stagData.phi[layer,:,:]
            if ftype == 'Scalar':
                self.v = stagData.v[layer,:,:]
            else:
                self.vx = stagData.vx[layer,:,:]
                self.vy = stagData.vy[layer,:,:]
                self.vz = stagData.vz[layer,:,:]
                self.v = stagData.v[layer,:,:]
                self.P = stagData.P[layer,:,:]
                self.vtheta = stagData.vtheta[layer,:,:]
                self.vphi   = stagData.vphi[layer,:,:]
                self.vr     = stagData.vr[layer,:,:]
        elif axis == 2 or axis == 'y':
            self.im('  - Slicing on axis 2, slice on y')
            self.x  = stagData.x[:,layer,:]
            self.y  = stagData.y[:,layer,:]
            self.z  = stagData.z[:,layer,:]
            self.xc = stagData.xc[:,layer,:]
            self.yc = stagData.yc[:,layer,:]
            self.zc = stagData.zc[:,layer,:]
            self.r     = stagData.r[:,layer,:]
            self.theta = stagData.theta[:,layer,:]
            self.phi   = stagData.phi[:,layer,:]
            if ftype == 'Scalar':
                self.v = stagData.v[:,layer,:]
            else:
                self.vx = stagData.vx[:,layer,:]
                self.vy = stagData.vy[:,layer,:]
                self.vz = stagData.vz[:,layer,:]
                self.v = stagData.v[:,layer,:]
                self.P = stagData.P[:,layer,:]
                self.vtheta = stagData.vtheta[:,layer,:]
                self.vphi   = stagData.vphi[:,layer,:]
                self.vr     = stagData.vr[:,layer,:]
        elif axis == 3 or axis == 'z':
            self.im('  - Slicing on axis 3, slice on z')
            self.x  = stagData.x[:,:,layer]
            self.y  = stagData.y[:,:,layer]
            self.z  = stagData.z[:,:,layer]
            self.xc = stagData.xc[:,:,layer]
            self.yc = stagData.yc[:,:,layer]
            self.zc = stagData.zc[:,:,layer]
            self.r     = stagData.r[:,:,layer]
            self.theta = stagData.theta[:,:,layer]
            self.phi   = stagData.phi[:,:,layer]
            if ftype == 'Scalar':
                self.v = stagData.v[:,:,layer]
            else:
                self.vx = stagData.vx[:,:,layer]
                self.vy = stagData.vy[:,:,layer]
                self.vz = stagData.vz[:,:,layer]
                self.v = stagData.v[:,:,layer]
                self.P = stagData.P[:,:,layer]
                self.vtheta = stagData.vtheta[:,:,layer]
                self.vphi   = stagData.vphi[:,:,layer]
                self.vr     = stagData.vr[:,:,layer]
        else:
            raise SliceAxisError(axis)
        # -------
        #Squeezing and sort: remove useless dimension
        if self.geometry == 'annulus' and axis in [1,2,3]:
            if self.plan == 'yz' or self.plan == 'zy':
                self.im('  - Sorting grid accroding to phi direction')
                self.phi   = np.squeeze(self.phi)
                self.theta = np.squeeze(self.theta)
                sort_list = np.argsort(self.phi)
            elif self.plan == 'xz' or self.plan == 'zx':
                self.im('  - Sorting grid accroding to theta direction')
                self.theta = np.squeeze(self.theta)
                self.phi   = np.squeeze(self.phi)
                sort_list = np.argsort(self.theta)
            else:
                print('WARRNING - Strange plan for annulus...')
            # apply sorting
            self.theta = self.theta[sort_list]
            self.phi   = self.phi[sort_list]
            self.x  = np.squeeze(self.x)[sort_list]
            self.y  = np.squeeze(self.y)[sort_list]
            self.z  = np.squeeze(self.z)[sort_list]
            self.xc = np.squeeze(self.xc)[sort_list]
            self.yc = np.squeeze(self.yc)[sort_list]
            self.zc = np.squeeze(self.zc)[sort_list]
            self.r     = np.squeeze(self.r)[sort_list]
            if ftype == 'Scalar':
                self.v = np.squeeze(self.v)[sort_list]
            else:
                self.vx = np.squeeze(self.vx)[sort_list]
                self.vy = np.squeeze(self.vy)[sort_list]
                self.vz = np.squeeze(self.vz)[sort_list]
                self.v = np.squeeze(self.v)[sort_list]
                self.P = np.squeeze(self.P)[sort_list]
                self.vtheta = np.squeeze(self.vtheta)[sort_list]
                self.vphi   = np.squeeze(self.vphi)[sort_list]
                self.vr     = np.squeeze(self.vr)[sort_list]
        #exit
        self.im('Slicing done successfully!')



class SliceData():
    """
    Defines the StagData structure dynamically from geometry of the grid
    """
    def __new__(cls,geometry='cart3D'):
        """
        Force to have more than just 'duck typing' in Python: 'dynamical typing'
        <i> : geometry = str, geometry of the grid. Must be in ('cart2D',
                         'cart3D','yy','annulus') for cartesian 2D, 3D,
                         Yin-Yang or annulus geometry, respectively. By
                         default, geometry = 'cart3D'
        """
        if geometry == 'yy':
            return YinYangSliceData()
        elif geometry == 'cart2D' or geometry == 'cart3D':
            return CartesianSliceData(geometry)
        elif geometry == 'spherical':
            return SphericalSliceData(geometry)
        elif geometry == 'annulus':
            #raise GridGeometryInDevError(geometry)
            return SphericalSliceData(geometry)
        else:
            raise InputGridGeometryError(geometry)









class InterpolatedSliceData(MainSliceData):
    """
    Defines the structure of the InterpolatedSliceData object derived from MainSliceData type.
    This object corresponds to a simplified StagCartesianGeometry object.
    """
    def __init__(self):
        super().__init__()  # inherit all the methods and properties from MainSliceData
        # geometry:
        self.geom = ''      # Will be: 'rsg' ...
        # cartesian grid
        self.x = []         #Matrix of X coordinates meshed
        self.y = []         #Matrix of Y coordinates meshed
        self.z = []         #Matrix of Z coordinates meshed
        # natural geographical grid
        self.lon = []       #Matrix of Longitude coordinates (in deg)
        self.lat = []       #Matrix of Latitude coordinates (in deg)
        self.r   = []       #Matrix of Depth/Elevation coordinates
        self.theta = []
        self.phi   = []
        # Scalar field
        self.v = []         #Matrix of scalar field
        # Vectorial field
        self.vx = []        #Matrix of x-component of the velocity field for Cartesian grids
        self.vy = []        #Matrix of y-component of the velocity field for Cartesian grids
        self.vz = []        #Matrix of z-component of the velocity field for Cartesian grids
        self.vtheta = []    #Matrix of theta component of the velocity field
        self.vphi   = []    #Matrix of phi component of the velocity field
        self.vr     = []    #Matrix of radial component of the velocity field
        self.P  = []        #Matrix of Pressure field for Cartesian grids
        #
        # New geometry
        self.geometry = 'interpolated'
        self.nxi = 0        #Number of point in the x direction in the inteprolated grid
        self.nyi = 0        #Number of point in the y direction in the inteprolated grid
        self.nzi = 0        #Number of point in the z direction in the inteprolated grid
        # interpolation parameters:
        self.interpGeom   = None          # Defined during the interpolation: indicates the type of geometry used for the new grid
        self.spacing      = None          # Defined during the interpolation: parameter of the interpGeom
        self.innerradius  = None          # Defined during the interpolation: CMB radius
        self.outerradius  = None          # Defined during the interpolation: surface radius
        self.ntheta       = None          # Defined during the interpolation: number of points in the theta direction
        self.nz           = None          # Defined during the interpolation: number of points in the z (r) direction
        self.interpMethod = None          # Defined during the interpolation: method used for the interpolation

    
    def convert_to_stagData(self):
        """
        Return a stagData object with an annulus geometry
        """
        if self.interpGeom == 'rgA':
            # Creat the object
            sd = StagData(geometry='annulus')
            sd.geometry = 'annulus'
            sd.plan     = 'xy'
            sd.nx0 = 0   # Mean that the stagData object come from a convertion or is handmade
            sd.nx = self.nx
            sd.ny = self.ny
            sd.nz = self.nz
            sd.x  = np.expand_dims(self.x,2)
            sd.y  = np.expand_dims(self.y,2)
            sd.z  = np.expand_dims(self.z,2)
            sd.xc = []
            sd.yc = []
            sd.zc = []
            sd.r     = np.expand_dims(self.r,2)
            sd.theta = np.expand_dims(self.theta,2)
            sd.phi   = np.expand_dims(self.phi,2)
            sd.v  = np.expand_dims(self.v,2)
            if self.fieldNature == 'Vectorial':
                sd.vx = np.expand_dims(self.vx,2)
                sd.vy = np.expand_dims(self.vy,2)
                sd.vz = np.expand_dims(self.vz,2)
                sd.vtheta = np.expand_dims(self.vtheta,2)
                sd.vphi   = np.expand_dims(self.vphi,2)
                sd.vr     = np.expand_dims(self.vr,2)
                sd.P  = np.expand_dims(self.P,2)
        return sd






class MainCouldStagData:
    """
    Main class defining the highest level of inheritance
    for StagCloudData derived object
    """
    def __init__(self):
        """
        Parent builder
        """
        # ----- Generic ----- #
        self.pName = 'cloudStagData'
        self.verbose  = True           #Condition on the verbose output
        self.path     = ''              #The path to the stag file
        self.gfname   = ''              #Generic File name of the stag file
        self.cfname   = ''              #Current file name of the stag file (open on a drop)
        self.geometry = ''              #Geometry of the Stag grid
        self.indices  = None            #List of all indicies
        self.ibegin   = None            #First index
        self.iend     = None            #Last index
        self.istep    = None            #File index step
        self.nt       = 0               #Length of self.indicies
        self.ci       = -1              #Current index of self.indicies load on the cloud
                                        #if self.ci = -1, mean that nothing is contain in the cloud drop
        # ----- Data description ----- #
        self.fieldType   = 'Undefined'  #Field contained in the current object
        self.fieldNature = 'Undefined'  #Nature of the field: Scalar or Vectorial
        self.resampling = [] #Resampling Parameters
        self.simuAge = []    #Dimensionless age of the simulation
        self.ti_step = []    #Inner step of the stag simualtion state
        # ----- Data description ----- #
        self.drop = None     #An instance of cloud (like a rain drop)
                             #self.drop will have a type derived from MainStagData
        # Other
        self.BIN = None
        self.bin = None
    

    def im(self,textMessage):
        """Print verbose internal message. This function depends on the
        argument of self.verbose. If self.verbose == True then the message
        will be displayed on the terminal.
        <i> : textMessage = str, message to display
        """
        if self.verbose == True:
            print('>> '+self.pName+'| '+textMessage)
    
    
    def __intstringer(self,iint,strl):
        """
        --- Internal function ---
        This function transforms an input int 'iint' into a str format according
        to a string length 'strl' (condition: strl >= len(str(iint))  ).
        e.g. >> intstringer(4000,5)
            << '04000'
        <i>: iint = int, input integer you want to transform into a string
            strl = int, length of the output string
        <o>: ostr = str, output string
        """
        class BaseError(Exception):
            """Base class for exceptions raised"""
            pass
        class intstringError(BaseError):
            def __init__(self):
                super().__init__('the length of the intput integer is higher '+\
                                'than the length of the string requested length')
        ostr = str(iint)
        if len(ostr) <= strl:
            ostr = '0'*(strl-len(ostr))+ostr
            return ostr
        else:
            print('intstringError: the length of the intput integer is higher\n'+\
                'than the length of the string requested length')
            raise intstringError()
    

    def build(self,gpath,gfname,resampling=[1,1,1],beginIndex=-1, endIndex=-1,verbose=True,\
              indices=[],ibegin=None,iend=None,istep=1):
        """
        Build the Cloud data
        resampling, beginIndex, endIndex and verbose input parameters correspond to the
        same one in StagData.stagImport()
        """
        # -- Path and file
        self.gpath       = gpath
        self.gfname      = gfname
        self.resampling  = resampling
        # -- Geometry
        self.geometry    = _temp.geometry
        # -- Layers
        self.beginIndex = beginIndex
        self.endIndex   = endIndex
        # -- indices
        if len(indices) == 0 and np.logical_or(ibegin==None,iend==None):
            raise CloudBuildIndexError
        elif len(indices) == 0:
            self.ibegin  = ibegin
            self.iend    = iend
            self.istep   = istep
            self.indices = list(range(ibegin,iend,istep))
        else:
            self.indices = indices
            self.ibegin  = indices[0]
            self.iend    = indices[-1]
            self.istep   = 1
        self.nt = len(self.indices)
        self.verbose = verbose
        # -- Initiate
        self.simuAge = np.empty(self.nt)
        self.ti_step = np.empty(self.nt)
    

    def iterate(self):
        """
        Iterate on given index to build a drop
        """
        # --- Prepare index
        self.ci += 1
        ind = self.indices[self.ci]
        self.cfname = self.gfname%self.__intstringer(ind,5)
        # --- Build the drop
        self.drop = StagData(geometry=self.geometry)
        self.drop.verbose = self.verbose
        self.drop.stagImport(self.gpath, self.cfname, resampling=self.resampling,\
                             beginIndex=self.beginIndex, endIndex=self.endIndex)
        self.drop.stagProcessing()
        # --- Fill Cloud field
        self.simuAge[self.ci] = self.drop.simuAge
        self.ti_step[self.ci] = self.drop.ti_step
    

    def reset(self):
        """
        reset the value of self.ci
        """
        self.ci = -1
        # -- Initiate
        self.simuAge = np.empty(self.nt)*np.nan
        self.ti_step = np.empty(self.nt)*np.nan

    
    def cloud2VTK(self,fname,multifile=False,timepar=1,path='./',creat_pointID=False,extended_verbose=True):
        """
        timepar = int, defines the unit of time in the .xdmf file.
                  Have a look in pypstag.stagVTK.stagCloud2timeVTU documentation for more details
        """
        self.im('Requested: Build VTK from StagCloudData object')
        if self.geometry == 'cart2D' or self.geometry == 'annulus':
            raise VisuGridGeometryError(self.geometry,'cart3D or yy')
        else:
            from pypStag.stagVTK import stagCloud2timeVTU
            stagCloud2timeVTU(fname,self,multifile=multifile,timepar=timepar,path=path,\
                              creat_pointID=creat_pointID,verbose=self.verbose,extended_verbose=extended_verbose)



class YinYangCloudData(MainCouldStagData):
    """
    Defines the structure of the YinYangCloudData object derived from MainCouldStagData type.
    """
    def __init__(self):
        super().__init__()  # inherit all the methods and properties from MainCouldStagData
        self.geometry = 'yy'


class CartesianCloudData(MainCouldStagData):
    """
    Defines the structure of the CartesianCloudData object derived from MainCouldStagData type.
    """
    def __init__(self,geometry):
        super().__init__()  # inherit all the methods and properties from MainCouldStagData
        self.geometry = geometry
        self.plan  = None  # If cartesian 2D, contain information of the plan as 'xy' or 'yz etc
    

    def spacetimeMap(self,axis=0,layer=-1,plotparam=None,timepar=0,aspect_ratio=1):
        """
        timepar = 0, time axis = file number
        timepar = 1, time axis = simuAge
        timepar = 2, time axis = ti_step
        """
        self.im('Prepare a Space-Time diagram')
        from .stagViewer import spaceTimeDiagram
        spaceTimeDiagram(self,axis=axis,layer=layer,timepar=timepar,\
                     plotparam=plotparam,aspect_ratio=aspect_ratio)


class SphericalCloudData(MainCouldStagData):
    """
    Defines the structure of the SphericalCloudData object derived from MainCouldStagData type.
    """
    def __init__(self,geometry):
        super().__init__()  # inherit all the methods and properties from MainCouldStagData
        self.geometry = geometry
    
    def spacetimeMap(self,axis=0,layer=-1,plotparam=None,timepar=0,aspect_ratio=1):
        """
        timepar = 0, time axis = file number
        timepar = 1, time axis = simuAge
        timepar = 2, time axis = ti_step
        """
        self.im('Prepare a Space-Time diagram')
        from .stagViewer import spaceTimeDiagram
        spaceTimeDiagram(self,axis=axis,layer=layer,timepar=timepar,\
                     plotparam=plotparam,aspect_ratio=aspect_ratio)




class StagCloudData():
    """
    Defines the StagCloudData structure dynamically from geometry of the grid
    """
    def __new__(cls,geometry='cart3D'):
        """
        Force to have more than just 'duck typing' in Python: 'dynamical typing'
        <i> : geometry = str, geometry of the grid. Must be in ('cart2D',
                         'cart3D','yy','annulus') for cartesian 2D, 3D,
                         Yin-Yang or annulus geometry, respectively. By
                         default, geometry = 'cart3D'
        """
        _temp.geometry = geometry
        if geometry == 'yy':
            return YinYangCloudData()
        elif geometry == 'cart2D' or geometry == 'cart3D':
            return CartesianCloudData(geometry)
        elif geometry == 'spherical':
            return SphericalCloudData()
        elif geometry == 'annulus':
            #raise GridGeometryInDevError(geometry)
            return SphericalCloudData(geometry)
        else:
            raise InputGridGeometryError(geometry)






class StagBookData():
    """
    Defines the structure of the StagBookData object.
    """
    def __init__(self):
        super().__init__()  # inherit all the methods and properties from MainSliceData
        # generic
        self.pName   = 'stagBookData'
        self.verbose = True
        # geometry:
        self.geom = ''      # Will be: 'rsg' ...
        self.data = []      # list of the data that are loaded

    def im(self,textMessage):
        """Print verbose internal message. This function depends on the
        argument of self.verbose. If self.verbose == True then the message
        will be displayed on the terminal.
        <i> : textMessage = str, message to display
        """
        if self.verbose == True:
            print('>> '+self.pName+'| '+textMessage)

    
    def add_stagData(self,stagData):
        """
        """
        if self.geom == '' or self.geom == stagData.geom:
            if stagData.fieldType == 'Divergence':
                self.div = stagData
            elif stagData.fieldType == 'Vorticity':
                self.vor = stagData
            elif stagData.fieldType == 'Viscosity':
                self.eta = stagData
            elif stagData.fieldType == 'Temperature':
                self.t = stagData
            elif stagData.fieldType == 'Velocity':
                self.vp = stagData
            elif stagData.fieldType == 'Sigma max':
                self.smax = stagData
            elif stagData.fieldType == 'Damage':
                self.dam = stagData
            elif stagData.fieldType == 'Topography':
                self.cs = stagData
            elif stagData.fieldType == 'Density':
                self.rho = stagData
            elif stagData.fieldType == 'Lyapunov':
                self.ly = stagData
            elif stagData.fieldType == 'Stress':
                self.str = stagData
            elif stagData.fieldType == 'Poloidal':
                self.pol = stagData
            elif stagData.fieldType == 'Toroidal':
                self.tor = stagData
            elif stagData.fieldType == 'Strain Rate':
                self.ed = stagData
            elif stagData.fieldType == 'Composition':
                self.c = stagData
            elif stagData.fieldType == 'Melt Fraction':
                self.f = stagData
            elif stagData.fieldType == 'Age':
                self.age = stagData
            elif stagData.fieldType == 'Continents':
                self.nrc = stagData
            elif stagData.fieldType == 'Prot':
                self.prot = stagData
            else:
                self.fieldType = 'Error: Undetermined'
                raise FieldTypeInDevError(stagData.fname)
            self.data.append(stagData.fieldType)
            self.im('Data added to the current stagBookData instance: '+self.data[-1])
        else:
            raise GridGeometryIncompatibleError(stagData.geom,self.geom)
    
    def book2VTU(self):
        """
        """
        return 0











class _Temp():
    """
    """
    def __init__(self):
        self.geometry = 'Unknown'
_temp = _Temp()















class StagMetaData:
    """ This class contains all routines to read and treat metadata of stag output.
    Files able to analyse:
       - *_time.dat
       - *_rprof.dat
       - *_refstat.dat
       - *_torpol.dat
       - *_plates_analyse.dat
    """

    def __init__(self,geometry='cart3D'):
        """Builder function.
        <i> : self = instance parameter
              geometry = str, geometry of the grid. Must be in ('cart2D',
                         'cart3D','yy','annulus') for cartesian 2D, 3D,
                         Yin-Yang or annulus geometry, respectively. By
                         default, geometry = 'cart3D'
        """
        #Parametrization:
        self.pName = 'stagMeta'
        self.verbose = True
        self.path  = ''
        #Universal fields
        self.header =[]       #textual header
        self.istep = []       #Index of the iteration
        self.time  = []       #Corresponding time for istep
        self.ftype = ''       #Type of data loaded e.g. 'time', or 'rprof'
        # *_time.dat
        self.Ftop  = []       #
        self.Fbot  = []       #
        self.Fmax  = []       #
        self.Fmean = []       #
        self.Tmin  = []       #Minimal temperature for each time step
        self.Tmax  = []       #Maximal temperature for each time step
        self.Tmean = []       #Average temperature for each time step
        self.Vmin  = []       #Minimal velocity for each time step
        self.Vmax  = []       #Minimal velocity for each time step
        self.Vrms  = []       #Root mean square velocity for each time step
        self.eta_min  = []    #Minimal viscosity for each time step
        self.eta_max  = []    #Maximal viscosity for each time step
        self.eta_mean = []    #Average viscosity for each time step
        self.Ra_eff = []      #Effective Rayleigh number
        self.Nu_top = []      #Nusselt number on the top of the domain
        self.Nu_bot = []      #Nusselt number on the bottom of the domain
        self.Cmin  = []       #
        self.Cmax  = []       #
        self.Cmean = []       #
        self.erupt_rate     = []    #
        self.erutpa         = []    #
        self.erupt_heatflux = []    #
        self.entrainment    = []    #
        self.Cmass_error = []  #
        self.Hint = []         #Internal heat
        # _plates_analyse
        self.mobility = []
        self.plateness = []
        # *_rprof.dat
        self.rprof  = []    #Matrix of the read radial profile. Format: self.rprof[istep,layers]
        self.layers = []    #Matrix of the (radial) layers indices for rprof
        #Generic:
        self.BIN = None
        self.bin = None
    
    
    
    
    def im(self,textMessage):
        """Print verbose internal message. This function depends on the
        argument of self.verbose. If self.verbose == True then the message
        will be displayed on the terminal.
        <i> : textMessage = str, message to display
        """
        if self.verbose == True:
            print('>> '+self.pName+'| '+textMessage)
    
    
    
    
    def metaImport(self, directory, fname, ftype='implicit',rprof_column=0):
        """ This function reads a and import data from every types of meta data files
        of stag (*_time.dat, *_rprof.dat, *_refstat.dat, *_torpol.dat, *_plates_analyse.dat)
        and fill the appropriated fields of the current StagMetaData object.
        <i> : directory = str, path to reach the data file
              fname = str, name of the data file
              ftype = str, type of meta data file, have to be in ('time', 'rprof', 
                      'refstat', 'torpol', 'plates_analyse', 'implicit')
                      Note that if 'implicit', the type will be deduced directly from
                      the fname input value. It is usefull when you haven't rename
                      meta file from stag, because it's automatic!
               rprof_column = int, index of the column you want to extract when
                              reading a *_rprof.dat file. Notice that in a StagMetaData
                              object you can store just one rprof field.
        """
        self.path = Path(directory+fname) #creat a Path object
        allowedftype = ['time','rprof','refstat','torpol','plates_analyse','plates']
        self.im('Opening the file: '+fname)
        if ftype == 'implicit':
            ftype = fname.split('_')[1].split('.')[0]
            self.im('Type detected: '+ftype)
            self.ftype = ftype
        else:
            self.im('Type entered: '+ftype)
            self.ftype = ftype
        if ftype == 'time':
            try:
                (self.istep, self.time, self.Ftop, self.Fbot, self.Tmin, self.Tmean,\
                self.Tmax, self.Vmin, self.Vrms, self.Vmax, self.eta_min, self.eta_mean,\
                self.eta_max, self.Ra_eff, self.Nu_top, self.Nu_bot, self.Cmin,\
                self.Cmean, self.Cmax, self.Fmean, self.Fmax, self.erupt_rate,\
                self.erutpa, self.erupt_heatflux, self.entrainment, self.Cmass_error,\
                self.Hint) = reader_time(self.path)
            except:
                raise NoFileError(directory,fname)
        elif ftype == 'rprof':
            try:
                (self.header,self.istep,self.time,self.layers,self.rprof) = reader_rprof(self.path,rprof_column)
            except:
                raise NoFileError(directory,fname)
        elif ftype == 'plates_analyse' or 'plates':
            try:
                (self.istep, self.time, self.mobility, self.plateness) = reader_plates_analyse(self.path)
            except:
                raise NoFileError(directory,fname)
        else:
            raise MetaFileInappropriateError(ftype,allowedftype)
        return 1
    
    
    
    
    def check(self,field='temperature',xaxis='time',figsize=(15,7)):
        """
        Only for 'time' or 'plates_analyse' data !
        Need to be complete for other fileds
        """
        if xaxis == 'time':
            xgrid = self.time
            xlabel = 'time'
        else:
            xgrid = self.istep
            xlabel = 'istep'
        #--------------------
        if self.ftype == 'plates_analyse' or self.ftype == 'plates':
            field = 'all_plates'
            plt.figure(figsize=figsize)
            ax1 = plt.subplot(121)
            ax1.set_title('Plateness')
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel('Plateness')
            ax1.plot(xgrid,self.plateness)
            ax1.plot(xgrid,[np.mean(self.plateness)+np.std(self.plateness)]*len(self.time),c='orange',label='$\pm1\sigma$')
            ax1.plot(xgrid,[np.mean(self.plateness)-np.std(self.plateness)]*len(self.time),c='orange')
            ax1.plot(xgrid,[np.mean(self.plateness)]*len(self.time),c='blue',label='mean value')
            ax2 = plt.subplot(122)
            ax2.set_title('Plates mobility')
            ax2.set_xlabel(xlabel)
            ax1.set_ylabel('Mobility')
            ax2.plot(xgrid,self.mobility)
            ax2.plot(xgrid,[np.mean(self.mobility)-np.std(self.mobility)]*len(self.time),c='orange',label='$\pm1\sigma$')
            ax2.plot(xgrid,[np.mean(self.mobility)+np.std(self.mobility)]*len(self.time),c='orange')
            ax2.plot(xgrid,[np.mean(self.mobility)]*len(self.time),c='blue',label='mean value')
            plt.show()
        elif self.ftype == 'time':
            allowedField = ['temperature','viscosity','velocity','param','composition']
            plt.figure(figsize=figsize)
            if field == 'temperature':
                plt.suptitle('Temperature evolution')
                field1 = self.Tmin
                field2 = self.Tmax
                field3 = self.Tmean
                ylabel = 'Temperature'
                titles = ('Tmin','Tmax','Tmean')
            elif field == 'velocity':
                plt.suptitle('Velocity evolution')
                field1 = self.Vmin
                field2 = self.Vmax
                field3 = self.Vrms
                ylabel = 'Velocity'
                titles = ('Vmin','Vmax','Vrms')
            elif field == 'viscosity':
                plt.suptitle('Viscosity evolution')
                field1 = self.eta_min
                field2 = self.eta_max
                field3 = self.eta_mean
                ylabel = 'Viscosity'
                titles = ('Eta min','Eta max','Eta mean')
            elif field == 'param':
                plt.suptitle('fluid parameters')
                field1 = self.Ra_eff
                field2 = self.Nu_bot
                field3 = self.Nu_top
                ylabel = ''
                titles = ('Ra effective','Nu bottom','Nu top')
            elif field == 'composition':
                plt.suptitle('Composition evolution')
                field1 = self.Cmin
                field2 = self.Cmax
                field3 = self.Cmean
                ylabel = 'Composition'
                titles = ('Cmin','Cmax','Cmean')
            else:
                raise MetaCheckFieldUnknownError(field,allowedField)
            #--------------------
            ax1 = plt.subplot(131)
            ax1.set_title(titles[0])
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            ax1.plot(xgrid,field1)
            #--------------------
            ax2 = plt.subplot(132)
            ax2.set_title(titles[1])
            ax2.set_xlabel(xlabel)
            ax2.plot(xgrid,field2)
            #--------------------
            ax3 = plt.subplot(133)
            ax3.set_title(titles[2])
            ax3.set_xlabel(xlabel)
            ax3.plot(xgrid,field3)
            plt.show()



    def checkProf(self):
        """
        Only for rprof data !
        """
        plt.figure()
        plt.title('Rprof evolution with time')
        time = range(0,len(self.time),int(len(self.time)/20))
        for t in time:
            plt.plot(self.rprof[t,:],self.layers)
            plt.ylabel('depth layers')
        plt.show()































