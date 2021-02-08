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
from .stagError import NoFileError, InputGridGeometryError, GridGeometryError, GridGeometryInDevError, \
                       MetaCheckFieldUnknownError, MetaFileInappropriateError, FieldTypeInDevError, \
                       VisuGridGeometryError, StagTypeError, CloudBuildIndexError, SliceAxisError, \
                       IncoherentSliceAxisError





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
            if self.geometry != 'cart2D':
                raise GridGeometryError(self.geometry,'cart2D')
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
        self.slayers = new_slayers
        
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
        elif ''.join(n[0:1]) == 'w':
            self.fieldType = 'Vorticity'
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
    

    def stag2VTU(self,fname=None,path='./',ASCII=False,verbose=True):
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
            """
            self.im('Requested: Build VTU from StagData object')
            if self.geometry == 'cart2D' or self.geometry == 'annulus':
                raise VisuGridGeometryError(self.geometry,'cart3D or yy')
            if fname == None:
                import time
                (y,m,d,h,mins,secs,bin1,bin2,bin3) = time.localtime()
                fname = self.fname+'_'+str(d)+'-'+str(m)+'-'+str(y)+'_'+str(h)+'-'+str(mins)+'-'+str(secs)
                self.im('Automatic file name attribution: '+fname)
            #Importation of the stagVTK package
            from .stagVTK import stag2VTU
            stag2VTU(fname,self,path,ASCII=ASCII,verbose=verbose)
    
    







class StagCartesianGeometry(MainStagObject):
    """
    Defines the StagCartesianGeometry object, derived from MainStagObject
    This object is conditionally inherited in StagData.
    """
    def __init__(self,geometry):
        super().__init__()  # inherit all the methods and properties from MainStagObject
        self.geometry = geometry
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
        if self.geometry == 'cart2D':
            self.im('      - 2D cartesian grid geometry')
        else:
            self.im('      - 3D cartesian grid geometry')
        (self.x,self.y,self.z) = np.meshgrid(self.x_coords,self.y_coords,self.z_coords,indexing='ij')
        #Same operation but on index matrix:
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
        self.r1 = []        #Matrix of the radius of points for Yin grid
        self.r2 = []        #Matrix of the radius of points for Yang grid
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

        #Function for 3D psherical YY grids
        def rectangular2YY(x,y,z,rcmb):
            """Returns YY geometry for the two blocks in cartesian coordinates
            Yin grid (x1,y1,z1), Yang grid (x2,y2,z2) from a single rectangular
            grid."""
            if isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and isinstance(z,np.ndarray):
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
            else:
                print('TypeError: Error found in types of inputs!')
                print('Error number: 100')
                return ''
        
        def cartesian2spherical(x1,y1,z1,x2,y2,z2):
            """Converts cartesian coordinates of YY grid into spherical coordinates"""
            #Type converting
            x1 = np.array(x1)
            y1 = np.array(y1)
            z1 = np.array(z1)
            x2 = np.array(x2)
            y2 = np.array(y2)
            z2 = np.array(z2)
            #Yin grid
            r1 = np.sqrt(x1**2+y1**2+z1**2)
            theta1 = np.arctan2(np.sqrt(x1**2+y1**2),z1)
            phi1 = np.arctan2(y1,x1)
            #Yang grid
            r2 = np.sqrt(x2**2+y2**2+z2**2)
            theta2 = np.arctan2(np.sqrt(x2**2+y2**2),z2)
            phi2 = np.arctan2(y2,x2)
            return ((r1,theta1,phi1),(r2,theta2,phi2))
        
        #Creation of Yin-Yang grids:
        self.im('      - Creation of the Yin-Yang grids')
        ((self.x1_overlap,self.y1_overlap,self.z1_overlap),(self.x2_overlap,self.y2_overlap,self.z2_overlap)) = \
            rectangular2YY(self.X,self.Y,self.Z,self.rcmb)
        ((self.r1,theta1,phi1),(self.r2,theta2,phi2)) = \
            cartesian2spherical(self.x1_overlap,self.y1_overlap,self.z1_overlap,self.x2_overlap,self.y2_overlap,self.z2_overlap)

        ##Cut off the corners from grid #1, which seems to do #2:
        ##Build Redflags on wrong coordinates
        theta12 = np.arccos(np.multiply(np.sin(theta1),np.sin(phi1)))
        self.redFlags = np.where(np.logical_or(np.logical_and((theta12>np.pi/4),(phi1>np.pi/2)),\
                                                np.logical_and((theta12<3*np.pi/4),(phi1<-np.pi/2))))[0]

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
                self.v1_overlap = [] #Yin
                self.v2_overlap = [] #Yang
                for gid in goodIndex:
                    self.v1_overlap.append(V1[gid])
                    self.v2_overlap.append(V2[gid])
                
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
                self.vx1_overlap = [] #Yin
                self.vx2_overlap = [] #Yang
                self.vy1_overlap = []
                self.vy2_overlap = []
                self.vz1_overlap = []
                self.vz2_overlap = []
                self.P1_overlap  = []
                self.P2_overlap  = []
                for gid in goodIndex:
                    self.vx1_overlap.append(VX1[gid])
                    self.vx2_overlap.append(VX2[gid])
                    self.vy1_overlap.append(VY1[gid])
                    self.vy2_overlap.append(VY2[gid])
                    self.vz1_overlap.append(VZ1[gid])
                    self.vz2_overlap.append(VZ2[gid])
                    self.P1_overlap.append(P1[gid])
                    self.P2_overlap.append(P2[gid])
            
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
        
        # == Processing Finish !
        self.im('Processing of stag data done!')







class StagSphericalGeometry(MainStagObject):
    """
    Defines the StagSphericalGeometry object, derived from MainStagObject
    This object is conditionally inherited in StagData.
    """
    def __init__(self,geometry):
        super().__init__()  # inherit all the methods and properties from MainStagObject
        self.geometry = geometry
        # ----- Cartesian 2D and 3D geometries ----- #
        self.x  = []        #Matrix of X coordinates meshed (in spherical)
        self.y  = []        #Matrix of Y coordinates meshed (in spherical)
        self.z  = []        #Matrix of Z coordinates meshed (in spherical)
        self.xc = []        #Matrice of cartesian x coordinates (un-projected)
        self.yc = []        #Matrice of cartesian y coordinates (un-projected)
        self.zc = []        #Matrice of cartesian z coordinates (un-projected)
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
        if self.geometry == 'cart2D':
            self.im('      - 2D cartesian grid geometry')
        else:
            self.im('      - 3D cartesian grid geometry')
        (self.x,self.y,self.z) = np.meshgrid(self.x_coords,self.y_coords,self.z_coords,indexing='ij')
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
        def rectangular2YY(x,y,z,rcmb):
            """Returns YY geometry for the two blocks in cartesian coordinates
            Yin grid (x1,y1,z1), Yang grid (x2,y2,z2) from a single rectangular
            grid."""
            if isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and isinstance(z,np.ndarray):
                #Spherical coordinates:
                R = z+rcmb
                lat = np.pi/4 - x
                lon = y - 3*np.pi/4
                #Spherical grid
                x = np.multiply(np.multiply(R,np.cos(lat)),np.cos(lon))
                y = np.multiply(np.multiply(R,np.cos(lat)),np.sin(lon))
                z = np.multiply(R,np.sin(lat))
                return (x,y,z)
            else:
                print('TypeError: Error found in types of inputs!')
                print('Error number: 100')
                return ''
        
        #Creation of Yin-Yang grids:
        self.im('      - Creation of the spherical grids')
        (self.x,self.y,self.z) = rectangular2YY(self.x,self.y,self.z,self.rcmb)

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
            
            #Creation of non overlapping data matrices for Yin and Yang
            self.vx = np.array(self.vx)
            self.vy = np.array(self.vy)
            self.vz = np.array(self.vz)
            self.vr = np.array(self.vr)
            self.P = np.array(self.P)
            
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
            raise GridGeometryInDevError(geometry)
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
        # ----- Yin Yang geometry ----- #
        self.r1 = []        #Matrix of the radius of points for Yin grid
        self.r2 = []        #Matrix of the radius of points for Yang grid
        self.x1 = []        #Yin grid x matrix - non-overlapping grids:
        self.y1 = []        #Yin grid y matrix
        self.z1 = []        #Yin grid z matrix
        self.x2 = []        #Yang grid x matrix
        self.y2 = []        #Yang grid y matrix
        self.z2 = []        #Yang grid z matrix
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

        
    def stackyy(self):
        """
        Computes all stacked fields from YinYang grid
        """
        #Dynamic containers: Use CPU on each call
        nodp = self.x1.shape[0]
        nodv = self.v1.shape[0]
        self.im('Stack grid matrices')
        self.x              = np.zeros((nodp*2))
        self.x[0:nodp]      = self.x1
        self.x[nodp:2*nodp] = self.x2
        self.y              = np.zeros((nodp*2))
        self.y[0:nodp]      = self.y1
        self.y[nodp:2*nodp] = self.y2
        self.z              = np.zeros((nodp*2))
        self.z[0:nodp]      = self.z1
        self.z[nodp:2*nodp] = self.z2
        self.r              = np.zeros((nodp*2))
        self.r[0:nodp]      = self.r1
        self.r[nodp:2*nodp] = self.r2
        self.im('Stack fields')
        if self.fieldNature == 'Scalar':
            self.v              = np.zeros((nodv*2))
            self.v[0:nodv]      = self.v1
            self.v[nodv:2*nodv] = self.v2
        else:
            nodv = self.v1.shape[0]
            nods = self.vr1.shape[0]
            self.v              = np.zeros((nodv*2)) # norm
            self.v[0:nodv]      = self.v1
            self.v[nodv:2*nodv] = self.v2
            self.vx              = np.zeros((nods*2))
            self.vx[0:nods]      = self.vx1
            self.vx[nods:2*nods] = self.vx2
            self.vy              = np.zeros((nods*2))
            self.vy[0:nods]      = self.vy1
            self.vy[nods:2*nods] = self.vy2
            self.vz              = np.zeros((nods*2))
            self.vz[0:nods]      = self.vz1
            self.vz[nods:2*nods] = self.vz2
            self.P               = np.zeros((nods*2))
            self.P[0:nods]       = self.P1
            self.P[nods:2*nods]  = self.P2

        self.v = np.array(list(self.v1)+list(self.v2))
        self.vx = np.array(list(self.vx1)+list(self.vx2))
        self.vy = np.array(list(self.vy1)+list(self.vy2))
        self.vz = np.array(list(self.vz1)+list(self.vz2))
        self.P  = np.array(list(self.P1)+list(self.P2))
        self.vtheta = np.array(list(self.vtheta1)+list(self.vtheta2))
        self.vphi   = np.array(list(self.vphi1)+list(self.vphi2))
        self.vr     = np.array(list(self.vr1)+list(self.vr2))
        self.im('Stacking done successfully!')


    def sliceExtractor(self,stagData,layer):
        """
        Extract a depth slice in a stagData.StagYinYangGeometry object.
        <i> : stagData = stagData.StagYinYangGeometry
              layer    = int, index of the stagData layer that will be extracted in
                         the new SliceData object.
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
        self.x2 = stagData.x2.reshape(NxNy,Nz)[:,layer]
        self.y2 = stagData.y2.reshape(NxNy,Nz)[:,layer]
        self.z2 = stagData.z2.reshape(NxNy,Nz)[:,layer]
        if ftype == 'Scalar':
            self.v1   = stagData.v1.reshape(NxNy,Nz)[:,layer]
            self.v2   = stagData.v2.reshape(NxNy,Nz)[:,layer]
            # empty
            self.vx1   = stagData.vx1
            self.vx2   = stagData.vx2
            self.vy1   = stagData.vy1
            self.vy2   = stagData.vy2
            self.vz1   = stagData.vz1
            self.vz2   = stagData.vz2
            self.vr1   = stagData.vr1
            self.vr2   = stagData.vr2
            self.vtheta1   = stagData.vtheta1
            self.vtheta2   = stagData.vtheta2
            self.vphi1     = stagData.vphi1
            self.vphi2     = stagData.vphi2
            self.P1   = stagData.P1
            self.P2   = stagData.P2
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
        self.stackyy()
        self.im('Extraction done successfully!')
        self.im('    - layer        = '+str(layer))
        self.im('    - pts in slice = '+str(NxNy))
    




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
    

    def sliceExtractor(self,stagData,axis=0,layer=0):
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
              axis = int, slice direction according to:
                    axis = 0  -> slice on x (x=layer,y=:,z=:)
                    axis = 1  -> slice on y (x=:,y=layer,z=:)
                    axis = 2  -> slice on z (x=:,y=:,z=layer)
              layer    = int, index of the stagData layer that will be extracted in
                         the new SliceData object.
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
            if axis == 0:
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
            elif axis == 1:
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
            elif axis == 2:
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
                        axis = é    -> expand on z
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
        elif geometry == 'annulus':
            raise GridGeometryInDevError(geometry)
        else:
            raise InputGridGeometryError(geometry)




class InterpolatedSliceData(MainSliceData):
    """
    Defines the structure of the InterpolatedSliceData object derived from MainSliceData type.
    This object corresponds to a simplified StagCartesianGeometry object.
    """
    def __init__(self):
        super().__init__()  # inherit all the methods and properties from MainSliceData
        self.x = []         #Matrix of X coordinates meshed
        self.y = []         #Matrix of Y coordinates meshed
        self.z = []         #Matrix of Z coordinates meshed
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
        self.interpGeom   = ''          # Defined during the interpolation: indicates the type of geometry used for the new grid
        self.spacing      = ''          # Defined during the interpolation: parameter of the interpGeom
        self.interpMethod = ''          # Defined during the interpolation: method used for the interpolation
        self.deg          = False       # Defined during the interpolation: bool, for interpGeom == 'rgS' only ! if deg is True, then the
                                        # x,y,z on output will be lon,lat,r repsectivelly






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
        self.verbose  = False           #Condition on the verbose output
        self.path     = ''              #The path to the stag file
        self.gfname   = ''              #Generic File name of the stag file
        self.cfname   = ''              #Current file name of the stag file (open on a drop)
        self.geometry = ''              #Geometry of the Stag grid
        self.indices  = None            #List of all indicies
        self.ibegin   = None            #First index
        self.iend     = None            #Last index
        self.step     = None            #File index step
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
    

    def build(self,gpath,gfname,resampling=[1,1,1],verbose=False,\
              indices=[],ibegin=None,iend=None,istep=1):
        """
        Build the Cloud data
        """
        # -- Path and file
        self.gpath       = gpath
        self.gfname      = gfname
        self.resampling  = resampling
        # -- Geometry
        self.geometry    = _temp.geometry
        # -- indices
        if len(indices) == 0 and np.logical_or(ibegin==None,iend==None):
            raise CloudBuildIndexError
        elif len(indices) == 0:
            self.ibegin = ibegin
            self.iend   = iend
            self.step   = step
            self.indices = list(range(ibegin,iend,step))
        else:
            self.indices = indices
            self.ibegin  = indices[0]
            self.iend    = indices[-1]
            self.step    = 1
        self.nt = len(self.indices)
        self.verbose = verbose
        # -- Initiate
        self.simuAge = np.empty(self.nt)*np.nan
        self.ti_step = np.empty(self.nt)*np.nan
    

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
        self.drop.stagImport(self.gpath, self.cfname, resampling=self.resampling)
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

    
    def stag2VTK(self):
        """
        """
        self.bin = self.verbose
        self.verbose = True
        self.im('Requested: Build VTK from StagCloudData object')
        self.verbose = self.bin
        if self.geometry == 'cart2D' or self.geometry == 'annulus':
            raise VisuGridGeometryError(self.geometry,'cart3D or yy')
        return 1



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
    

    def spacetimeMap(self,layer=-1,plotparam=None,timepar=0,aspect_ratio=1):
        """
        timepar = 0, time axis = file number
        timepar = 1, time axis = simuAge
        timepar = 2, time axis = ti_step
        """
        self.im('Prepare a Space-Time diagram')
        from .stagViewer import spaceTimeDiagram
        spaceTimeDiagram(self,layer=layer,timepar=timepar,\
                     plotparam=plotparam,aspect_ratio=aspect_ratio)


class SphericalCloudData(MainCouldStagData):
    """
    Defines the structure of the SphericalCloudData object derived from MainCouldStagData type.
    """
    def __init__(self):
        super().__init__()  # inherit all the methods and properties from MainCouldStagData
        self.geometry = 'spherical'




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
            raise GridGeometryInDevError(geometry)
        else:
            raise InputGridGeometryError(geometry)















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































