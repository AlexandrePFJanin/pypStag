# -*- coding: utf-8 -*-
"""
@author: Alexandre Janin
@aim:    pypStag core
"""

# External dependencies:
from pathlib import Path
import numpy as np
import gc

# Internal dependencies:
from .io import fields, extract_fieldName, f_split_fname_default
from .generics import im, resampling_coord, intstringer
from .planetaryModels import Earth
from .geotransform import xyz2latlon, latlon2xyz, bend_rectangular2Spherical_YY, bend_rectangular2Spherical,\
                          velocity_pole_projecton, rotation_matrix_3D, ecef2enu_stagYY
from .errors import NoFileError, InputGridGeometryError, GridGeometryError, fieldTypeError, \
                    PypStagDimensionError,VisuGridGeometryError, StagBaseError, fieldNatureError


# ----------------- CLASSES -----------------


class RawData:
    """
    Data structure for raw binary data
    """
    def __init__(self):
        # status
        self.allocated = False # status flag to keep track of the memory usage
        # raw binary data descriptor
        self.header = []    # raw header of stag file
        self.flds = []      # raw fields of stag file
        # coordinate matrix in the header (modified by the resampling)
        self.x_coords = []  # x
        self.y_coords = []  # y
        self.z_coords = []  # z
        # processing mask (integer resampling)
        self.xind   = []    # x
        self.yind   = []    # y
        self.zind   = []    # z
        # Original number of points in the three directions before the resampling
        self.nx0 = 0        # x 
        self.ny0 = 0        # y
        self.nz0 = 0        # z
    
    @property
    def meshmask(self):
        """
        Building of mesh flags (after resampling)
        """
        (Xind,Yind,Zind) = np.meshgrid(self.xind,self.yind,self.zind, indexing='ij')
        Xind = Xind.reshape(Xind.shape[0]*Xind.shape[1]*Xind.shape[2])
        Yind = Yind.reshape(Yind.shape[0]*Yind.shape[1]*Yind.shape[2])
        Zind = Zind.reshape(Zind.shape[0]*Zind.shape[1]*Zind.shape[2])
        return np.array(np.multiply(np.multiply(Xind,Yind),Zind), dtype=bool)
    
    def deallocate(self,option='min'):
        """Explicitly free space by deallocating memory and invoke the Garbage Collector.
        
        Args:
            option (str): 'no' or 'none' do nothing
                          'min' deallocates a minimum amount of data
                          'all' deallocates all the data
                          'all+' same as 'all'
        """
        if option not in ('no','min','all','all+'):
            raise ValueError()
        else:
            self.allocated = False
        if option == 'no' or option == 'none':
            pass
        else:
            if option == 'min':
                del self.header
                del self.flds
                del self.xind
                del self.yind
                del self.zind
            if option == 'all' or option == 'all+':
                del self.x_coords
                del self.y_coords
                del self.z_coords
            # Garbage collector
            gc.collect()
            #reset the fields
            if option == 'min':
                self.header = []
                self.flds = []
                self.xind = []
                self.yind = []
                self.zind = []
            if option == 'all' or option == 'all+':
                self.x_coords = []
                self.y_coords = []
                self.z_coords = []


class _Buffer():
    """
    Buffer class for complex type instantiation
    """
    def __init__(self):
        self.geometry = 'Unknown'
_buffer = _Buffer() # special instance


class Mesh():
    """
    Universal cartesian mesh structure
    """
    def __init__(self,spherical=False,dimension=3,hiddenStatus=False):
        # --- Mesh
        self.check(dimension)
        self.spherical = spherical  # is the mesh spherical
        self.dimension = dimension  # number of dimensions
        self.x = None
        if dimension > 1:
            self.y = None
        if dimension > 2:
            self.z = None
        if spherical:
            self.r     = None   # radial coordinates (adim)
            self.phi   = None   # longitude (radians)
            if dimension > 2:
                self.theta = None   # colatitude (radians)
        # --- Status
        self.hiddenStatus = hiddenStatus
        if not hiddenStatus:
            self.allocated = False  # status flag to keep track of the memory usage

    @property
    def lon(self):
        """Returns longitudes in degrees E"""
        return self.phi*180/np.pi
    
    @property
    def lat(self):
        """Returns latitudes in degrees N"""
        return 90-self.theta*180/np.pi


    @property
    def pointID(self):
        """Point ID field."""
        nod = 1
        for i in range(len(self.x.shape)):
            nod *= self.x.shape[i]
        ids = np.arange(nod)
        ids = ids.reshape(self.x.shape)
        return ids

    def check(self,dim):
        if dim > 3 or dim < 1:
            raise PypStagDimensionError(dim,dtype=False)
        if not isinstance(dim,int):
            raise PypStagDimensionError(dim,dtype=True)

    def deallocate(self):
        """Mesh deallocation: Explicitly free space by deallocating
        memory and invoke the Garbage Collector."""
        del self.x
        if self.dimension > 1:
            del self.y
        if self.dimension > 2:
            del self.z
        if self.spherical:
            del self.r
            del self.phi
            if self.dimension > 2:
                del self.theta
        # Garbage collector
        gc.collect()
        #reset the fields
        self.x = None
        if self.dimension > 1:
            self.y = None
        if self.dimension > 2:
            self.z = None
        if self.spherical:
            self.r     = None
            self.phi   = None
            if self.dimension > 2:
                self.theta = None
        # status change
        if not self.hiddenStatus:
            self.allocated = False

    

class OverlapMesh():
    """
    Data structure for Yin Yang overlapping fields
    """
    def __init__(self, scalar=True, pressure=False):
        self.yin  = PartialYinYang('yin',  scalar=scalar, pressure=pressure)
        self.yang = PartialYinYang('yang', scalar=scalar, pressure=pressure)
        pass

class PartialYinYang(Mesh):
    """
    Data structure of the Yin or Yang grid.
    """
    def __init__(self, name, scalar=True, pressure=False):
        super().__init__(spherical=True, dimension=3, hiddenStatus=False)
        if name in ('yin','yang'):
            self.name = name   # name of the mesh
        else:
            print("ERROR: the name of the partial yinyang mesh have to be in ['yin','yang']")
            raise ValueError()
        # --- fields
        self.scalar = scalar
        self.pressure = pressure
        self.v = None
        if not scalar:
            self.vx = None
            self.vy = None
            self.vz = None
            self.vr     = None
            self.vtheta = None
            self.vphi   = None
        if pressure:
            self.P = None
    
    @property
    def vlon(self):
        """Returns the longitude component of the velocity, positive towards the East"""
        return self.vphi

    @property
    def vlat(self):
        """Returns the latitude component of the velocity, positive towards the North"""
        return -self.vtheta
    
    def deallocate(self):
        # call the deallocate function from Mesh
        super().deallocate()
        # deallocate also the fields
        del self.v
        if not self.scalar:
            del self.vx
            del self.vy
            del self.vz
            del self.vr
            del self.vtheta
            del self.vphi
        if self.pressure:
            del self.P
        # Garbage collector
        gc.collect()
        #reset the fields
        self.v = None
        if not self.scalar:
            self.vx = None
            self.vy = None
            self.vz = None
            self.vr = None
            self.vtheta = None
            self.vphi = None
        if self.pressure:
            self.P
        # status change
        self.allocated = False
        return 
        


class MainStagObject(Mesh):
    """
    Main class defining the highest level of inheritance
    for StagData derived object
    """
    def __init__(self, spherical=False, planetaryModel=Earth):
        """
        Parent builder
        """
        super().__init__(spherical=spherical, dimension=3, hiddenStatus=True) # force dimension = 3, will be decrease later if needed, hide the mesh allocation status flag
        # main
        self.pName = 'stagData'
        self.verbose = True             # condition on the verbose output
        self.verbose_structure = False  # [not programmed] option on the automatic structure of the verbose output if active
        # about the binary
        self.path  = ''                 # the path to the stag file
        self.fname = ''                 # file name of the stag file
        # Mesh geometry
        self.spherical = spherical      # option controling the creation of a spherical mesh
        # planetary model
        self.planetaryModel = planetaryModel    # pypStag planetary model
        # about the field
        self.fieldName = 'Temperature'  # field contained in the current object
        self.scalar = True              # nature of the field: Scalar or Vectorial
        self.pressure = False           # need to creat an additional 'Pressure' field? Condition the creation of self.P
        # construction fields and mesh
        self.resampling = []            # resampling Parameters
        self.beginIndex = -1            # deepest z layer index
        self.endIndex   = -1            # shallowest z layer index
        self.raw = RawData()            # init a Raw data object
        self.ntb = 0                    # number of blocks, e.g. 2 for yinyang, 1 for 3D cartesian grid
        self.aspect = []                # aspect ratio of the mesh set in stagyy
        self.bot_temp = 1.0             # bottom temperature imposed in stagyy
        # Depth
        self.slayers = []               # matrix of selected layers (same size as self.raw.z_coord)
        self.depths = []                # matrix of depths in real Earth for each layers
        self.rcmb = 0                   # radius of the Core-Mantle Boundary (i.e. the deepest point of the mesh)
        # Time
        self.time = 0                   # dimensionless age of the simulation
        self.timestep = 0               # real time step of the stag simualtion
        # Dimensions after resampling
        self.nx  = 0                    # number of points in the x direction
        self.ny  = 0                    # number of points in the y direction
        self.nz  = 0                    # number of points in the z direction



    def im(self,textMessage,error=False,end=False):
        """Verbose output for internal messages."""
        im(textMessage,self.pName,self.verbose,error=error,structure=self.verbose_structure,end=end)



    def stagImport(self, directory, fname, beginIndex=-1, endIndex=-1, surface=False, resampling=[1,1,1], f_fieldName=f_split_fname_default,f_param=(5)):
        """
        Reads a stag binary data file using the module pypStag.io.fields
        and fill the current data structure.
        
        Args:
            directory (str): path to the directory in which the binary file is located
            fname (str): name of the data file
            beginIndex (int,optional): deepest index for considered layers.
                            If beginIndex=-1, the deepest index is 0, i.e. the deepest generated by stag.
                            Defaults, beginIndex = -1
            endIndex (int,optional): shallowest index for considered layers.
                            If endIndex=-1, the shallowest index is the shallowest index generated by stag.
                            Defaults, endIndex = -1
            surface (bool, optional): if set to True then, keep only the surface of the
                            model. This parameter overwrites the values of beginIndex and endIndex
                            if you had set one. The third value of resampling will also be forced to 1.
            resampling (list, size=(,3), int): matrix containing the resampling parameters (int) on X, Y
                            and Z axis as: resampling = [resampling_on_X,resampling_on_Y,resampling_on_Z]
                            Defaults, resampling=[1,1,1], means no resampling.
        """
        self.im('Reading and resampling: '+fname,end=True)
        # - Path and options
        if directory[-1] != '/':
            directory += '/'
        self.path  = Path(directory+fname) #creat a Path object
        self.fname = fname
        self.resampling = resampling

        # - Test the geometry:
        if self.geometry not in ('cart2D','cart3D','yy','spherical','annulus'):
            raise InputGridGeometryError(self.geometry)
        # - Read Stag binary files:
        try:
            (self.raw.header,self.raw.flds) = fields(self.path)
        except:
            raise NoFileError(directory,fname)

        #Strcuture for 'flds' variables:
        #  [Var][x-direction][y_direction][z-direction][block_index]
        self.raw.x_coords = self.raw.header.get('e1_coord')
        self.raw.y_coords = self.raw.header.get('e2_coord')
        self.raw.z_coords = self.raw.header.get('e3_coord')

        # number of blocks, 2 for yinyang or cubed sphere
        self.ntb = self.raw.header.get('ntb')

        # input aspect ratio of stagyy
        self.aspect = self.raw.header.get('aspect')

        # bottom temperature imposed in stagyy
        self.bot_temp = self.raw.header.get('bot_temp')

        #Conditioning the 2D/3D geometry problem:
        if not isinstance(self.raw.x_coords, np.ndarray):
            self.raw.x_coords = np.array([self.raw.x_coords])
            self.im('  - 2D data detected: plan yz')
            if self.geometry not in ['cart2D','annulus']:
                raise GridGeometryError(self.geometry,'cart2D or annulus')
            if self.geometry == 'annulus':
                self.im('  - annulus -> equatorial slice detected (nx0=1)')
        elif not isinstance(self.raw.y_coords, np.ndarray):
            self.raw.y_coords = np.array([self.raw.y_coords])
            self.im('  - 2D data detected: plan xz')
            if self.geometry not in ['cart2D','annulus']:
                raise GridGeometryError(self.geometry,'cart2D or annulus')
            if self.geometry == 'annulus':
                self.im('  - annulus -> spherical axi-symmetric detected (ny0=1)')
        elif not isinstance(self.raw.z_coords, np.ndarray):
            self.raw.z_coords = np.array([self.raw.z_coords])
            self.im('  - 2D data detected: plan xy')
            if self.geometry != 'cart2D':                           #Note: a xy_annulus is forbidden by StagYY
                raise GridGeometryError(self.geometry,'cart2D')
        else:
            self.im('  - 3D data detected')
            if self.geometry not in ['cart3D','yy','spherical']:
                raise GridGeometryError(self.geometry,'cart3D, yy or spherical')
            if self.ntb == 1:
                self.im('    -> Grid geometry compatible with cart3D or spherical')
                if self.geometry == 'yy':
                    raise GridGeometryError(self.geometry,'cart3D or spherical')
            elif self.ntb == 2:
                self.im('    -> YinYang grid detected')
                if self.geometry == 'cart3D' or self.geometry == 'spherical':
                    raise GridGeometryError(self.geometry,'yy')

        self.raw.nx0 = len(self.raw.x_coords)
        self.raw.ny0 = len(self.raw.y_coords)
        self.raw.nz0 = len(self.raw.z_coords)
        self.nx  = len(self.raw.x_coords)
        self.ny  = len(self.raw.y_coords)
        self.nz  = len(self.raw.z_coords)
        self.im("  - Original grid geometry:")
        self.im("    - Nx = "+str(self.raw.nx0))
        self.im("    - Ny = "+str(self.raw.ny0))
        self.im("    - Nz = "+str(self.raw.nz0))

        Nz = len(self.raw.header.get('e3_coord'))

        # surface
        if surface:
            self.beginIndex = self.nz-1
            self.endIndex   = self.nz
            resampling[2]   = 1 #forced
        else:
            #attribution values of default parameters
            if beginIndex == -1:
                self.beginIndex = 0
            else:
                self.beginIndex = beginIndex
            if endIndex == -1:
                self.endIndex = self.nz
            else:
                self.endIndex = endIndex

        self.slayers = np.arange(1,self.nz+1)
        self.rcmb = self.raw.header.get('rcmb')
        self.time = self.raw.header.get('ti_ad')
        self.timestep = self.raw.header.get('ti_step')
        self.im("  - Time:")
        self.im("    - Adim time = "+str(self.time))
        self.im("    - Time step = "+str(self.timestep))
        
        (self.raw.x_coords, self.raw.xind) = resampling_coord(self.raw.x_coords,resampling[0])
        (self.raw.y_coords, self.raw.yind) = resampling_coord(self.raw.y_coords,resampling[1])
        (self.raw.z_coords, self.raw.zind) = resampling_coord(self.raw.z_coords,resampling[2])

        ## Re-mapping of the zind matrix according to the range of depth considered
        #  -1- Work on indexes:
        zindtemp = np.zeros(Nz)
        for i in range(Nz):
            if i >= self.beginIndex and i < self.endIndex:
                zindtemp[i] = 1
        multi = np.multiply(self.raw.zind,zindtemp)
        if np.count_nonzero(multi) == 0:
            self.raw.zind = zindtemp
        else:
            self.raw.zind = multi
        # -2- Work on coordinates matrix
        indexNewZCoord = np.where(self.raw.zind == 1)[0]
        ztemp = self.raw.header.get('e3_coord')
        new_z_coords = []
        new_slayers   = []
        for ind in indexNewZCoord:
            new_z_coords.append(ztemp[ind])
            new_slayers.append(self.slayers[ind])    #Follows self.z_coord
        self.raw.z_coords = new_z_coords
        self.slayers = np.array(new_slayers)
        
        #Update the geometrical variable defining the grid
        self.nx  = len(self.raw.x_coords)
        self.ny  = len(self.raw.y_coords)
        self.nz  = len(self.raw.z_coords)
        self.im("  - New grid geometry:")
        self.im("    - Number of dimensions: "+str(self.dimension))
        self.im("    - Nx = "+str(self.nx))
        self.im("    - Ny = "+str(self.ny))
        self.im("    - Nz = "+str(self.nz))
        
        #Compute depths according to the input planetary model:
        width_mantle = self.planetaryModel.mantle_bottom_depth - self.planetaryModel.mantle_up_depth
        self.depths = [(1-self.raw.z_coords[i])*width_mantle+self.planetaryModel.mantle_up_depth for i in range(self.nz)]
        self.depths = np.array(sorted(self.depths,reverse=True)) #sorted as self.raw.z_coord
        
        #Find the fieldName:
        self.fieldName, self.pressure = extract_fieldName(fname,f_fieldName,f_param)
        
        #Scalar field?
        if self.raw.flds.shape[0] == 1:
            self.im('  - Scalar field detected')
            self.scalar = True
        else:
            self.im('  - Vectorial field detected: '+str(self.raw.flds.shape[0])+' dimensions')
            self.scalar = False
        
        # status change: keep track of the memory allocation
        self.raw.allocated = True

        #Out
        self.im('    -> '+self.fieldName)
        self.im('Reading and resampling operations done!',end=True)
    

    def stag2VTU(self,fname=None,path='./',ASCII=False,return_only=False,creat_pointID=False,verbose=True):
            """
            Integration of the pypStag.vtk module for a fast call on pypStag.stagData.StagData instance.
            This function creats '.vtu' or 'xdmf/h5' file readable with Paraview to efficiently 
            visualize 3D data contained in the StagData object. This function works directly
            on non overlapping StagData fields.
            Note also that the internal field stagData.slayers of the stagData object
            must be filled.
            
            Args:
                fname (str) = name of the exported file without any extention
                            Defaults, creat a file name combining the stag binary file name
                            and the current date and time.
                path (str) = path where you want to export the visualization file.
                            Defaults, path='./'
                ASCII (bool) = If set to True then output format will be 'ASCII .vtu' (Not recommended).
                            Else, the data will be stored in a binary .h5 file and the 
                            instruction will be stored in a .xdmf file (Recommended option).
                            Defaults, ASCII = Fasle
                creat_pointID (bool): If set to True then creats a new field in the paraview
                            file corresponding to the list of points ID. The ID of mesh point
                            is unique and correspond to their position on the StagData mesh
                            (similar as the internal StagData field 'pointID').
                            This option can be usefull very usefull when post processing
                            StagYY data with MAPT3 or TTK softwares.
                            WARNING: This option is only available if ASCII = False
                                     (i.e. .h5/.xdmf outputs)
                            Defaults: creat_pointID=False
                return_only (bool): Option controling the creation of output files.
                            If set to True then, no paraview files are created and you
                            will get in return the essential contruction fields.
                            (Usefull for debug).
                            If set to True, then returns:
                                Points, ElementNumbers, vstack, pointID
                            Defaults, return_only = False
                verbose (bool): Option controlling the verbose output specifically for the
                            pypStag.vtk module.
                            Defaults, verbose = True.
            """
            # Declaration here of the vtk module to avoid circular import
            from .vtk import stag2VTU
            self.im('Build paraview visualization from pypStag.StagData object instance',end=True)
            if self.geometry not in ['cart2D','cart3D','yy','annulus','spherical']:
                # not supported yet
                raise VisuGridGeometryError(self.geometry,'cart2D or cart3D or yy or annulus or spherical')
            if fname == None:
                import time
                (y,m,d,h,mins,secs,_,_,_) = time.localtime()
                d = intstringer(d,2)
                m = intstringer(d,2)
                y = intstringer(y,4)
                h = intstringer(h,2)
                mins = intstringer(mins,2)
                secs = intstringer(secs,2)
                fname = self.fname+'_'+d+'-'+m+'-'+y+'_'+h+'-'+mins+'-'+secs
                self.im('Automatic file name attribution: '+fname)
            if not return_only:
                stag2VTU(fname,self,path,ASCII=ASCII,creat_pointID=creat_pointID,return_only=return_only,verbose=verbose)
            else:
                Points,ElementNumbers,vstack,pointID = stag2VTU(fname,self,path,ASCII=ASCII,creat_pointID=creat_pointID,return_only=return_only,verbose=verbose)
                return Points,ElementNumbers,vstack,pointID
    
    






class StagCartesianGeometry(MainStagObject):
    """
    Defines the pypStag.stagData.StagCartesianGeometry object, derived from MainStagObject
    This object is conditionally inherited in StagData.
    """
    def __init__(self, geometry, planetaryModel=Earth):
        super().__init__(spherical=False, planetaryModel=planetaryModel)  # inherit all the methods and properties from MainStagObject
        self.geometry = geometry
        self.plan     = None
        self.reorganized = False # bool, flag indicating if you reorganized the mesh for 2D object
        # ----- Cartesian 2D and 3D geometries ----- #
        self.v = []         #Matrix of scalar field (or norm of vectorial fields)
        if not self.scalar:
            self.vx = []        #Matrix of x-component of the vectorial field for Cartesian grids
            self.vy = []        #Matrix of y-component of the vectorial field for Cartesian grids
            self.vz = []        #Matrix of z-component of the vectorial field for Cartesian grids
        if self.pressure:
            self.P  = []        #Matrix of Pressure field for Cartesian grids
    
    @property
    def layers(self):
        layers = np.zeros(self.x.shape, dtype=np.int32)
        for i in range(self.nz):
            layers[:,:,i] = i+1
        return layers

    def stagProcess(self,deallocate='all', reorganize=True):
        """
        This function processes the raw stag data read with the function
        pypStag.stagData.stagImport for a Cartesian geometry.

        Args:
            deallocate (str): Explicitly free space by deallocating memory
                            and invoke the Garbage Collector.
                            Options:
                                deallocate = 'no' or 'none': do nothing
                                deallocate = 'min':  deallocates the raw data
                                deallocate = 'all':  same as 'min'
                                deallocate = 'all+': same as 'min'
            reorganize (bool): If set to True, in the case of a cartesian 2D object,
                            pypStag will reorganize the mesh to have the slice
                            in the xy-plan (cartesian geometry), whatever the plan,
                            removing the unsued third dimension and here, always the z-axis.
                            If reorganize is set to True then, you will transform the
                            current object in a real 2D object (dimension = 2).
                            NOTE: Does not affect cartesian 3D objects.
        """
        self.im('Processing stag Data:',end=True)
        self.im('  - Grid Geometry')
        # Meshing
        (self.x,self.y,self.z) = np.meshgrid(self.raw.x_coords,self.raw.y_coords,self.raw.z_coords,indexing='ij')
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
        # Mesh mask:
        goodIndex = np.arange(self.raw.nx0*self.raw.ny0*self.raw.nz0)[self.raw.meshmask]
        #Processing of the field according to its scalar or vectorial nature:
        if self.scalar:
            self.im('      - Create data grid for scalar field')
            (Nx, Ny, Nz) = self.raw.header.get('nts')
            V = self.raw.flds[0,:,:,:,0].reshape(Nx*Ny*Nz)
            self.v = V[goodIndex].reshape(self.nx,self.ny,self.nz)
        elif not self.scalar:
            self.im('      - Create data grid for vectorial field')
            (Nx, Ny, Nz) = self.raw.header.get('nts')
            temp_vx = self.raw.flds[0][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_vy = self.raw.flds[1][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_vz = self.raw.flds[2][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            self.vx = temp_vx[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.vy = temp_vy[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.vz = temp_vz[goodIndex].reshape(self.nx,self.ny,self.nz)
            self.v  = np.sqrt(self.vx**2+self.vy**2+self.vz**2) #the norm
        if self.pressure:
            temp_P  = self.raw.flds[3][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            self.P  = temp_P[goodIndex].reshape(self.nx,self.ny,self.nz)
        # Reorganized the fields in case of a 2D cartesian object: remove a dimension (z)
        # on the mesh and the fields, and keep only/transfert the slice to the xy-plan,
        # -> allows to save RAM and is more coherent with a 2D object.
        # Just keep in mind that when you ask for x then, in stagyy if may be y or z for instance.
        if self.geometry == 'cart2D':
            if reorganize:
                self.reorganized = True
            else:
                self.reorganized = False
            self.im('      - Reorganization of the mesh for a 2D cartesian object: '+str(self.reorganized))
            self.im('           -> Number of dimensions:  2 -> 3')
        else:
            self.reorganized = False
        # Apply the reorganisation
        if self.reorganized:
            self.dimension = 2
            if self.plan == 'xy': # already good, just need to remove the empty mesh dimension
                # --- reorganize
                # the mesh
                self.nz = 0
                self.x   = self.x.squeeze()
                self.y   = self.y.squeeze()
                del self.z
                # the associated fields
                self.v = self.v.squeeze()
                if not self.scalar:
                    self.vx = self.vx.squeeze()
                    self.vy = self.vy.squeeze()
                    del self.vz
                if self.pressure:
                    self.P = self.P.squeeze()
                # call the garbage collector
                gc.collect()
            elif self.plan == 'xz':
                # --- reorganize
                # the mesh
                self.ny  = self.nz
                self.nz  = 0
                self.x   = self.x.squeeze()
                self.y   = self.z.squeeze().copy()
                del self.z
                # the associated fields
                self.v = self.v.squeeze()
                if not self.scalar:
                    self.vx = self.vx.squeeze()
                    self.vy = self.vz.squeeze().copy()
                    del self.vz
                if self.pressure:
                    self.P = self.P.squeeze()
                # call the garbage collector
                gc.collect()
            elif self.plan == 'yz':
                # --- reorganize
                # the mesh
                self.nx  = self.ny
                self.ny  = self.nz
                self.nz  = 0
                self.x   = self.y.squeeze().copy()
                self.y   = self.z.squeeze().copy()
                del self.z
                # the associated fields
                self.v = self.v.squeeze()
                if not self.scalar:
                    self.vx = self.vy.squeeze().copy()
                    self.vy = self.vz.squeeze().copy()
                    del self.vz
                if self.pressure:
                    self.P = self.P.squeeze()
                # call the garbage collector
                gc.collect()
        # Deallocate the memory from raw data
        if deallocate not in ('no','none','min','all','all+'):
            raise ValueError()
        else:
            if deallocate == 'no' or deallocate == 'none':
                pass
            else:
                self.im('Deallocate unsued data')
                self.raw.deallocate(deallocate)
        # == Processing Finish !
        self.im('Processing of stag data done!',end=True)
    
    



class StagYinYangGeometry(MainStagObject):
    """
    Defines the pypStag.stagData.StagYinYangGeometry object, derived from MainStagObject
    This object is conditionally inherited in StagData
    """
    def __init__(self, planetaryModel=Earth):
        super().__init__(spherical=True, planetaryModel=planetaryModel)  # inherit all the methods and properties from MainStagObject
        self.geometry = 'yy'
        self.unbended = Mesh()  # data structure for the unbended 'cubic' YY mesh
        self.overlap  = OverlapMesh(scalar=self.scalar, pressure=self.pressure)
        self.yin  = PartialYinYang('yin',  scalar=self.scalar, pressure=self.pressure)
        self.yang = PartialYinYang('yang', scalar=self.scalar, pressure=self.pressure)
        # ----- Yin Yang geometry ----- #
        self.layers  = []    # matrix of layer's index meshed
        self.yinyang = []    # np.array, same size as self.x and equal to 0 when 'yin' and 1 when 'yang'
        # Stacked Yin Yang grid
        self.v      = []    #stacked scalar field (or norm of the vectorial field)
        if not self.scalar:
            #stacked vectorial fields
            self.vx     = []
            self.vy     = []
            self.vz     = []
            self.vtheta = []
            self.vphi   = []
            self.vr     = []
        if self.pressure:
            # add an additional pressure field
            self.P      = []
    

    @property
    def vlon(self):
        """Returns the longitude component of the velocity, positive towards the East"""
        return self.vphi

    @property
    def vlat(self):
        """Returns the latitude component of the velocity, positive towards the North"""
        return -self.vtheta


    def stagProcess(self, deallocate='all', build_overlapping_field=False):
        """
        This function processes the raw stag data read with the function
        pypStag.stagData.stagImport for a Yin-Yang geometry.

        Args:
            deallocate (str): Explicitly free space by deallocating memory
                            and invoke the Garbage Collector.
                            Options:
                                deallocate = 'no' or 'none': do nothing
                                deallocate = 'min':  deallocates the raw and unbended data
                                deallocate = 'all':  same as 'min' + deallocates overlapping mesh
                                deallocate = 'all+': same as 'all' + deallocates the yin and yang mesh
            build_overlapping_field (bool, optional): Option controlling the
                            construction of fields for the overlapping mesh.
                            Defaults, set to False
        """
        self.im('Processing stag Data:')
        self.im('  - Grid Geometry')
        self.im('      - Yin-Yang grid geometry')
        self.im('      - Pre-processing of coordinates matrices')
        (self.unbended.x,self.unbended.y,self.unbended.z) = np.meshgrid(self.raw.x_coords,self.raw.y_coords,self.raw.z_coords, indexing='ij')
        self.unbended.x = self.unbended.x.flatten()
        self.unbended.y = self.unbended.y.flatten()
        self.unbended.z = self.unbended.z.flatten()

        #Same operation but on layers matrix:
        (_, _, self.layers) = np.meshgrid(self.raw.x_coords,self.raw.y_coords,self.slayers, indexing='ij')
        self.layers = self.layers.reshape(self.layers.shape[0]*self.layers.shape[1]*self.layers.shape[2])

        #Creation of Yin and Yang grids:
        self.im('      - Creation of the Yin and Yang grids from overlapping grids')
        (
        (self.overlap.yin.x,  self.overlap.yin.y,  self.overlap.yin.z),\
        (self.overlap.yang.x, self.overlap.yang.y, self.overlap.yang.z)\
        ) = \
        bend_rectangular2Spherical_YY(self.unbended.x,self.unbended.y,self.unbended.z,self.rcmb)

        (self.yin.theta,  self.yin.phi,  self.yin.r)  = xyz2latlon(self.overlap.yin.x,  self.overlap.yin.y,  self.overlap.yin.z)
        (self.yang.theta, self.yang.phi, self.yang.r) = xyz2latlon(self.overlap.yang.x, self.overlap.yang.y, self.overlap.yang.z)

        ##Remove the corners of the overlapping mesh: Build Redflags on wrong coordinates
        theta12  = np.arccos(np.multiply(np.sin(self.yin.theta),np.sin(self.yin.phi)))
        redFlags = np.logical_or(np.logical_and((theta12>np.pi/4),(self.yin.phi>np.pi/2)),\
                                 np.logical_and((theta12<3*np.pi/4),(self.yin.phi<-np.pi/2)))   # bool mask
        goodIndex = np.ones(len(self.overlap.yin.x),dtype=bool)
        goodIndex[redFlags] = False

        #Assembly Yin and Yang grids
        self.im('      - Construction of the Yin and Yang sub-grids')
        self.yin.x      = self.overlap.yin.x[goodIndex]
        self.yin.y      = self.overlap.yin.y[goodIndex]
        self.yin.z      = self.overlap.yin.z[goodIndex]
        self.yang.x     = self.overlap.yang.x[goodIndex]
        self.yang.y     = self.overlap.yang.y[goodIndex]
        self.yang.z     = self.overlap.yang.z[goodIndex]
        self.yin.r      = self.yin.r[goodIndex]
        self.yang.r     = self.yang.r[goodIndex]
        self.yin.theta  = self.yin.theta[goodIndex]
        self.yang.theta = self.yang.theta[goodIndex]
        self.yin.phi    = self.yin.phi[goodIndex]
        self.yang.phi   = self.yang.phi[goodIndex]
        self.layers = self.layers[goodIndex]
        self.layers = self.layers.astype(int)
        
        # Extract the scalar or the vectorial field V: V1 on Yin, V2 on Yang
        self.im('  - Construction of the data field:')
        # reset goodIndex
        self.im('      - Processing of redFlags')
        goodIndex = np.ones(len(self.overlap.yin.x),dtype=bool)
        if build_overlapping_field:
            goodIndex_4overlap = goodIndex.copy()
        goodIndex[redFlags] = False

        #Two different types of field: Scalar or Vectorial
        self.im('      - Prepare data for the each sub grid')
        if self.scalar:
            self.im('      - Create data grid for scalar field')
            tempField = self.raw.flds[0].reshape(self.raw.flds.shape[1]*self.raw.flds.shape[2]*self.raw.flds.shape[3],2)
            meshmask = self.raw.meshmask
            V1 = tempField[meshmask,0]
            V2 = tempField[meshmask,1]
            # Overlapping fields
            if build_overlapping_field:
                self.im('         - Overlapping field requested')
                self.overlap.yin.v  = V1[goodIndex_4overlap]
                self.overlap.yang.v = V2[goodIndex_4overlap]
            #Apply redFlags on goodindex:
            self.yin.v  = V1[goodIndex]
            self.yang.v = V2[goodIndex]

        elif not self.scalar:
            self.im('      - Create data grid for vectorial field')
            tempField_vx = self.raw.flds[0][0:self.raw.nx0,0:self.raw.ny0,:,:].reshape(self.raw.nx0*self.raw.ny0*self.raw.nz0,2)
            tempField_vy = self.raw.flds[1][0:self.raw.nx0,0:self.raw.ny0,:,:].reshape(self.raw.nx0*self.raw.ny0*self.raw.nz0,2)
            tempField_vz = self.raw.flds[2][0:self.raw.nx0,0:self.raw.ny0,:,:].reshape(self.raw.nx0*self.raw.ny0*self.raw.nz0,2)
            meshmask = self.raw.meshmask
            VX1 = tempField_vx[meshmask,0]
            VX2 = tempField_vx[meshmask,1]
            VY1 = tempField_vy[meshmask,0]
            VY2 = tempField_vy[meshmask,1]
            VZ1 = tempField_vz[meshmask,0]
            VZ2 = tempField_vz[meshmask,1]

            #Transformation of the vectorial field from internal coords -> Cartesian coords
            self.im('      - Vectorial field transformation: Internal -> Cartesian')
            (tX,tY,tZ) = np.meshgrid(self.raw.x_coords,self.raw.y_coords,self.raw.z_coords,indexing='ij')
            tX = tX.flatten()
            tY = tY.flatten()
            tZ = tZ.flatten()
            #R = tZ + self.rcmb
            lat = np.pi/4 - tX
            lon = tY - 3*np.pi/4
            # --- on Yin ---
            Vtheta = VX1
            Vphi   = VY1
            Vr     = VZ1
            VX1    =    Vtheta*np.sin(lat)*np.cos(lon) - Vphi*np.sin(lon) + Vr*np.cos(lat)*np.cos(lon)
            VY1    =    Vtheta*np.sin(lat)*np.sin(lon) + Vphi*np.cos(lon) + Vr*np.cos(lat)*np.sin(lon)
            VZ1    = -1*Vtheta*np.cos(lat)                                + Vr*np.sin(lat)
            vr1 = Vr
            # --- on Yang --- 
            Vtheta = VX2
            Vphi   = VY2
            Vr     = VZ2
            VX2    = -1*(Vtheta*np.sin(lat)*np.cos(lon) - Vphi*np.sin(lon) + Vr*np.cos(lat)*np.cos(lon))
            VZ2    =     Vtheta*np.sin(lat)*np.sin(lon) + Vphi*np.cos(lon) + Vr*np.cos(lat)*np.sin(lon)
            VY2    =  -1*Vtheta*np.cos(lat)                                + Vr*np.sin(lat)
            vr2 = Vr
            #Discharge of the memory
            discharge = True
            if discharge:
                del tX
                del tY
                del tZ
                del Vtheta
                del Vphi
                del Vr
                # Garbage collector
                gc.collect()
            if build_overlapping_field:
                self.im('         - Overlapping field requested')
                #Re-sampling
                self.overlap.yin.vx  = VX1[goodIndex_4overlap]
                self.overlap.yang.vx = VX2[goodIndex_4overlap]
                self.overlap.yin.vy  = VY1[goodIndex_4overlap]
                self.overlap.yang.vy = VY2[goodIndex_4overlap]
                self.overlap.yin.vz  = VZ1[goodIndex_4overlap]
                self.overlap.yang.vz = VZ2[goodIndex_4overlap]

            #Apply redFlags on goodindex:
            self.yin.vx  = VX1[goodIndex]
            self.yin.vy  = VY1[goodIndex]
            self.yin.vz  = VZ1[goodIndex]
            self.yang.vx = VX2[goodIndex]
            self.yang.vy = VY2[goodIndex]
            self.yang.vz = VZ2[goodIndex]
            
            #Radial velocities
            self.yin.vr  = vr1[goodIndex]
            self.yang.vr = vr2[goodIndex]
            
            #Tranformation of velocities from cartesian to spherical:
            self.im('      - Conversion of Velocities: Cartesian -> Spherical')
            lat1 = np.arctan2(np.sqrt(self.yin.x**2+self.yin.y**2),self.yin.z)
            lon1 = np.arctan2(self.yin.y,self.yin.x)
            lat2 = np.arctan2(np.sqrt(self.yang.x**2+self.yang.y**2),self.yang.z)
            lon2 = np.arctan2(self.yang.y,self.yang.x)
            
            Vlat1 =  self.yin.vx*(np.cos(lon1)*np.cos(lat1))  + self.yin.vy*(np.sin(lon1)*np.cos(lat1))  - self.yin.vz*(np.sin(lat1))
            Vlon1 = -self.yin.vx*(np.sin(lon1))               + self.yin.vy*(np.cos(lon1))
            Vlat2 =  self.yang.vx*(np.cos(lon2)*np.cos(lat2)) + self.yang.vy*(np.sin(lon2)*np.cos(lat2)) - self.yang.vz*(np.sin(lat2))
            Vlon2 = -self.yang.vx*(np.sin(lon2))              + self.yang.vy*(np.cos(lon2))
            
            self.yin.vtheta  = Vlat1
            self.yang.vtheta = Vlat2
            self.yin.vphi    = Vlon1
            self.yang.vphi   = Vlon2

            #Fill .v1 and .v2 with the L2-norm of the velocity
            self.yin.v   = np.sqrt(self.yin.vx**2+self.yin.vy**2+self.yin.vz**2) #the norm
            self.yang.v  = np.sqrt(self.yang.vx**2+self.yang.vy**2+self.yang.vz**2) #the norm
        
        # Pressure
        if self.pressure:
            self.im('      - Create data grid for pressure field')
            # reset goodIndex
            self.im('      - Re-Processing of redFlags for the pressure field')
            goodIndex = np.ones(len(self.overlap.yin.x),dtype=bool)
            if build_overlapping_field:
                goodIndex_4overlap = goodIndex.copy()
            goodIndex[redFlags] = False
            # raw data
            tempField_P  = self.raw.flds[3][0:self.raw.nx0,0:self.raw.ny0,:,:].reshape(self.raw.nx0*self.raw.ny0*self.raw.nz0,2)
            meshmask = self.raw.meshmask
            P1 = tempField_P[meshmask,0]
            P2 = tempField_P[meshmask,1]
            if build_overlapping_field:
                self.overlap.yin.P  = P1[goodIndex_4overlap]
                self.overlap.yang.P = P2[goodIndex_4overlap]
            if self.pressure:
                self.yin.P   = P1[goodIndex]
                self.yang.P  = P2[goodIndex]

        # Stitch Yin and Yang
        self.im('  - Stitch Yin and Yang together')
        self.im('     - Stack the grids')
        self.mergeGrid()
        self.im('     - Stack the fields')
        self.mergeFields()

        # status change: keep track of the memory allocation
        self.unbended.allocated = True 
        self.overlap.yin.allocated  = True
        self.overlap.yang.allocated = True
        self.yin.allocated  = True
        self.yang.allocated = True
        
        #Deallocate the memory from raw data
        if deallocate not in ('no','none','min','all','all+'):
            raise ValueError()
        else:
            if deallocate == 'no' or deallocate == 'none': # keep raw, unbended and overlap
                pass
            else:
                # if 'min' or 'all' or 'all+'
                self.im('Deallocate unsued data')
                self.im('   - raw data')
                self.im('   - unbended data')
                self.raw.deallocate(deallocate)
                self.unbended.deallocate()
                if deallocate == 'all':
                    self.im('   - overlapping data')
                    self.overlap.yin.deallocate()
                    self.overlap.yang.deallocate()
                if deallocate == 'all+':
                    self.im('   - partial yin and yang mesh and data')
                    self.yin.deallocate()
                    self.yang.deallocate()
        # == Processing Finish !
        self.im('Processing of stag data done!')
    
    
    def splitGrid(self):
        """ This function split the loaded grid (x->x1+x2,
        for instance and do the operation for x, y, z, r,
        theta and phi)"""
        self.im('Split the Yin-Yang grid into a Yin and a Yang grid (e.g. x->x1+x2)')
        nYinYang = len(self.x)
        nYin     = int(nYinYang/2)
        self.yin.x = self.x[0:nYin]
        self.yin.y = self.y[0:nYin]
        self.yin.z = self.z[0:nYin]
        self.yin.r = self.r[0:nYin]
        self.yin.theta = self.theta[0:nYin]
        self.yin.phi = self.phi[0:nYin]
        self.yang.x = self.x[nYin:nYinYang]
        self.yang.y = self.y[nYin:nYinYang]
        self.yang.z = self.z[nYin:nYinYang]
        self.yang.r = self.r[nYin:nYinYang]
        self.yang.theta = self.theta[nYin:nYinYang]
        self.yang.phi = self.phi[nYin:nYinYang]
        

    def mergeGrid(self):
        """ This function merge the loaded sub-grids (x1+x2->x,
        for instance and do the operation for x, y, z, r, theta
        and phi)"""
        self.im('Merge Yin and Yang grids (e.g. x1+x2->x)')
        n = self.yin.x.shape[0]
        self.yinyang = np.stack((np.zeros(n),np.ones(n))).reshape(2*n)
        self.x       = np.stack((self.yin.x,self.yang.x)).reshape(2*n)
        self.y       = np.stack((self.yin.y,self.yang.y)).reshape(2*n)
        self.z       = np.stack((self.yin.z,self.yang.z)).reshape(2*n)
        self.r       = np.stack((self.yin.r,self.yang.r)).reshape(2*n)
        self.theta   = np.stack((self.yin.theta,self.yang.theta)).reshape(2*n)
        self.phi     = np.stack((self.yin.phi,self.yang.phi)).reshape(2*n)
    
    
    def splitFields(self):
        """ This function split the loaded fields on the all mesh (v and if available, vx, vy, vz,
        vphi, vtheta, vr and P) into the Yin-Yang subgrid: v1, v2 (and vx1,vx2,vy1,vy2...)"""
        self.im('Split the Yin-Yang fields into a Yin and a Yang field (e.g. v->v1+v2)')
        nYinYang = len(self.x)
        nYin     = int(nYinYang/2)
        self.yin.v  = self.v[0:nYin]
        self.yang.v = self.v[nYin:nYinYang]
        if not self.scalar:
            self.yin.vx = self.vx[0:nYin]
            self.yin.vy = self.vy[0:nYin]
            self.yin.vz = self.vz[0:nYin]
            self.yin.vr = self.vr[0:nYin]
            self.yin.vtheta = self.vtheta[0:nYin]
            self.yin.vphi = self.vphi[0:nYin]
            self.yang.vx = self.vx[nYin:nYinYang]
            self.yang.vy = self.vy[nYin:nYinYang]
            self.yang.vz = self.vz[nYin:nYinYang]
            self.yang.vr = self.vr[nYin:nYinYang]
            self.yang.vtheta = self.vtheta[nYin:nYinYang]
            self.yang.vphi = self.vphi[nYin:nYinYang]
        if self.pressure:
            self.yin.P  = self.P[0:nYin]
            self.yang.P = self.P[nYin:nYinYang]
    
    
    def mergeFields(self):
        """ This function merge the loaded fields from the sub-meshes (Yin and Yang) to
        the entire YY (Yin+Yang). i.e. merge v1+v2 -> v (and vx1+vx2->vx, vy1+vy2->vy ...
        if vectorial)."""
        self.im('Merge Yin and Yang fields (e.g. v1+v2->v)')
        n = self.yin.x.shape[0]
        self.v       = np.stack((self.yin.v,self.yang.v)).reshape(2*n)
        if not self.scalar:
            self.vx      = np.stack((self.yin.vx,self.yang.vx)).reshape(2*n)
            self.vy      = np.stack((self.yin.vy,self.yang.vy)).reshape(2*n)
            self.vz      = np.stack((self.yin.vz,self.yang.vz)).reshape(2*n)
            self.vtheta  = np.stack((self.yin.vtheta,self.yang.vtheta)).reshape(2*n)
            self.vphi    = np.stack((self.yin.vphi,self.yang.vphi)).reshape(2*n)
            self.vr      = np.stack((self.yin.vr,self.yang.vr)).reshape(2*n)
        if self.pressure:
            self.P       = np.stack((self.yin.P,self.yang.P)).reshape(2*n)
    
    
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
            if not self.scalar:
                vprof = self.vx.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vy':
            if not self.scalar:
                vprof = self.vy.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vz':
            if not self.scalar:
                vprof = self.vz.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vr':
            if not self.scalar:
                vprof = self.vr.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vtheta':
            if not self.scalar:
                vprof = self.vtheta.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'vphi':
            if not self.scalar:
                vprof = self.vphi.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Vectorial')
        elif field == 'P':
            if not self.pressure:
                vprof = self.P.reshape((NxNy,Nz))[ids,:]
            else:
                raise fieldNatureError('Pressure')
        return vprof,coordinates

    
    def substract_rotation(self,rot,verbose=True):
        """
        Substract the angular rotation 'rot' defined by the tuple (wx,wy,wz) to the
        whole velocity field, layer by layer.

        NOTE: Applied only to the merged field. To apply it to the Yin and Yang sub
              meshes, use the function splitFields.

        Args:
            rot (list/tuple/np.ndarray): Angular rotation rates to substract.
                                         rot is defined as:  rot=(wx,wy,wz)
                                         where the components are in cartesian.
                                         wx, wy and wz have to be in RAD/MYR
        """
        self.im('Substract an angular rotation to the whole mantle velocity field')
        wx,wy,wz = rot
        self.im('Angular rotation to substract:')
        self.im('   - wx: %s  rad/Myr'%str(wx))
        self.im('   - wy: %s  rad/Myr'%str(wy))
        self.im('   - wz: %s  rad/Myr'%str(wz))
        if self.scalar or 'Velocity' not in self.fieldName:
            raise fieldTypeError('Velocity')
        else:
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
            self.im('  -> Iterative computation on the %s depth layers'%str(self.nz))
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
            self.im('Velocity reprojection: Done')
    
    
    def grid_rotation(self,axis='x',theta=1*np.pi/180,R=None):
        """
        Function for the rotation of the grid defined as either (1) a rotation
        around a given cartesian axis (x,y,z) or (2) as 3D rotation matrix.
        If the field loaded in the current class instance is a vectorial field
        then, rotate also this field (not need for a scalar field).
        
        NOTE: Applied only to the merged field. To apply it to the Yin and Yang sub
              meshes, use the function splitGrid for the mesh and splitFields for 
              the loaded fields.

        Args:
            axis (str): Cartesian axis around which you want to rotate.
                        Have to be in ['x','y','z']
                        Defaults, axis = 'x'
            theta (int/float): Angle for the rotation around a cartesian axis
                        set with the argument 'axis'.
                        Values in *RADIANS*.
                        Defaults, theta = 1*np.pi/180
            R (np.ndarray, shape = 3x3): Rotation matrix defining the rotation
                        of the grid.
                        If set, then take the priority over the value of 'axis'.
                        Defaults, R=None
        """
        if self.scalar:
            self.im('Rotation of the grid')
        else:
            self.im('Rotation of both the grid and the vectorial fields')
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
        if self.scalar and 'Velocity' in self.fieldName:
            self.im('  -> Vectors rotation:')
            vx = R[0,0]*self.vx+R[0,1]*self.vy+R[0,2]*self.vz
            vy = R[1,0]*self.vx+R[1,1]*self.vy+R[1,2]*self.vz
            vz = R[2,0]*self.vx+R[2,1]*self.vy+R[2,2]*self.vz
            self.vx = vx
            self.vy = vy
            self.vz = vz
            self.vphi, self.vtheta, self.vr = ecef2enu_stagYY(self.x,self.y,self.z,self.vx,self.vy,self.vz)
        





class StagSphericalGeometry(MainStagObject):
    """
    Defines the pypStag.stagData.StagSphericalGeometry object, derived from MainStagObject
    This object is conditionally inherited in StagData
    """
    def __init__(self, geometry, planetaryModel=Earth):
        super().__init__(spherical=True, planetaryModel=planetaryModel)  # inherit all the methods and properties from MainStagObject
        self.geometry = geometry
        self.plan     = None # stay None for 3D spherical and get a value for annulus
        self.reorganized = False # bool, flag indicating if you reorganized the mesh for 2D object
        self.cartesian = Mesh(spherical=False, dimension=3)
        # Fields
        self.v      = []    #stacked scalar field (or norm of the vectorial field)
        if not self.scalar:
            #stacked vectorial fields
            self.vx     = []
            self.vy     = []
            self.vz     = []
            self.vtheta = []
            self.vphi   = []
            self.vr     = []
        if self.pressure:
            # add an additional pressure field
            self.P      = []
    
    @property
    def vlon(self):
        """Returns the longitude component of the velocity, positive towards the East"""
        return self.vphi

    @property
    def vlat(self):
        """Returns the latitude component of the velocity, positive towards the North"""
        return -self.vtheta

    @property
    def layers(self):
        layers = np.zeros(self.x.shape, dtype=np.int32)
        for i in range(self.nz):
            layers[:,:,i] = i+1
        return layers
    
    
    def stagProcess(self, deallocate='all', reorganize=True):
        """
        This function processes the raw stag data read with the function
        pypStag.stagData.stagImport for a Spherical geometry.

        Args:
            deallocate (str): Explicitly free space by deallocating memory
                            and invoke the Garbage Collector.
                            Options:
                                deallocate = 'no' or 'none': do nothing
                                deallocate = 'min':  deallocates the raw data
                                deallocate = 'all':  same as 'min' + the cartesian mesh geometry
                                deallocate = 'all+': same as 'all'
            reorganize (bool): If set to True, in the case of a spherical annulus,
                            pypStag will reorganize the mesh to have the annulus
                            in the xy-plan (cartesian geometry) and the phi-r-plan
                            (spherical geometry), removing the axis z and theta.
                            If reorganize is set to True then, you will transform the
                            current object in a real 2D object (dimension = 2).
                            NOTE: Does not affect spherical 3D objects.
        """
        self.im('Processing stag Data:')
        self.im('  - Grid Geometry')
        # Meshing
        (self.cartesian.x,self.cartesian.y,self.cartesian.z) = np.meshgrid(self.raw.x_coords,self.raw.y_coords,self.raw.z_coords,indexing='ij')
        # Geometry
        if self.geometry == 'spherical':
            self.im('      - 3D spherical grid geometry')
        elif self.geometry == 'annulus':
            self.im('      - 2D annulus grid geometry')
            if  self.cartesian.x.shape[0] == 1:
                self.im('      - data detected: plan yz')
                self.im('  - annulus -> equatorial slice detected (nx0=1)')
                self.plan = 'yz'
            elif self.cartesian.x.shape[1] == 1:
                self.im('      - data detected: plan xz')
                self.im('  - annulus -> spherical axi-symmetric detected (ny0=1)')
                self.plan = 'xz'
            elif self.cartesian.x.shape[2] == 1:
                self.im('      - data detected: plan xy')
                self.plan = 'xy'
                raise InputGridGeometryError('xy-spherical annulus')
            else:
                self.im('ERROR: issue detected on the shape of the mesh\nshape: %s'%str(self.cartesian.x.shape),error=True)
        # Mesh mask:
        goodIndex = np.arange(self.raw.nx0*self.raw.ny0*self.raw.nz0)[self.raw.meshmask]

        # Creation of the spherical grid:
        self.im('      - Creation of the spherical grid')
        if self.geometry == 'spherical':
            (self.x,self.y,self.z) = latlon2xyz(self.cartesian.x,self.cartesian.y,self.cartesian.z+self.rcmb)
        elif self.geometry == 'annulus':
            if self.plan == 'yz':
                (self.x,self.y,self.z) = latlon2xyz(np.zeros(self.cartesian.x.shape),self.cartesian.y,self.cartesian.z+self.rcmb)
            elif self.plan == 'xz':
                (self.x,self.y,self.z) = latlon2xyz(self.cartesian.x,np.zeros(self.cartesian.y.shape),self.cartesian.z+self.rcmb)
        (self.theta,self.phi,self.r) = xyz2latlon(self.x,self.y,self.z)

        # Reorganized the fields in case of a 2D annulus: remove a dimension (z) on the mesh and keep only/transfert the annulus to the xy-plan,
        # (cartesian coordinates) and the phi-r-plan (spherical coordinates)
        # -> allows to save RAM and is more coherent with a 2D object.
        # Just keep in mind that when you ask for x then, in stagyy if may be y or z for instance.
        if self.geometry == 'annulus':
            if reorganize:
                self.reorganized = True
            else:
                self.reorganized = False
            self.im('      - Reorganization of the mesh for a 2D annulus: '+str(self.reorganized))
            self.im('           -> Number of dimensions:  2 -> 3')
        else:
            self.reorganized = False
        
        #Processing of the field according to its scalar or vectorial nature:
        if self.scalar:
            self.im('      - Create data grid for scalar field')
            (Nx, Ny, Nz) = self.raw.header.get('nts')
            V = self.raw.flds[0,:,:,:,0].reshape(Nx*Ny*Nz)
            self.v = V[goodIndex].reshape(self.nx,self.ny,self.nz)

        elif not self.scalar:
            self.im('      - Create data grid for vectorial field')
            self.im('          - Extract vectorial field in ENU reference frame')
            (Nx, Ny, Nz) = self.raw.header.get('nts')
            temp_vtheta  = self.raw.flds[0][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_vphi    = self.raw.flds[1][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            temp_vr      = self.raw.flds[2][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            self.vtheta  = temp_vtheta.reshape((self.nx,self.ny,self.nz))
            self.vphi    = temp_vphi.reshape((self.nx,self.ny,self.nz))
            self.vr      = temp_vr.reshape((self.nx,self.ny,self.nz))
            self.im('          - Transform vectors from ENU to ECEF reference frame')
            lat = 0
            lon = self.phi
            self.vx =    self.vtheta*np.sin(lat)*np.cos(lon) - self.vphi*np.sin(lon) + self.vr*np.cos(lat)*np.cos(lon)
            self.vy =    self.vtheta*np.sin(lat)*np.sin(lon) + self.vphi*np.cos(lon) + self.vr*np.cos(lat)*np.sin(lon)
            self.vz =    self.vtheta*np.cos(lat)                                     - self.vr*np.sin(lat)
            #Fill .v with the L2-norm of the velocity
            self.v  = np.sqrt(self.vx**2+self.vy**2+self.vz**2) #the norm
            self.v  = self.v.reshape(self.nx,self.ny,self.nz)
        
        if self.pressure:
            self.im('      - Create data grid for pressure field')
            (Nx, Ny, Nz) = self.raw.header.get('nts')
            temp_P  = self.raw.flds[3][0:Nx,0:Ny,:].reshape(Nx*Ny*Nz)
            self.P  = temp_P[goodIndex].reshape(self.nx,self.ny,self.nz)
        
        # Apply the reorganisation
        if self.reorganized:
            self.dimension = 2
            if self.plan == 'yz': # already good, just need to remove the empty mesh dimension
                # --- reorganize
                # the mesh
                self.nx  = self.ny
                self.ny  = self.nz
                self.nz  = 0
                self.x   = self.x.squeeze() # squeeze to have array[along phi, along r]
                self.y   = self.y.squeeze()
                self.phi = self.phi.squeeze()
                self.r   = self.r.squeeze()
                del self.z
                del self.theta
                # the associated fields
                self.v = self.v.squeeze()
                if not self.scalar:
                    self.vx = self.vx.squeeze()
                    self.vy = self.vy.squeeze()
                    self.vr = self.vr.squeeze()
                    self.vphi = self.vphi.squeeze()
                    del self.vz
                    del self.vtheta
                if self.pressure:
                    self.P = self.P.squeeze()
                # call the garbage collector
                gc.collect()
                pass
            elif self.plan == 'xz':
                # then, rename the axis (empty axis = y, keep x and change y to z)
                self.y = self.x
                self.x = self.z
                self.z = np.zeros(self.x.shape)
                # then, recompute the spherical coordinates
                (self.theta,self.phi,self.r) = xyz2latlon(self.x,self.y,self.z)
                # --- reorganize
                # the mesh
                self.ny  = self.nz
                self.nz  = 0
                self.x   = self.x.squeeze() # squeeze to have array[along phi, along r]
                self.y   = self.y.squeeze()
                self.phi = self.phi.squeeze()
                self.r   = self.r.squeeze()
                del self.z
                del self.theta
                # the associated fields
                self.v = self.v.squeeze()
                if not self.scalar:
                    # reorganize fields
                    temp_vphi   = self.vphi.copy()  # null
                    temp_vy     = self.vy.copy()    # null
                    self.vphi   = self.vtheta.squeeze()
                    self.vtheta = temp_vphi.squeeze()
                    self.vr     = self.vr.squeeze()
                    self.vy = self.vx.squeeze()
                    self.vx = self.vz.squeeze()
                    self.vz = temp_vy.squeeze()
                    del self.vz
                    del self.vtheta
                if self.pressure:
                    self.P = self.P.squeeze()
                # call the garbage collector
                gc.collect()
                
        # Deallocate the memory from raw data
        if deallocate not in ('no','none','min','all','all+'):
            raise ValueError()
        else:
            if deallocate == 'no' or deallocate == 'none': # keep everything
                pass
            else:
                # if 'min' or 'all' or 'all+'
                self.im('Deallocate unsued data')
                self.im('   - raw data')
                self.raw.deallocate(deallocate)
                if deallocate == 'all' or deallocate == 'all+':
                    self.im('   - cartesian mesh geometry')
                    self.cartesian.deallocate()
        # == Processing Finish !
        self.im('Processing of stag data done!')







class StagData():
    """
    Defines the StagData structure dynamically from the geometry of the grid.
    """
    def __new__(cls, geometry='cart3D', planetaryModel=Earth):
        """
        Force to have more than just 'duck typing' in Python: 'dynamical typing'
        
        Args:
            geometry (str): geometry of the grid. Must be in ('cart2D',
                        'cart3D','yy','annulus','spherical) for cartesian 2D, 3D,
                        Yin-Yang, annulus and 3D spherical (part of sphere)
                        geometries, respectively.
                        Defaults, geometry = 'cart3D'
        """
        if geometry == 'yy':
            return StagYinYangGeometry(planetaryModel=planetaryModel)
        elif geometry == 'cart2D' or geometry == 'cart3D':
            return StagCartesianGeometry(geometry, planetaryModel=planetaryModel)
        elif geometry == 'annulus' or geometry == 'spherical':
            return StagSphericalGeometry(geometry, planetaryModel=planetaryModel)
        else:
            raise InputGridGeometryError(geometry, planetaryModel=planetaryModel)




