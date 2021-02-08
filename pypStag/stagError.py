# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:46:12 2019

@author: Alexandre Janin
"""


"""Exceptions raised by pypStag"""


class PypStagError(Exception):
    """ Main class for all pypStag """
    pass




class PackageWarning(PypStagError):
    """Raised when a precise package is needed"""
    def __init__(self,pack):
        super().__init__('Error package import!\n'+\
             '>> the following package is needed:'+pack)


class NoFileError(PypStagError):
    """Raised when stagData.import find no file during the importation"""
    def __init__(self,directory,fname):
        super().__init__('Error on the input data !\n'+\
             '>> The expected following file does not exist !\n'+\
             ' | File requested:  '+fname+'\n'+\
             ' | On directory  :  '+directory)


class StagTypeError(PypStagError):
    """Raised unexpected type"""
    def __init__(self,givenType,expectedType):
        super().__init__('Error on input type\n'+\
            'Unexpected type given: '+str(givenType)+'\n'+\
            'Expected type for input is: '+str(expectedType))



class InputGridGeometryError(PypStagError):
    """Raised when stagData.import have a wrong input geometry"""
    def __init__(self,geom):
        super().__init__('Error on the input geometry!\n'+\
             "The proposed geometry '"+geom+"' is not contained in\n"+\
             'the allowed geometries supported by stagData object.')


class CloudBuildIndexError(PypStagError):
    """Raised when stagCloudData have a wrong index input"""
    def __init__(self,geom):
        super().__init__('Error on the input index!\n'+\
             "You have to set an 'indices' list or set a begining and end index and a file step.")


class GridGeometryInDevError(PypStagError):
    """Raised when stagData.import have an unconform input geometry"""
    def __init__(self,geom):
        super().__init__('Error on the input geometry !\n'+\
             "The input geometry '"+geom+"' is not suported now\n"+\
             'in the current version of pypStag... Be patient and take a coffee!')


class FieldTypeInDevError(PypStagError):
    """Raised when stagData.import have an unknown fieldType not yet supported"""
    def __init__(self,fieldType):
        super().__init__('Error on the input stagData.fieldType !\n'+\
             "The input fieldType '"+fieldType+"' is not supported now\n"+\
             'in the current versin of pypStag... Be patient and take a coffee !')


class GridGeometryError(PypStagError):
    """Raised when the geometry of a stagData object is not the expected
    geometry"""
    def __init__(self,INgeom,EXgeom):
        super().__init__('Error on the input geometry !\n'+\
             "The input geometry '"+INgeom+"' you chose during the construction\n"+\
             'of the StagData object is not the one expected here!\n'+\
             'Expected geometry corresponding to your input file: '+EXgeom)


class VisuGridGeometryError(PypStagError):
    """Raised when the geometry of a stagData object is not the expected
    geometry for a visualization tool"""
    def __init__(self,INgeom,EXgeom):
        super().__init__('Error on the input geometry !\n'+\
             "The input geometry '"+INgeom+"' of your StagData object is incoherent\n"+\
             'with the function you are using or its input parameters!\n'+\
             'Expected geometry: '+EXgeom)


class GridInterpolationError(PypStagError):
    """Raised when unknown input for interpolation grid"""
    def __init__(self,interpGeom):
        super().__init__('Error on the proposed interpolation grid!\n'+\
             "The selected grid geometry '"+interpGeom+"' is not supported\n"+\
             'for the moment or is wrong!')


class fieldTypeError(PypStagError):
    """Raised unexpected field type"""
    def __init__(self,expectedFieldtype):
        super().__init__('Error on the StagData Field Type\n'+\
            'Unexpected value of StagData.fiedType\n'+\
            'StagData.fieldType must be here: '+expectedFieldtype)


class SliceAxisError(PypStagError):
    """Raised when unknown axis is set in input"""
    def __init__(self,wrongaxis):
        super().__init__('Error in input axis!\n'+\
            'Unexpected value of axis: '+str(wrongaxis))


class IncoherentSliceAxisError(PypStagError):
    """Raised when incoherent axis is set in input, incoherent according
    to the grid geometry."""
    def __init__(self,wrongaxis):
        super().__init__('Error in input axis!\n'+\
            'Incoherent value of axis: '+str(wrongaxis)+', with the grid geomtry:')


class MetaFileInappropriateError(PypStagError):
    """Raised when the reader function of StagMetaData recieved an inappropriate
    file."""
    def __init__(self,ftype,allowedType):
        super().__init__('Error on the input of the meta file reader!\n'+\
             'Inappropriate meta file in input.\n'+\
             'Type you entered: '+ftype+'\n'+\
             'Type must be in: \n'+\
             str(allowedType))


class MetaCheckFieldUnknownError(PypStagError):
    """Raised when the reader function of StagMetaData recieved an inappropriate
    file."""
    def __init__(self,field,allowedFields):
        super().__init__('Error on the input field of the StagMetaData.check() function\n'+\
             'Unknown field: '+field+'\n'+\
             'The input field must be in: \n'+\
             str(allowedFields))





