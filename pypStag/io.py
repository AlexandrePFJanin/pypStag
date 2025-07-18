# -*- coding: utf-8 -*-
"""
@author: Alexandre Janin
@aim:    input/output module
@reading routines of binary files: adapted from the package Stagpy developed by Adrien Morison
 -> See the Stagpy doc here: https://github.com/StagPython/StagPy
"""

# External dependencies:
from functools import partial
import numpy as np
from itertools import product

# Internal dependencies:
from .path import path2fieldNamesParing, path2fieldNamesParingDefaults
from .errors import ParsingError, FieldTypeInDevError


# ----------------- FUNCTIONS -----------------

class HeaderInfo:
    """
    Data structure for binary headers
    NOTE: Adapted from Stagpy.
    """
    def __init__(self, path2file, magic, nval, sfield, readbin,  header):
        self.path2file = path2file
        self.magic = magic
        self.nval = nval
        self.sfield = sfield
        self.readbin = readbin
        self.header = header

# ----------------- FUNCTIONS -----------------

def _readbin(fid, fmt='i', nwords=1, file64=False, unpack=True):
    """Read n words of 4 or 8 bytes with fmt format (depends on header)

    Arg:
        fid: (_io.BufferedReader): Opened file instance (e.g. fieldfile.open("rb") as fid)
        fmt (str): type of data in ['i', 'f', 'b'].
                   'i', 'f' or 'b' for integer, float or bytes, respectively.
        nword (int): number of words that will be read
        file64 (bool, optional): If 8 bytes words: set to True,
                                 else if 4 bytes words, set to False.
                                 Defaults: file64 = Fasle
        unpack (bool, optional): If the read data is stored by numpy in a list:
                                 unpack the result.
                                 Default: unpack = True

    Return:
        elts (numpy.ndarray): Array of elements if more than one element and unpack set to True

    NOTE: Adapted from Stagpy.
    """
    if fmt in 'if':
        fmt += '8' if file64 else '4'
    elts = np.fromfile(fid, fmt, nwords)
    if unpack and len(elts) == 1:
        elts = elts[0]
    return elts



def binheader(filepath, fid):
    """Read the header of the StagYY binary file.

    Args:
        fieldfile (pathlib.PosixPath): Path to the binary file.
        fid (_io.BufferedReader): Opened file instance (e.g. fieldfile.open("rb") as fid)

    Returns:
        myheader (pypStag.io.HeaderInfo): Binary header of the 
                                          StagYY file.
    NOTE: Adapted from Stagpy.
    """
    readbin = partial(_readbin, fid)
    magic = readbin()
    if magic > 8000:  # 64 bits
        magic -= 8000
        readbin()  # need to read 4 more bytes
        readbin = partial(readbin, file64=True)
    # check nb components
    nval = 1
    sfield = False
    if magic > 400:
        nval = 4
    elif magic > 300:
        nval = 3
    elif magic > 100:
        sfield = True
    magic %= 100
    if magic < 9 or magic > 12:
        raise ParsingError(filepath, f"{magic=} not supported")
    # store header info
    myheader = HeaderInfo(filepath, magic, nval, sfield, readbin, {})
    header = myheader.header
    # extra ghost point in horizontal direction
    header["xyp"] = int(nval == 4)  # magic >= 9
    # total number of values in relevant space basis
    # (e1, e2, e3) = (theta, phi, radius) in spherical geometry
    #              = (x, y, z)            in cartesian geometry
    header["nts"] = readbin(nwords=3)
    # number of blocks, 2 for yinyang or cubed sphere
    header["ntb"] = readbin()  # magic >= 7
    # aspect ratio
    header["aspect"] = readbin("f", 2)
    # number of parallel subdomains
    header["ncs"] = readbin(nwords=3)  # (e1, e2, e3) space
    # number of blocks
    header["ncb"] = readbin()  # magic >= 8
    # r - coordinates
    # rgeom[0:self.nrtot+1, 0] are edge radial position
    # rgeom[0:self.nrtot, 1] are cell-center radial position
    header["rgeom"] = readbin("f", header["nts"][2] * 2 + 1)  # magic >= 2
    header["rgeom"] = np.resize(header["rgeom"], (header["nts"][2] + 1, 2))
    # radiux CMB
    header["rcmb"] = readbin("f")  # magic >= 7
    # time step
    header["ti_step"] = readbin()  # magic >= 3
    # time adim
    header["ti_ad"] = readbin("f")  # magic >= 3
    header["erupta_total"] = readbin("f")  # magic >= 5
    # add A.Janin 18.07.25
    if magic >= 12:
        header["erupta_TTG"] = readbin("f")
        header["intruda"]    = readbin("f", 2)
        header["TTGmass"]    = readbin("f", 3)
    else:
        header["erupta_TTG"] = None
        header["intruda"]    = None
        header["TTGmass"]    = None
    header["bot_temp"] = readbin("f")  # magic >= 6
    header["core_temp"] = readbin("f") if magic >= 10 else 1
    header["ocean_mass"] = readbin("f") if magic >= 11 else 0.0
    # magic >= 4
    header["e1_coord"] = readbin("f", header["nts"][0])
    header["e2_coord"] = readbin("f", header["nts"][1])
    header["e3_coord"] = readbin("f", header["nts"][2])
    return myheader



def fields(fieldfile):
    """Extract fields data from a StagYY binary file.

    Args:
        fieldfile (pathlib.PosixPath): Path to the binary file.

    Returns:
        (header, fields) = tuple (types: pypSatg.io.HeaderInfo, numpy.ndarray).
                           header is directly read with pypStag.io.binheader()
                           fields is indexed by as follows:
                           (variable, x-direction, y-direction, z-direction, block).
    
    NOTE: Adapted from Stagpy.
    """
    if not fieldfile.is_file():
        return None
    with fieldfile.open("rb") as fid:
        hdr = binheader(fieldfile, fid)
        header = hdr.header

        # number of points in (e1, e2, e3) directions PER CPU
        npc = header["nts"] // header["ncs"]
        # number of blocks per cpu
        nbk = header["ntb"] // header["ncb"]
        # number of values per 'read' block
        npi = (
            (npc[0] + header["xyp"])
            * (npc[1] + header["xyp"])
            * npc[2]
            * nbk
            * hdr.nval
            )
        header["scalefac"] = hdr.readbin("f") if hdr.nval > 1 else 1
        # read field:
        flds = np.zeros((
            hdr.nval,
            header["nts"][0] + header["xyp"],
            header["nts"][1] + header["xyp"],
            header["nts"][2],
            header["ntb"],
            ))
        # loop over parallel subdomains
        for icpu in product(
            range(header["ncb"]),
            range(header["ncs"][2]),
            range(header["ncs"][1]),
            range(header["ncs"][0]),
            ):
            # read the data for one CPU
            data_cpu = hdr.readbin("f", npi) * header["scalefac"]

            # icpu is (icpu block, icpu z, icpu y, icpu x)
            # data from file is transposed to obtained a field
            # array indexed with (x, y, z, block), as in StagYY
            flds[
                :,
                icpu[3] * npc[0] : (icpu[3] + 1) * npc[0] + header["xyp"],  # x
                icpu[2] * npc[1] : (icpu[2] + 1) * npc[1] + header["xyp"],  # y
                icpu[1] * npc[2] : (icpu[1] + 1) * npc[2],  # z
                icpu[0] * nbk : (icpu[0] + 1) * nbk,  # block
            ] = np.transpose(
                data_cpu.reshape(
                    (
                        nbk,
                        npc[2],
                        npc[1] + header["xyp"],
                        npc[0] + header["xyp"],
                        hdr.nval,
                    )
                )
            )
        if hdr.sfield:
            # for surface fields, variables are written along z direction
            flds = np.swapaxes(flds, 0, 3)
    return header, flds



def f_split_fname_default(fname,param=(5)):
    """Default function to split the field descriptor (e.g. 'eta'
    for a viscosity file).
    
    Args:
        fname (str): stagyy binary file name.
                     e.g fname='llsvp_eta00001'
        param (tupe): list of parameters used by this function
                    to extract the descriptor.
                    Here:
                    param = (ndigits)
                        ndigits (int,optional):
                                number of digits in the stag
                                binary output index.
                                Defaults: ndigits=5
    """
    (ndigits) = param
    fname = fname.split('_')[-1]
    return fname[0:len(fname)-ndigits]


def extract_fieldName(fname,f_split,param):
    """Returns the field name according to the name of the binary file.
    Use the file 'pypStag.data.stagyy-fields'
    Args:
        fname (str): name of the stagyy binary file.
                    e.g. 'llsvp-t00005'
    
    Return:
        fieldName (str): from the file 'pypStag.data.stagyy-fields'
        addfield (bool): set to True if an additional field (eg.
                        Pressure for 'vp' binaries) need to build.
    """
    fname = fname.split('_')[-1]
    ff = f_split(fname,param)
    found = False
    addfield = False
    # search for local
    with open(path2fieldNamesParing,'r') as data:
        for line in data:
            if '->' in line:
                line = line.strip().split('->')
                descriptor = line[0].strip()
                fieldName  = line[1].strip()
                if len(line) == 3 and line[2].strip() == '+':
                    addfield = True
                else:
                    addfield = False
                if descriptor == ff:
                    found = True
                    break
            else:
                pass
    # search for defaults
    if not found:
        with open(path2fieldNamesParingDefaults,'r') as data:
            for line in data:
                if '->' in line:
                    line = line.strip().split('->')
                    descriptor = line[0].strip()
                    fieldName  = line[1].strip()
                    if len(line) == 3 and line[2].strip() == '+':
                        addfield = True
                    else:
                        addfield = False
                    if descriptor == ff:
                        found = True
                        break
                else:
                    pass
    # out
    if not found:
        raise FieldTypeInDevError(fname)
    return fieldName, addfield