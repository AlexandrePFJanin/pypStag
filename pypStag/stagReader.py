# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:23:18 2019

@author: Alexandre Janin
"""

from functools import partial
import numpy as np
from itertools import product


class StagpyError(Exception):

    """Base class for exceptions raised by StagPy.

    Note:
        All exceptions derive from this class. To catch any error that might be
        raised by StagPy due to invalid requests/missing data, you only need to
        catch this exception.
    """

    pass


class ParsingError(StagpyError):

    """Raised when a parsing error occurs."""

    def __init__(self, faulty_file, msg):
        """Initialization of instances:

        Args:
            faulty_file (pathlike): path of the file where a parsing problem
                was encountered.
            msg (str): error message.

        Attributes:
            file (pathlike): path of the file where a parsing problem was
                encountered.
            msg (str): error message.
        """
        self.file = faulty_file
        self.msg = msg
        super().__init__(faulty_file, msg)


def _readbin(fid, fmt='i', nwords=1, file64=False, unpack=True):
    """Read n words of 4 or 8 bytes with fmt format.

    fmt: 'i' or 'f' or 'b' (integer or float or bytes)
    4 or 8 bytes: depends on header

    Return an array of elements if more than one element.

    Default: read 1 word formatted as an integer.
    """
    if fmt in 'if':
        fmt += '8' if file64 else '4'
    elts = np.fromfile(fid, fmt, nwords)
    if unpack and len(elts) == 1:
        elts = elts[0]
    return elts


def fields(fieldfile, only_header=False, only_istep=False):
    """Extract fields data.
    Function derived from stagpy and adapated by Alexandre Janin
    Args:
        fieldfile (:class:`pathlib.Path`): path of the binary field file.
        only_header (bool): when True (and :data:`only_istep` is False), only
            :data:`header` is returned.
        only_istep (bool): when True, only :data:`istep` is returned.
    Returns:
        depends on flags.: :obj:`int`: istep
            If :data:`only_istep` is True, this function returns the time step
            at which the binary file was written.
        :obj:`dict`: header
            Else, if :data:`only_header` is True, this function returns a dict
            containing the header informations of the binary file.
        :class:`numpy.array`: fields
            Else, this function returns the tuple :data:`(header, fields)`.
            :data:`fields` is an array of scalar fields indexed by variable,
            x-direction, y-direction, z-direction, block.
    """
    # something to skip header?
    if not fieldfile.is_file():
        return None
    header = {}
    with fieldfile.open('rb') as fid:
        readbin = partial(_readbin, fid)
        magic = readbin()
        if magic > 8000:  # 64 bits
            magic -= 8000
            readbin()  # need to read 4 more bytes
            readbin = partial(readbin, file64=True)
        # check nb components
        nval = 1
        if magic > 400:
            nval = 4
        elif magic > 300:
            nval = 3
        magic %= 100
        # extra ghost point in horizontal direction
        header['xyp'] = int(magic >= 9 and nval == 4)
        # total number of values in relevant space basis
        # (e1, e2, e3) = (theta, phi, radius) in spherical geometry
        #              = (x, y, z)            in cartesian geometry
        header['nts'] = readbin(nwords=3)
        # number of blocks, 2 for yinyang or cubed sphere
        header['ntb'] = readbin() if magic >= 7 else 1
        # aspect ratio
        header['aspect'] = readbin('f', 2)
        # number of parallel subdomains
        header['ncs'] = readbin(nwords=3)  # (e1, e2, e3) space
        header['ncb'] = readbin() if magic >= 8 else 1  # blocks
        # r - coordinates
        # rgeom[0:self.nrtot+1, 0] are edge radial position
        # rgeom[0:self.nrtot, 1] are cell-center radial position
        if magic >= 2:
            header['rgeom'] = readbin('f', header['nts'][2] * 2 + 1)
        else:
            header['rgeom'] = np.array(range(0, header['nts'][2] * 2 + 1))\
                * 0.5 / header['nts'][2]
        header['rgeom'].resize((header['nts'][2] + 1, 2))
        header['rcmb'] = readbin('f') if magic >= 7 else None
        header['ti_step'] = readbin() if magic >= 3 else 0
        if only_istep:
            return header['ti_step']
        header['ti_ad'] = readbin('f') if magic >= 3 else 0
        header['erupta_total'] = readbin('f') if magic >= 5 else 0
        header['bot_temp'] = readbin('f') if magic >= 6 else 1
        if magic >= 4:
            header['e1_coord'] = readbin('f', header['nts'][0])
            header['e2_coord'] = readbin('f', header['nts'][1])
            header['e3_coord'] = readbin('f', header['nts'][2])
        else:
            # could construct them from other info
            raise ParsingError(fieldfile,
                               'magic >= 4 expected to get grid geometry')
        if only_header:
            return header
        # READ FIELDS
        # number of points in (e1, e2, e3) directions PER CPU
        npc = header['nts'] // header['ncs']
        # number of blocks per cpu
        nbk = header['ntb'] // header['ncb']
        # number of values per 'read' block
        npi = (npc[0] + header['xyp']) * (npc[1] + header['xyp']) * npc[2] * \
            nbk * nval
        header['scalefac'] = readbin('f') if nval > 1 else 1
        flds = np.zeros((nval,
                         header['nts'][0] + header['xyp'],
                         header['nts'][1] + header['xyp'],
                         header['nts'][2],
                         header['ntb']))
        # loop over parallel subdomains
        for icpu in product(range(header['ncb']),
                            range(header['ncs'][2]),
                            range(header['ncs'][1]),
                            range(header['ncs'][0])):
            # read the data for one CPU
            data_cpu = readbin('f', npi) * header['scalefac']
            # icpu is (icpu block, icpu z, icpu y, icpu x)
            # data from file is transposed to obtained a field
            # array indexed with (x, y, z, block), as in StagYY
            flds[:,
                 icpu[3] * npc[0]:(icpu[3] + 1) * npc[0] + header['xyp'],  # x
                 icpu[2] * npc[1]:(icpu[2] + 1) * npc[1] + header['xyp'],  # y
                 icpu[1] * npc[2]:(icpu[1] + 1) * npc[2],  # z
                 icpu[0] * nbk:(icpu[0] + 1) * nbk  # block
                 ] = np.transpose(data_cpu.reshape(
                     (nbk, npc[2], npc[1] + header['xyp'],
                      npc[0] + header['xyp'], nval)))
    return header, flds



def reader_time(path2file):
    """ This function reads a *_time.dat file and return the appropriated fields
    of the current StagMetaData object.
    <i> : directory = str, path to reach the data file
          fname = str, name of the data file
    <o> : list of metadata
    """
    with open(path2file,'r') as data:
        BIN = data.readline() #header
        nod = len(data.readlines())
    with open(path2file,'r') as data:
        BIN = data.readline() #header
        output01 = np.zeros(nod)
        output02 = np.zeros(nod)
        output03 = np.zeros(nod)
        output04 = np.zeros(nod)
        output05 = np.zeros(nod)
        output06 = np.zeros(nod)
        output07 = np.zeros(nod)
        output08 = np.zeros(nod)
        output09 = np.zeros(nod)
        output10 = np.zeros(nod)
        output11 = np.zeros(nod)
        output12 = np.zeros(nod)
        output13 = np.zeros(nod)
        output14 = np.zeros(nod)
        output15 = np.zeros(nod)
        output16 = np.zeros(nod)
        output17 = np.zeros(nod)
        output18 = np.zeros(nod)
        output19 = np.zeros(nod)
        output20 = np.zeros(nod)
        output21 = np.zeros(nod)
        output22 = np.zeros(nod)
        output23 = np.zeros(nod)
        output24 = np.zeros(nod)
        output25 = np.zeros(nod)
        output26 = np.zeros(nod)
        output27 = np.zeros(nod)
        i = 0
        for line in data:
            line = line.strip().split()     #essential step
            output01[i] = float(line[0])
            output02[i] = float(line[1])
            output03[i] = float(line[2])
            output04[i] = float(line[3])
            output05[i] = float(line[4])
            output06[i] = float(line[5])
            output07[i] = float(line[6])
            output08[i] = float(line[7])
            output09[i] = float(line[8])
            output10[i] = float(line[9])
            output11[i] = float(line[10])
            output12[i] = float(line[11])
            output13[i] = float(line[12])
            output14[i] = float(line[13])
            output15[i] = float(line[14])
            output16[i] = float(line[15])
            output17[i] = float(line[16])
            output18[i] = float(line[17])
            output19[i] = float(line[18])
            output20[i] = float(line[19])
            output21[i] = float(line[20])
            output22[i] = float(line[21])
            output23[i] = float(line[22])
            output24[i] = float(line[23])
            output25[i] = float(line[24])
            output26[i] = float(line[25])
            output27[i] = float(line[26])
            i += 1
    return (output01,output02,output03,output04,output05,output06,output07,output08,output09,output10,\
            output11,output12,output13,output14,output15,output16,output17,output18,output19,output20,\
            output20,output21,output23,output24,output25,output26,output27)



def reader_rprof(path2file,column_index=0):
    """ This function reads a *_rprof.dat file and return the appropriated fields
    of the current StagMetaData object.
    <i> : directory = str, path to reach the data file
          fname = str, name of the data file
          column_index = int, index of the column you want to extract when
                         reading a *_rprof.dat file.
    <o> : (istep, time, layers, field)
          itsep  = np.array, list of the istep
          time   = np.array, list of corresponding values of time
          layers = np.array, list of z layers index
          field  = np.array 2D, matrix of the field rprof you extract with
                   the column_index
    """
    #1. Compute the number of zlayers
    nlayer    = 0  #number of layer in the z direction
    lenHeader = 0  #nmuber of lines in the header
    header = []    #textual header
    compute = 'undertermined'
    with open(path2file,'r') as data:
        for line in data:
            BIN = line.strip().split()
            if len(BIN) != 0:
                if BIN[0] == '*******************step:':
                    if compute == 'True':
                        compute = 'False'
                    else:
                        compute = 'True'
                else: #condition 1 to be a header
                    if compute == 'undertermined':
                        lenHeader += 1
                        header.append(BIN)
                if compute   == 'True':
                    nlayer += 1
                elif compute == 'False':
                    break
                else:
                    pass
            else: #condition 2 to be a header
                if compute == 'undertermined':
                    lenHeader += 1
                    header.append(BIN)
        nlayer = nlayer - 1
    #Creation of a layers list:
    layers = np.linspace(1,nlayer,nlayer)
    #2. Compute the number of time steps
    with open(path2file,'r') as data:
        nod = len(data.readlines())  #nod  = total number of lines
    print('>> rprof reader | Total number of line: '+str(nod))
    print('>> rprof reader | Number of nlayers   : '+str(nlayer))
    nots = int((nod-lenHeader)/(nlayer+1))   #nots = number of time steps
    print('>> rprof reader | Number of time steps: '+str(nots))
    print('>> rprof reader | Number data in field: '+str(nots*nlayer))
    #3. Read data
    field = np.zeros(nots*nlayer)
    istep = np.zeros(nots)
    time  = np.zeros(nots)
    #Preparing of the indices
    n = 0 #for all lines read
    i = 0 #for indices in field
    j = 0 #for indices in istep and time
    with open(path2file,'r') as data:
        for line in data :
            if len(line.strip().split()) != 0:
                if n%(nlayer+1) == 0:
                    istep[j] = int(line.strip().split()[1])
                    time[j]  = float(line.strip().split()[5])
                    j += 1
                else:
                    temp = len(line.strip().split())
                    field[i] = float(line.strip().split()[column_index])
                    i += 1
                n += 1
    print('>> rprof reader | Number of columns: '+str(temp))
    #4. Reshape field
    field = np.reshape(field,(nots,nlayer))
    return (header, istep, time, layers, field)
    



def reader_plates_analyse(path2file):
    """ This function reads a *_plates_analyse.dat file and return the appropriated fields
    of the current StagMetaData object.
    <i> : directory = str, path to reach the data file
          fname = str, name of the data file
    <o> : (istep, time, layers, field)
          itsep  = np.array, list of the istep
          time   = np.array, list of corresponding values of time
          mobility = np.array, list of plate mobility
          plateness  = np.array,list of plateness
    """
    #1. Init
    istep = []
    time  = []
    mobility = []
    plateness = []
    with open(path2file,'r') as data:
        data.readline()  #remove header 
        for line in data:
            line = line.strip().split()
            istep.append(int(line[0]))
            time.append(float(line[1]))
            mobility.append(float(line[3]))
            plateness.append(float(line[5]))
    istep = np.array(istep)
    time  = np.array(time)
    mobility = np.array(mobility)
    plateness = np.array(plateness)
    return (istep, time, mobility, plateness)




