# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 18:38:55 2019

@author: Alexandre Janin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
import colorsys


def readCPT(fileName):
    """
    Function that reads CPT file and returns values and RGB code
    """
    values = []
    RGB = []
    with open(fileName,'r') as data:
        for line in data:
            line = line.strip().split()     #essential step
            head = line[0]
            if len(line[0])>1:
                head = line[0].split('#')
                if len(head)>1:
                    head = '#'
            if head != '#' and line[0] not in ['F','B','N']:
                values.append(float(line[0]))
                values.append(float(line[4]))
                RGB.append([float(line[1]),float(line[2]),float(line[3])])
                RGB.append([float(line[5]),float(line[6]),float(line[7])])
    rmList = []
    for i in range(len(values)-1):
        if values[i] == values[i+1] and RGB[i] == RGB[i+1]:
            rmList.append(i)
    rmList = sorted(rmList, reverse=True)
    for delId in rmList:
        del values[delId]
        del RGB[delId]
    RGB_intTuple = []
    for rgb in RGB:
        tuple_rgb = (int(rgb[0]),int(rgb[1]),int(rgb[2]))
        RGB_intTuple.append(tuple_rgb)
    return values,RGB_intTuple



def readCLR(fileName):
    """
    Function that reads CLR file and returns values and RGB code
    """
    values = []
    RGB = []
    with open(fileName,'r') as data:
        for line in data:
            line = line.strip().split()     #essential step
            if line[0] == 'COLORMAP':
                self.bin = line[0]
            else:
                values.append(float(line[0]))
                RGB.append([float(line[1]),float(line[2]),float(line[3])])
    RGB_intTuple = []
    for rgb in RGB:
        tuple_rgb = (int(rgb[0]),int(rgb[1]),int(rgb[2]))
        RGB_intTuple.append(tuple_rgb)
    return values,RGB_intTuple



def readXML(fileName):
    """
    Function that reads XML file and returns values and RGB code
    """
    values = []
    RGB = []
    with open(fileName,'r') as data:
        self.bin = data.readline() #header
        self.bin = data.readline() #header
        for line in data:
            if line.strip().split()[0] != '</ColorMap>' and line.strip().split()[0] != '</ColorMaps>':
                line = line.strip().split("\"")     #essential step
                values.append(float(line[1]))
                RGB.append([float(line[5]),float(line[7]),float(line[9])])
    RGB_intTuple = []
    for rgb in RGB:
        tuple_rgb = (int(rgb[0]),int(rgb[1]),int(rgb[2]))
        RGB_intTuple.append(tuple_rgb)
    return values,RGB_intTuple



def GCMTcolormap(fileName,reverse=False):
    """
    Function coming from the GCMT (Generic Color Mapping Tool) package.
    Copyright 2021 Alexandre Janin, all rights reserved.

    This function returns a usable colormap for python from a GMT-like file
    format in input.
    File extension supported: .cpt, .crl and .xml
    <i> : fileName = str, name (and exact path) of the colormap you want to
                     use in python
          reverse = bool, if reverse == False: the colormap return by this
                    function will be the same as in the input file. if 
                    reverse == True, this function will reverse the read
                    colormap.
    <o> : GMT_import = matplotlib.colors.LinearSegmentedColormap object
    <e.g.> : cmap = GCMTcolormap('../GMT_hot.cpt')
    """
    GMTPath = None
    if type(GMTPath) == type(None):
        filePath = fileName
    else:
        filePath = GMTPath+"/"+ fileName +".cpt"
    try:
        f = open(filePath)
    except:
        print("file ",filePath, "not found")
        return None
    lines = f.readlines()
    f.close()

    fileExtension = fileName.split('.')[1]
    #print('Format of the selected colormap file: '+fileExtension)
    if fileExtension == 'cpt':
        x,RGB = readCPT(fileName)
    elif fileExtension == 'clr':
        x,RGB = readCLR(fileName)
    elif fileExtension == 'xml':
        x,RGB = readXML(fileName)
    x = np.array(x)
    RGB = np.array(RGB)
    r = RGB[:,0]
    g = RGB[:,1]
    b = RGB[:,2]
    colorModel = "RGB"
    for l in lines:
        ls = l.split()
        if l[0] == "#":
            if ls[-1] == "HSV":
                colorModel = "HSV"
                continue
            else:
                continue
    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr,gg,bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
            r[i] = rr ; g[i] = gg ; b[i] = bb
    if colorModel == "HSV":
        for i in range(r.shape[0]):
            rr,gg,bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
            r[i] = rr ; g[i] = gg ; b[i] = bb
    if colorModel == "RGB":
        r = r/255.
        g = g/255.
        b = b/255.
    xNorm = (x - x[0])/(x[-1] - x[0])
    red = []
    blue = []
    green = []
    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
      
    # >> Construction de la colorbar utilisable dans un plot:
    # voir cm de bsemap pour l'exemple d'utilisation
    _LUTSIZE = rcParams['image.lut']
    datad={}
    gmtColormap_dict = colorDict
    GMT_import = colors.LinearSegmentedColormap('CPT_imported_GCMT',\
                                                gmtColormap_dict,\
                                                _LUTSIZE)
    datad['CPT_imported_GCMT'] = gmtColormap_dict

    def _revcmap(data):
        data_r = {}
        for key,val in data.items():
            val = list(val)
            valrev = val[::-1]
            valnew = []
            for a,b,c in valrev:
                valnew.append((1.-a,b,c))
            data_r[key]=valnew
        return data_r
      
    _cmapnames = list(datad.keys())
    for _cmapname in _cmapnames:
        _cmapname_r = _cmapname+'_r'
        _cmapdat_r = _revcmap(datad[_cmapname])
        datad[_cmapname_r] = _cmapdat_r
        locals()[_cmapname_r] = colors.LinearSegmentedColormap(_cmapname_r, _cmapdat_r, _LUTSIZE)

    #sequence to reverse the colorbar:
    if reverse == True:
        gmtColormap_dict_r = datad['CPT_imported_GCMT_r']
        GMT_import_r = colors.LinearSegmentedColormap('CPT_imported_GCMT_r',\
                                                      gmtColormap_dict_r,\
                                                      _LUTSIZE)
        GMT_import = GMT_import_r
    return GMT_import




def colorBar(Val,minVal,maxVal,cptFile):
    """ This function return a RBG tuple corresponding to the value 'Val' 
    between values 'minVal' and 'maxVal' on the colormap described by the .cpt
    file at the path 'cptFile'
    """
    finalRGB = (0,0,0)
    RGBvalues = []   #list of of two tuples per interval
    colorValues = [] #list of values framing a RBG range
    
    #Cpt file reading operation
    file = open(cptFile)
    lines = file.readlines()
    file.close()
    for li in lines:
        ls = li.split()
        exitCondition = False
        if len(ls[0])>1:
            if ls[0][0] == "#" or ls[0][0] in ["B","F","N"]:
                exitCondition = True
        else:
            if ls[0] == "#" or ls[0] in ["B","F","N"]:
                exitCondition = True            
        if exitCondition == False:
            RGBvalues.append([(ls[1],ls[2],ls[3]),(ls[5],ls[6],ls[7])])
            colorValues.append([ls[0],ls[4]])
    index      = 0
    normVal    = (Val-minVal)/(maxVal-minVal)
    normMinCpt = float(colorValues[0][0])
    normMaxCpt = float(colorValues[len(colorValues)-1][1])
    normColorValues = []
    for i in range(len(colorValues)):
        normColorValues.append([(float(colorValues[i][0])-normMinCpt)/(normMaxCpt-normMinCpt),(float(colorValues[i][1])-normMinCpt)/(normMaxCpt-normMinCpt)])
    #First test: if normVal is out the minVal, maxVal interval: = staturation
    if normVal<=0:
        index = 0
        temp1 = np.where(np.array(normColorValues) == np.amin(normColorValues))[0][0]
        R = RGBvalues[temp1][0][0]
        G = RGBvalues[temp1][0][1]
        B = RGBvalues[temp1][0][2]
        finalRGB = (int(R),int(G),int(B))
    elif normVal>=1:
        index = len(normColorValues)-1
        temp1 = np.where(np.array(normColorValues) == np.amax(normColorValues))[0][0]
        R = RGBvalues[temp1][1][0]
        G = RGBvalues[temp1][1][1]
        B = RGBvalues[temp1][1][2]
        finalRGB = (int(R),int(G),int(B))
    else:
        #If No saturation of the colormap:
        for i in range(len(normColorValues)):
            if normVal>normColorValues[i][0] and normVal<=normColorValues[i][1]:
                index = i
                prop1 = (normColorValues[i][1]-normVal)/(normColorValues[i][1]-normColorValues[i][0])
                prop2 = 1-prop1
                R = prop1*float(RGBvalues[i][0][0])+prop2*float(RGBvalues[i][1][0])
                G = prop1*float(RGBvalues[i][0][1])+prop2*float(RGBvalues[i][1][1])
                B = prop1*float(RGBvalues[i][0][2])+prop2*float(RGBvalues[i][1][2])
                finalRGB = (int(R),int(G),int(B))
                break
    return finalRGB



def colorBar_pyqtgraph(Vals,minVal,maxVal,cptFile,glColor=True):
    """ This function return a RBG tuple corresponding to the value 'Val' 
    between values 'minVal' and 'maxVal' on the colormap described by the .cpt
    file at the path 'cptFile'
    """
    
    import pyqtgraph as pg
    
    finalRGB = []    #final output
    RGBvalues = []   #list of of two tuples per interval
    colorValues = [] #list of values framing a RBG range
    
    #Cpt file reading operation
    file = open(cptFile)
    lines = file.readlines()
    file.close()
    for li in lines:
        ls = li.split()
        exitCondition = False
        if len(ls[0])>1:
            if ls[0][0] == "#" or ls[0][0] in ["B","F","N"]:
                exitCondition = True
        else:
            if ls[0] == "#" or ls[0] in ["B","F","N"]:
                exitCondition = True            
        if exitCondition == False:
            RGBvalues.append([(ls[1],ls[2],ls[3]),(ls[5],ls[6],ls[7])])
            colorValues.append([ls[0],ls[4]])
    
    for Val in Vals:
        normVal    = (Val-minVal)/(maxVal-minVal)
        normMinCpt = float(colorValues[0][0])
        normMaxCpt = float(colorValues[len(colorValues)-1][1])
        normColorValues = []
        for i in range(len(colorValues)):
            normColorValues.append([(float(colorValues[i][0])-normMinCpt)/(normMaxCpt-normMinCpt),(float(colorValues[i][1])-normMinCpt)/(normMaxCpt-normMinCpt)])
        #First test: if normVal is out the minVal, maxVal interval: = staturation
        if normVal<=np.amin(normColorValues):
            temp1 = np.where(np.array(normColorValues) == np.amin(normColorValues))[0][0]
            R = RGBvalues[temp1][0][0]
            G = RGBvalues[temp1][0][1]
            B = RGBvalues[temp1][0][2]
            RGBtuple = (int(R),int(G),int(B))
        elif normVal>=np.amax(normColorValues):
            temp1 = np.where(np.array(normColorValues) == np.amax(normColorValues))[0][0]
            R = RGBvalues[temp1][1][0]
            G = RGBvalues[temp1][1][1]
            B = RGBvalues[temp1][1][2]
            RGBtuple = (int(R),int(G),int(B))
        else:
            #If No saturation of the colormap:
            for i in range(len(normColorValues)):
                if normVal>normColorValues[i][0] and normVal<=normColorValues[i][1]:
                    prop1 = (normColorValues[i][1]-normVal)/(normColorValues[i][1]-normColorValues[i][0])
                    prop2 = 1-prop1
                    R = prop1*float(RGBvalues[i][0][0])+prop2*float(RGBvalues[i][1][0])
                    G = prop1*float(RGBvalues[i][0][1])+prop2*float(RGBvalues[i][1][1])
                    B = prop1*float(RGBvalues[i][0][2])+prop2*float(RGBvalues[i][1][2])
                    RGBtuple = (int(R),int(G),int(B))
                    break
        if glColor:
            finalRGB.append(pg.glColor(RGBtuple))
        else:
            finalRGB.append(RGBtuple)
    return finalRGB