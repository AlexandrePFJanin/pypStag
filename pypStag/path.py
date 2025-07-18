# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    pypStag: list of path
"""


# External dependencies
import os
import pypStag

# -----------------------------
# Path to data used by pypStag
# -----------------------------

# Path to the stagYY field names
path2pypStag = os.path.abspath(pypStag.__file__)
path2fieldNamesParing         = '/'.join(path2pypStag.split('/')[0:-1])+'/fields/stagyy-fields-local'
path2fieldNamesParingDefaults = '/'.join(path2pypStag.split('/')[0:-1])+'/fields/stagyy-fields-defaults'
            

