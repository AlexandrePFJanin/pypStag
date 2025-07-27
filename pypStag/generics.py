# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Generic routines
"""

# External dependencies:
import numpy as np
from termcolor import colored


# ----------------- FUNCTIONS -----------------


def im(textMessage,pName,verbose,error=False,warn=False,structure=True,end=True):
    """Print verbose internal message. This function depends on the
    argument verbose. If verbose, then the message will be displayed
    in the terminal.
    
    Args:
        textMessage = str, message to display
        pName = str, name of the subprogram
        verbose = bool, condition for the verbose output
    """
    if verbose and not error:
        if not warn:
            msgc = None
        else:
            msgc = 'yellow'
        if structure:
            if end:
                print(colored('>> '+pName+'| ','blue')+colored('--- ','magenta')+colored(textMessage, msgc))
            else:
                print(colored('>> '+pName+'| ','blue')+colored(' : ','magenta')+colored(textMessage, msgc))
        else:
            print(colored('>> '+pName+'| ','blue')+colored(textMessage, msgc))
    if error:
        #print error message
        print(colored('>> '+pName+'| --- ----- ---','red'))
        print(colored('>> '+pName+'| --- ERROR ---','red'))
        print(colored('>> '+pName+'| --- ----- ---','red'))
        print(colored('>> '+pName+'| '+textMessage,'red'))
        raise AssertionError()


def line_count(filename):
    """
    Returns the number of lines in a file.
    """
    import subprocess
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])


def intstringer(iint,strl):
    """
    Transforms an input integer 'iint' into a string format according
    to a string length 'strl' (condition: strl >= len(str(iint))  ).
    e.g. >> intstringer(4000,5)
         << '04000'
    Args:
        iint = int, input integer you want to transform into a string
        strl = int, length of the output string
    Returns:
        ostr = str, output string
    """
    class BaseError(Exception):
        """Base class for exceptions raised"""
        pass
    class intstringError(BaseError):
        def __init__(self):
            super().__init__('The length of the intput integer is higher '+\
                             'than the length of the string requested length')
    ostr = str(iint)
    if len(ostr) <= strl:
        ostr = '0'*(strl-len(ostr))+ostr
        return ostr
    else:
        raise intstringError()


def resampling_coord(coords,sampling):
    """This function resamples coords vector according to the sampling rate.
    Sampling must be an integer. If sampling ==1 the output is the input.
    If sampling == 2, the output is twince smaller than the input.
    Returns the new coordinate vector after the resampling and the mask of
    keeped (1) and removed (0) elements.
    """
    ind  = np.arange(len(coords))
    mask = np.zeros(len(ind),dtype=bool)
    mask[ind[::sampling]] = True
    mask[-1] = True # make sure to keep the first and the last index
    new_coords = coords[mask].copy()
    mask.dtype = np.int8
    return (new_coords, mask)
