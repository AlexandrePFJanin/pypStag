# -*- coding: utf-8 -*-
"""
Created on Wed Sep 9 09:32:14 2020

@author: Alexandre
"""


"""
This script contains stagViewer mod for efficient plots
"""





class PlotParam:
    """
    Class defining generic plot parameters that can be used automatically
    """
    def __init__(self,fignum=0,title='',suptitle='',xlabel='',ylabel='',figsize=(7,7),show=True,\
                 log10=False,nol=20,maxVal='Default',minVal='Default',\
                 cmap='vik',reverseCMAP=False,\
                 save=False,path='Default',name='preview.png',format='Default',dpi=500):
        import pathlib
        #general figure parameters
        self.fignum   = fignum     # index of the figure
        self.title    = title      # title of the plot
        self.suptitle = suptitle   # superior title of the figure
        self.figsize  = figsize    # size of the figure
        self.show     = show       # if True show the figure else close the figure (plt.close(fig))
        # x/y label parametrisation
        self.xlabel      = xlabel     # label on x axis
        self.ylabel      = ylabel     # label on y axis
        self.labelpad    = 2
        self.size_label  = 7
        self.xlabel_rotation = None
        self.ylabel_rotation = None
        #field parameters
        self.log10   = log10      # if True plot the np.log10 of the field
        self.nol     = nol        # number of levels
        self.maxVal  = maxVal     # upper limit for the color map
        self.minVal  = minVal     # lower limit for the color map
        #color map parameters
        self.crameri_path = '/home/alexandre/Alexandre/Ptoleme/cpt/ScientificColourMaps4/'
        if cmap == 'vik':         #path of the cmap of GCMT
            self.cmap = self.crameri_path + 'vik/vik.cpt'
        elif cmap == 'oslo':
            self.cmap = self.crameri_path + 'oslo/oslo.cpt'
        self.reverseCMAP = reverseCMAP  # if True will reverse the color map
        #color bar position
        self.cbar_orientation ='vertical'
        self.cbar_position = 'right'  #can be 'top','bottom','right','left'
        self.cbar_size = '5%'
        self.cbar_pad = 0.05
        self.cbar_labelsize = 6
        self.cbar_label = ''
        self.cbar_drawedges = False
        #export parameters
        self.save      = save     # if True save the figure with the export parameters
        if path == 'Default':     # path for the exported file
            self.path  = str(pathlib.Path.cwd())+'/' #set the current path of the run
        else:
            self.path  = path 
        self.name      = name     # name (with extension) of the file generated
        self.format    = format
        self.dpi       = dpi      # dots per inches
        #ticks parameters
        self.size_suptitle = 10
        self.size_title    = 8
        self.size_ticks    = 6
        #spine
        self.spine_linewidth  = 0.5

    


    def update(self,axis=None):
        """
        This function will just update all the field correctly. Have to be done
        before using the class instance (better), for example compil correctly 
        cmap shortcuts.
        If an axis is put into argument of this function, this function will
        automatically apply all 'cosmetic' parametrisation contained into the
        current instance of PlotParam class like tick_param, xlabel, title, etc.
        """
        #cmap shortcuts managments
        if self.cmap == 'vik':         #path of the cmap of GCMT
            self.cmap        = self.crameri_path + 'vik/vik.cpt'
        elif self.cmap == 'oslo':
            self.cmap        = self.crameri_path + 'oslo/oslo.cpt'
        #exporting format managment
        if self.format == 'Default':
            self.format = self.name.split('.')[1] # format of the file (default: extracted from the name)
        else:
            if len(self.name.split('.')) > 1: #means that the file name already contains an extension
                self.name = self.name.split('.')[0]+self.format #add the good format
            else: #means that the user didn't give a file name with format extension
                self.name += self.format
        # ========================== If axis is given in input: 
        if axis != None:
            self.set_ticks(axis)
            self.set_labels(axis)
            self.set_title(axis)
            self.set_spines(axis)

    
    def set_ticks(self,axis):
        """
        This function apply automatically on the given axis parametrisation
        of ticks and ticks only
        """
        axis.tick_params(axis='both', which='major', labelsize=self.size_ticks) 
        axis.tick_params(axis='both', which='minor', labelsize=self.size_ticks)


    def set_labels(self,axis):
        """
        This function apply automatically on the given axis parametrisation
        of labels and labels only
        """
        if self.xlabel_rotation != None:
            axis.set_xlabel(self.xlabel,size=self.size_label, rotation=self.xlabel_rotation, labelpad=self.labelpad)
        else:
            axis.set_xlabel(self.xlabel,size=self.size_label, labelpad=self.labelpad)
        if self.ylabel_rotation != None:
            axis.set_ylabel(self.ylabel,size=self.size_label, rotation=self.ylabel_rotation, labelpad=self.labelpad)
        else:
            axis.set_ylabel(self.ylabel,size=self.size_label, labelpad=self.labelpad)
    

    def set_title(self,axis):
        """
        This function apply automatically on the given axis parametrisation
        of axis title and axis title only
        """
        axis.set_title(self.title,size=self.size_title)
    

    def set_spines(self,axis):
        """
        This function apply automatically on the given axis parametrisation
        of axis spines and axis spines only
        """
        axis.spines["top"].set_linewidth(self.spine_linewidth)
        axis.spines["bottom"].set_linewidth(self.spine_linewidth)
        axis.spines["right"].set_linewidth(self.spine_linewidth)
        axis.spines["left"].set_linewidth(self.spine_linewidth)


    def set_cbar(self,figure,axis,cmap):
        """
        This function return a cbar (<class 'matplotlib.colorbar.Colorbar'> object)
        using all colorbar parameters in the current class instance
        to parameters in the current instance of PlotParam.
        <e.g> : >> fig, ax1 = plt.subplots()
                >> cmap = ax1.contourf(data)
                >> plotparam.cbar(fig,ax1,cmap)    #not more difficult...
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axis)
        cax = divider.append_axes(self.cbar_position, size=self.cbar_size, pad=self.cbar_pad)
        cbar = figure.colorbar(cmap, cax=cax, orientation=self.cbar_orientation, label=self.cbar_label,\
                               drawedges=self.cbar_drawedges)
        cbar.ax.tick_params(labelsize=self.cbar_labelsize)
        return cbar




