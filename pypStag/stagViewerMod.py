# -*- coding: utf-8 -*-
"""
Created on Wed Sep 9 09:32:14 2020

@author: Alexandre
"""


"""
This script contains stagViewer mod for efficient plots
"""

from matplotlib.colors import ListedColormap
import matplotlib
import cartopy.crs as ccrs


class PlotParam:
    """
    Class defining generic plot parameters that can be used automatically
    """
    def __init__(self,fignum=0,title='',suptitle='',xlabel='',ylabel='',figsize=(7,7),show=True,aspect_ratio=1,aspect=None,\
                 projection=ccrs.Robinson(),transform=ccrs.PlateCarree(),gridlines=True,mapticks=True,\
                 antialiased=False,\
                 linecolor='black',linewidth=1,\
                 vscale=None,arrow_width=None,qscale=10,noa=1000,\
                 cbar=True,cbar_location='bottom',cbar_axis=None,cbar_shrink=0.8,cbar_pad=0.05,cbar_aspect=30,cbar_label=None,\
                 log10=False,nol=20,vmax=None,vmin=None,s=5,lw=1.25,edgecolor=None,\
                 cmap='vik',reverseCMAP=False,alpha=1,\
                 save=False,path='Default',name='preview.png',format='Default',dpi=500,\
                 rticks=False,theta_ticks=110):
        import pathlib
        #general figure parameters
        self.fignum   = fignum     # index of the figure
        self.title    = title      # title of the plot
        self.suptitle = suptitle   # superior title of the figure
        self.figsize  = figsize    # size of the figure
        self.show     = show       # if True show the figure else close the figure (plt.close(fig))
        self.aspect_ratio = aspect_ratio # aspect ratio of the figure
        self.aspect   = aspect     # in ['equal',None]
        # mapping and projection
        self.projection = projection
        self.transform  = transform  # ccrs.Geodetic() or ccrs.PlateCarree()   [default: ccrs.PlateCarree()]
        self.gridlines  = gridlines
        self.mapticks   = mapticks   # bool, if you want to display geo ticks on your map
        # rendering
        self.antialiased = antialiased
        # x/y label parametrisation
        self.xlabel      = xlabel     # label on x axis
        self.ylabel      = ylabel     # label on y axis
        self.labelpad    = 2
        self.size_label  = 7
        self.xlabel_rotation = None
        self.ylabel_rotation = None
        # ticks
        self.rticks   = rticks      # Bool, If you want to display radial ticks on polar plot
        self.theta_ticks = theta_ticks # int/float, *DEGREES* angle to represent the ticks in an annulus plot
        #field parameters
        self.log10   = log10      # if True plot the np.log10 of the field
        self.nol     = nol        # number of levels
        self.vmax    = vmax       # upper limit for the color map
        self.vmin    = vmin       # lower limit for the color map
        #color map parameters
        self.crameri_path = '/home/alexandre/Alexandre/Ptoleme/cpt/ScientificColourMaps4/'
        if cmap == 'vik':         #path of the cmap of GCMT
            self.cmap     = self.crameri_path + 'vik/vik.cpt'
            self.cmaptype = 'cmap'
        elif cmap == 'oslo':
            self.cmap     = self.crameri_path + 'oslo/oslo.cpt'
            self.cmaptype = 'cmap'
        elif isinstance(cmap,ListedColormap):
            self.cmap     = cmap
            self.cmaptype = 'matplotlib'
        else:
            self.cmap     = cmap
            self.cmaptype = 'perso'
        self.reverseCMAP = reverseCMAP  # if True will reverse the color map
        # arrows
        self.noa         = noa         # number of arrows
        self.vscale      = vscale
        self.arrow_width = arrow_width
        self.qscale      = qscale
        # scatter plot parameter
        self.s  = s         # size of points
        self.lw = lw        # linewidth parameter
        self.edgecolor = edgecolor # edgecolor of point
        self.alpha = alpha
        #
        self.linewidth = linewidth
        self.linecolor = linecolor
        #color bar position
        self.cbar = cbar
        self.cbar_axis = cbar_axis # [left, bottom, width, height]
        self.cbar_orientation ='horizontal'
        self.cbar_location = cbar_location  #can be 'top','bottom','right','left'
        self.cbar_shrink = cbar_shrink
        self.cbar_pad = cbar_pad
        self.cbar_aspect = cbar_aspect
        self.cbar_labelsize = 10
        self.cbar_label = cbar_label
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
            self.cmap     = self.crameri_path + 'vik/vik.cpt'
            self.cmaptype = 'cmap'
        elif self.cmap == 'lajolla':
            self.cmap     = self.crameri_path + 'lajolla/lajolla.cpt'
            self.cmaptype = 'cmap'
        elif self.cmap == 'davos':
            self.cmap     = self.crameri_path + 'davos/davos.cpt'
            self.cmaptype = 'cmap'
        elif self.cmap == 'oslo':
            self.cmap     = self.crameri_path + 'oslo/oslo.cpt'
            self.cmaptype = 'cmap'
        elif isinstance(self.cmap,ListedColormap):
            self.cmaptype = 'matplotlib'
        else:
            self.cmaptype = 'perso'
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




