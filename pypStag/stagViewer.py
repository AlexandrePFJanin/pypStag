# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:45:14 2019

@author: Alexandre Janin
"""


"""
This script contains routines for efficient 2D/3D plots
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from .stagError import VisuGridGeometryError, StagTypeError
from .stagData import StagData, StagCartesianGeometry, StagYinYangGeometry
from .stagData import SliceData, CartesianSliceData, YinYangSliceData
from .stagData import InterpolatedSliceData
from .stagViewerMod import PlotParam
from .stagInterpolator import sliceInterpolator





def im(textMessage,pName,verbose):
    """Print verbose internal message. This function depends on the
    argument of self.verbose. If self.verbose == True then the message
    will be displayed on the terminal.
    <i> : textMessage = str, message to display
          pName = str, name of the subprogram
          verbose = bool, condition for the verbose output
    """
    if verbose == True:
        print('>> '+pName+'| '+textMessage)




def align_yaxis_twinx(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:,1] / (extrema[:,1] - extrema[:,0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0,1] = extrema[0,0] + tot_span * (extrema[0,1] - extrema[0,0])
    extrema[1,0] = extrema[1,1] + tot_span * (extrema[1,0] - extrema[1,1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]




def align_axis_x_subplots(ax, ax_target):
        """Make x-axis of `ax` aligned with `ax_target` in figure"""
        posn_old, posn_target = ax.get_position(), ax_target.get_position()
        ax.set_position([posn_target.x0, posn_old.y0, posn_target.width, posn_old.height])




def stagCartesian3DMap(stagData,plotparam='Default',plan='yz',indexing=0,\
                       stagVelocity=None,veloScale=None,veloWidth=None,\
                       aspect_ratio=1,Qscale=1000):
    """
    Makes a plot of stagData.StagCartesianGeometry object field according
    to a projection plan and an indexing on this plan.
    <i> : stagData = stagData.StagCartesianGeometry object
          plotparam = stagViewerMod.PlotParam object describing all your plot
                      parameters (Default, plotparam is automatically generated)
          plan = str, plan direction on which you want a figure.
                 plan have to take values in ['xy','yx','xz','zx','yz','zy']
                 (Default, plan = 'yz')
          indexing = int, index of the plan in the input stagData geometry.
                     If plan = 'xy', indexing refers to depth level.
                     If stagData.geometry = 'cart2D' indexing have to be 0
                     (Default, indexing = 0)
          stagVelocity = stagData.StagCartesianGeometry object containing a
                         velocity field (stagVelocity.fieldType = 'Velocity')
                         If given, will plot the vector field of velocity
                         according to the same indexing and plan.
                         (Default, stagVelocity = None)
          veloScale = int/float, length scaling factor for velocity
                      vectors if a stagVelocity is given.
                      (Default, veloScale = None)
          veloWidth = int/float, width scaling factor for velocity
                      vectors if a stagVelocity is given.
                      (Default, veloWidth = None)
          Qscale = int, length of the legend vector
                   (Default, Qscale = 1000)
          aspect_ratio = int, aspect ratio of the plot
                         (Default, aspect_ratio = 1)
    """
    pName = 'stagCartesian3DMap'
    # Test geometry:
    if stagData.geometry != 'cart2D' and stagData.geometry != 'cart3D':
        raise VisuGridGeometryError(stagData.geometry,'cart2D or cart3D')
    # Test colormap:
    try:
        from .cptReader import GCMTcolormap
        cmap = GCMTcolormap(plotparam.cmap,reverse=plotparam.reverseCMAP)
    except:
        im("WARNING: Unknown colormap file",pName,True)
        cmap = plt.cm.seismic
    if plotparam == 'Default':
        plotparam = PlotParam()
    else:
        plotparam.update()
    # Log10:
    if plotparam.log10:
        stagfield = np.log10(stagData.v)
    else:
        stagfield = stagData.v
    # title
    if plotparam.title == '':
        field = stagData.fieldType
        if plotparam.log10:
            field = 'log10('+stagData.fieldType+')'
        title = 'stagCartesian3DMap: plan='+plan+' field='+field
    else:
        title = plotparam.title
    # others
    loc = indexing
    if plotparam.minVal == 'Default':
        minVal = np.amin(stagfield)
    else:
        minVal = plotparam.minVal
    if plotparam.maxVal == 'Default':
        maxVal = np.amax(stagfield)
    else:
        maxVal = plotparam.maxVal
    levels = np.linspace(minVal,maxVal,plotparam.nol)
    
    # ------- Figure --------
    
    #---- Type I : velocity and 'cut' map: Multiple axis plot
    if isinstance(stagVelocity,StagCartesianGeometry) and plan != 'xy' and plan != 'yx':

        kw = {'height_ratios':[1,4]}
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw=kw,figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
        # Adjust inter subplots space
        plt.subplots_adjust(hspace=0.03)
        # SUBPLOT 1: Horizontal and vertical velocity
        if plan == 'yz' or plan == 'zy':
            ax1.plot([np.amin(stagVelocity.y[indexing,:,-1]),np.amax(stagVelocity.y[indexing,:,-1])],[0,0],'--',color='black',alpha=0.5)
            ax1.plot(stagVelocity.y[indexing,:,-1],stagVelocity.vy[indexing,:,-1],'o',color='#FF6600',alpha=0.7)
            ax1.plot(stagVelocity.y[indexing,:,-1],stagVelocity.vy[indexing,:,-1],'-',color='#FF6600',label=r'$V_y^{surf}$')
            ax1.set_ylabel('Horizontal Velocity',color='#FF6600')
            ax1.tick_params(axis='y', labelcolor='#FF6600')
            ax1bis = ax1.twinx()   # instantiate a second axes that shares the same x-axis
            ax1bis.plot(stagVelocity.y[indexing,:,-1],stagVelocity.vz[indexing,:,-1],'o',color='#0202D1',alpha=0.7)
            ax1bis.plot(stagVelocity.y[indexing,:,-1],stagVelocity.vz[indexing,:,-1],'-',color='#0202D1',label=r'$V_z^{surf}$',alpha=1)
            ax1bis.set_ylabel('Vertcial Velocity',color='#0202D1')
            ax1bis.tick_params(axis='y', labelcolor='#0202D1')
            align_yaxis_twinx(ax1,ax1bis)    # Give the same 0
            ax1.legend(loc='lower left')     # legend
            ax1bis.legend(loc='lower right')  # legend
            ax1.set_title(title)
        elif plan == 'xz' or plan == 'zx':
            ax1.plot([np.amin(stagVelocity.x[:,indexing,-1]),np.amax(stagVelocity.x[:,indexing,-1])],[0,0],'--',color='black',alpha=0.5)
            ax1.plot(stagVelocity.x[:,indexing,-1],stagVelocity.vx[:,indexing,-1],'o',color='#FF6600',alpha=0.7)
            ax1.plot(stagVelocity.x[:,indexing,-1],stagVelocity.vx[:,indexing,-1],'-',color='#FF6600',label=r'$V_x^{surf}$')
            ax1.set_ylabel('Horizontal Velocity',color='#FF6600')
            ax1.tick_params(axis='y', labelcolor='#FF6600')
            ax1bis = ax1.twinx()   # instantiate a second axes that shares the same x-axis
            ax1bis.plot(stagVelocity.x[:,indexing,-1],stagVelocity.vz[:,indexing,-1],'o',color='#0202D1',alpha=0.7)
            ax1bis.plot(stagVelocity.x[:,indexing,-1],stagVelocity.vz[:,indexing,-1],'-',color='#0202D1',label=r'$V_z^{surf}$',alpha=1)
            ax1bis.set_ylabel('Vertcial Velocity',color='#0202D1')
            ax1bis.tick_params(axis='y', labelcolor='#0202D1')
            align_yaxis_twinx(ax1,ax1bis)    # Give the same 0
            ax1.legend(loc='lower left')     # legend
            ax1bis.legend(loc='lower right')  # legend
            ax1.set_title(title)
        
        if plan == 'xz' or plan == 'zx':
            cmap = ax2.contourf(stagData.x[:,loc,:],stagData.z[:,loc,:],stagfield[:,loc,:],levels=levels,cmap=cmap,extend='both')
            Q = ax2.quiver(stagVelocity.x[:,loc,:].flatten(),stagVelocity.z[:,loc,:].flatten(),stagVelocity.vx[:,loc,:].flatten(),stagVelocity.vz[:,loc,:].flatten(),\
                        scale=veloScale,width=veloWidth,label='Velocity field')
            qq = ax2.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
            ax2.legend(loc='lower right')
            ax2.set_xlabel('x-axis')
            ax2.set_ylabel('z-axis')

        elif plan == 'yz' or plan == 'zy':
            cmap = ax2.contourf(stagData.y[loc,:,:],stagData.z[loc,:,:],stagfield[loc,:,:],levels=levels,cmap=cmap,extend='both')
            Q = ax2.quiver(stagVelocity.y[loc,:,:].flatten(),stagVelocity.z[loc,:,:].flatten(),stagVelocity.vy[loc,:,:].flatten(),stagVelocity.vz[loc,:,:].flatten(),\
                        scale=veloScale,width=veloWidth,label='Velocity field')
            qq = ax2.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
            ax2.legend(loc='lower right')
            ax2.set_xlabel('y-axis')
            ax2.set_ylabel('z-axis')

        #To align axis:
        align_axis_x_subplots(ax2,ax1)
    
    #---- Type II : Velocity but iso-depth map: Single axe plot
    elif isinstance(stagVelocity,StagCartesianGeometry):

        fig, ax = plt.subplots(1,1,figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
        ax.set_title(title)
        cmap = ax.contourf(stagData.x[:,:,loc],stagData.y[:,:,loc],stagfield[:,:,loc],levels=levels,cmap=cmap,extend='both')
        Q = ax.quiver(stagVelocity.x[:,:,loc].flatten(),stagVelocity.y[:,:,loc].flatten(),stagVelocity.vx[:,:,loc].flatten(),stagVelocity.vy[:,:,loc].flatten(),\
                    scale=veloScale,width=veloWidth,label='Horizontal Velocity field')
        qq = ax.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
        ax.legend(loc='lower right')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
    
    #---- Type III : No velocity: Single axe plot
    else:

        fig, ax = plt.subplots(1,1,figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
        ax.set_title(title)

        if plan == 'xz' or plan == 'zx':
            cmap = ax.contourf(stagData.x[:,loc,:],stagData.z[:,loc,:],stagfield[:,loc,:],levels=levels,cmap=cmap,extend='both')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('z-axis')

        elif plan == 'yz' or plan == 'zy':
            cmap = ax.contourf(stagData.y[loc,:,:],stagData.z[loc,:,:],stagfield[loc,:,:],levels=levels,cmap=cmap,extend='both')
            ax.set_xlabel('y-axis')
            ax.set_ylabel('z-axis')
        
        elif plan == 'xy' or plan == 'yx':
            cmap = ax.contourf(stagData.x[:,:,loc],stagData.y[:,:,loc],stagfield[:,:,loc],levels=levels,cmap=cmap,extend='both')
            ax.set_xlabel('x-axis')
            ax.set_ylabel('y-axis')

    # Now adding the colorbar
    cbaxes = fig.add_axes([0.91, 0.3, 0.01, 0.4]) # [left, bottom, width, height]
    cbar = plt.colorbar(cmap,cax=cbaxes)

    # --- end ---
    if plotparam.save:
        print("Save images under:\n"+plotparam.path+plotparam.name)
        plt.savefig(plotparam.path+plotparam.name,dpi=plotparam.dpi)
    if plotparam.show:
        fig.show()
    else:
        plt.close(fig)








def sliceMap(sliceData,plotparam='Default',aspect_ratio=1,sliceVelocity=None,\
    vresampling=[1,1],veloScale=None,veloWidth=None,Qscale=1000, \
    projection=None):
    """
    Makes a map of stagData.InterpolatedSliceData or stagData.CartesianSliceData object.
    <i> : sliceData = stagData.InterpolatedSliceData or stagData.CartesianSliceData object
          plotparam = stagViewerMod.PlotParam object describing all your plot
                      parameters (Default, plotparam is automatically generated)
          sliceVelocity = stagData.InterpolatedSliceData or stagData.CartesianSliceData
                          containing a velocity field (sliceVelocity.fieldType = 'Velocity')
                          If given, will plot the vector field of velocity
                          according to the same indexing and plan.
                          (Default, sliceVelocity = None)
          vresampling = list/tuple/array of size 2. Define the x and y
                        resampling parameters, repsectivelly.
          veloScale = int/float, length scaling factor for velocity
                      vectors if a sliceVelocity is given.
                      (Default, veloScale = None)
          veloWidth = int/float, width scaling factor for velocity
                      vectors if a sliceVelocity is given.
                      (Default, veloWidth = None)
          Qscale = int, length of the legend vector
                   (Default, Qscale = 1000)
          aspect_ratio = int, aspect ratio of the plot
                         (Default, aspect_ratio = 1)
    """
    pName = 'sliceMap'
    im('Creation of the sliceMap',pName,True)
    #Typing:
    if not isinstance(sliceData,InterpolatedSliceData) and not isinstance(sliceData,CartesianSliceData):
        raise StagTypeError(str(type(sliceData)),'stagData.InterpolatedSliceData or stagData.CartesianSliceData')
    # Test geometry:
    if sliceData.geometry == 'cart2D' or sliceData.geometry == 'annulus':
        raise VisuGridGeometryError(sliceData.geometry,'interpolated or cart3D')
    # Test colormap:
    try:
        from .cptReader import GCMTcolormap
        cmap = GCMTcolormap(plotparam.cmap,reverse=plotparam.reverseCMAP)
    except:
        im("WARNING: Unknown colormap file",pName,True)
        cmap = plt.cm.seismic
    if plotparam == 'Default':
        plotparam = PlotParam()
    else:
        plotparam.update()
    # Log10:
    if plotparam.log10:
        im('Requested: log10',pName,True)
        slicefield = np.log10(sliceData.v)
    else:
        slicefield = sliceData.v
    # title
    if plotparam.title == '':
        field = sliceData.fieldType
        if plotparam.log10:
            field = 'log10('+sliceData.fieldType+')'
        title = 'sliceMap: field='+field
    else:
        title = plotparam.title
    # others
    if plotparam.minVal == 'Default':
        minVal = np.amin(slicefield)
    else:
        minVal = plotparam.minVal
    if plotparam.maxVal == 'Default':
        maxVal = np.amax(slicefield)
    else:
        maxVal = plotparam.maxVal
    levels = np.linspace(minVal,maxVal,plotparam.nol)

    # ------- Figure --------
    
    #---- Type I : InterpolatedSliceData
    if isinstance(sliceData,InterpolatedSliceData):
        im('  - sliceMap from InterpolatedSliceData',pName,True)
        if isinstance(sliceVelocity,InterpolatedSliceData):
            im('  - velocities: True',pName,True)
            fig = plt.figure(figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
            ax = fig.add_subplot(1, 1, 1,projection=projection)
            ax.set_title(title)
            X = sliceData.x.reshape(sliceData.nxi,sliceData.nyi)
            Y = sliceData.y.reshape(sliceData.nxi,sliceData.nyi)
            V = slicefield.reshape(sliceData.nxi,sliceData.nyi)
            if projection == None:
                cmap = ax.contourf(X,Y,V,levels=levels,cmap=cmap,extend='both')
                ax.set_xlabel('x-axis')
                ax.set_ylabel('y-axis')
            else:
                cmap = ax.imshow(V.T, extent=(0,360,90,-90),cmap=cmap,transform=ccrs.PlateCarree(),vmin=minVal,vmax=maxVal)
            X  = sliceVelocity.x.reshape(sliceVelocity.nxi,sliceVelocity.nyi)[::vresampling[0],::vresampling[1]]
            Y  = sliceVelocity.y.reshape(sliceVelocity.nxi,sliceVelocity.nyi)[::vresampling[0],::vresampling[1]]
            #
            Vx = sliceVelocity.vphi.reshape(sliceVelocity.nxi,sliceVelocity.nyi)[::vresampling[0],::vresampling[1]]
            Vy = -sliceVelocity.vtheta.reshape(sliceVelocity.nxi,sliceVelocity.nyi)[::vresampling[0],::vresampling[1]]
            #
            if projection != None:
                Q = ax.quiver(X.flatten(),Y.flatten(),Vx.flatten(),Vy.flatten(),\
                             scale=veloScale,width=veloWidth,label='Horizontal Velocity field',\
                             transform=ccrs.PlateCarree())
            else:
                Q = ax.quiver(X.flatten(),Y.flatten(),Vx.flatten(),Vy.flatten(),\
                             scale=veloScale,width=veloWidth,label='Horizontal Velocity field')
            qq = ax.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
            ax.legend(loc='lower right')
        else:
            im('  - velocities: False',pName,True)
            fig = plt.figure(figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            ax.set_title(title)
            X = sliceData.x.reshape(sliceData.nxi,sliceData.nyi)
            Y = sliceData.y.reshape(sliceData.nxi,sliceData.nyi)
            V = slicefield.reshape(sliceData.nxi,sliceData.nyi)
            if projection == None:
                cmap = ax.contourf(X,Y,V,levels=levels,cmap=cmap,extend='both')
                ax.set_xlabel('x-axis')
                ax.set_ylabel('y-axis')
            else:
                cmap = ax.imshow(V.T, extent=(0,360,90,-90),cmap=cmap,transform=ccrs.PlateCarree(),vmin=minVal,vmax=maxVal)
        # Now adding the colorbar
        cbaxes = fig.add_axes([0.91, 0.3, 0.01, 0.4]) # [left, bottom, width, height]
        cbar = plt.colorbar(cmap,cax=cbaxes)
        # --- end ---
        if plotparam.save:
            print("Save images under:\n"+plotparam.path+plotparam.name)
            plt.savefig(plotparam.path+plotparam.name,dpi=plotparam.dpi)
        if plotparam.show:
            fig.show()
        else:
            plt.close(fig)
    
    #---- Type II : CartesianSliceData
    else:
        im('  - sliceMap from CartesianSliceData',pName,True)
        if isinstance(sliceVelocity,CartesianSliceData):
            im('  - velocities: True',pName,True)
            # Expand all dimension on a copy of the input data
            im('  - Clone the SliceData objects for dimExpand',pName,True)
            sliceClone         = sliceData
            sliceVelocityCloce = sliceVelocity
            im('  - Prepare data for plot operation',pName,True)
            sliceClone.dimExpand(sliceData.axis)
            sliceVelocityCloce.dimExpand(sliceData.axis)
            sliceVelocity = StagCartesianGeometry('cart3D') #adap, need to be a StagCartesianGeometry
            sliceVelocity.x = sliceVelocityCloce.x
            sliceVelocity.y = sliceVelocityCloce.y
            sliceVelocity.z = sliceVelocityCloce.z
            sliceVelocity.vx = sliceVelocityCloce.vx
            sliceVelocity.vy = sliceVelocityCloce.vy
            sliceVelocity.vz = sliceVelocityCloce.vz
            if sliceData.axis == 0:
                plan = 'yz'
            elif sliceData.axis == 1:
                plan = 'xz'
            elif sliceData.axis == 2:
                plan = 'xy'
            im('  - Call stagCartesian3DMap',pName,True)
            stagCartesian3DMap(sliceClone,plotparam=plotparam,plan=plan,indexing=0,aspect_ratio=aspect_ratio,\
                stagVelocity=sliceVelocity,veloScale=veloScale,veloWidth=veloWidth,Qscale=Qscale)
        else:
            im('  - velocities: False',pName,True)
            # Expand all dimension on a copy of the input data
            im('  - Clone the SliceData input object for dimExpand',pName,True)
            sliceClone = sliceData
            im('  - Prepare data for plot operation',pName,True)
            sliceClone.dimExpand(sliceData.axis)
            if sliceData.axis == 0:
                plan = 'yz'
            elif sliceData.axis == 1:
                plan = 'xz'
            elif sliceData.axis == 2:
                plan = 'xy'
            im('  - Call stagCartesian3DMap',pName,True)
            stagCartesian3DMap(sliceClone,plotparam=plotparam,plan=plan,indexing=0,aspect_ratio=aspect_ratio)
            






def spaceTimeDiagram(stagCloudData,layer=-1,timepar=0,\
                     plotparam='Default',aspect_ratio=1):
    """
    """
    pName = 'spaceTimeDiagram'
    # read the first file to initiate field    
    stagCloudData.iterate()
    sld = SliceData(geometry=stagCloudData.geometry)
    sld.verbose = False
    sld.sliceExtractor(stagCloudData.drop,layer)
    stagCloudData.reset()
    tSlice  = np.zeros((stagCloudData.nt,len(sld.v)))
    axedata = np.zeros((stagCloudData.nt,len(sld.v)))
    simuAge = np.zeros((stagCloudData.nt,len(sld.v)))
    ti_step = np.zeros((stagCloudData.nt,len(sld.v)))
    fileid  = np.zeros((stagCloudData.nt,len(sld.v)))
    # true run
    from tqdm import tqdm
    if stagCloudData.drop.fieldNature == 'Scalar':
        for i in tqdm(stagCloudData.indices):
            stagCloudData.iterate() 
            sld = SliceData(geometry=stagCloudData.geometry)
            sld.verbose = False
            sld.sliceExtractor(stagCloudData.drop,layer)
            # --- write ---
            tSlice[i,:]    = sld.v
            simuAge[i,:]   = np.ones(len(sld.v))*stagCloudData.drop.simuAge
            ti_step[i,:]   = np.ones(len(sld.v))*stagCloudData.drop.ti_step
            fileid[i,:]    = np.ones(len(sld.v))*i
            if sld.plan in ['xz','zx']:
                    axedata[i,:] = sld.x
            elif sld.plan in ['yz','zy']:
                    axedata[i,:] = sld.y
            else:
                    print('ERROR: break operation')
                    break
    #---------- Figure ----------#
    # Test colormap:
    try:
        from .cptReader import GCMTcolormap
        cmap = GCMTcolormap(plotparam.cmap,reverse=plotparam.reverseCMAP)
    except:
        im("WARNING: Unknown colormap file",pName,True)
        cmap = plt.cm.seismic
    plotparam.update()
    # Log10:
    if plotparam.log10:
        slicefield = np.log10(tSlice)
    else:
        slicefield = tSlice
    # title
    title = 'Space-Time diagram: '+stagCloudData.drop.fname+': '+stagCloudData.drop.fieldType
    # others
    if plotparam.minVal == 'Default':
        minVal = np.amin(slicefield)
    else:
        minVal = plotparam.minVal
    if plotparam.maxVal == 'Default':
        maxVal = np.amax(slicefield)
    else:
        maxVal = plotparam.maxVal
    # build figure instance
    fig = plt.figure(figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    if timepar == 0:
        ax.set_xlabel('Time (file index)')
        timex = fileid
    elif timepar == 1:
        ax.set_xlabel('Time (internal simulation ages)')
        timex = simuAge
    elif timepar == 2:
        ax.set_xlabel('Time (simulation time steps)')
        timex = ti_step
    # plot
    cmap = ax.pcolormesh(timex,axedata,slicefield,shading='nearest',cmap=cmap,vmin=minVal,vmax=maxVal)
    ax.set_ylabel('Space')
    # Now adding the colorbar
    cbaxes = fig.add_axes([0.91, 0.3, 0.01, 0.4]) # [left, bottom, width, height]
    cbar = plt.colorbar(cmap,cax=cbaxes)
    # --- end ---
    if plotparam.save:
        print("Save images under:\n"+plotparam.path+plotparam.name)
        plt.savefig(plotparam.path+plotparam.name,dpi=plotparam.dpi)
    if plotparam.show:
        fig.show()
    else:
        plt.close(fig)








def seal3DScatter(x, y, z, v, cptFile, minVal, maxVal, genericSize=5, redHighlight=None, verbose=True, colormod=''):
    """
    This function is an alternative to matplotlib 3D and allows plot scatter
    data very efficiently using OpenGL objects contains into a pyQt5
    application.
    <i>: x, y and z = list/np.array, correspond to coordinates 
                      matrix for each point of the scatter plot
         genericSize = int/float, defines the size of points [default: 1]
         colorChoice = str, defines the color map used for the scatter plot
         verbose = bool, condition for the verbose output
    """
    
    # Specific import:
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    from tqdm import tqdm
    from .cptReader import colorBar_pyqtgraph


    pName = 'seal3DScatter'
    if verbose:
        print()
    im('Welcome in seal3D',pName,verbose)
    im('Enjoy the experience !',pName,verbose)
    im('',pName,verbose)
    im('Procceding to plot operation with seal3D',pName,verbose)
    im('Erreur sur la carte graphique: ',pName,verbose)
    im("    https://groups.google.com/forum/#!topic/pyqtgraph/2M2hwIW-vgM",pName,verbose)
    # ====================
    # Windows    
    left=250;top=150;width=1300;height=800    
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.setCameraPosition(elevation=10)
    w.pan(0,0,0)
    w.opts['distance'] = 10
    w.show()
    w.setWindowTitle('Seal3D')
    w.setGeometry(left,top,width,height)
    ax = gl.GLAxisItem()
    w.addItem(ax)
    # ====================
    # Plot in OpenGL
    pts = []
    colors = []
    size = []
    im('Computation on all points:',pName,verbose)
    if colormod == 'geom':
        colors = []
        distMax = np.amax(10*np.sqrt(np.array(x)**2+np.array(y)**2))/1.2
        for i in tqdm(range(len(x))):
            pts.append([x[i],y[i],z[i]])    
            size.append(genericSize)
            colors.append(pg.glColor( (10*np.sqrt(x[i]**2+y[i]**2),distMax) ))
    else:
        for i in tqdm(range(len(x))):
            pts.append([x[i],y[i],z[i]])    
            size.append(genericSize)
        colors = colorBar_pyqtgraph(v,minVal,maxVal,cptFile)
    # ====================
    #Highlight selected points with the redHighlight flag:
    if redHighlight != None:
        redHighlight = sorted(redHighlight, reverse=True)
        for hIndex in redHighlight:
            size[hIndex] = genericSize*3
            colors[hIndex] = pg.glColor((255,0,0))
    pts    = np.array(pts)
    size   = np.array(size)
    colors = np.array(colors)
    plt = gl.GLScatterPlotItem(pos=pts, color=colors, size=size)
    plt.setGLOptions('translucent')
    w.addItem(plt)
    # ====================
    # Start Qt event loop unless running in interactive mode.
    if __name__ == 'stagViewer':
        im('Thanks to use seal3D !',pName,verbose)
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()





def seal3DSurface(x, y, z, v, cptFile='Default', minV='Default', maxV='Default', function=None,\
                  drawEdges=False, drawFaces=True, smoothing=False, verbose=True):
    """Advance routine for 3D surface plot. This function will plot the convex
    hull of an input scatter define with 3 matrices of coordinates: x,y and z
    and a matrix (v) for the field on each points.
    <i> : x = list/np.ndarray, x coordinates of the scatter: shape = N
          y = list/np.ndarray, y coordinates of the scatter: shape = N
          z = list/np.ndarray, z coordinates of the scatter: shape = N
          v = list/np.ndarray, field of the scatter: shape = N
          <options>
          cptFile = str, path to the color map under the .cpt format you want
                    for the plot (Default = '.../Binaries/stagCTP_T.cpt')
          minV = int/float, minimum value for the colormap
                 (Default = np.amin(v))
          maxV = int/float, maximum value for the colormap
                 (Default = np.amax(v))
          function = function object, function that will be applied on the
                     field before plotting (e.g. numpy.log10)
          drawEdges = bool, condition to draw Edges
          drawFaces = bool, condition to draw Faces
          smoothing = bool, condition to apply a smoothing
          verbose = bool, condition for the verbose output
    """

    # Specific import:
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    from tqdm import tqdm
    from ..cptReader import colorBar_pyqtgraph


    def _surface_triangulation(x,y,z,verbose=True):
        """ This function computes a convex hull triangulation on an input 
        scatter and will adapt the output for pyqtgraph. 
        <i> : x = list/np.ndarray, x coordinates of the scatter: shape = N
              y = list/np.ndarray, y coordinates of the scatter: shape = N
              z = list/np.ndarray, z coordinates of the scatter: shape = N
              verbose = bool, condition for the verbose output
        """
        pName = 'seal3DSurface'
        if verbose:
            print()
        im('Stag Visualization ToolKit',pName,verbose)
        im('Requested: scatter -> seal3D',pName,verbose)
        # ====================
        # 1) Take the surface of the 2 grids, patch together and triangulate
        im('    - Triangulation on convex hull',pName,verbose)
        # ====================
        # Triangulation of the surface using a convex hull algorithm
        from scipy.spatial import ConvexHull
        points  = [[x[i],y[i],z[i]] for i in range(len(x))]
        triGrid = ConvexHull(points).simplices # simple way to grid it
        return (points,triGrid)
    # ====================
    # Plotting routine
    pName = 'seal3DSurface'
    # ====================
    # windows parameters
    left=250;top=150;width=1300;height=800
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.setCameraPosition(elevation=10)
    w.pan(0,0,0)
    w.opts['distance'] = 10
    w.show()
    w.setWindowTitle('Seal3D')
    w.setGeometry(left,top,width,height)
    ax = gl.GLAxisItem()
    w.addItem(ax)
    # ====================
    # Triangulation
    (points,triGrid) = _surface_triangulation(x,y,z,verbose=verbose)
    # ====================
    # Plot Parameters
    verts  = np.array(points)
    faces  = triGrid
    if function == None:
        v_plot = v
    else:
        v_plot = function(v)
    if minV == 'Default':
        minVal = np.amin(v_plot)
    else:
        minVal = minV
    if maxV == 'Default':
        maxVal = np.amax(v_plot)
    else:
        maxVal = maxV
    if cptFile == 'Default':
        cptFile = 'C:/Alexandre/Bibliotheka Alexandrina/6- Stage Assimilation/Binaries/stagCTP_T.cpt'
    # ====================
    # Construction of colors vector
    co = colorBar_pyqtgraph(v_plot,minVal,maxVal,cptFile,glColor=False)# np.random.random(size=(Points.shape[0], 4))#
    colors = np.array([[co[i][0]/255, co[i][1]/255, co[i][2]/255,1] for i in range(len(co))])
    # ====================
    # Surface plot
    ## Mesh item will automatically compute face normals.
    im('    - Preparation of the interface',pName,verbose)
    m1 = gl.GLMeshItem(vertexes=verts, faces=faces, vertexColors=colors, drawEdges=False, drawFaces=True, smooth=False)
    m1.setGLOptions('translucent')
    w.addItem(m1)
    #====================
    # Start Qt event loop unless running in interactive mode.
    if __name__ == 'stagViewer' or __name__=='__main__':
        im('Thanks to use seal3D !',pName,verbose)
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()





def seal3D_YYSurface(stagData,selectedLayer='surface',log10=False,drawEdges=False, drawFaces=True, smoothing=False, verbose=True):
    """Advance routine for 3D plot. This script is based on stagVTU module 
    of the stag visialuzation toolkit (stagVTK)
    <i> : stagData = StagData object, StagData to display
          selectedLayer = 'surface' or int, if int corresponds to the index of 
                          the stagData that you want to display
          log10 = bool, if True, then diplay the np.log10 of the current field
                  stored in the StagData
          drawEdges = bool, condition to draw Edges
          drawFaces = bool, condition to draw Faces
          smoothing = bool, condition to smooth the resulting display and avoid
                      as possible the aliasing
          verbose = bool, condition for the verbose output
    """

    # Specific import:
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    from tqdm import tqdm
    from .cptReader import colorBar_pyqtgraph


    def _surface_triangulation(stagData,selectedLayer='surface',verbose=True):
        """ This function creat '.vtu' file readable with Paraview to efficiently 
        visualize 3D data contain in a stagData object. This function works directly
        on non overlapping stagData object.
        Note also that the internal field stagData.slayers of the stagData object
        must be filled.
        <i> : stagData = stagData object, stagData object that will be transform
                         into .vtu file
              selectedLayer = 'surface' or int, if int corresponds to the index
                              of the stagData that you want to display
              verbose = bool, condition for the verbose output
        """
        pName = 'seal3D_YYSurface'
        if verbose:
            print()
        im('Stag Visualization ToolKit',pName,verbose)
        im('Requested: stagData -> seal3D',pName,verbose)
    
        im('    - Grid preparation',pName,verbose)
        # ====================
        #Re-formating initial grids data
        X_yin  = stagData.x1
        Y_yin  = stagData.y1
        Z_yin  = stagData.z1
        X_yang = stagData.x2
        Y_yang = stagData.y2
        Z_yang = stagData.z2
        # ====================
        Nz   = len(stagData.slayers)      #Number of depth layers
        NxNy = int(len(stagData.x1)/Nz)   #Number of points for each layers
        # ====================
        x1     = X_yin.reshape(NxNy,Nz)
        x2     = X_yang.reshape(NxNy,Nz)
        X_yin  = x1.tolist()
        X_yang = x2.tolist()
        y1     = Y_yin.reshape(NxNy,Nz)
        y2     = Y_yang.reshape(NxNy,Nz)
        Y_yin  = y1.tolist()
        Y_yang = y2.tolist()
        z1     = Z_yin.reshape(NxNy,Nz)
        z2     = Z_yang.reshape(NxNy,Nz)
        Z_yin  = z1.tolist()
        Z_yang = z2.tolist()
        # ====================
        #Re-organisation of data to have X,Y and Z grid matrices organized by depths:
        X = []
        Y = []
        Z = []
        for j in range(Nz):
            x1t = [x1[i][j] for i in range(NxNy)]
            x2t = [x2[i][j] for i in range(NxNy)]
            X.append(x1t+x2t)
            y1t = [y1[i][j] for i in range(NxNy)]
            y2t = [y2[i][j] for i in range(NxNy)]
            Y.append(y1t+y2t)
            z1t = [z1[i][j] for i in range(NxNy)]
            z2t = [z2[i][j] for i in range(NxNy)]
            Z.append(z1t+z2t)    
        # ====================
        # 1) Take the surface of the 2 grids, patch together and triangulate
        im('    - Triangulation on convex hull',pName,verbose)
        # NotaBene: _s for the surface layer
        X_s    = X[Nz-1]
        Y_s    = Y[Nz-1]
        Z_s    = Z[Nz-1]
        # ====================
        # Triangulation of the surface using a convex hull algorithm
        from scipy.spatial import ConvexHull
        points      = [[X_s[i],Y_s[i],Z_s[i]] for i in range(len(X_s))]
        triYingYang = ConvexHull(points).simplices # simple way to grid it
        # ====================
        # 2) Create a 3D grid with tetrahedron elements
        # Number all gridpoints we have
        NUM_1       = np.array(range(0,NxNy*Nz))
        NUMBER_1    = NUM_1.reshape((NxNy,Nz), order='F')
        NUMBER_2    = NUMBER_1 + NxNy*Nz
        #NUM_2       = np.array(range(0,NxNy*Nz))+NxNy*Nz
        # ====================
        # Make a loop over all levels
        ElementNumbers      = []
        if selectedLayer == 'surface':
            for iz in range(Nz-1):
                if iz == Nz-2:
                    num_upper1      = NUMBER_1[:,iz+1]
                    num_upper2      = NUMBER_2[:,iz+1]
                    num_upper       = list(num_upper1) + list(num_upper2)
                    num_lower1      = NUMBER_1[:,iz]
                    num_lower2      = NUMBER_2[:,iz]
                    num_lower       = list(num_lower1) + list(num_lower2)
                    # ====================
                    num_tri = [[num_upper[t[0]], \
                                num_upper[t[1]], \
                                num_upper[t[2]], \
                                num_lower[t[0]], \
                                num_lower[t[1]], \
                                num_lower[t[2]]] for t in triYingYang]
                    ElementNumbers.extend(num_tri)
        else:
            for iz in range(Nz-1):
                num_upper1      = NUMBER_1[:,iz+1]
                num_upper2      = NUMBER_2[:,iz+1]
                num_upper       = list(num_upper1) + list(num_upper2)
                num_lower1      = NUMBER_1[:,iz]
                num_lower2      = NUMBER_2[:,iz]
                num_lower       = list(num_lower1) + list(num_lower2)
                # ====================
                num_tri = [[num_upper[t[0]], \
                            num_upper[t[1]], \
                            num_upper[t[2]], \
                            num_lower[t[0]], \
                            num_lower[t[1]], \
                            num_lower[t[2]]] for t in triYingYang]
                ElementNumbers.extend(num_tri)
        # ====================
        # Convert data into correct vector format
        im('    - Convert data into correct vector format',pName,verbose)
        Points = [list(np.array(x1).reshape((NxNy*Nz), order='F')) + list(np.array(x2).reshape((NxNy*Nz), order='F')), \
                  list(np.array(y1).reshape((NxNy*Nz), order='F')) + list(np.array(y2).reshape((NxNy*Nz), order='F')), \
                  list(np.array(z1).reshape((NxNy*Nz), order='F')) + list(np.array(z2).reshape((NxNy*Nz), order='F'))]
        Points = np.array(Points).transpose()
        # ====================
        if stagData.fieldNature == 'Scalar' or stagData.fieldNature == '':
            V_yin  = np.array(stagData.v1).reshape(NxNy,Nz)
            V_yang = np.array(stagData.v2).reshape(NxNy,Nz)
            # ====================
            vstack = list(V_yin.reshape((NxNy*Nz), order='F')) + \
                     list(V_yang.reshape((NxNy*Nz),order='F'))
        # ====================
        if stagData.fieldNature == 'Vectorial':
            # ------ Vx ------
            V_yinx  = np.array(stagData.vx1).reshape(NxNy,Nz)
            V_yangx = np.array(stagData.vx2).reshape(NxNy,Nz)
            # ====================
            vstackx = list(V_yinx.reshape((NxNy*Nz), order='F')) + \
                      list(V_yangx.reshape((NxNy*Nz),order='F'))
            # ------ Vy ------
            V_yiny  = np.array(stagData.vy1).reshape(NxNy,Nz)
            V_yangy = np.array(stagData.vy2).reshape(NxNy,Nz)
            # ====================
            vstacky = list(V_yiny.reshape((NxNy*Nz), order='F')) + \
                      list(V_yangy.reshape((NxNy*Nz),order='F'))
            # ------ Vz ------
            V_yinz  = np.array(stagData.vz1).reshape(NxNy,Nz)
            V_yangz = np.array(stagData.vz2).reshape(NxNy,Nz)
            # ====================
            vstackz = list(V_yinz.reshape((NxNy*Nz), order='F')) + \
                      list(V_yangz.reshape((NxNy*Nz),order='F'))
            # ------ Vr ------
            V_yinr  = np.array(stagData.vr1).reshape(NxNy,Nz)
            V_yangr = np.array(stagData.vr2).reshape(NxNy,Nz)
            # ====================
            vstackr = list(V_yinr.reshape((NxNy*Nz), order='F')) + \
                      list(V_yangr.reshape((NxNy*Nz),order='F'))
            # ------ Vtheta ------
            V_yintheta  = np.array(stagData.vtheta1).reshape(NxNy,Nz)
            V_yangtheta = np.array(stagData.vtheta2).reshape(NxNy,Nz)
            # ====================
            vstacktheta = list(V_yintheta.reshape((NxNy*Nz), order='F')) + \
                          list(V_yangtheta.reshape((NxNy*Nz),order='F'))
            # ------ Vphi ------
            V_yinphi  = np.array(stagData.vphi1).reshape(NxNy,Nz)
            V_yangphi = np.array(stagData.vphi2).reshape(NxNy,Nz)
            # ====================
            vstackphi = list(V_yinphi.reshape((NxNy*Nz), order='F')) + \
                        list(V_yangphi.reshape((NxNy*Nz),order='F'))
            # ------ P ------
            V_yinp  = np.array(stagData.P1).reshape(NxNy,Nz)
            V_yangp = np.array(stagData.P2).reshape(NxNy,Nz)
            # ====================
            vstackp = list(V_yinp.reshape((NxNy*Nz), order='F')) + \
                      list(V_yangp.reshape((NxNy*Nz),order='F'))
            # ====================
            vstack = (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)
        return (NxNy,Nz,Points,ElementNumbers,vstack)
    # ====================
    # Plotting routine
    pName = 'seal3DSurface'
    # ====================
    left=250;top=150;width=1300;height=800
    # ====================
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    #w.setBackgroundColor(pg.glColor((10,10)))
    w.setCameraPosition(elevation=10)
    w.pan(0,0,0)
    w.opts['distance'] = 10
    w.show()
    w.setWindowTitle('Seal3D')
    w.setGeometry(left,top,width,height)
    ax = gl.GLAxisItem()
    w.addItem(ax)
    # ====================
    (NxNy,Nz,Points,ElementNumbers,vstack) = _surface_triangulation(stagData,\
                                                                    selectedLayer=selectedLayer,\
                                                                    verbose=verbose)
    # ====================
    verts = Points
    faces = np.array(ElementNumbers)
    if log10:
        v_plot = np.log10(vstack)
    else:
        v_plot = vstack
    # ====================
    minVal = np.amin(v_plot)
    maxVal = np.amax(v_plot)
    cptFile = 'C:/Alexandre/Bibliotheka Alexandrina/6- Stage Assimilation/Binaries/stagCTP_T.cpt'
    # ====================
    co = colorBar_pyqtgraph(v_plot,minVal,maxVal,cptFile,glColor=False)# np.random.random(size=(Points.shape[0], 4))#
    colors = np.array([[co[i][0]/255, co[i][1]/255, co[i][2]/255,1] for i in range(len(co))])
    # ====================
    # Mesh item will automatically compute face normals.
    m1 = gl.GLMeshItem(vertexes=verts, faces=faces, vertexColors=colors, drawEdges=drawEdges, drawFaces=drawFaces, smooth=smoothing)
    m1.setGLOptions('translucent')
    w.addItem(m1)
    # ====================
    # Start Qt event loop unless running in interactive mode.
    if __name__ == 'stagViewer' or __name__=='__main__':
        im('Thanks to use seal3D !',pName,verbose)
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()








def _surface_triangulation(stagData,selectedLayer='surface',verbose=True):
    """ This function creat '.vtu' file readable with Paraview to efficiently 
    visualize 3D data contain in a stagData object. This function works directly
    on non overlapping stagData object.
    Note also that the internal field stagData.slayers of the stagData object
    must be filled.
    <i> : stagData = stagData object, stagData object that will be transform
                     into .vtu file
          path = str, path where you want to export your new .vtu file.
                 [Default: path='./']
          ASCII = bool, if True, the .vtu file will be write in ASCII mode
                  if not, in binary mode. [Default, ASCII=True]
    """
    pName = 'sealTri'
    if verbose:
        print()
    im('Stag Visualization ToolKit',pName,verbose)
    im('Requested: stagData -> .vtu',pName,verbose)

    im('    - Grid preparation',pName,verbose)
    # =======================================
    #Re-formating initial grids data
    X_yin  = stagData.x1
    Y_yin  = stagData.y1
    Z_yin  = stagData.z1
    X_yang = stagData.x2
    Y_yang = stagData.y2
    Z_yang = stagData.z2

    Nz   = len(stagData.slayers)      #Number of depth layers
    NxNy = int(len(stagData.x1)/Nz)   #Number of points for each layers
    
    x1     = X_yin.reshape(NxNy,Nz)
    x2     = X_yang.reshape(NxNy,Nz)
    X_yin  = x1.tolist()
    X_yang = x2.tolist()
    y1     = Y_yin.reshape(NxNy,Nz)
    y2     = Y_yang.reshape(NxNy,Nz)
    Y_yin  = y1.tolist()
    Y_yang = y2.tolist()
    z1     = Z_yin.reshape(NxNy,Nz)
    z2     = Z_yang.reshape(NxNy,Nz)
    Z_yin  = z1.tolist()
    Z_yang = z2.tolist()
    
    #Re-organisation of data to have X,Y and Z grid matrices organized by depths:
    X = []
    Y = []
    Z = []
    for j in range(Nz):
        x1t = [x1[i][j] for i in range(NxNy)]
        x2t = [x2[i][j] for i in range(NxNy)]
        X.append(x1t+x2t)
        y1t = [y1[i][j] for i in range(NxNy)]
        y2t = [y2[i][j] for i in range(NxNy)]
        Y.append(y1t+y2t)
        z1t = [z1[i][j] for i in range(NxNy)]
        z2t = [z2[i][j] for i in range(NxNy)]
        Z.append(z1t+z2t)    
    
    # =========================================================================
    # 1) Take the surface of the 2 grids, patch together and triangulate
    # =========================================================================
    im('    - Triangulation on convex hull',pName,verbose)
    
    # NotaBene: _s for the surface layer
    X_s    = X[Nz-1]
    Y_s    = Y[Nz-1]
    Z_s    = Z[Nz-1]
    
    # =======================================
    # Triangulation of the surface using a convex hull algorithm
    from scipy.spatial import ConvexHull
    
    points      = [[X_s[i],Y_s[i],Z_s[i]] for i in range(len(X_s))]
    triYingYang = ConvexHull(points).simplices # simple way to grid it
    
    # =========================================================================
    # 2) Create a 3D grid with tetrahedron elements
    # =========================================================================
    
    # Number all gridpoints we have
    NUM_1       = np.array(range(0,NxNy*Nz))
    NUMBER_1    = NUM_1.reshape((NxNy,Nz), order='F')
    NUMBER_2    = NUMBER_1 + NxNy*Nz
    #NUM_2       = np.array(range(0,NxNy*Nz))+NxNy*Nz
    
    # Make a loop over all levels
    ElementNumbers      = []
    if selectedLayer == 'surface':
        for iz in range(Nz-1):
            if iz == Nz-2:
                num_upper1      = NUMBER_1[:,iz+1]
                num_upper2      = NUMBER_2[:,iz+1]
                num_upper       = list(num_upper1) + list(num_upper2)
                num_lower1      = NUMBER_1[:,iz]
                num_lower2      = NUMBER_2[:,iz]
                num_lower       = list(num_lower1) + list(num_lower2)
                
                num_tri = [[num_upper[t[0]], \
                            num_upper[t[1]], \
                            num_upper[t[2]], \
                            num_lower[t[0]], \
                            num_lower[t[1]], \
                            num_lower[t[2]]] for t in triYingYang]
                ElementNumbers.extend(num_tri)
    else:
        for iz in range(Nz-1):
            num_upper1      = NUMBER_1[:,iz+1]
            num_upper2      = NUMBER_2[:,iz+1]
            num_upper       = list(num_upper1) + list(num_upper2)
            num_lower1      = NUMBER_1[:,iz]
            num_lower2      = NUMBER_2[:,iz]
            num_lower       = list(num_lower1) + list(num_lower2)
            
            num_tri = [[num_upper[t[0]], \
                        num_upper[t[1]], \
                        num_upper[t[2]], \
                        num_lower[t[0]], \
                        num_lower[t[1]], \
                        num_lower[t[2]]] for t in triYingYang]
            ElementNumbers.extend(num_tri)
            
    
    # =======================================
    # Convert data into correct vector format
    
    im('    - Convert data into correct vector format',pName,verbose)

    Points = [list(np.array(x1).reshape((NxNy*Nz), order='F')) + list(np.array(x2).reshape((NxNy*Nz), order='F')), \
              list(np.array(y1).reshape((NxNy*Nz), order='F')) + list(np.array(y2).reshape((NxNy*Nz), order='F')), \
              list(np.array(z1).reshape((NxNy*Nz), order='F')) + list(np.array(z2).reshape((NxNy*Nz), order='F'))]
    Points = np.array(Points).transpose()
    
    # ===================
    if stagData.fieldNature == 'Scalar' or stagData.fieldNature == '':
        
        V_yin  = np.array(stagData.v1).reshape(NxNy,Nz)
        V_yang = np.array(stagData.v2).reshape(NxNy,Nz)
    
        vstack = list(V_yin.reshape((NxNy*Nz), order='F')) + \
                 list(V_yang.reshape((NxNy*Nz),order='F'))
    
    # ===================
    if stagData.fieldNature == 'Vectorial':
        # ------ Vx ------
        V_yinx  = np.array(stagData.vx1).reshape(NxNy,Nz)
        V_yangx = np.array(stagData.vx2).reshape(NxNy,Nz)
        
        vstackx = list(V_yinx.reshape((NxNy*Nz), order='F')) + \
                  list(V_yangx.reshape((NxNy*Nz),order='F'))
        # ------ Vy ------
        V_yiny  = np.array(stagData.vy1).reshape(NxNy,Nz)
        V_yangy = np.array(stagData.vy2).reshape(NxNy,Nz)
        
        vstacky = list(V_yiny.reshape((NxNy*Nz), order='F')) + \
                  list(V_yangy.reshape((NxNy*Nz),order='F'))
        # ------ Vz ------
        V_yinz  = np.array(stagData.vz1).reshape(NxNy,Nz)
        V_yangz = np.array(stagData.vz2).reshape(NxNy,Nz)
        
        vstackz = list(V_yinz.reshape((NxNy*Nz), order='F')) + \
                  list(V_yangz.reshape((NxNy*Nz),order='F'))
        # ------ Vr ------
        V_yinr  = np.array(stagData.vr1).reshape(NxNy,Nz)
        V_yangr = np.array(stagData.vr2).reshape(NxNy,Nz)
        
        vstackr = list(V_yinr.reshape((NxNy*Nz), order='F')) + \
                  list(V_yangr.reshape((NxNy*Nz),order='F'))
        # ------ Vtheta ------
        V_yintheta  = np.array(stagData.vtheta1).reshape(NxNy,Nz)
        V_yangtheta = np.array(stagData.vtheta2).reshape(NxNy,Nz)
        
        vstacktheta = list(V_yintheta.reshape((NxNy*Nz), order='F')) + \
                      list(V_yangtheta.reshape((NxNy*Nz),order='F'))
        # ------ Vphi ------
        V_yinphi  = np.array(stagData.vphi1).reshape(NxNy,Nz)
        V_yangphi = np.array(stagData.vphi2).reshape(NxNy,Nz)
        
        vstackphi = list(V_yinphi.reshape((NxNy*Nz), order='F')) + \
                    list(V_yangphi.reshape((NxNy*Nz),order='F'))
        
        # ------ P ------
        V_yinp  = np.array(stagData.P1).reshape(NxNy,Nz)
        V_yangp = np.array(stagData.P2).reshape(NxNy,Nz)
        
        vstackp = list(V_yinp.reshape((NxNy*Nz), order='F')) + \
                  list(V_yangp.reshape((NxNy*Nz),order='F'))
        
        vstack = (vstackx,vstacky,vstackz,vstackr,vstacktheta,vstackphi,vstackp)
    
    return (NxNy,Nz,Points,ElementNumbers,vstack)

























