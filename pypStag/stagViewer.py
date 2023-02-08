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
import random
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from .stagError import VisuGridGeometryError, StagTypeError, StagMapFieldError, StagMapUnknownFieldError, InputGridGeometryError_Viewer, fieldTypeError,StagOptionInDevError
from .stagData import StagCartesianGeometry, StagYinYangGeometry, StagSphericalGeometry
from .stagData import SliceData, CartesianSliceData
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



def Rgt(lat,lon):
    """
    Rgt =  geocentric to topocentric rotation matrix

    For Venu = (Ve,Vn,Vz) and Vxyz = (Vx,Vy,Vz)

    Venu = np.dot(Rgt,Vxyz)
    Vxyz = np.dot(Rgt_inv,Venu)

    ** Lat, Lon coordinates in RADIANS **
    """
    return np.array([[-np.sin(lon),np.cos(lon),0],\
                     [-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],\
                     [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
    


def align_axis_x_subplots(ax, ax_target):
        """Make x-axis of `ax` aligned with `ax_target` in figure"""
        posn_old, posn_target = ax.get_position(), ax_target.get_position()
        ax.set_position([posn_target.x0, posn_old.y0, posn_target.width, posn_old.height])



def refresh_figure(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()






def __builder(stagData,field='v',axis=0,normal=[1,0,0],layer=-1,title=None,plotparam=None,verbose=True):
    """
    """
    pName = 'builder'
    # --- Test geometry:
    if stagData.geometry != 'cart2D' and stagData.geometry != 'cart3D' and \
       stagData.geometry != 'annulus' and stagData.geometry != 'yy' and\
       stagData.geometry != 'spherical':
        raise VisuGridGeometryError(stagData.geometry,'cart2D or cart3D or annulus')
    # --- Test PlotParam
    if plotparam is None:
        plotparam = PlotParam()
    plotparam.update()
    # --- Test colormap:
    if plotparam.cmaptype == 'matplotlib':
        cmap = plotparam.cmap
        if plotparam.reverseCMAP:
            cmap = cmap.reversed()
    else:
        try:
            from .cptReader import GCMTcolormap
            cmap = GCMTcolormap(plotparam.cmap,reverse=plotparam.reverseCMAP)
        except:
            im("WARNING: Unknown colormap file",pName,verbose)
            cmap = plt.cm.seismic
    # --- Test the field
    if field == 'scalar' or field == 'v':
        stagfield = stagData.v
    elif field == 'vx':
        if stagData.fieldNature == 'Vectorial':
            stagfield = stagData.vx
        else:
            raise StagMapFieldError(field,stagData.geometry,stagData.fieldNature)
    elif field == 'vy':
        if stagData.fieldNature == 'Vectorial':
            stagfield = stagData.vy
        else:
            raise StagMapFieldError(field,stagData.geometry,stagData.fieldNature)
    elif field == 'vz':
        if stagData.fieldNature == 'Vectorial':
            stagfield = stagData.vz
        else:
            raise StagMapFieldError(field,stagData.geometry,stagData.fieldNature)
    elif field == 'vphi':
        if stagData.fieldNature == 'Vectorial':
            stagfield = stagData.vphi
        else:
            raise StagMapFieldError(field,stagData.geometry,stagData.fieldNature)
    elif field == 'vtheta':
        if stagData.fieldNature == 'Vectorial':
            stagfield = stagData.vtheta
        else:
            raise StagMapFieldError(field,stagData.geometry,stagData.fieldNature)
    elif field == 'vr':
        if stagData.fieldNature == 'Vectorial':
            stagfield = stagData.vr
        else:
            raise StagMapFieldError(field,stagData.geometry,stagData.fieldNature)
    elif field == 'p' or field == 'P':
        if stagData.fieldNature == 'Vectorial':
            stagfield = stagData.P
        else:
            raise StagMapFieldError(field,stagData.geometry,stagData.fieldNature)
    else:
        raise StagMapUnknownFieldError(field)
    # --- Log10:
    if plotparam.log10:
        stagfield = np.log10(stagfield)
    # --- Title
    if title is None: #no title set
        if plotparam.title == '':  # no generic title -> build auto title
            field = stagData.fieldType
            if plotparam.log10:
                field = 'log10('+stagData.fieldType+')'
            if axis == 0 or axis == 'annulus':
                title = 'stagMap: axis='+str(axis)+' normal='+str(normal)+' field='+field
            else:
                title = 'stagMap: axis='+str(axis)+' layer='+str(layer)+' field='+field
        else: # generic title
            title = plotparam.title
    # --- vmin and vmax
    if plotparam.vmin is None:
        vmin = np.amin(stagfield)
    else:
        vmin = plotparam.vmin
    if plotparam.vmax is None:
        vmax = np.amax(stagfield)
    else:
        vmax = plotparam.vmax
    # --- leveling
    levels = np.linspace(vmin,vmax,plotparam.nol)
    # --- outputs
    return plotparam,title,cmap,stagfield,vmin,vmax,levels
    
    



def contourfMap(stagData,field='v',stagVelocity=None,\
                axis=1,normal=None,layer=-1,method='tri',interp_method='nearest',locate=None,\
                pollenPlot=False,pollenData=None,pollenField='v',plotparam_pollen=None,\
                title=None,plotparam=None,plotparam_velo=None,verbose=True):
    """
    """
    if isinstance(stagData,StagCartesianGeometry):
        if stagData.geometry == 'cart2D':
            output = contourfMap_cart2D(stagData,field=field,stagVelocity=stagVelocity,title=title,plotparam=plotparam,plotparam_velo=plotparam_velo,verbose=verbose)
        elif stagData.geometry == 'cart3D':
            output = contourfMap_cart3D(stagData,field=field,stagVelocity=stagVelocity,axis=axis,normal=normal,layer=layer,method=method,interp_method=interp_method,title=title,plotparam=plotparam,plotparam_velo=plotparam_velo,verbose=verbose)
        else:
            raise InputGridGeometryError_Viewer(stagData.geometry)
    elif isinstance(stagData,StagSphericalGeometry):
        if stagData.geometry == 'spherical':
            raise InputGridGeometryError_Viewer(stagData.geometry)
        elif stagData.geometry == 'annulus':
            output = contourfMap_annulus(stagData,field=field,stagVelocity=stagVelocity,pollenPlot=pollenPlot,pollenData=pollenData,layer=layer,pollenField=pollenField,title=title,plotparam=plotparam,plotparam_pollen=plotparam_pollen,plotparam_velo=plotparam_velo,verbose=verbose)
        else:
            raise InputGridGeometryError_Viewer(stagData.geometry)
    elif isinstance(stagData,StagYinYangGeometry):
        output = contourfMap_yy(stagData,field=field,stagVelocity=stagVelocity,axis=axis,normal=normal,layer=layer,method=method,interp_method=interp_method,locate=locate,title=title,pollenPlot=pollenPlot,pollenData=pollenData,pollenField=pollenField,plotparam_pollen=plotparam_pollen,plotparam=plotparam,plotparam_velo=plotparam_velo,verbose=verbose)
    return output



def contourfMap_cart2D(stagData,field='v',stagVelocity=None,title=None,plotparam=None,plotparam_velo=None,verbose=True):
    """
    """
    pName = 'contourfMap'
    plotparam,title,cmap,stagfield,vmin,vmax,levels = __builder(stagData,field=field,axis=None,normal=None,layer=None,title=title,plotparam=plotparam,verbose=verbose)

    fig, ax = plt.subplots(1,1,figsize=(plotparam.figsize[0]*plotparam.aspect_ratio,plotparam.figsize[1]))
    if plotparam.aspect == 'equal':
        ax.set_aspect('equal', 'box')
    ax.set_title(title)

    loc = 0
    if stagData.plan == 'xz' or stagData.plan == 'zx':
        cmap4bar = ax.contourf(stagData.x[:,loc,:],stagData.z[:,loc,:],stagfield[:,loc,:],levels=levels,cmap=cmap,extend='both')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('z-axis')

    elif stagData.plan == 'yz' or stagData.plan == 'zy':
        cmap4bar = ax.contourf(stagData.y[loc,:,:],stagData.z[loc,:,:],stagfield[loc,:,:],levels=levels,cmap=cmap,extend='both')
        ax.set_xlabel('y-axis')
        ax.set_ylabel('z-axis')
    
    elif stagData.plan == 'xy' or stagData.plan == 'yx':
        cmap4bar = ax.contourf(stagData.x[:,:,loc],stagData.y[:,:,loc],stagfield[:,:,loc],levels=levels,cmap=cmap,extend='both')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        
    # cbar
    if plotparam.cbar == True:
        if plotparam.cbar_location is not None:
            #cbaxes = fig.add_axes([0.3, 0.1, 0.4, 0.01])
            cbar   = fig.colorbar(cmap4bar, ax=[ax], location=plotparam.cbar_location, shrink=plotparam.cbar_shrink, pad=plotparam.cbar_pad,aspect=plotparam.cbar_aspect)
        elif cbaxes is not None:
            cbaxes = fig.add_axes(plotparam.cbar_axis) # [left, bottom, width, height]
            cbar   = fig.colorbar(cmap4bar,cax=cbaxes,orientation=plotparam.cbar_orientation)
        
        if plotparam.cbar_label is not None:
            clabel = plotparam.cbar_label
        else:
            clabel = stagData.fieldType
        cbar.set_label(clabel,size=plotparam.cbar_labelsize)
    
    if isinstance(stagVelocity,StagCartesianGeometry):
        if stagVelocity.fieldType is not 'Velocity':
            raise fieldTypeError('Velocity')
        else:
            # Quiver
            if stagData.plan == 'xz' or stagData.plan == 'zx':
                xa = stagVelocity.x[:,loc,:].flatten()
                ya = stagVelocity.z[:,loc,:].flatten()
                vx = stagVelocity.vx[:,loc,:].flatten()
                vy = stagVelocity.vz[:,loc,:].flatten()
            elif stagData.plan == 'yz' or stagData.plan == 'zy':
                xa = stagVelocity.y[loc,:,:].flatten()
                ya = stagVelocity.z[loc,:,:].flatten()
                vx = stagVelocity.vy[loc,:,:].flatten() - np.mean(stagVelocity.vy[loc,:,:].flatten())
                vy = stagVelocity.vz[loc,:,:].flatten()
            elif stagData.plan == 'xy' or stagData.plan == 'yx':
                xa = stagVelocity.x[:,:,loc].flatten()
                ya = stagVelocity.y[:,:,loc].flatten()
                vx = stagVelocity.vx[:,:,loc].flatten()
                vy = stagVelocity.vy[:,:,loc].flatten()
            if plotparam_velo is not None:
                veloScale = plotparam_velo.vscale
                veloWidth = plotparam_velo.arrow_width
                Qscale = plotparam_velo.qscale
                noa = plotparam_velo.noa
            else:
                Qscale = int(np.mean(np.sqrt(vx**2+vy**2)))
                noa = 1000
                if noa > len(xa):
                    while noa > len(xa):
                        noa = int(noa/10)
                veloScale = None
                veloWidth = None
            # noa = number of Arrows
            id = list(range(len(xa)))
            mask = random.sample(id,noa)
            xa = xa[mask]
            ya = ya[mask]
            vx = vx[mask]
            vy = vy[mask]
            Q  = ax.quiver(xa,ya,vx,vy,scale=veloScale,width=veloWidth)
            qq = ax.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
    elif stagVelocity is not None:
        raise StagTypeError(type(stagVelocity),str(StagCartesianGeometry))
    # Plot
    plt.show()
    return 1


def contourfMap_cart3D(stagData,field='v',stagVelocity=None,axis=1,normal=None,layer=-1,method='tri',interp_method='nearest',title=None,plotparam=None,plotparam_velo=None,verbose=True):
    """
    axis must be in:
        axis = 0                                 # for a slice computed with the normal to the slicing plan (as for YY annulus)
        axis = 'xy' or axis = 'yx' or axis = 1   # [Default]
        axis = 'xz' or axis = 'zx' or axis = 2
        axis = 'yz' or axis = 'zy' or axis = 3
    """
    pName = 'contourfMap'
    if axis == 0:
        if normal is None:
            normal = [1,0,0]
    plotparam,title,cmap,stagfield,vmin,vmax,levels = __builder(stagData,field=field,axis=None,normal=None,layer=None,title=title,plotparam=plotparam,verbose=verbose)

    fig, ax = plt.subplots(1,1,figsize=(plotparam.figsize[0]*plotparam.aspect_ratio,plotparam.figsize[1]))
    if plotparam.aspect == 'equal':
        ax.set_aspect('equal', 'box')
    ax.set_title(title)

    loc = layer
    if axis == 'xz' or axis == 'zx' or axis == 2:
        cmap4bar = ax.contourf(stagData.x[:,loc,:],stagData.z[:,loc,:],stagfield[:,loc,:],levels=levels,cmap=cmap,extend='both')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('z-axis')

    elif axis == 'yz' or axis == 'zy' or axis == 3:
        cmap4bar = ax.contourf(stagData.y[loc,:,:],stagData.z[loc,:,:],stagfield[loc,:,:],levels=levels,cmap=cmap,extend='both')
        ax.set_xlabel('y-axis')
        ax.set_ylabel('z-axis')
    
    elif axis == 'xy' or axis == 'yx' or axis == 1:
        cmap4bar = ax.contourf(stagData.x[:,:,loc],stagData.y[:,:,loc],stagfield[:,:,loc],levels=levels,cmap=cmap,extend='both')
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
    
    elif axis == 0:
        # --- Slicing
        sld = SliceData(geometry=stagData.geometry)
        sld.slicing(stagData,axis=axis,normal=normal,layer=layer,interp_method=interp_method)  # give an axis and here a normal vector
        plotparam,title,cmap,stagfield,vmin,vmax,levels = __builder(sld,field=field,axis=axis,normal=normal,layer=layer,title=title,plotparam=plotparam,verbose=verbose)
        # --- Plot
        if method == 'interp':
            # --- Direct mapping: no need to interpolate, it's already done for the yy->annulus slice
            if plotparam.aspect == 'equal':
                ax.set_aspect('equal', 'box')
            cmap4bar = ax.pcolormesh(sld.x,sld.y,stagfield, cmap=cmap, vmin=plotparam.vmin, vmax=plotparam.vmax,antialiased=plotparam.antialiased)
        if method == 'tri':
            if plotparam.aspect == 'equal':
                ax.set_aspect('equal', 'box')
            ax.set_title(title)
            cmap4bar = ax.contourf(sld.x,sld.y,stagfield,levels=levels,cmap=cmap,extend='both',antialiased=plotparam.antialiased)
        #
        ax.invert_yaxis()
        #
    else:
        raise StagOptionInDevError()
        
    # cbar
    if plotparam.cbar == True:
        if plotparam.cbar_location is not None:
            #cbaxes = fig.add_axes([0.3, 0.1, 0.4, 0.01])
            cbar   = fig.colorbar(cmap4bar, ax=[ax], location=plotparam.cbar_location, shrink=plotparam.cbar_shrink, pad=plotparam.cbar_pad,aspect=plotparam.cbar_aspect)
        elif cbaxes is not None:
            cbaxes = fig.add_axes(plotparam.cbar_axis) # [left, bottom, width, height]
            cbar   = fig.colorbar(cmap4bar,cax=cbaxes,orientation=plotparam.cbar_orientation)
        
        if plotparam.cbar_label is not None:
            clabel = plotparam.cbar_label
        else:
            clabel = stagData.fieldType
        cbar.set_label(clabel,size=plotparam.cbar_labelsize)
    
    if isinstance(stagVelocity,StagCartesianGeometry):
        if stagVelocity.fieldType is not 'Velocity':
            raise fieldTypeError('Velocity')
        else:
            # Quiver
            if axis == 'xz' or axis == 'zx' or axis == 2:
                xa = stagVelocity.x[:,loc,:].flatten()
                ya = stagVelocity.z[:,loc,:].flatten()
                vx = stagVelocity.vx[:,loc,:].flatten()
                vy = stagVelocity.vz[:,loc,:].flatten()
            elif axis == 'yz' or axis == 'zy' or axis == 3:
                xa = stagVelocity.y[loc,:,:].flatten()
                ya = stagVelocity.z[loc,:,:].flatten()
                vx = stagVelocity.vy[loc,:,:].flatten()
                vy = stagVelocity.vz[loc,:,:].flatten()
            elif axis == 'xy' or axis == 'yx' or axis == 1:
                xa = stagVelocity.x[:,:,loc].flatten()
                ya = stagVelocity.y[:,:,loc].flatten()
                vx = stagVelocity.vx[:,:,loc].flatten()
                vy = stagVelocity.vy[:,:,loc].flatten()
            elif axis == 0:
                sldv = SliceData(geometry=stagData.geometry)
                sldv.slicing(stagVelocity,axis=axis,normal=normal,layer=layer,interp_method=interp_method)  # give an axis and here a normal vector
                xa   = sldv.x.flatten()
                ya   = sldv.y.flatten()
                vx   = sldv.vx.flatten()
                vy   = sldv.vy.flatten()
            if plotparam_velo is not None:
                veloScale = plotparam_velo.vscale
                veloWidth = plotparam_velo.arrow_width
                Qscale = plotparam_velo.qscale
                noa = plotparam_velo.noa
            else:
                Qscale = int(np.mean(np.sqrt(vx**2+vy**2)))
                noa = 1000
                if noa > len(xa):
                    while noa > len(xa):
                        noa = int(noa/10)
                veloScale = None
                veloWidth = None
            # noa = number of Arrows
            id = list(range(len(xa)))
            mask = random.sample(id,noa)
            xa = xa[mask]
            ya = ya[mask]
            vx = vx[mask]
            vy = vy[mask]
            Q  = ax.quiver(xa,ya,vx,vy,scale=veloScale,width=veloWidth)
            qq = ax.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
    elif stagVelocity is not None:
        raise StagTypeError(type(stagVelocity),str(StagCartesianGeometry))
    # Plot
    plt.show()
    return 1


def contourfMap_annulus(stagData,field='v',stagVelocity=None,pollenPlot=False,pollenData=None,layer=-1,pollenField='v',title=None,plotparam=None,plotparam_pollen=None,plotparam_velo=None,verbose=True):
    """
    layer permet de choisir la depth du pollenPlot
    """
    pName = 'contourfMap'
    # fixed aspect_ratio for annulus:
    aspect_ratio = 1.4
    
    plotparam,title,cmap,stagfield,vmin,vmax,levels = __builder(stagData,field=field,axis=None,normal=None,layer=layer,title=title,plotparam=plotparam,verbose=verbose)
    
    #---- Type I : Velocity
    if isinstance(stagVelocity,StagSphericalGeometry):
        if stagVelocity.fieldType is not 'Velocity':
            raise fieldTypeError('Velocity')
        else:
            # prepare plot
            rmin = np.amax(stagData.r) + 0.15
            rmax = rmin+1.1
            pollen_phi  = stagVelocity.phi[0,:,layer]  #surface phi coord
            pollen_vphi = stagVelocity.vphi[0,:,layer]
            pollen_vr   = stagVelocity.vr[0,:,layer]
            # rescale pollenfield between 0 and 1
            maxValvphi = np.amax(pollen_vphi)
            minValvphi = np.amin(pollen_vphi)
            maxValvr   = np.amax(pollen_vr)
            minValvr   = np.amin(pollen_vr)
            if maxValvphi != minValvphi: #vphi_surface
                pollen_vphi = (pollen_vphi-minValvphi)/(maxValvphi-minValvphi)
            else:
                pollen_vphi = pollen_vphi*0+0.5 # fixed to 0.5
            if maxValvr != minValvr: #vr_surface
                pollen_vr = (pollen_vr-minValvr)/(maxValvr-minValvr)
            else:
                pollen_vr = pollen_vr*0+0.5 # fixed to 0.5
            # compute smooth dydx = d(svphi)/phi + d(svr)/phi  in absolute value
            def derivative(x,y):
                diff = np.diff(y)/np.diff(x)
                dydx = np.ones(len(diff)+1)*diff[-1]
                dydx[0:-1] = diff
                return dydx
            dydx = abs(derivative(pollen_phi,pollen_vphi))+abs(derivative(pollen_phi,pollen_vr))
            def gaussian1D(x,x0,sx,A=1):
                return A*np.exp(-((x-x0)/sx)**2)
            fdy  = np.zeros(len(dydx))
            for i in range(len(dydx)):
                fdy += gaussian1D(pollen_phi,pollen_phi[i],2*np.pi/stagVelocity.ny,dydx[i])
            dydx = fdy
            maxdydx   = np.amax(dydx)
            mindydx   = np.amin(dydx)
            if maxdydx != mindydx: #vphi_surface
                dydx = (dydx-mindydx)/(maxdydx-mindydx)
            else:
                dydx = dydx*0+0.5 # fixed to 0.5
            # add rmin
            pollen_vphi += rmin
            pollen_vr   += rmin
            dydx        += rmin
    elif stagVelocity is not None:
        raise StagTypeError(type(stagVelocity),str(StagSphericalGeometry))
    
    #---- Pollenplot of surface field

    if pollenPlot:
        # local extrema
        if plotparam_pollen is not None:
            if plotparam_pollen.vmin != None:
                locminVal = plotparam_pollen.vmin
            else:
                locminVal = vmin
            if plotparam_pollen.vmax != None:
                locmaxVal = plotparam_pollen.vmax
            else:
                locmaxVal = vmax
        else:
            locminVal = vmin
            locmaxVal = vmax
        if isinstance(pollenData,StagSphericalGeometry):
            # r axis
            rmin = np.amax(pollenData.r) + 0.15
            rmax = rmin+1.1
            plotparam_pollen,empty2,empty3,pollenfield,empty4,empty5,empty6 = __builder(pollenData,field=pollenField,axis=None,normal=None,layer=layer,title=title,plotparam=plotparam_pollen,verbose=verbose)
            pollenfield = pollenfield[0,:,layer]
            if locmaxVal != locminVal:
                pollenfield = (pollenfield-locminVal)/(locmaxVal-locminVal)
            else:
                pollenfield = pollenfield*0+0.5 # fixed to 0.5
            # add rmin
            pollenfield += +rmin
        else:
            # r axis
            rmin = np.amax(stagData.r) + 0.15
            rmax = rmin+1.1
            plotparam_pollen,empty2,empty3,pollenfield,empty4,empty5,empty6 = __builder(stagData,field=pollenField,axis=None,normal=None,layer=layer,title=title,plotparam=plotparam_pollen,verbose=verbose)
            pollenfield = pollenfield[0,:,layer]
            if locmaxVal != locminVal:
                pollenfield = (pollenfield-locminVal)/(locmaxVal-locminVal)
            else:
                pollenfield = pollenfield*0+0.5 # fixed to 0.5
            # add rmin
            pollenfield += +rmin
    else:
        rmax = np.amax(stagData.r)
        
    #---- THE PLOT:
    fig = plt.figure(figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
    ax  = fig.add_subplot(111, polar=True)
    if plotparam.aspect == 'equal':
        ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_yticklabels([]) # hide radial label
    ax.set_rlim(0, rmax)
    # - Prepare the data for annulus
    # sorting
    sort_list = np.argsort(stagData.phi[0,:,0])
    phi = stagData.phi[0,:,:][sort_list,:]
    R   = stagData.r[0,:,:][sort_list,:]
    V   = stagfield[0,:,:][sort_list,:]
    # close the annulus
    if stagData.plan == 'yz':
        dtheta = (2*np.pi)/stagData.ny
    elif stagData.plan == 'xz':
        dtheta = (2*np.pi)/stagData.nx
    elif stagData.plan == 'xy':
        dtheta = (2*np.pi)/stagData.nx
    # compute plot data
    if stagData.nx0 != 0:
        wrp_PHI = np.concatenate((phi, phi[-1:] + dtheta))
        wrp_R   = np.concatenate((R, R[0:1, :]), axis=0)
        wrp_V   = np.concatenate((V, V[0:1, :]), axis=0)
    else:
        # Mean that the current stagData object have been made manualy
        wrp_PHI = np.squeeze(stagData.phi)
        wrp_R   = np.squeeze(stagData.r)
        wrp_V   = np.squeeze(stagfield)
    # plot
    cmap4bar = ax.contourf(wrp_PHI,wrp_R,wrp_V,levels=levels,cmap=cmap,extend='both')

    # cbar
    if plotparam.cbar == True:
        if plotparam.cbar_location is not None:
            #cbaxes = fig.add_axes([0.3, 0.1, 0.4, 0.01])
            cbar   = fig.colorbar(cmap4bar, ax=[ax], location=plotparam.cbar_location, shrink=plotparam.cbar_shrink, pad=plotparam.cbar_pad)
        elif cbaxes is not None:
            cbaxes = fig.add_axes(plotparam.cbar_axis) # [left, bottom, width, height]
            cbar   = fig.colorbar(cmap4bar,cax=cbaxes,orientation=plotparam.cbar_orientation)
        
        if plotparam.cbar_label is not None:
            clabel = plotparam.cbar_label
        else:
            clabel = stagData.fieldType
        cbar.set_label(clabel,size=plotparam.cbar_labelsize)
        
    #---- OTHER PLOT:  QUIVER
    if isinstance(stagVelocity,StagSphericalGeometry):
        if stagVelocity.fieldType is not 'Velocity':
            raise fieldTypeError('Velocity')
        else:
            # - Prepare the data for annulus
            # sorting
            sort_list = np.argsort(stagVelocity.phi[0,:,0])
            phi  = stagVelocity.phi[0,:,:][sort_list,:]
            R    = stagVelocity.r[0,:,:][sort_list,:]
            Vphi = stagVelocity.vphi[0,:,:][sort_list,:]
            Vr   = stagVelocity.vr[0,:,:][sort_list,:]
            # close the annulus
            if stagVelocity.plan == 'yz':
                dtheta = (2*np.pi)/stagVelocity.ny
            elif stagVelocity.plan == 'xz':
                dtheta = (2*np.pi)/stagVelocity.nx
            elif stagVelocity.plan == 'xy':
                dtheta = (2*np.pi)/stagVelocity.nx
            # compute plot data
            if stagVelocity.nx0 != 0:
                wrp_PHI  = np.concatenate((phi, phi[-1:] + dtheta))
                wrp_R    = np.concatenate((R, R[0:1, :]), axis=0)
                wrp_Vphi = np.concatenate((Vphi, Vphi[0:1, :]), axis=0)
                wrp_Vr   = np.concatenate((Vr  , Vr[0:1, :]),   axis=0)
            else:
                # Mean that the current stagData object have been made manualy
                wrp_PHI  = np.squeeze(stagVelocity.phi)
                wrp_R    = np.squeeze(stagVelocity.r)
                wrp_Vphi = np.squeeze(stagVelocity.vphi)
                wrp_Vr   = np.squeeze(stagVelocity.vr)
            
            # Quiver does not work properly for a polar figure.
            # -> The vectors are represented as lines with a circle at the origin of each arrows
            xa   = wrp_PHI.flatten()
            ya   = wrp_R.flatten()
            vx   = wrp_Vphi.flatten()
            vy   = wrp_Vr.flatten()
            #
            if plotparam_velo is not None:
                veloScale = plotparam_velo.vscale
                veloWidth = plotparam_velo.arrow_width
                Qscale = plotparam_velo.qscale
                noa = plotparam_velo.noa
            else:
                Qscale = int(np.mean(np.sqrt(vx**2+vy**2)))
                noa = 1000
                veloScale = 3e4
                veloWidth = 0.7
            # noa = number of Arrows
            id = list(range(len(xa)))
            mask = random.sample(id,noa)
            xa = xa[mask]
            ya = ya[mask]
            vx = vx[mask]
            vy = vy[mask]
            #
            sc = veloScale
            ax.scatter(xa,ya,s=1,marker='o',color='black')
            for i in range(len(xa)): # represention with lines
                ax.plot([xa[i],xa[i]+vx[i]/sc],[ya[i],ya[i]+vy[i]/sc],color='black',linewidth=veloWidth)
    elif stagVelocity is not None:
        raise StagTypeError(type(stagVelocity),str(StagSphericalGeometry))        

    if pollenPlot:
        ax.fill_between(wrp_PHI[:,-1],rmin,1+rmin,color='black',alpha=0.05,linewidth=0)
        if isinstance(pollenData,StagSphericalGeometry):
            ax.plot(np.concatenate((pollenData.phi[0,:,layer],pollenData.phi[0,0:1,layer])),np.concatenate((pollenfield,pollenfield[0:1])),label=pollenData.fieldType+' at '+str(int(stagData.depths[layer]))+' km')
        else:
            ax.plot(np.concatenate((stagData.phi[0,:,layer],stagData.phi[0,0:1,layer])),np.concatenate((pollenfield,pollenfield[0:1])),label='Field at '+str(int(stagData.depths[layer]))+' km')
        # -> here below to plot the grid of the pollenPlot (!! concatenate to close the circle !!)
        ax.plot(np.concatenate((stagData.phi[0,:,layer],stagData.phi[0,0:1,layer])),np.concatenate((stagData.phi[0,:,layer]*0+0.00+rmin,stagData.phi[0,0:1,layer]*0+0.00+rmin)),'--',c='black',lw='0.7',alpha=0.5)
        ax.plot(np.concatenate((stagData.phi[0,:,layer],stagData.phi[0,0:1,layer])),np.concatenate((stagData.phi[0,:,layer]*0+1.00+rmin,stagData.phi[0,0:1,layer]*0+1.00+rmin)),'--',c='black',lw='0.7',alpha=0.5)
        if plotparam.gridlines:
            ax.plot(np.concatenate((stagData.phi[0,:,layer],stagData.phi[0,0:1,layer])),np.concatenate((stagData.phi[0,:,layer]*0+0.25+rmin,stagData.phi[0,0:1,layer]*0+0.25+rmin)),'--',c='black',lw='0.5',alpha=0.3)
            ax.plot(np.concatenate((stagData.phi[0,:,layer],stagData.phi[0,0:1,layer])),np.concatenate((stagData.phi[0,:,layer]*0+0.50+rmin,stagData.phi[0,0:1,layer]*0+0.50+rmin)),'--',c='black',lw='0.5',alpha=0.3)
            ax.plot(np.concatenate((stagData.phi[0,:,layer],stagData.phi[0,0:1,layer])),np.concatenate((stagData.phi[0,:,layer]*0+0.75+rmin,stagData.phi[0,0:1,layer]*0+0.75+rmin)),'--',c='black',lw='0.5',alpha=0.3)
        plt.legend(loc='lower left', bbox_to_anchor=(-0.2, 0))
        if plotparam.rticks:
            import matplotlib.ticker as mticker
            label_format = '{:,.2f}'
            if pollenData != None:
                rticks_loc = [locminVal,locminVal+(locmaxVal-locminVal)/2,locmaxVal]
            else:
                rticks_loc = [locminVal,locminVal+(locmaxVal-locminVal)/2,locmaxVal] #[vmin,vmin+(vmax-vmin)/2,vmax]
            loc = [rmin,rmin+0.5,rmin+1]
            ax.yaxis.set_major_locator(mticker.FixedLocator(loc))
            ax.set_yticklabels([label_format.format(x) for x in rticks_loc])
            theta_ticks = plotparam.theta_ticks #110
            ax.plot([theta_ticks*np.pi/180]*3,[rmin,rmin+0.5,rmin+1],'-',c='black',lw='0.5',alpha=0.5)
            ax.plot([theta_ticks*np.pi/180]*3,[rmin,rmin+0.5,rmin+1],'+',c='black',lw='0.5',alpha=0.5)
            ax.set_rlabel_position([theta_ticks]) #rlabel angla
    plt.show()
    
    return 1


def contourfMap_yy(stagData,field='v',axis=0,normal=None,layer=-1,stagVelocity=None,method='tri',interp_method='nearest',locate=None,title=None,pollenPlot=True,pollenData=None,pollenField='v',plotparam=None,plotparam_velo=None,plotparam_pollen=None,verbose=True):
    """
        axis     = int or str, parameter describing the axis of the slice (default, axis=0).
                    axis = 0  or axis = 'annulus'
                                -> an annulus-slice 
                                The normal of the plan containing the annulus is 
                                is given in the 'normal' input argument
                    axis = 1  or axis = 'layer'
                                -> a r-constant-slice (depth-slice, a.k.a iso r)
                                The layer index for the slice is given in the 'layer' input argument
        layer    = int, only for axis >= 1 (Default: layer = -1):
                    if axis == 1:
                            layer will be the value of the stagData.slayers that will be extracted in
                            the new SliceData object.
                            e.g. if you chose layer = 109 then you will extract the layer of the stagData where
                                stagData.slayers == 109
                    else:
                            layer will be the index of the slice on the cartesian mesh.
                            e.g. if axis == 'xy', layer will be the index of the depth
                    
                   
        normal   = list/array, (only if axis == 0), vector of coordinates corresponding
                    to the normal to the plan containing the annlus.
                    This definition is consistent with the normal of the slicing plan in the Paraview software!
                    normal = (nx,ny,nz)
                    Default: normal = (1,0,0)
                    
    method = str: can be 'tri' or 'interp'. This parameter defines the graphical method of representation.
             if method = 'tri' then the map will be generated directly from the sliceData with triangulation (tricontourf)
             if method = 'interp' then the ma will be generated using first an interpolation on a regular lon/lat grid and
                then displayed with pcolormesh
             NOTE: we recommend the 'tri' (triangulation) method
             [default, method = 'tri']
    """
    pName = 'contourfMap'
    if not isinstance(normal,np.ndarray):
        if normal == 'test':
            vx = 0.5717144633095191
            vy = -0.6168453865626955
            vz = -0.540966118642404
            normal = [vx,vy,vz]
    if axis == 0 or axis == 'annulus':
        if normal is None:
            normal = [1,0,0]
    # --- Slicing
    sld = SliceData(geometry=stagData.geometry)
    sld.slicing(stagData,axis=axis,normal=normal,layer=layer,interp_method=interp_method)  # give an axis and here a normal vector
    plotparam,title,cmap,stagfield,vmin,vmax,levels = __builder(sld,field=field,axis=axis,normal=normal,layer=layer,title=title,plotparam=plotparam,verbose=verbose)
    # --- Cases
    if axis == 0 or axis == 'annulus':
        if locate is not None:
            xi   = np.zeros(len(locate)); yi   = np.zeros(len(locate)); zi = np.zeros(len(locate))
            loni = np.zeros(len(locate)); lati = np.zeros(len(locate)); ri = np.zeros(len(locate))
            for i in range(len(locate)):
                xi[i],yi[i],zi[i],loni[i],lati[i],ri[i] = sld.locate_on_annulus_slicing(stagData,locate[i],normal)
        if method == 'interp':
            # --- Direct mapping: no need to interpolate, it's already done for the yy->annulus slice
            fig = plt.figure(figsize=(plotparam.figsize[0]*plotparam.aspect_ratio,plotparam.figsize[1]))
            ax  = fig.add_subplot(111)
            if plotparam.aspect == 'equal':
                ax.set_aspect('equal', 'box')
            cmap4bar = ax.pcolormesh(sld.x,sld.y,stagfield, cmap=cmap, vmin=plotparam.vmin, vmax=plotparam.vmax,antialiased=plotparam.antialiased)
            if locate is not None:
                ax.scatter(xi,yi,c='red',s=20)
        if method == 'tri':
            rmax = np.amax(sld.r)
            if locate:
                rmax_locate = 0.25
                rmax += rmax_locate
            #---- THE PLOT:
            fig = plt.figure(figsize=(plotparam.figsize[0]*plotparam.aspect_ratio,plotparam.figsize[1]))
            ax  = fig.add_subplot(111, polar=True)
            if plotparam.aspect == 'equal':
                ax.set_aspect('equal', 'box')
            ax.set_title(title)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.set_yticklabels([]) # hide radial label
            ax.set_rlim(0, rmax)
            # plot
            cmap4bar = ax.contourf(sld.phi,sld.r,stagfield,levels=levels,cmap=cmap,extend='both',antialiased=plotparam.antialiased)
            if locate is not None:
                import matplotlib.colors as colors
                import matplotlib.cm as cmx
                cNorm     = colors.Normalize(vmin=0, vmax=len(locate)-1)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='jet')
                for i in range(len(loni)):
                    ax.arrow(loni[i], rmax, 0, -(rmax_locate-rmax_locate/2), alpha = 1, width = 0.010,
                         edgecolor = scalarMap.to_rgba(i), facecolor=scalarMap.to_rgba(i), zorder = 5)
                    if loni[i] < np.pi:
                        horizontalalignment = 'left'
                    else:
                        horizontalalignment = 'rigth'
                    verticalalignment = 'bottom'
                    ax.annotate('point '+str(i+1), xy = (loni[i],rmax), xytext = (loni[i],rmax+rmax/15),\
                         verticalalignment=verticalalignment,horizontalalignment=horizontalalignment)
                #ax.scatter(loni,ri,c='red',s=20)
            
            if isinstance(stagVelocity,StagYinYangGeometry):
                if stagVelocity.fieldType is not 'Velocity':
                    raise fieldTypeError('Velocity')
                else:
                    # --- Slicing
                    sldv = SliceData(geometry=stagData.geometry)
                    sldv.slicing(stagVelocity,axis=axis,normal=normal,layer=layer,interp_method=interp_method)  # give an axis and here a normal vector
                    # Quiver does not work properly for a polar figure.
                    # -> The vectors are represented as lines with a circle at the origin of each arrows
                    xa   = sldv.phi.flatten()
                    ya   = sldv.r.flatten()
                    vx   = sldv.vphi.flatten()
                    vy   = sldv.vr.flatten()
                    #
                    if plotparam_velo is not None:
                        veloScale = plotparam_velo.vscale
                        veloWidth = plotparam_velo.arrow_width
                        Qscale = plotparam_velo.qscale
                        noa = plotparam_velo.noa
                    else:
                        Qscale = int(np.mean(np.sqrt(vx**2+vy**2)))
                        noa = 1000
                        veloScale = 3e4
                        veloWidth = 0.7
                    # noa = number of Arrows
                    id = list(range(len(xa)))
                    mask = random.sample(id,noa)
                    xa = xa[mask]
                    ya = ya[mask]
                    vx = vx[mask]
                    vy = vy[mask]
                    #
                    sc = veloScale
                    ax.scatter(xa,ya,s=1,marker='o',color='black')
                    for i in range(len(xa)): # represention with lines
                        ax.plot([xa[i],xa[i]+vx[i]/sc],[ya[i],ya[i]+vy[i]/sc],color='black',linewidth=veloWidth)
    
            #---- Pollenplot of surface field
            if pollenPlot:
                # local extrema
                if plotparam_pollen is not None:
                    if plotparam_pollen.vmin != None:
                        locminVal = plotparam_pollen.vmin
                    else:
                        locminVal = vmin
                    if plotparam_pollen.vmax != None:
                        locmaxVal = plotparam_pollen.vmax
                    else:
                        locmaxVal = vmax
                else:
                    locminVal = vmin
                    locmaxVal = vmax
                if isinstance(pollenData,StagYinYangGeometry):
                    sld_pollen = SliceData(geometry=pollenData.geometry)
                    sld_pollen.slicing(pollenData,axis=axis,normal=normal,layer=layer,interp_method=interp_method)
                    pollenData = sld_pollen
                    plotparam_pollen,empty2,empty3,pollenfield,empty4,empty5,empty6 = __builder(pollenData,field=pollenField,axis=None,normal=None,layer=layer,title=title,plotparam=plotparam_pollen,verbose=verbose)
                    pollenfield = pollenfield[layer,:]
                    # r axis
                    rmin = np.amax(pollenData.r) + 0.15
                    rmax = rmin+1.1
                    #pollenfield = pollenData.v[layer,:]
                    # log10
                    #if plotparam_pollen is not None:
                    #    if plotparam_pollen.log10:
                    #        pollenfield = np.log10(pollenfield)
                    # rescale pollenfield between 0 and 1
                    if locmaxVal != locminVal:
                        pollenfield = (pollenfield-locminVal)/(locmaxVal-locminVal)
                    else:
                        pollenfield = pollenfield*0+0.5 # fixed to 0.5
                    # add rmin
                    pollenfield += +rmin
                else:
                    # r axis
                    rmin = np.amax(sld.r) + 0.15
                    rmax = rmin+1.1
                    plotparam_pollen,empty2,empty3,pollenfield,empty4,empty5,empty6 = __builder(sld,field=pollenField,axis=None,normal=None,layer=layer,title=title,plotparam=plotparam_pollen,verbose=verbose)
                    pollenfield = pollenfield[layer,:]
                    # Log10:
                    #if plotparam.log10:
                    #    pollenfield = np.log10(sld.v[layer,:])
                    #else:
                    #    pollenfield = sld.v[layer,:]
                    # rescale pollenfield between 0 and 1
                    if locmaxVal != locminVal:
                        pollenfield = (pollenfield-locminVal)/(locmaxVal-locminVal)
                    else:
                        pollenfield = pollenfield*0+0.5 # fixed to 0.5
                    # add rmin
                    pollenfield += +rmin
            else:
                rmax = np.amax(sld.r)
            
            # THE PLOT
            if pollenPlot:
                ax.set_ylim(0,rmax)
                ax.fill_between(sld.phi[-1,:],rmin,1+rmin,color='black',alpha=0.05,linewidth=0)
                if isinstance(pollenData,StagSphericalGeometry):
                    ax.plot(np.concatenate((pollenData.phi[layer,:],pollenData.phi[layer,0:1])),np.concatenate((pollenfield,pollenfield[0:1])),label=pollenData.fieldType+' at '+str(int(stagData.depths[layer]))+' km')
                else:
                    ax.plot(np.concatenate((sld.phi[layer,:],sld.phi[layer,0:1])),np.concatenate((pollenfield,pollenfield[0:1])),label='Field at '+str(int(stagData.depths[layer]))+' km')
                # -> here below to plot the grid of the pollenPlot (!! concatenate to close the circle !!)
                ax.plot(np.concatenate((sld.phi[layer,:],sld.phi[layer,0:1])),np.concatenate((sld.phi[layer,:]*0+0.00+rmin,sld.phi[layer,0:1]*0+0.00+rmin)),'--',c='black',lw='0.7',alpha=0.5)
                ax.plot(np.concatenate((sld.phi[layer,:],sld.phi[layer,0:1])),np.concatenate((sld.phi[layer,:]*0+1.00+rmin,sld.phi[layer,0:1]*0+1.00+rmin)),'--',c='black',lw='0.7',alpha=0.5)
                if plotparam.gridlines:
                    ax.plot(np.concatenate((sld.phi[layer,:],sld.phi[layer,0:1])),np.concatenate((sld.phi[layer,:]*0+0.25+rmin,sld.phi[layer,0:1]*0+0.25+rmin)),'--',c='black',lw='0.5',alpha=0.3)
                    ax.plot(np.concatenate((sld.phi[layer,:],sld.phi[layer,0:1])),np.concatenate((sld.phi[layer,:]*0+0.50+rmin,sld.phi[layer,0:1]*0+0.50+rmin)),'--',c='black',lw='0.5',alpha=0.3)
                    ax.plot(np.concatenate((sld.phi[layer,:],sld.phi[layer,0:1])),np.concatenate((sld.phi[layer,:]*0+0.75+rmin,sld.phi[layer,0:1]*0+0.75+rmin)),'--',c='black',lw='0.5',alpha=0.3)
                plt.legend(loc='lower left', bbox_to_anchor=(-0.2, 0))
                if plotparam.rticks:
                    import matplotlib.ticker as mticker
                    label_format = '{:,.2f}'
                    if pollenData != None:
                        rticks_loc = [locminVal,locminVal+(locmaxVal-locminVal)/2,locmaxVal]
                    else:
                        rticks_loc = [locminVal,locminVal+(locmaxVal-locminVal)/2,locmaxVal] #[vmin,vmin+(vmax-vmin)/2,vmax]
                    loc = [rmin,rmin+0.5,rmin+1]
                    ax.yaxis.set_major_locator(mticker.FixedLocator(loc))
                    ax.set_yticklabels([label_format.format(x) for x in rticks_loc])
                    theta_ticks = plotparam.theta_ticks #110
                    ax.plot([theta_ticks*np.pi/180]*3,[rmin,rmin+0.5,rmin+1],'-',c='black',lw='0.5',alpha=0.5)
                    ax.plot([theta_ticks*np.pi/180]*3,[rmin,rmin+0.5,rmin+1],'+',c='black',lw='0.5',alpha=0.5)
                    ax.set_rlabel_position([theta_ticks]) #rlabel angla
    
    
    if axis == 1 or axis == 'layer':
        if normal is not None:
            im('Compute the projection of the slicing annulus plan at the surface of the sphere',pName,verbose)
            small = 1e-10           # to avoid to divide by 0
            normal[0] += small      #
            normal[1] += small      #
            normal[2] += small      #
            a = normal[0]
            b = normal[1]
            c = normal[2]
            # Compute the projection of the plan
            nplan   = 10000 #5000000
            lonp = np.linspace(0,360,nplan)*np.pi/180
            latp = np.zeros(nplan)
            rp   = np.array(stagData.z_coords)+stagData.rcmb
            rp   = np.ones(nplan)*rp[layer]
            xp = rp*np.cos(latp)*np.cos(lonp)
            yp = rp*np.cos(latp)*np.sin(lonp)
            zp = rp*np.sin(latp)
            # Compute the normal vectors to the slicing plan:
            normalu = np.array([1,-a/b,0])
            normalv = np.array([a/b,1,-(a**2+b**2)/(c*b)])
            normalw = np.array([a,b,c])
            base_plan = np.array([normalu,normalv,normalw])
            base_cart = np.eye(3)
            mpassage = np.dot(np.linalg.inv(base_plan),base_cart)
            xpp = np.zeros(nplan)
            ypp = np.zeros(nplan)
            zpp = np.zeros(nplan)
            for i in range(nplan):
                xpp[i],ypp[i],zpp[i] = np.dot(np.array([xp[i],yp[i],zp[i]]),mpassage.T)
                xpp[i],ypp[i],zpp[i] = rp[0]*np.array([xpp[i],ypp[i],zpp[i]])/np.linalg.norm(np.array([xpp[i],ypp[i],zpp[i]]))
            rpp   = np.sqrt(xpp**2+ypp**2+zpp**2)
            latpp = -(np.arctan2(np.sqrt(xpp**2+ypp**2),zpp)*180/np.pi-90)
            lonpp = np.arctan2(ypp,xpp)*180/np.pi
            # ---
            # Compute the thickness of the slice:
            R = np.sqrt((4*np.pi*2.19**2)/(stagData.nx*stagData.ny))*3
            # Plan equation:
            #gind = np.where(abs(a*sld.x+b*sld.y+c*sld.z) <= R)[0]
            #lonNormal  = sld.phi[gind]*180/np.pi; latNormal  = -(sld.theta[gind]*180/np.pi-90)
  
        if method == 'interp':
            # --- interpolation on a regular lon/lat 2D grid
            totSurf   = np.count_nonzero(stagData.layers == stagData.slayers[0])*2
            opti_size = np.sqrt(totSurf/2)
            spacing   = 180/opti_size     # spacing (in deg) between each new points (in lon and in lat)
            # Build the regular grid
            from .stagInterpolator import sliceYYInterpolator_mapping
            lonRG,latRG,vInterp = sliceYYInterpolator_mapping(sld,field,spacing=spacing,interpMethod='nearest',\
                                                            verbose=True,log10=plotparam.log10,deg=True)
        # --- Mapping
        fig = plt.figure(figsize=(plotparam.figsize[0]*plotparam.aspect_ratio,plotparam.figsize[1]))
        ax = fig.add_subplot(1,1,1, projection=plotparam.projection)
        ax.set_title(title)
        if plotparam.aspect == 'equal':
            ax.set_aspect('equal', 'box')
        ax.set_global()
        if plotparam.gridlines:
            gllinewidth = 1
            glalpha = 0.5
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=plotparam.mapticks,
                  linewidth=gllinewidth, color='gray', alpha=glalpha, linestyle='--')
            gl.rotate_labels = False
        if method == 'interp':
            cmap4bar = ax.pcolormesh(lonRG,latRG,vInterp, cmap=cmap, vmin=plotparam.vmin, vmax=plotparam.vmax,antialiased=plotparam.antialiased, transform=plotparam.transform)
        elif method == 'tri':
            cmap4bar = ax.tricontourf(sld.phi*180/np.pi,-(sld.theta*180/np.pi-90),stagfield, cmap=cmap, levels=levels,extend='both',transform=plotparam.transform)
        if normal is not None:
            #ax.scatter(lonNormal,latNormal,c='red',s=1,transform=ccrs.PlateCarree(),alpha=0.5)
            ax.plot(lonpp,latpp,color=plotparam.linecolor,linewidth=plotparam.linewidth,transform=ccrs.Geodetic())
            # ========== EXTRACT THE FIELD ALONG THE SLICING PLAN =================
            from scipy.interpolate import griddata
            points = np.array([(sld.phi[i]*180/np.pi,-(sld.theta[i]*180/np.pi-90)) for i in range(len(sld.phi))])
            v_profile = griddata(points, stagfield, (lonpp, latpp), method='nearest')
            # =================================================================================
    
        if isinstance(stagVelocity,StagYinYangGeometry):
            if stagVelocity.fieldType is not 'Velocity':
                raise fieldTypeError('Velocity')
            else:
                # --- Slicing
                sldv = SliceData(geometry=stagVelocity.geometry)
                sldv.slicing(stagVelocity,axis=axis,normal=normal,layer=layer,interp_method=interp_method)  # give an axis and here a normal vector
    
                xa   = sldv.phi*180/np.pi ; xa = xa.flatten()
                ya   = -(sldv.theta*180/np.pi-90) ; ya = ya.flatten()
                vx   = sldv.vphi.flatten()
                vy   = -sldv.vtheta.flatten()
                #
                if plotparam_velo is not None:
                    veloScale = plotparam_velo.vscale
                    veloWidth = plotparam_velo.arrow_width
                    Qscale = plotparam_velo.qscale
                    noa = plotparam_velo.noa
                else:
                    Qscale = int(np.mean(np.sqrt(vx**2+vy**2)))
                    noa = 1000
                    veloScale = None
                    veloWidth = None
                # noa = number of Arrows
                id = list(range(len(xa)))
                mask = random.sample(id,noa)
                xa = xa[mask]
                ya = ya[mask]
                vx = vx[mask]
                vy = vy[mask]
                # quiver
                Q  = ax.quiver(xa,ya,vx,vy,scale=veloScale,width=veloWidth,transform=plotparam.transform)
                qq = ax.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
            
    
    # cbar
    if plotparam.cbar == True:
        if plotparam.cbar_location is not None:
            #cbaxes = fig.add_axes([0.3, 0.1, 0.4, 0.01])
            cbar   = fig.colorbar(cmap4bar, ax=[ax], location=plotparam.cbar_location, shrink=plotparam.cbar_shrink, pad=plotparam.cbar_pad)
        elif cbaxes is not None:
            cbaxes = fig.add_axes(plotparam.cbar_axis) # [left, bottom, width, height]
            cbar   = fig.colorbar(cmap4bar,cax=cbaxes,orientation=plotparam.cbar_orientation)
        
        if plotparam.cbar_label is not None:
            clabel = plotparam.cbar_label
        else:
            clabel = stagData.fieldType
        cbar.set_label(clabel,size=plotparam.cbar_labelsize)
    # --- end ---
    if plotparam.save:
        im("Save images under:\n"+plotparam.path+plotparam.name,pName,verbose)
        plt.savefig(plotparam.path+plotparam.name,dpi=plotparam.dpi)
    if plotparam.show:
        fig.show()
    else:
        plt.close(fig)
    # return
    if axis == 1 or axis == 'layer':
        if normal is not None:
            return v_profile
        else:
            return 0
    return 0
                










def spaceTimeDiagram(stagCloudData,field='v',axis=0,layer=-1,timepar=0,\
                     plotparam='Default',aspect_ratio=1,verbose=True):
    """
    Generate a space-time diagram on a CloudStagData creating  and stacking slices
    on a 2D CloudStagData.drop.

    <i> stagCloudData = pypStag.stagData.stagClouData object which contain
                        info you want to display.
                          * IMPORTANT *
                          stagCloudData have to represent a time series of 2D stagData!
                          (annulus or cart2D) and is incompatble for the moment with 3D geometries
        field = str, select here the type of field contained in the input.
                structure that you want to display. Identical to the 'field'
                option of the stagMap function. Differents values of 'field':
                <for all stagData>:
                    field = 'scalar' or 'v': take the 'v' scalar field of the stagData object
                <only for vectorial stagData>:
                    field = 'vx' or 'vy' or 'vz', take the corresponding cartesian component of a 
                            vetorial field
                    field = 'vr' or 'vtheta' or 'vphi', take the corresponding spherical component
                            of a vectorial field
                    field = 'P' or 'p', take the Pressure field of the input stagData
    """
    pName = 'spaceTimeDiagram'
    # Test geometry:
    if stagCloudData.geometry != 'cart2D' and stagCloudData.geometry != 'annulus':
        raise VisuGridGeometryError(stagCloudData.geometry,'cart2D or annulus')
    # read the first file to initiate field    
    stagCloudData.iterate()
    sld = SliceData(geometry=stagCloudData.geometry)
    sld.verbose = False
    sld.sliceExtractor(stagCloudData.drop,axis=axis,layer=layer)
    stagCloudData.reset()
    tSlice  = np.zeros((stagCloudData.nt,len(sld.v)))
    axedata = np.ones((stagCloudData.nt,len(sld.v)))
    simuAge = np.zeros((stagCloudData.nt,len(sld.v)))
    ti_step = np.zeros((stagCloudData.nt,len(sld.v)))
    fileid  = np.zeros((stagCloudData.nt,len(sld.v)))
    # true run
    from tqdm import tqdm
    im('Selected field: '+field,pName,verbose)
    for i in tqdm(range(stagCloudData.nt)):
        j = stagCloudData.indices[i]
        stagCloudData.iterate() 
        sld = SliceData(geometry=stagCloudData.geometry)
        sld.verbose = False
        sld.sliceExtractor(stagCloudData.drop,axis=axis,layer=layer)
        # ---------------------
        # selection of the good field:
        if field == 'scalar' or field == 'v':
            stagfield = sld.v
        elif field == 'vx':
            if sld.fieldNature == 'Vectorial':
                stagfield = sld.vx
            else:
                raise StagMapFieldError(field,stagCloudData.geometry,sld.fieldNature)
        elif field == 'vy':
            if sld.fieldNature == 'Vectorial':
                stagfield = sld.vy
            else:
                raise StagMapFieldError(field,stagCloudData.geometry,sld.fieldNature)
        elif field == 'vz':
            if sld.fieldNature == 'Vectorial':
                stagfield = sld.vz
            else:
                raise StagMapFieldError(field,stagCloudData.geometry,sld.fieldNature)
        elif field == 'vphi':
            if sld.fieldNature == 'Vectorial' and stagCloudData.geometry == 'annulus':
                stagfield = sld.vphi
            else:
                raise StagMapFieldError(field,stagCloudData.geometry,sld.fieldNature)
        elif field == 'vtheta':
            if sld.fieldNature == 'Vectorial' and stagCloudData.geometry == 'annulus':
                stagfield = sld.vtheta
            else:
                raise StagMapFieldError(field,stagCloudData.geometry,sld.fieldNature)
        elif field == 'vr':
            if sld.fieldNature == 'Vectorial' and stagCloudData.geometry == 'annulus':
                stagfield = sld.vr
            else:
                raise StagMapFieldError(field,stagCloudData.geometry,sld.fieldNature)
        elif field == 'p' or field == 'P':
            if sld.fieldNature == 'Vectorial':
                stagfield = sld.P
            else:
                raise StagMapFieldError(field,stagCloudData.geometry,sld.fieldNature)
        else:
            raise StagMapUnknownFieldError(field)
        # --- write ---
        tSlice[i,:]    = stagfield
        simuAge[i,:]   = np.ones(len(sld.v))*stagCloudData.drop.simuAge
        ti_step[i,:]   = np.ones(len(sld.v))*stagCloudData.drop.ti_step
        fileid[i,:]    = np.ones(len(sld.v))*i
    
    # --- the grid ---
    if stagCloudData.geometry == 'annulus':
        axedata = axedata*sld.phi
    elif stagCloudData.geometry == 'cart2D' or stagCloudData.geometry == 'cart3D':
        if sld.plan in ['xz','zx']:
            axedata = axedata*sld.x
        elif sld.plan in ['yz','zy']:
            axedata = axedata*sld.y
        else:
            print('ERROR: break operation')
    #---------- Figure ----------#
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






def yyTectoMap2(sd_cont,sd_vor,sd_div,sd_vp,plotparam=None,\
                rotation=None,noa = 1000,veloScale=None,veloWidth=5e-4,Qscale=0,\
                addtext=None,verbose=True,max_vor = 6e4):
    """
    sd_vor,sd_div,sd_vp
    """
    pName = 'yyTectoMap2'
    from .stagInterpolator import sliceYYInterpolator_mapping

    # --- Prepare the figure
    fig = plt.figure(figsize=(plotparam.figsize[0]*plotparam.aspect_ratio,plotparam.figsize[1]))
    ax = fig.add_subplot(1, 1, 1,projection=plotparam.projection)
    ax.set_title("yyTectoMap")
    ax.set_global()
    
    if plotparam.gridlines:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.5, linestyle='--')
        gl.xlabels_top = True
        gl.ylabels_left = True
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator([-60,-30,0,30,60])
        gl.xlocator = mticker.FixedLocator([-180,-90,0,90,180])
    
    
    # --- Vorticity
    # slicing
    sld = SliceData(geometry=sd_vor.geometry)
    sld.slicing(sd_vor,axis='layer',normal=None,layer=-1)
    lon,lat,v = sliceYYInterpolator_mapping(sld,field='v',spacing=0.4)
    # plot
    
    
    cmap = plt.cm.Greys
    cmap = ax.imshow(abs(v.T),extent=(-180,180,90,-90),cmap=cmap,transform=ccrs.PlateCarree(),vmin=0,vmax=max_vor)
    cbaxes = fig.add_axes([0.91, 0.3, 0.01, 0.4]) # [left, bottom, width, height]
    cbar_vor = plt.colorbar(cmap,cax=cbaxes)
    #cbar_vor.set_label('abs(vorticity)')
    cbar_vor.ax.set_title('abs(vorticity)')
    
    # --- Continents
    # slicing
    sld = SliceData(geometry=sd_cont.geometry)
    sld.slicing(sd_cont,axis='layer',normal=None,layer=-1)
    lon,lat,v = sliceYYInterpolator_mapping(sld,field='v',spacing=0.4)
    # processing
    countcont = np.unique(v)
    vmin_cont = countcont[1]
    vmax_cont = countcont[-1]
    bind = v == 0
    v[bind] = np.nan
    # plot
    cmap4cont = plt.cm.get_cmap('viridis', len(countcont)-1)
    cmap_cont = ax.imshow(v.T, extent=(-180,180,90,-90),cmap=cmap4cont,transform=ccrs.PlateCarree(),vmin=vmin_cont,vmax=vmax_cont,alpha=0.5)
    cbaxes_cont = fig.add_axes([0.3, 0.05, 0.4, 0.01]) # [left, bottom, width, height]
    cbar_cont = plt.colorbar(cmap_cont,cax=cbaxes_cont,orientation='horizontal')
    cbar_cont.set_label('continents')
    
    
    # --- Divergence
    # slicing
    sld = SliceData(geometry=sd_div.geometry)
    sld.slicing(sd_div,axis='layer',normal=None,layer=-1)
    lon,lat,v = sliceYYInterpolator_mapping(sld,field='v',spacing=0.4)
    # processing
    limit = 10000
    minValdiv = -1.25e5
    maxValdiv = +1.25e5
    bind = abs(v) < limit
    v[bind] = np.nan
    # plot
    try:
        from .cptReader import GCMTcolormap
        colormapdiv = GCMTcolormap(plotparam.crameri_path+'vik/vik.cpt',reverse=plotparam.reverseCMAP)
    except:
        im("WARNING: Unknown colormap file",pName,verbose)
        colormapdiv = plt.cm.seismic
    # Add to plot
    cmap_div = ax.imshow(v.T, extent=(-180,180,90,-90),cmap=colormapdiv,transform=ccrs.PlateCarree(),vmin=minValdiv,vmax=maxValdiv)
    cbaxes_div = fig.add_axes([0.05, 0.3, 0.01, 0.4]) # [left, bottom, width, height]
    cbar_div = plt.colorbar(cmap_div,cax=cbaxes_div)
    #cbar_div.set_label('divergence')
    cbar_div.ax.set_title('divergence')
    
    
    # --- velocity
    # slicing
    sld = SliceData(geometry=sd_vp.geometry)
    sld.slicing(sd_vp,axis='layer',normal=None,layer=-1)
    lon,lat,vx = sliceYYInterpolator_mapping(sld,field='vphi',spacing=0.4)
    lon,lat,vy = sliceYYInterpolator_mapping(sld,field='vtheta',spacing=0.4)
    # process
    vlat  = -vy.flatten()
    vlon  =  vx.flatten()
    lon = lon.flatten()*np.pi/180
    lat = lat.flatten()*np.pi/180
    # resample
    id = list(range(len(lon)))
    mask = random.sample(id,noa)
    vlat = vlat[mask]
    vlon = vlon[mask]
    lat = lat[mask]
    lon = lon[mask]
    #vlat  = -vy[::vresampling[0],::vresampling[1]].flatten()
    #vlon  =  vx[::vresampling[0],::vresampling[1]].flatten()
    #lon = lon[::vresampling[0],::vresampling[1]].flatten()*np.pi/180
    #lat = lat[::vresampling[0],::vresampling[1]].flatten()*np.pi/180
    #
    
    if rotation is not None:
        temp1,temp2,vx = sliceYYInterpolator_mapping(sld,field='vx',spacing=0.4)
        temp1,temp2,vy = sliceYYInterpolator_mapping(sld,field='vy',spacing=0.4)
        temp1,temp2,vz = sliceYYInterpolator_mapping(sld,field='vz',spacing=0.4)
        #vx = vx[::vresampling[0],::vresampling[1]].flatten()
        #vy = vy[::vresampling[0],::vresampling[1]].flatten()
        #vz = vz[::vresampling[0],::vresampling[1]].flatten()
        #
        vx = vx.flatten()
        vy = vy.flatten()
        vz = vz.flatten()
        # resample
        vx = vx[mask]
        vy = vy[mask]
        vz = vz[mask]        
        #
        wx,wy,wz = rotation
        # --- fipole
        R = 2.19889718407
        x = R*np.cos(lat)*np.cos(lon)
        y = R*np.cos(lat)*np.sin(lon)
        z = R*np.sin(lat)
        # cartfipole
        N  = len(x)
        vxfit = np.zeros(N)
        vyfit = np.zeros(N)
        vzfit = np.zeros(N)
        for i in range(N):
            xi,yi,zi = x[i],y[i],z[i]
            A        = np.array([[0, zi, -yi],\
                                [-zi, 0, xi],\
                                [yi, -xi, 0]])
            vxfit[i],vyfit[i],vzfit[i] = np.dot(A,np.array([wx,wy,wz]))
        # then
        nod2 = len(lon)
        dvlon  = np.zeros(nod2)     # lon misfit
        dvlat  = np.zeros(nod2)     # lat misfit
        dvr    = np.zeros(nod2)     # r misfit
        vlonfit = np.zeros(nod2)    # lon v fit
        vlatfit = np.zeros(nod2)    # lat v fit
        rfit   = np.zeros(nod2)     # r v fit
        londv  = np.zeros(nod2)     # lon v data
        latdv  = np.zeros(nod2)     # lat v data
        rdv    = np.zeros(nod2)     # r v data
        # iterate
        for i in range(nod2):
            lon_i = lon[i]*np.pi/180
            lat_i = lat[i]*np.pi/180
            dvxi   = vx[i]-vxfit[i]
            dvyi   = vy[i]-vyfit[i]
            dvzi   = vz[i]-vzfit[i]
            vxifit = vxfit[i]
            vyifit = vyfit[i]
            vzifit = vzfit[i]
            vxd = vx[i]
            vyd = vy[i]
            vzd = vz[i]
            Rmatrix = Rgt(lat_i,lon_i)
            dVxyz     = np.array([dvxi,dvyi,dvzi])          # residual
            Vxyz_fit  = np.array([vxifit,vyifit,vzifit])    # fit
            Vxyz_d    = np.array([vxd,vyd,vzd])             # data
            dvlon[i],dvlat[i],dvr[i]      = np.dot(Rmatrix,dVxyz)
            vlonfit[i],vlatfit[i],rfit[i] = np.dot(Rmatrix,Vxyz_fit)
            londv[i],latdv[i],rdv[i]      = np.dot(Rmatrix,Vxyz_d)
        # modify the velocity that will be displayed
        vlon = dvlon
        vlat = dvlat
    
    # plot
    lon = lon*180/np.pi
    lat = lat*180/np.pi
    Q = ax.quiver(lon,lat,vlon,vlat,scale=veloScale,width=veloWidth,transform=ccrs.PlateCarree())
    qq = ax.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
    
    
    # Add text
    if addtext is not None:
        ax.text(-0.1,0.9,addtext,transform=ax.transAxes)
    # --- end ---
    if plotparam.save:
        im("Save images under:\n"+plotparam.path+plotparam.name,pName,verbose)
        plt.savefig(plotparam.path+plotparam.name,dpi=plotparam.dpi)
    if plotparam.show:
        plt.show()
    else:
        plt.close(fig)









def yyTectoMap(sliceData,plotparam='Default',aspect_ratio=1,sliceVelocity=None,\
    vresampling=[1,1],veloScale=None,veloWidth=None,Qscale=1000, \
    projection=None,div=None,cont=None,velo=None,hspot_centroidIDs=None,\
    addtext=None,verbose=True):
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
    pName = 'yyTectoMap'
    im('Creation of the sliceMap',pName,verbose)
    #Typing:
    if not isinstance(sliceData,InterpolatedSliceData) and not isinstance(sliceData,CartesianSliceData):
        raise StagTypeError(str(type(sliceData)),'stagData.InterpolatedSliceData or stagData.CartesianSliceData')
    # Test geometry:
    #if sliceData.geometry == 'cart2D' or sliceData.geometry == 'annulus':
    #    raise VisuGridGeometryError(sliceData.geometry,'interpolated or cart3D')
    # Test colormap:
    if plotparam.cmaptype == 'matplotlib':
        cmap = plotparam.cmap
    else:
        try:
            from .cptReader import GCMTcolormap
            cmap = GCMTcolormap(plotparam.cmap,reverse=plotparam.reverseCMAP)
        except:
            im("WARNING: Unknown colormap file",pName,verbose)
            cmap = plt.cm.seismic
        if plotparam == 'Default':
            plotparam = PlotParam()
        else:
            plotparam.update()
    # Log10:
    if plotparam.log10:
        im('Requested: log10',pName,verbose)
        slicefield = np.log10(sliceData.v)
    else:
        slicefield = sliceData.v
    # title
    if plotparam.title == '':
        field = 'Vor '
        if div != None:
            field += '+ Div '
        if cont != None:
            field += '+ Contients '
        if velo != None:
            field += '+ Velo '
        if list(hspot_centroidIDs) != None:
            field += '+ Hotspots '
        title = 'yyTectoMap: fields='+field
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
        im('  - sliceMap from InterpolatedSliceData',pName,verbose)
        if isinstance(sliceVelocity,InterpolatedSliceData):
            im('  - velocities: True',pName,verbose)
            fig = plt.figure(figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
            ax = fig.add_subplot(1, 1, 1,projection=projection)
            ax.set_title(title)
            X = sliceData.x.reshape(sliceData.nxi,sliceData.nyi)
            Y = sliceData.y.reshape(sliceData.nxi,sliceData.nyi)
            V = slicefield.reshape(sliceData.nxi,sliceData.nyi)
            if projection == None:
                im('  - projection: None',pName,verbose)
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
                im('  - projection: True',pName,verbose)
                Q = ax.quiver(X.flatten(),Y.flatten(),Vx.flatten(),Vy.flatten(),\
                             scale=veloScale,width=veloWidth,label='Horizontal Velocity field',\
                             transform=ccrs.PlateCarree())
                #ax.set_extent([-np.pi, +np.pi, -np.pi/2, +np.pi/2], ccrs.PlateCarree())
            else:
                Q = ax.quiver(X.flatten(),Y.flatten(),Vx.flatten(),Vy.flatten(),\
                             scale=veloScale,width=veloWidth,label='Horizontal Velocity field')
            qq = ax.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
            ax.legend(loc='lower right')
        else:
            im('  - velocities: False',pName,verbose)
            fig = plt.figure(figsize=(plotparam.figsize[0]*aspect_ratio,plotparam.figsize[1]))
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            ax.set_title(title)
            X = sliceData.x.reshape(sliceData.nxi,sliceData.nyi)
            Y = sliceData.y.reshape(sliceData.nxi,sliceData.nyi)
            V = slicefield.reshape(sliceData.nxi,sliceData.nyi)
            if projection == None:
                im('  - projection: None',pName,verbose)
                cmap = ax.contourf(X,Y,V,levels=levels,cmap=cmap,extend='both')
                ax.set_xlabel('x-axis')
                ax.set_ylabel('y-axis')
            else:
                im('  - projection: True',pName,verbose)
                cmap = ax.imshow(V.T, extent=(0,360,90,-90),cmap=cmap,transform=ccrs.PlateCarree(),vmin=minVal,vmax=maxVal)
                if div != None:
                    limit = 10000
                    # Process Div: Put to NaN all points with a abs(value) < limit
                    Vdiv = div.v.reshape(div.nxi,div.nyi)
                    minValdiv = -1.25e5
                    maxValdiv = +1.25e5
                    bind = abs(Vdiv) < limit
                    Vdiv[bind] = np.nan
                    # Cmap
                    try:
                        from .cptReader import GCMTcolormap
                        colormapdiv = GCMTcolormap(plotparam.crameri_path+'vik/vik.cpt',reverse=plotparam.reverseCMAP)
                    except:
                        im("WARNING: Unknown colormap file",pName,verbose)
                        colormapdiv = plt.cm.seismic
                    # Add to plot
                    cmap_div = ax.imshow(Vdiv.T, extent=(0,360,90,-90),cmap=colormapdiv,transform=ccrs.PlateCarree(),vmin=minValdiv,vmax=maxValdiv)
                    cbaxes_div = fig.add_axes([0.05, 0.3, 0.01, 0.4]) # [left, bottom, width, height]
                    cbar_div = plt.colorbar(cmap_div,cax=cbaxes_div)
                    cbar_div.set_label('divergence')
                if cont != None:
                    # Process Cont Contours lines
                    Xcont = cont.lon.reshape(cont.nxi,cont.nyi)
                    Ycont = cont.lat.reshape(cont.nxi,cont.nyi)
                    Vcont = cont.v.reshape(cont.nxi,cont.nyi)
                    # Add to plot:
                    countcont = np.unique(Vcont)
                    vmin_cont = countcont[1]
                    vmax_cont = countcont[-1]
                    bind = Vcont == 0
                    Vcont[bind] = np.nan

                    cmap4cont = plt.cm.get_cmap('viridis', len(countcont)-1)
                    cmap_cont = ax.imshow(Vcont.T, extent=(0,360,90,-90),cmap=cmap4cont,transform=ccrs.PlateCarree(),vmin=vmin_cont,vmax=vmax_cont,alpha=0.5)
                    cbaxes_cont = fig.add_axes([0.3, 0.05, 0.4, 0.01]) # [left, bottom, width, height]
                    cbar_cont = plt.colorbar(cmap_cont,cax=cbaxes_cont,orientation='horizontal')
                    cbar_cont.set_label('continents')
                    """
                    CS    = ax.contour(Xcont,Ycont,Vcont,transform=ccrs.PlateCarree())
                    # Remove small contients
                    d = []
                    for level in CS.collections:
                        for kp,path in reversed(list(enumerate(level.get_paths()))):
                            # go in reversed order due to deletions!

                            # include test for "smallness" of your choice here:
                            # I'm using a simple estimation for the diameter based on the
                            #    x and y diameter...
                            verts = path.vertices # (N,2)-shape array of contour line coordinates
                            diameter = np.max(verts.max(axis=0) - verts.min(axis=0))
                            d.append(diameter)
                            if diameter<50: # threshold to be refined for your actual dimensions!
                                del(level.get_paths()[kp])  # no remove() for Path objects:(
                    # this might be necessary on interactive sessions: redraw figure
                    plt.gcf().canvas.draw()
                    """
                if velo != None:
                    X  = velo.lon.reshape(velo.nxi,velo.nyi)[::vresampling[0],::vresampling[1]]
                    Y  = velo.lat.reshape(velo.nxi,velo.nyi)[::vresampling[0],::vresampling[1]]
                    #
                    Vx = velo.vphi.reshape(velo.nxi,velo.nyi)[::vresampling[0],::vresampling[1]]
                    Vy = -velo.vtheta.reshape(velo.nxi,velo.nyi)[::vresampling[0],::vresampling[1]]
                    #
                    im('  - projection: True',pName,verbose)
                    Q = ax.quiver(X.flatten(),Y.flatten(),Vx.flatten(),Vy.flatten(),\
                                  scale=veloScale,width=veloWidth,transform=ccrs.PlateCarree())
                    qq = ax.quiverkey(Q,0.95,-0.1,Qscale,'v='+str(Qscale),labelpos='E')
                    #ax.legend(loc='lower right')
                
                if list(hspot_centroidIDs) != None:
                    xc = []; yc = []
                    nc = len(hspot_centroidIDs)
                    for i in range(nc):
                        xc.append(sliceData.lon.flatten()[hspot_centroidIDs[i]])
                        yc.append(sliceData.lat.flatten()[hspot_centroidIDs[i]])
                    ax.scatter(xc,yc,marker='D',c='red',s=20,alpha=1,transform=ccrs.PlateCarree(),label='hotspots')
                    ax.legend(loc = 4)

        # Now adding the colorbar
        cbaxes = fig.add_axes([0.91, 0.3, 0.01, 0.4]) # [left, bottom, width, height]
        cbar = plt.colorbar(cmap,cax=cbaxes,label='vorticity')
        # Add text
        if addtext != None:
            ax.text(-0.1,0.9,addtext,transform=ax.transAxes)
        # --- end ---
        if plotparam.save:
            im("Save images under:\n"+plotparam.path+plotparam.name,pName,verbose)
            plt.savefig(plotparam.path+plotparam.name,dpi=plotparam.dpi)
        if plotparam.show:
            fig.show()
        else:
            plt.close(fig)












