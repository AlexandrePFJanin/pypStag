# -*- coding: utf-8 -*-
"""
Created on Sun Jul 04 15:22:00 2021

@author: Alexandre
"""

import matplotlib.pyplot as plt
import numpy as np
from .stagData import StagData
from .stagError import StagTypeError,InputGridGeometryError,fieldTypeError


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



def get_xzy_scoords(stagData,lon,lat,verbose=True):
    """
    Function that return the x,y and z coord of a point at the surface of a YY model
    from a given lon lat coordinates.
    
    <i> : stagData = pypStag.stagData.StagYinYangGeometry
          lon,lat = float,float coordinates [in DEGREES] of the point
    """
    if stagData.geometry != 'yy':
        raise StagTypeError(str(type(stagData)),'pypStag.stagData.StagYinYangGeometry')
    else:
        from .stagData  import SliceData
        pName = 'get_xzy_scoords'
        im('Search the nearest point to:',pName,verbose=verbose)
        im('lon = '+str(lon)+', lat = '+str(lat),pName,verbose=verbose)
        sld = SliceData(geometry=stagData.geometry)
        sld.verbose = False
        sld.slicing(stagData,axis='layer',normal=None,layer=-1)
        long = sld.phi*180/np.pi
        latg = -(sld.theta*180/np.pi-90)
        dist = np.sqrt((long-lon)**2+(latg-lat)**2)
        gind = np.where(dist == np.amin(dist))[0][0]
        x,y,z = sld.x[gind],sld.y[gind],sld.z[gind]
        im('   -> Point index: '+str(gind),pName,verbose=verbose)
        im('   -> found lon/lat: '+str(int(long[gind]*100)/100)+', '+str(int(latg[gind]*100)/100),pName,verbose=verbose)
        return x,y,z



def get_normal_2points(stagData,point1,point2,verbose=True):
    """
    Compute the normal vector of the plan passing by (lon1,lat1), (lon2,lat2) and
    by the origin (the center of the Earth: 0,0,0). The norm of this vector is set to 1.
    This function have been made to be used with the slicing pachage to define slicing plan.
    
    <i> : stagData = pypStag.stagData.StagYinYangGeometry
          point1 = [lon1,lat1] = [float,float] coordinates [in DEGREES] of the point 1
          point2 = [lon2,lat2] = [float,float] coordinates [in DEGREES] of the point 2
    """
    pName = 'get_normal_2points'
    lon1,lat1 = point1
    lon2,lat2 = point2
    im('Compute the normal vector of the plan passing by:',pName,verbose=verbose)
    im('   -> lon1,lat1 = '+str(lon1)+', '+str(lat1),pName,verbose=verbose)
    im('   -> lon2,lat2 = '+str(lon2)+', '+str(lat2),pName,verbose=verbose)
    im('   -> x0,y0,z0  = 0, 0, 0',pName,verbose=verbose)
    # --- Compute the xyz coordinates of the two points
    x1,y1,z1 = get_xzy_scoords(stagData,lon1,lat1,verbose=verbose)
    x2,y2,z2 = get_xzy_scoords(stagData,lon2,lat2,verbose=verbose)
    # --- Formulation of the problem under the form AX = B
    A = np.array([[x1, y1, z1],\
                  [x2, y2, z2],\
                  [1,  1,  1]]).T
    normal = np.dot(np.array([0,0,1]),np.linalg.inv(A))
    im('Done',pName,verbose=verbose)
    return normal



def automatic_hotspot_detection(slice350kmt,slice350kmv,remove_smaller=10,plot=False,verbose=True):
    """
    This function detects automatically hotspots using an interpolatedsliceData on temperature
    and an interpolatedsliceData on velocity. The slices have to be cut arround 191 km depth
    (corresponding to the layer index 350km in the LLSVP2019 model).
    The detection is based on to threshold : when vr >= vr_threshold .AND.
    t >= t_threshold then the point is considered as a hotspot.
    The function will then clusterize the resulting detected hotspot points and
    will return the indices of points in the input interpolatedslideData forming a cluster
    and the indices of the centroid for all hotspots.
    <i> : slice350kmt  = pypStag.stagData.InterpolatedSliceData from a temperature SliceData
                       cut on layer=350km (depth=350km,layer=100)
          slice350kmvp = pypStag.stagData.InterpolatedSliceData from a velocity-pressure
                       SliceData cut on layer=350km (depth=350km,layer=100)
          vr_threshold = int/float, threhold on vr
          t_threshold  = int/float, threhold on t
          plot = bool, if true then plot a map of the hotspots detection and their clustering
          remove_smaller = int, remove hotspots when they have less point than remove_smaller
                           Mean that if remove_smaller, remove nothing
    """
    pName = 'AHD'
    im('Compute AHD after Arnould et al 2020',pName,verbose)
    im('Compute the radial heat advection',pName,verbose)
    a = slice350kmt.v * slice350kmv.vr
    #ma1 = a <= 8000
    #ma2 = a >= 2021
    #gind = np.where(ma1*ma2)[0]
    ma = a >= 8253 # From Arnould et al., 2020: Corresponds to a value of 190 K.m.yr^{-1} (defined as the upper mantle threhold)
    #ma = a >= 5000
    gind = np.where(ma)[0]
    
    xhot = slice350kmt.x[gind]
    yhot = slice350kmt.y[gind]
    zhot = slice350kmt.z[gind]

    # define a metric: dist between 3 points
    threshold = 3 * np.sqrt((slice350kmt.x[0]-slice350kmt.x[1])**2+(slice350kmt.y[0]-slice350kmt.y[1])**2+(slice350kmt.z[0]-slice350kmt.z[1])**2)

    donefull    = []  # list of examined points
    clusterized = []  # list of hotspots segmentation (clustering)
    undonefull  = list(range(len(xhot))) # list of un-examined points
    # clustering
    im('Hotspots clustering on '+str(len(xhot))+' points',pName,verbose)
    while len(undonefull) != 0:
        cluster = [undonefull[0]]
        done    = []
        n = 0
        while len(done) < len(cluster):
            # search undone point in the current cluster
            undone = [c for c in cluster if c not in done]
            i = undone[0] # take the first undone point
            xref = xhot[i]
            yref = yhot[i]
            zref = zhot[i]
            dist = np.sqrt((xref-xhot)**2+(yref-yhot)**2+(zref-zhot)**2)
            cind = np.where(dist <= threshold)[0]
            if len(cind) != 0:
                for gid in cind:
                    if gid not in cluster:
                        cluster.append(gid)
            done.append(i)
            n += 1
        clusterized.append(cluster)
        donefull += done
        for i in range(len(done)):
            undonefull.remove(done[i])

    nc = len(clusterized)  # number of clusters
    im('Number of detected hotspots: '+str(nc),pName,verbose)

    im('Retrieve indicies and compute centroids',pName,verbose)
    im('Remove hotspots smaller than '+str(remove_smaller)+' points',pName,verbose)
    centroidid    = []
    clusterizedid = []
    # get the indices of cluster in the interpolatedslideData
    # and compute the index of centroid for all the clusters
    # (also in the indices of the input interpolatedslideData)
    for i in range(nc):
        seg = np.array(clusterized[i])
        if len(seg) > remove_smaller:
            clusterizedid.append(gind[seg])
            xseg = xhot[seg]
            yseg = yhot[seg]
            zseg = zhot[seg]
            # search the centroid
            nod = len(xseg)
            maxdist = np.zeros(nod)
            for j in range(nod):
                refx,refy,refz = xseg[j],yseg[j],zseg[j]
                dist = np.sqrt((xseg-refx)**2+(yseg-refy)**2+(zseg-refz)**2)
                maxdist[j] = np.amax(dist)
            # get centroid
            cid = np.where(maxdist == np.amin(maxdist))[0][0]
            centroidid.append(int(gind[seg[cid]]))
    centroidid = np.array(centroidid)
    nc = len(clusterizedid)  # number of clusters
    im('Number of cleaned hotspots: '+str(nc),pName,verbose)
    
        
    if plot:
        im('Map of autodetected hotspots',pName,verbose)
        import cartopy.crs as ccrs
        proj = ccrs.Robinson()
        lon = slice350kmt.phi.flatten()*180/np.pi
        #lat = -(90-slice350kmt.theta.flatten()*180/np.pi)
        lat = (90-slice350kmt.theta.flatten()*180/np.pi)
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
        ax.set_title('Map of autodetected hotspots')
        ax.set_global()
        x = []; y = []; v = []
        xc = []; yc = []
        for i in range(nc):
            seg = np.array(clusterizedid[i])
            x += list(lon[seg])
            y += list(lat[seg])
            v += list(lon[seg].copy()*0+i) # index of the cluster
            xc.append(lon[centroidid[i]])
            yc.append(lat[centroidid[i]])
        ax.scatter(lon,lat,c=a,transform=ccrs.PlateCarree())
        ax.scatter(x,y,c=v,cmap=plt.cm.viridis,s=5,alpha=1,transform=ccrs.PlateCarree(),label='hotspot points')
        ax.scatter(xc,yc,marker='D',c='orange',s=10,alpha=1,transform=ccrs.PlateCarree(),label='hotspot centroids')
        plt.legend()
        plt.show()
    
    im('Detection and clustring of hotspots done!',pName,verbose)
    return clusterizedid,centroidid


def mantleDrag(stagDataVelo,plot=True):
    """
    This function return the map of the mantle drag
    coefficient as defined by Coltice et al 2019
    <i> stagDataVelo must be a pypStag.stagData.StagYinYangGeometry
        which have the fieldType = 'Velocity'
    """
    if stagDataVelo.geometry != 'yy':
        raise InputGridGeometryError(stagDataVelo.geometry)
    if stagDataVelo.fieldType != 'Velocity':
        raise fieldTypeError('Velocity')
    # --- Define the sublithospheric depth of the comptutation
    dist = abs(stagDataVelo.depths - 240)
    ind  = np.where(dist == np.amin(dist))[0][0]
    NxNy = int(stagDataVelo.v.shape[0]/stagDataVelo.nz)
    Nz   = stagDataVelo.nz
    vsurf = np.empty((NxNy,3),dtype=np.float64)
    vsurf[:,0] = stagDataVelo.vx.reshape(NxNy,Nz)[:,-1]
    vsurf[:,1] = stagDataVelo.vy.reshape(NxNy,Nz)[:,-1]
    vsurf[:,2] = stagDataVelo.vz.reshape(NxNy,Nz)[:,-1]
    vd = np.empty((NxNy,3),dtype=np.float64)
    vd[:,0] = stagDataVelo.vx.reshape(NxNy,Nz)[:,ind]
    vd[:,1] = stagDataVelo.vy.reshape(NxNy,Nz)[:,ind]
    vd[:,2] = stagDataVelo.vz.reshape(NxNy,Nz)[:,ind]
    vsurfn = np.sqrt(vsurf[:,0]**2+vsurf[:,1]**2+vsurf[:,2]**2)
    vrms   = np.mean(vsurfn)
    dot    = vsurf[:,0]*vd[:,0]+vsurf[:,1]*vd[:,1]+vsurf[:,2]*vd[:,2]
    D = (vsurfn - dot/vsurfn)/vrms
    lon = stagDataVelo.phi.reshape(NxNy,Nz)[:,-1]*180/np.pi
    lat = (90-stagDataVelo.theta.reshape(NxNy,Nz)[:,-1]*180/np.pi)
    if plot:
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
        ax.set_title('Mantle drag coefficient')
        ax.set_global()
        cmap = plt.cm.get_cmap('RdBu', 8)
        cmap = ax.scatter(lon,lat,c=D,s=1,cmap=cmap,vmin=-3,vmax=+3,transform=ccrs.PlateCarree())
        plt.colorbar(cmap)
        plt.show()
    return lon,lat, D



def lithosphere_thickness(stagData,plot=True):
    """
    This function computes the lithosphere thickness from a
    pypStag.stagData  *TEMPERATURE*  in  *YY*  geometry.
    """
    if stagData.geometry != 'yy':
        raise InputGridGeometryError(stagData.geometry)
    if stagData.fieldType != 'Temperature':
        raise fieldTypeError('Temperature')
    from tqdm import tqdm
    Nz = stagData.nz
    x = stagData.depths
    dx = x[0:Nz-1]-x[1:Nz]
    NxNy = int(stagData.v.shape[0]/Nz)
    lithothickness = np.zeros(NxNy)
    Y = stagData.v.reshape(NxNy,Nz)
    for pID in tqdm(range(NxNy)):
        y = Y[pID,:]
        dy = y[0:Nz-1]-y[1:Nz]
        lithoID1 = np.where(dy/dx <= 0)[0]
        if len(lithoID1) > 0:
            lithoID1 = lithoID1[-1]+1
        else:
            lithoID1 = 0
        lithoID2 = np.where(dy/dx <= 0.0005)[0]
        if len(lithoID2) > 0:
            lithoID2 = lithoID2[-1]+1
        else:
            lithoID2 = 0
        if lithoID1 >= lithoID2:
            lithoID = lithoID1
            #determination = 'inflexion'
        else:
            lithoID = lithoID2
            #determination = 'slope'
        lithoID  = max(lithoID1,lithoID2)
        lithothickness[pID] = x[lithoID]
    if plot:
        import cartopy.crs as ccrs
        lon = stagData.phi.reshape(NxNy,Nz)[:,-1]*180/np.pi
        lat = (90-stagData.theta.reshape(NxNy,Nz)[:,-1]*180/np.pi)
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
        ax.set_title('Lithosphere thickness (in km)')
        ax.set_global()
        cmap = ax.scatter(lon,lat,c=lithothickness,s=1,transform=ccrs.PlateCarree())
        plt.colorbar(cmap)
        plt.show()
    return lithothickness
    
    



























