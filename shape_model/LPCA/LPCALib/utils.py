import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import shape_model.LPCA.LPCALib.kernels as lpca_kernels
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.sparse.csgraph as csgraph
import cv2
import matplotlib.colors as colors 
# import pyvista as pv
import SimpleITK as sitk
from numpy import matlib


def plot_contour(contour_data,args='.',xlim=None, ylim=None):
    contour_data=np.concatenate((contour_data,contour_data[0:2]))
    plt.plot(contour_data[0::2],contour_data[1::2],args)
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])
    plt.axis('equal')
    plt.axis('off')

def plot_2d_multi_object_contour(contour_data, obj_indicator, color_map, args='.', xlim=None, ylim=None):
    unique_objects=np.unique(obj_indicator)
    for i in range(0,len(unique_objects)):
#        contour_data=np.asarray(contour_data).flatten()
        current_contour=contour_data[obj_indicator==unique_objects[i]]        
        current_contour=np.concatenate((current_contour,current_contour[0:2]))
        plt.plot(current_contour[0::2],current_contour[1::2],color_map[i]+args)
        if xlim is not None:
            plt.xlim(xlim[0],xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0],ylim[1])
        plt.axis('equal')
        plt.axis('off')

def plot_2d_multi_object_contour_matrix(contour_data, obj_indicator, color_map, args='.',xlim=None, ylim=None):
    if len(contour_data.shape)>1:
        num_cols=min(contour_data.shape[1],3)
        num_rows=np.ceil(contour_data.shape[1]/3).astype(int)
        num_plots=contour_data.shape[1]
        data=contour_data
    else:
        num_cols=1
        num_rows=1
        num_plots=1
        data=np.asmatrix(contour_data).T
    
    for idx in range(1,num_plots+1):
        plt.subplot(num_rows,num_cols,idx)
        plot_2d_multi_object_contour(data[:,idx-1],obj_indicator,color_map,args,xlim,ylim)
        plt.gca().invert_yaxis()

def plot_hand_matrix(contour_data):
    if len(contour_data.shape)>1:
        num_cols=min(contour_data.shape[1],3)
        num_rows=np.ceil(contour_data.shape[1]/3).astype(int)
        num_plots=contour_data.shape[1]
        data=contour_data
    else:
        num_cols=1
        num_rows=1
        num_plots=1
        data=np.asmatrix(contour_data).T
    
    for idx in range(1,num_plots+1):
        plt.subplot(num_rows,num_cols,idx)
        plot_contour(data[:,idx-1],'k-')
        plt.gca().invert_yaxis()

def plot_2d_contour_subpace_model(model, dims, alpha,xlim=None, ylim=None):
    for i in range(0,dims):
        #-alpha
        plt.subplot(3,dims,i+1)
        plot_contour(model.translation_vector-np.asmatrix(model.basis[:,i]*np.sqrt(model.eigenvalues[i,0])*alpha),'k-',xlim, ylim)
        plt.gca().invert_yaxis()
        #mean
        plt.subplot(3,dims,dims+i+1)
        plot_contour(model.translation_vector,'k-',xlim, ylim)
        plt.gca().invert_yaxis()
        #+alpha
        plt.subplot(3,dims,2*dims+i+1)
        plot_contour(model.translation_vector+np.asmatrix(model.basis[:,i]*np.sqrt(model.eigenvalues[i,0])*alpha),'k-',xlim, ylim)
        plt.gca().invert_yaxis()

def plot_2d_multi_object_contour_subpace_model(model, dims, alpha, obj_indicator, color_map, args='.',xlim=None, ylim=None):
    for i in range(0,dims):
        #-alpha
        plt.subplot(3,dims,i+1)
        plot_2d_multi_object_contour(model.translation_vector-(model.basis[:,i]*np.sqrt(model.eigenvalues[i,0])*alpha),obj_indicator, color_map, args,xlim,ylim)
        plt.gca().invert_yaxis()
        #mean
        plt.subplot(3,dims,dims+i+1)
        plot_2d_multi_object_contour(model.translation_vector,obj_indicator, color_map, args,xlim,ylim)
        plt.gca().invert_yaxis()
        #+alpha
        plt.subplot(3,dims,2*dims+i+1)
        plot_2d_multi_object_contour(model.translation_vector+(model.basis[:,i]*np.sqrt(model.eigenvalues[i,0])*alpha),obj_indicator, color_map, args,xlim,ylim)
        plt.gca().invert_yaxis()

def plot_3d_masked_displ_field(field, slice_indicator, mask_flattended, img, max_mag=None, axis_aspect_ratio=1):

    #convert masked field to full field
    field_full=np.zeros((mask_flattended.shape[0],1))
    field_full[np.where(mask_flattended==1)]=field
    field_full=field_full.reshape((img.shape[0],img.shape[1],img.shape[2],3),order='F')

    #extract selected slice
    if slice_indicator[0] > 0:    
        field_slice=field_full[slice_indicator[0],:,:,:][...,(1,2)]
        img_slice=img[slice_indicator[0],:,:]
    elif slice_indicator[1] > 0:
        field_slice=field_full[:,slice_indicator[1],:,:][...,(0,2)]
        img_slice=img[:,slice_indicator[1],:]
    elif slice_indicator[2] > 0:
        field_slice=field_full[:,:,slice_indicator[2],:][...,(0,1)]
        img_slice=img[:,:,slice_indicator[2]]

    #convert field to HSV representation
    hsv=np.zeros((field_slice.shape[0],field_slice.shape[1],3))
    mag,ang=cv2.cartToPolar(np.float32(field_slice)[...,0],np.float32(field_slice)[...,1])
    ang=ang/(2*np.pi)
    if max_mag is None:
        mag[np.nonzero(mag)]=mag[np.nonzero(mag)]/np.max(mag)
    else:
        mag[np.nonzero(mag)]=mag[np.nonzero(mag)]/max_mag
        mag[np.where(mag>1)]=1

    hsv[...,1]=1
    hsv[...,0]=ang
    hsv[...,2]=mag
    rgb=colors.hsv_to_rgb(hsv)
    rgba=np.concatenate((rgb,np.zeros((rgb.shape[0],rgb.shape[1],1))),axis=2)
    rgba[...,3]=mag

    plt.imshow(img_slice,cmap=plt.get_cmap('gray'),aspect=axis_aspect_ratio)
    plt.imshow(rgba,aspect=axis_aspect_ratio)
    plt.axis('off')        

def plot_3d_masked_displ_field_subspace_model(model, dims, alpha, slice_indicator, mask_flattended, img, max_mag=None, axis_aspect_ratio=1):
    for i in range(0,dims):
        #-alpha
        plt.subplot(3,dims,i+1)
        plot_3d_masked_displ_field(model.translation_vector-np.asmatrix(model.basis[:,i]*np.sqrt(model.eigenvalues[i,0])*alpha),slice_indicator, mask_flattended, img, max_mag, axis_aspect_ratio)
        #mean
        plt.subplot(3,dims,dims+i+1)
        plot_3d_masked_displ_field(model.translation_vector,slice_indicator, mask_flattended, img, max_mag, axis_aspect_ratio)
        #+alpha
        plt.subplot(3,dims,2*dims+i+1)
        plot_3d_masked_displ_field(model.translation_vector+np.asmatrix(model.basis[:,i]*np.sqrt(model.eigenvalues[i,0])*alpha),slice_indicator, mask_flattended, img, max_mag, axis_aspect_ratio)


# def plot_3d_mesh(points, base_mesh, plotter=None):
# #    surf=pv.PolyData(np.hstack([points[0::3], points[1::3],points[2::3]]),np.int64(np.reshape(np.hstack([(3*np.ones(topology.shape[0]))[:,np.newaxis],topology]),(-1,1))))
#
#     vtk_points=pv.vtk_points(pv.np.hstack([points[0::3], points[1::3],points[2::3]]))
#     base_mesh.SetPoints(vtk_points)
#
#     if plotter is None:
#         plotter=pv.BackgroundPlotter()
#         plotter.add_mesh(base_mesh,smooth_shading=True)
#         plotter.view_xy()
#         plotter.set_background('w')
#         plotter.show()
#     else:
#         plotter.add_mesh(base_mesh,smooth_shading=True)

# def plot_3d_mesh_subspace_model(model, dims, alpha):
#     plotter=pv.BackgroundPlotter(shape=(3,dims),border_color=[1,1,1],border_width=0)
#     for i in range(0,dims):
#         #-alpha
#         plotter.subplot(0,i)
#         plot_3d_mesh(model.translation_vector-np.asmatrix(model.basis[:,i]*np.sqrt(model.eigenvalues[i,0])*alpha),model.base_mesh,plotter)
#         #mean
#         plotter.subplot(1,i)
#         plot_3d_mesh(model.translation_vector,model.base_mesh,plotter)
#         #+alpha
#         plotter.subplot(2,i)
#         plot_3d_mesh(model.translation_vector+np.asmatrix(model.basis[:,i]*np.sqrt(model.eigenvalues[i,0])*alpha),model.base_mesh,plotter)
#
#     plotter.link_views()
#     plotter.view_xy()
#     plotter.set_background([1,1,1])
#     plotter.hide_axes()

def mean_error_2d_contour(gt,pred):
    return np.mean(np.sqrt(np.square(gt[0::2,:]-pred[0::2,:])+np.square(gt[1::2,:]-pred[1::2,:])),axis=0)

def mean_error_3d_shape(gt,pred):
    return np.mean(np.sqrt(np.square(gt[0::3,:]-pred[0::3,:])+np.square(gt[1::3,:]-pred[1::3,:])+np.square(gt[2::3,:]-pred[2::3,:])),axis=0)

def mean_error_3d_shape_new(gt,pred):
    temp=(np.array(gt)[:,:,None]-np.array(pred)[:,None,:])**2
    return np.mean(np.sqrt(temp[0::3,...]+temp[1::3,...]+temp[2::3,...]),axis=0)

def mean_error_3d_field(gt,pred):
    size=np.long(gt.shape[0]/3)
    diff=np.square(gt-pred)
    return np.mean(np.sqrt(diff[0:size,:]+diff[size:size*2,:]+diff[size*2:,:]),axis=0)
#    return np.mean(np.sqrt(np.square(gt[0:size]-pred[0:size])+np.square(gt[size:size*2]-pred[size:size*2])+np.square(gt[size*2:]-pred[size*2:])),axis=0)

#gt is expected to be a displ field; pred will be converted from velo to displ prior to comparison!!!!
def mean_error_3d_field_svf_masked(gt,pred,mask=None):
    size=np.long(gt.shape[0]/3)
    diff=np.square(gt-pred)
    return np.mean(np.sqrt(diff[0:size,:]+diff[size:size*2,:]+diff[size*2:,:]),axis=0)
#    return np.mean(np.sqrt(np.square(gt[0:size]-pred[0:size])+np.square(gt[size:size*2]-pred[size:size*2])+np.square(gt[size*2:]-pred[size*2:])),axis=0)

def compute_simple_contour_distance_matrix(num_points):
    dist_matrix=np.zeros((num_points,num_points))

    pos_dist=np.array(range(0,num_points))
    neg_dist=np.array(range(num_points,0,-1))
    for i in range(0,num_points):
        dist_matrix[i,:]=np.min([np.abs(pos_dist),neg_dist],axis=0)
        pos_dist=pos_dist-1
        neg_dist=neg_dist+1

    temp=np.kron(dist_matrix,np.ones((2,2)))
    return np.min([temp.T,temp],axis=0)

def compute_euclidean_2d_point_distance_matrix(base_contour):
    dist_matrix=np.sqrt((base_contour[0::2]-np.matlib.repmat(base_contour[0::2],1,base_contour[0::2].shape[0]).T)**2+(base_contour[1::2]-np.matlib.repmat(base_contour[1::2],1,base_contour[1::2].shape[0]).T)**2)    

    return np.kron(dist_matrix,np.ones((2,2)))

#highly inefficient (=very slow) implementation!!!!
def compute_bspline_geodesic_2d_point_distance_matrix(base_contour):
    tck_pos,u_pos=interpolate.splprep([np.concatenate([base_contour[0::2].flatten(),base_contour[0::2][0]]), np.concatenate([base_contour[1::2].flatten(),base_contour[1::2][0]])], s=0, per=True)

    deriv_norm_pos=lambda x: np.linalg.norm(interpolate.splev(x,tck_pos,der=1))

    max_dist=integrate.quad(deriv_norm_pos,0,1)[0]
    
    geo_dist_matrix=np.zeros([np.int(base_contour.shape[0]/2),np.int(base_contour.shape[0]/2)])

    for i in range(0,geo_dist_matrix.shape[0]):
        for j in range(0,geo_dist_matrix.shape[0]):
            curr_result=np.abs(integrate.quad(deriv_norm_pos,u_pos[i],u_pos[j])[0])
            geo_dist_matrix[i,j]=np.minimum(curr_result,max_dist-curr_result)

    return np.kron(geo_dist_matrix,np.ones((2,2)))
    

def compute_pseudo_geodesic_2d_point_distance_matrix(base_contour):
    euclidean_dist_matrix=np.sqrt((base_contour[0::2]
                                   -np.matlib.repmat(base_contour[0::2],1,base_contour[0::2].shape[0]).T)**2
                                  +(base_contour[1::2]
                                    -np.matlib.repmat(base_contour[1::2],1,base_contour[1::2].shape[0]).T)**2)
    
    succ=np.diag(euclidean_dist_matrix,1)
    succ=np.concatenate((succ,[euclidean_dist_matrix[0,-1]])) # complete +1 'diagonal'

    geo_dist_matrix=np.zeros(euclidean_dist_matrix.shape)

    for i in range(0,euclidean_dist_matrix.shape[1]):
        curr_succ=np.roll(succ,-i)
        curr_succ_a=np.concatenate(([0],np.cumsum(curr_succ[0:-1])))
        curr_succ_b=np.concatenate(([0],np.flip(np.cumsum(np.flip(curr_succ[1:],axis=0)),axis=0)))
        curr_dist=np.roll(np.minimum(curr_succ_a,curr_succ_b),+i)
        geo_dist_matrix[:,i]=curr_dist

    return np.kron(geo_dist_matrix,np.ones((2,2)))

def compute_multi_object_pseudo_euclidean_geodesic_2d_point_distance_matrix(base_contour,obj_indicator):
    euclidean_dist_matrix=np.kron(np.sqrt((base_contour[0::2]
                                           -np.matlib.repmat(base_contour[0::2],1,base_contour[0::2].shape[0]).T)**2
                                          +(base_contour[1::2]
                                            -np.matlib.repmat(base_contour[1::2],1,base_contour[1::2].shape[0]).T)**2),
                                  np.ones((2,2)))

    geo_dist_matrix=np.ones(euclidean_dist_matrix.shape)*np.finfo(euclidean_dist_matrix.dtype).max
    max_geo_dist=0

    unique_objects=np.unique(obj_indicator)
    for i in range(0,len(unique_objects)):
        curr_object=base_contour[obj_indicator==unique_objects[i]]
        
        curr_geo_dist=compute_pseudo_geodesic_2d_point_distance_matrix(curr_object)
        max_geo_dist=np.maximum(np.max(curr_geo_dist),max_geo_dist)

        idcs=np.where(obj_indicator==unique_objects[i])
        print(curr_geo_dist.shape)
        print(idcs)
        idx_min=np.min(idcs)
        idx_max=np.max(idcs)
        geo_dist_matrix[idx_min:idx_max+1,idx_min:idx_max+1]=curr_geo_dist

    combined_dist=np.minimum(geo_dist_matrix,euclidean_dist_matrix+max_geo_dist+1)
#    combined_dist=np.minimum(geo_dist_matrix,max_geo_dist)

    return combined_dist,max_geo_dist

def compute_multi_object_pseudo_euclidean_geodesic_shortest_path_2d_point_distance_matrix(base_contour,obj_indicator,eta,kappa):
    euclidean_dist_matrix=np.kron(np.sqrt((base_contour[0::2]
                                           -np.matlib.repmat(base_contour[0::2],1,base_contour[0::2].shape[0]).T)**2
                                          +(base_contour[1::2]
                                            -np.matlib.repmat(base_contour[1::2],1,base_contour[1::2].shape[0]).T)**2),
                                  np.ones((2,2)))


    geo_dist_matrix=np.ones(euclidean_dist_matrix.shape)*np.finfo(euclidean_dist_matrix.dtype).max

    unique_objects=np.unique(obj_indicator)
    for i in range(0,len(unique_objects)):
        curr_object=base_contour[obj_indicator==unique_objects[i]]
        
        curr_geo_dist=compute_pseudo_geodesic_2d_point_distance_matrix(curr_object)

        idcs=np.where(obj_indicator==unique_objects[i])
        idx_min=np.min(idcs)
        idx_max=np.max(idcs)
        geo_dist_matrix[idx_min:idx_max+1,idx_min:idx_max+1]=curr_geo_dist

    combined_dist=np.minimum(geo_dist_matrix,euclidean_dist_matrix*eta+kappa)
    dist_matrix=csgraph.shortest_path(combined_dist[0::2,0::2],directed=False)
    
    return np.kron(dist_matrix,np.ones((2,2)))


def compute_pseudo_geodesic_3d_point_distance_matrix(base_contour):
    euclidean_dist_matrix = np.sqrt((base_contour[0::3]
                                     - np.matlib.repmat(base_contour[0::3], 1, base_contour[0::3].shape[0]).T) ** 2
                                    + (base_contour[1::3]
                                       - np.matlib.repmat(base_contour[1::3], 1, base_contour[1::3].shape[0]).T) ** 2
                                    + (base_contour[2::3]
                                       - np.matlib.repmat(base_contour[2::3], 1, base_contour[2::3].shape[0]).T) ** 2)

    succ = np.diag(euclidean_dist_matrix, 1)
    succ = np.concatenate((succ, [euclidean_dist_matrix[0, -1]]))  # complete +1 'diagonal'

    geo_dist_matrix = np.zeros(euclidean_dist_matrix.shape)

    for i in range(0, euclidean_dist_matrix.shape[1]):
        curr_succ = np.roll(succ, -i)
        curr_succ_a = np.concatenate(([0], np.cumsum(curr_succ[0:-1])))
        curr_succ_b = np.concatenate(([0], np.flip(np.cumsum(np.flip(curr_succ[1:], axis=0)), axis=0)))
        curr_dist = np.roll(np.minimum(curr_succ_a, curr_succ_b), +i)
        geo_dist_matrix[:, i] = curr_dist

    return np.kron(geo_dist_matrix, np.ones((3, 3)))


def compute_multi_object_pseudo_euclidean_geodesic_shortest_path_3d_point_distance_matrix(base_contour, obj_indicator,
                                                                                          eta, kappa):
    euclidean_dist_matrix = np.kron(np.sqrt((base_contour[0::3]
                                             - np.matlib.repmat(base_contour[0::3], 1,
                                                                base_contour[0::3].shape[0]).T) ** 2
                                            + (base_contour[1::3]
                                               - np.matlib.repmat(base_contour[1::3], 1,
                                                                  base_contour[1::3].shape[0]).T) ** 2
                                            + (base_contour[2::3]
                                               - np.matlib.repmat(base_contour[2::3], 1,
                                                                  base_contour[2::3].shape[0]).T) ** 2),
                                    np.ones((3, 3)))

    geo_dist_matrix = np.ones(euclidean_dist_matrix.shape) * np.finfo(euclidean_dist_matrix.dtype).max

    unique_objects = np.unique(obj_indicator)
    for i in range(0, len(unique_objects)):
        curr_object = base_contour[obj_indicator == unique_objects[i]]

        curr_geo_dist = compute_pseudo_geodesic_3d_point_distance_matrix(curr_object)

        idcs = np.where(obj_indicator == unique_objects[i])


        idx_min = np.min(idcs)
        idx_max = np.max(idcs)
        geo_dist_matrix[idx_min:idx_max + 1, idx_min:idx_max + 1] = curr_geo_dist

    combined_dist = np.minimum(geo_dist_matrix, euclidean_dist_matrix * eta + kappa)
    dist_matrix = csgraph.shortest_path(combined_dist[0::3, 0::3], directed=False)

    return np.kron(dist_matrix, np.ones((3, 3)))


def corrcov(cov_matrix):
    sigma=np.sqrt(np.diag(cov_matrix))
    corr_matrix=cov_matrix/(np.asmatrix(sigma).T@np.asmatrix(sigma))
    corr_matrix=(corr_matrix.T+corr_matrix)/2
    np.fill_diagonal(corr_matrix,1)
    return corr_matrix,sigma

def covcorr(corr_matrix,sigmas):
    return np.diag(sigmas)@corr_matrix@np.diag(sigmas)

def higham_closest_corr_matrix(corr_matrix,max_iterations=1000,tol=1e-5):    
    X=Y=corr_matrix
    correction_matrix=np.zeros(corr_matrix.shape)
    diffX=tol+1
    diffY=tol+1
    diffXY=tol+1

    i=1
    while np.max([diffX,diffY,diffXY]) > tol and i <= max_iterations:
#        print('iter: '+str(i))
        Xold=np.copy(X)
        R=Y-correction_matrix
        
        #projection onto space of psd matrices
        eig_vals,eig_vecs=np.linalg.eig(R)
        X=eig_vecs*np.diag(np.max([eig_vals,np.zeros(eig_vals.shape)],axis=0))*eig_vecs.T
        X=(X.T+X)/2
        correction_matrix=X-R

        Yold=np.copy(Y)
        #projection onto space of matrices with unit diagonal
        Y=np.copy(X)
        np.fill_diagonal(Y,1)

        #compute differences
        diffX=np.linalg.norm(X-Xold)/np.linalg.norm(X)
        diffY=np.linalg.norm(Y-Yold)/np.linalg.norm(Y)
        diffXY=np.linalg.norm(Y-X)/np.linalg.norm(Y)        
#        print('X: '+str(diffX)+' Y: '+str(diffY)+' XY: '+str(diffXY))
                                    
        i=i+1

    #Higham usually returns X but here we use Y to ensure that the diagonal elements are 1 --> no correction performed after last step!
    return Y
#    return X

def merge_subspace_models_closest_rotation(modelA, modelB,decorrelation=False,decorrelation_mode='full'):
    """
    Merges two subspaces of different dimensions (rank(A)<=rank(B)) by 
    finding the closest (same dimension) subspace to B that fully contains A.

    Based on:
        KE YE AND LEK-HENG LIM: DISTANCE BETWEEN SUBSPACES OF
                    DIFFERENT DIMENSIONS, http://arxiv.org/abs/1407.0900v1
    """
    if modelA.basis.shape[1] >= modelB.basis.shape[1]:
        return modelA.translation_vector,modelA.basis,modelA.eigenvalues

#    print('size A: '+str(modelA.basis.shape[1])+' size B: '+str(modelB.basis.shape[1]))
#    print('model A evs: '+str(modelA.eigenvalues)+' sum: '+str(np.sum(modelA.eigenvalues)))
#    calc_weights=modelA.basis.T@(training_data-modelA.translation_vector)
#    calc_variance=(calc_weights@calc_weights.T)/(training_data.shape[1]-1)
#    print('calc variance A: '+str(np.diag(calc_variance)))

#    print('model B evs: '+str(modelB.eigenvalues)+' sum: '+str(np.sum(modelB.eigenvalues)))
#    calc_weights=modelB.basis.T@(training_data-modelB.translation_vector)
#    calc_variance=(calc_weights@calc_weights.T)/(training_data.shape[1]-1)
#    print('calc variance B: '+str(np.diag(calc_variance)))

    U,S,Vt=np.linalg.svd(modelA.basis.T@modelB.basis)
    V=Vt.T

#    print('A: '+str(modelA.basis.shape)+' B: '+str(modelB.basis.shape))
#    print('U: '+str(U.shape)+' V: '+str(V.shape))

    rotA=modelA.basis@U
    rotB=modelB.basis@V
    new_basis=np.zeros((modelA.basis.shape[0],modelB.basis.shape[1]))
    new_basis[:,0:modelA.basis.shape[1]]=rotA
    new_basis[:,modelA.basis.shape[1]:]=rotB[:,modelA.basis.shape[1]:]

    rotA_evs=U.T@np.diagflat(modelA.eigenvalues)@U
    rotB_evs=Vt@np.diagflat(modelB.eigenvalues)@V

#    print('sum A: '+str(np.sum(modelA.eigenvalues))+' sum B: '+str(np.sum(modelB.eigenvalues)))
#    print('rotA_evs: '+str(rotB_evs)+' sum: '+str(np.sum(rotB_evs)))
#    calc_evs=rotB.T@(modelB.basis@np.sqrt(np.diag(np.array(modelB.eigenvalues).flatten())))
#    calc_evs=calc_evs@calc_evs.T
#    print('calc_evs: '+str(calc_evs)+' sum: '+str(np.sum(calc_evs)))

#    calc_evs_nb=new_basis.T@(modelB.basis@np.sqrt(np.diag(np.array(modelB.eigenvalues).flatten())))
#    calc_evs_nb=calc_evs_nb@calc_evs_nb.T
#    print('calc_evs_nb: '+str(calc_evs_nb)+' sum: '+str(np.sum(calc_evs_nb)))

    new_evs_old=np.zeros((modelB.basis.shape[1],modelB.basis.shape[1]))
    new_evs_old[0:rotA_evs.shape[0],0:rotA_evs.shape[1]]=rotA_evs
    new_evs_old[rotA_evs.shape[0]:,rotA_evs.shape[1]:]=rotB_evs[rotA_evs.shape[0]:,rotA_evs.shape[1]:]

    new_evs=np.zeros((modelB.basis.shape[1],modelB.basis.shape[1]))
    new_evs[0:rotA_evs.shape[0],0:rotA_evs.shape[1]]=rotA_evs*0.5
    b_evs_new_basis=new_basis.T@rotB@rotB_evs@rotB.T@new_basis
    new_evs=new_evs+(b_evs_new_basis*0.5)

#    print('new_evs: '+str(new_evs))
#    print('new_evs_old: '+str(new_evs_old))

    new_evs=new_evs_old


    if decorrelation:
        if decorrelation_mode == 'full':
            U,S,Vt=np.linalg.svd(new_basis@new_evs@new_basis.T)
            new_basis=U[:,0:new_basis.shape[1]]
#            print('sum before correction: '+str(np.sum(new_evs)))
            new_evs=S*(np.sum(modelA.eigenvalues)/np.sum(S))
            new_evs=new_evs[0:new_basis.shape[1]]
        elif decorrelation_mode == 'kernel':
            L=np.linalg.cholesky(new_evs)
            new_basis,new_evs=eig_fast_spsd_kernel(new_basis@L,[(lpca_kernels.CovKernel(1),None,'data',None,1)],new_evs.shape[0],sampling_factor=2)
            new_evs=new_evs*(np.sum(modelB.eigenvalues)/np.sum(new_evs))
            new_evs=new_evs[0:new_basis.shape[1]]                        
    else:
        new_evs=modelB.eigenvalues

#    print('new_evs: '+str(new_evs))

#    print('rotA.T@rot_B: '+str(rotA.T@rotB[:,rotA.shape[1]:]))
#    print('rotA.T@rot_B: '+str(rotA.T@rotB[:,0:rotA.shape[1]]))
#    print('rotA.T@rot_B: '+str(np.diag(rotA.T@rotB)))
    print("Merging2")
    return modelA.translation_vector,new_basis,new_evs

def merge_subspace_models_closest_rotation_decorr(modelA, modelB):
    return merge_subspace_models_closest_rotation(modelA, modelB,True,'full')

def merge_subspace_models_closest_rotation_decorr_kernel(modelA, modelB):
    return merge_subspace_models_closest_rotation(modelA, modelB,True,'kernel')

def eig_fast_spsd_kernel(data,kernel_list,rank,sampling_factor=None):
    """
    Computes the eigendecomposition of a low-rank kernel matrix implicitly derived
    from the data vectors (column vectors!) and the list of kernels (order matters!; left to right!)
    using a randomized sampling approach

    Method is based on 

    Towards More Efficient SPSD Matrix Approximation and CUR Matrix Decomposition by Wang et al.
    http://www.jmlr.org/papers/volume17/15-190/15-190.pdf

    Input:

    kernels --> list of kernels to be applied where each entry consists of 5 elements (kernel, concat_op, data_string, dist_func, weight)
        kernel --> instance of a kernel derived from KernelBase
        concat_op --> any operator that can be applied to numpy objects (np.add, np.mult,...)
        data_string --> indicates whether this kernel operates on the 'data' or the coord distances 'dist'
        dist_func --> distance function derived from DistanceBase. Only used when data_string='dist'
        weight --> multiplicative weight applied to the kernel result before concatenation; i.e. enables weighted linear combination of different kernels (concat_op = np.add)
    """
    
    n = data.shape[0]
    m = data.shape[1]
    if sampling_factor is None:
        sampling_factor=np.int32(np.ceil(np.sqrt(rank*n))) 
#    print('sampling_factor: '+str(sampling_factor))
#    sampling_factor=1
    num_samples=rank*sampling_factor


    #first sketch --> nystroem approximation
    uni_sampling = np.sort(np.random.choice(n,rank,replace=False))# uniform sampling
    if kernel_list[0][2] == 'data':
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(data,data[uni_sampling,:])
    else:
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(kernel_list[0][3].dist(np.arange(0,data.shape[0]),uni_sampling))
    if len(kernel_list) > 1:
        for i in range(1,len(kernel_list)):
            if kernel_list[i][2] == 'data':
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(data,data[uni_sampling,:]))
            else:
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(kernel_list[i][3].dist(np.arange(0,data.shape[0]),uni_sampling)))

    Q,_temp,=np.linalg.qr(sketch)
#    print('sketch1: '+str(sketch.shape[0])+' x '+str(sketch.shape[1]))
    #second sketch
    sampling_prob=np.sum(np.square(Q),axis=1)
    sampling_prob=sampling_prob/np.sum(sampling_prob)
    
    lev_sampling=np.sort(np.random.choice(n,num_samples,replace=True,p=np.ravel(sampling_prob)))
    lev_sampling=np.unique(np.concatenate((lev_sampling,uni_sampling)))
    QInv=np.linalg.pinv(Q[lev_sampling,:])
#    print('kernel_list: '+str(kernel_list))
    if kernel_list[0][2] == 'data':
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(data[lev_sampling,:],data[lev_sampling,:])
    else:
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(kernel_list[0][3].dist(lev_sampling,lev_sampling))

    if len(kernel_list) > 1:
        for i in range(1,len(kernel_list)):
            #print('kernel_list['+str(i)+']: '+str(kernel_list[i]))
            if kernel_list[i][2] == 'data':
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(data[lev_sampling,:],data[lev_sampling,:]))    
            else:
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(kernel_list[i][3].dist(lev_sampling,lev_sampling)))

#    print('sketch2: '+str(sketch.shape[0])+' x '+str(sketch.shape[1]))

    U=QInv@sketch@QInv.T

#old version used in PhD thesis for computation of eigenvectors based on CUR decomposition
#    QInv=None

#    UC,SC,VCt=np.linalg.svd(Q,full_matrices=False)
#    Q=None

#    Z=(np.diag(SC)@VCt)@U@(np.diag(SC)@VCt).T
#    eigenvalues,VZ=np.linalg.eig(0.5*(Z.T+Z))
#    eigenvalues=np.real(eigenvalues)
#    VZ=np.real(VZ)
#    eigenvectors=UC@VZ

    UC,SC,_=np.linalg.svd(U,full_matrices=False)

    idx=np.argsort(-SC)
    eigenvectors=(Q@UC)[:,idx]
    eigenvalues=SC[idx]

    return eigenvectors,eigenvalues

def eig_nystrom_kernel(data,kernel_list,rank,sampling_factor=None):
    """
    Computes the eigendecomposition of a low-rank kernel matrix implicitly derived
    from the data vectors (column vectors!) and the list of kernels (order matters!; left to right!)
    using the Nystrom method

    Implementation follows MATLAB code from 

    https://arxiv.org/pdf/1505.07570.pdf

    Input:

    kernels --> list of kernels to be applied where each entry consists of 5 elements (kernel, concat_op, data_string, dist_func, weight)
        kernel --> instance of a kernel derived from KernelBase
        concat_op --> any operator that can be applied to numpy objects (np.add, np.mult,...)
        data_string --> indicates whether this kernel operates on the 'data' or the coord distances 'dist'
        dist_func --> distance function derived from DistanceBase. Only used when data_string='dist'
        weight --> multiplicative weight applied to the kernel result before concatenation; i.e. enables weighted linear combination of different kernels (concat_op = np.add)
    """
    
    n = data.shape[0]
    m = data.shape[1]
    if sampling_factor is None:
        sampling_factor=.8
#    print('sampling_factor: '+str(sampling_factor))
#    sampling_factor=1
    num_samples=np.int32(np.ceil(rank*sampling_factor))


    #first sketch --> nystroem approximation
    uni_sampling = np.sort(np.random.choice(n,rank,replace=False))# uniform sampling
    if kernel_list[0][2] == 'data':
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(data,data[uni_sampling,:])
    else:
        sketch = kernel_list[0][4]*kernel_list[0][0].apply(kernel_list[0][3].dist(np.arange(0,data.shape[0]),uni_sampling))
    if len(kernel_list) > 1:
        for i in range(1,len(kernel_list)):
            if kernel_list[i][2] == 'data':
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(data,data[uni_sampling,:]))
            else:
                sketch=kernel_list[i][1](sketch,kernel_list[i][4]*kernel_list[i][0].apply(kernel_list[i][3].dist(np.arange(0,data.shape[0]),uni_sampling)))

    W=sketch[uni_sampling,:]
    UW,SW,_=np.linalg.svd(W)

    SW=1/np.sqrt(SW[0:num_samples])
    UW=np.multiply(UW[:,0:num_samples],SW.T)
    L=sketch@UW

    UC,SC,_=np.linalg.svd(L,full_matrices=False)

    idx=np.argsort(-SC)
    eigenvectors=UC[:,idx]
    eigenvalues=SC[idx]**2

    return eigenvectors,eigenvalues


def eig_kernel(data,kernel_list,rank):

    if kernel_list[0][2] == 'data':
        kernel_matrix = kernel_list[0][4]*kernel_list[0][0].apply(data,data)
    else:
        kernel_matrix = kernel_list[0][4]*kernel_list[0][0].apply(kernel_list[0][3].dist(np.arange(0,data.shape[0]),np.arange(0,data.shape[0])))

    if len(kernel_list) > 1:
        for i in range(1,len(kernel_list)):
            if kernel_list[i][2] == 'data':
                kernel_matrix=kernel_list[i][1](kernel_matrix,kernel_list[i][4]*kernel_list[i][0].apply(data,data))    
            else:
                kernel_matrix=kernel_list[i][1](kernel_matrix,kernel_list[i][4]*kernel_list[i][0].apply(kernel_list[i][3].dist(np.arange(0,data.shape[0]),np.arange(0,data.shape[0]))))

    eig_vals,eig_vecs=sp.linalg.eigh(kernel_matrix,eigvals=(kernel_matrix.shape[0]-rank,kernel_matrix.shape[0]-1))

    return eig_vecs,eig_vals

def kernel_based_matrix_embedding(matrix,kernel=None):
    if kernel is None:
        kernel=lpca_kernels.GaussianKernel(5)

    print('matrix: '+str(matrix.shape))
    
    return kernel.apply(matrix,matrix)

def masked_svf_scaling_and_squaring(np_field,mask_flattened,ref_velo_field_itk, accuracy=4):

    ref_velo_field_np=sitk.GetArrayFromImage(ref_velo_field_itk)
    ref_velo_field_flat=np.ravel(ref_velo_field_np,order='F')*0
    ref_velo_field_flat[np.where(mask_flattened==1)]=np.squeeze(np.array(np_field[:]))
    ref_velo_field_np=ref_velo_field_flat.reshape(ref_velo_field_np.shape,order='F')        
        
    #scale field according to the desired integration accuracy/steps (2^steps)
    ref_velo_field_np=ref_velo_field_np/(2**accuracy)
    velo=sitk.GetImageFromArray(ref_velo_field_np,isVector=True)
    velo.CopyInformation(ref_velo_field_itk)
    
    #prepare warper for squaring step (velo_n+1 = velo_n \circ velo_n)
    warper=sitk.WarpImageFilter()
    warper.SetInterpolator(sitk.sitkLinear)
    warper.SetOutputParameteresFromImage(ref_velo_field_itk)
    
    #squaring step
    for i in range(accuracy):
        temp=warper.Execute(velo,velo)
        velo=velo+temp

    velo_np=sitk.GetArrayFromImage(velo)
    velo_masked=np.ravel(velo_np,order='F')[np.where(mask_flattened==1)]
        
    return velo_masked,velo

def svf_scaling_and_squaring(np_field,ref_velo_field_itk, accuracy=4):
        
    #scale field according to the desired integration accuracy/steps (2^steps)
    ref_velo_field_np=np_field/(2**accuracy)
    velo=sitk.GetImageFromArray(ref_velo_field_np,isVector=True)
    velo.CopyInformation(ref_velo_field_itk)
    
    #prepare warper for squaring step (velo_n+1 = velo_n \circ velo_n)
    warper=sitk.WarpImageFilter()
    warper.SetInterpolator(sitk.sitkLinear)
    warper.SetOutputParameteresFromImage(ref_velo_field_itk)
    
    #squaring step
    for i in range(accuracy):
        temp=warper.Execute(velo,velo)
        velo=velo+temp

    velo_np=sitk.GetArrayFromImage(velo)
        
    return velo_np


    
    

    

    

    

    
    

    

    
    
        
        

    
