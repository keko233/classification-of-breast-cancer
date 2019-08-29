from scipy.spatial import Delaunay
import cv2
import numpy as np



def get_triangle_matrix(matrix_0,triangle_maxlen,discrete_distance):

	matrix=np.copy(matrix_0)
	
	triangle_matrix=np.full((matrix.shape[0],matrix.shape[1]),0,dtype=np.uint8)
	

	
	point_filter=np.full((matrix.shape[0],matrix.shape[1]),0,dtype=np.int)
	point_filter[::discrete_distance,::discrete_distance]=1
	matrix=point_filter*matrix
	
	if np.max(matrix)!=0:

		
		npy=np.argwhere(matrix!=0)
		if npy.size>0:

			
			npy[:,[0,1]]=npy[:,[1,0]]
			
			
			tri = Delaunay(npy)
			
			
			eadge_npy=np.array([npy[tri.vertices[:,0]],npy[tri.vertices[:,1]],npy[tri.vertices[:,2]]])
			
			
			hypot_npy=np.array([np.sqrt(np.power(eadge_npy[0,:,0]-eadge_npy[1,:,0],2)+np.power(eadge_npy[0,:,1]-eadge_npy[1,:,1],2)),
								np.sqrt(np.power(eadge_npy[0,:,0]-eadge_npy[2,:,0],2)+np.power(eadge_npy[0,:,1]-eadge_npy[2,:,1],2)),
								np.sqrt(np.power(eadge_npy[1,:,0]-eadge_npy[2,:,0],2)+np.power(eadge_npy[1,:,1]-eadge_npy[2,:,1],2))])
			
			
			eadge_npy=np.swapaxes(eadge_npy,1,0)
			hypot_npy=np.swapaxes(hypot_npy,1,0)
			
			
			eadge_npy=eadge_npy[(hypot_npy[:,0]<=triangle_maxlen) &
								(hypot_npy[:,1]<=triangle_maxlen) & 
								(hypot_npy[:,2]<=triangle_maxlen)]
			
			
			cv2.fillPoly(triangle_matrix, eadge_npy, 1)

	return triangle_matrix
