import math
import numpy as np
import SuperpixelSegmentation as s



def computeL2DistGT(xVector,R_gt,W_gt,nodes,edges):
	"""
	Computes the L2 greater than distance between corresponding points(edge).
	Calculates result by matrix-multiplication from right to left.
	Resulting formula= [x,R_gt]^T * ( A_gt^T * ( W_gt * ( A_gt * [x,R_gt] )))
	
	:param: xVector greyscale values of nodes
			R_gt 	slack variable for each edge
			W_gt 	weights matrix (greater than)
			nodes 	list of points/centroids
			edges	list of edges
			
	:return: L2 distance
	"""
	
	numNodes = len(xVector)
	numEdges = W_gt.size
	
	#xR_eq : |N|+|E| x 1 (vector?)
	#xVector size = |N| , R_gt size = |E|
	xR_eq = np.array(xVector.extend(R_gt))
	
	#A_gt : |E| x |N|+|E|
	A_gt = np.zeros([numEdges, numNodes + numEdges], dtype=int)
	
	#Index of current edge
	p = 0
	#Fill the matrix A_gt
	for ((x0,y0),(x1,y1)) in edges:
		#Find index i,j = position of points in node list
		i_pos = nodes.index((x0,y0)) 
		j_pos = nodes.index((x1,y1)) 
		
		A_gt[p][i_pos] = 1
		A_gt[p][j_pos] = -1
		A_gt[p][numNodes+p] = -1
	

	result = A_gt.dot(xR_eq)	#works?
	result = W_gt.dot(result)
	result = (A_gt.transpose()).dot(result)
	result = (xR_eq.transpose()).dot(result)
	
	return result
	
	
def getWeigthMatrices(lines,imagePath):
	"""
	Calculates the diagonal weight matrices. 
	The weights of two points are probabilities indicating 
	which point is farer respectively nearer from each other in the image.
	
	:param: lines edges (p1,p2)
			imagePath path to the image
	:return: (weight-matrix greater than, weight-matrix less than)
	"""
	
	# load the image and convert it to a greyscale image
	image = io.imread(imagePath, as_gray=True)
	
	numEdges = len(lines)
	#Intialize weight matrices(size |E|x|E|) with all zeros
	W_gt = np.zeros([numEdges, numEdges], dtype=int)
	W_lt = np.zeros([numEdges, numEdges], dtype=int)
	W_eq = np.zeros([numEdges, numEdges], dtype=int)
	
	edgeIndex=0
	
	for((x0,y0),(x1,y1)) in lines:
		(w_gt, w_lt) = calc(x0,y0,x1,y1,image)
		
		W_gt[edgeIndex][edgeIndex] = w_gt
		W_lt[edgeIndex][edgeIndex] = w_lt
	
		edgeIndex = edgeIndex + 1
		
	return (W_gt,W_lt)

def calc(x0,y0,x1,y1,image):
	return (0.2,0.8)
	
def main():
	imagePath = "living_room_small.jpg"
	(points, lines) = s.calcSegmentation(imagePath)
	
	print(points)
	print(lines)
	
	xVector = s.getPixelValues(points,imagePath)
	print("xVector=",xVector)
	print("xVector length=",len(xVector))
	print("Points length=",len(points))
	
	x= [1,1]
	R_gt = [1]
	xR_eq = np.array(xVector.extend(R_gt))
	A_gt = np.zeros([numEdges, numNodes + numEdges], dtype=int)
	
	
if __name__ == "__main__":
    main()