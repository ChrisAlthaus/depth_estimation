import math
import numpy as np
import SuperpixelSegmentation as s
import ProblemSolver as solv
from skimage import io


def computeL2Dist(xVector,A_xy,R_xy,W_xy):
	"""
	Computes the L2 distance between corresponding points(edge).
	Calculates result by matrix-multiplication from right to left.
	Resulting formula= [x,R_gt]^T * ( A_gt^T * ( W_gt * ( A_gt * [x,R_gt] )))
	
	:param: xVector greyscale values of nodes
			R_xy 	slack variable for each edge (GT,LT or EQ)
			A_xy	multiplication help matrix
			W_xy	weights matrix (GT,LT or EQ)
			
	:return: L2 distance
	"""
	
	#xR_xy : |N|+|E| x 1 (vector?)
	#xVector size = |N| , R_xy size = |E|
	xR_xy = np.array(xVector + R_xy)
	
	#Printing matrices and dimensions
	print()
	print("x= \n" , xVector)
	print("R_xy \n=" , R_xy)
	print("->xR_xy= \n" , xR_xy)
	print("A_xy= \n" , A_xy)
	print("W_xy= \n" , W_xy)
	
	print("dimensions R_xy=" , len(R_xy))
	print("dimensions xR_xy=" , xR_xy.shape)
	print("dimensions A_xy=" , A_xy.shape)
	print("dimensions W_xy=" , W_xy.shape)
	
	#Computing dot products
	result = A_xy.dot(xR_xy)
	#print("A * xR=" , result)
	print("dimensions A * xR=" , result.shape)
	result = W_xy.dot(result)
	#print("W * A * xR=" , result)
	print("dimensions W * A * xR=" , result.shape)
	result = (A_xy.transpose()).dot(result)
	#print("A^T * W * A * xR=" , result)
	print("dimensions A^T * W * A * xR=" , result.shape)
	result = (xR_xy.transpose()).dot(result)
	#print("xR^T * A^T * W * A * xR=" , result)
	print("dimensions xR^T * A^T * W * A * xR=" , result.shape)
	print("->xR=" , xR_xy.transpose())
	print("result=",result)
	print()
	
	return result
	
def computeSmoothTerm(xVector,A_s,W_s,b_s):
	"""
	Computes the smoothness term. Allows to enforce smoothness in the image.
	Calculates result by matrix-multiplication from right to left.
	Resulting formula= x^T * ( A_s^T * ( W_s * ( A_s * x ))) + x^T * b_s
	
	:param:	xVector greyscale values of nodes
			A_s 	multiplication help matrix
			W_s 	matrix of smoothness weights
			b_s		vector (format = list) for more flexible smoothing, size=|N|
			
	:return smoothness value
	"""
	
	#Printing matrices and dimensions
	print()
	print("x= \n" , xVector)
	print("A_s= \n" , A_s)
	print("W_s= \n" , W_s)
	print("b_s= \n" , b_s)
	
	print("dimensions A_s=" , len(A_s))
	print("dimensions W_s=" , W_s.shape)
	print("dimensions b_s=" , len(b_s))
	
	x = np.array(xVector)
	
	result = A_s.dot(x)
	print("dimensions A_s * x=" , result.shape)
	result = W_s.dot(result)
	print("dimensions W_s * A_s * x=" , result.shape)
	result = (A_s.transpose()).dot(result)
	print("dimensions A_s^T * W_s * A_s * x=" , result.shape)
	result = (x.transpose()).dot(result)
	print("dimensions x^T * A_s^T * W_s * A_s * x=" , result.shape)
	
	result = result + (x.transpose()).dot(b_s)
	print("dimensions A_s^T * W_s * A_s * x + x^T * b_s=" , result.shape)
	
	
	return result
	
	

	
def getWeigthMatrices(nodes, edges,imagePath,meanLuminances):
	"""
	Calculates the diagonal weight matrices. 
	The weights of two points are probabilities indicating 
	which point is farer respectively nearer from each other in the image.
	
	:param: nodes segment centroid list
			edges lines (p1,p2)
			imagePath path to the image
			meanLuminances list with mean luminance values of segments
			
	:return: (weight-matrix greater than, weight-matrix less than,
			  weight-matrix equal, weight-matrix smoothing)
	"""
	
	# load the image and convert it to a greyscale image
	image = io.imread(imagePath, as_gray=True)
	
	numEdges = len(edges)
	#Intialize weight matrices(size |E|x|E|) with all zeros
	W_gt = np.zeros([numEdges, numEdges], dtype=float)
	W_lt = np.zeros([numEdges, numEdges], dtype=float)
	W_eq = np.zeros([numEdges, numEdges], dtype=float)
	W_s = np.zeros([numEdges, numEdges], dtype=float)
	
	edgeIndex=0
	
	for((x0,y0),(x1,y1)) in edges:
		(w_gt, w_lt) = calc(x0,y0,x1,y1,image)
		
		W_gt[edgeIndex][edgeIndex] = w_gt
		W_lt[edgeIndex][edgeIndex] = w_lt
		# assumption that equals result form gt and lt
		W_eq[edgeIndex][edgeIndex] = 1 - abs(w_gt-w_lt)
	
		edgeIndex = edgeIndex + 1
	
	#Smoothing parameter
	p = 1
	
	#Index of current edge
	edgeIndex=0
	#Fill the matrix W_s
	for ((x0,y0),(x1,y1)) in edges:
		#Find index i,j = position of points in node list
		i_pos = nodes.index((x0,y0)) 
		j_pos = nodes.index((x1,y1)) 
		
		lum_p0 = meanLuminances[i_pos]
		lum_p1 = meanLuminances[j_pos]
		
		W_s[edgeIndex][edgeIndex] = math.exp( (-1/p) * (abs( lum_p0 - lum_p1 ))**2 )
		
		edgeIndex = edgeIndex +1
		

		
	return (W_gt,W_lt,W_eq,W_s)

def calc(x0,y0,x1,y1,image):
	return (0.2,0.8)
	
def getSlackVariables(edges):
	"""
	Get a slack variables for each edge. 
	Computed with different normal distributions.
	
	:param: edges list of edges
	
	:return: (slack vector for greater than, slack vector for less than,
			  slack vector for equals)
	"""
	
	stdDevGT= 0.001
	stdDevLT= 0.001
	stdDevEQ= 0.001
	
	meanGT= math.log(2)
	meanLT= math.log(2)
	meanEQ= 0
	
	#TODO: how compute slack variables with normal distribution (mapping?)
	
	numEdges = len(edges)
	
	R_gt = [1] * numEdges;
	R_lt = [1] * numEdges;
	R_eq = [1] * numEdges;
	
	return (R_gt,R_lt,R_eq)

def getMatricesA(nodes,edges):
	"""
	Computes the matrices A_gt, A_lt and A_eq. Used for term computations.
	:param:	nodes	list of segments centroids
			edges 	list of edges
			
	:return: (A_gt, A_lt, A_eq, A_s)
	"""
	
	numEdges = len(edges)
	numNodes = len(nodes)
	
	#A_xy : |E| x |N|+|E|
	A_gt = np.zeros([numEdges, numNodes + numEdges],dtype="float64")
	A_lt = np.zeros([numEdges, numNodes + numEdges],dtype="float64")
	A_s = np.zeros([numEdges, numNodes],dtype="float64")
	
	#Index of current edge
	p = 0
	#Fill the matrix A_gt ( size=|E|x|E|+|N| )
	for ((x0,y0),(x1,y1)) in edges:
		#Find index i,j = position of points in node list
		i_pos = nodes.index((x0,y0)) 
		j_pos = nodes.index((x1,y1)) 
		
		A_gt[p][i_pos] = 1
		A_gt[p][j_pos] = -1
		A_gt[p][numNodes+p] = -1
		
		p = p + 1
	
	#Set A_lt = - A_gt
	for row in range(A_gt.shape[0]):
		for col in range(A_gt.shape[1]):
			if(A_lt[row][col]!= 0):
				A_lt[row][col] = A_gt[row][col] * (-1);
	
	#A_eq == A_gt
	A_eq = np.copy(A_gt)
	
	#Index of current edge
	p = 0
	#Fill the matrix A_s ( size=|E|x|N| )
	for ((x0,y0),(x1,y1)) in edges:
		#Find index i,j = position of points in node list
		i_pos = nodes.index((x0,y0)) 
		j_pos = nodes.index((x1,y1)) 
		
		A_s[p][i_pos] = 1
		A_s[p][j_pos] = -1
		
		p = p + 1
	
	return (A_gt, A_lt, A_eq, A_s)
	
def runningTests():
	"""
	Runs L2 distance computation with different setups. 
	Then comparing results with results computed by hand.
	"""	
	
	#Test 1 Setup: two nodes, one connecting edge
	#Correct result: 0.25
	print("Test 1:")
	nodes = [(0,0),(100,100)]
	edges = [( (0,0),(100,100) )] 
	xVector= [1,0.5] #aka pixel intensities
	R_gt = [1]
	
	numEdges = len(edges)
	(A_gt, A_lt, A_eq, A_s) = getMatricesA(nodes,edges)
	print("A_gt= \n", A_gt , "\n")
	print("A_lt= \n", A_lt, "\n")
	print("A_eq= \n", A_eq, "\n")
	print("A_s= \n", A_s, "\n")
	
	W_gt = np.zeros([numEdges, numEdges],dtype="float64")
	W_gt[0][0] = 1	#(x0,y0)> (x1,y1)
	
	res = computeL2Dist(xVector,A_gt,R_gt,W_gt)
	print("L2DistGT=", res)
	
	#Test 2 Setup: four nodes, four edges 
	#Correct results: 
	#		L2DistGT = 2.925
	#		SmoothTerm = 0.045
	print("Test 2:")
	nodes = [(0,0),(0,100),(100,100),(100,0)]
	edges = [( (0,0),(0,100) ), ( (0,100),(100,100) ), 
			 ( (100,100),(100,0) ), ( (100,0), (0,0) )] 
	xVector= [0.1,0.2,0.4,0.4] #aka pixel intensities
	R_gt = [1,1,1,1]
	
	numEdges = len(edges)
	(A_gt, A_lt, A_eq, A_s) = getMatricesA(nodes,edges)
	print("A_gt= \n", A_gt , "\n")
	print("A_lt= \n", A_lt, "\n")
	print("A_eq= \n", A_eq, "\n")
	print("A_s= \n", A_s, "\n")
	
	W_gt = np.zeros([numEdges, numEdges], dtype="float64")
	W_gt[0][0] = 0.8	#P1>P2
	W_gt[1][1] = 0.7	#P2>P3
	W_gt[2][2] = 0.9	#P3>P4
	W_gt[3][3] = 0.1	#P4>P1
	
	W_s = np.copy(W_gt)
	
	b_s = [0,0,0,0]
	
	res = computeL2Dist(xVector,A_gt,R_gt,W_gt)
	print("L2DistGT= ",res)
	
	res= computeSmoothTerm(xVector,A_s,W_s,b_s)
	print("SmoothTerm= ",res)
	
	#Test 3: Test e-function calculation
	#Correct result = 0.999000499833375
	print("calculation:" ,math.exp( (-1/10) * (abs( 0.8 - 0.9 ))**2 ))
	
	
def matrixMulTest():
	
	x = [1,0,1,0]
	x_array = np.array(x)
	y = [1,2,3,4]
	y_array = np.array(y)
	
	result = x_array.transpose().dot(y)
	print("x^t * y : ")
	print(x_array.transpose() , " * " ,y_array , " = " , result )


def main():
	#runningTests()
	
	imagePath = "living_room_small.jpg"
	(points, lines, segmentMatrix, meanLuminances) = s.calcSegmentation(imagePath)
	
	#runningTests()
	print(points)
	print(lines)
	
	xVector = s.getPixelValues(points,imagePath)
	(W_gt, W_lt, W_eq, W_s) = getWeigthMatrices(points,lines,imagePath,meanLuminances)
	(R_gt, R_lt, R_eq) = getSlackVariables(lines)
	(A_gt, A_lt, A_eq, A_s) = getMatricesA(points,lines)
	
	#print("xVector=",xVector)
	print("xVector length=",len(xVector))
	print("Points length=",len(points))
	
	distGT = computeL2Dist(xVector,A_gt,R_gt,W_gt)
	#distLT = computeL2Dist(xVector,A_lt,R_lt,W_lt)
	#distEQ = computeL2Dist(xVector,A_eq,W_eq)
	
	print("distGT=",distGT)
	#print("distLT=",distLT)
	#print("distEQ=",distEQ)
	
	numSegments = len(meanLuminances)
	xSolution = solv.solveProblem(numSegments)
	
	s.floodfillImage(segmentMatrix,xSolution,imagePath)

	s.showPlot()
	
if __name__ == "__main__":
    main()