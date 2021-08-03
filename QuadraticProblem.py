import math
import numpy as np
import SuperpixelSegmentation
from HelperFunctions import plotImage,saveImage
from skimage import io
from prediction import predict
import config
import logging

logger = logging.getLogger("main_logger")

class QuadraticProblem:
	""" 
	Represents the optimization problem.  
	
	Attributes:
		A_gt, A_lt, A_eq, A_s 	Help matrices for among other L2Dist computation
		W_gt, W_lt, W_eq, W_s	Weight matrices with weights = depth probabilities
		b_s 					Smoothing term for flexible smoothing
		superpixels				Reference to the segmentation object
	"""
	
	
	
	def __init__(self, superpixels,index,b_s ,p):
	
		self.A_gt, self.A_lt, self.A_eq, self.A_s = [None] * 4
		self.W_gt, self.W_lt, self.W_eq, self.W_s = [None] * 4
		self.b_s = None
		self.p = None
		self.superpixels = None
		self.imageNumber = index
		
		if superpixels is None:
			raise ValueError('Segmentation setup not initialized! Please first initialize.')
		
		self.superpixels = superpixels
		self.setupEnvironment(b_s,p)
	
	def getWeigthMatrices(self,nodes, edges, meanLuminances):
		"""
		Calculates the diagonal weight matrices. 
		The weights of two points are probabilities indicating 
		which point is farer respectively nearer from each other in the image.
		
		:param: nodes segment centroid list
				edges lines (p1,p2)
				meanLuminances list with mean luminance values of segments
				
		:return: (weight-matrix greater than, weight-matrix less than,
				  weight-matrix equal, weight-matrix smoothing)
		"""
		#print("nodes=",nodes)
		#print("edges=",edges)
		
		image = self.superpixels.imageRaw / 255
		# load the image and convert it to a greyscale image
		#image = io.imread(self.superpixels.imagePath).astype(dtype=np.float32)/255
		
		print("Image shape=", image.shape, image.max(), image.min())
		
		numEdges = len(edges)
		#Intialize weight matrices(size |E|x|E|) with all zeros
		W_gt = np.zeros([numEdges, numEdges], dtype="float32")
		W_lt = np.zeros([numEdges, numEdges], dtype="float32")
		W_eq = np.zeros([numEdges, numEdges], dtype="float32")
		W_s = np.zeros([numEdges, numEdges], dtype="float32")
		
		
		list_points_a = list()
		list_points_b = list()
		
		for((x0,y0),(x1,y1)) in edges:
			list_points_a.append((x0,y0))
			list_points_b.append((x1,y1))
		
		logger.debug("Number of point pairs for prediction=%d"%len(edges))
		
		logger.debug("Calculate weights of edges with pre-trained model:")
		weights = self.calc(list_points_a,list_points_b,image)
		
		var = np.var(weights) #, axis=1) # Computing variance for comparison/testing
		print("Weight Variances = ", var)

		logger.debug("Calculate weights of edges with pre-trained model done.")
		print("weights=",weights[1:20])
		for edgeIndex in range(len(edges)):
			w_gt = weights[edgeIndex][0]
			w_lt = weights[edgeIndex][1]
			
			if(edgeIndex%200 == 0):
				logger.debug("weigths of edge %d = ( %f , %f )" ,edgeIndex,w_gt,w_lt)
				
			W_gt[edgeIndex][edgeIndex] = w_gt
			W_lt[edgeIndex][edgeIndex] = w_lt
			# assumption that equals result form gt and lt
			W_eq[edgeIndex][edgeIndex] = 1 - abs(w_gt-w_lt)
		
		
		#Index of current edge
		edgeIndex=0
		#Fill the matrix W_s
		for ((x0,y0),(x1,y1)) in edges:
			#Find index i,j = position of points in node list
			i_pos = nodes.index((x0,y0)) 
			j_pos = nodes.index((x1,y1)) 
			
			lum_p0 = meanLuminances[i_pos]
			lum_p1 = meanLuminances[j_pos]
			
			W_s[edgeIndex][edgeIndex] = math.exp( (-1/self.p) * (abs( lum_p0 - lum_p1 ))**2 )
			
			edgeIndex = edgeIndex +1
		
		
		print("W_gt shape:",W_gt.shape)
		print("W_lt shape:",W_lt.shape)
		print("W_eq shape:",W_eq.shape)
		print("W_s shape:",W_s.shape)
		
		return (W_gt,W_lt,W_eq,W_s)
	
	def plotRawWeightData(self):
		"""
		Used to visualize the raw weight data from the models prediction.
		Higher values of a node indicate a higher dominance for the greater than property.
		"""
		
		nodes = self.superpixels.nodes
		edges = self.superpixels.edges
		
		greaterThanVisualization = [0]*len(nodes)
		for edgeIndex in range(len(edges)):
			(x0,y0),(x1,y1) = edges[edgeIndex]
			
			
			if( self.W_gt[edgeIndex][edgeIndex] > 0.5 ):
				pos = nodes.index((x0,y0)) 
				greaterThanVisualization[pos] = greaterThanVisualization[pos] + 1
			#else:
			#	pos = nodes.index((x1,y1)) 
			#	greaterThanVisualization[pos] = greaterThanVisualization[pos] + 1
		
		if(max(greaterThanVisualization) == 0):
			print("Don't plot raw weight data, since all elements in array are zero.")
			return
		normVisualization = [i/max(greaterThanVisualization) for i in greaterThanVisualization]

		vImage = self.superpixels.floodfillImage(normVisualization)	
		
		print("vImage shape:", vImage.shape)
		saveImage(vImage, "Visualization Greater Than","%d/visualization_predicted_weights"%self.imageNumber)
	

	def setSmoothnessParameters(self,b_s, p):
		"""
		Sets the specified smoothing term paramter b for more flexibility in smoothing.
		
		:param  	b_s smoothing term
					p smoothing term for weights
		:return: b_s 	vector/list of size |number nodes|
		"""
		logger.debug("Setting smoothing parameters: b_s = %f, p = %f"%(b_s,p))
		numNodes = len(self.superpixels.nodes)
		#b_s_mean = b_s
		#stdDev = 0.001
		#self.b_s = np.array( np.random.normal(b_s_mean, stdDev, numNodes), dtype="float32" ) #maybe better with normal distr. initialization ?!
		self.b_s = [b_s] * numNodes	#TODO: how to compute this term??
		self.p = p
		
		
	def calc(self,list_a,list_b,image):
		"""
		Calculates depth probabilities for all given edges.
		
		:param:	list_a	list of points p1
				list_b	list of points p2
				image 	target image to infer probabilities
		
		:return: list of 2-D tuple (probability p1>p2 ,probability p2>p1)
		"""
		# load params
		params = config.DIWConfig()

		probabilities = list()

		predictions = list()

		#Calculate predictions by calling predict function multiple times,
		#in order to prevent too big input data (>2GB) tf error
		
		for batch in predict(params, image, list_a, list_b):
			print("batch type", type(batch))
			print("batch=",batch)
			predictions.extend(list(batch))
			print("shape of prediction: ", type(predictions))
			print("predictions[0]: ",predictions[0])

		for i in range(len(predictions)):
			probabilities.append(predictions[i]['probabilities'])
		
		return probabilities
		
	
	def getSlackVariables(self,mode): 
		"""
		Get a test slack variables for each edge. 
		Computed with different normal distributions.
		
		:param: edges list of edges
				mode  shading/reflectance or depth recovery
				
		:return: (slack vector for greater than, slack vector for less than,
				  slack vector for equals)
		"""
		
		stdDevGT= None
		stdDevLT= None
		stdDevEQ= None
		
		meanGT= math.log(2,10)
		meanLT= math.log(2,10)
		meanEQ= 0
		
		if(mode=="SHADE_REFLECT"):
			stdDevGT= math.pow(0.001,2)
			stdDevLT= math.pow(0.001,2)
			stdDevEQ= math.pow(0.001,2)
		elif(mode=="DEPTH-RECOVERY"):
			stdDevGT= math.pow(4,2)
			stdDevLT= math.pow(4,2)
			stdDevEQ= math.pow(0.1,2)
		else:
			raise ValueError('Please select a valid mode/target.')
		
		numEdges = len(self.superpixels.edges)
	
		R_gt = np.array( np.random.normal(meanGT, stdDevGT, numEdges), dtype="float32" )
		R_lt = np.array( np.random.normal(meanLT, stdDevLT, numEdges), dtype="float32" )
		R_eq = np.array( np.random.normal(meanEQ, stdDevEQ, numEdges), dtype="float32" )
		
		
		#R_gt and R_lt should be positive-valued
		#for i in range(numEdges):
		#	if R_gt[i] < 0:
		#		R_gt[i] = 0
		#	if R_lt[i] < 0:
		#		R_lt[i] = 0
		
		return (R_gt,R_lt,R_eq)

	def getMatricesA(self,nodes,edges):
		"""
		Computes the matrices A_gt, A_lt and A_eq. Used for term computations.
		:param:	nodes	list of segments centroids
				edges 	list of edges
				
		:return: (A_gt, A_lt, A_eq, A_s)
		"""
		
		numEdges = len(edges)
		numNodes = len(nodes)
		
		#A_xy : |E| x |N|+|E|
		A_gt = np.zeros([numEdges, numNodes + numEdges],dtype="float32")
		A_lt = np.zeros([numEdges, numNodes + numEdges],dtype="float32")
		A_s = np.zeros([numEdges, numNodes],dtype="float32")
		
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
		A_lt = np.copy((-1)*A_gt)
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
	
	
	def setupEnvironment(self, b_s , p):
		"""
		Used to init the computation of the segmentation. 
		Saves the implicit weight and helper matrices locally.
		
		:param  	b_s smoothing term
					p smoothing term for weights
		"""
		
		#Calculates segmentation, superpixel centroids and mean luminances 
		print("Calculate superpixels..")
		(nodes,edges,segmentation,meanLuminances) = self.superpixels.calcSegmentation()

		print("Calculate superpixels done.")
		
		self.setSmoothnessParameters(b_s, p)
		
		print("Computing matrices for optimization formula..")
		logger.debug("Computing matrices for optimization formula")
		(self.W_gt, self.W_lt, self.W_eq, self.W_s) = self.getWeigthMatrices(nodes,edges,meanLuminances)
		(self.A_gt, self.A_lt, self.A_eq, self.A_s) = self.getMatricesA(nodes,edges)
		print("Computing matrices for optimization formula done.")
		
	
	def runningTests(self):
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
		(A_gt, A_lt, A_eq, A_s) = self.getMatricesA(nodes,edges)
		print("A_gt= \n", A_gt , "\n")
		print("A_lt= \n", A_lt, "\n")
		print("A_eq= \n", A_eq, "\n")
		print("A_s= \n", A_s, "\n")
		
		W_gt = np.zeros([numEdges, numEdges],dtype="float32")
		W_gt[0][0] = 1	#(x0,y0)> (x1,y1)
		
		res = self.computeL2Dist(xVector,R_gt,A_gt,W_gt)
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
		(A_gt, A_lt, A_eq, A_s) = self.getMatricesA(nodes,edges)
		print("A_gt= \n", A_gt , "\n")
		print("A_lt= \n", A_lt, "\n")
		print("A_eq= \n", A_eq, "\n")
		print("A_s= \n", A_s, "\n")
		
		W_gt = np.zeros([numEdges, numEdges], dtype="float32")
		W_gt[0][0] = 0.8	#P1>P2
		W_gt[1][1] = 0.7	#P2>P3
		W_gt[2][2] = 0.9	#P3>P4
		W_gt[3][3] = 0.1	#P4>P1
		
		W_s = np.copy(W_gt)
		
		b_s = [0,0,0,0]
		
		res = self.computeL2Dist(xVector,R_gt,A_gt,W_gt)
		print("L2DistGT= ",res)
		
		res= self.computeSmoothTerm(xVector,A_s,W_s,b_s)
		print("SmoothTerm= ",res)
		
		#Test 3: Test e-function calculation
		#Correct result = 0.999000499833375
		print("calculation:" ,math.exp( (-1/10) * (abs( 0.8 - 0.9 ))**2 ))
			
