# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.util import img_as_float32
from skimage.color import rgb2gray
from skimage import io
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import math
import numpy as np
import logging
import time

logger = logging.getLogger("main_logger")
 

class SuperpixelSegmentation:
	""" 
	Contains all operations used for image superpixel computation.
	
	Attributes:
		imagePath		Path to the image, which shall be segemented
		numSrSegments		Real calc. superpixel number
		maxNumSegments	Number of wanted segments (real calc. superpixel number can differ)
		segments		Segementation matrix [heigth][width]
		nodes 			List of segementation centroids
		edges			List of neighbouring node-pairs
		meanLuminance	Mean luminance of each superpixel/segement
		k				Number of nearest points for edge computation
	
	"""
	
	def __init__(self, imageRef, index, k=3 , percentage=0.03, k_lrange=[2,3,4], p_lrange= [0.12,0.18,0.25]):
		
		self.imagePath = None
		self.imageRaw = None	
		self.imageGrey = None
		self.imageSolution = None
		self.imageNumber = index
		
		self.numLrSegments = None
		self.numSrSegments = None
		self.segments = None
		self.meanLuminances = list()
		self.k = k
		self.percentage = percentage
		
		self.k_lrange = k_lrange
		self.p_lrange = p_lrange
		
		self.nodes = list()
		self.edges = list()
		
		logger.debug("Load images..")
		#if imageRef is a path to an image
		if isinstance(imageRef, str):
			self.imagePath = imageRef
			self.loadImages()
		#else imageRef is the raw image values
		else:
			self.imageRaw = imageRef
			self.imageGrey = img_as_float32(rgb2gray(imageRef))
			self.imageSolution = img_as_float32(rgb2gray(imageRef))
			logger.debug("imageGrey shape= %s"%str(self.imageGrey.shape))
			logger.debug("imageGrey type= %s"%str(type(self.imageGrey[0][0])))
			logger.debug("raw image shape=%s"%str(self.imageRaw.shape))
			logger.debug("data type of raw image = %s",str(type(self.imageRaw[0][0])))
		logger.debug("Load images done.")	
		
		
		logger.debug("Set superpixel sizes:")
		logger.debug("Normal: p = %0.1f %% , k = %d "%(self.percentage*100,k))
		for k_level,p_level in zip(k_lrange,p_lrange):
			logger.debug("LRange: p = %0.1f %% , k = %d "%(p_level*100,k_level))

		
	def getNumberSegmentsDynamic(self,percentage):
		"""
		Sets the maxNumSegments with reference to the given image width and height.
		Authors setup: superpixel size = 3% of the image width 
		"""
		imgHeight = len(self.imageGrey)
		imgWidth = len(self.imageGrey[0])
		
		widthSuperPixel = imgWidth * percentage
		#equal-sized grid over the image
		numSuperpixels = (imgWidth/ widthSuperPixel) * (imgHeight/widthSuperPixel)
		
		return int(numSuperpixels)
		
	def loadImages(self):
		"""
		Load image in raw(rgb) and grey format.
		"""
		
		# load the image (R,G,B) and convert it to a floating point data type
		self.imageRaw = img_as_float(io.imread(self.imagePath))
		
		# load the image and convert it to a greyscale image
		self.imageGrey = io.imread(self.imagePath, as_gray=True)
		
		# load the image and convert it to a greyscale image
		self.imageSolution = io.imread(self.imagePath, as_gray=True)
		
	
	def slicSegmentation(self,maxNumSegments):
		"""
		Calculates the Segmentation as a matrix with height and width of the image.
		Matrix entries are the number of the segementation. 
		Sets number of segments (class variable) with reference to actual found segments.
	
		"""
		
		# apply SLIC and extract (approximately) the supplied number of segments
		# runtime max_iter = 10 -> ca 1s, max_iter = 30 -> ca 2s
		segmentation = slic(self.imageRaw, n_segments = maxNumSegments, sigma = 0, compactness = 25, max_iter =100)
		
		#Computes actual superpixel number
		tmp = list()
		for i in range(len(segmentation)):
			for j in range(len(segmentation[i])):
				index = segmentation[i][j]
				
				if index not in tmp:
					tmp.append(index)
				
		numSuperpixels = len(tmp)
		
		return (segmentation, numSuperpixels)
    
	def plotSegmentation(self,segmentation,nodes,title):
		"""
		Plots superpixel boundaries on the image.
		"""
		# show the output of SLIC
		fig = plt.figure("Superpixels -- %d segments" % len(nodes))
		plt.imshow(mark_boundaries(self.imageRaw, segmentation))
		
		for (x,y) in nodes:
			plt.scatter(x, y, s=1, c='red', marker='o')
		plt.axis("off")
		
		fig.savefig("imgEval/%d/%s.png"%(self.imageNumber,title),format='png',dpi=100)
		plt.close(fig)
		
	
	
	def segmentsCenters(self,segmentation,numSPixels):
		"""
		Computes a list of the segments centroids, with data from the segment_matrix.
		Centroids coordinates are computed with the sum of the values of points in x- or y-direction
		of one segmentnumber divided by the number of points in this segement.
		
		"""
		
		# Temp list to compute the centroids [(s0_sumx,s0_sumy,s0_no_pixels),...,(sn_sumx,sn_sumy,sn_no_pixels)]
		# tmp = [[0,0,0]] * numSrSegments
		tmp = list()
		#print("Number of segments=", self.numSrSegments)

		for i in range(numSPixels):
			tmp.append([0,0,0])
		
		for i in range(len(segmentation)):
			for j in range(len(segmentation[i])):
				index = segmentation[i][j] #number of the segment

				tmp[index][0] = tmp[index][0] + j # j=x axis
				tmp[index][1] = tmp[index][1]+ i	# i=y axis
				tmp[index][2] = tmp[index][2] + 1

		# Assign nodes to list of segment centroids ([(s0_x,s0_y),...,(sn_x,sn_y)]),
		# sorted by occurance of segments in row-wise processing
		
		nodes_centroids = list()
		for i in range(numSPixels):
			number_points = tmp[i][2]
			if number_points != 0:
				x_center = tmp[i][0] / number_points
				y_center = tmp[i][1] / number_points
				nodes_centroids.append((x_center,y_center))
		
		#Sanity check
		imgHeight = len(self.imageRaw)
		imgWidth = len(self.imageRaw[0])
		range_errors = 0
		
		for (x,y) in nodes_centroids:
			if(x<0 or y<0 or x>imgWidth or y>imgHeight):
				range_errors = range_errors + 1
		
		if(range_errors>0):
			raise ValueError("%d centroids are out of the image dimensions."%range_errors)
		
		return nodes_centroids
		
	def plotEdges(self,nodes,edges,k,p):
		"""
		Draw edges into plot.
		"""
		
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
		cLen = len(colors)
		i=0
		#plt.ioff()
		fig = plt.figure("Edges between centroids of superpixels")
		#plt.imshow(self.imageRaw,cmap='gray',vmin=0,vmax=1)
		plt.imshow(mark_boundaries(self.imageRaw, self.segments))
		#Draw centroids as red points into the image/plot.		
		for (x,y) in nodes:
			plt.scatter(x, y, s=1, c='red', marker='o')
		
		for ((x1,y1) , (x2,y2)) in edges:
			p1 = [x1,x2]
			p2 = [y1,y2]
			#Choosing next color
			#plt.plot( p1 , p2 ,marker='o',color= colors[int(i/3)%(cLen-1)] , linewidth=1 , markersize=3 )
			plt.plot( p1 , p2 ,marker='o',color='k' ,linewidth=0.5,markersize=2 )
			
			i = i +1
			
		fig.savefig("imgEval/%d/edges_visualization_k=%0.2f_p=%0.2f.png"%(self.imageNumber,k,p),format='png',dpi=100)
		plt.close(fig)
	
	def calculateNearestNeighbors(self,centroids,k):
		"""
		For each point of the input list the k-nearest points are found and
		the resulting k lines are saved.
				
		:return: list of line tuples ((x_p1,y_p1),(x_p2,y_p2))
		"""
		edges = list()
		count = 0 # to delete
		
		centroids_tmp = np.asarray(centroids)
		
		start_time = time.time()
		nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', metric='euclidean').fit(centroids_tmp)
		distances, indices = nbrs.kneighbors(centroids_tmp)
		print("--- %s seconds --- for calculating edges." % (time.time() - start_time))

		for i in range(len(indices)):
			p1 = centroids[i]
			for j in range(len(indices[i]))[1:]:
				index_to = indices[i][j]
				p2 = centroids[index_to]
				if ((p2,p1) not in edges):
					edges.append((p1,p2))
				else:
					count = count + 1

		print("edges size=",len(edges))
		print("%d edges are discarded."%count)		
		return edges
	
	def removeDuplicateEdges(self,edges):
		edges_result = list()
		count = 0
		for ((x1,y1),(x2,y2)) in edges:
			if( ((x1,y1),(x2,y2)) not in edges_result and ((x2,y2),(x1,y1)) not in edges_result):
				edges_result.append(((x1,y1),(x2,y2)))
			else:
				count = count + 1

		print("edges size=",len(edges_result))
		print("%d edges are discarded."%count)		
		return edges_result
	
	def getLRangeEdges(self,c_normal, c_lrange, k_level):
		"""
		:params		c_normal centroids short distance
					c_lrange centroids large distance
		"""
		edges_lr = list()
		
		def calculateDistance(p1,p2):
			x1,y1 =p1[0],p1[1]
			x2,y2 = p2[0],p2[1]
			dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
			return dist
			
		map_superpixels = dict()
		
		for c_lr in c_lrange:
			least_dist_node = c_normal[0]
			dist = calculateDistance(c_normal[0],c_lr)
			centroid = c_normal[0]
			
			for c_n in c_normal:
				dist_tmp = calculateDistance(c_n,c_lr)
				if(dist_tmp<dist):
					dist = dist_tmp
					centroid = c_n
			map_superpixels[c_lr] = centroid
		
		if(len(c_lrange)<= k_level):
			raise ValueError("Too few centroids for long range connections, please decrease size of superpixels.")
					
		edges_lr_raw = self.calculateNearestNeighbors(c_lrange,k_level)	
		for (p1,p2) in edges_lr_raw:
			p1_real = map_superpixels[p1]
			p2_real = map_superpixels[p2]
			if( (p2_real,p1_real) not in edges_lr ):
				edges_lr.append((p1_real,p2_real))
		
		return edges_lr
			
	def calcSegmentation(self):
		"""
		Sets the parameters for the segmentation and line element computation 
		and launches the computations. 
		
		:return: (segmentation centroids list, edges list ,mean luminance value list)
		"""
		#Calculate SLIC superpixel segmentation
		numSeg_normal = self.getNumberSegmentsDynamic(self.percentage)
		segmentation, numSuperpixels = self.slicSegmentation(numSeg_normal);
		self.segments = segmentation
		self.numSrSegments = numSuperpixels
		
		#Calculate centroids of the superpixels
		centroids = self.segmentsCenters(segmentation,numSuperpixels);
		self.nodes = centroids
		
		#Nearest neighbour computation for getting edges
		edges = self.calculateNearestNeighbors(centroids,self.k)
		self.plotEdges(centroids,edges,self.k,self.percentage)
		
		#Determine mean superpixel/segement luminances
		meanLuminances = self.meanSegmentLuminance(segmentation,numSuperpixels)
		self.meanLuminances = meanLuminances
		
		print("Superpixels sr-connectivity=",len(centroids))
		print("Edges sr-connectivity=",len(edges))
		self.plotSegmentation(segmentation,centroids,"superpixel_edges_srange k= %d p= %d"%(self.k,self.percentage))
		
		for k_level,p_level in zip(self.k_lrange,self.p_lrange):
			#Compute segmentation for long range connections
			numSeg_lrange = self.getNumberSegmentsDynamic(p_level)
			
			seg_lrange, numS_lrange = self.slicSegmentation(numSeg_lrange)
			
			centroids_lrange = self.segmentsCenters(seg_lrange,numS_lrange)
			
			edges_lr = self.getLRangeEdges(centroids, centroids_lrange, k_level)
			print("Superpixels lr-connectivity=",len(centroids_lrange))
			print("Edges lr-connectivity=",len(edges_lr))
			self.plotSegmentation(seg_lrange,centroids_lrange,"superpixel_edges_lrange k= %d p= %0.2f"%(k_level,p_level))
			self.plotEdges(centroids_lrange,edges_lr,k_level,p_level)
			
			edges.extend(edges_lr)
		
		print(self.removeDuplicateEdges([((1,2),(3,4)),((2,1),(4,3)),((3,4),(1,2))]))
		print("edges type=", type(edges[0]))
		print("edges= ",edges[:100])
		edges = self.removeDuplicateEdges(edges)
		self.edges = edges

		print("Edges total=",len(edges))

		
		return (centroids,edges,segmentation,meanLuminances)
		
	def floodfillImage(self, fillValues):
		"""
		Used to fill the superpixels in the image with the computed values 
		from the optimization algorithm.

		:param:	fillValues	list with segment no. -> greylevel ( 1=white , 0=black)
		
		"""
		
		imgHeight = len(self.imageGrey)
		imgWidth = len(self.imageGrey[0])
		
		image = np.ones([imgHeight,imgWidth],dtype="float32")

		for i in range(imgHeight):
			for j in range(imgWidth):
				segmentNumber = self.segments[i][j]
				image[i][j] = fillValues[segmentNumber]
		
		return image	
		
			
	def meanSegmentLuminance(self,segmentMatrix,numSuperpixels):
		"""
		Computes the mean luminance of each segment respectively superpixel.
		
		:return: list of mean greylevel values with index = segement number,
				 sorted by occurance of segments in row-wise processing
		"""

		#(x,y) : x= sum of luminance, y= number of points (in this segment)
		tmp = [[0,0] for i in range(numSuperpixels)]
		
		if self.imageGrey.max() > 1:
			raise ValueError("Image in range [0,1] expected for mean luminance calculation.")
		
		for i in range(len(segmentMatrix)):
			for j in range(len(segmentMatrix[i])):
				index = segmentMatrix[i][j]
				
				tmp[index][0] = tmp[index][0] + self.imageGrey[i][j]
				tmp[index][1] = tmp[index][1] + 1
	
		meanLuminances = list()
		for (sum,count) in tmp:
			meanLuminances.append(sum/count)
			
		return meanLuminances
		
