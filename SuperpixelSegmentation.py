# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import math
 

class SuperpixelSegmentation:
	""" 
	Contains all operations used for image superpixel computation.
	
	Attributes:
		imagePath		Path to the image, which shall be segemented
		numSegments		Number of wanted segments (real calc. superpixel number can differ)
		segments		Segementation matrix [heigth][width]
		nodes 			List of segementation centroids
		edges			List of neighbouring node-pairs
		meanLuminance	Mean luminance of each superpixel/segement
		k				Number of nearest points for edge computation
	
	"""
	
	imagePath = None
	imageRaw = None
	imageGrey = None
	imageSolution = None
	
	numSegments = None
	segments = None
	meanLuminances = list()
	k = 3
	
	nodes = list()
	edges = list()
	
	
	def __init__(self, imagePath, numSegments):
		self.imagePath = imagePath
		self.numSegments = numSegments
		
		self.loadImages()
		
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
		
	
	def slicSegmentation(self):
		"""
		Calculates the Segmentation as a matrix with height and width of the image.
		Matrix entries are the number of the segementation.
	
		"""
		
		# apply SLIC and extract (approximately) the supplied number of segments
		segments = slic(self.imageRaw, n_segments = self.numSegments, sigma = 5)
		self.segments = segments
		
		# show the output of SLIC
		fig = plt.figure("Superpixels -- %d segments" % (self.numSegments))
		plt.imshow(mark_boundaries(self.imageRaw, segments))
		
		plt.axis("off")
		
		# Printing for validation
		#for i in range(len(segments)):
		#	print("segements list[",i,"] = " , segments[i][:24]);

	def segmentsCenters(self):
		"""
		Returns a list of the segments centroids, with data from the segment_matrix.
		Centroids coordinates are computed with the sum of the values of points in x- or y-direction
		of one segmentnumber divided by the number of points in this segement.
		
		"""
		
		# Temp list to compute the centroids [(s0_sumx,s0_sumy,s0_no_pixels),...,(sn_sumx,sn_sumy,sn_no_pixels)]
		# tmp = [[0,0,0]] * numSegments
		tmp = list()
		
		for i in range(self.numSegments):
			tmp.append([0,0,0])

		for i in range(len(self.segments)):
			for j in range(len(self.segments[i])):
				index = self.segments[i][j] #number of the segement
				
				tmp[index][0] = tmp[index][0] + j # j=x axis
				tmp[index][1] = tmp[index][1]+ i	# i=y axis
				tmp[index][2] = tmp[index][2] + 1

		# Assign nodes to list of segment centroids ([(s0_x,s0_y),...,(sn_x,sn_y)]),
		# sorted by occurance of segments in row-wise processing
		
		for i in range(self.numSegments):
			number_points = tmp[i][2]
			if number_points != 0:
				x_center = tmp[i][0] / number_points
				y_center = tmp[i][1] / number_points
				self.nodes.append((x_center,y_center))

	def drawCentroids(self):
		"""
		Draw centroids as red points into the image/plot.
		"""
		
		for (x,y) in self.nodes:
			plt.scatter(x, y, s=2, c='red', marker='o')

	def drawLines(self):
		"""
		Draw lines into plot.
		"""
		
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
		cLen = len(colors)
		i=0
		for ((x1,y1) , (x2,y2)) in self.edges:
			p1 = [x1,x2]
			p2 = [y1,y2]
			
			#Choosing next color
			#plt.plot( p1 , p2 ,marker='o',color= colors[int(i/3)%(cLen-1)] , linewidth=1 , markersize=3 )
			plt.plot( p1 , p2 ,marker='o',color='k' ,linewidth=1,markersize=3 )
			
			i = i +1
			
	def calculateNearestNeighbors(self):
		"""
		For each point of the input list the k-nearest points are found and
		the resulting k lines are saved.
				
		:return: list of line tuples ((x_p1,y_p1),(x_p2,y_p2))
		"""
		
		for (x1,y1) in self.nodes:
			distances = list()
			for (x2,y2) in self.nodes:
				dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
				if dist!= 0:
					distances.append((x2,y2,dist))
			#Sort distance list with ascending order of dist
			distances = sorted( distances, key=lambda x: x[-1] )
			#print("Sorted= ",distances[:10])
			#Pick k elements with minimal distance
			for i in range(self.k):
				#Round/cast to integer points
				p1 = (x1,y1)
				p2 = (distances[i][0],distances[i][1])
				
				self.edges.append((p1,p2))
		
		
	def calcSegmentation(self):
		"""
		Sets the parameters for the segmentation and line element computation 
		and launches the computations. 
		
		:return: (segmentation centroids list, edges list ,mean luminance value list)
		"""
		
		#Calculate SLIC superpixel segmentation
		self.slicSegmentation();
		
		#Calculate centroids of the superpixels
		self.segmentsCenters();
		self.drawCentroids()
		
	    #Nearest neighbour computation for getting edges
		self.calculateNearestNeighbors()
		self.drawLines()
		
		#Get integer point coordinates to match edges aka lines coordinates #TODO: int cast necessary
		#for (x,y) in segmentCentroids:
		#	points.append((int(x),int(y)))
			
		#Determine mean superpixel/segement luminances
		self.meanSegmentLuminance()
		
		return (self.nodes,self.edges,self.segments,self.meanLuminances)
		
	def floodfillImage(self, fillValues):
		"""
		Used to fill the superpixels in the image with the computed values 
		from the optimization algorithm.

		:param:	fillValues	list with segment no. -> greylevel ( 1=white , 0=black)
		
		"""
		
		
		#print("Previous image=\n",self.imageGrey)
		
		imgHeight = len(self.imageGrey)
		imgWidth = len(self.imageGrey[0])
		
		#print("Segementation matrix:",self.segments)
		
		for i in range(imgHeight):
			for j in range(imgWidth):
				segmentNumber = self.segments[i][j]
				self.imageSolution[i][j] = fillValues[segmentNumber]
		
		#print("Resulting image=\n",self.imageSolution)		
		
		# show the output of SLIC
		fig = plt.figure("Superpixels -- %d segments painted" % len(fillValues) )
		plt.imshow(self.imageSolution,cmap='gray',vmin=0,vmax=1)
		
		plt.axis("off")
			
	def meanSegmentLuminance(self):
		"""
		Computes the mean luminance of each segment respectively superpixel.
		
		:return: list of mean greylevel values with index = segement number,
				 sorted by occurance of segments in row-wise processing
		"""
		
		segmentMatrix = self.segments
		
		#(x,y) : x= sum of luminance, y= number of points (in this segment)
		tmp = [[0,0]]* self.numSegments
		print(tmp)
		
		for i in range(len(segmentMatrix)):
			for j in range(len(segmentMatrix)):
				index = segmentMatrix[i][j]
				tmp[index][0] = tmp[index][0] + self.imageGrey[i][j]
				tmp[index][1] = tmp[index][1] + 1
		
		for (sum,count) in tmp:
			self.meanLuminances.append(sum/count)
		
			
	def getPixelValues(self):
		"""
		Computes the greyscale values of each given point and returns these.
			
		:return: list of greyscale values 
		"""
		values = list()
		
		if self.image is None:
			raise ValueError('Image not loaded.')
		
		for p in self.nodes:
			x = p[0]
			y = p[1]
			values.append(self.image[x][y])
		
		return values
		
	def showPlot(self):
		# show the plot
		plt.show()
		#print("Centroids= ",segmentCentroids)
		#print()
