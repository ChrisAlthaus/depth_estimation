# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import math
 


def getSlicSegmentation(imagePath, numberSegments):
	"""
	Get the Segmentation as a matrix with height and width of the image.
	Matrix entries are the number of the segementation.
	
	:param imagePath: the path to the image
	:return: segementation matrix 
	"""
	
	# load the image and convert it to a floating point data type
	image = img_as_float(io.imread(imagePath))
	
	# loop over the number of segments
	
	# apply SLIC and extract (approximately) the supplied number of segments
	segments = slic(image, n_segments = numberSegments, sigma = 5)
	 
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numberSegments))
	plt.imshow(mark_boundaries(image, segments))
	
	plt.axis("off")
	
	# Printing for validation
	for i in range(len(segments)):
		print("segements list[",i,"] = " , segments[i][:24]);
	
	
	return segments;

def getSegmentsCenters(segment_matrix,numSegments):
	"""
	Returns a list of the segments centroids, with data from the segment_matrix.
	Centroids coordinates are computed with the sum of the values of points in x- or y-direction
	of one segmentnumber divided by the number of points in this segement.
	
	:param 	segment_matrix: Segementation matrix [heigth][width]
			numSegments: Number of segments in the matrix
	:return: List of segement centroids ([(s0_x,s0_y),...,(sn_x,sn_y)])
	"""
	segment_centroids = list()
	
	#Temp list to compute the centroids [(s0_sumx,s0_sumy,s0_no_pixels),...,(sn_sumx,sn_sumy,sn_no_pixels)]
	#tmp = [[0,0,0]] * numSegments
	tmp = list()
	
	for i in range(numSegments):
		tmp.append([0,0,0])

	for i in range(len(segment_matrix)):
		for j in range(len(segment_matrix[i])):
			index = segment_matrix[i][j] #number of the segement
			
			tmp[index][0] = tmp[index][0] + j # j=x axis
			tmp[index][1] = tmp[index][1]+ i	# i=y axis
			tmp[index][2] = tmp[index][2] + 1

	
	for i in range(numSegments):
		number_points = tmp[i][2]
		if number_points != 0:
			x_center = tmp[i][0] / number_points
			y_center = tmp[i][1] / number_points
			segment_centroids.append((x_center,y_center))
	
	return segment_centroids

def drawCentroids(segmentCentroids):
	"""
	Draw centroids as red points into the image/plot.
	"""
	
	for (x,y) in segmentCentroids:
		plt.scatter(x, y, s=2, c='red', marker='o')

def drawLines(lines):
	"""
	Draw lines into plot.
	"""
	
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
	cLen = len(colors)
	i=0
	for ((x1,y1) , (x2,y2)) in lines:
		p1 = [x1,x2]
		p2 = [y1,y2]
		
		#Choosing next color
		#plt.plot( p1 , p2 ,marker='o',color= colors[int(i/3)%(cLen-1)] , linewidth=1 , markersize=3 )
		plt.plot( p1 , p2 ,marker='o',color='k' ,linewidth=1,markersize=3 )
		
		i = i +1
		
def calculateNearestNeighbors(points, k):
	"""
	For each point of the input list the k-nearest points are found and
	the resulting k lines are saved. 
	
	:param:	points list of points (x,y)
			k number of nearest points
			
	:return: list of line tuples ((x_p1,y_p1),(x_p2,y_p2))
	"""
	#print("points = ",points )
	lines= list()
	for (x1,y1) in points:
		distances = list()
		for (x2,y2) in points:
			dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
			if dist!= 0:
				distances.append((x2,y2,dist))
		#Sort distance list with ascending order of dist
		distances = sorted( distances, key=lambda x: x[-1] )
		#print("Sorted= ",distances[:10])
		#Pick k elements with minimal distance
		for i in range(k):
			#Round/cast to integer points
			p1 = (int(x1),int(y1))
			p2 = (int(distances[i][0]),int(distances[i][1]))
			
			lines.append((p1,p2))
		
	#print("Lines= ")
	#for i in lines:
	#	print(i)
	#print()
	
	return lines
	
def calcSegmentation(imagePath):
	"""
	Sets the parameters for the segmentation and line element computation 
	and launches the computations.
	
	:return: (segmentation centroids list, edges list )
	"""
	numberSegments = 300
	segments = getSlicSegmentation(imagePath,numberSegments);
	
	segmentCentroids = getSegmentsCenters(segments,numberSegments);
	drawCentroids(segmentCentroids)
	
	#Choose k nearest neighbours for edge computation
	k=3
	lines = calculateNearestNeighbors(segmentCentroids, k)
	drawLines(lines)
	
	#Get integer point coordinates
	points = list()
	for (x,y) in segmentCentroids:
		points.append((int(x),int(y)))
		
		
	return (points,lines)

def getPixelValues(points,imagePath):
	"""
	Computes the greyscale values of each given point and returns these.
	
	:param: centroids list of centroids in float format
			imagePath local path to image
			
	:return: list of greyscale values 
	"""
	values = list()
	# load the image and convert it to a greyscale image
	image = io.imread(imagePath, as_gray=True)
	
	
	for p in points:
		x = p[0]
		y = p[1]
		values.append(image[x][y])
	
	#print("Integer Points=", points)
	print("Image=",image)
	
	return values
	
def showPlot():
	# show the plot
	plt.show()
	#print("Centroids= ",segmentCentroids)
	#print()
