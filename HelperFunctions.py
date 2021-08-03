
import matplotlib.pyplot as plt 
import numpy as np
import logging

logger = logging.getLogger("main_logger")

def plotImage(image,title):
		"""
		Plots given image.
		"""
		fig = plt.figure(title)
		plt.imshow(image,cmap='gray',vmin=0,vmax=1)
		
		plt.axis("off")
		plt.show()
		
def plotImages(imageSolution,groundTruthImage,superpixels,solver):
	"""
	Helper function to plot calculated images in one iteration for validation of results.
	"""
	
	#Plot solution image
	fig = plt.figure("solution image")
	plt.imshow(imageSolution,cmap='gray')
	plt.colorbar()
	
	#Plot ground truth image
	plt.figure("ground truth image")
	plt.imshow(groundTruthImage,cmap='gray')
	
	#Plot mean luminance image for comparison
	meanImage = superpixels.floodfillImage(superpixels.meanLuminances)
	fig = plt.figure("mean luminances of image")
	plt.imshow(meanImage,cmap='gray',vmin=0,vmax=1)
	
	#Plot learning graph of optimization solving 
	solver.plotLearningGraph()
	#Plot superpixel visualization
	superpixels.plotSegmentation()
	superpixels.plotEdges()
	
	plt.show()
	plt.close('all')
	
	
def saveImage(image,title,file_name):
	"""
	Saves given image.
	"""
	"""print("Saving image")
	fig = plt.figure(title)
	fig.figImage(image)
	fig.savefig("imgEval/%s.png"%title, format='png', dpi=100)"""
	
	
	fig = plt.figure(title)
	ax1 = fig.add_subplot(111)
	logger.debug("min of saved image = %f, max of saved image= %f"%(image.min(),image.max()))
	im1 = ax1.imshow(image,cmap='gray') #,vmin=0,vmax=1)
	fig.colorbar(im1,ax=ax1,shrink=0.5)
	fig.savefig("imgEval/%s.png"%file_name, format='png', dpi=100)
	fig.clf()
	
	print("Saving image done.")

def scale(image, newMin, newMax):
	"""
	Scales an image (numpy array) to the specified min and max.
	"""
	
	return (( (image- image.min())* (newMax -  newMin) )/(image.max()-image.min())) + newMin

def plotGraph(point_array,title,file_name):
	"""
	Plots 1 graph into one plot.
	"""
	fig = plt.figure(title)
	print(point_array)
	x_values = np.asarray([p[0] for p in point_array])
	y_values = np.asarray([p[1] for p in point_array])
	
	plt.plot(x_values,y_values)
	fig.savefig("imgEval/%s.png"%file_name, format='png', dpi=100)
	fig.clf()

def plotGraphs(data_array,title,file_name):
	"""
	Plots n graphs into one plot.
	params: 	data_array  list of shape [ [[p,b],[[X,Y]]] ,..,[[p,b],[[X,Y]]]]
	"""
	fig = plt.figure(title)
	print("data array=",data_array)

	for graph_data in data_array:
		params = graph_data[0]
		points = graph_data[1:-1]
		print("points=",points)
		x_values = np.asarray([p[0] for p in points])
		y_values = np.asarray([p[1] for p in points])
		
		plt.plot(x_values,y_values,label="p=%.04f b=%.04f"%(params['p'],params['b']))
	plt.legend(loc='upper left')
	fig.savefig("imgEval/%s.png"%(file_name), format='png', dpi=100)
	plt.close(fig)

