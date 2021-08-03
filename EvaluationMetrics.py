import math
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import logging

import h5py

logger= logging.getLogger("main_logger")

class EvaluationMetrics:
	"""
	Metrics for evaluation between solved image and ground truth image.
	
	Formulas from the papers:
	1.	Deep Convolutional Neural Fields for Depth Estimation from a Single ImageFayao by
		Liu, Chunhua Shen, Guosheng Lin
	2.	Depth Map Prediction from a Single Imageusing a Multi-Scale Deep Network by
		David Eigen, Christian Puhrsch, Rob Fergus
	"""
	
	#Either for comparison with ground truth raw or normal(edited) depth images
	evaluationMode = None
	
	#Dataset for computation of global min and max depth/raw depth
	#f = h5py.File('C:/Program Files/TensorFlow/datasets/nyu_depth_v2_labeled.mat')
	f = h5py.File('nyu_depth_v2_labeled.mat')
	
	#Min and max depths of the dataset
	minRD = None
	maxRD = None
	minD = None
	maxD = None
	
	#Evaluation metrics over all processed images
	absRelMean = 0.0
	sqrRelMean = 0.0
	rmseLinearMean = 0.0
	rmseLogMean = 0.0
	
	#Evaluation metrics for current processed image
	absRel = 0.0
	sqrRel = 0.0
	rmseLinear = 0.0
	rmseLog = 0.0
	
	#Used to scale both calulated and ground truth image to equal interval
	scale_min = 1e-9
	
	def __init__(self,evalMode=None):
		self.evaluationMode = evalMode
		#self.computeScaleRanges()

	def computeScaleRanges(self): #deprecated/ not used
		"""
		Compute the min and max depth/raw depths over all images in the dataset.
		"""
		
		logger.info("Computing min and max depth/raw depths over all images in the dataset...")
		rawDepths = self.f['rawDepths']
		
		minRD = 1000
		maxRD = 0
		count_zero_rd = 0

		for i in range(rawDepths.shape[0]):
			tmpMin = rawDepths[i].min()
			if(tmpMin==0):
				count_zero_rd = count_zero_rd +1
			if(tmpMin<minRD):
				minRD = tmpMin
			tmpMax = rawDepths[i].max()
			if(tmpMax>maxRD):
				maxRD = tmpMax
		
		self.minRD = minRD
		self.maxRD = maxRD
		
		logger.debug("number img with min = 0 : %d"%count_zero_rd)
		logger.debug("minD = %d"% minRD)
		logger.debug("maxD = %d"% maxRD)	
		
		depths = self.f['depths']
		
		minD = 1000
		maxD = 0
		count_zero_d = 0

		for i in range(depths.shape[0]):
			tmpMin = depths[i].min()
			if(tmpMin==0):
				count_zero_d = count_zero_d +1
			if(tmpMin<minD):
				minD = tmpMin
			tmpMax = depths[i].max()
			if(tmpMax>maxD):
				maxD = tmpMax	
				
		self.minD = minD
		self.maxD = maxD
		
		logger.debug("number img with min = 0 : %d"%count_zero_d)
		logger.debug("minD = %d"% minD)
		logger.debug("maxD = %d"% maxD)
	
		
	def scale(self, image, newMin, newMax):
		"""
		Scales an image (numpy array) to the specified min and max.
		"""
		
		"""scaler = MinMaxScaler(feature_range=(min, max))
		scaler = scaler.fit(self.image)
		self.imageScaledD = scaler.transform([self.image])"""
		
		return (( (image- image.min())* (newMax -  newMin) )/(image.max()-image.min())) + newMin
		
	def evaluateMean(self,results):
		"""
		Create evaluation statistics for input list of pairs(image, groundTruth).
		"""
		absRelSum = 0.0
		sqrRelSum = 0.0
		rmseLinearSum = 0.0
		rmseLogSum = 0.0
		
		numImages = len(results)
		
		logger.info("Evaluation mode= %s"%self.evaluationMode)
		
		for (image, imgGroundTruth) in results:
			(absRel,sqrRel,rmseLinear,rmseLog) = self.evaluate(image,imgGroundTruth)
			
			absRelSum = absRelSum + absRel
			sqrRelSum = sqrRelSum + sqrRel
			rmseLinearSum = rmseLinearSum + rmseLinear
			rmseLogSum = rmseLogSum + rmseLog
			
			print(absRelSum,sqrRelSum,rmseLinearSum,rmseLogSum)
		
		print("numImages=",numImages)
		print("absRelSum=",absRelSum)
		print("sqrRelSum=",sqrRelSum)
		print("rmseLinearSum=",rmseLinearSum)
		print("rmseLogSum=",absRelSum)
		self.absRelMean = absRelSum/numImages
		self.sqrRelMean = sqrRelSum/numImages
		self.rmseLinearMean = rmseLinearSum/numImages
		self.rmseLogMean = rmseLogSum/numImages
			
	
	def evaluate(self,image,imgGroundTruth):
		"""
		Computes either raw depth or depth evaluation for one image pair.
		"""
		absRel = 0.0
		sqrRel = 0.0
		rmseLinear = 0.0
		rmseLog = 0.0
		
		
		if(self.evaluationMode == "RAW-DEPTH"):
			(absRel,sqrRel,rmseLinear,rmseLog) = self.evaluateRawDepths(image,imgGroundTruth)
			return (absRel,sqrRel,rmseLinear,rmseLog)
		elif(self.evaluationMode == "NORMAL-DEPTH"):
			(absRel,sqrRel,rmseLinear,rmseLog) = self.evaluateDepths(image,imgGroundTruth)
			return (absRel,sqrRel,rmseLinear,rmseLog)
		else:
			raise ValueError("No valid depth evaluation mode.")
	
	
	def evaluateRawDepths(self,image,imgRawDepths):
		"""
		Computes the evaluations with comparison to the corresponding raw depth image.
		Before both images are scaled.
		Then different metrics are computed.
		"""
	
		image = self.scale(image,self.scale_min, 1)
		
		counter = 0
		for i in range(len(image)):
			for j in range(len(image[0])):
				if(image[i][j] == 0):
					counter = counter + 1
		logger.debug("number zeros=%d"%counter)
		
		logger.debug( "new min of calculated image= %.10f, max = %.10f"%(image.min(), image.max()) )
		
		# Scale groundtruth image to [0,1], because invalid points (depth = 0)
		# shouldn't be considered for evaluation.
		imgGroundTruthRD = self.scale(imgRawDepths,0,1)
		logger.debug( "new min of ground truth raw depth image= %.10f, max = %.10f"%(imgGroundTruthRD.min(),imgGroundTruthRD.max()) )
	
		
		absRel = self.absRel(image,imgGroundTruthRD)
		sqrRel = self.sqrtRel(image,imgGroundTruthRD)
		rmseLinear = self.rmseLinear(image,imgGroundTruthRD)
		rmseLog = self.rmseLog(image,imgGroundTruthRD)
			
		return (absRel,sqrRel,rmseLinear,rmseLog)
		
		
	def evaluateDepths(self,image,imgDepths):
		"""
		Computes the evaluations with comparison to the corresponding normal depth image.
		Before both images are scaled.
		Then different metrics are computed.
		"""
		image = self.scale(image,self.scale_min, 1)
		counter = 0
		for i in range(len(image)):
			for j in range(len(image[0])):
				if(image[i][j] == 0):
					counter = counter + 1
		logger.debug("number zeros=%d"%counter)
		
		logger.debug( "new min of calculated image= %f, max = %f"%(image.min(), image.max()) )
		print(image.min(),image.max())
		
		# Scale groundtruth image to [0,1], because invalid points (depth = 0)
		# shouldn't be considered for evaluation.
		imgGroundTruthD = self.scale(imgDepths,0,1)
		logger.debug( "new min of ground truth raw depth image= %f, max = %f"%(imgGroundTruthD.min(),imgGroundTruthD.max()) )
		print(imgGroundTruthD.min(),imgGroundTruthD.max())	
		
		absRel = self.absRel(image,imgGroundTruthD)
		sqrRel = self.sqrtRel(image,imgGroundTruthD)
		rmseLinear = self.rmseLinear(image,imgGroundTruthD)
		rmseLog = self.rmseLog(image,imgGroundTruthD)
			
		return (absRel,sqrRel,rmseLinear,rmseLog)
		
	def printEvaluations(self):
		"""
		Prints the mean absRel, sqrRel, RMSLin and RMSLog( 0 if raw depths) scores.
		"""
		print("Evaluation mean statistics:")
		print(" absRel = %f \n sqrRel = %f \n RMSLin = %f \n RMSLog = %f"\
			%(self.absRelMean, self.sqrRelMean, self.rmseLinearMean, self.rmseLogMean)	)
			
		"""logger.info("Evaluation mean statistics:")
		logger.info(" absRel = %f \n sqrRel = %f \n RMSLin = %f \n RMSLog = %f"\
			%(self.absRelMean, self.sqrRelMean, self.rmseLinearMean, self.rmseLogMean)	)"""

	def validationError(image,imageGroundTruth,type='sqrt'):
	    
		width = len(image[0])
		height = len(image)
		
		def scale(image, newMin, newMax):
			return (( (image- image.min())* (newMax -  newMin) )/(image.max()-image.min())) + newMin
			
		imageMin = image.min()
		imageMax = image.max()
		imageTMin = imageGroundTruth.min()
		imageTMax = imageGroundTruth.max()
		
		if(imageMin<1e-4 or imageMin>1 or imageMax<1e-4 or imageMax>1):
			image = scale(image,1e-4,1)
		if(imageTMin<0 or imageTMin>1 or imageTMax<0 or imageTMax>1):
			logger.debug("Image ground truth depth not in range 0-1. Setting range..")
			imageGroundTruth = scale(image,0,1)
		
		if(type=='abs'):
			absError = 0
			numPixels = 0
			for i in range(height):
				for j in range(width):
					if(image[i][j] == 0):
						raise ValueError('Image depth = 0 not allowed.')
					if(imageGroundTruth[i][j] == 0): #depth=0 -> no depth information available
						continue
					absError = absError + (abs(image[i][j] - imageGroundTruth[i][j]) / imageGroundTruth[i][j])
					numPixels = numPixels + 1
			return (1/numPixels) * absError
		elif(type=='sqrt'):
			se = 0	#squared error
			numPixels = 0
		
			for i in range(height):
				for j in range(width):
					if(image[i][j] == 0):
						raise ValueError('Image depth = 0 not allowed.')
					if(imageGroundTruth[i][j] == 0): #depth=0 -> no depth information available
						continue
					se = se + math.pow(image[i][j] - imageGroundTruth[i][j],2)/imageGroundTruth[i][j]
					numPixels = numPixels + 1
			return (1/numPixels) * se
		else:
			raise ValueError("Please choose either abs or sqrt validation error metric.")
		

			
	def absRel(self,image,imageGroundTruth):
		"""
		Normalized sum of absolute errors aka absolute relative difference (paper: abs rel).
		Pixel with raw depth = 0, indicates no valid depth (NYU specification).
		Formula: 1/T * |D*-D|/D* .
		
		:return: error value(lower->better)
		"""
		#print("absRel")
		width = len(image[0])
		height = len(image)
		
		absError = 0
		numPixels = 0
		
		for i in range(height):
			for j in range(width):
				if(image[i][j] == 0):
					raise ValueError('Image depth = 0 not allowed.')
				if(imageGroundTruth[i][j] == 0): #depth=0 -> no depth information available
					continue
				absError = absError + (abs(image[i][j] - imageGroundTruth[i][j]) / imageGroundTruth[i][j])
				numPixels = numPixels + 1
		
		return (1/numPixels) * absError
		
		
	def sqrtRel(self,image,imageGroundTruth):
		"""
		Normalized sum of squared errors aka squared relative difference (paper: sqr rel).
		For raw depth ground truth image (raw depth range = [0,max]).
		Pixel with raw depth = 0, indicates no valid depth (NYU specification).
		Formula: 1/T * |D-D*|^2 /D* .
		
		:return: error value (lower->better)
		"""
		#print("sqrtRel")
		width = len(image[0])
		height = len(image)
		
		se = 0	#squared error
		numPixels = 0
		
		for i in range(height):
			for j in range(width):
				if(image[i][j] == 0):
					raise ValueError('Image depth = 0 not allowed.')
				if(imageGroundTruth[i][j] == 0): #depth=0 -> no depth information available
					continue
				
				se = se + math.pow(image[i][j] - imageGroundTruth[i][j],2)/imageGroundTruth[i][j]
				numPixels = numPixels + 1
				#if(j%10000 == 0):
				#	print(math.pow(image[i][j] - imageGroundTruth[i][j],2)/imageGroundTruth[i][j])
		#print("numPixels=",numPixels)
		return (1/numPixels) * se
		
	
	def rmseLinear(self,image,imageGroundTruth):
		"""
		Root mean squared error linear (paper: RMS(lin)).
		Pixel with raw depth = 0, indicates no valid depth (NYU specification).
		Formula: sqrt( 1/T * (D-D*)^2).
		
		:return: error value (lower->better)
		"""
		#print("rmseLinear")
		width = len(image[0])
		height = len(image)
		
		se = 0	#squared error
		numPixels = 0
		
		for i in range(height):
			for j in range(width):
				if(image[i][j] == 0):
					raise ValueError('Image depth = 0 not allowed.')
				if(imageGroundTruth[i][j] == 0): #depth=0 -> no depth information available
					continue
				se = se + math.pow(imageGroundTruth[i][j] - image[i][j],2)
				#if(j%10000 == 0):
				#	print(math.pow(imageGroundTruth[i][j] - image[i][j],2))
				numPixels = numPixels + 1
				
		return math.sqrt( (1/numPixels) * se)
	
	def rmseLog(self,image,imageGroundTruth):
		"""
		Root mean squared error logaritmic (paper: RMS(log)).
		For normal depth ground truth image.
		To evaluated image should be scaled before.
		Formula: sqrt( 1/T * (log(D)-log(D*))^2).
		
		:return: error value (lower->better)
		"""
		#print("rmseLog")
		width = len(image[0])
		height = len(image)
		
		se = 0	#squared error
		numPixels = 0
		
		for i in range(height):
			for j in range(width):
				if(image[i][j] == 0):
					raise ValueError('Image depth = 0 not allowed.')
				if(imageGroundTruth[i][j] == 0):
					continue
				se = se + math.pow( math.log(imageGroundTruth[i][j],10) - math.log(image[i][j] ,10), 2)	
				#if(j%10000 == 0):
				#	print(math.pow( math.log(imageGroundTruth[i][j],10) - math.log(image[i][j] ,10), 2))
				numPixels = numPixels + 1
				
		return math.sqrt( (1/numPixels) * se)


