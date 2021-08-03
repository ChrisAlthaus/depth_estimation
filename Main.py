from SuperpixelSegmentation import SuperpixelSegmentation
from QuadraticProblem import QuadraticProblem
from ProblemSolver import ProblemSolver
from EvaluationMetrics import EvaluationMetrics
from HelperFunctions import plotImage,plotImages,plotGraphs,scale

import math
import h5py
import matplotlib.pyplot as plt 

import os
import logging
import argparse
from copy import copy, deepcopy
from skimage.transform import rescale
from skimage import util

logger = logging.getLogger("main_logger")

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)

c_format = logging.Formatter('%(levelname)s:%(message)s')
c_handler.setFormatter(c_format)

logger.addHandler(c_handler)

#f = h5py.File('C:/Program Files/TensorFlow/datasets/nyu_depth_v2_labeled.mat')
f = h5py.File('datasets/nyu_depth_v2_labeled.mat')

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true', help="increase output verbosity")
	return parser.parse_args()

def main():
		args = parse_args()
		
		loglevel = None
		
		if args.verbose:
			loglevel = logging.DEBUG
		else:
			loglevel = logging.INFO
		
		logging.getLogger("main_logger").setLevel(loglevel)
	
		for handler in logging.getLogger("main_logger").handlers:
			handler.setLevel(loglevel)
			
		
		images = f['images'][()]
		depths = f['depths']
		rawDepths = f['rawDepths']
		
		modes = ["SHADE_REFLECT","DEPTH-RECOVERY"]
		evalModes = ["RAW-DEPTH", "NORMAL-DEPTH"]
		
		optMode = modes[1]
		evalMode = evalModes[0]
		
		eval = EvaluationMetrics(evalMode)
		results = list()
		print("Started depth prediction and optimization")
		
		
		#pValues = [10,1,0.1,0.01]
		#bValues = [10,1,0.1,0.01]
		#pValues = [0.001,0.01,0.0001,0.1]
		#bValues = [0.001,0.01,0.0001,0.1]
		pValues = [0.01,0.001]
		bValues = [0.001]
		#pValues = [0.1]
		#bValues = [0.0001]
		
		grid_search = [(x,y) for x in pValues for y in bValues]
		pValues = [ tuple[0] for tuple in grid_search ]
		bValues = [ tuple[1] for tuple in grid_search ]
		
		print("Hyperparameter setting:")
		print("p: ", pValues)
		print("b: ", bValues)
		print("\n")

		k_normal = 2
		p_normal = 0.03
		k_lr = [20,20,15]
		p_lr = [0.1,0.15,0.2]
		
		#optProblem = QuadraticProblem("test",None,None)
		#solver = ProblemSolver(optProblem)
		#solver.testL2TermComputation()
		#exit(1)
		scores = list()
		evalImageNum = 1
		
		test_image = images[evalImageNum].transpose((2,1,0))
		#test_image = images[evalImageNum]
		rescaled_image = util.img_as_ubyte(rescale(test_image, 2, anti_aliasing=False))
		
		"""fig = plt.figure("original image:")
		plt.imshow(test_image,cmap='gray')
		fig.savefig("imgEval/original.png", format='png', dpi=100)
		
		fig = plt.figure("rescaled image:")
		plt.imshow(rescaled_image,cmap='gray')
		fig.savefig("imgEval/rescaled.png", format='png', dpi=100)
		
		print("rescaled image type =", type(rescaled_image[0][0][0]))
		print("original image type =", type(images[evalImageNum][0][0][0]))
		"""
	
		imgList = [rescaled_image] *len(pValues)

		#for i in range(images.shape[0]):
		for i in range(len(imgList)):	
			#image = images[i].transpose((2,1,0))
			#image = imgList[i].transpose((2,1,0))
			image = imgList[i]
			
			groundTruthImage = None
			
			if(evalMode == "RAW-DEPTHS"):
				groundTruthImage = rawDepths[evalImageNum].transpose((1,0))
				groundTruthImage = scale(groundTruthImage,0,1)
				groundTruthImage = util.img_as_ubyte(rescale(groundTruthImage, 2, anti_aliasing=False))

			else: #"NORMAL-DEPTHS"
				groundTruthImage = depths[evalImageNum].transpose((1,0))
				groundTruthImage = scale(groundTruthImage,0,1)
				groundTruthImage = util.img_as_ubyte(rescale(groundTruthImage, 2, anti_aliasing=False))

			print("ground truth image: min = %d max = %d"%(groundTruthImage.min(),groundTruthImage.max()))
			print("Calculation for image %d"%i)
			
			result_dir = os.path.join('imgEval','%d'%i)
			if not os.path.exists(result_dir):
				os.makedirs(result_dir)
			#logger.debug("image %d's type= %s"%(i,type(image[0][0][0])))
			logger.debug("image %d's shape= %s"%(i,image.shape))
			
			#Setup superpixel segmentation: sets number of superpixels 
			#with reference to images width and height
		
			superpixels = SuperpixelSegmentation(image,i, k=k_normal , percentage=p_normal, k_lrange=k_lr, p_lrange= p_lr)
			
		
			#Setup optimization problem parameters, calc superpixels, get edges probalilities
			optProblem = QuadraticProblem(superpixels,i,bValues[i],pValues[i])
			
			# Plot predicted raw weight values on the image for validation
			optProblem.plotRawWeightData()
			
			#superpixels.plotSegmentation()
			# Solve the optimization problem
			solver = ProblemSolver(optProblem)
			
			print("Solving optimization problem..")
			xSolution, err = solver.solveProblemTfAdam(optMode,groundTruthImage)
			scores.append(err)
			print("x_s=",xSolution)
			print("Solving optimization problem done.")
			
			print("Calculation for image %d done."%i)
			logger.info("Calculation for image %d done."%i)
			#Normalize solution
			#xSolution /= xSolution.max()	#needed?
			#xSolution=scale(xSolution, 0, 1)
			
			imageSolution= superpixels.floodfillImage(xSolution)
		
			#Evaluation of solution image
			results.append((imageSolution,groundTruthImage))
			
			imgHeight = len(imageSolution)
			imgWidth = len(imageSolution[0])
			error_score = EvaluationMetrics.validationError(imageSolution,groundTruthImage)
			
			file = open('imgEval/evaluation_statistics.txt', 'a')
			file.write("Image %d \n"%i)
			log_entry = "k_normal: "+str(k_normal) +", p_normal: "+str(p_normal) +", k_lr: "+ str(k_lr) +", p_lr: "+ str(p_lr) + "\n\n"
			file.write(log_entry)
			log_entry = "b: "+str(bValues[i]) +", p: "+str(pValues[i]) +", error_score: "+ str(error_score) + "\n\n"
			file.write(log_entry)
		
		
			plt.axis("off")
			
			if(i%1 ==0):
				fig = plt.figure("ground thruth image")
				ax1 = fig.add_subplot(111)								
				im1 = ax1.imshow(groundTruthImage,cmap='gray')
				fig.colorbar(im1,ax=ax1,shrink=0.5)
				fig.savefig("imgEval/%d/ground_truth_image.png"%i,format='png',dpi=100)
				
				fig = plt.figure("mean luminance image")
				meanImage = superpixels.floodfillImage(superpixels.meanLuminances)
				ax1 = fig.add_subplot(111)								
				im1 = ax1.imshow(meanImage,cmap='gray',vmin=0,vmax=1)
				fig.colorbar(im1,ax=ax1,shrink=0.5)
				fig.savefig("imgEval/%d/image_meanLuminances.png"%i,format='png',dpi=100)
									
				fig = plt.figure("final solution image %d"%i)
				ax1 = fig.add_subplot(111)
				im1 = ax1.imshow(imageSolution,cmap='gray')
				fig.colorbar(im1,ax=ax1,shrink=0.5)
				fig.savefig("imgEval/%d/result_b_s%.4f_p%.4f.png"%(i,bValues[i],pValues[i]),format='png',dpi=100)
			
			print("\n\n")
			plt.close('all')
		
		print("Finished with depth prediction and optimization")
		
		print("Creating error & cost summary plots...")
		#scores[ [ [dict],[[x1,c1],[x2,c2],..],[[x1,e1],[x2,e2],..] ], [ [dict],[[x1,c1],[x2,c2],..],[[x1,e1],[x2,e2],..] ] ]
		print("scores=",scores)
		c_scores = [item[0] for item in scores]
		e_scores = [item[1] for item in scores]
		print("c_scores=",c_scores)
		print("e_scores=",e_scores)
		plotGraphs(c_scores,"cost scores for dif p and b's","cost_scores_overall")
		plotGraphs(e_scores,"error scores for dif p and b's","error_scores_overall")
		print("Creating error & cost summary plots done.")
		
		print("Started Evaluation")
		eval.evaluateMean(results)
		print("Finished Evaluation")
		
		eval.printEvaluations()
		

	
		
if __name__ == "__main__":
	main()
