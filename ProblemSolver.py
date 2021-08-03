	
import math
import numpy as np
import time
from QuadraticProblem import QuadraticProblem
import tensorflow as tf
from HelperFunctions import saveImage, plotGraph, scale
from EvaluationMetrics import EvaluationMetrics

import matplotlib.pyplot as plt

import logging

logger = logging.getLogger("main_logger")

class ProblemSolver:
	"""
	This class specifies the fitness function for an easy to use interface.
	This class allows to solve the optimization problem with the tensorflow framework.
	
	Attributes:
		problem 	Reference to the relevant optimization problem
	"""
	
	def __init__(self, problem):
		#Reference to QuadraticProblem 
		self.problem = None
		
		#Weights for L2 distance terms
		self.lamb_eq = None	
		self.lamb_gt = None
		self.lamb_lt = None
		self.lamb_s = None
		
		#used to plot fitness over each iteration
		self.graphFitnessIteration = list()
	
		if problem is None:
			raise ValueError('Optimization problem not initialized! Please first initialize.')
		
		self.problem = problem
		
		#for debugging outputs
		self.tmpIndex = problem.imageNumber

	def getL2Term(self,xVector,R_xy,A_xy,W_xy):
		"""
		Returns L2 distance term as a tensor.
		Matrix-multiplication from right to left.
		Resulting formula= [x,R_gt]^T * ( A_gt^T * ( W_gt * ( A_gt * [x,R_gt] )))
		
		:param: xVector greyscale values of nodes
				R_xy 	slack variable for each edge (GT,LT or EQ)
				A_xy	multiplication help matrix
				W_xy	weights matrix (GT,LT or EQ)
				
		:return: L2 distance term (tensor)
		"""
		
		#xR_xy : |N|+|E| x 1 (vector?)
		#xVector size = |N| , R_xy size = |E|
		#print("x= \n" , xVector)
		#print("R_xy \n=" , R_xy)
		#print("dimensions x=" , xVector.shape)
		#print("dimensions R_xy=" , R_xy.shape)
	
		#Concat into one vector and convert into vector shape
		xR_xy = tf.concat([xVector,R_xy],0)
		xR_xy = tf.transpose([xR_xy])
		
		#Printing matrices and dimensions
		#print()
		#print("x= \n" , xVector)
		#print("R_xy \n=" , R_xy)
		#print("->xR_xy= \n" , xR_xy)
		#print("A_xy= \n" , A_xy)
		#print("W_xy= \n" , W_xy)
		
		#print("dimensions x=" , xVector.shape)
		#print("dimensions R_xy=" , R_xy.shape)
		#print("dimensions xR_xy=" , xR_xy.shape)
		#print("dimensions A_xy=" , A_xy.shape)
		#print("dimensions W_xy=" , W_xy.shape)
		
		#Computing dot products
		result = tf.matmul(A_xy,xR_xy)
		#print("A * xR=" , result)
		#print("dimensions A * xR=" , result.shape)
		
		result = tf.matmul(W_xy,result)
		#print("W * A * xR=" , result)
		#print("dimensions W * A * xR=" , result.shape)
		
		result = tf.matmul(tf.transpose(A_xy),result)
		#print("A^T * W * A * xR=" , result)
		#print("dimensions A^T * W * A * xR=" , result.shape)
		
		result = tf.matmul(tf.transpose(xR_xy),result)
		#print("xR^T * A^T * W * A * xR=" , result)
		#print("dimensions xR^T * A^T * W * A * xR=" , result.shape)
		#print("->xR=" , xR_xy.transpose())
		#print("result=",result)
		#print()
		
		#Convert into a 1-D shape respectively a single number
		result = tf.reshape(result,[-1])
		
		return result
		
	def getSmoothTerm(self,x,A_s,W_s,b_s,mode):
		"""
		Returns the smoothness term as a tensor representation. Allows to enforce smoothness in the image.
		Matrix-multiplication from right to left.
		Resulting formula= x^T * ( A_s^T * ( W_s * ( A_s * x ))) + x^T * b_s
		
		:param:	x 		greyscale values of nodes
				A_s 	multiplication help matrix
				W_s 	matrix of smoothness weights
				b_s		vector (format = list) for more flexible smoothing, size=|N|
				mode  	shading/reflectance or depth recovery
				
		:return smoothness term in tensor representation
		"""
	
		#Convert into vector shape
		x= tf.transpose([x])
		b_s = tf.transpose([b_s])
		
		result = tf.matmul(A_s,x)
		#print("dimensions A_s * x=" , result.shape)
		
		result = tf.matmul(W_s,result)
		#print("dimensions W_s * A_s * x=" , result.shape)
		
		result = tf.matmul(tf.transpose(A_s),result)
		#print("dimensions A_s^T * W_s * A_s * x=" , result.shape)
		
		result = tf.matmul(tf.transpose(x),result)
		#print("dimensions x^T * A_s^T * W_s * A_s * x=" , result.shape)
		
		if(mode=="SHADE_REFLECT"):
			print("SHADE_REFLECT: Adding b_s to smoothness term")
		result = tf.add(result, tf.matmul(tf.transpose(x),b_s))
		
		#print("dimensions A_s^T * W_s * A_s * x + x^T * b_s=" , result.shape)
		
		#Convert into a 1-D shape respectively a single number
		result = tf.reshape(result,[-1])
		
		return result
		
	def getSlackTerm(self,R_gt,R_lt,R_eq,mode):
		"""
		Returns the sum of slack terms as a tensor representation.
		
		:param:	R_gt	slack variables for greater than computations, size |E|x1
				R_lt 	slack variables for less than computations, size |E|x1
				R_eq	slack variables for equals than computations, size |E|x1
				mode  	shading/reflectance or depth recovery
				
		:return sum of slack terms in tensor representation
		"""
		
		stdDevGT= None
		stdDevLT= None
		stdDevEQ= None
		
		meanGT= tf.constant(math.log(2,10),dtype=tf.float32)
		meanLT= tf.constant(math.log(2,10),dtype=tf.float32)
		meanEQ= tf.constant(0,dtype=tf.float32)
		
		if(mode=="SHADE_REFLECT"):
			stdDevGT= tf.constant(0.001,dtype=tf.float32)
			stdDevLT= tf.constant(0.001,dtype=tf.float32)
			stdDevEQ= tf.constant(0.001,dtype=tf.float32)
		elif(mode=="DEPTH-RECOVERY"):
			stdDevGT= tf.constant(4,dtype=tf.float32)
			stdDevLT= tf.constant(4,dtype=tf.float32)
			stdDevEQ= tf.constant(0.1,dtype=tf.float32)
		else:
			raise ValueError('Please select a valid mode/target.')
			
		#Computing the sum in the second section of the formula
		slackTerm = 0
		#print(R_eq.shape[0])
		numEdges = R_eq.shape[0]
		
		#for i in range(numEdges):
			
		#	tmp = tf.add_n( [tf.divide( tf.pow(R_eq[i],2.0), tf.pow(stdDevEQ,2.0) ),
		#						  tf.divide( tf.pow( tf.subtract(R_gt[i],meanGT), 2 ), tf.pow(stdDevGT,2) ),
		#				          tf.divide( tf.pow(tf.subtract(R_lt[i],meanLT), 2 ),tf.pow(stdDevLT, 2))] )
								  
						
			#slackTerm = tf.add(tmp ,slackTerm)
		slackTerm = tf.reduce_sum( tf.add_n( [tf.divide( tf.pow(R_eq,2.0), tf.pow(stdDevEQ,2.0) ),
											tf.divide( tf.pow( tf.subtract(R_gt,meanGT), 2 ), tf.pow(stdDevGT,2) ),
											tf.divide( tf.pow(tf.subtract(R_lt,meanLT), 2 ),tf.pow(stdDevLT, 2))] )	)
		return slackTerm
		
	def getFitnessFunction(self,x,R_gt,R_lt,R_eq,mode):
		"""
		Returns the final fitness function (aka optimization function) in tensor representation.
		
		:param:	x 		greyscale values of nodes
				R_gt	slack variables for greater than computations, size |E|x1
				R_lt 	slack variables for less than computations, size |E|x1
				R_eq	slack variables for equals than computations, size |E|x1
				mode  	shading/reflectance or depth recovery
				
		:return smoothness term in tensor representation
		"""
		logger.debug("Load loss terms..")
		L_eq = self.getL2Term(x ,R_eq ,self.problem.A_eq ,self.problem.W_eq )
		L_gt = self.getL2Term(x ,R_gt ,self.problem.A_gt ,self.problem.W_gt )
		L_lt = self.getL2Term(x ,R_lt ,self.problem.A_lt ,self.problem.W_lt )
		logger.debug("Load loss terms done.")
		
		logger.debug("Load b_s..")
		b_s = tf.convert_to_tensor(self.problem.b_s ,dtype=tf.float32) #todo: how to set b_s
		logger.debug("Load b_s done.")
		
		logger.debug("Load L_s ..")
		L_s = self.getSmoothTerm(x ,self.problem.A_s ,self.problem.W_s, b_s,"DEPTH-RECOVERY")
		logger.debug("Load L_s done.")	
		
		L2Term = tf.add_n([tf.multiply(self.lamb_eq, L_eq), tf.multiply(self.lamb_gt, L_gt), 
							   tf.multiply(self.lamb_lt, L_lt), tf.multiply(self.lamb_s,L_s)])
		#L2Term = tf.add_n([tf.multiply(self.lamb_gt, L_gt), tf.multiply(self.lamb_lt, L_lt), tf.multiply(self.lamb_s,L_s)])

		"""
		if(mode=="SHADE_REFLECT"):
			L_s = self.getSmoothTerm(x ,self.problem.A_s ,self.problem.W_s, b_s)
			L2Term = tf.add_n([tf.multiply(self.lamb_eq, L_eq), tf.multiply(self.lamb_gt, L_gt), 
							   tf.multiply(self.lamb_lt, L_lt), tf.multiply(self.lamb_s,L_s)])
		elif(mode=="DEPTH-RECOVERY"): # no smoothness term( paper p.6: "Unlike in the case of reflectance estimation,
									  #					there is no constraint on the inequality magnitudes for depth differences" ?!)
			L_s = self.getSmoothTerm(x ,self.problem.A_s ,self.problem.W_s, b_s)
			L2Term = tf.add_n([tf.multiply(self.lamb_eq, L_eq), tf.multiply(self.lamb_gt, L_gt), 
							   tf.multiply(self.lamb_lt, L_lt), tf.multiply(self.lamb_s,L_s)])
		else:
			raise ValueError('Please select a valid mode/target.')
		"""	
		
		logger.debug("Load slack term..")
		slackTerm = self.getSlackTerm(R_gt,R_lt,R_eq,mode)
		logger.debug("Load slack term done.")
		
		f = tf.add( L2Term, slackTerm)
		
		return f
		
	
	def ClipIfNotNone(self,grad):
		"""
		Used to clip the gradients to prevent too strong drifting in one direction.
		"""
		if grad is None:
			return grad
		return tf.clip_by_value(grad, -400, 400)
	
	def setResultMode(self,mode):
		"""
		Sets the weights for the L2 terms with reference of the objective.
		
		:param: mode  shading/reflectance or depth recovery
		"""
		if(mode=="SHADE_REFLECT"):
			self.lamb_eq = tf.constant(5,dtype=tf.float32)	
			self.lamb_gt = tf.constant(1,dtype=tf.float32)
			self.lamb_lt = tf.constant(1,dtype=tf.float32)
			self.lamb_s = tf.constant(0.5,dtype=tf.float32)
		elif(mode=="DEPTH-RECOVERY"):
			self.lamb_eq = tf.constant(1,dtype=tf.float32)	
			self.lamb_gt = tf.constant(1,dtype=tf.float32)
			self.lamb_lt = tf.constant(1,dtype=tf.float32)
			self.lamb_s = tf.constant(10,dtype=tf.float32)
		else:
			raise ValueError('Please select a valid mode/target.')
			
	def getConstraints(self,mode):
		"""
		Returns the upper and lower bound with reference to the given mode/objective.
		
		:param: mode  shading/reflectance or depth recovery
		
		:return: (lower_bound, upper_bound)
		"""
		if(mode=="SHADE_REFLECT"):
			return (-np.infty,0)
		elif(mode=="DEPTH-RECOVERY"):
			return (0,10)
		else:
			raise ValueError('Please select a valid mode/target.')
		
	def solveProblemTfAdam(self,mode, validImage):
		"""
		Solves the optimization problem with the adam optimizer.
		
		:param: mode  shading/reflectance or depth recovery
		
		:return: result of optimization, x numpy array with greyscale values
		"""
		tf.reset_default_graph()
		
		#Begin with x as the superpixels mean luminaces and
		#define constraints for intervall [-inf,0] for log reflectance
		x_shape = tf.placeholder(tf.float32, shape=(self.problem.superpixels.numSrSegments))
		lower_bound, upper_bound = self.getConstraints(mode)
		x = tf.Variable(initial_value= x_shape,
						constraint=lambda x: tf.clip_by_value(x, lower_bound, upper_bound),
						dtype=tf.float32)
						
		(R_gt,R_lt,R_eq) = self.problem.getSlackVariables(mode)	
		logger.debug("data type(R_gt)=%s"%R_gt.dtype.name)
		logger.debug("data type(R_lt)=%s"%R_lt.dtype.name)
		logger.debug("data type(R_eq)=%s"%R_eq.dtype.name)
		
		if(mode=="SHADE_REFLECT"):
			R_gt =  tf.Variable(initial_value=R_gt ,dtype=tf.float32,	#float32 better?
								constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
			R_lt = tf.Variable(initial_value=R_lt ,dtype=tf.float32,
								constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
			R_eq = tf.Variable(initial_value=R_eq ,dtype=tf.float32,
								constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
		elif(mode=="DEPTH-RECOVERY"): # no inequality magnitudes ?! -> paper p.6 
			R_gt =  tf.Variable(initial_value=R_gt ,dtype=tf.float32)
			R_lt = tf.Variable(initial_value=R_lt ,dtype=tf.float32)
			R_eq = tf.Variable(initial_value=R_eq ,dtype=tf.float32,
								constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
		else:
			raise ValueError('Please select a valid mode/target.')
		
		
		logger.debug("R_tensors loaded")
		
		self.setResultMode(mode)		
		logger.debug("Compute fitness function..")
		f = self.getFitnessFunction(x,R_gt,R_lt,R_eq,mode)
		logger.debug("Compute fitness function done.")
		
		adam = tf.train.AdamOptimizer(learning_rate=0.03)
		
		train = adam.minimize(f, var_list=[x,R_gt,R_lt,R_eq])

		
		#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		sess = tf.Session()
		
		logger.info("init variables... ")
		init = tf.global_variables_initializer()
		logger.info("init variables done. ")
		
		#Replace placeholder with mean luminaces for tensors > 2GB
		sess.run(init, feed_dict={x_shape: self.problem.superpixels.meanLuminances})
		
		logger.info("Solving Optimization Problem for %s"% mode)
		
		iterationNumber = 6000
		errors = 0
		
		logger.debug("Number of iterations of optimizer= %d"%iterationNumber)
		tf.get_default_graph().finalize()
		logger.debug("Graph finalized.")
		#tf.train.start_queue_runners(sess)
		
		imgHeight = len(validImage)
		imgWidth = len(validImage)
		
		validImage = scale(validImage,0,1)
		#for plot and graph labeling
		p_b_params = dict()
		p_b_params['p'] = self.problem.p
		p_b_params['b'] = self.problem.b_s[0]
		
		error_score = list([p_b_params])
		cost_score = list([p_b_params])
		
		for step in range(1,iterationNumber+1):
			#Train one iteration of adam optimization
			
			train_v,f_v = sess.run([train,f])
			#logger.debug("Training step finished")
			
			self.graphFitnessIteration.append((step,f_v))
			

			if step % 10 == 0:
				#logger.debug("%d , %.10f"%(step, f_v))
				print(step, f_v)
				"""logger.debug("x=%f"% sess.run(x))"""
				
				#logger.debug("Compute constraint errors..")
				#counting constraint errors
				"""errors_gt = sum(1 for i in sess.run(R_gt) if i<0)
				errors_lt = sum(1 for i in sess.run(R_lt) if i<0)
				errors_eq = sum(1 for i in sess.run(R_eq) if i<0)
				errors_x = sum(1 for i in sess.run(x) if i<0)"""
				
				#errors = errors + errors_gt + errors_lt + errors_eq + errors_x
				#logger.debug("Compute constraint errors done.")
		
			if step % 200 == 0:
				tmpImage = self.problem.superpixels.floodfillImage(sess.run(x))

				if(step%400 == 0):
					saveImage(tmpImage,"","%d/image_b_s%.4f_p%.4f_step%d"%(self.tmpIndex,p_b_params['b'], p_b_params['p'], step))
				
				#error_score = (abs(validImage - tmpImage)).sum()/(imgHeight*imgWidth)
				error = EvaluationMetrics.validationError(tmpImage,validImage)
				error_score.append((step,error))
				
				cost_score.append((step,f_v))
		print("error score=",error_score)
		#plotGraph(error_score[1:-1],"error( vs. ground truth)","error_graph_%d_b_s%.4f_p%.4f"%\
		#									(self.tmpIndex,p_b_params['b'], p_b_params['p']))
		#plotGraph(cost_score[1:-1],"fitness","cost_score_graph_%d_b_s%.4f_p%.4f"%\
		#									(self.tmpIndex,p_b_params['b'], p_b_params['p']))
		scores = [np.asarray(cost_score),np.asarray(error_score)]
		
		xSolution = sess.run(x)
		logger.debug("data type(x_result)=%s"%xSolution.dtype.name)
		
		if(mode=="SHADE_REFLECT"):
			for i in range(len(xSolution)):
				xSolution[i] = min( np.exp(xSolution[i]),1 ) # Paper section 4.1 end 
			
		if(errors>0):
			logger.warning("constraint errors= %d"%errors)
		#print("x solution=",xSolution)	
		sess.close()
		
	
		return xSolution,scores
	
	def plotLearningGraph(self):
		"""
		Plots the learning graph of the Gradient Descent Optimizer.
		"""
		plt.figure("Learning Graph")
		plt.title("Learning Graph")
		plt.xlabel('iterations')
		plt.ylabel('fitness')
	
		plt.plot(self.graphFitnessIteration)
		plt.clf()
		
	
	def testL2TermComputation(self):
	
		#x = np.array([0.1,0.2,0.4,0.4])
		x = tf.Variable(initial_value= np.array([0.1,0.2,0.4,0.4]),
						constraint=lambda x: tf.clip_by_value(x, lower_bound, upper_bound),
						dtype=tf.float32)
		R_gt = np.array( [1,1,1,1], dtype="float32" )
		R_lt = np.array( [1,1,1,1], dtype="float32" )
		R_eq = np.array( [1,1,1,1], dtype="float32" )
		
		#b_s = np.array( [0.5,0.5,0.5,0.5], dtype="float32" )
		b_s = tf.convert_to_tensor([0.5,0.5,0.5,0.5] ,dtype=tf.float32)
	
		numEdges = 4
		W_gt = np.zeros([numEdges, numEdges], dtype="float32")
		W_lt = np.zeros([numEdges, numEdges], dtype="float32")
		W_eq = np.zeros([numEdges, numEdges], dtype="float32")
		W_s = np.zeros([numEdges, numEdges], dtype="float32")
		
		W_gt[0][0] = 0.8
		W_gt[1][1] = 0.7
		W_gt[2][2] = 0.9
		W_gt[3][3] = 0.1
		
		W_eq[0][0] = 0.8
		W_eq[1][1] = 0.7
		W_eq[2][2] = 0.9
		W_eq[3][3] = 0.1
		
		W_lt[0][0] = 0.8
		W_lt[1][1] = 0.7
		W_lt[2][2] = 0.9
		W_lt[3][3] = 0.1
		
		W_s[0][0] = 0.8
		W_s[1][1] = 0.7
		W_s[2][2] = 0.9
		W_s[3][3] = 0.1
		
		optProblem = QuadraticProblem("test",None,None)
		
		nodes = [(0,0),(0,100),(100,100),(100,0)]
		edges = [( (0,0),(0,100) ), ( (0,100),(100,100) ), 
				 ( (100,100),(100,0) ), ( (100,0), (0,0) )] 
		
		(A_gt, A_lt, A_eq, A_s) = optProblem.getMatricesA(nodes,edges)
		print("A_gt= \n", A_gt , "\n")
		print("A_lt= \n", A_lt, "\n")
		print("A_eq= \n", A_eq, "\n")
		print("A_s= \n", A_s, "\n")
		
		#A_gt = np.zeros([numEdges, numNodes + numEdges],dtype="float32")
		#A_lt = np.zeros([numEdges, numNodes + numEdges],dtype="float32")
		#A_s = np.zeros([numEdges, numNodes],dtype="float32")
		
		L_eq = self.getL2Term(x ,R_eq ,A_eq ,W_eq )
		L_gt = self.getL2Term(x ,R_gt ,A_gt ,W_gt )
		L_lt = self.getL2Term(x ,R_lt ,A_lt ,W_lt )	
		L_s = self.getSmoothTerm(x ,A_s ,W_s, b_s)
		slackTerm = self.getSlackTerm(R_gt,R_lt,R_eq,"DEPTH-RECOVERY")
		sess = tf.Session()
		
		logger.info("init variables... ")
		init = tf.global_variables_initializer()
		sess.run(init)
		logger.info("init variables done. ")

		print(sess.run(L_eq))
		print(sess.run(L_gt))
		print(sess.run(L_lt))
		print(sess.run(L_s))
		print(sess.run(slackTerm))

	
