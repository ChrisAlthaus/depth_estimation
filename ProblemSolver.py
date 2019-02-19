import math
import numpy as np
import time
import QuadraticProblem
import tensorflow as tf


class ProblemSolver:
	"""
	This class specifies the fitness function for an easy to use interface.
	
	Attributes:
		problem 	Reference to the relevant optimization problem
	"""
	
	#Reference to QuadraticProblem 
	problem = None
	
	def __init__(self, problem):
		if problem is None:
			raise ValueError('Optimization problem not initialized! Please first initialize.')
		
		self.problem = problem
		
		
	def getL2Term(self,xVector,R_xy,A_xy,W_xy):
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
		print("x= \n" , xVector)
		print("R_xy \n=" , R_xy)
		#print("dimensions x=" , xVector.shape)
		#print("dimensions R_xy=" , R_xy.shape)
		#xR_xy = np.concatenate((xVector.eval(),R_xy))
		#Concat into one vector and convert into vector shape
		xR_xy = tf.concat([xVector,R_xy],0)
		xR_xy = tf.transpose([xR_xy])
		#Printing matrices and dimensions
		"""print()
		print("x= \n" , xVector)
		print("R_xy \n=" , R_xy)
		print("->xR_xy= \n" , xR_xy)
		print("A_xy= \n" , A_xy)
		print("W_xy= \n" , W_xy)
		
		print("dimensions x=" , len(xVector))
		print("dimensions R_xy=" , len(R_xy))
		print("dimensions xR_xy=" , xR_xy.shape)
		print("dimensions A_xy=" , A_xy.shape)
		print("dimensions W_xy=" , W_xy.shape)"""
		
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
		
		result = tf.reshape(result,[-1])
		
		return result
		
	def getSmoothTerm(self,x,A_s,W_s,b_s):
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
		"""print()
		print("x= \n" , xVector)
		print("A_s= \n" , A_s)
		print("W_s= \n" , W_s)
		print("b_s= \n" , b_s)
		
		print("dimensions A_s=" , len(A_s))
		print("dimensions W_s=" , W_s.shape)
		print("dimensions b_s=" , len(b_s))
		"""
		#x = np.array(xVector)
		
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
		
		result = tf.add(result, tf.matmul(tf.transpose(x),b_s))
		#print("dimensions A_s^T * W_s * A_s * x + x^T * b_s=" , result.shape)
		result = tf.reshape(result,[-1])
		
		return result
		
	def getFitnessFunction(self,x,R_gt,R_lt,R_eq):
		
		lamb_eq = tf.constant(5,dtype=tf.float64)	
		lamb_gt = tf.constant(1,dtype=tf.float64)
		lamb_lt = tf.constant(1,dtype=tf.float64)
		lamb_s = tf.constant(0.5,dtype=tf.float64)
		
		L_eq = self.getL2Term(x ,R_eq ,self.problem.A_eq ,self.problem.W_eq )
		L_gt = self.getL2Term(x ,R_gt ,self.problem.A_gt ,self.problem.W_gt )
		L_lt = self.getL2Term(x ,R_lt ,self.problem.A_lt ,self.problem.W_lt )
		
		b_s = tf.convert_to_tensor(self.problem.b_s ,dtype=tf.float64)
		L_s = self.getSmoothTerm(x ,self.problem.A_s ,self.problem.W_s, b_s)
		
		print("Leq= ", L_eq)
		print("lamb_eq ", lamb_eq)
		
		L2Term = tf.add_n([tf.multiply(lamb_eq, L_eq), tf.multiply(lamb_gt, L_gt), 
						   tf.multiply(lamb_lt, L_lt), tf.multiply(lamb_s,L_s)])
						
		slackTerm = self.getSlackTerm(R_gt,R_lt,R_eq)
						   
		f = tf.add( L2Term, slackTerm)
		
		return f
	
	def ClipIfNotNone(self,grad):
		if grad is None:
			return grad
		return tf.clip_by_value(grad, -400, 400)
	
	def solveProblemTf(self):
		
		x = tf.Variable(initial_value= np.array(self.problem.superpixels.meanLuminances), dtype=tf.float64)
		(R_gt,R_lt,R_eq) = self.problem.getTestSlackVariables()
		R_1 = tf.convert_to_tensor(R_gt,dtype=tf.float64)
		R_2 = tf.convert_to_tensor(R_lt, dtype=tf.float64)
		R_3 = tf.convert_to_tensor(R_eq, dtype=tf.float64)
		
		f = self.getFitnessFunction(x,R_1,R_2,R_3)
		print("f=",f)
		
		opt = tf.train.GradientDescentOptimizer(0.0035)
		grads_and_vars = opt.compute_gradients(f, [x])
		
		clipped_grads_and_vars = [(self.ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
		

		train = opt.apply_gradients(clipped_grads_and_vars)

		sess = tf.Session()

		init = tf.global_variables_initializer()
		sess.run(init)
		
		for step in range(10):
			#print("grads_and_vars=",sess.run(grads_and_vars))
			sess.run(train)
			#print(sess.run(grads_and_vars))
			
			#if step % 10 == 0:
				#print(sess.run(grads_and_vars))
				#print(sess.run(clipped_grads_and_vars))
			
			print(step, sess.run(f))
		
		xSolution = sess.run(x)
		print("Solved! x= ",xSolution)
		return xSolution
						   
		
		
	def getSlackTerm(self,R_gt,R_lt,R_eq):
		
		stdDevGT= tf.constant(0.001,dtype=tf.float64)		#global variables?
		stdDevLT= tf.constant(0.001,dtype=tf.float64)
		stdDevEQ= tf.constant(0.001,dtype=tf.float64)
		
		meanGT= tf.constant(math.log(2),dtype=tf.float64)
		meanLT= tf.constant(math.log(2),dtype=tf.float64)
		meanEQ= tf.constant(0,dtype=tf.float64)
		
		#Computing the sum in the second section of the formula
		slackTerm = 0
		print(R_eq.shape[0])
		numEdges = R_eq.shape[0]
		
		for i in range(numEdges):
			
			tmp = tf.add_n( [tf.divide( tf.pow(R_eq[i],2.0), tf.pow(stdDevEQ,2.0) ),
								  tf.divide( tf.pow( tf.subtract(R_gt[i],meanGT), 2 ), tf.pow(stdDevGT,2) ),
						          tf.divide( tf.pow(tf.subtract(R_lt[i],meanLT), 2 ),tf.pow(stdDevLT, 2))] )
						
			slackTerm = tf.add(tmp ,slackTerm)
			
		return slackTerm
		
	def solveProblem(self):
		"""
		Used to simulate the optimization algorithm.
		Returns test result.
		
		:return: x fill value of each segmentnumber
		"""
		numSegments = len(self.problem.superpixels.nodes)
		
		#x = np.linspace(0,1,numSegments).tolist()
		x = self.problem.superpixels.meanLuminances
		(R_gt,R_lt,R_eq) = self.problem.getTestSlackVariables()
		
		startTime = time.time()
		for i in range(100):
			self.f(x, R_eq , R_gt , R_lt)
		timeTaken = time.time() - startTime
		print("100 fitness function calls took: %f seconds" % timeTaken)
		
		
		print("Solved! x= ", x)
		
		return x


	def f(self, xVector, R_eq , R_gt , R_lt):
		"""
		Fitness function for evaluating the given parameters.
		
		:param:	xVector 	list/vector with greyscale value of each segment
				R_eq		list/vector with slack variable for each edge
				R_gt		list/vector with slack variable for each edge
				R_lz		list/vector with slack variable for each edge
		
		:return: fitness value 
		
		"""
		
		lamb_eq = 5	
		lamb_gt = 1
		lamb_lt = 1
		lamb_s = 0.5
		
		stdDevGT= 0.001		#global variables?
		stdDevLT= 0.001
		stdDevEQ= 0.001
		
		meanGT= math.log(2)
		meanLT= math.log(2)
		meanEQ= 0
		
		
		
		#Computing the first section of the formula
		(L_eq, L_gt, L_lt, L_s) = self.problem.compL2Distances(xVector, R_eq , R_gt , R_lt)
		
		L2Term = lamb_eq * L_eq + lamb_gt * L_gt + lamb_lt * L_lt + lamb_s * L_s
		
		
		#Computing the sum in the second section of the formula
		slackSum = 0
		numEdges = len(R_eq)
		
		for i in range(numEdges):
			
			slackTerm = ( (R_eq[i]**2)/(stdDevEQ**2) ) +( ( (R_gt[i]-meanGT)**2 )/(stdDevGT**2) ) + \
						( ( (R_lt[i]-meanLT)**2 )/(stdDevLT**2) )
						
			slackSum = slackSum + slackTerm
			
		result = L2Term + slackSum
		
		return result
		
		
	
	
	
	