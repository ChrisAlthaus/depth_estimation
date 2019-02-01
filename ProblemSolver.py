import math
import numpy as np
import time
import QuadraticProblem


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
		
	def solveProblem(self):
		"""
		Used to simulate the optimization algorithm.
		Returns test result.
		
		:return: x fill value of each segmentnumber
		"""
		numSegments = len(self.problem.superpixels.nodes)
		
		x = np.linspace(0,1,numSegments).tolist()
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
		
		:param:	xVector 	numpy vector with greyscale value of each segment
				R_eq		numpy vector with slack variable for each edge
				R_gt		numpy vector with slack variable for each edge
				R_lz		numpy vector with slack variable for each edge
		
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
		
		
	
	
	
	