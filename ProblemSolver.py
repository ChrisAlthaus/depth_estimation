import numpy as np

def solveProblem(numSegments):
	"""
	Used to simulate the optimization algorithm.
	Returns test result.
	
	:param:	numSegments number of segments
	
	:return: x fill value of each segmentnumber
	"""
	
	x = np.linspace(0,1,numSegments)
	print("Solved! x= ", x)
	
	return x



	