from SuperpixelSegmentation import SuperpixelSegmentation
from QuadraticProblem import QuadraticProblem
from  ProblemSolver import ProblemSolver

#import TfOptimization as tfSrc


def main():
		
		imagePath = "living_room_small.jpg"
		numberSegments = 300
		
		superpixels = SuperpixelSegmentation(imagePath,numberSegments)
		
		optProblem = QuadraticProblem(superpixels)
		
		#Plot image for comparison
		superpixels.floodfillImage(superpixels.meanLuminances,0)
		
		solver = ProblemSolver(optProblem)
		
		# Solve the optimization problem
		xSolution = solver.solveProblemTf()
		
		print("x solution=", xSolution)
		
		superpixels.floodfillImage(xSolution,1)

		superpixels.showPlot()
		
if __name__ == "__main__":
	main()