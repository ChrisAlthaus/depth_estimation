from SuperpixelSegmentation import SuperpixelSegmentation
from QuadraticProblem import QuadraticProblem
from  ProblemSolver import ProblemSolver

def main():
		imagePath = "living_room_small.jpg"
		numberSegments = 300
		
		superpixels = SuperpixelSegmentation(imagePath,numberSegments)
		
		optProblem = QuadraticProblem(superpixels)
		
		solver = ProblemSolver(optProblem)
		
		# Solve the optimization problem
		xSolution = solver.solveProblem()
		
		superpixels.floodfillImage(xSolution)

		superpixels.showPlot()
		
if __name__ == "__main__":
	main()