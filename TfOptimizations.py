import tensorflow as tf
import numpy as np
	
	
def ClipIfNotNone(grad):
	if grad is None:
		return grad
	return tf.clip_by_value(grad, -400, 400)
	
def fitness_function(x):
	result = tf.add( tf.pow(x,2.0), 10 )
	return result

def getTerm(x):
	return tf.pow(x,2.0)
	
	
def minOptimization():
	
	#@tf.RegisterGradient("ExternalGradient")
	#def _custom_external_grad(unused_op, grad):
		# I don't know yet how to compute a gradient
		# From Tensorflow documentation:
	#	return grad, tf.negative(grad)
	
	x_init_value = np.array([4, 5])
	x_var = tf.Variable(x_init_value, dtype=tf.float32)
	
	x = tf.Variable(initial_value= 100, dtype=tf.float32)
	# Loss function
	#f = tf.add(tf.pow(tf.subtract(1.0, x_var[0]), 2.0), 
    #       tf.multiply(100.0, tf.pow(tf.subtract(x_var[1],tf.pow(x_var[0], 2.0)), 2.0)))
		   
	#g = tf.pow(x,2.0)
	f = tf.add(getTerm(x),10)
	
	
	#g = tf.get_default_graph()
	#with g.gradient_override_map({"PyFunc": "ExternalGradient"}):
	#	f =  tf.py_func(fitness_function, [x], [tf.float32])[0]
	#f = tf.py_func(fitness_function, [x], tf.float32)

	
	#f = my_model(x_var) - 0.5 + g(x_var)
	# Define the optimizer
	
	opt = tf.train.GradientDescentOptimizer(0.0035)
	#grads_and_vars = opt.compute_gradients(f, [x_var])
	grads_and_vars = opt.compute_gradients(f, [x])
	
	clipped_grads_and_vars = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
	
	
	#clipped_grads_and_vars = [(tf.clip_by_value(g, -10., 10.), v) for g, v in grads_and_vars]

	train = opt.apply_gradients(clipped_grads_and_vars)
	
	#opt = tf.train.GradientDescentOptimizer(0.0035)
	#train = opt.minimize(f)

	sess = tf.Session()

	init = tf.global_variables_initializer()
	sess.run(init)
	
	for step in range(800):
		
		sess.run(train)
		
		if step % 10 == 0:
			print(sess.run(grads_and_vars))
			print(sess.run(clipped_grads_and_vars))
		
			#print(step, sess.run(x_var[0]), sess.run(x_var[1]), sess.run(f))
			print(step, sess.run(x),sess.run(f))
		
		
	#optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(f) 

	#sess = tf.Session()

	print("Tensorflow result= ",sess.run(x_var[0]), sess.run(x_var[1]), sess.run(f))
