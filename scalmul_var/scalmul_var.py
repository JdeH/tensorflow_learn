import tensorflow as tf

session = tf.Session ()

# Creating variables use constructors rather than functions
# They are not yet initialised, merely an object is creaed
x0Var = tf.Variable ([3.3], dtype = tf.float64, name = 'x0')
x1Phold = tf.placeholder (tf.float64, name = 'x1')
yNode = tf.multiply (x0Var, x1Phold, 'y')


session.run (tf.global_variables_initializer ())
print (session.run ([x0Var, x1Phold, yNode], {x1Phold: [10, 20, 30]}))

writer = tf.summary.FileWriter ('log', graph = session.graph)

# x1 and xy are drawn as operations, x0 is drawn as a rectangular box connected to an init 'auxiliary' operation (aux. node)
