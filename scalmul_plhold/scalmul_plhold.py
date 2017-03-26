import tensorflow as tf

session = tf.Session ()

x0Phold = tf.placeholder (tf.float64, name = 'x0')
x1Phold = tf.placeholder (tf.float64, name = 'x1')
yNode = tf.multiply (x0Phold, x1Phold, name = 'y')

print (session.run ([x0Phold, x1Phold, yNode], {x0Phold: 3.3, x1Phold: 10}))

writer = tf.summary.FileWriter ('log', graph = session.graph)

# x0, x1 and y are all drawn as operations
