import tensorflow as tf

session = tf.Session ()

x0Node = tf.constant (3.3, name = 'x0')
x1Node = tf.constant (10.0, name = 'x1')
yNode = tf.multiply (x0Node, x1Node, 'y')

print (session.run ([x0Node, x1Node, yNode]))

writer = tf.summary.FileWriter ('log', graph = session.graph)

# x0 and x1 are drawn as tensors (or rather scalars), and y is drawn as an operation
