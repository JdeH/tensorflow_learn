import tensorflow as tf

A = tf.constant ([0, 1, 2, 3], shape = [2, 2])
x = tf.constant ([4, 5], shape = [2, 1])
b = tf.constant ([6, 7], shape = [2, 1])
y = tf.matmul (A, x) + b    # Use @ i.s.o. matmul a.s.a. tensorflow supports it

session = tf.Session ()

print (tf.Session () .run ([A, x, b, y]))

writer = tf.summary.FileWriter ('log', graph = session.graph)
