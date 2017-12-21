import tensorflow as tf

def embedding(inputs):
	# Creates embedding layer [700, 50] from one hot encoded inputs [700, 22].
	# relu is default activation
	embedding_layer = tf.contrib.layers.fully_connected(inputs, 50)
	return embedding_layer
