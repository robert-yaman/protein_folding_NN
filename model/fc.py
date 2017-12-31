import tensorflow as tf

def fc(recurrent_layer, convolutional_layer, training):
	def true_(): return tf.constant(0.5)
	def false_(): return tf.constant(1.0)
	keep_prob = tf.cond(training, true_, false_)

	# recurrent_layer has shape [batch_size, 700,600]
	# convolutional_layer has shape [batch_size, 700, 192]
	combined_layer = tf.concat([recurrent_layer, convolutional_layer], axis=2)
	with tf.name_scope('final_fc1'):
		intermediate_layer1 = tf.contrib.layers.fully_connected(
			tf.nn.dropout(combined_layer, keep_prob), 300)

	with tf.name_scope('final_fc2'):
		intermediate_layer2 = tf.contrib.layers.fully_connected(
		tf.nn.dropout(intermediate_layer1, keep_prob), 300)

	with tf.name_scope('logits'):
		logits = tf.contrib.layers.fully_connected(intermediate_layer2, 9)

	return logits
