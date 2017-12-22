import tensorflow as tf

def rnns(input_layer, training):
	# Three BRNNs with 0.5 dropout and 300 hidden units
	# Inputs is shape [700, 192]

	def _cell():
			# Use default tanh activation.
			cell = tf.contrib.rnn.GRUCell(num_units=300)
			# Dropout if we are training.
			def true_(): return tf.constant(0.5)
			def false_(): return tf.constant(1.0)
			keep_prob = tf.cond(training, true_, false_)
			return tf.nn.rnn_cell.DropoutWrapper(cell, 
				output_keep_prob=keep_prob)

	with tf.variable_scope("rnn_1"):
		# Outputs is a tuple (fw, bw) of [1, 700, 300]
		outputs1, _ = tf.nn.bidirectional_dynamic_rnn(_cell(), _cell(), 
			inputs=tf.expand_dims(input_layer, 0), dtype=tf.float32)

	with tf.variable_scope("rnn_2"):
		outputs2, _ = tf.nn.bidirectional_dynamic_rnn(_cell(), _cell(),
			inputs=tf.concat(outputs1, 2), dtype=tf.float32)

	with tf.variable_scope("rnn_3"):
		outputs3, _ = tf.nn.bidirectional_dynamic_rnn(_cell(), _cell(),
			inputs=tf.concat(outputs2, 2), dtype=tf.float32)

	return tf.squeeze(tf.concat(outputs3, 2))