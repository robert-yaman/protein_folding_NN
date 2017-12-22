import tensorflow as tf

import embedding
import fc
import cnn
import rnn

class ProteinFoldingModel(object):
	def __init__(self, input_tensor, training):
		self.embedding_layer = embedding.embedding(input_tensor)
		self.cnn_layer = cnn.cnns(self.embedding_layer, training)
		self.rnn_layer = rnn.rnns(self.cnn_layer, training)
		self.logits = fc.fc(self.rnn_layer, self.cnn_layer, training)
		self.readout = tf.nn.softmax(self.logits)
