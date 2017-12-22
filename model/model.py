import data
import embedding
import cnn
import rnn

class ProteinFoldingModel(object):
	def __init__(self, input_tensor, training):
		embedding_layer = embedding.embedding(input_tensor)
		cnn_layer = cnn.cnns(embedding_layer, training)
		rnn_layer = rnn.rnns(cnn_layer, training)
		print "CURRENT SHAPE: "
		print rnn_layer.shape
