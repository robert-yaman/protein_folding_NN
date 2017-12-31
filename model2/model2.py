import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model.model import ProteinFoldingModel

import model.rnn as rnn

class ProteinFoldingModel2(ProteinFoldingModel):
	def _recurrentLayer(self, training):
		# Override superclass.
		return rnn.rnns(self.embedding_layer, training)