# Predicting Protein Secondary Structure with Deep Learning

In this project, we attempt to use a neural network to predict protein 
secondary structure from peptide sequences. Inputs are 700-dimensional 
sequences of amino acids, and outputs are Q8 classification of secondary
protein structure.

Our model is slightly simpler than the original model from the paper.
In the paper, there were two major input sources: the amino acid sequence and
a 22-dimensional profile of features obtained from PSI-BLAST. Here, we only use
the amino acid sequence. Additionally, the model trains on a 4 dimensional
label on the protein's solvent accessibility, which we leave out.

## The model

[Pipeline](proteinrnn_pipeline.png "Pipeline")

The model has 4 stages:
- Embedding stage: The inputs is a sparse 21-dimensional one-hot encoding of 
the amino acid type. Since dense vectors are better for learning, the first
layer is a 50-dimensional embedding.
- Convolutional stage: We have 3 1-D convolutions along the direction of the
amino acid sequence. The layers have heights 3, 7, and 11 and output 64 
filters. The filters each undergo batch normalizations, then are concatenated 
for an output of shape [700, 192]. This layer captures effects that amino 
acids have in conjunction with their immediate neighbors.
- Recurrent stage: This stage consists of three stacked bi-directional RNNs.
The cells in each network are gated recurrent units (GRUs). Each cell has 300 hidden units and a dropout rate of 0.5. This layer captures long range effects of the amino acid sequence.
- Fully conneted stage: The final stage consists of two fully connected layers,
then a final 9-dimensional output.

## Get Data

Data is taken from two protein databases. Training and validation data is taken
from CB6133 [Wang and Dunbrack, 2003] and testing data is from CB 513 [Cuff 
and Barton, 1999]. This mirror the initial paper, but leaves out the CAS10 and CAS11 databases.

Luckily, data was pre-prepared. run ```bash get_data.bash```. Taken from [delta2323's chainer implementation](https://github.com/delta2323/BMI219-2017-ProteinFolding).

## Reference

Guoli Wang and Roland L Dunbrack. Pisces: a protein sequence culling server.Bioinformatics, 19(12):1589–1591, 2003.

James A Cuff and Geoffrey J Barton. Evaluation and improvement of multiple sequence methods for protein secondary structure prediction. Proteins: Structure, Function, and Bioinformatics, 34(4):508– 519, 1999. 

Li, Z., & Yu, Y. (2016). Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks. arXiv preprint arXiv:1604.07176.

## Results

I trained the model with a batch size of 64 over twelve epochs. Our ending Q8 accuracy was 53%. After about 3 epochs we began to overfit, since we achieved maximum accuracy of 62%, after which point it monotonically decreased.

This is worse than the model in the original paper, possibly because we did not use solvent accessibility in training.

## TODOS

- Take into account solvent accessibility during training and see how much it
helps the model.