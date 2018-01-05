# Model 2

In the original model I was confused by the fact that the input to the recurrent layer is the output of the convolutional layer. This seems strange for two reasons:

- It's hard to conceptualize the output of a convolution as a step in a sequence given that adjacent convolutions encode some of the same information.

- A single input of the recurrent layer contained outputs of multiple different filter sizes (3, 7, and 11).

I wanted to see if it improved the model to have the input of the convolutional and recurrent layers be the same embedding layer, and then concatenate the two at the end.

This modification made no significant difference to the performance of the model over twelve epochs.