import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    """
    The encoder generates a single output vector that embodies the input sequence meaning.
    The general procedure is as follows:
        1. In each step, a word will be fed to a network and it generates
         an output and a hidden state.
        2. For the next step, the hidden step and the next word will
         be fed to the same network (W) for updating the weights.
        3. In the end, the last output will be the representative of the input sentence (called the "context vector").
    """
    def __init__(self, hidden_size, input_size, batch_size, num_layers=1):
        """
        * For nn.LSTM, same input_size & hidden_size is chosen.
        :param input_size: The size of the input vocabulary
        :param hidden_size: The hidden size of the RNN.
        :param batch_size: The batch_size for mini-batch optimization.
        :param num_layers: Number of RNN layers. Default: 1
        :param bidirectional: If the encoder is a bi-directional LSTM. Default: False
        """
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # The input should be transformed to a vector that can be fed to the network.
        self.embedding = nn.Embedding(input_size, embedding_dim=hidden_size)

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)


    def forward(self, input, hidden):
        # Make the data in the correct format as the RNN input.
        embedded = self.embedding(input).view(1, 1, -1)
        rnn_input = embedded
        # The following descriptions of shapes and tensors are extracted from the official Pytorch documentation:
        # output-shape: (seq_len, batch, num_directions * hidden_size): tensor containing the output features (h_t) from the last layer of the LSTM
        # h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state
        # c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state
        output, (h_n, c_n) = self.lstm(rnn_input, hidden)
        return output, (h_n, c_n)

    def initHidden(self):

        encoder_state = [torch.zeros(self.num_layers, 1, self.hidden_size),
                              torch.zeros(self.num_layers, 1, self.hidden_size)]
        return encoder_state
