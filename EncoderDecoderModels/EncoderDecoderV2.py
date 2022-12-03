from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()

        # set the encoder input dimesion , embbed dimesion, hidden dimesion, and number of layers
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # initialize the embedding layer with input and embbed dimention
        self.embedding = nn.Embedding(input_dim, self.embbed_dim)
        # intialize the GRU to take the input dimetion of embbed, and output dimention of hidden and
        # set the number of gru layers
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, src):
        embedded = self.embedding(src) #.view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()
        # set the encoder output dimension, embed dimension, hidden dimension, and number of layers
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # initialize every layer with the appropriate dimension.
        # For the decoder layer, it will consist of an embedding, GRU, a Linear layer and a Log softmax activation function.
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # reshape the input to (1, batch_size)
        # input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)#nn.GRU(12, 12, 1)(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source):
        input_length = source.shape[1]  # get the input length (number of words in sentence)
        batch_size = source.shape[0]# target.shape[1]
       # vocab_size = self.decoder.output_dim

        # initialize a variable to hold the predicted outputs
        outputs = torch.zeros(input_length, batch_size, input_length)

        # encode every word in a sentence
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])

        # use the encoder’s hidden layer as the decoder hidden
        decoder_hidden = encoder_hidden

        # add a token before the first predicted word
        decoder_input = (torch.zeros(batch_size, config.emb_size)).to(torch.int64)
        #decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        # topk is used to get the top K value over a list
        # predict the output word from the current target word. If we enable the teaching force,
        # then the #next decoder input is the next word, else, use the decoder output highest value.
        probs = []
        for t in range(input_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden.unsqueeze(0))
            outputs[t] = decoder_output
            probs_for_current_batch = F.log_softmax(decoder_output)
            probs.append(probs_for_current_batch)


        return probs


