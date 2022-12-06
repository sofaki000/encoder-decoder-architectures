import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class LinearModel(nn.Module):
     def __init__(self, input_size,emb_size,hidden_size, output_size):
        super(LinearModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=emb_size)
        self.layer_1 = nn.Linear(in_features=emb_size, out_features=hidden_size)
        self.layer_2 = nn.Linear(in_features=hidden_size, out_features=output_size)

     def forward(self, input):
        embedded_input = self.embedding(input.to(torch.int64))
        output = self.layer_1(embedded_input)
        output = F.tanh(output)
        output = self.layer_2(output)
        output = F.tanh(output)
        return F.softmax(output)

class Seq2SeqModel(nn.Module):
   def __init__(self, encoder, decoder):
      super(Seq2SeqModel, self).__init__()
      self.encoder= encoder
      self.decoder = decoder

   def forward(self, input):
      encoder_output, hidden = self.encoder(input)

      # the decoder output is the INDEX of sequence we suggest
      return self.decoder(input, encoder_output)

class EncoderModel(nn.Module):
   def __init__(self, input_size, emb_size, hidden_size, output_size):
      super(EncoderModel, self).__init__()

      self.hidden_size = hidden_size
      self.sequence_length = input_size

      # layers
      self.embedding = nn.Embedding(num_embeddings=input_size,
                                    embedding_dim=emb_size)
      self.lstm = nn.LSTM(emb_size, hidden_size,batch_first=True)

      # self.layer_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size*2)

   def forward(self, input):
      # embedded_input = self.embedding(input.to(torch.int64)) # (bs, seq_len, emb_size)
      embedded_input = self.embedding(input)  # (bs, seq_len, emb_size)

      # pernaei mia mono fora to embedded input
      output, hidden = self.lstm(embedded_input)

      return output, hidden

class DecoderModel(nn.Module):
   def __init__(self, input_size, emb_size, hidden_size, output_size):
      super(DecoderModel, self).__init__()

      self.hidden_size = hidden_size
      self.sequence_length = input_size
      self.emb_size = emb_size

      # if lstm is last layer:
      # self.lstm = nn.LSTMCell(self.emb_size,# LSTMCell's input is always batch first
      #                         output_size )

      #otherwise:
      self.lstm = nn.LSTMCell(self.emb_size,   self.hidden_size)
      self.layer_2 = nn.Linear(in_features=self.hidden_size , out_features=output_size)
   def init_state(self, encoder_outputs):
      encoder_outputs = encoder_outputs.transpose(1,0)
      return encoder_outputs[-1]
   def forward(self, input, encoder_output):

      cell_state = self.init_state(encoder_output)

      sequence_length = input.shape[1]
      batch_size = input.shape[0]

      hidden = torch.zeros([batch_size, self.hidden_size])   # hidden_initial

      tour_logp = []
      tour_idx = []

      visited_mask = [ False for i in range(sequence_length)]

      decoder_input = torch.zeros(batch_size, self.emb_size)
      probs = []
      # pernaei ena ena ta tokens kai bgazei to probability tou kathena element gia to
      # sygkekrimeno koutaki
      for i in range(self.sequence_length): # gia kathe thesi, tha broume poio element theloume pio poly
         out, hidden = self.lstm(decoder_input, (hidden, cell_state))

         out = self.layer_2(out)

         out = F.softmax(out)
         probs.append(out)
         # mask = [0 for i in range(4)]
         # indexes_of_visited_cities = [i for i, x in enumerate(visited_mask) if visited_mask[i] == True]
         #
         # for k in range(len(indexes_of_visited_cities)):
         #     mask[indexes_of_visited_cities[k]]  = 1
         #
         # if False: # taking next position from distribution
         #    m = torch.distributions.Categorical(probs)
         #    ptr = m.sample()
         #    logp = m.log_prob(ptr)
         # else:
         #    prob, ptr = torch.max(probs - torch.tensor(mask), 1)  # Greedy
         #    logp = prob.log()

         # tour_logp.append(logp.unsqueeze(1))
         # tour_idx.append(ptr.data.unsqueeze(1))
         # visited_mask[ptr.data.item()] = True


         # hidden = hidden[0]
      # tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
      # tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)
      probs = torch.stack(probs, dim=1)  # (bs, M, L) #make to tensor
      return probs



class DecoderAttentionModel(nn.Module):
   def __init__(self, input_size, emb_size, hidden_size, output_size):
      super(DecoderAttentionModel, self).__init__()

      self.hidden_size = hidden_size
      self.sequence_length = input_size
      self.emb_size = emb_size

      self.lstm = nn.LSTMCell(self.emb_size,# LSTMCell's input is always batch first
                          self.hidden_size )

      self.layer_2 = nn.Linear(in_features=self.hidden_size , out_features=output_size)
      self.weight_size = config.weight_size
      self.W1 = nn.Linear(hidden_size,config.weight_size, bias=False)  # blending encoder
      self.W2 = nn.Linear(hidden_size, config.weight_size, bias=False)  # blending decoder
      self.vt = nn.Linear(config.weight_size, 1, bias=False)  # scaling sum of enc and dec by v.T


   def init_state(self, encoder_outputs):
      encoder_outputs = encoder_outputs.transpose(1,0)
      return encoder_outputs[-1]
   def forward(self, input, encoder_output):

      cell_state = self.init_state(encoder_output)

      sequence_length = input.shape[1]
      batch_size = input.shape[0]

      hidden = torch.zeros([batch_size, self.hidden_size])   # hidden_initial


      decoder_input = torch.zeros(batch_size, self.emb_size)
      probs = []
      # pernaei ena ena ta tokens kai bgazei to probability tou kathena element gia to
      # sygkekrimeno koutaki
      for i in range(self.sequence_length): # gia kathe thesi, tha broume poio element theloume pio poly
         out, hidden = self.lstm(decoder_input, (hidden, cell_state))

         # DEN MAS NOIAZEI TO OUTPUT TOU DECODER
         # MONO TO HIDDEN STATE TOU
         # output = self.layer_2(out)
         # out = F.softmax(output)

         blend1 = self.W1(encoder_output.transpose(1, 0))  # (L, bs, W)
         blend2 = self.W2(hidden)  # (bs, W)
         blend_sum = F.tanh(blend1 + blend2)  # (L, bs, W)
         out = self.vt(blend_sum).squeeze()  # (L, bs)
         out = F.log_softmax(out.transpose(0, 1).contiguous(), -1)  # (bs, L)
         probs.append(out)


      probs = torch.stack(probs, dim=1)  # (bs, M, L) #make to tensor
      return probs