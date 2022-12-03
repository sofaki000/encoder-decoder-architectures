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

class EncoderModel(nn.Module):
   def __init__(self, input_size, emb_size, hidden_size, output_size):
      super(EncoderModel, self).__init__()

      self.hidden_size = hidden_size
      self.sequence_length = input_size

      # layers
      self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=emb_size)
      self.lstm = nn.LSTM(emb_size, hidden_size)
      self.layer_2 = nn.Linear(in_features=hidden_size, out_features=output_size)

   def forward(self, input):
      embedded_input = self.embedding(input.to(torch.int64))

      batch_size = 1

      hidden_initial = Variable(torch.zeros(batch_size, self.hidden_size))
      c_0 = Variable(torch.zeros(batch_size, self.hidden_size))

      # pernaei mia mono fora to embedded input
      _, (final_hidden_state, final_cell_state) = self.lstm(embedded_input, (hidden_initial, c_0))

      output = self.layer_2(final_hidden_state)
      return F.softmax(output)

class DecoderModel(nn.Module):
   def __init__(self, input_size, emb_size, hidden_size, output_size):
      super(DecoderModel, self).__init__()

      self.hidden_size = hidden_size
      self.sequence_length = input_size

      # layers
      # self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=emb_size)
      # self.lstm = nn.LSTM(emb_size, hidden_size)
      self.lstm = nn.LSTM(config.node_features, hidden_size) # nn.LSTM(input_size, hidden_size)
      self.layer_2 = nn.Linear(in_features=hidden_size, out_features=output_size)

   def forward(self, input):
      # embedded_input = self.embedding(input.to(torch.int64))

      sequence_length = input.shape[0]
      batch_size = 1

      hidden_initial = Variable(torch.zeros(batch_size, self.hidden_size))
      c_0 = Variable(torch.zeros(batch_size, self.hidden_size))

      hidden = hidden_initial
      probs = []

      tour_logp = []
      tour_idx = []

      visited_mask = [ False for i in range(sequence_length)]

      # pernaei ena ena ta tokens kai bgazei to probability tou kathena element gia to
      # sygkekrimeno koutaki
      for i in range(self.sequence_length): # gia kathe thesi, tha broume poio element theloume pio poly
         out, hidden = self.lstm(input[i].unsqueeze(0).view(1, -1)) # pairnei 1 thesi

         output = self.layer_2(out)

         probs = F.softmax(output)

         mask = [0 for i in range(4)]
         indexes_of_visited_cities = [i for i, x in enumerate(visited_mask) if visited_mask[i] == True]
         for k in range(len(indexes_of_visited_cities)):
             #probs[0][index_of_visited_city] = 0 #np.Inf
             mask[indexes_of_visited_cities[k]]  = 1

         if False: # taking next position from distribution
            m = torch.distributions.Categorical(probs)
            ptr = m.sample()
            logp = m.log_prob(ptr)
         else:
            prob, ptr = torch.max(probs - torch.tensor(mask), 1)  # Greedy
            logp = prob.log()

         counter = 0
         # while visited_mask[ptr.data.item()] is True: # xanapernoume deigma mexri na ftasei se polh pou den exoume paei
         #    ptr = m.sample()
         #
         #    counter +=1
         #
         #    if counter==10:
         #       print("Sorry, surpassed 10 tries ")
         #       break


         tour_logp.append(logp.unsqueeze(1))
         tour_idx.append(ptr.data.unsqueeze(1))
         visited_mask[ptr.data.item()] = True

      #return torch.cat(probs)

      tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
      tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

      return tour_idx, tour_logp
