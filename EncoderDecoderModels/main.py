import torch
from matplotlib import pyplot as plt

import config
from data_utilities import generate_data
from EncoderDecoderModels.EncoderDecoder import EncoderDecoderNetwork
from train import train, test

input_seq_len = 4
input, targets = generate_data.make_seq_data(config.total_size, input_seq_len)

# Convert to torch tensors
input = (torch.LongTensor(input))     # (N, L)
targets = (torch.LongTensor(targets)) # (N, L)

data_split = (int)(config.total_size * 0.9)
train_X = input[:data_split]
train_Y = targets[:data_split]
test_X = input[data_split:]
test_Y = targets[data_split:]
# pointer_network = PointerNetwork(input_seq_len, config.emb_size, config.weight_size, input_seq_len)
# losses_pointer_network = train(pointer_network, train_X, train_Y, config.batch_size, config.n_epochs)
# print('----Test result---')
# test(pointer_network, test_X, test_Y)
#
# plt.plot(losses_pointer_network, range(config.n_epochs+1))
# plt.xlabel("Epochs")
# plt.ylabel("Loss per epoch")
# plt.savefig('pointer_network.png')
# plt.clf()
# print("---------------------- for encoder decoder only ----------------------")


# encoder = Encoder(input_dim=inp_size, hidden_dim=config.hidden_size, embbed_dim=config.emb_size, num_layers=1)
# decoder = Decoder(output_dim=inp_size, hidden_dim=config.hidden_size, embbed_dim=config.emb_size, num_layers=1)
# encoder_decoder = Seq2Seq(encoder, decoder)
encoder_decoder = EncoderDecoderNetwork(input_seq_len, config.emb_size, config.weight_size, input_seq_len)
losses_encoder_decoder_network = train(encoder_decoder, train_X, train_Y, config.batch_size, config.n_epochs)
print('----Test result---')
test(encoder_decoder, test_X, test_Y)

plt.plot(losses_encoder_decoder_network, range(config.n_epochs+1))
plt.xlabel("Epochs")
plt.ylabel("Loss per epoch")

plt.savefig('encoder_decoder_network_v2.png')