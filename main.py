import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utilities.dataset import Dataset
from models.model import EncoderModel, Seq2SeqModel, DecoderPointerAttentionModel, DecoderModel

import os
import config
from plot_utilities import plot_losses, plot_accuracies
from train import train_model

experiments_dir = "results/seq2seq"
os.makedirs(experiments_dir, exist_ok=True)
losses_file_name = f"{experiments_dir}/{config.loss_file_name}.png"
acc_file_name = f"{experiments_dir}/{config.acc_file_name}.png"


train_data = Dataset(num_samples=config.train_size,
                    num_nodes=config.input_sequence_length)
test_data = Dataset(num_samples=config.train_size,
                    num_nodes=config.input_sequence_length)
train_loader = DataLoader(train_data,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=0)
test_loader = DataLoader(train_data,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=0)

encoder = EncoderModel(input_size=config.input_sequence_length,
                    emb_size=config.emb_size,
                    hidden_size=config.hidden_size,
                    output_size=config.input_sequence_length)
decoder = DecoderPointerAttentionModel(input_size=config.input_sequence_length,
                                       emb_size=config.emb_size,
                                       hidden_size=config.hidden_size,
                                       output_size=config.input_sequence_length)

decoder = DecoderModel(input_size=config.input_sequence_length,
                                       emb_size=config.emb_size,
                                       hidden_size=config.hidden_size,
                                       output_size=config.input_sequence_length)


model = Seq2SeqModel(encoder, decoder)

train_losses, accuracies_per_epochs = train_model(model, train_loader, test_loader)
plot_losses(train_losses, losses_file_name)
plot_accuracies(accuracies_per_epochs, acc_file_name)
