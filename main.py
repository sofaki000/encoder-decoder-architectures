import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import torch.nn.functional as F
import numpy as np
from data_utilities import generate_data
from model import DecoderModel, EncoderModel, Seq2SeqModel
import os
import config
experiments_dir = "results"
os.makedirs(experiments_dir, exist_ok=True)
losses_file_name = f"{experiments_dir}/enc_dec_loss_lr_smallest_more_neurons.png"
acc_file_name = f"{experiments_dir}/enc_dec_acc_lr_smallest_more_neurons.png"

inputs, targets = generate_data.task_data_for_predicting_same_sequence(config.train_size,
                                                                       config.input_sequence_length)
encoder = EncoderModel(input_size=config.input_sequence_length,
                    emb_size=config.emb_size,
                    hidden_size=config.hidden_size,
                    output_size=config.input_sequence_length)
decoder = DecoderModel(input_size=config.input_sequence_length,
                    emb_size=config.emb_size,
                    hidden_size=config.hidden_size,
                    output_size=config.input_sequence_length)

model = Seq2SeqModel(encoder, decoder)

model.train()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
train_losses = []
accuracies = []

for epoch in range(config.epochs):
    running_loss = .0
    train_acc = .0
    correct = 0
    for i in range(config.train_size):
        input = torch.tensor(inputs[i], dtype=torch.float32)
        target = torch.tensor(targets[i])

        tour_idx, tour_logp = model(input)

        loss = F.nll_loss(tour_idx.squeeze(0) * tour_logp.squeeze(0), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # indexes = np.argmax(output.detach().numpy(), axis=1)
        #predicted = input[np.argmax(F.softmax(output).detach().numpy(), axis=1)].numpy()
        print(f'Predicted: {tour_idx.numpy()[0]}')
        print(f'Expected:{target.numpy()}')
        if np.array_equal(tour_idx.numpy()[0], target.numpy()):
            print("Equal!!!")

        #correct += (tour_idx.detach().numpy() == target.numpy()).float().sum()
        train_acc = np.count_nonzero(tour_idx.detach().numpy() == target.numpy()) # / config.input_sequence_length
        running_loss += loss.item()

    #accuracy = 100 * correct /  config.train_size
    final_train_acc = train_acc / config.train_size
    train_losses.append(running_loss)
    accuracies.append(final_train_acc)
    print(f'Epoch {epoch}: Accuracy={final_train_acc}, Loss={running_loss}')

plt.plot(train_losses)
plt.savefig(losses_file_name)
plt.clf()

plt.plot(accuracies)
plt.savefig(acc_file_name)