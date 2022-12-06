import torch
from matplotlib import pyplot as plt
from torch import optim
import torch.nn.functional as F
from data_utilities import generate_data
from model import DecoderModel, EncoderModel, Seq2SeqModel, DecoderAttentionModel
import os
import config
experiments_dir = "results"
os.makedirs(experiments_dir, exist_ok=True)
losses_file_name = f"{experiments_dir}/{config.loss_file_name}.png"
acc_file_name = f"{experiments_dir}/{config.acc_file_name}.png"


def test(model, X, Y):
    probs = model(X)  # (bs, M, L)
    _v, indices = torch.max(probs, 2)  # (bs, M)

    correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind, y in zip(indices, Y)])

    accuracy = correct_count / len(X) * 100

    print(f'Acc: {accuracy:.2f}% ({correct_count}/{len(X)})')
    return accuracy

inputs, targets = generate_data.make_seq_data(config.train_size,  config.input_sequence_length)
#generate_data.task_data_for_predicting_same_sequence(config.train_size,   config.input_sequence_length)

data_split = (int)(config.train_size * 0.9)
train_X = inputs[:data_split]
train_Y = targets[:data_split]
test_X = inputs[data_split:]
test_Y = targets[data_split:]

encoder = EncoderModel(input_size=config.input_sequence_length,
                    emb_size=config.emb_size,
                    hidden_size=config.hidden_size,
                    output_size=config.input_sequence_length)
decoder = DecoderAttentionModel(input_size=config.input_sequence_length,
                    emb_size=config.emb_size,
                    hidden_size=config.hidden_size,
                    output_size=config.input_sequence_length)

model = Seq2SeqModel(encoder, decoder)

model.train()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
train_losses = []
accuracies = []

input = (torch.LongTensor(inputs))     # (N, L)
targets = (torch.LongTensor(targets)) # (N, L)

for epoch in range(config.n_epochs):
    running_loss = .0
    train_acc = .0
    correct = 0
    N = input.size(0)
    for i in range(0, N - config.batch_size, config.batch_size): # with batch size
        x = input[i:i + config.batch_size]  # (bs, L)
        y = targets[i:i + config.batch_size]  # (bs, M)

        probs = model(x)

        # why he suggests doing that?
        # probs = probs.view(-1 , input.size(1))
        # y = y.view(-1)  # (bs*M)
        # whats the difference? F.nll_loss(torch.stack(probs, dim=1).view(-1,input.size(1)), y.view(-1))
        loss = F.nll_loss(probs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # indexes = np.argmax(output.detach().numpy(), axis=1)
        #predicted = input[np.argmax(F.softmax(output).detach().numpy(), axis=1)].numpy()
        # print(f'Predicted: {tour_idx.numpy()[0]}')
        # print(f'Expected:{target.numpy()}')
        # if np.array_equal(tour_idx.numpy()[0], target.numpy()):
        #     print("Equal!!!")

        # #correct += (tour_idx.detach().numpy() == target.numpy()).float().sum()
        # train_acc = np.count_nonzero(tour_idx.detach().numpy() == target.numpy()) # / config.input_sequence_length
        running_loss += loss.item()

    if epoch % 2 == 0:
        for i in range(0, N - config.batch_size, config.batch_size):  # with batch size
            x = input[i:i + config.batch_size]  # (bs, L)
            y = targets[i:i + config.batch_size]  # (bs, M)
            test(model, x, y)

            acc = test(model,  x, y)
            accuracies.append(acc)

    train_losses.append(running_loss)


    print(f'Epoch {epoch}:, Loss={running_loss}')

plt.plot(train_losses)
plt.savefig(losses_file_name)
plt.clf()

plt.plot(accuracies)
plt.savefig(acc_file_name)

for i in range(0, N - config.batch_size, config.batch_size):  # with batch size
    x = input[i:i + config.batch_size]  # (bs, L)
    y = targets[i:i + config.batch_size]  # (bs, M)
    test(model, x, y)