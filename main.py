import torch
from matplotlib import pyplot as plt
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utilities.dataset import Dataset
from model import DecoderModel, EncoderModel, Seq2SeqModel, DecoderAttentionModel
import os
import config
experiments_dir = "results"
os.makedirs(experiments_dir, exist_ok=True)
losses_file_name = f"{experiments_dir}/{config.loss_file_name}.png"
acc_file_name = f"{experiments_dir}/{config.acc_file_name}.png"
import statistics

def test(model, X, Y):
    probs = model(X)  # (bs, M, L)
    _v, indices = torch.max(probs, 2)  # (bs, M)

    correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind, y in zip(indices, Y)])

    accuracy = correct_count / len(X) * 100

    print(f'Acc: {accuracy:.2f}% ({correct_count}/{len(X)})')
    return accuracy

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
decoder = DecoderAttentionModel(input_size=config.input_sequence_length,
                    emb_size=config.emb_size,
                    hidden_size=config.hidden_size,
                    output_size=config.input_sequence_length)

model = Seq2SeqModel(encoder, decoder)

model.train()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
train_losses = []

accuracies_per_epochs = []

for epoch in range(config.n_epochs):
    accuracy_at_epoch = []
    running_loss = .0
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch

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

        running_loss += loss.item()

    if epoch % 2 == 0:
         for  batch_test_idx, test_batch in enumerate(test_loader):
             x , y = test_batch
             test(model, x, y)

             acc = test(model,  x, y)
             accuracy_at_epoch.append(acc)
         mean_acc = statistics.mean(accuracy_at_epoch)
         print(f'Mean acc:{mean_acc:.2f}%')
         accuracies_per_epochs.append(mean_acc)

    train_losses.append(running_loss)


    print(f'Epoch {epoch}:, Loss={running_loss}')

plt.plot(train_losses)
plt.savefig(losses_file_name)
plt.clf()

plt.plot(accuracies_per_epochs)
plt.savefig(acc_file_name)

# for i in range(0, N - config.batch_size, config.batch_size):  # with batch size
#     x = input[i:i + config.batch_size]  # (bs, L)
#     y = targets[i:i + config.batch_size]  # (bs, M)
#     test(model, x, y)