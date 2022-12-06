import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utilities.dataset import Dataset
from models.model import EncoderModel, Seq2SeqModel, DecoderPointerAttentionModel
import os
import config
from plot_utilities import plot_losses, plot_accuracies
import statistics

def test(model, X, Y):
    probs = model(X)  # (bs, M, L)
    _v, indices = torch.max(probs, 2)  # (bs, M)

    correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind, y in zip(indices, Y)])

    accuracy = correct_count / len(X) * 100

    print(f'Acc: {accuracy:.2f}% ({correct_count}/{len(X)})')
    return accuracy

def train_model(model, train_loader, test_loader):
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
            loss =  F.nll_loss(probs, y)

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

    return train_losses, accuracies_per_epochs