from matplotlib import pyplot as plt

def plot_losses(train_losses, losses_file_name):
    plt.plot(train_losses)
    plt.savefig(losses_file_name)
    plt.clf()


def plot_accuracies(accuracies_per_epochs, acc_file_name):
    plt.plot(accuracies_per_epochs)
    plt.savefig(acc_file_name)