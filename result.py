import matplotlib
import matplotlib.pyplot as plt
import numpy as np

path = './deconv_model/'
# total_loss = np.load(path + 'total_loss.npy', allow_pickle=True)
total_loss = np.load(path + 'loss_arr.npy', allow_pickle=True)
# loss = np.load(path + 'loss_arr (2).npy', allow_pickle=True)
# total_loss1 = np.concatenate((total_loss, loss[:5]), axis=0)
# save_loss = path + 'total_loss.npy'
# np.save(save_loss, total_loss1, allow_pickle=True)
# print(total_loss)
# print(loss[:5])
# quit()

# Visualizing the average losses at every epoch
fig, axs = plt.subplots(2, sharex=True, sharey=False)
losses = np.array(total_loss)
axs[0].plot(losses.T[0], label='Generator', color='Green')
axs[1].plot(losses.T[1], label='Discriminator', color='Red')
fig.suptitle("Training Losses")
axs[0].set(ylabel="Loss")
axs[1].set(xlabel="Epoch", ylabel="Loss")
fig.legend()
plt.savefig('{}loss_plot.png'.format(path))