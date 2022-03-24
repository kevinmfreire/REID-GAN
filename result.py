import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Visualize one loss only
path = './rigan_model/'
total_loss = np.load(path + 'total_loss.npy', allow_pickle=True)
# total_loss = np.load(path + 'loss_arr.npy', allow_pickle=True)
# loss = np.load(path + 'loss_arr (1).npy', allow_pickle=True)
# total_loss1 = np.concatenate((total_loss, loss), axis=0)
# losses = np.array(total_loss1)
# save_loss = path + 'total_loss.npy'
# np.save(save_loss, losses[:51], allow_pickle=True)
# print(total_loss)
# print(loss[:5])
# quit()

# Visualize all losses
# mse_loss = np.load('./MSE/loss_arr.npy', allow_pickle=True)
# msessmp_loss = np.load('./MSESSMP/loss_arr.npy', allow_pickle=True)
# ncssmp = np.load('./NCSSMP/total_loss.npy', allow_pickle=True)
# ssmp_loss = np.load('./SSMP/loss_arr.npy', allow_pickle=True)

######################################### Visualizing the average losses at every epoch #####################################

fig, axs = plt.subplots(2, sharex=True, sharey=False)
losses = np.array(total_loss)
axs[0].plot(losses.T[0], label='Generator', color='Green')
axs[1].plot(losses.T[1], label='Discriminator', color='Red')
fig.suptitle("Training Losses")
axs[0].set(ylabel="Loss")
axs[1].set(xlabel="Epoch", ylabel="Loss")
fig.legend()
plt.savefig('{}loss_plot.png'.format(path))

# save_loss = './rigan_model/loss_arr.npy'
# np.save(save_loss, losses[:26], allow_pickle=True)

################################ Visual comparison of losses between different methods #######################################

# mse, msessmp, ncssmp, ssmp = np.array(mse_loss), np.array(msessmp_loss), np.array(ncssmp), np.array(ssmp_loss)
# fig, axs = plt.subplots(2, sharex=True, sharey=False)
# axs[0].plot(mse.T[0], label='M1', color='Green')
# axs[0].plot(msessmp.T[0], label='M2', color='Red')
# axs[0].plot(ncssmp.T[0], label='M3', color='Black')
# axs[0].plot(ssmp.T[0], label='M4', color='Blue')
# axs[1].plot(mse.T[1], color='Green')
# axs[1].plot(msessmp.T[1], color='Red')
# axs[1].plot(ncssmp.T[1], color='Black')
# axs[1].plot(ssmp.T[1], color='Blue')
# fig.suptitle("Training Losses")
# axs[0].set(ylabel="Loss")
# axs[1].set(xlabel="Epoch", ylabel="Loss")
# fig.legend()
# plt.savefig('{}loss_plot.png'.format(path))