from reader import CustomImageDataset
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from visualization import show_data
from model_dcgan import Generator, Discriminator
from tqdm import tqdm

def train(G, D, train_dataloader, optimizer_G, optimizer_D,
          epochs, device, gen_loss_type, content_loss_lambda=0.9):
    G.train()  # set to training mode
    D.train()
    D_losses, G_losses = [], []
    D_losses_vs_epochs, G_losses_vs_epochs = [], []
    criterion = nn.BCELoss()

    for i in tqdm(range(1, epochs + 1)):
        for batch_idx, (inp_noisy, inp_target) in enumerate(train_dataloader):
            # inp_noisy = torch.unsqueeze(inp_noisy,1).to(device, dtype=torch.float)
            inp_noisy = inp_noisy.to(device).float()
            inp_target = inp_target.to(device).float()
            # inp_target = torch.unsqueeze(inp_target,1).to(device, dtype=torch.float)
            G.zero_grad()

            if gen_loss_type == 'modified':
                y = torch.ones(inp_noisy.size(0), 1).to(device)
                G_out = G(inp_noisy)
                D_out = D(G_out)
                G_adv_loss = criterion(D_out, y)
                G_adv_loss.backward()

            elif gen_loss_type == 'standard':
                y = torch.zeros(inp_noisy.size(0), 1).to(device)  # zeros instead of ones
                G_out = G(inp_noisy)
                D_out = D(G_out)
                G_adv_loss = criterion(D_out, y)
                neg_G_loss = -G_adv_loss
                (neg_G_loss).backward()  # Minimizing the negative loss is as maximizing the loss
            else:
                raise ValueError('loss_type should be either \'standard\' or \'modified\' ')

            G_content_loss = torch.mean((G_out - inp_target).abs())
            G_loss = content_loss_lambda * G_content_loss + (1-content_loss_lambda) * G_adv_loss

            # gradient backprop & optimize ONLY G's parameters
            optimizer_G.step()
            G_losses.append(G_loss.item())

            # We can train the discriminator multiple times per each generator iteration, but we chose to do it once.
            for k in range(1):
                D.zero_grad()
                # train discriminator on target (real) samples
                y_real = torch.ones(inp_target.size(0), 1).to(device)

                D_out = D(inp_target)
                D_real_loss = criterion(D_out, y_real)

                # train discriminator on fake samples
                x_fake = G(inp_noisy)
                y_fake = torch.zeros(inp_noisy.size(0), 1).to(device)

                D_out = D(x_fake)
                D_fake_loss = criterion(D_out, y_fake)

                # Backprop
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                optimizer_D.step()

            D_losses.append(D_loss.item())

        print('Epoch[%d/%d]: Loss Disc: %.3f, Loss Gen: %.3f'
              % ((i), epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
        G_losses_vs_epochs.append(torch.mean(torch.FloatTensor(G_losses)))
        D_losses_vs_epochs.append(torch.mean(torch.FloatTensor(D_losses)))

    return G_losses_vs_epochs, D_losses_vs_epochs


def sample(input, device):
    G.eval()  # set to inference mode
    input = input.to(device).float()
    with torch.no_grad():
        generated = G(input).detach().cpu()
    return generated

print('checkpoint')

# DEFINE CONSTANTS
root_dir = '/home/dsi/idancohen/Master_Simulations/Generative_Models_Project_Fixed_Data/'
# Number of workers for dataloader
workers = 1
# Batch size during training
batch_size = 64
# Number of training epochs
num_epochs = 40
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.


# Set device:
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# Dataset and Dataloader
training_data = CustomImageDataset(root_dir, split_parts=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers)


# Plot some training images and their targets:
train_rev, train_target = next(iter(train_dataloader))
fig, axes = plt.subplots(10,2)

l = []
for i in range(10):
    l.append(train_rev[i].squeeze())
    l.append(train_target[i].squeeze())

for j, (ax, im) in enumerate(zip(axes.ravel(), l)):
    m=ax.imshow(im)
    m.set_clim(vmin=-1, vmax=1)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(m, cax=cbar_ax)
    # ax.axis('off')
    if j == 0:
        ax.set_title('Reverberated Data Samples')
    if j == 1:
        ax.set_title('Target Data Samples')
plt.show()


test_data = CustomImageDataset(root_dir, is_test=True, normalize=False, split_parts=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,  num_workers=workers)

G = Generator(device).to(device)
D = Discriminator(device).to(device)

optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)

G_losses, D_losses = train(G, D, train_dataloader, optimizer_G, optimizer_D, epochs=40,
                               device=device, gen_loss_type='modified', content_loss_lambda=0.9)

model_save_folder = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Denoising_DCGAN/'
G.save_gen(epoch=40, save_dir=model_save_folder)

# Learning curve
plt.figure()
plt.plot(G_losses)
plt.plot(D_losses)
plt.legend(['Generator', 'Discriminator'])
plt.title('Loss vs Epoch')

# Show inputs and their sampled outputs:
inp_noisy = next(iter(train_dataloader))[0]
show_data(np.squeeze(inp_noisy))



generated = sample(inp_noisy, device)
show_data(torch.squeeze(generated).cpu().detach().numpy())


test_data = CustomImageDataset(root_dir, is_test=True,split_parts=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


###############################
# Plot test images
###############################
test_rev, test_target = next(iter(test_loader))
test_generated = sample(test_rev,device)

fig, axes = plt.subplots(10,3)
l = []
for i in range(10):
    l.append(test_rev[i].cpu().squeeze())
    l.append(test_target[i].squeeze())
    l.append(test_generated[i].squeeze())

for j, (ax, im) in enumerate(zip(axes.ravel(), l)):
    m=ax.imshow(im)
    m.set_clim(vmin=-1, vmax=1)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(m, cax=cbar_ax)
    # ax.axis('off')
    if j == 0:
        ax.set_title('Reverberated Data Samples')
    if j == 1:
        ax.set_title('Target Data Samples')
    if j == 2:
        ax.set_title('Generated Data Samples')
plt.show()

###############################
# Apply generator on test data, and show the output:
###############################
data_loadpath = '/home/dsi/idancohen/Master_Simulations/Generative_Models_Project_Fixed_Data/Test:room=[6,6,2.4], src=[3,3,1.2]/beta = 0.160/data_phase.npy'
data = np.load(data_loadpath)
data_savepath = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Denoising_DCGAN/Cleaned_Test_Data_beta=0.160_src=[3,3,1.2]/'

data = data/np.pi

data_left = data[:,:,:125]
data_right = data[:,:,125:]

data_left = np.pad(data_left, ((0, 0),(0,0), (3, 0)), 'constant', constant_values=0)
data_right = np.pad(data_right, ((0, 0),(0,0), (3, 0)), 'constant', constant_values=0)

data_left_torch = torch.tensor(data_left).reshape([data_left.shape[0], 1, data_left.shape[1],data_left.shape[2]]).to(device)
data_right_torch = torch.tensor(data_right).reshape([data_left.shape[0], 1, data_left.shape[1],data_left.shape[2]]).to(device)

data_generated_left = np.zeros([data.shape[0],7,128])
data_generated_right = np.zeros([data.shape[0],7,128])
# Due to lack in gpu memory - apply in batches:
for k in range(data.shape[0]//batch_size):
    data_generated_left[k*batch_size:(k+1)*batch_size] = np.squeeze(G(data_left_torch[k*batch_size:(k+1)*batch_size].float()).detach().cpu().numpy())
    data_generated_right[k*batch_size:(k+1)*batch_size] = np.squeeze(G(data_right_torch[k*batch_size:(k+1)*batch_size].float()).detach().cpu().numpy())


data_generated = np.concatenate([data_generated_left,data_generated_right], axis=2)

# plt.figure()
fig, ax = plt.subplots(1,2)
ax[0].imshow(data_generated.reshape([-1,256])[:770])
ax[0].set_title('Generated Data')
# compare with target data:
target_data_path ='/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/root_dir_temp/Test:room=[6,6,2.4],src=[3,3,1.2]/Target (gpuRIR)/data_phase.npy'
target_data = np.load(target_data_path)
target_data = target_data / np.pi  # Apply the same normalization we did to training data
ax[1].imshow(target_data.reshape([-1,250])[:770])
ax[1].set_title('Target Data')
plt.tight_layout()

np.save(data_savepath + 'data_phase_epoch40.npy',data_generated)

# Load a saved generator:
gen_path = "/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Denoising_DCGAN/Gen_Epoch_0.pt"
G = Generator(device).to(device)
G.load_model(gen_path)