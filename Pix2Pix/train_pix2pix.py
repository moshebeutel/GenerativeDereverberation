import numpy as np
from reader import CustomImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_pix2pix import _weights_init, Generator, PatchGAN
from tqdm.auto import tqdm

class Pix2Pix(nn.Module):

    def __init__(self, in_channels, out_channels, device, learning_rate=0.0002, lambda_recon=200):
        super().__init__()
        self.device = device
        self.gen = Generator(in_channels, out_channels).to(device)
        self.patch_gan = PatchGAN(in_channels + out_channels).to(device)
        self.lambda_recon = lambda_recon
        self.lr = learning_rate

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

        self.D_losses_vs_epochs = []
        self.G_losses_vs_epochs = []
        self.epoch = 0

    @property
    def generator(self):
        return self.gen

    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)
        disc_logits = self.patch_gan(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)

        return adversarial_loss + self.lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()

        # fake_images = self.gen(conditioned_images)
        fake_logits = self.patch_gan(fake_images, conditioned_images)
        real_logits = self.patch_gan(real_images, conditioned_images)
        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def configure_optimizers(self):
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        self.disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=self.lr)
        return self.disc_opt, self.gen_opt

    def train(self, device, train_dataloader, epochs, save_every=None, save_dir=None):

        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.patch_gan.requires_grad_(True)
        # self.gen.requires_grad_(True)
        self.gen.train()  # set to training mode
        self.patch_gan.train()  # set to training mode

        self.D_losses, self.G_losses = [], []

        for i in tqdm(range(1, epochs + 1), desc='Epoch Loop',position=0, leave=True):
            self.epoch += 1
            pbar = tqdm(enumerate(train_dataloader), desc='Mini-batch Loop', position=1, leave=True)
            for batch_idx, inputs in tqdm(enumerate(train_dataloader), desc='Mini-batch Loop',
                                          total=len(train_dataloader), position=1, leave=True):
                condition, real = inputs
                real = real.to(self.device)
                condition = condition.to(self.device)
                self.gen.zero_grad()
                self.patch_gan.zero_grad()

                # Train patch discriminator
                disc_loss = self._disc_step(real_images=real, conditioned_images=condition)
                disc_loss.backward()
                self.D_losses.append(disc_loss)
                self.disc_opt.step()

                # Train generator
                gen_loss = self._gen_step(real_images=real, conditioned_images=condition)
                gen_loss.backward()
                self.G_losses.append(gen_loss)
                self.gen_opt.step()

            print('Epoch[%d]: Loss Disc: %.3f, Loss Gen: %.3f'
                  % (self.epoch, torch.mean(torch.FloatTensor(self.D_losses)), torch.mean(torch.FloatTensor(self.G_losses))))

            self.G_losses_vs_epochs.append(torch.mean(torch.FloatTensor(self.G_losses)))
            self.D_losses_vs_epochs.append(torch.mean(torch.FloatTensor(self.D_losses)))

            if save_every is not None and not self.epoch % save_every:
                self.save_model(save_dir=save_dir)


    def save_model(self,save_dir=None, final=False):
        if save_dir is None:
            save_dir = ''
        elif not save_dir.endswith('/'):
            save_dir += '/'
        if final:
            save_dir += 'Final: '

        torch.save({'epoch': self.epoch, 'model_state_dict': self.gen.state_dict(),
                    'optimizer_state_dict': self.gen_opt.state_dict(), 'Gen_losses': self.G_losses_vs_epochs, },
                   save_dir + f'Gen_Epoch_{self.epoch}_loss_{self.G_losses_vs_epochs[-1]:.3f}.pt')
        torch.save({'epoch': self.epoch, 'model_state_dict': self.patch_gan.state_dict(),
                    'optimizer_state_dict': self.disc_opt.state_dict(), 'Disc_losses': self.D_losses_vs_epochs, },
                   save_dir + f'Disc_Epoch_{self.epoch}_loss_{self.D_losses_vs_epochs[-1]:.3f}.pt')
        return

    def load_model(self, gen_path, disc_path):
        """

        :param gen_path: path to generator state_dict
        :param disc_path: path to discriminator state_dict
        :return: The model generator, discriminator, optimizers and epoch attributes are loaded with the
        parameters from the state_dicts.
        """

        if not (hasattr(self, 'gen_opt') and hasattr(self, 'disc_opt')):
            self.configure_optimizers()

        gen_state_dict = torch.load(gen_path, map_location=device)
        self.epoch = gen_state_dict.get('epoch')
        self.gen.load_state_dict(gen_state_dict.get('model_state_dict'))
        self.gen_opt.load_state_dict(gen_state_dict.get('optimizer_state_dict'))
        self.G_losses_vs_epochs = gen_state_dict.get('Gen_losses')


        disc_state_dict = torch.load(disc_path, map_location=device)
        self.patch_gan.load_state_dict(disc_state_dict.get('model_state_dict'))
        self.disc_opt.load_state_dict(disc_state_dict.get('optimizer_state_dict'))
        self.D_losses_vs_epochs = disc_state_dict.get('Disc_losses')

        return


    def test(self, batch):
        """
        :param batch: batch of images loaded on device, to feed into the generator. Dims: [batch_size, 1, 7, 128]
        :return: output is returned on cpu
        """

        self.gen.eval()
        with torch.no_grad():
            generated = self.gen(batch)
            generated = generated.detach().cpu()
        return generated

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params


root_dir = '/home/dsi/idancohen/Master_Simulations/Generative_Models_Project_Fixed_Data/'
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Number of workers for dataloader
workers = 16
# Batch size during training
batch_size = 64
# Number of training epochs
num_epochs = 40
# Number of channels in the training images. For color images this is 3
nc = 1
# Learning rate for optimizer
lr = 0.0002
# Set device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Dataset and Dataloader
training_data = CustomImageDataset(root_dir, split_parts=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers)


# Plot a some training images and their targets:
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



adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 200

pix2pix = Pix2Pix(nc, nc,device=device, learning_rate=lr, lambda_recon=lambda_recon)
pix2pix.configure_optimizers()

save_path = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Pix2Pix/Pix2Pix_lambda_recon=200/'
# pix2pix.train(device,train_dataloader, epochs=num_epochs)
pix2pix.train(device,train_dataloader, epochs=num_epochs)
pix2pix.save_model(save_dir=save_path)


#### Plot Learning Curves:
# Plot Generator and Discriminator learning curves:
fig = plt.figure()
gen_losses = [a.cpu() for a in pix2pix.G_losses_vs_epochs]
disc_losses = [a.cpu() for a in pix2pix.D_losses_vs_epochs]
plt.plot(gen_losses)
plt.plot(disc_losses)
plt.legend(['Generator', 'Discriminator'])
plt.title('Loss vs Epoch')
fig.savefig(save_path + 'learning_curve_40.png')


test_data = CustomImageDataset(root_dir, is_test=True,split_parts=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

##### Plot test images
test_rev, test_target = next(iter(test_loader))
test_rev = test_rev.to(device)
test_generated = pix2pix.test(test_rev)
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
    if j == 0:
        ax.set_title('Reverberated Data Samples')
    if j == 1:
        ax.set_title('Target Data Samples')
    if j == 2:
        ax.set_title('Generated Data Samples')
plt.show()


############## Apply generator on test data, and show the output:
data_loadpath = '/home/dsi/idancohen/Master_Simulations/Generative_Models_Project_Fixed_Data/Test:room=[6,6,2.4], src=[3,3,1.2]/beta = 0.160/data_phase.npy'
data = np.load(data_loadpath)
data_savepath = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Pix2Pix/Pix2Pix_lambda_recon=200/Cleaned_Test_Data_beta=0.160_src=[3,3,1.2]/'

# Normalizing to range [-1,1]
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
    data_generated_left[k*batch_size:(k+1)*batch_size] = np.squeeze(pix2pix.gen(data_left_torch[k*batch_size:(k+1)*batch_size]).detach().cpu().numpy())
    data_generated_right[k*batch_size:(k+1)*batch_size] = np.squeeze(pix2pix.gen(data_right_torch[k*batch_size:(k+1)*batch_size]).detach().cpu().numpy())


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

# Save the cleaned data:
np.save(data_savepath + 'data_phase_epoch40.npy',data_generated)

##############################
# Load a Saved Model:
##############################

adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 200

pix2pix = Pix2Pix(nc, nc,device=device, learning_rate=lr, lambda_recon=lambda_recon)
pix2pix.configure_optimizers()

gen_path = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Pix2Pix/Pix2Pix_lambda_recon=200/Gen_Epoch_40_loss_13.066.pt'
disc_path = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Pix2Pix/Pix2Pix_lambda_recon=200/Disc_Epoch_40_loss_0.078.pt'

pix2pix.load_model(gen_path, disc_path)

