import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from torchvision.io import read_image
import torch.nn.functional as F


# DEFINE CONSTANTS

# Root directory for dataset
root_dir = 'Data/room=[6,6,2.4]'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# DEFINE DATASET

class CustomImageDataset(Dataset):
    """"""
    def __init__(self, root_dir, is_test=False, normalize=False, split_parts=False):
        """
        :param root_dir:
        :param is_test:
        :param normalize: If not False, should be a nd.array of the form
         np.array([mean_left,std_left,mean_right,std_right])
        """
        dirs = os.listdir(root_dir)
        dirs = [dir for dir in dirs if dir.startswith('Test') == is_test]
        self.revs = None
        self.targets = None

        for dir in dirs:
            rev_target_dirs = os.listdir(os.path.join(root_dir, dir))
            rev_dir = [d for d in rev_target_dirs if d.startswith('beta = 0.160')][0]
            target_dir = [d for d in rev_target_dirs if d.startswith('Target')][0]
            rev = os.path.join(root_dir, dir, rev_dir, 'data_phase.npy' if is_test else 'data_phase.npy')
            a = np.load(rev)

            if self.revs is None:
                self.height = a.shape[1]
                self.width = a.shape[2]
                if split_parts:
                    self.width = self.width // 2

            if normalize is not False:
                a[:,:,:self.width//2] = (a[:,:,:self.width//2] - normalize[0])/normalize[1]
                a[:,:,self.width//2:] = (a[:,:,self.width//2:] - normalize[2])/normalize[3]

            if split_parts:
                a = np.concatenate([a[:, :, :125], a[:, :, 125:]], axis=0)

            if self.revs is None:
                self.revs = np.copy(a)
            else:
                self.revs = np.vstack((self.revs, a))
            target = os.path.join(root_dir, dir, target_dir, 'data_phase.npy')
            b = np.load(target)

            # Not sure if we need to scale the target as well
            # if normalize is not False:
            #     b[:,:,:self.width//2] = (b[:,:,:self.width//2] - normalize[0])/normalize[1]
            #     b[:,:,self.width//2:] = (b[:,:,self.width//2:] - normalize[2])/normalize[3]

            if split_parts:
                b = np.concatenate([b[:, :, :125], b[:, :, 125:]], axis=0)

            if self.targets is None:
                self.targets = np.copy(b)
            else:
                self.targets = np.vstack((self.targets, b))

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        reverberated_image = self.revs[idx, :, :] / np.pi
        target_image = self.targets[idx, :, :] / np.pi
        pad = 128 - self.width
        # reverberated_image = F.pad(reverberated_image, (pad, 0, 0, 0), mode='constant', value=0)
        # target_image = F.pad(target_image, (pad, 0, 0, 0), mode='constant', value=0)
        reverberated_image = np.pad(reverberated_image, ((0, 0), (pad, 0)), 'constant', constant_values=0)
        target_image = np.pad(target_image, ((0, 0), (pad, 0)), 'constant', constant_values=0)

        reverberated_image = reverberated_image.reshape(1, 7, -1)
        target_image = target_image.reshape(1, 7, -1)
        return torch.from_numpy(reverberated_image), torch.from_numpy(target_image)


#
# class CustomImageDataset(Dataset):
#     def __init__(self, root_dir, is_test=False, split_parts=False):
#         dirs = os.listdir(root_dir)
#         dirs = [dir for dir in dirs if dir.startswith('Test') == is_test]
#         self.revs = None
#         self.targets = None
#
#         for dir in dirs:
#             rev_target_dirs = os.listdir(os.path.join(root_dir, dir))
#             rev_dir = [d for d in rev_target_dirs if d.startswith('beta')][0]
#             target_dir = [d for d in rev_target_dirs if d.startswith('Target')][0]
#             rev = os.path.join(root_dir, dir, rev_dir, 'data_phase.npy' if is_test else 'Reverberated/data_phase.npy')
#             a = np.load(rev)
#             if split_parts:
#                 a = np.concatenate([a[:, :, :126], a[:, :, 126:]], axis=0)
#             if self.revs is None:
#                 self.revs = np.copy(a)
#             else:
#                 self.revs = np.vstack((self.revs, a))
#             target = os.path.join(root_dir, dir, target_dir, 'data_phase.npy')
#             b = np.load(target)
#             if split_parts:
#                 b = np.concatenate([b[:, :, :126], b[:, :, 126:]], axis=0)
#             if self.targets is None:
#                 self.targets = np.copy(b)
#             else:
#                 self.targets = np.vstack((self.targets, b))
#
#
#     def __len__(self):
#         return self.targets.shape[0]
#
#     def __getitem__(self, idx):
#         reverberated_image = self.revs[idx, :, :]
#         target_image = self.targets[idx, :, :]
#         return torch.from_numpy(reverberated_image), torch.from_numpy(target_image)


if __name__ == '__main__':
    # LOAD DATA

    training_data = CustomImageDataset(root_dir)
    test_data = CustomImageDataset(root_dir, is_test=True)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot a training image
    real_batch = next(iter(train_dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(real_batch[0][0, :, :])
    plt.show()