import os
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PYTHONHASHSEED']=str(42)
import tensorflow as tf
from Loca import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from visualization import *
import random as python_random


np.random.seed(42)
python_random.seed(42)
tf.compat.v1.set_random_seed(42)

tf.config.list_physical_devices('GPU')
# To change gpu device:
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU')


print('debug checkpoint')

# Load the input tensor:

# save_dir = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Denoising_DCGAN/Spatial_Reconstruction/'
save_dir = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Pix2Pix/Pix2Pix_lambda_recon=200/Spatial_Reconstruction/'
burst_loc_dir = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/cleaned_results/'
# load_dir = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Denoising_DCGAN/Cleaned_Test_Data_beta=0.160_src=[3,3,1.2]/'
load_dir = '/home/dsi/idancohen/Pytorch_Projects/GenerativeModelsProject/Pix2Pix/Pix2Pix_lambda_recon=200/Cleaned_Test_Data_beta=0.160_src=[3,3,1.2]/'

data = np.load(load_dir + '/' + 'data_phase_epoch80.npy')
burst_locations = np.load(burst_loc_dir + '/burst_locations.npy')


show_data(data[0:110])


# Divide into training and validation:

N, M, d = data.shape
indices = np.random.permutation(N)
indices_train, indices_val = indices[:N * 9 // 10], indices[N * 9 // 10:]
#
data_train = data[indices_train, :, :]
data_val = data[indices_val, :, :]

# Predict variance:
pred_var = lambda radius, count : radius ** 2 * (count / (count - 1) / 2)
burst_var = pred_var(0.033, 6)
burst_var = burst_var


# # Setting input parameters to the LOCA model
params = {}
params['bursts_var'] = burst_var

params['activation_enc'] = 'l_relu' # The activation function defined in the encoder
params['activation_dec'] = 'tanh' # The activation function defined in the decoder
params['dropout'] = None
params['l2_reg'] = False

params['encoder_layers'] = [d, 200,200,200, 200,200,2]  # The amount of neurons in each layer of the encoder
params['decoder_layers'] = [200, 200, 200, 200,200,d]  # The amount of neurons in each layer of the decoder


model = Loca(**params)

evaluate_every = 100
batch_size = 512
batch_size = 1024
batch_size = 2048
batch_size = 4096

large_lrs = [3e-3, 1e-3, 7e-4]
medium_lrs = [3e-4, 1e-4, 3e-5]
small_lrs = [1e-5, 3e-6, 1e-6]


for lr in large_lrs[0:1]:
    for i in range(1):
        model.train(data_train, amount_epochs=1000, lr=lr, batch_size=batch_size, evaluate_every=evaluate_every,
                    data_val=data_val, verbose=True, train_only_decoder=False, mutual_train=True, tol=None,
                    initial_training=False, whlr_reclr_ratio=1,
                    # save_best=False,)
                    save_best=True)

for lr in medium_lrs:
    for i in range(5):
        model.train(data_train, amount_epochs=1000, lr=lr, batch_size=batch_size,evaluate_every=evaluate_every,
                    data_val=data_val, verbose=True, train_only_decoder=False, mutual_train=True, tol=None,
                    initial_training=False, save_best=True, whlr_reclr_ratio=1)


model.best_white
model.best_rec
fig = visualize_embedding_xy_separated_2D(model, data, burst_locations, centers_only=False)
# visualize_embedding_xy_separated_2D(model, data, burst_locations, centers_only=True)
fig.savefig(save_dir + 'reconstruction_epoch_40')










