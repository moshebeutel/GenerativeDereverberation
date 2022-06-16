# See E. Peterfreund, O. Lindenbaum, F. Dietrich, T. Bertalan, M. Gavish, I.G. Kevrekidis and R.R. Coifman,
# "LOCA: LOcal Conformal Autoencoder for standardized data coordinates",
# https://arxiv.org/abs/2004.07234
#
#
# -----------------------------------------------------------------------------
# Author: Erez Peterfreund , Ofir Lindenbaum
#         erezpeter@cs.huji.ac.il  , ofir.lindenbaum@yale.edu , 2020
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import math
import tensorflow

if tensorflow.__version__[0] == '2':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

else:
    import tensorflow as tf

import numpy as np


# Validates whether all the values of x are in the range of [minVal,maxVal]
# x- should be a numpy array
def val_num(x, minVal=None, maxVal=None, errMsg=None):
    if (minVal is not None):
        if x < minVal:
            if errMsg is None:
                raise Exception('The value ' + str(x) + ' should be bigger then ' + str(minVal))
            raise Exception(errMsg + ' Actual val: ' + str(x))

    if not (maxVal is None):
        if x > maxVal:
            if errMsg is None:
                raise Exception('The value ' + str(x) + ' should be below ' + str(maxVal))
            raise Exception(errMsg + ' Actual val: ' + str(x))


# Calculates the covariance of the data
# x- N x M x d tensorflow tensor, where N is the amount of bursts, M is the
#                      amount of samples in a burst and d is the coordinates
def tf_cov(x):
    val_num(len(np.shape(x)), minVal=3, maxVal=3, errMsg='The data should be a 3-d tensor.')

    x_no_bias = x - tf.reduce_mean(x, axis=1, keepdims=True)

    cov_x = tf.matmul(tf.transpose(x_no_bias, [0, 2, 1]), x_no_bias) / tf.cast(tf.shape(x)[1] - 1, tf.float32)
    return cov_x


def get_activation_layer(inputTensor, currActivation):
    if currActivation == 'relu':
        outputTensor = tf.nn.relu(inputTensor)
    elif currActivation == 'l_relu':
        outputTensor = tf.nn.leaky_relu(inputTensor)
    elif currActivation == 'sigmoid':
        outputTensor = tf.nn.sigmoid(inputTensor)
    elif currActivation == 'tanh':
        outputTensor = tf.nn.tanh(inputTensor)
    elif currActivation == 'none':
        outputTensor = inputTensor
    else:
        raise Exception('Error: ' + str(model.act_type_dec) +
                        ' is not supported. The activations that are supported- relu,l_relu,tanh,sigmoid,none')

    return outputTensor


# The function returns the output of the net and its weights that are defined by the input args. The final layer of the
# network include only an affine transformation and no activation function.
# INPUT:
#         input_tensor- a tensor where its last dimension includes the coordinates of the different samples.
#         input_dim- a positive integer. The dimension of the data.
#         layers- a list that includes the amount of neurons in each layer
#         act_type- a string defining the activation function (see get_activation_layer())
#         amount_layers_created - an integer. The amount of layers created so far for the net.
#
# OUTPUT:
#         layer_out - a 3 dimensional tensor (? x ? x output dimension) of type tensorflow variable.
#                     This variable represents the final layer of the generated neural network
#         nnweights - A list that contains the weights and biases variables that were used in this network
#         rate - a dropout rate - number between 0 and 1.
#
def generateCoder(input_tensor, input_dim, layers, act_type, amount_layers_created=0, rate=None,
                  dec_1st_scaled=False, l2_reg=False):
    layer_out = input_tensor
    prev_node_size = input_dim
    nnweights = []

    for i in range(len(layers)):
        layer_name = 'layer' + str(amount_layers_created + i)
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable('weights', [prev_node_size, layers[i]], \
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

            biases = tf.get_variable('biases', [layers[i]], \
                                     initializer=tf.constant_initializer(0))

            nnweights.append(weights)
            nnweights.append(biases)

            layer_out = (tf.tensordot(layer_out, weights, axes=[[-1], [0]]) + biases)

            l2_weights_list= []
            if l2_reg:
                l2_weights_list.append(weights)
                # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in l2Loss_list) * lossL2

            # if i == 0 and dec_1st_scaled == True:
            #     layer_out = (tf.tensordot(layer_out / 3, weights, axes=[[-1], [0]]) + biases)
            #
            # else:
            #     layer_out = (tf.tensordot(layer_out, weights, axes=[[-1], [0]]) + biases)


            if i == 0 and dec_1st_scaled == True:
                # layer_out = tf.nn.tanh(layer_out/10)
                layer_out = layer_out  # Linear
            elif i < len(layers)-1:
                layer_out = get_activation_layer(layer_out, act_type)

            # The activation layer will not be used on the final layer
            # if i < len(layers) - 1:
            #     layer_out = get_activation_layer(layer_out, act_type)
            #


            prev_node_size = layers[i]
        if rate is not None:
            if i != len(layers)-1: # As long as we are not in the last layer of either the encoder or decoder
                with tf.variable_scope('dropout'+layer_name):
                    layer_out = tf.nn.dropout(layer_out, rate=rate)


    return layer_out, nnweights, l2_weights_list


# The function gets a Loca class object and generates a neural network that is saved in the net fields
def generateNeuralNet(net):
    net.LR = tf.placeholder(tf.float32, shape=(), name="init")

    net.X = tf.placeholder(tf.float32, [None, None, net.input_dim])

    net.embedding, net.nnweights_enc, net.l2_weights = generateCoder(net.X, net.input_dim, net.encoder_layers[1:], net.act_type, \
                                                     amount_layers_created=0, rate=net.dropout, l2_reg=net.l2_regularization)

    net.reconstruction, net.nnweights_dec, _ = generateCoder(net.embedding, net.embedding_dim, net.decoder_layers[1:], \
                                                          net.act_type_dec,
                                                          amount_layers_created=len(net.encoder_layers) - 1,
                                                          rate=net.dropout, dec_1st_scaled=True)


# The Tensorflow implementation of LOCA
class Loca(object):
    # The init function.
    # INPUT:
    #             - bursts_var - float32. The assumed variance of each burst ( see sigma squared in the paper).
    #             - encoder_layers - A list that includes the amount of neurons in each layer of the generated encoder.
    #             - decoder_layers - A list that includes the amount of neurons in each layer of the generated decoder.
    #             - activation_enc - 'relu'/'l_relu','sigmoid','tanh','none' - the activation function that will be used
    #                                 in the encoder
    #             - activation_dec(Optional) - 'relu'/'l_relu','sigmoid','tanh','none' - the activation function that
    #                                 will be used in the decoder. If not supplied activation_enc will be used.
    #             - dropout(Optional) - float32. A number between 0 and 1 that will define how much neurons in a layer
    #                                   are set to zero during training phase. (Same as tf.nn.droput.rate)

    def __init__(self, bursts_var, encoder_layers, decoder_layers, activation_enc,
                 activation_dec=None, dropout=None, l2_reg=False):

        # Neural net params
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dropout = None
        self.input_dim = self.encoder_layers[0]
        self.embedding_dim = self.encoder_layers[-1]
        self.output_dim = self.decoder_layers[-1]
        self.l2_regularization = False

        if self.decoder_layers[-1] != self.input_dim:
            raise Exception('The final layer of the decoder should have the same dimension as the input')

        self.act_type = activation_enc
        if activation_dec is not None:
            self.act_type_dec = activation_dec
        else:
            self.act_type_dec = self.act_type

        if dropout is not None:
            if dropout > 1 or dropout < 0:
                raise Exception('Dropout rate should be a number between 0 and 1')
            else:
                self.dropout = dropout

        if l2_reg is not False:
            if isinstance(l2_reg, (int, float)) and not isinstance(l2_reg, bool):  # if l2_reg is a number
                 self.l2_lambda = l2_reg  # L2 regularization lambda parameter
                 self.l2_regularization = True
            else:
                raise Exception('l2_reg should be either False or a number value for the regularization lambda parameter')


        self.nnweights_enc = []
        self.nnweights_dec = []

        # Loss related params
        self.burst_var = bursts_var

        # Training history
        self.epochs_done = 0
        self.lr_training = []

        self.best_weights = None  # Determined by the validation loss
        self.best_loss = np.inf
        self.best_rec = np.inf
        self.best_white = np.inf

        # My addition:
        self.train_white_losses_list = []
        self.train_recon_losses_list = []
        self.train_l2_losses_list = []
        self.val_white_losses_list = []
        self.val_recon_losses_list = []
        self.val_l2_losses_list = []

        # Generate neural network
        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            generateNeuralNet(self)

            # Cyclic loss:
            # self.embedding_mod2pi = tf.math.mod((tf.math.mod(self.embedding, 2 * math.pi)) + math.pi, 2 * math.pi) - math.pi
            # normalized_cov_code = tf_cov(self.embedding_mod2pi) / (self.burst_var) - \
            #                       tf.expand_dims(tf.eye(self.embedding_dim), axis=0)

            normalized_cov_code = tf_cov(self.embedding) / (self.burst_var) - \
                                  tf.expand_dims(tf.eye(self.embedding_dim), axis=0)

            self.white_loss = tf.reduce_sum(tf.reduce_mean(normalized_cov_code ** 2, axis=0))

            self.rec_loss = tf.reduce_mean(tf.reduce_sum((self.reconstruction - self.X) ** 2, axis=-1))

            # Cyclic loss:
            # self.rec_loss = tf.reduce_mean(tf.reduce_sum((tf.math.mod(self.reconstruction, 2*math.pi) - tf.math.mod(self.X, 2*math.pi)) ** 2, axis=-1))
            # self.rec_loss = tf.reduce_mean(tf.reduce_sum((tf.math.mod(tf.math.mod(self.reconstruction-self.X, 2*math.pi) + math.pi, 2*math.pi) - math.pi )** 2, axis=-1))

            # # import tensorflow as tf
            # a = tf.math.mod((tf.math.mod(3.01 - (-3.003), 2*math.pi)) + math.pi, 2*math.pi) - math.pi
            # a = tf.math.mod((tf.math.mod(1.0 - 2.0, 2 * math.pi)) + math.pi, 2 * math.pi) - math.pi
            # # a = tf.math.mod((tf.math.mod(3.11 - (-3.13), 2*3.14)) + math.pi, 2*math.pi) - math.pi
            # # # a = (tf.math.mod(3.11,2 * 3.14) - tf.math.mod(-3.13, 2 * 3.14))
            # # # a = (tf.math.mod(0.1,2 * 3.14) - tf.math.mod(-0.1, 2 * 3.14))
            # # # a = (tf.math.mod(3.14 + 3.11, 2 * 3.14) - tf.math.mod(3.14 - 3.13, 2 * 3.14))
            # with tf.compat.v1.Session() as sess:
            #     print(sess.run(a))

            # with tf.compat.v1.Session() as sess:
            #     a = (tf.math.mod(3.01 - (-3.003), 3.14))
            #     a.eval(sess)


            if self.l2_regularization:
                self.l2_loss = tf.add_n([tf.nn.l2_loss(weight_mat) for weight_mat in self.l2_weights]) * self.l2_lambda
                self.train_step_whitening = tf.train.AdamOptimizer(self.LR).minimize(self.white_loss + self.l2_loss,
                                                                                     var_list=self.nnweights_enc)
                self.train_step_recon = tf.train.AdamOptimizer(self.LR).minimize(self.rec_loss + self.l2_loss, \
                                                                                 var_list=self.nnweights_dec + self.nnweights_enc)
                self.total_loss = self.white_loss + self.rec_loss + self.l2_loss  # My

            else:
                self.train_step_whitening = tf.train.AdamOptimizer(self.LR).minimize(self.white_loss,
                                                                                     var_list=self.nnweights_enc)
                self.train_step_recon = tf.train.AdamOptimizer(self.LR).minimize(self.rec_loss, \
                                                                                 var_list=self.nnweights_dec + self.nnweights_enc)
                self.total_loss = self.white_loss + self.rec_loss  # My

            self.train_step_recon_dec = tf.train.AdamOptimizer(self.LR).minimize(self.rec_loss,
                                                                                 var_list=self.nnweights_dec)

            self.train_step_total = tf.train.AdamOptimizer(self.LR).minimize(self.total_loss,
                                                                             var_list=self.nnweights_dec + self.nnweights_enc)  # My


            init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

        self.sess.run(init_op)

    # The function trains the neural net on data_train. If evaluate_every is given the function evaluates the neural net's
    # performance on data_train and data_val(if given). If data_val and evaluate_every is given the function will restore
    # the best version of the neural network based before returning. The best version means the neural network with the lowest
    # loss on data_val throughout the evluations steps.
    #
    # data_train- a Nx M x d tensor of data, where N indicates the amount of bursts,
    #                                  M indicates the amount of points in each burst
    #                                  and d indicates the dimension of the each sample
    # amount_epochs-   int. The amount of epochs to run.
    # lr-          float32. The learning that will be used throughout the training
    # batch_size - int. The batch size that will be used in the GD.
    # data_val(Optional)-    Same as in train_data but for validation (the values of m and d of this
    #                        tensor should be the same as in train_data).
    # evaluate_every(Optional) - int. The amount of epochs that will be passed between the evaluation of the losses
    #                                 based on the training data (data_train) and validation data (data_val) if is given.
    # verbose(Optional):          Boolean - Enables the printing of the losses evaluated evalutate_every epochs.
    # train_only_decoder(Optional):    Boolean. If True the training will only apply optimize the reconstruction loss,
    #                                  and will update only the weights in the decoder.
    # save_best(Optional)-      Boolean - if True, best weights are kept in the end of training session, based on the
    #                           best validation data loss.
    # tol(Optional)-          int. If not 'None', tol will represent the amount of epochs we evaluate without seeing
    #                               improvement in the 'best_loss' parameter. Training will be stopped when the tolerance
    #                               is reached, and the best weights will be loaded. Use only when data_val and evaluate_every
    #                               parameters are defined.
    def train(self, data_train, amount_epochs, lr=0.1, batch_size=None, data_val=None, evaluate_every=None,
              verbose=False, train_only_decoder=False, save_best=True, tol=None, mutual_train=False,
              initial_training=False, whlr_reclr_ratio=1):

        N_train = data_train.shape[0]
        N_val = 0

        if data_val is not None:
            if data_val.shape[0] > 0:
                N_val = data_val.shape[0]

        if batch_size is None:
            batch_size = N_train

        val_num(len(np.shape(data_train)), minVal=3, maxVal=3, errMsg='data_train should be a 3d tensor')
        val_num(batch_size, minVal=1, errMsg='btach_size should be at least 1')

        k = 0  # tol index
        for epoch in range(amount_epochs):

            for i in range(0, N_train, batch_size):

                if i + batch_size <= N_train:
                    indexes = np.arange(i, i + batch_size)
                else:
                    indexes = np.mod(np.arange(i, i + batch_size), N_train)

                batch_xs = data_train[indexes, :, :]

                if mutual_train:
                    _ = self.sess.run([self.train_step_total], feed_dict={self.X: batch_xs, self.LR: lr})
                else:
                    if initial_training == False:
                        r = 2
                    else:
                        r = 90
                    if not train_only_decoder:
                        if epoch % r:  # happens r-1 times in r epochs
                            _ = self.sess.run([self.train_step_whitening], feed_dict={self.X: batch_xs, self.LR: lr})
                        else:  # happens 1 time in r epochs
                            _ = self.sess.run([self.train_step_recon], feed_dict={self.X: batch_xs, self.LR: lr/whlr_reclr_ratio})


                    else:  # train only the decoder using the reconstruction loss
                        _ = self.sess.run([self.train_step_recon_dec], feed_dict={self.X: batch_xs, self.LR: lr})

            # Evaluation stage
            if evaluate_every is not None:
                if (epoch + 1) % evaluate_every == 0 and (verbose or early_stopping):

                    overall_train_white_loss = 0.
                    overall_train_rec_loss = 0.
                    overall_train_l2_loss = 0.
                    # Train
                    for i in range(0, N_train, batch_size):
                        max_ind = np.min([N_train, i + batch_size])
                        batch_xs = data_train[i:max_ind, :, :]

                        if self.l2_regularization:
                            rec_loss_train, white_loss_train, l2_loss_train = \
                                self.sess.run([self.rec_loss, self.white_loss, self.l2_loss], feed_dict={ self.X: batch_xs})
                        else:
                            rec_loss_train, white_loss_train = self.sess.run([self.rec_loss, self.white_loss],
                                                                             feed_dict={self.X: batch_xs})

                        overall_train_white_loss += white_loss_train * (max_ind - i) / N_train
                        overall_train_rec_loss += rec_loss_train * (max_ind - i) / N_train
                        if self.l2_regularization:
                            overall_train_l2_loss += l2_loss_train * (max_ind - i) / N_train

                    self.train_white_losses_list.append(overall_train_white_loss)
                    self.train_recon_losses_list.append(overall_train_rec_loss)
                    if self.l2_regularization:
                        self.train_l2_losses_list.append(overall_train_l2_loss)

                    if overall_train_white_loss is np.nan or overall_train_rec_loss is np.nan:
                        print("Training stopped after Epoch:", '%04d' % (self.epochs_done + 1), ", after",
                              'loss reached value - \'nan\'')
                        self.epochs_done += 1
                        self.load_weights_lowest_val()

                    if data_val is not None:
                        # Validation
                        overall_val_white_loss = 0.
                        overall_val_rec_loss = 0.
                        overall_val_l2_loss = 0.
                        for i in range(0, N_val, batch_size):
                            max_ind = np.min([N_val, i + batch_size])
                            batch_xs = data_val[i:max_ind, :, :]
                            if self.l2_regularization:
                                rec_loss_val, white_loss_val, l2_loss_val =\
                                    self.sess.run([self.rec_loss, self.white_loss, self.l2_loss], feed_dict={ self.X: batch_xs})
                            else:
                                rec_loss_val, white_loss_val = self.sess.run([self.rec_loss, self.white_loss],
                                                                             feed_dict={self.X: batch_xs})

                            overall_val_white_loss += white_loss_val * (max_ind - i) / N_val
                            overall_val_rec_loss += rec_loss_val * (max_ind - i) / N_val
                            if self.l2_regularization:
                                overall_val_l2_loss += l2_loss_val * (max_ind - i) / N_val

                        self.val_white_losses_list.append(overall_val_white_loss)
                        self.val_recon_losses_list.append(overall_val_rec_loss)
                        if self.l2_regularization:
                            self.val_l2_losses_list.append(overall_val_l2_loss)

                    if verbose:
                        if self.l2_regularization==False:
                            overall_train_l2_loss = 0
                            overall_val_l2_loss = 0

                        if data_val is not None:
                            print("Epoch:", '%04d' % (self.epochs_done + 1), "Train : white=",
                                  "{:.5f}".format(overall_train_white_loss), \
                                  "rec={:.5f}".format(overall_train_rec_loss),\
                                  "l2={:.5f}".format(overall_train_l2_loss), "     Val: : white=",
                                  "{:.5f}".format(overall_val_white_loss), \
                                  "rec={:.5f}".format(overall_val_rec_loss),\
                                  "l2={:.5f}".format(overall_val_l2_loss))
                        else:
                            print("Epoch:", '%04d' % (self.epochs_done + 1), "Train : white=",
                                  "{:.5f}".format(overall_train_white_loss), \
                                  "rec={:.5f}".format(overall_train_rec_loss))

                    # Saves the best version of the neural network based on its validation loss
                    if data_val is not None:
                        if (self.best_loss > overall_val_rec_loss + overall_val_white_loss):

                            self.best_weights = self.get_current_weights()
                            self.best_loss = overall_val_rec_loss + overall_val_white_loss
                            self.best_rec = overall_val_rec_loss
                            self.best_white = overall_val_white_loss
                            k = 0
                        else:
                            k += 1
                            if k == tol:
                                print("Training stopped after Epoch:", '%04d' % (self.epochs_done + 1), ", after",
                                      '%d' % tol, "evaluations without improvement")
                                self.epochs_done += 1
                                self.load_weights_lowest_val()
                                return

            self.epochs_done += 1
        if data_val is not None and save_best:
            self.load_weights_lowest_val()

    def eval_whitening_loss(self, data, batch_size=100):
        val_num(len(np.shape(data)), minVal=3, maxVal=3, errMsg='data should be a 3d tensor')

        N = data.shape[0]
        overall_white_loss = 0.

        for i in range(0, N, batch_size):
            max_ind = np.min([N, i + batch_size])
            batch_xs = data[i:max_ind, :, :]
            white_loss = self.sess.run(self.white_loss, feed_dict={self.X: batch_xs})

            overall_white_loss += white_loss * (max_ind - i) / N
        return overall_white_loss

    def eval_recon_loss(self, data, batch_size=100):
        val_num(len(np.shape(data)), minVal=3, maxVal=3, errMsg='data should be a 3d tensor')

        N = data.shape[0]
        overall_recon_loss = 0.

        for i in range(0, N, batch_size):
            max_ind = np.min([N, i + batch_size])
            batch_xs = data[i:max_ind, :, :]
            recon_loss = self.sess.run(self.rec_loss, feed_dict={self.X: batch_xs})

            overall_recon_loss += recon_loss * (max_ind - i) / N
        return overall_recon_loss

    def eval_total_loss(self, data, batch_size=100):
        total_loss = self.eval_recon_loss(data, batch_size) + self.eval_whitening_loss(data, batch_size)
        return total_loss

    # The function inputs the data into the neural network and returns the embedding and reconstruction of the data.
    # INPUT:
    #         data-  a 2 or 3 dimensional tensor, where its last dimension indicates the coordinates of the data.
    #
    # OUTPUT:
    #         embedding - same structure as data. Includes the embedding of the different input datapoints.
    #         reconstruction - same structure as data. Includes the reconstruction of the different input datapoints.
    def test(self, data):
        val_num(len(np.shape(data)), minVal=2, maxVal=3, errMsg='The data should be a 2d or 3d tensor.')

        new_data = data + 0.
        if len(np.shape(data)) == 2:
            new_data = np.expand_dims(new_data, axis=1)

        embedding, reconstruction = self.sess.run([self.embedding, self.reconstruction], feed_dict={self.X: new_data})

        if len(np.shape(data)) == 2:
            return embedding[:, 0, :], reconstruction[:, 0, :]

        return embedding, reconstruction

        # The function inputs the data into the neural network and returns the embedding and reconstruction of the data.

    # INPUT:
    #         data-  a 2/3 dimensional tensor, where its last dimension indicates the coordinates of the data. It will include
    #                the input for the embedding layer of the neural network.
    #
    # OUTPUT:
    #         newData- me structure as data. Includes the reconstruction of the different input datapoints.
    def decode(self, data):
        val_num(len(np.shape(data)), minVal=2, maxVal=3, errMsg='The data should be a 2d or 3d tensor.')
        decoder_weights = self.get_current_weights()[1]

        newData = data + 0.
        if len(np.shape(data)) == 2:
            newData = np.expand_dims(newData, axis=1)

        for i in range(0, len(decoder_weights), 2):
            newData = np.matmul(newData, decoder_weights[i]) + decoder_weights[i + 1]

            # The last layer of the decoder is linear
            if i < len(decoder_weights) - 2:
                if self.act_type_dec == 'relu':
                    newData = np.maximum(newData, 0)

                elif self.act_type_dec == 'l_relu':
                    alpha = 0.2
                    newData = np.maximum(newData, 0) + alpha * np.minimum(newData, 0)
                elif self.act_type_dec == 'tanh':
                    newData = np.tanh(newData)
                elif self.act_type_dec == 'sigmoid':
                    newData = 1 / (1 + np.exp(-newData))
                elif self.act_type_dec == 'none':
                    newData = newData
                else:
                    raise Exception('Error: ' + str(self.act_type_dec) + \
                                    ' is not supported. The activations that are supported- relu,l_relu,tanh,sigmoid,none')

        if len(np.shape(data)) == 2:
            newData = newData[:, 0, :]
        return newData

    # The method updates the neural network weights with the weights of the neural network that achieved
    # the lowest validation loss throughout training
    def load_weights_lowest_val(self):
        self.load_weights(self.best_weights[0], self.best_weights[1])

    # The function returns the weights of the current neural network as two lists.
    # OUTPUT:
    #        we - A list that contains the neural network weights of the encdoer.
    #        wd - A list that contains the neural network weights of the decoder.
    def get_current_weights(self):
        we = self.sess.run(self.nnweights_enc)
        wd = self.sess.run(self.nnweights_dec)
        return we, wd

    # The function loads the given weights into the neural network.
    # INPUT:
    #         encoderWeights - A list containing the values of the encoder part of the neural network.
    #         decoderWeights - A list containing the values of the decoder part of the neural network.
    def load_weights(self, encoderWeights, decoderWeights):
        for i in range(len(encoderWeights)):
            self.assign_variable(self.nnweights_enc[i], encoderWeights[i])
            # self.sess.run(self.nnweights_enc[i].assign(encoderWeights[i]))

        for i in range(len(decoderWeights)):
            self.assign_variable(self.nnweights_dec[i], decoderWeights[i])
            # self.sess.run(self.nnweights_dec[i].assign(decoderWeights[i]))

    def assign_variable(self, variable, value):
        # if tensorflow.__version__[0] == '2':
        #     variable.assign(value)
        #     print('version is 2')
        # else:
        self.sess.run(variable.assign(value))
