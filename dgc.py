import logging
import math
import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils
from datahandler import datashapes
from models import encoder, decoder


class DGC(object):

    def __init__(self, opts, tag):
        tf.reset_default_graph()
        logging.error('Building the Tensorflow Graph')
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.opts = opts

        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        self.add_inputs_placeholders()

        self.add_training_placeholders()
        sample_size = tf.shape(self.sample_points)[0]

        enc_mean, enc_sigmas = encoder(opts, inputs=self.sample_points,
                                       is_training=self.is_training, y=self.labels)

        enc_sigmas = tf.clip_by_value(enc_sigmas, -50, 50)
        self.enc_mean, self.enc_sigmas = enc_mean, enc_sigmas

        eps = tf.random_normal((sample_size, opts['zdim']),
                               0., 1., dtype=tf.float32)
        self.encoded = self.enc_mean + tf.multiply(
            eps, tf.sqrt(1e-8 + tf.exp(self.enc_sigmas)))
        # self.encoded = self.enc_mean + tf.multiply(
        #     eps, tf.exp(self.enc_sigmas / 2.))

        (self.reconstructed, self.reconstructed_logits), self.probs1 = \
            decoder(opts, noise=self.encoded,
                    is_training=self.is_training)
        self.correct_sum = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.probs1, axis=1), self.labels), tf.float32))
        (self.decoded, self.decoded_logits), _ = decoder(opts, reuse=True, noise=self.sample_noise,
                                                         is_training=self.is_training)

        self.loss_cls = self.cls_loss(self.labels, self.probs1)
        self.loss_mmd = self.mmd_penalty(self.sample_noise, self.encoded)
        self.loss_recon = self.reconstruction_loss(
            self.opts, self.sample_points, self.reconstructed)
        self.objective = self.loss_recon + opts['lambda'] * self.loss_mmd + self.loss_cls

        self.tag = tag

        logpxy = []
        dimY = opts['n_classes']
        N = sample_size
        S = opts['sampling_size']
        x_rep = tf.tile(self.sample_points, [S, 1, 1, 1])
        for i in range(dimY):
            y = tf.fill((N * S,), i)
            mu, log_sig = encoder(opts, inputs=x_rep, reuse=True, is_training=False, y=y)
            eps2 = tf.random_normal((N * S, opts['zdim']), 0., 1., dtype=tf.float32)
            z = mu + tf.multiply(
                eps2, tf.sqrt(1e-8 + tf.exp(log_sig)))
            z_sample = tf.random_normal((tf.shape(z)[0], opts['zdim']), 0., 1., dtype=tf.float32)

            (mu_x, _), logit_y = decoder(opts, reuse=True, noise=z, is_training=False)
            logp = -tf.reduce_sum((x_rep - mu_x) ** 2, axis=[1, 2, 3])
            log_pyz = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit_y)
            mmd_loss = self.mmd_penalty(z_sample, z)
            bound = 0.5 * logp + log_pyz + opts['lambda'] * mmd_loss
            bound = tf.reshape(bound, [S, N])
            bound = self.logsumexp(bound) - tf.log(float(S))
            logpxy.append(tf.expand_dims(bound, 1))
        logpxy = tf.concat(logpxy, 1)
        y_pred = tf.nn.softmax(logpxy)
        self.eval_probs = y_pred

        if opts['e_pretrain']:
            self.loss_pretrain = self.pretrain_loss()
        else:
            self.loss_pretrain = None

        self.add_optimizers()
        self.add_savers()

    def log_gaussian_prob(self, x, mu=0.0, log_sig=0.0):
        logprob = -(0.5 * np.log(2 * np.pi) + log_sig) \
                  - 0.5 * ((x - mu) / tf.exp(log_sig)) ** 2
        ind = list(range(1, len(x.get_shape().as_list())))
        return tf.reduce_sum(logprob, ind)

    def logsumexp(self, x):
        x_max = tf.reduce_max(x, 0)
        x_ = x - x_max
        tmp = tf.log(tf.clip_by_value(tf.reduce_sum(tf.exp(x_), 0), 1e-20, np.inf))
        return tmp + x_max

    def add_inputs_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        data = tf.placeholder(
            tf.float32, [None] + shape, name='real_points_ph')
        label = tf.placeholder(tf.int64, shape=[None], name='label_ph')
        noise = tf.placeholder(
            tf.float32, [None] + [opts['zdim']], name='noise_ph')

        self.sample_points = data
        self.sample_noise = noise
        self.labels = label

    def add_training_placeholders(self):
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.lr_decay = decay
        self.is_training = is_training

    def pretrain_loss(self):
        opts = self.opts
        mean_pz = tf.reduce_mean(self.sample_noise, axis=0, keepdims=True)
        mean_qz = tf.reduce_mean(self.encoded, axis=0, keepdims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        cov_pz = tf.matmul(self.sample_noise - mean_pz,
                           self.sample_noise - mean_pz, transpose_a=True)
        cov_pz /= opts['e_pretrain_sample_size'] - 1.
        cov_qz = tf.matmul(self.encoded - mean_qz,
                           self.encoded - mean_qz, transpose_a=True)
        cov_qz /= opts['e_pretrain_sample_size'] - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    def add_savers(self):
        saver = tf.train.Saver(max_to_keep=11)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        if self.enc_mean is not None:
            tf.add_to_collection('encoder_mean', self.enc_mean)
            tf.add_to_collection('encoder_var', self.enc_sigmas)
        tf.add_to_collection('encoder', self.encoded)
        tf.add_to_collection('decoder', self.decoded)

        self.saver = saver

    def cls_loss(self, labels, logits):
        return tf.reduce_mean(tf.reduce_sum(  # FIXME
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)))

    def mmd_penalty(self, sample_pz, sample_qz):
        opts = self.opts
        sigma2_p = 1.
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        Cbase = 2. * opts['zdim'] * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat

    def reconstruction_loss(self, opts, real, reconstr):
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.5 * tf.reduce_mean(loss)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        return tf.train.AdamOptimizer(lr, beta1=opts["adam_beta1"])

    def add_optimizers(self):
        opts = self.opts
        lr = opts['lr']
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        ae_vars = encoder_vars + decoder_vars

        # Auto-encoder optimizer
        opt = self.optimizer(lr, self.lr_decay)
        self.ae_opt = opt.minimize(loss=self.objective,
                                   var_list=encoder_vars + decoder_vars)

        # Encoder optimizer
        if opts['e_pretrain']:
            opt = self.optimizer(lr)
            self.pretrain_opt = opt.minimize(loss=self.loss_pretrain,
                                             var_list=encoder_vars)
        else:
            self.pretrain_opt = None
        if opts['LVO']:
            self.lvo_opt = opt.minimize(loss=self.objective, var_list=encoder_vars)

    def sample_pz(self, num=100, z_dist=None, labels=None):
        opts = self.opts
        if z_dist is None:
            mean = np.zeros(opts["zdim"])
            cov = np.identity(opts["zdim"])
            noise = np.random.multivariate_normal(mean, cov, num).astype(np.float32)
            return noise
        assert labels is not None
        means, covariances = z_dist
        noise = np.array([
            np.random.multivariate_normal(means[e], covariances[e])
            for e in labels
        ])
        return noise

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = 200
        batch_size = opts['e_pretrain_sample_size']
        for step in range(steps_max):
            train_size = data.num_points
            data_ids = np.random.choice(train_size, min(train_size, batch_size),
                                        replace=False)
            batch_images = data.data[data_ids].astype(np.float)
            batch_labels = data.labels[data_ids].astype(np.int64)
            batch_noise = self.sample_pz(batch_size)

            [_, loss_pretrain] = self.sess.run(
                [self.pretrain_opt,
                 self.loss_pretrain],
                feed_dict={self.sample_points: batch_images,
                           self.labels: batch_labels,
                           self.sample_noise: batch_noise,
                           self.is_training: True})

            if opts['verbose']:
                logging.error('Step %d/%d, loss=%f' % (
                    step, steps_max, loss_pretrain))

            if loss_pretrain < 0.1:
                break

    def augment_batch(self, x, y):
        class_cnt = self.class_cnt

        max_class_cnt = max(class_cnt)
        n_classes = len(class_cnt)
        x_aug_list = []
        y_aug_list = []
        aug_rate = self.opts['aug_rate']
        if aug_rate <= 0:
            return x, y
        aug_nums = [aug_rate * (max_class_cnt - class_cnt[i]) for i in range(n_classes)]
        rep_nums = [aug_num / class_cnt[i] for i, aug_num in enumerate(aug_nums)]
        for i in range(n_classes):
            idx = (y == i)
            if rep_nums[i] <= 0.:
                x_aug_list.append(x[idx])
                y_aug_list.append(y[idx])
                continue
            n_c = np.count_nonzero(idx)
            if n_c == 0:
                continue
            x_aug_list.append(
                np.repeat(x[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
            y_aug_list.append(
                np.repeat(y[idx], repeats=math.ceil(1 + rep_nums[i]), axis=0)[:math.floor(n_c * (1 + rep_nums[i]))])
        if len(x_aug_list) == 0:
            return x, y
        x_aug = np.concatenate(x_aug_list, axis=0)
        y_aug = np.concatenate(y_aug_list, axis=0)
        return x_aug, y_aug

    def train(self, data):
        opts = self.opts
        self.class_cnt = [np.count_nonzero(data.labels == n) for n in range(opts['n_classes'])]
        if opts['verbose']:
            logging.error(opts)
        losses = []
        losses_rec = []
        losses_match = []
        losses_cls = []

        batches_num = math.ceil(data.num_points / opts['batch_size'])
        self.num_pics = opts['plot_num_pics']
        self.sess.run(tf.global_variables_initializer())

        if opts['e_pretrain']:
            logging.error('Pretraining the encoder')
            self.pretrain_encoder(data)
            logging.error('Pretraining the encoder done.')

        self.start_time = time.time()
        counter = 0
        decay = 1.
        wait = 0

        for epoch in range(opts["epoch_num"]):
            # Update learning rate if necessary
            start_time = time.time()
            if opts['lr_schedule'] == "manual":
                if epoch == 30:
                    decay = decay / 2.
                if epoch == 50:
                    decay = decay / 5.
                if epoch == 100:
                    decay = decay / 10.
            elif opts['lr_schedule'] == "manual_smooth":
                enum = opts['epoch_num']
                decay_t = np.exp(np.log(100.) / enum)
                decay = decay / decay_t

            elif opts['lr_schedule'] != "plateau":
                assert type(opts['lr_schedule']) == float
                decay = 1.0 * 10 ** (-epoch / float(opts['lr_schedule']))

            # Save the model
            if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
                self.saver.save(self.sess,
                                os.path.join(opts['work_dir'],
                                             'checkpoints',
                                             'trained'),
                                global_step=counter)

            acc_total = 0.
            loss_total = 0.

            for it in tqdm(range(batches_num)):
                start_idx = it * opts['batch_size']
                end_idx = start_idx + opts['batch_size']
                batch_images = data.data[start_idx:end_idx]
                batch_labels = data.labels[start_idx:end_idx]
                if opts['augment_z'] is True:
                    batch_images, batch_labels = self.augment_batch(batch_images, batch_labels)
                train_size = len(batch_labels)
                batch_noise = self.sample_pz(len(batch_images))
                if opts['LVO'] is True:
                    _ = self.sess.run(self.lvo_opt, feed_dict={self.sample_points: batch_images,
                                                               self.sample_noise: batch_noise,
                                                               self.labels: batch_labels,
                                                               self.lr_decay: decay,
                                                               self.is_training: True})

                feed_d = {
                    self.sample_points: batch_images,
                    self.sample_noise: batch_noise,
                    self.labels: batch_labels,
                    self.lr_decay: decay,
                    self.is_training: True}

                (_, loss, loss_rec, loss_cls, loss_match, correct) = self.sess.run(
                    [self.ae_opt,
                     self.objective,
                     self.loss_recon,
                     self.loss_cls,
                     self.loss_mmd,
                     self.correct_sum],
                    feed_dict=feed_d)
                acc_total += correct / train_size
                loss_total += loss

                if opts['lr_schedule'] == "plateau":
                    if epoch >= 30:
                        if loss < min(losses[-20 * batches_num:]):
                            wait = 0
                        else:
                            wait += 1
                        if wait > 10 * batches_num:
                            decay = max(decay / 1.4, 1e-6)
                            logging.error('Reduction in lr: %f' % decay)
                            wait = 0

                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_match.append(loss_match)
                losses_cls.append(loss_cls)

                counter += 1

            # Print debug info
            now = time.time()
            # Auto-encoding test images
            [loss_rec_test, loss_cls_test] = self.sess.run(
                [self.loss_recon, self.loss_cls],
                feed_dict={self.sample_points: data.test_data[:self.num_pics],
                           self.labels: data.test_labels[:self.num_pics],
                           self.is_training: False})

            debug_str = 'EPOCH: %d/%d, BATCH/SEC:%.2f' % (
                epoch + 1, opts['epoch_num'],
                float(counter) / (now - self.start_time))
            debug_str += ' (TOTAL_LOSS=%.5f, RECON_LOSS=%.5f, ' \
                         'MATCH_LOSS=%.5f, ' \
                         'CLS_LOSS=%.5f, ' \
                         'RECON_LOSS_TEST=%.5f, ' \
                         'CLS_LOSS_TEST=%.5f, ' % (
                             losses[-1], losses_rec[-1],
                             losses_match[-1], losses_cls[-1], loss_rec_test, loss_cls_test)
            logging.error(debug_str)

            training_acc = acc_total / batches_num
            avg_loss = loss_total / batches_num
            print("Train loss: %.5f, Train acc: %.5f, Time: %.5f" % (avg_loss, training_acc, time.time() - start_time))

            if (self.opts['eval_strategy'] == 1 and (epoch + 1) % 5 == 0) or \
                    self.opts['eval_strategy'] == 2 and ((0 < epoch <= 20) or (epoch > 20 and epoch % 3 == 0)):
                self.evaluate(data, epoch)

            if epoch > 0 and epoch % 10 == 0:
                self.saver.save(self.sess,
                                os.path.join(opts['work_dir'],
                                             'checkpoints',
                                             'trained-final'),
                                global_step=epoch)

    def evaluate(self, data, epoch):
        batch_size = self.opts['batch_size'] // 10
        batches_num = math.ceil(len(data.test_data) / batch_size)
        probs = []
        for it in tqdm(range(batches_num)):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size
            prob = self.sess.run(
                self.eval_probs,
                feed_dict={self.sample_points: data.test_data[start_idx:end_idx],
                           self.is_training: False})
            probs.append(prob)
        probs = np.concatenate(probs, axis=0)
        predicts = np.argmax(probs, axis=-1)
        asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi = utils.get_test_metrics(data.test_labels, predicts)
        print("EPOCH: %d, ASCA=%.5f, PRE=%.5f, REC=%.5f, SPE=%.5f, F1_ma=%.5f, F1_mi=%.5f, G_ma=%.5f, G_mi=%.5f" % (
            epoch, asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi))
