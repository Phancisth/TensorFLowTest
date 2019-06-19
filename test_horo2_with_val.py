import matplotlib.pyplot as plt
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import keras
# This code has been tested with TensorFlow 1.6

from tensorflow.examples.tutorials.mnist import input_data


def accuracy(predictions, labels):
    '''
    Accuracy of a given set of predictions of size (N x n_classes) and
    labels of size (N x n_classes)
    '''
    return np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) * 100.0 / labels.shape[0]


batch_size = 100
layer_ids = ['hidden1', 'hidden2', 'out']
layer_sizes = [32, 64, 10]

hvd.init()

# Inputs and Labels
train_inputs = tf.placeholder(tf.float32, shape=[None, 32], name='train_inputs')
train_labels = tf.placeholder(tf.float32, shape=[None], name='train_labels')

# Weight and Bias definitions
for idx, lid in enumerate(layer_ids):
    with tf.variable_scope(lid):
        w = tf.get_variable('weights', shape=[layer_sizes[idx], layer_sizes[idx + 1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.05))
        b = tf.get_variable('bias', shape=[layer_sizes[idx + 1]],
                            initializer=tf.random_uniform_initializer(-0.1, 0.1))

# Calculating Logits
logits = train_inputs
for lid in layer_ids:
    with tf.variable_scope(lid, reuse=True):
        w, b = tf.get_variable('weights'), tf.get_variable('bias')
        if lid != 'out':
            logits = tf.nn.relu(tf.matmul(h, w) + b, name=lid + '_output')
        else:
            logits = tf.nn.xw_plus_b(h, w, b, name=lid + '_output')

tf_predictions = tf.nn.softmax(h, name='predictions')

# Calculating Loss
tf_loss = tf.losses.softmax_cross_entropy(train_labels, logits)

# Optimizer
tf_learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

global_step = tf.train.get_or_create_global_step()

optimizer = tf.train.RMSPropOptimizer(0.001 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)
train_op = optimizer.minimize(tf_loss, global_step=global_step)
grads_and_vars = optimizer.compute_gradients(tf_loss)
tf_loss_minimize = optimizer.minimize(tf_loss)

# Name scope allows you to group various summaries together
# Summaries having the same name_scope will be displayed on the same row
with tf.name_scope('performance'):
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    # Create a scalar summary object for the loss so it can be displayed
    tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

    # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
    tf_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
    # Create a scalar summary object for the accuracy so it can be displayed
    tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

# Gradient norm summary
for g, v in grads_and_vars:
    if 'hidden2' in v.name and 'weights' in v.name:
        with tf.name_scope('gradients'):
            tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g ** 2))
            tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
            break
# Merge all summaries together
performance_summaries = tf.summary.merge([tf_loss_summary, tf_accuracy_summary])

image_size = 28
n_channels = 1
n_classes = 10
n_train = 55000
n_valid = 5000
n_test = 10000
n_epochs = 25


# Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.

cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
if not os.path.exists(cache_dir):
    try:
        os.mkdir(cache_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
            pass
        else:
            raise


# Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
# Horovod: adjust number of steps based on number of GPUs.
hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.StopAtStepHook(last_step=20000 // hvd.size()),
        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': tf_loss},every_n_iter=10),
]


# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Horovod checkpoints
checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None

# Start session
session = tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,hooks=hooks,config=config)

tf.global_variables_initializer().run()

accuracy_per_epoch = []
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

for epoch in range(n_epochs):
    loss_per_epoch = []
    for i in range(n_train // batch_size):

        # =================================== Training for one step ========================================
        batch = mnist_data.train.next_batch(batch_size)  # Get one batch of training data
        if i == 0:
            # Only for the first epoch, get the summary data
            # Otherwise, it can clutter the visualization
            l, _, gn_summ = session.run(
                train_op,
                feed_dict={train_inputs: batch[0].reshape(batch_size, image_size * image_size),
                train_labels: batch[1],
                tf_learning_rate: 0.001 * hvd.size()})

            summ_writ1er.add_summary(gn_summ, epoch)
        else:
            # Optimize with training data
            l, _ = session.run(
                train_op,
                feed_dict={train_inputs: batch[0].reshape(batch_size, image_size * image_size),
                train_labels: batch[1],
                tf_learning_rate: 0.001 * hvd.size()})
        loss_per_epoch.append(l)

    print('Average loss in epoch %d: %.5f' % (epoch, np.mean(loss_per_epoch)))
    avg_loss = np.mean(loss_per_epoch)

    # ====================== Calculate the Validation Accuracy ==========================
    valid_accuracy_per_epoch = []
    for i in range(n_valid // batch_size):
        valid_images, valid_labels = mnist_data.validation.next_batch(batch_size)
        valid_batch_predictions = session.run(
            tf_predictions, feed_dict={train_inputs: valid_images.reshape(batch_size, image_size * image_size)})
        valid_accuracy_per_epoch.append(accuracy(valid_batch_predictions, valid_labels))

    mean_v_acc = np.mean(valid_accuracy_per_epoch)
    print('\tAverage Valid Accuracy in epoch %d: %.5f' % (epoch, np.mean(valid_accuracy_per_epoch)))

    # ===================== Calculate the Test Accuracy ===============================
    accuracy_per_epoch = []
    for i in range(n_test // batch_size):
        test_images, test_labels = mnist_data.test.next_batch(batch_size)
        test_batch_predictions = session.run(
            tf_predictions, feed_dict={train_inputs: test_images.reshape(batch_size, image_size * image_size)}
        )
        accuracy_per_epoch.append(accuracy(test_batch_predictions, test_labels))

    print('\tAverage Test Accuracy in epoch %d: %.5f\n' % (epoch, np.mean(accuracy_per_epoch)))
    avg_test_accuracy = np.mean(accuracy_per_epoch)

    # Execute the summaries defined above
    summ = session.run(train_op, feed_dict={tf_loss_ph: avg_loss, tf_accuracy_ph: avg_test_accuracy})

    # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
    summ_writer.add_summary(summ, epoch)

session.close()


