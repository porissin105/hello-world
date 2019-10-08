# hello-world
tensorflow vgg

from __future__ import division, print_function, absolute_import
from data import load_vgg
import tensorflow as tf
import numpy as np
import os
import time
import tensorflow.contrib.slim as slim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # close the warning
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # use gpu 0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4  # use 40 % gpu memory

model_path = "C:\\Users\student4\Desktop\Wistar_Rat"  # save path

# Training Parameters
data_sample = 122332
epoch = 100                  # 100 runs with 55000 train data
total_batch = 400            # In each runs , train data 55000 = 200 x 275 (total_batch x batch_size)
batch_size = 305
display_step = 50
keep_prob = 0.7

# Learning rate decay
learn_init = 0.005
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learn_init, global_step, total_batch, 0.9, staircase=True)

# Shuffle coefficient for all data
allran_num = np.arange(0,data_sample)
np.random.shuffle(allran_num)
allInt = allran_num.astype(int)
Indexall = allInt.tolist()
# Shuffle coefficient in train step
ran_num = np.arange(0,data_sample)
np.random.shuffle(ran_num)
Int = ran_num.astype(int)
Index = Int.tolist()

# Load data
trainData, trainlabel, testData, testlabel = load_vgg()

# print(trainData.shape)
# print(trainlabel.shape)
# Shuffle
trainData=trainData[Indexall]
trainlabel=trainlabel[Indexall]
# print(trainData)
# placeholder
X = tf.placeholder(tf.float32, [None, 1,1536,1])
Y = tf.placeholder(tf.float32, [None, 2])
is_training = tf.placeholder(tf.bool)


x = tf.convert_to_tensor(X, np.float32)
weight_decay = 0.00001
batch_norm_decay = 0.997
batch_norm_epsilon = 1e-5
batch_norm_scale = True
batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training,
  }


with slim.arg_scope([slim.conv2d, slim.fully_connected],
      weights_regularizer = slim.l2_regularizer(weight_decay),
      weights_initializer = slim.variance_scaling_initializer(),
      activation_fn = tf.nn.relu,
      normalizer_fn = slim.batch_norm,
      normalizer_params = batch_norm_params):
    with slim.arg_scope([slim.dropout], keep_prob=0.7, is_training=is_training):
        with slim.arg_scope([slim.max_pool2d], kernel_size=[1, 2], stride=[2, 2]):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                conv1 = slim.repeat(x, 3, slim.conv2d, 16, [1, 3])
                pool1 = slim.max_pool2d(conv1)
                conv2 = slim.repeat(pool1, 3, slim.conv2d, 32, [1, 3])
                pool2 = slim.max_pool2d(conv2)
                conv3 = slim.repeat(pool2, 3, slim.conv2d, 64, [1, 3])
                pool3 = slim.max_pool2d(conv3)
                flatten = slim.flatten(pool3)
                fc1 = slim.fully_connected(flatten, 1024)
                drop1 = slim.dropout(fc1)
                fc2 = slim.fully_connected(drop1, 1024)
                drop2 = slim.dropout(fc2)
                output = slim.fully_connected(drop2, 2, activation_fn=None)


# Output layer & Accuracy
prediction = tf.nn.softmax(output)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Batch_normalization tensorflow official
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

best_test_accuracy = 0
count = 0
counter = 0

with tf.Session(config=config) as sess:
    sess.run(init)

    for epoch in range(1, epoch+1):

        result_train = 0
        result_test = 0
        print("--------------Epoch" + str(epoch) + "----------------")
        trainData = trainData[Index]
        trainlabel = trainlabel[Index]
        for step in range(1, total_batch+1):
            sess.run(train_op, feed_dict={X: trainData[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
                                          Y: trainlabel[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
                                          is_training:True})

            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={X: trainData[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
                                                    Y: trainlabel[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
                                                    is_training: False})
                print("Step " + str(step) + ", Training Accuracy= " + "{:.4f}".format(acc))


        for x in range(100):
            result_train_1=sess.run(accuracy,
                                    feed_dict={X: trainData[x*100:(x+1)*100],
                                               Y: trainlabel[x*100:(x+1)*100],
                                               is_training: False})
            result_train = result_train + result_train_1 * (1 / 100)
        for y in range(100):
            result_test_1 = sess.run(accuracy,
                                    feed_dict={X: testData[y*100:(y+1)*100],
                                               Y: testlabel[y*100:(y+1)*100],
                                               is_training: False})
            result_test = result_test + result_test_1 * (1 / 100)
        # early stop
        count = count + 1
        counter = counter + 1
        if best_test_accuracy <= result_test and result_train - result_test < 0.1:
            best_test_accuracy = result_test
            print("***** best test  accuracy :" + "{:.4f}".format(best_test_accuracy) + " ****")
            print("***** best train accuracy :" + "{:.4f}".format(result_train) + " ****")
            count = 0
            save_path = saver.save(sess, model_path, write_meta_graph=False)
        else:
            print("Testing  accuracy=" + "{:.4f}".format(result_test))
            print("Training accuracy=" + "{:.4f}".format(result_train))

        if count == 20:
            break
        if result_train - result_test < 0.1:
            counter = 0
        if counter == 20:
            break

# Restore weights & biases variable
print("----------------------Restore------------------------------")
with tf.Session() as sess:

    saver.restore(sess, model_path)
    result_test = 0
    result_train = 0

    for x in range(100):
        result_train_1 = sess.run(accuracy,
                                  feed_dict={X: trainData[x * 100:(x + 1) * 100],
                                             Y: trainlabel[x * 100:(x + 1) * 100],
                                             is_training: False})
        result_train = result_train + result_train_1 * (1 / 100)
    for y in range(100):
        result_test_1 = sess.run(accuracy,
                                  feed_dict={X: testData[y * 100:(y + 1) * 100],
                                             Y: testlabel[y * 100:(y + 1) * 100],
                                      is_training: False})
        result_test = result_test + result_test_1 * (1 / 100)
    print("Best testing  accuracy=" + "{:.4f}".format(result_test))
    print("Best training accuracy=" + "{:.4f}".format(result_train))
