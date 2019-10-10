from load_rat import wistar_rat_data
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

""" OK """
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # close the warning
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # use gpu 0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4  # use 40 % gpu memory

model_path = "C:/Users/student5/Desktop/practice/save/model.ckpt"  # save path

# Training Parameters
epoch = 100
total_batch = 400
batch_size = 305
display_step = 50
weight_decay = 0.00001

# Learning rate decay
learn_init = 0.05
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learn_init, global_step, total_batch, 0.9, staircase=True)

# placeholder
x = tf.placeholder(tf.float32, [None, 1, 1536, 1])
y = tf.placeholder(tf.float32, [None, 2])
is_training = tf.placeholder(tf.bool)

batch_norm_params = {
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training
}

train_data, train_label, test_data, test_label = wistar_rat_data()


# NN
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    weights_initializer=slim.variance_scaling_initializer(),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.dropout], keep_prob=0.7, is_training=is_training):
        with slim.arg_scope([slim.conv2d], kernel_size=[1, 3], stride=[1, 1], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], kernel_size=[1, 2], stride=[1, 2], padding='SAME'):
                # block 1
                conv1 = slim.repeat(x, 2, slim.conv2d, 32)  # 1x384x128
                max_pool1 = slim.max_pool2d(conv1)  # 1x192x128
                conv2 = slim.repeat(max_pool1, 2, slim.conv2d, 64)  # 1x48x256
                max_pool2 = slim.max_pool2d(conv2)  # 1x24x256
                # block 2

                # block 3
                flat = slim.flatten(max_pool2)  # 6144
                fc1 = slim.fully_connected(flat, 128)
                fc2 = slim.fully_connected(fc1, 2, activation_fn=None)

prediction = tf.nn.softmax(fc2)
# Test accuracy
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# loss function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2, labels=y))

# Batch normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

""" train """
best_test_accuracy = 0
count = 0
counter = 0

with tf.Session(config=config) as sess:
    sess.run(init)

    for epoch in range(1, epoch+1):
        result_train = 0
        result_test = 0
        print("--------------Epoch" + str(epoch) + "----------------")

        # shuffle every epoch
        train_data, train_label = shuffle(train_data, train_label)

        for step in range(1, total_batch+1):
            sess.run(train_op, feed_dict={x: train_data[0 + batch_size * (step - 1):
                                                        batch_size + batch_size * (step - 1)],
                                          y: train_label[0 + batch_size * (step - 1):
                                                         batch_size + batch_size * (step - 1)],
                                          is_training: True})

            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: train_data[0 + batch_size * (step - 1):
                                                                  batch_size + batch_size * (step - 1)],
                                                    y: train_label[0 + batch_size * (step - 1):
                                                                   batch_size + batch_size * (step - 1)],
                                                    is_training: False})
                print("Step " + str(step) + ", Training Accuracy= " + "{:.4f}".format(acc))

        for i in range(122):
            result_train_1 = sess.run(accuracy,
                                      feed_dict={x: train_data[i*1000:(i+1)*1000],
                                                 y: train_label[i*1000:(i+1)*1000],
                                                 is_training: False})
            result_train = result_train + result_train_1 * (1 / 122)

        for i in range(10):
            result_test_1 = sess.run(accuracy,
                                     feed_dict={x: test_data[i*1000:(i+1)*1000],
                                                y: test_label[i*1000:(i+1)*1000],
                                                is_training: False})
            result_test = result_test + result_test_1 * (1/10)

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

    for i in range(122):
        result_train_1 = sess.run(accuracy,
                                  feed_dict={x: train_data[i * 1000:(i + 1) * 1000],
                                             y: train_label[i * 1000:(i + 1) * 1000],
                                             is_training: False})
        result_train = result_train + result_train_1 * (1 / 122)

    for i in range(10):
        result_test_1 = sess.run(accuracy,
                                 feed_dict={x: test_data[i*1000:(i+1)*1000],
                                            y: test_label[i*1000:(i+1)*1000],
                                            is_training: False})
    print("Best testing  accuracy=" + "{:.4f}".format(result_test))
    print("Best training accuracy=" + "{:.4f}".format(result_train))
