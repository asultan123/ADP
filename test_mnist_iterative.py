from __future__ import print_function
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import cleverhans.tf2.attacks as attacks
# from cleverhans.utils_tf import model_eval
import os
from utils import *
from model import resnet_v1
from keras_wraper_ensemble import KerasModelWrapper

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 30:
        lr *= 1e-2
    elif epoch > 15:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def model_eval(sess, x, y, predictions, X_test=None, Y_test=None, feed=None, args=None):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """
    global _model_eval_cache
    args = _ArgsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument " "must be supplied.")

    # Define accuracy symbolically
    key = (y, predictions)
    if key in _model_eval_cache:
        correct_preds = _model_eval_cache[key]
    else:
        correct_preds = tf.equal(tf.argmax(y, axis=-1), tf.argmax(predictions, axis=-1))
        _model_eval_cache[key] = correct_preds

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        X_cur = np.zeros((args.batch_size,) + X_test.shape[1:], dtype=X_test.dtype)
        Y_cur = np.zeros((args.batch_size,) + Y_test.shape[1:], dtype=Y_test.dtype)
        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                _logger.debug("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            Y_cur[:cur_batch_size] = Y_test[start:end]
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)
            cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)

            accuracy += cur_corr_preds[:cur_batch_size].sum()

        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return accuracy


# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Computed depth from supplied model parameter n
n = 3
depth = n * 6 + 2
version = 1

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
print(model_type)
print('Attack method is %s'%FLAGS.attack_method)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
clip_min = 0.0
clip_max = 1.0
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    clip_min -= x_train_mean
    clip_max -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# Define input TF placeholder
# x = tf.compat.v1.placeholder(tf.float32, shape=(None, 28, 28, 1))
# y = tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes))
# sess = tf.compat.v1.Session()
# tf.compat.v1.keras.backend.set_session(sess)

# Prepare model pre-trained checkpoints directory.
save_dir = os.path.join(os.getcwd(), FLAGS.dataset+'_EE_LED_saved_models'+str(FLAGS.num_models)+'_lamda'+str(FLAGS.lamda)+'_logdetlamda'+str(FLAGS.log_det_lamda)+'_'+str(FLAGS.augmentation))
model_name = 'model.%s.h5' % str(FLAGS.epoch).zfill(3)
filepath = os.path.join(save_dir, model_name)
print('Restore model checkpoints from %s'% filepath)

# Prepare baseline model pre-trained checkpoints directory.
save_dir_baseline = os.path.join(os.getcwd(), FLAGS.dataset+'_EE_LED_saved_models'+str(FLAGS.num_models)+'_lamda0.0_logdetlamda0.0_'+str(FLAGS.augmentation))
model_name_baseline = 'model.%s.h5' % str(FLAGS.baseline_epoch).zfill(3)
filepath_baseline = os.path.join(save_dir_baseline, model_name_baseline)
print('Restore baseline model checkpoints from %s'% filepath_baseline)



#Creat model
model_input = Input(shape=input_shape)
model_dic = {}
model_out = []
for i in range(FLAGS.num_models):
    model_dic[str(i)] = resnet_v1(input=model_input, depth=depth, num_classes=num_classes, dataset=FLAGS.dataset)
    model_out.append(model_dic[str(i)][2])
model_output = tf.keras.layers.concatenate(model_out)
model = Model(inputs=model_input, outputs=model_output)
model_ensemble = tf.keras.layers.Average()(model_out)
model_ensemble = Model(inputs=model_input, outputs=model_ensemble)



#Creat baseline model
model_input_baseline = Input(shape=input_shape)
model_dic_baseline = {}
model_out_baseline = []
for i in range(FLAGS.num_models):
    model_dic_baseline[str(i)] = resnet_v1(input=model_input_baseline, depth=depth, dataset=FLAGS.dataset)
    model_out_baseline.append(model_dic_baseline[str(i)][2])
model_output_baseline = tf.keras.layers.concatenate(model_out_baseline)
model_baseline = Model(inputs=model_input_baseline, outputs=model_output_baseline)
model_ensemble_baseline = tf.keras.layers.Average()(model_out_baseline)
model_ensemble_baseline = Model(inputs=model_input_baseline, outputs=model_ensemble_baseline)



#Get individual models
wrap_ensemble = KerasModelWrapper(model_ensemble)
wrap_ensemble_baseline = KerasModelWrapper(model_ensemble_baseline)



#Load model
model.load_weights(filepath)
model_baseline.load_weights(filepath_baseline)



# Initialize the attack method
if FLAGS.attack_method == 'MadryEtAl':
    att = attacks.MadryEtAl(wrap_ensemble)
    att_baseline = attacks.MadryEtAl(wrap_ensemble_baseline)
elif FLAGS.attack_method == 'FastGradientMethod':
    att = attacks.FastGradientMethod(wrap_ensemble)
    att_baseline = attacks.FastGradientMethod(wrap_ensemble_baseline)
elif FLAGS.attack_method == 'MomentumIterativeMethod':
    att = attacks.MomentumIterativeMethod(wrap_ensemble)
    att_baseline = attacks.MomentumIterativeMethod(wrap_ensemble_baseline)
elif FLAGS.attack_method == 'BasicIterativeMethod':
    att = attacks.BasicIterativeMethod(wrap_ensemble)
    att_baseline = attacks.BasicIterativeMethod(wrap_ensemble_baseline)

# Consider the attack to be constant
eval_par = {'batch_size': 100}
eps_ = FLAGS.eps
print('eps is %.3f'%eps_)
if FLAGS.attack_method == 'FastGradientMethod':
    att_params = {'eps': eps_,
               'clip_min': clip_min,
               'clip_max': clip_max}
else:
    att_params = {'eps': eps_,
                'eps_iter': eps_*1.0/10,
               'clip_min': clip_min,
               'clip_max': clip_max,
               'nb_iter': 10}
adv_x = tf.stop_gradient(att.generate(x, **att_params))
adv_x_baseline = tf.stop_gradient(att_baseline.generate(x, **att_params))
preds = model_ensemble(adv_x)
preds_baseline = model_ensemble_baseline(adv_x_baseline)
acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
acc_baseline = model_eval(sess, x, y, preds_baseline, x_test, y_test, args=eval_par)
print('adv_ensemble_acc: %.3f adv_ensemble_baseline_acc: %.3f'%(acc,acc_baseline))