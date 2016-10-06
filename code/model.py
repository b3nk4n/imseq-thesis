import tensorflow as tf
import tensorlight as tl
from tf.contrib.layers import *

class ConvAutoencoderModel(tl.model.AbstractModel):    
  def __init__(self, weight_decay=0.0):
    super(ConvAutoencoderModel, self).__init__(weight_decay)
        
  %%@tl.utils.attr.override%%
  def inference(self, inputs, targets, feeds, is_train, device_scope, memory_device):
    with tf.variable_scope("Encoder"):
      conv1 = tl.network.conv2d("Conv1", inputs, 4, (5, 5), (2, 2),
                                   weight_init=xavier_initializer_conv2d(),
                                   regularizer=l2_regularizer(self.weight_decay),
                                   activation=tf.nn.relu)
      conv1_bn = batch_norm(conv1, is_training=is_train, scope="conv1_bn")
      conv2 = tl.network.conv2d("Conv2", conv1_bn, 8, (3, 3), (2, 2),
                                   weight_init=xavier_initializer_conv2d(),
                                   regularizer=l2_regularizer(self.weight_decay),
                                   activation=tf.nn.relu)
      conv2_bn = batch_norm(conv2, is_training=is_train, scope="conv2_bn")
      learned_rep = conv2_bn
    with tf.variable_scope("Decoder"):
      convt = tl.network.conv2d_transpose("Convt1", learned_rep, 4, (3, 3), (2, 2),
                                              weight_init=tl.init.bilinear_initializer(),
                                              regularizer=l2_regularizer(self.weight_decay),
                                              activation=tf.nn.relu)
      convt_bn = batch_norm(convt, is_training=is_train, scope="convt_bn")
      return tl.network.conv2d_transpose("Convt2", convt_bn, 1, (5, 5), (2, 2),
                                              weight_init=tl.init.bilinear_initializer(), 
                                              regularizer=l2_regularizer(self.weight_decay),
                                              activation=tf.nn.sigmoid)
    
  %%@tl.utils.attr.override%%
  def loss(self, predictions, targets, device_scope):
    return tl.loss.bce(predictions, targets)
    
  %%@tl.utils.attr.override%%
  def evaluation(self, predictions, targets, device_scope):
    psnr = tl.image.psnr(predictions, targets)
    sharpdiff = tl.image.sharp_diff(predictions, targets)
    ssim = tl.image.ssim(predictions, targets)
    return {"psnr": psnr, "sharpdiff": sharpdiff, "ssim": ssim}
