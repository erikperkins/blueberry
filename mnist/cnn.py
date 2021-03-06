import os
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from numpy import random, expand_dims, uint8, array, shape, argmax
from PIL import Image
from PIL.ImageOps import expand

(_, (images, _)) = load_data(f'{os.getcwd()}/data/mnist.npz')

def digit_image(id):
  record = images[id]
  array = record.reshape((28, 28))
  return Image.fromarray(255 - array)

def mnist_input(id = None, buffer = None):
  if id:
    return images[id]
  if buffer:
    image = Image.open(buffer)
    normalized = normalize(image)

    image_array = array(normalized)[:,:,3] / 255.0
    return image_array.reshape((784,))

def normalize(image):
  cropped = image.crop(image.getbbox())
  padded = Image.new('RGBA', (max(cropped.size), max(cropped.size)))
  size = max(cropped.size)
  offset = (int((size - cropped.width)/2), int((size - cropped.height)/2))
  padded.paste(cropped, offset)

  normalized = expand(padded, int(padded.height/10))
  normalized.convert('LA')
  normalized.thumbnail((28, 28), Image.ANTIALIAS)
  return normalized

class Classifier():
  def __init__(self):
    tf.reset_default_graph()
    self.build()

  def build(self):
    self.PROB = tf.placeholder(tf.float32)

    def layer(inputs, kernel_shape, bias_shape, multiply, activate):
      self.weights = tf.get_variable(
        "weights",
        kernel_shape,
        initializer = tf.random_normal_initializer()
      )
      self.biases = tf.get_variable(
        "biases",
        bias_shape,
        initializer = tf.constant_initializer(0.0)
      )
      return activate(multiply(inputs, self.weights) + self.biases)

    def max_pool(h):
      return tf.nn.max_pool(
        h,
        ksize = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME'
      )

    def conv(inputs, weights):
      return tf.nn.conv2d(
        inputs,
        weights,
        strides = [1, 1, 1, 1],
        padding = 'SAME'
      )

    def flatten(h):
      return tf.reshape(h, [-1, 7 * 7 * 64])

    def dropout(h, keep_prob):
      return tf.nn.dropout(h, keep_prob)

    def apply_network(inputs):
      with tf.variable_scope("conv1") as scope:
        activate = lambda a: max_pool(tf.nn.relu(a))
        layer1 = layer(inputs, [5, 5, 1, 32], [32], conv, activate)
        scope.reuse_variables()

      with tf.variable_scope("conv2") as scope:
        activate = lambda a: flatten(max_pool(tf.nn.relu(a)))
        layer2 = layer(layer1, [5, 5, 32, 64], [64], conv, activate)
        scope.reuse_variables()

      with tf.variable_scope("conn1"):
        activate = lambda a: dropout(tf.nn.relu(a), self.PROB)
        layer3 = layer(layer2, [7 * 7 * 64, 1024], [1024], tf.matmul, activate)
        scope.reuse_variables()

      with tf.variable_scope("conn2"):
        activate = lambda a: a
        output_layer = layer(layer3, [1024, 10], [10], tf.matmul, activate)
        scope.reuse_variables()
        return output_layer

    self.x = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_ = tf.placeholder(tf.float32, shape = [None, 10])
    self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

    with tf.variable_scope("images") as scope:
      try:
        self.tensor = apply_network(self.x_image)
        self.graph = tf.get_default_graph()
      except ValueError:
        scope.reuse_variables()
        self.tensor = apply_network(self.x_image)
        self.graph = tf.get_default_graph()

    self.saver = tf.train.Saver()

  def classify(self, input):
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 2
    config.inter_op_parallelism_threads = 2

    with tf.Session(config = config, graph = self.graph) as session:
      session.graph.finalize()

      self.saver.restore(session, f'{os.getcwd()}/data/mnist.ckpt')
      datum = expand_dims(input, axis = 0).reshape(1, 784)
      output = session.run(
        self.tensor,
        feed_dict = { self.x: datum, self.PROB: 1.0 }
      )
      (classification,) = argmax(output, 1)

      return int(classification)
