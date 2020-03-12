import os

from tensorflow.keras import layers
from tensorflow.keras import Model
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
  
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

def build_inception_deepNet(num_classes):
  local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

  pre_trained_model = InceptionV3(input_shape = (300, 300, 3), 
                                  include_top = False, 
                                  weights = None)

  pre_trained_model.load_weights(local_weights_file)

  for layer in pre_trained_model.layers:
    layer.trainable = False
    
  # pre_trained_model.summary()

  last_layer = pre_trained_model.get_layer('mixed7')
  print('last layer output shape: ', last_layer.output_shape)
  last_output = last_layer.output
  # Flatten the output layer to 1 dimension
  x = layers.Flatten()(last_output)
  # Add a fully connected layer with 1,024 hidden units and ReLU activation
  x = layers.Dense(1024, activation='relu')(x)
  # Add a dropout rate of 0.2
  x = layers.Dropout(0.2)(x)                  
  # Add a final sigmoid layer for classification
  x = layers.Dense  (num_classes, activation=None)(x)           
  #net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
  #model = Model( pre_trained_model.input, x) 

  return x
