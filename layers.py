import tensorflow, numpy

# Layers return their outputs, states (if recurrent), and trainable parameters

def reconv(data, state, filters, kernel_size, count):
  # The layer takes input from both its own output (hence filters) and the input data.
  # The kernel's number of input elements is based on the kernel size, not the
  # featuremap/input size height/width.
  in_channels = data.get_shape().as_list()[-1]
  fan_in = (filters + in_channels) * kernel_size * kernel_size

  # Glorot initialization
  weights = tensorflow.get_variable(
    "recurrent_weights_" + str(count),
    initializer=numpy.random.normal(scale=numpy.sqrt(2 / (fan_in + filters)),
                                    size=(kernel_size, kernel_size,
                                          filters + in_channels, filters),
                                    ).astype('f'),
    dtype=tensorflow.float32,
  )
  biases = tensorflow.get_variable(
    name="recurrent_biases_" + str(count),
    initializer=numpy.zeros((1, filters), dtype=numpy.float32),
    dtype=tensorflow.float32,
  )

  def step(step_data, step_state):
    joined = tensorflow.concat([step_data, step_state[0]], axis=-1)
    output = tensorflow.tanh(
      tensorflow.nn.conv2d(joined,
                           filter=weights,
                           strides=1,
                           padding='SAME') + biases
    )
    return output, [output]

  # rnn() takes a function for each timestep, plus data and initial states.
  # Data should be [batch, time_steps, other_dims], but initial states omit the time
  # dimension, since all states after the initial will be produced by the step function
  _, features, new_state = tensorflow.keras.backend.rnn(
    step,
    data,
    [state],
    unroll=True  # Optimize by unrolling the loop rnn() runs
  )
  return features, new_state, [weights, biases]

def to_logits(data, outshape, batch_size, seq_len):
  # Flatten output featuremap
  flattened = tensorflow.reshape(data, (batch_size, seq_len, -1))
  fan_in = flattened.get_shape().as_list()[-1]

  weights = tensorflow.get_variable(
    "output_weights",
    initializer=numpy.random.normal(scale=numpy.sqrt(2 / (fan_in + outshape)),
                                    size=(fan_in, outshape)).astype('f'),
    dtype=tensorflow.float32
  )
  biases = tensorflow.get_variable(
    "output_biases",
    initializer=numpy.zeros((1, outshape), dtype=numpy.float32),
    dtype=tensorflow.float32,
  )
  outputs = tensorflow.matmul(flattened, weights) + biases
  return outputs, [weights, biases]
