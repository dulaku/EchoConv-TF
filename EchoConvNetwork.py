import tensorflow, keras, numpy
from tensorflow.python.ops import nccl_ops
import matplotlib.pyplot as plt
import time

import config, layers, dataloader, plot

tensorflow_config = tensorflow.ConfigProto()
tensorflow_config.gpu_options.allow_growth = True  # Claim GPU memory as needed
session = tensorflow.Session(config=tensorflow_config)

#############################
# Build placeholder tensors #
#############################

input_placeholders = [
  [tensorflow.placeholder(
     tensorflow.float32,
     [config.BATCH_SIZE, config.SEQ_LEN, input_size, input_size, 1]
   ) for input_size in config.INPUTS
  ] for gpu in range(config.GPUS)
]

# State at each layer
state_placeholders = [
  [tensorflow.placeholder(
     tensorflow.float32,
     [config.BATCH_SIZE, state_size[0], state_size[0], state_size[1]]
   ) for state_size in config.STATES
  ] for gpu in range(config.GPUS)
]

# Classifier targets - no shape for the target shape since it's just an int
target_placeholders = [
  [tensorflow.placeholder(
    tensorflow.int32,
    [config.BATCH_SIZE, config.SEQ_LEN]
  )] for gpu in range(config.GPUS)
]

#################################
# MODEL ARCHITECTURE DEFINITION #
#################################
def build_model(gpu):
  if gpu != 0:
    reuse = True
  else:
    reuse = False
  with tensorflow.device('/gpu:' + str(gpu)),\
       tensorflow.variable_scope(tensorflow.get_variable_scope(), reuse=reuse):
    states, variables = [], []

    # Build first layer from inputs
    features, state, params = layers.reconv(
      input_placeholders[gpu][0],  # Input (placeholder tensor)
      state_placeholders[gpu][0],  # States (list of placeholder tensors)
      config.FILTERS[0],           # Filter count (int)
      config.KERNEL_SIZES[0],      # Kernel size  (int)
      0                            # Layer index (int)
    )
    states.append(state)
    variables.append(params)

    # Loop over further layers
    for layer in range(1, len(config.FILTERS)):
      features, state, params = layers.reconv(
        features,                        # Input (real tensor)
        state_placeholders[gpu][layer],  # States (list of placeholder tensors)
        config.FILTERS[layer],           # Filter count (int)
        config.KERNEL_SIZES[layer],      # Kernel size (int)
        layer                            # Layer index (int)
      )
      states.append(state)
      variables.append(params)
    scores, params = layers.to_logits(features,
                                      config.CLASSES[0],  # Number of classes (int)
                                      config.BATCH_SIZE,  # Batch size (int)
                                      config.SEQ_LEN)     # Sequence length (int)
    variables.append(params)

    loss = tensorflow.reduce_mean(
        tensorflow.nn.sparse_softmax_cross_entropy_with_logits(
          logits=scores, labels=target_placeholders[gpu][0]
        )
      )

    metric = tensorflow.contrib.metrics.accuracy(
      labels=target_placeholders[gpu][0],
      predictions=tensorflow.argmax(scores, axis=-1, output_type=tensorflow.int32)
    )

  return scores, loss, states, variables, metric


def clone_model_across_gpus(dev0_scores,
                            dev0_loss,
                            dev0_states,
                            dev0_variables,
                            dev0_metrics):
  scores = [dev0_scores]              # Per GPU predictions
  losses = [dev0_loss]                # Per GPU losses
  states = [dev0_states]              # Per GPU output states
  learn_variables = [dev0_variables]  # Per GPU learnable parameters
  metrics = [dev0_metrics]            # Per GPU model accuracy
  variables = []                      # Per GPU ALL variables
  optimizers = []                     # Per GPU optimizers
  grads = []                          # Per GPU gradients
  steps = []                          # Per GPU training step tensors

  # Clone the model across GPUs
  for gpu in range(1, config.GPUS):
    with tensorflow.name_scope('GPU_%d' % gpu), \
         tensorflow.device('/gpu:%d' % gpu):
      dev_scores, dev_loss, dev_states, dev_variables, dev_metrics = build_model(gpu)
      scores.append(dev_scores)
      losses.append(dev_loss)
      states.append(dev_states)
      learn_variables.append(dev_variables)
      metrics.append(dev_metrics)

  # Create each copy's optimizer and a record for gradients
  for gpu in range(config.GPUS):
    with tensorflow.device('/gpu:%d' % gpu):
      optimizers.append(tensorflow.train.AdamOptimizer())
      dev_grads = optimizers[-1].compute_gradients(
        losses[gpu],
        var_list=learn_variables[gpu],
        gate_gradients=tensorflow.train.Optimizer.GATE_NONE
      )

      # compute_gradients returns a list of [gradient, variable] pairs; split it up
      grads.append([dev_grads[grad][0] for grad in range(len(dev_grads))])
      variables.append([dev_grads[grad][1] for grad in range(len(dev_grads))])

  # Compute summed gradient across devices with nccl
  with tensorflow.name_scope('SumAcrossGPUs'), tensorflow.device(None):
    shared_gradient = []
    for gradient in zip(*grads):
      shared_gradient.append(nccl_ops.all_sum(gradient))

  # Apply the gradient to each GPU's model
  # Scale gradients from sum to mean across GPUs, then clip.
  for gpu in range(config.GPUS):
    with tensorflow.device('/gpu:%d' % gpu):
      clipped = [tensorflow.clip_by_norm(grad[gpu] / config.GPUS, config.GRADIENT_CLIP)
                 for grad in shared_gradient]
      steps.append(
        optimizers[gpu].apply_gradients(zip(clipped, variables[gpu]))
      )

  return scores, losses, states, steps, metrics

def split_data_for_gpus(batch_data):
  batch_inputs = [batch_data[0][gpu * config.BATCH_SIZE:(gpu + 1) * config.BATCH_SIZE]
                  for gpu in range(config.GPUS)]
  batch_targets = [batch_data[1][gpu * config.BATCH_SIZE:(gpu + 1) * config.BATCH_SIZE]
                   for gpu in range(config.GPUS)]
  return batch_inputs, batch_targets

def get_fresh_states():
  # Return a list of each layer's initial (zeroed) state on each GPU
  return [[numpy.zeros((config.BATCH_SIZE,
                        state_size[0],
                        state_size[0],
                        state_size[1]))
           for state_size in config.STATES]
          for gpu in range(config.GPUS)]

with tensorflow.Session() as sess:
  # Misc matplotlib setup
  plt.ion()
  plt.figure()
  plt.show()

  ####################
  # Initialize model #
  ####################

  scores, losses, states, variables, metric = build_model(0)
  scores, losses, states, train_steps, metrics = clone_model_across_gpus(
    scores, losses, states, variables, metric
  )
  train_losses = []
  val_losses = []

  saver = tensorflow.train.Saver()

  try:
    saver.restore(sess, "./save/model.ckpt")
  except Exception as e:
    print("Could not load model: ", str(e))
    print("Starting from scratch...")
    sess.run(tensorflow.global_variables_initializer())

  ######################
  # Set up dataloaders #
  ######################

  train_loader = dataloader.MNISTLoader(
    data_dir=config.TRAIN_DIR,
    batch_size=config.BATCH_SIZE * config.GPUS,  # Get a batch for each GPU
    seq_len=config.SEQ_LEN,
    echo_lag=config.ECHO_LAG,
    shuffle=True
  )
  train_queuer = keras.utils.OrderedEnqueuer(train_loader, use_multiprocessing=True)
  train_queuer.start(workers=15, max_queue_size=15)
  train_generator = train_queuer.get()

  val_loader = dataloader.MNISTLoader(
    data_dir=config.VALIDATION_DIR,
    batch_size=config.BATCH_SIZE * config.GPUS,  # Get a batch for each GPU
    seq_len=config.SEQ_LEN,
    echo_lag=config.ECHO_LAG,
    shuffle=True
  )
  val_queuer = keras.utils.OrderedEnqueuer(val_loader, use_multiprocessing=True)
  val_queuer.start(workers=15, max_queue_size=15)
  val_generator = val_queuer.get()

  #################
  # TRAINING LOOP #
  #################
  for epoch_id in range(config.EPOCHS):
    print("New data, epoch", epoch_id)

    train_loader.on_epoch_end()
    val_loader.on_epoch_end()
    saver.save(sess, "./save/model.ckpt")

    current_state = get_fresh_states()
    feed_dict = {state_placeholders[gpu][state]: current_state[gpu][state]
                 for gpu in range(config.GPUS)
                 for state in range(len(config.STATES))}

    start_time = time.time()
    for batch_id in range(len(train_loader)):
      train_batch = next(train_generator)

      # Split inputs and targets into separate batches for each GPU
      train_inputs, train_targets = split_data_for_gpus(train_batch)

      feed_dict.update({input_placeholders[gpu][input_data]: train_inputs[gpu]
                        for gpu in range(config.GPUS)
                        for input_data in range(len(config.INPUTS))})
      feed_dict.update({target_placeholders[gpu][target]: train_targets[gpu]
                        for gpu in range(config.GPUS)
                        for target in range(len(config.CLASSES))})

      # Execute a single batch's training step
      train_results = sess.run(
        losses + metrics + train_steps + states + scores,
        feed_dict=feed_dict
      )

      # Slice apart results
      train_loss = numpy.mean(train_results[:config.GPUS], keepdims=False)
      train_metric = numpy.mean(train_results[config.GPUS:2 * config.GPUS])
      current_state = train_results[3 * config.GPUS: 4 * config.GPUS]
      train_losses.append(train_loss)

      # Update input states for next batch (which is made up of the next sequence)
      # Can just grab element 0 since this is one state per layer; would require some
      # refactoring for multiple states per layer, e.g. LSTM
      feed_dict.update({state_placeholders[gpu][layer] : current_state[gpu][layer][0]
                        for gpu in range(config.GPUS)
                        for layer in range(len(config.STATES))})


      if batch_id % 25 == 0:
        end_time = time.time()
        print("Step", batch_id,
              "Loss", train_loss,
              "Acc", train_metric,
              "Time", end_time - start_time)
        plot.plot(train_losses,
                  val_losses,
                  train_results[-1],  # Show samples from the final GPU batch
                  train_inputs[-1],
                  config.ECHO_LAG
                 )
        start_time = time.time()

    # Before starting validation, reset input states fed to the model
    current_state = get_fresh_states()
    feed_dict = {state_placeholders[gpu][state]: current_state[gpu][state]
                 for gpu in range(config.GPUS)
                 for state in range(len(config.STATES))}

    # Instead of printing regularly during validation, print the mean at the end;
    # these tables store the intermediate values as I was too lazy to compute the
    # running mean properly
    val_accuracy_storage = []
    val_loss_storage = []

    ###################################
    # VALIDATION LOOP AT END OF EPOCH #
    ###################################
    for val_batch_id in range(len(val_loader)):
      val_batch = next(val_generator)

      # Split inputs and targets into separate batches for each GPU
      val_inputs, val_targets = split_data_for_gpus(val_batch)

      feed_dict.update({input_placeholders[gpu][input_data]: val_inputs[gpu]
                        for gpu in range(config.GPUS)
                        for input_data in range(len(config.INPUTS))})
      feed_dict.update({target_placeholders[gpu][target]: val_targets[gpu]
                        for gpu in range(config.GPUS)
                        for target in range(len(config.CLASSES))})

      # Run a validation computation step. Note no train_steps being computed here!
      train_results = sess.run(
        losses + metrics + states + scores,
        feed_dict=feed_dict
      )

      # Split the results into useful pieces
      val_metric = numpy.mean(train_results[config.GPUS:2 * config.GPUS])
      current_state = train_results[2 * config.GPUS:3 * config.GPUS]
      feed_dict.update({state_placeholders[gpu][state]: current_state[gpu][state][0]
                        for gpu in range(config.GPUS)
                        for state in range(len(config.STATES))})

      val_accuracy_storage.append(val_metric)
      val_loss_storage.append(numpy.mean(train_results[:config.GPUS], keepdims=False))

    # Condense the collected statistics for the validation loop and print them.
    # Store the validation loss for display alongside the training loss.
    val_loss = numpy.mean(val_loss_storage)
    val_acc = numpy.mean(val_accuracy_storage)
    val_losses.append([len(train_losses), val_loss])
    print("*******************")
    print("VALIDATION: Loss", val_loss, "Acc", val_acc)
    print("*******************")

# Save a copy of the final plot for later reference and close the plot display window
plt.savefig('Example.png')
plt.ioff()