import keras, numpy
import os

class MNISTLoader(keras.utils.Sequence):
  def __init__(self, data_dir, batch_size=8, seq_len=15, echo_lag=3, shuffle=True):
    self.shape = (28, 28, 1)
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.echo_lag = echo_lag
    self.shuffle = shuffle

    # Walk the data directory; each directory contains all images of a digit
    # We map each example to its correct label
    self.targets = {example : digit
                    for digit in os.listdir(self.data_dir)
                    for example in os.listdir(
                                     os.path.join(self.data_dir, digit)
                                   )}

    # Could iterate through self.targets.keys() elsewhere, but this is more readable
    self.samples = [example for example in self.targets.keys()]

    self.on_epoch_end() # Shuffle, initialize leftovers

  def on_epoch_end(self):
    self.indices = numpy.arange(len(self.samples))
    # Keep track of the targets for inputs to close to the end of this batch's
    # sequences - these need to appear at the start of the next batch.
    self.leftover_targets = numpy.zeros((self.batch_size, self.echo_lag),
                                        dtype=numpy.int32)
    if self.shuffle:
      numpy.random.shuffle(self.indices)

  def __data_generation(self, sample_ids):
    image_batch = numpy.zeros((self.batch_size, self.seq_len, *self.shape),
                              dtype=numpy.float32)
    class_batch = numpy.zeros((self.batch_size, self.seq_len),
                              dtype=numpy.int32)

    # Pull in the leftovers from the previous batch's sequences
    for timestep in range(self.echo_lag):
      for batch_id in range(self.batch_size):
        class_batch[batch_id][timestep] = self.leftover_targets[batch_id][timestep]

    # Follow up with the data from this batch
    for timestep in range(self.seq_len):
      for batch_id in range(self.batch_size):
        example = sample_ids[timestep * self.batch_size + batch_id]
        filedir = os.path.join(self.data_dir, self.targets[example])
        filepath = os.path.join(filedir, example)
        imgarr = keras.preprocessing.image.load_img(filepath,
                                                    color_mode='grayscale')

        imgarr = keras.preprocessing.image.img_to_array(imgarr)
        image_batch[batch_id][timestep] = imgarr / 127.5 - 1.0

        target_class = int(self.targets[example])
        if timestep < self.seq_len - self.echo_lag:
          class_batch[batch_id][timestep + self.echo_lag] = target_class
        else:
          leftover_step = timestep + self.echo_lag - self.seq_len
          self.leftover_targets[batch_id][leftover_step] = target_class

    return image_batch, class_batch

  def __len__(self):
    return len(self.samples) // self.batch_size // self.seq_len

  def __getitem__(self, index):
    indices = self.indices[index * self.batch_size * self.seq_len:
                           (index + 1) * self.batch_size * self.seq_len]
    samples = [self.samples[i] for i in indices]
    return self.__data_generation(samples)
