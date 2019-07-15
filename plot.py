from scipy.special import softmax
import matplotlib.pyplot as plt
import numpy

def plot(train_losses, val_losses, scores, inputs, echo_lag):
  plt.subplot(2, 3, 1)    # 2 x 3 grid of plots; we'll plot to the first
  plt.cla()               # Clear the plot
  plt.ylim(0, 4)          # Set a max Y to avoid graph squashing from occasional spikes
  plt.plot(train_losses)  # Plot a blue line graph for training losses

  # Plot red dots for validation losses
  plt.plot([point[0] for point in val_losses],
           [point[1] for point in val_losses],
           'ro')

  for selection in range(5):
    # Get a prediction that's sure to be in the same sequence as its source image
    individual_scores = scores[selection][echo_lag + 1]
    # Convert scores to highest-probability class
    probs = softmax(individual_scores, axis=-1)
    prediction = numpy.argmax(probs)
    confidence = probs[prediction]

    plt.subplot(2, 3, selection + 2)
    plt.cla()
    plt.imshow(inputs[selection][1].squeeze(), cmap='gray')
    plt.title(str(prediction) + ": %2d" % int(100*confidence) + "%")

  plt.draw()
  plt.pause(0.0001)
