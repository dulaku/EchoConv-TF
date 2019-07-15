# Overview

This model accepts sequences of shuffled MNIST images and emits the digit of each image 3 steps after it was observed. The model is quite bad at its job, reaching about 74% validation set accuracy. However, being a good model isn't the point.

This model exists as a template for Tensorflow models. I put this together to learn some essential skills for developing models in Tensorflow; the model includes the basic features I'd like to have conveniently to hand when developing real projects.

* Recurrence
* Convolution
* Model Saving/Loading
* Training Visualization
* Multithreaded Data Loading
* Multi-GPU Training

The model is meant to be extended, so some parts of it are more complicated than strictly necessary. For example, the model creates an array for storing input placeholder tensors, even though the model takes only a single input. At the same time, I've tried to do this in the most readable way possible, both for my own future self's sake and that of anyone interested.

The model's efficiency could almost certainly be improved; its readability definitely could be. Feedback on these points is extremely welcome.

# Acknowledgements

This code is descended from the work of many people smarter than me. I highly recommend checking their work out.

* [Tensorflow Recurrent Neural Network tutorial](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767) by Erik Hallström
	** Some of the API calls in this are out of date but the groundwork is solid
* [Multi-GPU Training in ProGAN](https://github.com/tkarras/progressive_growing_of_gans/blob/master/tfutil.py) by NVIDIA/Tero Karras
	** Most of the multi-GPU logic is under the Optimizer class
* [Keras Dataloaders](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) by Afshine Amidi and Shervine Amidi

# Requirements

This model expects the following Python packages:

* `tensorflow`, `tensorflow-gpu`
* `keras`
* `matplotlib`
* `numpy`
* `scipy`

I'm not sure how you'd get `tensorflow` and `keras` without `numpy` etc., but listing them anyway since they're directly used.

In addition, this model uses the NVIDIA Collective Communications Library ([NCCL](https://developer.nvidia.com/nccl)) functions from Tensorflow. Actually getting your machine in a position to use that is an adventure left to the reader - in my case it required compiling Tensorflow from source with some very specific configuration choices.

Finally, the model expects a copy of the MNIST dataset in PNG form in the current working directory, under `mnist_png`. At time of writing you can get a copy [here](https://github.com/myleott/mnist_png).
