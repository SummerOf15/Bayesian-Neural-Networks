import os
from comet_ml import Experiment
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.datasets import mnist
from load_dataset import MNISTSequence
import matplotlib
matplotlib.use('Agg') # backend model
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
tfd=tfp.distributions
try:
    import seaborn as sns  # pylint: disable=g-import-not-at-top
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

experiment = Experiment()
IMAGE_SHAPE=[28,28,1]
NUM_TRAIN_EXAMPLES=60000
NUM_HELDOUT_EXAMPLES=10000
NUM_CLASSES=10

learning_rate=0.001
num_epochs=10
batch_size=128
data_dir="data/"
save_model_dir="model/"
num_monte_carlo=30

def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    """Save a PNG plot with histograms of weight means and stddevs.

    Args:
    names: A Python `iterable` of `str` variable names.
      qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
    """
    fig = figure.Figure(figsize=(6, 3))
    canvas = backend_agg.FigureCanvasAgg(fig)

    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
    ax.set_title('weight means')
    ax.set_xlim([-1.5, 1.5])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
    ax.set_title('weight stddevs')
    ax.set_xlim([0, 1.])

    fig.tight_layout()
    canvas.print_figure(fname, format='png')
    print('saved {}'.format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=''):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.

    Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
    """
    fig = figure.Figure(figsize=(9, 3*n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3*i + 1)
        ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]), interpolation='None')

        ax = fig.add_subplot(n, 3, 3*i + 2)
        for prob_sample in probs:
            sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
            ax.set_ylim([0, 1])
        ax.set_title('posterior samples')

        ax = fig.add_subplot(n, 3, 3*i + 3)
        sns.barplot(np.arange(10), tf.reduce_mean(probs[:, i, :], axis=0), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title('predictive probs')
        fig.suptitle(title)
        fig.tight_layout()

    canvas.print_figure(fname, format='png')
    print('saved {}'.format(fname))


def create_model():
    """
    Creates a Keras model using LeNet-5 architecture
    Returns:
        model:compiled keras model
    """
    kl_divergence_function=(lambda p,q,ignore: tfd.kl_divergence(q,p)/tf.cast(NUM_TRAIN_EXAMPLES,dtype=tf.float32))
    model=tf.keras.models.Sequential([
        tfp.layers.Convolution2DFlipout(6,kernel_size=5, padding="SAME", kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2],padding="SAME"),
        tfp.layers.Convolution2DFlipout(16,kernel_size=5, padding="SAME", kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=[2,2],padding="SAME"),
        tfp.layers.Convolution2DFlipout(16, kernel_size=5, padding="SAME", kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(84,kernel_divergence_fn=kl_divergence_function,activation=tf.nn.relu),
        tfp.layers.DenseFlipout(NUM_CLASSES, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.softmax)
    ])

    optimizer=tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"],experimental_run_tf_function=False)
    return model


def train():
    if tf.io.gfile.exists(save_model_dir):
        tf.compat.v1.logging.warning(
            'Warning: deleting old log directory at {}'.format(save_model_dir))
        tf.io.gfile.rmtree(save_model_dir)
    tf.io.gfile.makedirs(save_model_dir)

    # load data
    print("Loading dataset ... ")
    train_set, test_set = mnist.load_data(path="F://program//pyBNNnumber//datasets//mnist.npz")
    train_seq=MNISTSequence(train_set, batch_size=batch_size)
    test_seq=MNISTSequence(test_set,batch_size=batch_size)
    # create model
    model=create_model()
    # model.build(input_shape=[None, 28, 28, 1])

    print(' ... Training convolutional neural network')
    for epoch in range(num_epochs):
        epoch_accuracy=[]
        epoch_loss=[]
        for step, (batch_x, batch_y) in enumerate(train_seq):
            batch_loss, batch_accuracy=model.train_on_batch(batch_x,batch_y)
            epoch_accuracy.append(batch_accuracy)
            epoch_loss.append(batch_loss)

            if step%100==0:
                print('Epoch: {}, Batch index: {}, Loss: {:.3f}, Accuracy: {:.3f}'.format(epoch, step, tf.reduce_mean(epoch_loss),tf.reduce_mean(epoch_accuracy)))
        model.save(save_model_dir+"model.h5")

        # Compute log prob of heldout set by averaging draws from the model:
        # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        # where model_i is a draw from the posterior p(model|train).
        print(' ... Running monte carlo inference')
        probs = tf.stack([model.predict(test_seq.images, verbose=1) for _ in range(num_monte_carlo)], axis=0)
        mean_probs = tf.reduce_mean(probs, axis=0)
        heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

        print(HAS_SEABORN)
        if HAS_SEABORN:
            print("visualization ... ")
            names = [layer.name for layer in model.layers if 'flipout' in layer.name]
            qm_vals = [layer.kernel_posterior.mean() for layer in model.layers if 'flipout' in layer.name]
            qs_vals = [layer.kernel_posterior.stddev() for layer in model.layers if 'flipout' in layer.name]
            plot_weight_posteriors(names, qm_vals, qs_vals,
                                   fname=os.path.join(save_model_dir,'epoch{}_weights.png'.format(epoch)))
            plot_heldout_prediction(test_seq.images, probs,
                                    fname=os.path.join(save_model_dir,'epoch{}_pred.png'.format(epoch)),
                                    title='mean heldout logprob {:.2f}'.format(heldout_log_prob))


if __name__ == "__main__":
    train()