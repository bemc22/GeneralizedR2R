import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
from tqdm import trange


# Data preparation
def get_dataset():
    mnist_train = torchvision.datasets.MNIST(root=".", train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root=".", train=False, download=True)

    train_imgs = jnp.array(mnist_train.data.numpy().astype("float32")) / 255.0
    test_imgs = jnp.array(mnist_test.data.numpy().astype("float32")) / 255.0

    train_imgs = jnp.reshape(train_imgs, (len(train_imgs), 28, 28, 1))
    test_imgs = jnp.reshape(test_imgs, (len(test_imgs), 28, 28, 1))

    return train_imgs, test_imgs


def add_gaussian_noise(data, noise_factor, key):
    noise = random.normal(key, data.shape) * noise_factor
    return data + noise


def add_poisson_noise(data, noise_factor, key):
    return random.poisson(key, data / noise_factor) * noise_factor

def add_gamma_noise(data, noise_factor, key):
    sample = random.gamma(key, noise_factor, data.shape)
    return sample * (data / noise_factor)


@jax.jit
def gaussian_corruptor(noisy_input, alpha, noise_level, key):
    noise = random.normal(key, noisy_input.shape) * noise_level
    return noisy_input + alpha * noise

@jax.jit
def poisson_corruptor(noisy_input, alpha, noise_level, key):
    # Compute z once and use its shape for sampling.
    z = noisy_input / noise_level
    # Directly use z.shape for the binomial sample without additional temporaries.
    w = random.binomial(key, n=z, p=alpha, shape=z.shape)
    return noise_level * (z - w) / (1 - alpha)

@jax.jit
def gamma_corruptor(noisy_input, alpha, noise_level, key):
    # Leverage broadcasting: noise_level and alpha are scalars.
    concentration1 = noise_level * alpha
    concentration0 = noise_level * (1 - alpha)
    # Specify the desired output shape directly.
    w = random.beta(key, concentration1, concentration0, shape=noisy_input.shape)
    return noisy_input * (1 - w) / (1 - alpha)


distribution = "gamma"
alpha = 0.2
rng = random.PRNGKey(13)
init_rng, shuffle_rng, corr_rng = random.split(rng, 3)

if distribution == "gaussian":
    noise_level = 0.1
    add_noise = add_gaussian_noise
    r2r_corruptor = lambda x: gaussian_corruptor(x, alpha, noise_level, corr_rng)
elif distribution == "poisson":
    noise_level = 0.1
    add_noise = add_poisson_noise
    r2r_corruptor = lambda x: poisson_corruptor(x, alpha, noise_level, corr_rng)
elif distribution == "gamma":
    noise_level = 5.0
    add_noise = add_gamma_noise
    r2r_corruptor = lambda x: gamma_corruptor(x, alpha, noise_level, corr_rng)


class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(8, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
        x = nn.ConvTranspose(8, (2, 2), (2, 2), padding="SAME")(x)
        x = nn.relu(nn.Conv(8, (3, 3), padding="SAME")(x))
        x = nn.ConvTranspose(1, (2, 2), (2, 2), padding="SAME")(x)
        return nn.Conv(1, (3, 3), padding="SAME")(x)


def r2r_loss_fn(params, batch):
    _input = batch["images"]
    y1 = r2r_corruptor(_input)
    y2 = (1 / alpha) * (_input - y1 * (1 - alpha))
    logits = Autoencoder().apply({"params": params}, y1)
    loss = jnp.mean(optax.l2_loss(logits, y2))
    return loss, logits


def create_train_state(rng, lr):
    model = Autoencoder()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def create_train_state(rng, lr):

    # The function init() generates model parameters for the model randomly.
    # To achieve this properly, it has to know the shape of the input to the model.
    # At this point, batch_size is regarded as 1 to have a parameter set for exact input data.

    model = Autoencoder()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):

    # One forward and backward pass are performed over a mini-batch of samples given as input
    # Then, by using computed gradients, the parameters of the model are updated by adam gradient descent.
    # Finally, step number and optimizer state are modified.

    grad_fn = jax.value_and_grad(r2r_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, batch)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def train_epoch(state, noisy_train_imgs, train_imgs, batch_size, epoch, shuffle_rng):

    num_train_imgs = len(train_imgs)
    steps_per_epoch = num_train_imgs // batch_size  # number of training steps per epoch

    # Shuffle the indices of training images for each epoch
    perms = jax.random.permutation(shuffle_rng, num_train_imgs)
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Use tqdm to display progress
    batch_loss = []
    with trange(steps_per_epoch, desc=f"Epoch {epoch}", unit="batch") as t:
        for perm in perms:
            batch = {"images": noisy_train_imgs[perm], "labels": train_imgs[perm]}
            state, loss = train_step(state, batch)
            batch_loss.append(loss)
            t.set_postfix(loss=float(loss))  # Update tqdm with the current loss
            t.update()

    average_loss = jnp.mean(jnp.array(batch_loss))
    print("Train epoch: %d, average loss: %.4f" % (epoch, average_loss))

    return state


def display_images(clean_data, noisy_data, predictions=None):
    # Determine the number of rows based on whether predictions are provided
    n = 10
    key = jax.random.PRNGKey(17)
    indices = jax.random.randint(key, (n,), 0, len(clean_data))

    clean_subset = clean_data[indices, :]
    noisy_subset = noisy_data[indices, :]
    pred_subset = predictions[indices, :] if predictions is not None else None

    # Adjust the layout based on the presence of predictions
    rows = 3 if predictions is not None else 2
    plt.figure(figsize=(20, 2 * rows))

    titles = (
        ["Clean Images", "Noisy Images", "Predicted Images"]
        if predictions is not None
        else ["Clean Images", "Noisy Images"]
    )

    for row, title in enumerate(titles):
        ax = plt.subplot(rows, 1, row + 1)
        ax.set_title(title, fontsize=16)
        ax.axis("off")

    for i, (clean_img, noisy_img) in enumerate(zip(clean_subset, noisy_subset)):
        ax = plt.subplot(rows, n, i + 1)
        plt.imshow(clean_img.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(rows, n, i + 1 + n)
        plt.imshow(noisy_img.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if predictions is not None:
            ax = plt.subplot(rows, n, i + 1 + 2 * n)
            plt.imshow(pred_subset[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


def eval_model(params, test_imgs, noisy_test_imgs, mc_samples=5):

    predictions = 0.0

    for i in range(mc_samples):
        y1 = r2r_corruptor(noisy_test_imgs)
        logits = Autoencoder().apply({"params": params}, y1)
        predictions += logits / mc_samples

    display_images(test_imgs, noisy_test_imgs, predictions)


train_imgs, test_imgs = get_dataset()


noisy_train_imgs = add_noise(train_imgs, noise_level, init_rng)
noisy_test_imgs = add_noise(test_imgs, noise_level, init_rng)

display_images(train_imgs, noisy_train_imgs)

lr = 0.001
batch_size = 128
num_epochs = 1

state = create_train_state(init_rng, lr)
for epoch in range(1, num_epochs + 1):
    state = train_epoch(
        state, noisy_train_imgs, train_imgs, batch_size, epoch, shuffle_rng
    )

eval_model(state.params, test_imgs, noisy_test_imgs)
