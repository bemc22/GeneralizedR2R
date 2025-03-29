import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

# This setup code is inspired by Santiago L. Valdarram's Keras tutorial "Convolutional autoencoder 
# for image denoising" (https://keras.io/examples/vision/autoencoder/)

# Data preparation
def get_dataset():
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=".", train=False, download=True, transform=transform)

    # Data are returned as tensors with shape [N, 1, 28, 28]
    train_imgs = mnist_train.data.unsqueeze(1).float() / 255.0
    test_imgs = mnist_test.data.unsqueeze(1).float() / 255.0

    return train_imgs, test_imgs


def add_gaussian_noise(data, noise_factor):
    noise = torch.randn_like(data) * noise_factor
    return data + noise


def add_poisson_noise(data, noise_factor):
    # Using torch.poisson for noise generation.
    return torch.poisson(data / noise_factor) * noise_factor


def add_gamma_noise(data, noise_factor):
    # Sample from a Gamma distribution with shape parameter 'noise_factor'.
    gamma_dist = torch.distributions.Gamma(noise_factor, 1)
    sample = gamma_dist.sample(data.shape)
    return sample * (data / noise_factor)


def gaussian_corruptor(noisy_input, alpha, noise_level):
    noise = torch.randn_like(noisy_input) * noise_level
    return noisy_input + alpha * noise


def poisson_corruptor(noisy_input, alpha, noise_level):
    # Compute z and use its shape for sampling.
    z = noisy_input / noise_level
    # For binomial sampling, we round z to the nearest integer.
    z_int = torch.round(z)
    binom = torch.distributions.Binomial(total_count=z_int, probs=alpha)
    w = binom.sample().to(noisy_input.dtype)
    return noise_level * (z - w) / (1 - alpha)


def gamma_corruptor(noisy_input, alpha, noise_level):
    # Concentration parameters computed using broadcasting.
    concentration1 = noise_level * alpha
    concentration0 = noise_level * (1 - alpha)
    beta_dist = torch.distributions.Beta(concentration1, concentration0)
    # Sample with the same shape as noisy_input.
    w = beta_dist.sample(noisy_input.shape)
    return noisy_input * (1 - w) / (1 - alpha)


# Distribution selection and random generator initialization
distribution = "gaussian"
alpha = 0.2
torch.manual_seed(13)
init_generator = torch.Generator()
init_generator.manual_seed(13)
shuffle_generator = torch.Generator()
shuffle_generator.manual_seed(42)



if distribution == "gaussian":
    noise_level = 0.1
    add_noise = add_gaussian_noise
    r2r_corruptor = lambda x: gaussian_corruptor(x, alpha, noise_level)
elif distribution == "poisson":
    noise_level = 0.1
    add_noise = add_poisson_noise
    r2r_corruptor = lambda x: poisson_corruptor(x, alpha, noise_level)
elif distribution == "gamma":
    noise_level = 5.0
    add_noise = add_gamma_noise
    r2r_corruptor = lambda x: gamma_corruptor(x, alpha, noise_level)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.deconv1(x)
        x = F.relu(self.conv4(x))
        x = self.deconv2(x)
        x = self.conv5(x)
        return x


def r2r_loss_fn(model, batch):
    _input = batch["images"]
    y1 = r2r_corruptor(_input)
    y2 = (1 / alpha) * (_input - y1 * (1 - alpha))
    logits = model(y1)
    loss = F.mse_loss(logits, y2)
    return loss, logits


def create_train_state(lr):
    model = Autoencoder()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def train_step(model, optimizer, batch):
    optimizer.zero_grad()
    loss, _ = r2r_loss_fn(model, batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, optimizer, noisy_train_imgs, train_imgs, batch_size, epoch, shuffle_generator):
    num_train_imgs = train_imgs.size(0)
    steps_per_epoch = num_train_imgs // batch_size

    # Shuffle the indices for each epoch.
    perms = torch.randperm(num_train_imgs, generator=shuffle_generator)[:steps_per_epoch * batch_size]
    perms = perms.view(steps_per_epoch, batch_size)

    batch_loss = []
    for perm in trange(steps_per_epoch, desc=f"Epoch {epoch}", unit="batch"):
        batch = {
            "images": noisy_train_imgs[perm].float(),
            "labels": train_imgs[perm].float()
        }
        loss = train_step(model, optimizer, batch)
        batch_loss.append(loss)
    average_loss = np.mean(batch_loss)
    print(f"Train epoch: {epoch}, average loss: {average_loss:.4f}")
    return model, optimizer


def display_images(clean_data, noisy_data, predictions=None):
    n = 10
    indices = torch.randint(0, clean_data.size(0), (n,))
    clean_subset = clean_data[indices]
    noisy_subset = noisy_data[indices]
    pred_subset = predictions[indices] if predictions is not None else None

    rows = 3 if predictions is not None else 2
    plt.figure(figsize=(20, 2 * rows))
    titles = ["Clean Images", "Noisy Images", "Predicted Images"] if predictions is not None else ["Clean Images", "Noisy Images"]

    for row, title in enumerate(titles):
        ax = plt.subplot(rows, 1, row + 1)
        ax.set_title(title, fontsize=16)
        ax.axis("off")

    for i in range(n):
        ax = plt.subplot(rows, n, i + 1)
        plt.imshow(clean_subset[i].squeeze().detach().cpu().numpy(), cmap="gray")
        plt.axis("off")
        ax = plt.subplot(rows, n, i + 1 + n)
        plt.imshow(noisy_subset[i].squeeze().detach().cpu().numpy(), cmap="gray")
        plt.axis("off")
        if predictions is not None:
            ax = plt.subplot(rows, n, i + 1 + 2 * n)
            plt.imshow(pred_subset[i].squeeze().detach().cpu().numpy(), cmap="gray")
            plt.axis("off")
    plt.show()


def eval_model(model, test_imgs, noisy_test_imgs, mc_samples=5):
    predictions = 0.0
    for i in range(mc_samples):
        y1 = r2r_corruptor(noisy_test_imgs)
        logits = model(y1)
        predictions += logits / mc_samples
    display_images(test_imgs, noisy_test_imgs, predictions)


# Main script execution
train_imgs, test_imgs = get_dataset()

noisy_train_imgs = add_noise(train_imgs, noise_level)
noisy_test_imgs = add_noise(test_imgs, noise_level)

display_images(train_imgs, noisy_train_imgs)

lr = 0.001
batch_size = 128
num_epochs = 30

model, optimizer = create_train_state(lr)
for epoch in range(1, num_epochs + 1):
    model, optimizer = train_epoch(model, optimizer, noisy_train_imgs, train_imgs, batch_size, epoch, shuffle_generator)

eval_model(model, test_imgs, noisy_test_imgs)
