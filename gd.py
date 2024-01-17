import os
import time

import torch
from torch import distributions as D

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

from utils import build_kl_hat, mixture_normal, seed_everything

RESULT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
try:
    os.mkdir(RESULT_DIR)
except FileExistsError:
    pass

device = "cpu"
if torch.cuda.is_available():
    print("Using CUDA!")
    device = "cuda"

seed_everything(609)  # hard seed
# seed_everything(166)  # easy seed
nb_density = 9

p = mixture_normal(nb_density, device=device)
kl_hat = build_kl_hat(p, compile=True)

# Sample data from the distributions
target_data = p.sample((nb_density * 100,))

nb_samples = 2**10
init_sample = D.Uniform(
    torch.tensor([0.0, 0.0], device=device), torch.tensor([1.0, 1.0], device=device)
).sample((nb_samples,))

# Plotting the data
plt.figure(figsize=(8, 6))
plt.scatter(
    target_data[:, 0].cpu().numpy(),
    target_data[:, 1].cpu().numpy(),
    alpha=0.6,
    label="Density Samples",
)
plt.scatter(
    init_sample[:, 0].cpu().numpy(),
    init_sample[:, 1].cpu().numpy(),
    alpha=0.6,
    color="red",
    label="Particles",
)
plt.title("Scatter Plot of Sampled Data from 9 Grid Gaussian with Particles")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.savefig(os.path.join(RESULT_DIR, "gd-init.png"))
plt.close()

#####################################
# Gradient descent
#####################################

X = torch.nn.Parameter(init_sample.clone())
optimizer = torch.optim.SGD([X], lr=2e-1, momentum=0, weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[2000, 4000], gamma=0.5
)
recorded_positions = []
recorded_ent = []
nb_epoch = 5000
t0 = time.time()
for i in range(nb_epoch):
    optimizer.zero_grad()

    kl_loss = kl_hat(X)

    if i % 1 == 0:
        recorded_ent.append(kl_loss.item())
        recorded_positions.append(X.cpu().detach().clone())

    kl_loss.backward()
    optimizer.step()
    scheduler.step()

    if i % 100 == 0:
        print(f"{i}/{nb_epoch} - kl_loss: {kl_loss.item()}")
print(f"Time taken by the optimization process: {time.time() - t0}")

# Creating the animation
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

# Scatter plot for the particles
axs[0].scatter(
    target_data[:, 0].cpu().numpy(), target_data[:, 1].cpu().numpy(), alpha=0.4
)
scat = axs[0].scatter(
    init_sample[:, 0].cpu().numpy(),
    init_sample[:, 1].cpu().numpy(),
    alpha=0.6,
    color="red",
)
axs[0].set_title("Optimization of Particle Positions")
axs[0].set_xlabel("X-axis")
axs[0].set_ylabel("Y-axis")
axs[0].grid(True)
axs[0].axis("equal")

# Line plot for the entropy
(line,) = axs[1].plot([], [], lw=2)
axs[1].set_yscale("log")
axs[1].set_xlim(0, len(recorded_ent))
axs[1].set_ylim(10**-1, 10**2)
axs[1].set_title("KL over Iterations")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Entropy")


def update(frame):
    scat.set_offsets(recorded_positions[frame])
    line.set_data(range(frame + 1), recorded_ent[: frame + 1])
    return scat, line


print("Dumping video of the optimization process")


t0 = time.time()
ani = FuncAnimation(fig, update, frames=range(len(recorded_ent)))
f = os.path.join(RESULT_DIR, "gd.mp4")
writervideo = FFMpegWriter(fps=60)
ani.save(f, writer=writervideo)
print(f"Time taken to dump the video: {time.time() - t0}")
