import os
import time
from collections import deque

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
compile = True
use_triton = False
if torch.cuda.is_available():
    print("Using CUDA!")
    device = "cuda"
    compile = False
    use_triton = True

seed_everything(609)  # hard seed
# seed_everything(166)  # easy seed
nb_density = 9

p = mixture_normal(nb_density, device=device)
kl_hat = build_kl_hat(p, compile=compile, use_triton=use_triton)

# Sample data from the distributions
target_data = p.sample((nb_density * 100,))

nb_samples = 2**10
init_sample = D.Uniform(torch.tensor([0.0, 0.0], device=device), torch.tensor([1.0, 1.0], device=device)).sample(
    (nb_samples,)
)

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
plt.savefig(os.path.join(RESULT_DIR, "gd-adam-stoc-init.png"))
plt.close()

#####################################
# Stochastic Gradient descent
#####################################

step_per_video_frame = 3
X = torch.nn.Parameter(init_sample.clone())
optimizer = torch.optim.Adam([X], lr=2e-1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500 * 2, 1000 * 2], gamma=0.3)
recorded_positions = []
recorded_ent = []
nb_iter = 1500 * 2
t0 = time.time()

bs = 32
nb_epoch = 1 + (nb_iter - 1) // bs
t = 0
kl_loss_sma_k = 30
kl_loss_sma_data = deque([])
kl_loss_sma = 0.0
for i in range(nb_epoch):
    stoch_idx = torch.randperm(nb_samples, device=device)

    for idx_start in range(0, nb_samples, bs):
        cur_bs = min(nb_samples - idx_start, bs)
        idxs = idx_start + torch.arange(cur_bs)
        X_stoc = X[stoch_idx[idxs]]

        optimizer.zero_grad()

        kl_loss = kl_hat(X_stoc)
        kl_loss_val = kl_loss.item()
        if t % step_per_video_frame == 0:
            recorded_ent.append(kl_loss_val)
            recorded_positions.append(X.cpu().detach().clone())

        kl_loss.backward()
        optimizer.step()
        scheduler.step()

        kl_loss_sma_data.append(kl_loss_val)
        if len(kl_loss_sma_data) <= kl_loss_sma_k:
            kl_loss_sma = torch.mean(torch.tensor(kl_loss_sma_data))
        else:
            el_removed = kl_loss_sma_data.popleft()
            kl_loss_sma += (kl_loss_val - el_removed) / kl_loss_sma_k
        t += 1

    print(f"{i}/{nb_epoch} - kl_loss_sma: {kl_loss_sma}")
print(f"Time taken by the optimization process: {time.time() - t0}")

# Creating the animation
fig, axs = plt.subplots(1, 2, figsize=(20 / 2, 10 / 2), dpi=48)  # 960x480 for 2 subplots

# Scatter plot for the particles
axs[0].scatter(target_data[:, 0].cpu().numpy(), target_data[:, 1].cpu().numpy(), alpha=0.4)
scat = axs[0].scatter(
    init_sample[:, 0].cpu().numpy(),
    init_sample[:, 1].cpu().numpy(),
    alpha=0.6,
    color="red",
)

# Line plot for the entropy
(line,) = axs[1].plot([], [], lw=2)
axs[1].set_yscale("log")
axs[1].set_xlim(0, len(recorded_ent) * step_per_video_frame)
axs[1].set_ylim(10**-1, 10**2)
axs[1].set_title("KL SMA over Iterations")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Entropy")


def update(frame):
    scat.set_offsets(recorded_positions[frame])
    line.set_data(
        range(0, (frame + 1) * step_per_video_frame, step_per_video_frame),
        recorded_ent[: frame + 1],
    )
    return scat, line


print("Dumping video of the optimization process")


t0 = time.time()
ani = FuncAnimation(fig, update, frames=range(len(recorded_ent)), repeat=False, blit=True)
f = os.path.join(RESULT_DIR, "gd-adam-stoc.mp4")
writervideo = FFMpegWriter(fps=30)
ani.save(f, writer=writervideo)
print(f"Time taken to dump the video: {time.time() - t0}")
