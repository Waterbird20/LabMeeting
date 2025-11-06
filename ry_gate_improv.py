import numpy as np
from numpy import linspace, cos, sin, ones, outer, pi, size
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.cm as cm

import torch

sphere_color="#FFDDDD"
sphere_alpha = 0.1
frame_color = 'gray'
frame_alpha = 0.1
frame_width = 1

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def show_label(ax, **opts):
    ax.scatter(-1,0,0, color="black", s=20)
    ax.scatter(1,0,0, color="black", s=20)
    ax.scatter(0,-1,0, color="black", s=20)
    ax.scatter(0,1,0, color="black", s=20)
    ax.scatter(0,0,-1, color="black", s=20)
    ax.scatter(0,0,1, color="black", s=20)

    ax.text(-1.2, 0, 0, "$-x$", fontsize=12, **opts)
    ax.text( 1.2, 0, 0,  "$x$", fontsize=12, **opts)

    ax.text(0, -1.2, 0, "$-y$", fontsize=12, **opts)
    ax.text(0,  1.2, 0,  "$y$", fontsize=12, **opts)

    ax.text(0, 0, -1.2, "$-z$", fontsize=12, **opts)
    ax.text(0, 0,  1.2,  "$z$", fontsize=12, **opts)

def plot_back(ax):
    # back half of sphere
    u = linspace(0, pi, 25)
    v = linspace(0, pi, 25)
    x = outer(cos(u), sin(v))
    y = outer(sin(u), sin(v))
    z = outer(ones(size(u)), cos(v))
    ax.plot_surface(x, y, z, rstride=2, cstride=2,
                            color=sphere_color, linewidth=0,
                            alpha=sphere_alpha)
    # wireframe
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                color=frame_color,
                                alpha=frame_alpha)
    # equator
    ax.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='z',
                    lw=frame_width, color=frame_color)
    ax.plot(1.0 * cos(u), 1.0 * sin(u), zs=0, zdir='x',
                    lw=frame_width, color=frame_color)

def plot_front(ax):
    # front half of sphere
    u = linspace(-pi, 0, 25)
    v = linspace(0, pi, 25)
    x = outer(cos(u), sin(v))
    y = outer(sin(u), sin(v))
    z = outer(ones(size(u)), cos(v))
    ax.plot_surface(x, y, z, rstride=2, cstride=2,
                            color=sphere_color, linewidth=0,
                            alpha=sphere_alpha)
    # wireframe
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                color=frame_color,
                                alpha=frame_alpha)
    # equator
    ax.plot(1.0 * cos(u), 1.0 * sin(u),
                    zs=0, zdir='z', lw=frame_width,
                    color=frame_color)
    ax.plot(1.0 * cos(u), 1.0 * sin(u),
                    zs=0, zdir='x', lw=frame_width,
                    color=frame_color)
    
X = torch.tensor([
    [0,1],
    [1,0]
], dtype = torch.complex128)

Y = torch.tensor([
    [0,-1j],
    [1j,0]
], dtype = torch.complex128)

Z = torch.tensor([
    [1,0],
    [0,-1]
], dtype = torch.complex128)

def U(t):
    return torch.linalg.matrix_exp(-1j*t*X)

initial_state = torch.zeros(2, dtype=torch.complex128)
initial_state[0] = 1
initial_state.reshape(2,1)

dt = 1e-3
tlist = torch.arange(0, 2*np.pi, dt)
expval = {
    "X" : torch.zeros(len(tlist), dtype=torch.float64),
    "Y" : torch.zeros(len(tlist), dtype=torch.float64),
    "Z" : torch.zeros(len(tlist), dtype=torch.float64)
}

expval["X"][0] = torch.einsum("i,ij,j", initial_state.conj_physical(), X, initial_state).real
expval["Y"][0] = torch.einsum("i,ij,j", initial_state.conj_physical(), Y, initial_state).real
expval["Z"][0] = torch.einsum("i,ij,j", initial_state.conj_physical(), Z, initial_state).real

for i, t in enumerate(tlist[1:]):
    state = U(t) @ initial_state
    expval["X"][i+1] = torch.einsum("i,ij,j", state.conj_physical(), X, state).real
    expval["Y"][i+1] = torch.einsum("i,ij,j", state.conj_physical(), Y, state).real
    expval["Z"][i+1] = torch.einsum("i,ij,j", state.conj_physical(), Z, state).real

# Reduce number of arrows for clarity (8 instead of 12)
plot_list = torch.arange(0, 8, 1) * torch.pi / dt / 8

# Use color gradient to show time evolution
colors = cm.coolwarm(np.linspace(0.2, 0.8, len(plot_list)))

# Set up figure
fig = plt.figure(figsize=(6, 6), facecolor="#ffffff")
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_facecolor("#ffffff")
ax.view_init(azim=-60, elev=30)

# Plot sphere
plot_back(ax)
plot_front(ax)

# Draw rotation axis (Y-axis) more prominently
ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'green', linewidth=2.5, alpha=0.6, label='Rotation Axis (X)', zorder=10)

# Add trajectory path
trajectory_indices = torch.arange(0, len(tlist), 20).int()
ax.plot(expval["X"][trajectory_indices], 
        expval["Y"][trajectory_indices], 
        expval["Z"][trajectory_indices], 
        'k--', linewidth=1.5, label='Trajectory', zorder=5, alpha=0.8)

# Plot initial state (special styling)
idx_initial = 0
a_initial = Arrow3D(xs=[0, expval["X"][idx_initial]], 
                    ys=[0, expval["Y"][idx_initial]], 
                    zs=[0, expval["Z"][idx_initial]], 
                    mutation_scale=22, lw=4, arrowstyle="-|>", 
                    color='darkblue', alpha=1.0, zorder=20)
ax.add_artist(a_initial)

# Plot intermediate states
for i in range(1, len(plot_list)-1):
    idx = int(plot_list[i])
    # Make intermediate arrows slightly shorter and thinner
    scale = 0.95
    a = Arrow3D(xs=[0, scale*expval["X"][idx]], 
                ys=[0, scale*expval["Y"][idx]], 
                zs=[0, scale*expval["Z"][idx]], 
                mutation_scale=18, lw=2.5, arrowstyle="-|>", 
                color=colors[i], alpha=0.9, zorder=15)
    ax.add_artist(a)

# Plot final state (special styling)
idx_final = int(plot_list[-1])
a_final = Arrow3D(xs=[0, expval["X"][idx_final]], 
                  ys=[0, expval["Y"][idx_final]], 
                  zs=[0, expval["Z"][idx_final]], 
                  mutation_scale=22, lw=4, arrowstyle="-|>", 
                  color='darkred', alpha=1.0, zorder=20)
ax.add_artist(a_final)

# Add rotation direction indicator (curved arrow in XZ plane)
theta_arc = np.linspace(0, 2*np.pi/3, 20)
arc_x = np.zeros_like(theta_arc)
arc_y = - 0.4 * np.sin(theta_arc)
arc_z = 0.4 * np.cos(theta_arc)
# ax.plot(arc_x, arc_y, arc_z, 'g-', linewidth=2, alpha=0.5, zorder=10)
# Add arrowhead to rotation indicator
# ax.quiver(arc_x[-2], arc_y[-2], arc_z[-2], 
#           arc_x[-1]-arc_x[-2], arc_y[-1]-arc_y[-2], arc_z[-1]-arc_z[-2],
#           color='green', alpha=0.5, arrow_length_ratio=3, zorder=10)

# Labels
show_label(ax)

# Add title and legend
# ax.set_title(r'$R_Y(\theta)$ Gate: Rotation around Y-axis ($\theta = 2\pi$)', fontsize=14, pad=20)
# ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Add text annotations for initial and final states
# ax.text(expval["X"][0]+0.1, expval["Y"][0], expval["Z"][0]+0.1, 
#         r'$|0\rangle$', fontsize=12, color='darkblue', weight='bold')


ax.axis("off")
fig.tight_layout(pad=1)
fig.savefig("/home/hun/LabMeeting/test.png", dpi=600)