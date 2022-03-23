import torch
from sim.crossbar.crossbar import crossbar
import sim.modules.Linear as linear
from sim.modules.Random import Random2

import matplotlib
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_statistics(array):
    mean = torch.mean(array)
    diffs = array - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    return mean, std, skews, kurtoses


device_params = {"Vdd": 0.2,
                 "r_wl": 5 + 1e-9,
                 "r_bl": 5 + 1e-9,
                 "m": 8,
                 "n": 8,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "r_in": 1e3,
                 "r_out": 1e3,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1/3,
                 "tile_rows": 8,
                 "tile_cols": 8,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20,
                 "r_on_stddev": 1e3,
                 "r_off_stddev": 1e4,
                 "p_stuck_on": 0.0,
                 "p_stuck_off": 0.0,
                 "method": "viability",
                 "viability": 0.01,
}

class Euler_Maruyama(torch.nn.Module):

    def __init__(self, f, g, cb, t0, t1, N):
        super(Euler_Maruyama, self).__init__()

        self.cb = cb
        self.rand = Random2(1, cb, bypass=True)
        self.N, self.t0, self.t1 = N, t0, t1
        self.t = torch.linspace(self.t0, self.t1, self.N+1)
        self.h = self.t[1] - self.t[0]
        self.f, self.g = f, g
        self.cbon = False

    def forward(self, Z0, W=None):

        if W is not None: Normal = W
        else: Normal = torch.zeros_like(self.t)
            
        t, dt = self.t, self.h
        Z = torch.zeros_like(self.t)
        Z[0] = Z0
        for i in range(self.N):
            if W is None: Normal[i+1] = self.rand()
            f_eval = f(Z[i].reshape(1, -1, 1)).view(Z[i].size())
            g_eval = g(Z[i].reshape(1, -1, 1)).view(Z[i].size())            
            Z[i+1] = Z[i] + f_eval * dt + g_eval * dt**0.5 * Normal[i+1]
        
        return Z, Normal

    def use_cb(self, state):
        self.rand.use_cb(state)
        self.f.use_cb(state)
        self.g.use_cb(state)

fig, ax2 = plt.subplots(1)
#fig = plt.figure(figsize=(8,8))
#gs = gridspec.GridSpec(4, 3)
#ax_main = plt.subplot(gs[0:3, :2])
#ax2 = plt.subplot(gs[0:3, 2:3],sharey=ax_main)

# Black Scholes SDE Parameters
r = 0.1
sigma = 0.2

cb = crossbar(device_params, deterministic=False)
f = linear.Linear(2, 1, cb, W=torch.ones(1, 1)*r, vbits=16)
g = linear.Linear(1, 1, cb, W=torch.ones(1, 1)*sigma, vbits=16)
Z0 = 50

# Solver Parameters
t0 = 0
t1 = 1
N = 100

solver = Euler_Maruyama(f, g, cb, t0, t1, N)
solver.use_cb(True)

# Simulate Trajectories using Crossbar
num_trajectories = 1000
all_trajectories = []
normal_trajectories = []
for i in range(num_trajectories):
    if i % 100 == 0: print(i)
    Z, normal = solver(Z0)
    all_trajectories.append(Z)
    normal_trajectories.append(normal)

# Bin ending values at time t1
end_val = torch.cat([t[-1].view(1) for t in all_trajectories]).detach().numpy().flatten()
vals, bins = np.histogram(end_val, bins=40, density=True)
ax2.plot((bins[1:] + bins[:-1])/2, vals, label='Crossbar RNG')

solver.use_cb(False)
all_ideal_trajectories = []
ideal_normal_trajectories = []
for trajectory in normal_trajectories:
    Z, normal = solver(Z0)#, trajectory)
    all_ideal_trajectories.append(Z)
    ideal_normal_trajectories.append(normal)

end_val_ideal = torch.cat([t[-1].view(1) for t in all_ideal_trajectories]).numpy().flatten()
vals, bins = np.histogram(end_val_ideal, bins=30, density=True)
ax2.plot((bins[1:] + bins[:-1])/2, vals, label='Ideal Monte Carlo')

# Analytic Solution for t1
wiener = lambda x, t: np.exp(-x**2 / (2*t)) / (2 * np.pi * t)**0.5
blackscholes = lambda W, t: Z0 * np.exp(sigma * W + (r - 0.5 * sigma**2) * t)
W = np.linspace(-4, 4, 1000)
pW = wiener(W, t1)
B = blackscholes(W, t1)
pB = (pW[1:] + pW[:-1])/2 * (W[1:] - W[:-1]) / (B[1:] - B[:-1])
ax2.plot((B[1:] + B[:-1])/2, pB, label='Analytic Solution', color='k')

#for i in range(min(num_trajectories, 10)):
#    ax_main.plot(np.linspace(0, 100, 101), all_trajectories[i], color=u'#1f77b4')

# Plot
ax2.legend()
plt.show()



"""
gauss = lambda x, mu, sig: np.exp(-(x-mu)**2 / (2 * sig**2)) / (2 * np.pi * sig**2)**0.5
gauss_w = lambda x, mu: gauss(x, mu, 3.0)
x = np.linspace(0, 120, 1000)
density = np.zeros_like(x)
for mu in end_val:
    density += gauss_w(x, mu)
density /= len(end_val)
"""
