
import numpy as np
from nistats import hemodynamic_models
import matplotlib.pyplot as plt

rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "serif",
    "text.latex.preview": True,
}
plt.rcParams.update(rc)

frame_times = np.linspace(0, 30, 61)
onset, amplitude, duration = 0.0, 1.0, 1.0
exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)


stim = np.zeros_like(frame_times)
stim[(frame_times > onset) * (frame_times <= onset + duration)] = amplitude

hrf_models = ["glover + derivative + dispersion"]


fig = plt.figure(figsize=(4, 4))
for i, hrf_model in enumerate(hrf_models):
    # obtain the signal of interest by convolution
    signal, name = hemodynamic_models.compute_regressor(
        exp_condition, hrf_model, frame_times, con_id='main',
        oversampling=16)

    # plot this
    # plt.fill(frame_times, stim, 'k', alpha=.5, label='stimulus')
    for j in range(signal.shape[1]):
        if name[j] == "main":
            plt.plot(frame_times, signal.T[j], label=name[j])
    plt.xlabel('time (s)')
    plt.ylabel("BOLD signal")
plt.savefig("./hrf.pdf", bbox_inches="tight")
