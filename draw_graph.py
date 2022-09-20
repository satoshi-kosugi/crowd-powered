import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(labelsize=15)
plt.xticks(np.linspace(0,30,7))
plt.yticks(np.linspace(0.475,0.650,8))
plt.grid(linestyle='dotted', linewidth=1)

BIQME_scores = {
    # legend name: [list of BIQME score files]
}

initial_scores = np.load("BIQME/originalBIQME_scores.npy")

for legend_name, npy_names in BIQME_scores.items():
    score = 0
    for npy_name in npy_names:
        score = score + np.load(npy_name).mean(axis=-1).flatten()[:30]
    score /= len(npy_names)
    score = np.concatenate([initial_scores.mean(keepdims=True), score], axis=0)
    plt.plot(np.arange(0, score.shape[0]), score, "", label=legend_name)

plt.legend(fontsize=15)
plt.savefig("BIQME.eps")
print("The graph is saved as BIQME.eps.")
