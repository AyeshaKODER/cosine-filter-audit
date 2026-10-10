'''import matplotlib.pyplot as plt
import numpy as np

# Empirical similarity data extracted directly from your complete logs
rounds = [1, 2, 3]
attacker_sims = [0.9995, 0.9994, 0.9982]
benign_avg_sims = [
    np.mean([-0.9986, -0.9987, -0.9987, -0.9981, -0.9987, -0.9985, -0.9984, -0.9987, -0.9986]),
    np.mean([-0.9984, -0.9980, -0.9977, -0.9980, -0.9968, -0.9980, -0.9990, -0.9985, -0.9980]),
    np.mean([-0.9950, -0.9922, -0.9936, -0.9959, -0.9955, -0.9942, -0.9936, -0.9945, -0.9970])
]

plt.figure(figsize=(8, 5))
plt.plot(rounds, attacker_sims, marker='o', color='crimson', linewidth=2, label='Malicious Node (Unflagged Attacker)')
plt.plot(rounds, benign_avg_sims, marker='s', color='royalblue', linewidth=2, linestyle='--', label='9 Benign Nodes (Inverted & Flagged)')

plt.axhline(y=0.7, color='black', linestyle=':', label='Cosine Threshold (0.7)')
plt.title('Reference-Point Inversion Under Large-Magnitude Attack (Scale = -15.0)', fontsize=11, fontweight='bold')
plt.xlabel('Federated Learning Round', fontsize=10)
plt.ylabel('Cosine Similarity against Round Mean', fontsize=10)
plt.xticks(rounds)
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.3)
plt.legend(loc='center left')

plt.savefig('reference_point_inversion.png', dpi=300, bbox_inches='tight')
print("🎉 Plot successfully generated and saved as 'reference_point_inversion.png'") '''

'''import matplotlib.pyplot as plt
import numpy as np

# Empirical tracking from your full 8-round output
rounds = list(range(1, 9))
attacker_sims = [-0.9936, -0.9534, -0.2756, 0.7046, 0.9615, 0.9739, 0.9773, 0.9811]

# Average similarity values for the clean nodes across the lifecycle
benign_avg_sims = [
    np.mean([0.9963, 0.9955, 0.9959, 0.9959, 0.9964, 0.9959, 0.9958, 0.9965, 0.9948]),
    np.mean([0.9731, 0.9835, 0.9658, 0.9675, 0.9651, 0.9718, 0.9768, 0.9678, 0.9607]),
    np.mean([0.5396, 0.5445, 0.5985, 0.5777, 0.7769, 0.6082, 0.6363, 0.4740, 0.6326]),
    np.mean([0.4086, 0.3166, 0.1526, 0.8649, 0.1741, 0.5000, 0.4393, 0.1690, -0.1126]),
    np.mean([0.0444, 0.7012, 0.0781, 0.0919, 0.2942, 0.6135, 0.4410, 0.3558, -0.1424]),
    np.mean([0.2112, 0.7707, -0.0343, 0.1222, -0.2154, 0.3464, 0.7809, -0.7194, 0.0128]),
    np.mean([-0.0532, 0.1687, 0.1626, 0.7771, 0.1445, 0.4669, 0.6002, 0.1710, -0.4306]),
    np.mean([-0.0055, 0.1982, -0.1004, 0.8008, 0.4486, 0.5144, 0.2685, -0.2281, 0.2625])
]

plt.figure(figsize=(9, 5.5))
plt.plot(rounds, attacker_sims, marker='o', color='crimson', linewidth=2.5, label='Malicious Node (Partition 0)')
plt.plot(rounds, benign_avg_sims, marker='s', color='royalblue', linewidth=2, linestyle='--', label='Benign Nodes (Average)')

plt.axhline(y=0.7, color='black', linestyle=':', alpha=0.7, label='Cosine Threshold (0.7)')
plt.axvline(x=3.5, color='darkorange', linestyle='--', alpha=0.5, label='Crossover Window')

plt.title('Temporal Erosion & Reference-Point Inversion Lock-In (Scale = -7.0)', fontsize=11, fontweight='bold')
plt.xlabel('Federated Learning Round', fontsize=10)
plt.ylabel('Cosine Similarity against Round Mean', fontsize=10)
plt.xticks(rounds)
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.25)
plt.legend(loc='lower left', fontsize=9)

plt.savefig('temporal_inversion_collapse.png', dpi=300, bbox_inches='tight')
print("🎉 Success! 8-round collapse figure saved as 'temporal_inversion_collapse.png'")

'''

"""
Temporal erosion figure, built directly from the real detection logs
pasted from 3 actual flwr runs (2 mean-reference seeds implicitly via
original + new, 2 previous-global-model seeds: original + new).
No synthetic/narrated data -- every point below is read off the actual
terminal output.
"""

'''
import matplotlib.pyplot as plt
import numpy as np

rounds = list(range(1, 9))

# Attacker's own cosine similarity each round, mean-reference mode
# Seed A = original run, Seed B = new run (this session)
attacker_mean_seedA = [-0.9947, -0.9955, -0.8733, 0.2448, 0.8675, 0.8231, 0.7959, 0.8556]
attacker_mean_seedB = [-0.9930, -0.9482, -0.3739, 0.6789, 0.9367, 0.9694, 0.9854, 0.9492]

# Attacker's own cosine similarity each round, previous-global-model mode
attacker_prev_seedA = [-0.9936, -0.9955, -0.8734, 0.2306, 0.8259, 0.8637, 0.9152, 0.9438]
attacker_prev_seedB = [-0.9936, -0.9961, -0.8827, 0.2306, 0.8259, 0.8637, 0.9152, 0.9438]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

for ax, (title, seedA, seedB) in zip(
    axes,
    [
        ("Mean-of-round reference ($r_t^{mean}$)", attacker_mean_seedA, attacker_mean_seedB),
        ("Previous-global-model reference ($r_t^{prev}$)", attacker_prev_seedA, attacker_prev_seedB),
    ],
):
    ax.plot(rounds, seedA, marker='o', color='crimson', label='Seed 1')
    ax.plot(rounds, seedB, marker='s', color='darkorange', linestyle='--', label='Seed 2')
    ax.axhline(y=0.7, color='black', linestyle=':', linewidth=1, label='Detection threshold ($\\tau=0.7$)')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Federated learning round')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Attacker cosine similarity to reference')
axes[0].legend(loc='lower right', fontsize=8)

fig.suptitle(r'Temporal erosion at fixed attack scale $\gamma=-7.0$: attacker similarity climbs above threshold and locks in, across both reference-point designs',
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig('temporal_erosion_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('temporal_erosion_figure.png', dpi=300, bbox_inches='tight')
print("Saved temporal_erosion_figure.pdf and .png")
'''

"""
Architecture/pipeline diagram: the detection-enforcement flow within
one federated round, matching the description in Section 3.2 and
Algorithm 1 of the paper.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set up clean canvas
fig, ax = plt.subplots(figsize=(10, 7.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis('off')

# Helper function to create clean text blocks
def box(x, y, w, h, text, color='#f4f4f6', edge='#4a4a4a', fontsize=9, style='round'):
    b = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"{style},pad=0.05,rounding_size=0.08",
                        linewidth=1.2, edgecolor=edge, facecolor=color)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return (x + w/2, y), (x + w/2, y + h)  # returns (bottom_center, top_center)

# Helper function for crisp arrows
def arrow(p1, p2, color='#2c3e50', style='-|>', lw=1.2, ls='-'):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                         linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(a)

# --- SECTION LABELS (Left Margin) ---
ax.text(0.2, 11.6, "SERVER\n(Initialize)", fontsize=9, fontweight='bold', color='#7f8c8d', ha='left', va='center')
ax.text(0.2, 9.6, "CLIENTS\n(Local)", fontsize=9, fontweight='bold', color='#7f8c8d', ha='left', va='center')
ax.text(0.2, 5.0, "SERVER\n(Defense & info)", fontsize=9, fontweight='bold', color='#7f8c8d', ha='left', va='center')

# --- 1. Server: configure_train ---
server_b, server_t = box(2.5, 11.0, 5.0, 1.2,
    r"Server: configure_train" + "\n" +
    r"Broadcasts attack config to evaluate baseline" + "\n" +
    r"(round $t$, magnitude $\gamma$, malicious IDs)",
    color='#fff9e6', edge='#d4ac0d', fontsize=9)

# --- 2. Clients Row ---
cx = [1.5, 4.2, 6.9]
labels = [
    r"Client 0" + "\n" + r"(Malicious, if triggered)" + "\n" + r"Local training" + "\n" + r"$\Delta w_t^0 \times \gamma$", 
    r"Client $k$" + "\n" + r"(Benign)" + "\n" + r"Standard local training" + "\n" + r"$\Delta w_t^k$", 
    r"Client $K$" + "\n" + r"(Benign)" + "\n" + r"Standard local training" + "\n" + r"$\Delta w_t^K$"
]
colors = ['#fdadad', '#d5f5e3', '#d5f5e3']
edges = ['#c0392b', '#27ae60', '#27ae60']

client_tops = []
client_bottoms = []
for x, lab, c, e in zip(cx, labels, colors, edges):
    b_, t_ = box(x, 8.8, 2.2, 1.6, lab, color=c, edge=e, fontsize=8.5)
    client_tops.append(t_)
    client_bottoms.append(b_)

# Arrows from Server to Clients
arrow((server_b[0] - 1.8, server_b[1]), (client_tops[0][0], client_tops[0][1]))
arrow((server_b[0], server_b[1]), (client_tops[1][0], client_tops[1][1]))
arrow((server_b[0] + 1.8, server_b[1]), (client_tops[2][0], client_tops[2][1]))

# --- 3. Server: aggregate_train -- Detection ---
det_b, det_t = box(2.0, 6.4, 6.0, 1.6,
    r"Server: aggregate_train (Defense Audit)" + "\n" +
    r"1. Compute reference point $r_t$ (mean or previous global model)" + "\n" +
    r"2. Calculate Cosine Similarity: $\mathrm{sim}(w_t^k, r_t)$" + "\n" +
    r"3. Flag anomalies if $\mathrm{sim} < \tau$",
    color='#eaecee', edge='#2c3e50', fontsize=9)

# Clean straight arrows from clients down to detection block
for b_ in client_bottoms:
    arrow((b_[0], b_[1]), (b_[0], det_t[1]))

# --- 4. Enforcement Switch (The Vulnerability Gap) ---
enf_b, enf_t = box(2.0, 4.2, 6.0, 1.4,
    "Enforcement Switch (Audit Finding)\n"
    "enforce=True: Exclude flagged clients (Intended Defense)\n"
    "enforce=False: Include all updates (Audited Bug Pattern)",
    color='#fce4d6', edge='#d35400', fontsize=9)
arrow((det_b[0], det_b[1]), (enf_t[0], enf_t[1]))

# --- 5. Aggregation ---
agg_b, agg_t = box(2.5, 2.2, 5.0, 1.2,
    "Weighted FedAvg Aggregation\n"
    "Aggregates updates over surviving client set\n"
    "to compute global step",
    color='#d5f5e3', edge='#27ae60', fontsize=9)
arrow((enf_b[0], enf_b[1]), (agg_t[0], agg_t[1]))

# --- 6. Global Model Update ---
glob_b, glob_t = box(3.2, 0.4, 3.6, 1.1,
    r"Updated Global Model" + "\n" + r"$w_{t+1}$", 
    color='#f2f4f4', edge='#7f8c8d', fontsize=9)
arrow((agg_b[0], agg_b[1]), (glob_t[0], glob_t[1]))

# --- 7. The Temporal Erosion Loop ---
path_x = 9.8
# From global model block rightward
ax.plot([glob_b[0] + 1.8, path_x], [glob_b[1] + 0.55, glob_b[1] + 0.55], color='#7f8c8d', linestyle='--', linewidth=1.2)
# Up the right margin
ax.plot([path_x, path_x], [glob_b[1] + 0.55, server_t[1] - 0.6], color='#7f8c8d', linestyle='--', linewidth=1.2)
# Back into the server block
arrow((path_x, server_t[1] - 0.6), (server_b[0] + 2.5, server_t[1] - 0.6), color='#7f8c8d', style='-|>', lw=1.2, ls='--')

# Intercept text for the loop
ax.text(path_x - 0.1, 5.5, "Temporal Erosion Loop\n(Model state drifts over\n" + r"sustained rounds $t \rightarrow t+1$)", 
        fontsize=8.5, color='#7f8c8d', ha='right', va='center', style='italic')

plt.tight_layout()
plt.savefig('architecture_diagram_fixed.png', dpi=300, bbox_inches='tight')
print("Successfully generated clean layout with mathematical formulas as 'architecture_diagram_fixed.png'")
