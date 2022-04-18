import torch
import numpy as np

def reliability_diagram(bin_count, errors, confidences, ax=None):
    assert len(errors) == len(confidences)
    
    bins = [[] for _ in range(bin_count)]
    for i, confidence in enumerate(confidences):
        bins[torch.floor(confidence * bin_count).int()].append(i)
    bin_accuracys = np.array([errors[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])
    mid = np.linspace(0, 1, bin_count)
    bin_errors = np.abs(np.array(bin_accuracys) - mid)
    bin_confidences = np.array([confidences[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="tab:grey", linestyle=(0, (1, 5)), linewidth=1)
    interval = np.arange(0, 1, 1 / bin_count)
    ax.bar(interval, bin_accuracys, 1 / bin_count, align="edge", color="b", edgecolor="k")
    ax.bar(interval, bin_errors, 1 / bin_count, bottom=np.minimum(bin_accuracys, mid), align="edge", color="mistyrose", alpha=0.5, edgecolor="r", hatch="/")
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlabel('Confidence', fontsize=14)

    for (x, count) in zip(interval, bins):
        ax.text(x + 0.5 * 1 / bin_count, 0.01, str(len(count)), color="white", fontsize=14, ha="center")

    ece = 0
    for i in range(bin_count):
        ece += len(bins[i]) * np.abs(bin_accuracys[i] - bin_confidences[i])
    ece /= len(confidences)

    mce = np.max(np.abs(bin_accuracys - bin_confidences))

    ace = calculate_ace(bin_count, errors, confidences)

    ident = [0.0, 1.0]
    ax.plot(ident,ident,linestyle='--',color="tab:grey")

    ax.text(0.08, 0.9, f"ECE: {ece:.2f}\nMCE: {mce:.2f}\nACE: {ace:.2f}", 
        transform=ax.transAxes, fontsize=16, verticalalignment="top", 
        bbox={"boxstyle": "square,pad=0.5", "facecolor": "white"})

    return ece, mce

def calculate_ace(bin_count, errors, confidences):
    sorted_conf, sort_indices = torch.sort(confidences)

    conf_chunks = torch.chunk(sorted_conf, bin_count)
    error_chunks = torch.chunk(errors[sort_indices], bin_count)

    bin_accuracys = torch.stack([chunk.mean() for chunk in error_chunks])
    bin_confidences = torch.stack([chunk.mean() for chunk in conf_chunks])

    print(f"ACE average confidences: {bin_confidences}")

    ace = 0
    for i in range(bin_count):
        ace += len(conf_chunks[i]) * (bin_accuracys[i] - bin_confidences[i]).abs()
    ace /= len(confidences)

    return ace