"""Plotting helpers. All figures are saved as PNGs; nothing pops up a window."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed series colours, assigned in order (blue, orange, aqua, yellow).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#3d3d3a"
MUTED = "#8a8a84"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cfcfca")
    ax.tick_params(colors=MUTED)
    ax.grid(True, color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)


def _label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def plot_loss_curve(history, path, title="Training curve"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, key in enumerate(["train_loss", "val_loss"]):
        ax.plot(history["step"], history[key], color=COLORS[i], linewidth=2,
                label=key.replace("_", " "))
        _label_end(ax, history["step"][-1], history[key][-1], f"{history[key][-1]:.3f}")
    _style(ax, title, "step", "cross-entropy loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_context_results(rows, path):
    """Val loss vs context length. rows: list of dicts from the experiment summary."""
    xs = [r["block_size"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = [("val_loss_mean", "avg over all positions"),
              ("last_pos_loss_mean", "at the last position (full context)")]
    for i, (key, label) in enumerate(series):
        ys = [r[key] for r in rows]
        errs = [r[key.replace("_mean", "_std")] for r in rows]
        ax.errorbar(xs, ys, yerr=errs if any(errs) else None, color=COLORS[i], linewidth=2,
                    marker="o", markersize=7, capsize=3, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 9 if i == 0 else -15),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    _style(ax, "Validation loss vs. context length", "context length (characters, log scale)",
           "validation loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_position(per_pos_by_block, path):
    """Loss at each position t, one line per model. per_pos_by_block: {block_size: list}."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (bs, losses) in enumerate(sorted(per_pos_by_block.items())):
        xs = list(range(1, len(losses) + 1))
        ax.plot(xs, losses, color=COLORS[i % len(COLORS)], linewidth=2, label=f"context {bs}")
        _label_end(ax, xs[-1], losses[-1], f"{bs}")
    ax.set_xscale("log", base=2)
    ticks = [2 ** k for k in range(0, max(len(v) for v in per_pos_by_block.values()).bit_length())]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    _style(ax, "Loss by how many characters the model could see",
           "characters of context available (log scale)", "validation loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention(maps, text, path, layer=0):
    """Heatmap of each head's attention weights in one layer for one input string.
    maps: model.attention_maps() after a forward pass on a batch of size 1.
    """
    heads = maps[layer]
    labels = [repr(c)[1:-1] or " " for c in text]
    fig, axes = plt.subplots(1, len(heads), figsize=(3.2 * len(heads), 3.6))
    for h, ax in enumerate(axes):
        ax.imshow(heads[h][0].cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"layer {layer} head {h}", fontsize=10, color=INK)
        ax.set_xticks(range(len(text)))
        ax.set_yticks(range(len(text)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("attends to (key)", fontsize=8, color=MUTED)
        if h == 0:
            ax.set_ylabel("predicting from (query)", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
