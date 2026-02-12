import matplotlib.pyplot as plt


def plot_two_tables_env_and_modes(path, w, z_max):
    plt.figure()

    # Tables (env)
    plt.plot([0, 1], [0, 0], linewidth=10, alpha=0.25, label="Left table (env)")
    plt.plot([2, 3], [0, 0], linewidth=10, alpha=0.25, label="Right table (env)")
    plt.plot([1, 2], [0, 0], linestyle=":", linewidth=2, label="Gap (env)")

    # Transition zones (visual guide)
    plt.axvspan(1.0 - w, 1.0, alpha=0.15, label="Left transition zone")
    plt.axvspan(2.0, 2.0 + w, alpha=0.15, label="Right transition zone")

    # Path
    plt.plot(path[:, 0], path[:, 1], marker="o", markersize=2, label="Planned path")

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.xlim(-0.1, 3.1)
    plt.ylim(-0.05, z_max + 0.1)
    plt.title("Multimodal planning with transition zones + free lift height")
    plt.legend(loc="upper right")
    plt.show()