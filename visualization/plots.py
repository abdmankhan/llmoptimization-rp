import matplotlib.pyplot as plt

def plot_pareto(solutions, pareto, baselines, path):
    plt.figure()

    # All solutions
    plt.scatter(
        [s["cost"] for s in solutions],
        [s["accuracy"] for s in solutions],
        alpha=0.25,
        label="All solutions"
    )

    # Pareto front
    plt.scatter(
        [p["cost"] for p in pareto],
        [p["accuracy"] for p in pareto],
        label="Optimized Pareto",
        marker="x",
        s=80
    )

    # Baseline points
    for b in baselines:
        plt.scatter(
            b["cost"],
            b["accuracy"],
            marker="o",
            s=120
        )
        plt.text(
            b["cost"],
            b["accuracy"],
            b["prompt"],
            fontsize=9,
            ha="right"
        )

    plt.xlabel("Total Cost")
    plt.ylabel("Expected Success")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
