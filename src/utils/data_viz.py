import matplotlib.pyplot as plt
import numpy as np


def plot_regression_results():
    # Data
    labels = ['Precision', 'Recall', 'F1 Score']

    # Data ordering [precision loss, precision win, recall loss, recall win, f1 loss, f1 win]
    metrics_everything = [0.6237, 0.6222, 0.6198, 0.6261, 0.6218, 0.6241]
    metrics_only_shortest_path = [0.6815, 0.5466, 0.2786, 0.8698, 0.3956, 0.6714]
    metrics_shortest_path = [0.6016, 0.6049, 0.6112, 0.5952, 0.6064, 0.6000]
    metrics_pagerank = [0.6282, 0.6292, 0.6306, 0.6269, 0.6294, 0.6280]
    metrics_combined = [0.4684, 0.7609, 0.7102, 0.5336, 0.5645, 0.6273]

    # Separate data into "Losses" and "Wins"
    losses = [
        metrics_everything[0::2],
        metrics_only_shortest_path[0::2],
        metrics_shortest_path[0::2],
        metrics_pagerank[0::2],
        metrics_combined[0::2],
    ]

    wins = [
        metrics_everything[1::2],
        metrics_only_shortest_path[1::2],
        metrics_shortest_path[1::2],
        metrics_pagerank[1::2],
        metrics_combined[1::2],
    ]

    # Color palette (CUD palette for accessibility)
    colors = ['#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#F0E442']

    # Plot parameters
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # width of each bar

    # Create plot for Losses
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - 2 * width, losses[0], width, label='Everything', color=colors[0])
    bar2 = ax.bar(x - width, losses[1], width, label='Only Shortest Path', color=colors[1])
    bar3 = ax.bar(x, losses[2], width, label='Shortest Path', color=colors[2])
    bar4 = ax.bar(x + width, losses[3], width, label='Pagerank', color=colors[3])
    bar5 = ax.bar(x + 2 * width, losses[4], width, label='Combined', color=colors[4])

    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Metrics')
    ax.set_title('Logistic Regression Metrics - Losses')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Create plot for Wins
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - 2 * width, wins[0], width, label='Everything', color=colors[0])
    bar2 = ax.bar(x - width, wins[1], width, label='Only Shortest Path', color=colors[1])
    bar3 = ax.bar(x, wins[2], width, label='Shortest Path', color=colors[2])
    bar4 = ax.bar(x + width, wins[3], width, label='Pagerank', color=colors[3])
    bar5 = ax.bar(x + 2 * width, wins[4], width, label='Combined', color=colors[4])

    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Metrics')
    ax.set_title('Logistic Regression Metrics - Wins')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.show()
    