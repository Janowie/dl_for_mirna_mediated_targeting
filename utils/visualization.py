import matplotlib.pyplot as plt


def plot_heatmap(data, labels):
    """
    Plots samples in a row.

    @param data: np.array - encoded samples
    @param labels: list - labels to display
    @return:
    """
    fig, ax = plt.subplots(1, len(data))
    for i, d in enumerate(data):
        ax[i].set_title(f"Label: {labels[i]}")
        ax[i].imshow(d.squeeze(), cmap='Blues')
