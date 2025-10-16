from matplotlib import pyplot as plt


def plot_map(data, show=True, save_name=None):
    data.to_dataarray().plot()
    if save_name is not None:
        path = "Figures/" + save_name + ".pdf"
        print(path)
        plt.savefig(path)
    if show:
        plt.show()