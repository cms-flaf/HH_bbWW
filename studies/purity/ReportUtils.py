import seaborn as sns
import matplotlib.pyplot as plt

def MakePlot(masspoint, purity, output_name, title):
    plt.grid()
    plot = sns.lineplot(x=masspoint, y=purity, marker='s', linestyle='')
    plot.set_title(title)
    plot.set(xlabel='m(X), [GeV]', ylabel='Purity')

    fig = plot.get_figure()
    fig.savefig(output_name)
    plt.close()


def SaveToFile(output_name, masspoint, purity):
    with open(output_name, 'w') as file:
        lines = [f"{masspoint[i]} {purity[i]}\n" for i in range(len(masspoint))]
        file.write("".join(lines))
