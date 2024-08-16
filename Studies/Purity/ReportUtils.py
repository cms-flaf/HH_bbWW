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


# data: list of pairs (masspoint, purity) for all taggers in fixed channel and resonance type
# markers: dactionary mapping tagger to marker type
def MakeCmpPlot(data, output_name, title, taggers, markers):
    plt.grid()
    
    for i, t in enumerate(data):
        masspoint, purity = t
        tagger = taggers[i]
        plot = sns.lineplot(x=masspoint, y=purity, marker=markers[tagger], label=tagger, linestyle='', legend=False)
        plot.set_title(title)
        plot.set(xlabel='m(X), [GeV]', ylabel='Purity')

    plt.legend(loc='upper right')
    fig = plot.get_figure()
    fig.savefig(output_name)
    plt.close()


def SaveToFile(output_name, masspoint, purity):
    with open(output_name, 'w') as file:
        lines = [f"{masspoint[i]} {purity[i]}\n" for i in range(len(masspoint))]
        file.write("".join(lines))
