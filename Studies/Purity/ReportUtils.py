import csv
import pandas as pd
import matplotlib.pyplot as plt


def MakeCmpPlot(df1, df2, label1, label2, title, name):
    plt.errorbar(df1['mX'], df1['purity'], 
                 yerr=[df1['purity'] - df1['low'], df1['high'] - df1['purity']], 
                 fmt='o', capsize=5, capthick=2, elinewidth=2, label=label1)

    plt.errorbar(df2['mX'], df2['purity'], 
                 yerr=[df2['purity'] - df2['low'], df2['high'] - df2['purity']], 
                 fmt='s', capsize=5, capthick=2, elinewidth=2, label=label2)
    
    plt.title(title)
    plt.xlabel("m(X)")
    plt.ylabel("Purity")
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.legend(loc='upper right')
    plt.savefig(name, format='png')
    plt.close()


def SaveToFile(output_name, data):
    with open(output_name, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(["mX", "purity", "low", "high"])
        for row in data:
            csv_out.writerow(row)
