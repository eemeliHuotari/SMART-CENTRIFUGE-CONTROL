import pandas as pd
import matplotlib.pyplot as plt

def hist(series, title, xlabel, outfile=None):
    plt.figure()
    plt.hist(series.dropna(), bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.show()
