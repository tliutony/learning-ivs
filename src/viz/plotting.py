"""
Plotting functions for visualizing data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def kde_ridgeplot(long_df, row_name="method", value_name="estimate", pal="Set2", figsize=(12, 4)):
    """Plots a ridgeplot of the data in long_df.

    Assumes long_df is in "long" format, with row_name and value_name columns.

    Sourced from: https://seaborn.pydata.org/examples/kde_ridgeplot.html

    TODO fix weirdly inconsistent overlap of plots
    """
    n_methods = long_df[row_name].nunique()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(
        long_df, row=row_name, hue=row_name, palette=pal,
        height=figsize[1] / n_methods*2
    )
    # set the size of the figure manually to solve seaborn warning: https://stackoverflow.com/a/56622260
    g.figure.set_figwidth(figsize[0])
    g.figure.set_figheight(figsize[1])

    # Draw the densities in a few steps
    g.map(sns.kdeplot, value_name, fill=True, alpha=1, linewidth=1.5, thresh=0)
    g.map(sns.kdeplot, value_name, clip_on=False, color="black", lw=1,  thresh=0)
    # Set the subplots to overlap
    # calculate negative space between plots
    hspace = (figsize[1] / n_methods) * -1
    #print(hspace)
    g.figure.subplots_adjust(hspace=hspace)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(left=True)

    # passing color=None to refline() uses the hue mapping
    # g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, value_name)