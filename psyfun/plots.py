import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from psyfun.config import *

LABELFONTSIZE = 8
plt.rcParams['figure.dpi'] = 180
plt.rcParams['axes.labelsize'] = LABELFONTSIZE
plt.rcParams['xtick.labelsize'] = LABELFONTSIZE 
plt.rcParams['ytick.labelsize'] = LABELFONTSIZE 
plt.rcParams['legend.fontsize'] = LABELFONTSIZE 
plt.rcParams['axes.titlesize'] = LABELFONTSIZE

def pval2stars(p, ns='n.s.', na='n/a'):
    if np.isnan(p):
        return na
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ns

def set_plotsize(w, h=None, ax=None):
    """
    Set the size of a matplotlib axes object in cm.

    Parameters
    ----------
    w, h : float
        Desired width and height of plot, if height is None, the axis will be
        square.

    ax : matplotlib.axes
        Axes to resize, if None the output of plt.gca() will be re-sized.

    Notes
    -----
    - Use after subplots_adjust (if adjustment is needed)
    - Matplotlib axis size is determined by the figure size and the subplot
      margins (r, l; given as a fraction of the figure size), i.e.
      w_ax = w_fig * (r - l)
    """
    if h is None: # assume square
        h = w
    w /= 2.54 # convert cm to inches
    h /= 2.54
    if not ax: # get current axes
        ax = plt.gca()
    # get margins
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    # set fig dimensions to produce desired ax dimensions
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def qc_grid(df, qc_columns=None, qcval2num=None, ax=None, xticklabels=None, cmap=cmaps['qc'],
           legend=True):
    if qcval2num is None:
        qcval2num = QCVAL2NUM
    if qc_columns is None:
        qc_columns = df.columns
    df_qc = df[qc_columns].replace(qcval2num)
    if ax is None:
        fig, ax = plt.subplots()
    qcmat = df_qc.values.T.astype(float)
    ax.matshow(qcmat, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(df_qc)))
    if type(xticklabels) == str:
        ax.set_xticklabels(df[xticklabels])
        ax.tick_params(axis='x', rotation=90)
    elif type(xticklabels) == list:
        xticklabels = df.apply(lambda x: '_'.join(x[xticklabels].astype(str)), axis='columns')
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='x', rotation=90)
    ax.set_yticks(np.arange(len(df_qc.columns)))
    ax.set_yticklabels(qc_columns)
    for xtick in ax.get_xticks():
        ax.axvline(xtick - 0.5, color='white')
    for ytick in ax.get_yticks():
        ax.axhline(ytick - 0.5, color='white')
    if legend:
        for key, val in qcval2num.items():
            ax.scatter(-1, -1, color=cmap(val), label=key)
        ax.set_xlim(left=-0.5)
        ax.set_ylim(top=-0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax