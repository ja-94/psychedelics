import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

# Map QC flags to numbers for plotting
QCVAL2NUM = {  
    np.nan: 0.,
    'NOT SET': 0.01,
    'NOT_SET': 0.01,
    'PASS': 1.,
    'WARNING': 0.66,
    'CRITICAL': 0.33,
    'FAIL': 0.1
}
# Create colormap for QC grid plots
QCCMAP = colors.LinearSegmentedColormap.from_list(
    'qc_cmap',
    [(0., 'white'), (0.01, 'gray'), (0.1, 'palevioletred'), (0.33, 'violet'), (0.66, 'orange'), (1., 'limegreen')],
    N=256
)

def qc_grid(df, qc_columns=None, qcval2num=None, ax=None, xticklabels=None,
           legend=True):
    if qcval2num is None:
        qcval2num = QCVAL2NUM
    if qc_columns is None:
        qc_columns = df.columns
    df_qc = df[qc_columns].replace(qcval2num)
    if ax is None:
        fig, ax = plt.subplots()
    qcmat = df_qc.values.T.astype(float)
    ax.matshow(qcmat, cmap=QCCMAP, vmin=0, vmax=1)
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
            ax.scatter(-1, -1, color=QCCMAP(val), label=key)
        ax.set_xlim(left=-0.5)
        ax.set_ylim(top=-0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax