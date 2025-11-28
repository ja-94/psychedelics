import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator

from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from psyfun import util
from psyfun.config import *

## FIXE: make this a function
plt.rcParams['figure.dpi'] = FIGUREDPI
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

def in2cm(inches):
    return inches * 2.54

def cm2in(cm):
    return cm / 2.54

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
    w = cm2in(w) # convert cm to inches
    h = cm2in(h)
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


def format_ticklabel(x):
    """
    Format tick label give tick value: 2 sig figs, scientific for large/small
    values.
    """
    if abs(x) >= 100 or (abs(x) < 0.01 and x != 0):
        return f'{x:.1e}'
    elif abs(x) >= 10:
        return f'{x:.0f}'
    elif abs(x) >= 1:
        return f'{x:.1f}'
    else:
        return f'{x:.2f}'



def auto_tick(ax, nticks=3, locator_class=MaxNLocator):
    """
    ax : ax.xaxis or ax.yaxis
    """
    if locator_class is None:
        # Get data limits
        data_min, data_max = ax.get_data_interval()

        # Generate evenly-spaced ticks
        ticks = np.linspace(data_min, data_max, nticks)
    else:
        locator = locator_class(nbins=nticks-1)
        ticks = locator.tick_values(*ax.get_view_interval())
    ax.set_ticks(ticks)
    ax.set_ticklabels([format_ticklabel(t) for t in ticks])


def clip_axes_to_ticks(ax=None, spines=['left', 'bottom'], ext={}):
    """
    Clip the axis lines to end at the minimum and maximum tick values.

    Parameters
    ----------
    ax : matplotlib.axes
        Axes to resize, if None the output of plt.gca() will be re-sized.

    spines : list
        Axes to keep and clip, axes not included in this list will be removed.
        Valid values include 'left', 'bottom', 'right', 'top'.

    ext : dict
        For each axis in ext.keys() ('left', 'bottom', 'right', 'top'),
        the axis line will be extended beyond the last tick by the value
        specified, e.g. {'left':[0.1, 0.2]} will results in an axis line
        that extends 0.1 units beyond the bottom tick and 0.2 unit beyond
        the top tick.
    """
    if ax is None:
        ax = plt.gca()
    spines2ax = {
        'left': ax.yaxis,
        'top': ax.xaxis,
        'right': ax.yaxis,
        'bottom': ax.xaxis
    }
    all_spines = ['left', 'bottom', 'right', 'top']
    for spine in spines:
        low = min(spines2ax[spine].get_majorticklocs())
        high = max(spines2ax[spine].get_majorticklocs())
        if spine in ext.keys():
            low += ext[spine][0]
            high += ext[spine][1]
        ax.spines[spine].set_bounds(low, high)
    for spine in [spine for spine in all_spines if spine not in spines]:
        ax.spines[spine].set_visible(False)


def qc_grid(
    df,
    qc_columns=None,
    qcval2num=None,
    ax=None,
    xticklabels=None,
    cmap=CMAPS['qc'],
    legend=True
    ):
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


def paired_plot(df, cols, groupby, ax=None, x=[0, 1], transform=None, **kwargs):
    """
    Plot paired changes in a metric from task00 to task01 for a single epoch.

    Parameters:
    -----------
    df : DataFrame
        Single unit data
    cols : [str, str]
        Columns to compare
    groupby : str
        Column to group data by
    transform : callable
        Function to apply to values before taking the mean
    """
    if ax is None:
        fig, ax = plt.subplots()
    assert len(cols) == len(x)
    for eid, data in df.groupby(groupby):
        if transform is not None:
            y = np.array([
                np.nanmean(transform(data[col].replace(0, np.nan))) for col in cols
                ])
        else:
            y = np.array([data[col].mean() for col in cols])
        ax.plot(x, y, **kwargs)
    return ax


def plot_proportion_by_group(
    df, col, group_by, condition_col='control_recording', sort_by=None,
    width=0.4, alpha=0.05, colors=None, orientation='vertical'
    ):
    """
    Plot mean of column by group and condition.

    Parameters:
    -----------
    df : DataFrame
        Data with boolean column to plot
    col : str
        Boolean column name (True = significant/positive result)
    group : str
        Grouping variable (e.g., 'coarse_region', 'subject')
    condition : str
        Condition variable for side-by-side bars (e.g., 'control_recording')
    sort_by : value from condition column, optional
        Sort groups by proportion in this condition (default: first condition)
    colors : dict, optional
        Map condition values to colors (e.g., {True: CONTROLCOLOR, False: LSDCOLOR})

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """

    n_conditions = df[condition_col].nunique()
    conditions = df[condition_col].unique()

    if colors is not None:
        assert len(colors) == n_conditions

    sorted_groups = util.sort_groups(
        df,  # dataframe to sort groups in
        col,  # column with values of interest
        group_by,  # grouping column
        aggfunc=np.mean,  # mean of T/F give proportion True
        reference_condition=(condition_col, sort_by) if sort_by is not None else sort_by  # region order determined by sorting LSD recording values
    )

    wpc = width / n_conditions
    x_offsets = np.arange(-width, width, width) + wpc

    fig, ax = plt.subplots()
    for condition, x_offset, color in zip(conditions, x_offsets, colors):
        # Get data for this condition
        df_condition = df.query(f'{condition_col} == @condition')

        # Counts per group
        n = df_condition.groupby(group_by).apply(lambda x: x[col].sum())
        N = df_condition.groupby(group_by).size()

        # Proportion of significant neurons pooled across recordings
        p = (n / N)
        y_neg, y_pos = proportion_confint(n, N, alpha=alpha)
        yerr = np.vstack([p - y_neg, y_pos - p])
        xpos = np.array([sorted_groups[group] for group in p.index])

        if orientation == 'vertical':
            ax.bar(xpos + x_offset, p, yerr=yerr, width=width, fc=color, ec='gray')
            ax.set_xticks(np.fromiter(sorted_groups.values(), dtype=int))
            ax.set_xticklabels(sorted_groups.keys())
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlim([-1, len(sorted_groups)])
        elif orientation == 'horizontal':
            ax.barh(xpos + x_offset, p, xerr=yerr, width=width, fc=color, ec='gray')
            ax.set_yticks(np.fromiter(sorted_groups.values(), dtype=int))
            ax.set_yticklabels(sorted_groups.keys())
            ax.tick_params(axis='y')
            ax.set_ylim([-1, len(sorted_groups)])

    return fig, ax


def plot_mean_by_group(
    df, col, group_by, condition_col=None, sort_by=None, agg_func=None,
    ascending=False, width=0.4, alpha=0.05, colors=None, orientation='vertical'
    ):
    """
    Plot mean of column by group and condition.

    Parameters:
    -----------
    df : DataFrame
        Data with boolean column to plot
    col : str
        Boolean column name (True = significant/positive result)
    group : str
        Grouping variable (e.g., 'coarse_region', 'subject')
    condition_col : str
        Condition variable for side-by-side bars (e.g., 'control_recording')
    sort_by : value from condition column, optional
        Sort groups by proportion in this condition (default: first condition)
    colors : dict, optional
        Map condition values to colors (e.g., {True: CONTROLCOLOR, False: LSDCOLOR})

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """

    n_conditions = df[condition_col].nunique()
    conditions = df[condition_col].unique()

    if colors is not None:
        assert len(colors) == n_conditions

    def _mean(group_data):
        return np.mean(group_data[col])

    if agg_func is None:
        agg_func = _mean

    sorted_groups = util.sort_groups(
        df,  # dataframe to sort groups in
        col,  # column with values of interest
        group_by,  # grouping column
        agg_func=agg_func,
        reference_condition=(condition_col, sort_by) if sort_by is not None else sort_by,  # region order determined by sorting LSD recording values
        ascending=ascending
    )

    wpc = width / n_conditions
    x_offsets = np.arange(-width, width, width) + wpc

    fig, ax = plt.subplots()
    for condition, x_offset, color in zip(conditions, x_offsets, colors):
        # Get data for this condition
        df_condition = df.query(f'{condition_col} == @condition').dropna(subset=col)

        # Counts per group
        means = df_condition.groupby(group_by).apply(
            lambda x: np.mean(x[col]), include_groups=False
            )
        sem = df_condition.groupby(group_by).apply(
            lambda x: stats.sem(x[col]), include_groups=False
            )
        xpos = np.array([sorted_groups[group] for group in means.index])

        if orientation == 'vertical':
            ax.bar(xpos + x_offset, means, yerr=sem, width=width, fc=color, ec='gray')
            ax.set_xticks(np.fromiter(sorted_groups.values(), dtype=int))
            ax.set_xticklabels(sorted_groups.keys())
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlim([-1, len(sorted_groups)])
        elif orientation == 'horizontal':
            ax.barh(xpos + x_offset, means, xerr=sem, height=width, fc=color, ec='gray')
            ax.set_yticks(np.fromiter(sorted_groups.values(), dtype=int))
            ax.set_yticklabels(sorted_groups.keys())
            ax.tick_params(axis='y')
            ax.set_ylim([-1, len(sorted_groups)])

    return fig, ax


def plot_difference_by_group(
    df, col, group_by, compare, condition_col, sort_by=None,
    ascending=False, width=0.4, alphas=None, colors=None, fill_color=None
    ):
    """
    Plot mean of column by group and condition.

    Parameters:
    -----------
    df : DataFrame
        Data with boolean column to plot
    col : str
        Boolean column name (True = significant/positive result)
    group : str
        Grouping variable (e.g., 'coarse_region', 'subject')
    compare : str
    condition : str
        Condition variable for side-by-side bars (e.g., 'control_recording')
    sort_by : value from compare column, optional
        Sort groups by proportion in this condition (default: first condition)
    colors : dict, optional
        Map condition values to colors (e.g., {True: CONTROLCOLOR, False: LSDCOLOR})

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    conditions = df[condition_col].unique()
    # ~assert len(conditions) == 2

    sorted_groups = util.sort_groups(
        df,  # dataframe to sort groups in
        col,  # column with values of interest
        group_by,  # grouping column
        aggfunc=np.mean,
        reference_condition=(compare, sort_by) if sort_by is not None else sort_by,  # region order determined by sorting LSD recording values
        ascending=ascending
    )

    bins = np.linspace(-1, 1, 51)
    xvals = np.linspace(-1, 1, 1000)

    fig, ax = plt.subplots()
    for group_name, group_data in df.groupby(group_by):
        for condition in conditions:
            condition_data = group_data.query(f'{condition_col} == @condition')
            if len(condition_data[compare].unique()) < 2:
                continue
            pdfs = condition_data.dropna(subset=col).groupby(compare)[col].apply(
                lambda x:
                    # ~np.histogram(x, bins=bins, weights=np.ones_like(x)/len(x))[0]
                    stats.gaussian_kde(x)(xvals)
                    if len(x) > 10 else np.full(len(xvals), np.nan)
                )
            # KS test
            from scipy.stats import ks_2samp
            data_control = condition_data.dropna(subset=col).query(f'{compare} == True')[col]
            data_lsd = condition_data.dropna(subset=col).query(f'{compare} == False')[col]
            ks_stat, ks_pval = ks_2samp(data_control, data_lsd)
            # Get specific conditions
            pdf_control = pdfs.loc[True]   # Assuming True = control
            pdf_lsd = pdfs.loc[False]       # Assuming False = LSD
            # Explicit subtraction (LSD - Control)
            diff = pdf_lsd - pdf_control
            ax.plot(
                # ~bins[:-1] + np.diff(bins).mean() / 2,
                xvals,
                # ~diff * 10 + sorted_groups[group_name],
                diff + sorted_groups[group_name],
                color=colors[condition],
                # ~color='black',
                alpha=alphas[condition]
                )
            ax.fill_between(
                # ~bins[:-1] + np.diff(bins).mean() / 2,
                xvals,
                # ~diff * 10 + sorted_groups[group_name],
                diff + sorted_groups[group_name],
                sorted_groups[group_name],
                where=diff >= 0,
                color=fill_color[0],
                alpha=0.25
                )
            ax.fill_between(
                # ~bins[:-1] + np.diff(bins).mean() / 2,
                xvals,
                # ~diff * 10 + sorted_groups[group_name],
                diff + sorted_groups[group_name],
                sorted_groups[group_name],
                where=diff < 0,
                color=fill_color[1],
                alpha=0.25
                )
            # Add significance stars
            if ks_pval < 0.001:
                stars = '***'
            elif ks_pval < 0.01:
                stars = '**'
            elif ks_pval < 0.05:
                stars = '*'
            else:
                stars = ''
            if stars:
                ax.text(
                    xvals[-1] + 0.05 * (xvals[-1] - xvals[0]),  # Slightly to the right of the line
                    diff[-1] + sorted_groups[group_name],  # At the end of the line
                    stars,
                    fontsize=12,
                    va='center',
                    color=colors[condition]
                )
        ax.axhline(sorted_groups[group_name], ls='--', color='gray')
    ax.set_yticks(np.fromiter(sorted_groups.values(), dtype=int))
    ax.set_yticklabels(sorted_groups.keys())
    ax.tick_params(axis='y')
    ax.set_ylim([-1, len(sorted_groups)])

    return fig, ax
