import re
from datetime import datetime
import numpy as np
from scipy import stats

from psyfun.config import *

def bootstrap_median_ci(data, n_bootstrap=1000, alpha=0.05):
    """
    Compute a bootstrap confidence interval for the median of an array.
    
    Parameters
    ----------
    data : array_like
        The data from which to compute the median.
    n_bootstrap : int, optional
        The number of bootstrap samples to use. Default is 1000.
    alpha : float, optional
        Significance level (e.g. 0.05 gives a 95% CI).
        
    Returns
    -------
    (ci_lower, ci_upper) : tuple of floats
        The lower and upper bounds of the bootstrap confidence interval.
    """
    medians = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        medians[i] = np.median(sample)
    ci_lower = np.percentile(medians, 100 * (alpha / 2))
    ci_upper = np.percentile(medians, 100 * (1 - alpha / 2))
    return ci_lower, ci_upper

def sort_groups(df, value_col, groupby_col, aggfunc=np.median,
                reference_condition=None, ascending=False):
    """
    Compute sorting order based on global data or a reference condition.
    
    Parameters
    ----------
    df : pd.DataFrame
        data to sort
    value_col : str 
        column containing values to sort by, will be aggregated within groups
    groupby_col : str
        column to group by
    aggfunc : callable
        function for aggregating values within each group (default: np.median)
    reference_condition : None, tuple
        if None, use all data to determine sort order, otherwise use only data 
        where condition_col takes on the given value (condition_col, value)
    ascending : bool
        if True, sort ascending; if False, sort descending (default: False)
    
    Returns
    -------
    sorted_regions : dict
        map specifying the sorted position for each group
    """
    # Get groups from data if not given
    groups = df[groupby_col].unique()
    # Filter data if using reference condition
    if reference_condition is not None:
        assert len(reference_condition) == 2
        ref_col, ref_val = reference_condition
        df = df[df[ref_col] == ref_val].copy()
    
    # Get groups and their aggregated values
    group_values = {}
    for group in groups:
        group_data = df[df[groupby_col] == group]
        group_values[group] = aggfunc(group_data[value_col])
    
    # Sort groups by their aggregated values
    sorted_groups = sorted(group_values.keys(), 
                          key=lambda x: group_values[x], 
                          reverse=not ascending)
    sorted_groups = {group: idx for idx, group in enumerate(sorted_groups)}
    
    return sorted_groups


def label_first_sessions(df, condition=None, label='first_session'):
    """
    Label the first experimental session for each subject.
    
    Parameters
    ----------
    df : pd.DataFrame 
        with columns 'subject', 'control_recording', 'start_time'
    
    Returns
    -------
    df : pd.DataFrame
        with additional 'first_lsd_session' boolean column
    """
    # Convert start_time to datetime
    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
        df['start_time'] = df['start_time'].apply(datetime.fromisoformat)
    
    # Filter for LSD sessions only and sort by subject and start_time
    if condition is not None:
        condition_col, condition_val = condition
        df_condition = (df[df[condition_col] == condition_val].sort_values(['subject', 'start_time']))
    else:
        df_condition = df
    
    # Mark the first experimental session for each subject
    first_mask = df_condition.groupby('subject').cumcount() == 0
    
    # Create the label column
    df[label] = False
    df.loc[df_condition[first_mask].index, label] = True
    
    return df


def sliding_epochs(df, t0='LSD_admin', epochs=postLSD_epoch_starts, length=postLSD_epoch_length, return_cols=False):
    prefix = re.sub(r'(_start|_stop)$', '', t0)
    # Prepare a dict to hold all new columns
    new_cols = {}
    for epoch in epochs:
        label = str(int(epoch))
        # Compute the start values for this epoch
        start_values = df[t0] + epoch
        stop_values = start_values + length
        new_cols[f'{prefix}_{label}_start'] = start_values
        new_cols[f'{prefix}_{label}_stop'] = stop_values
    # Create a DataFrame from new_cols and concatenate to original
    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    # Drop duplicate columns
    df = df.drop(columns=[col for col in new_cols_df.columns if col in df.columns])
    df = pd.concat([df, new_cols_df], axis=1)
    if return_cols:
        col_names = [col.rsplit('_', 1)[0] for col in new_cols if col.endswith('_start')]
        return df, col_names
    else:
        return df

def exponential_decay(x, alpha, beta, tau):
    return  alpha * np.exp(-x / tau) + beta

def bi_exponential_decay(x, tau1, tau2, p):
    return p * np.exp(-x / tau1) + (1 - p) * np.exp(-x / tau2)

def _get_exp_tau(eig):
    xx = np.arange(len(eig))
    yy = (eig - eig.min()) / (eig.max() - eig.min())
    try:
        (tau), pcov = optimize.curve_fit(lambda x, tau: exponential_decay(x, 1, 0, tau), xx, yy)
    except:
        tau = np.nan
    return tau

def _get_biexp_mean_lifetime(eig):
    xx = np.arange(len(eig))
    yy = (eig - eig.min()) / (eig.max() - eig.min())
    try:
        (tau1, tau2, p), pcov = curve_fit(bi_exponential_decay, xx, yy)
        mean_lifetime = p * tau1 + (1 - p) * tau2
    except:
        mean_lifetime = np.nan
    return mean_lifetime

def power_law_slope(eig, rank=100):
    if len(eig) < rank:
        rank = len(eig)
    xx = np.arange(rank) + 1
    res = stats.linregress(np.log(xx), np.log(eig[:rank]))
    return res.slope

def ngsc(a):
    return stats.entropy(a) / np.log2(len(a))

def angle(v1, v2):
    """
    Compute the angle between two vectors v1 and v2.
    
    Parameters
    ----------
    v1, v2 : array_like
        The two vectors (e.g. PC1 directions) between which to compute the angle.
    
    Returns
    -------
    angle_rad : float
        Angle in radians.
    angle_deg : float
        Angle in degrees.
    """
    # Compute dot product and norms.
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid numerical issues: clip the cosine value to the valid range [-1, 1]
    cos_theta = np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    # angle_deg = np.degrees(angle_rad)
    return angle_rad

## From Claude, for multi-level modeling
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_icc_oneway(data, groups):
    """
    Calculate one-way random effects ICC
    ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
    where k is average group size
    """
    df = pd.DataFrame({'value': data, 'group': groups})
    
    # Overall statistics
    grand_mean = df['value'].mean()
    n_groups = df['group'].nunique()
    total_n = len(df)
    
    # Group statistics
    group_stats = df.groupby('group')['value'].agg(['count', 'mean', 'var']).reset_index()
    group_stats.columns = ['group', 'n_j', 'mean_j', 'var_j']
    
    # Calculate sum of squares
    # Between groups sum of squares
    SSB = sum(group_stats['n_j'] * (group_stats['mean_j'] - grand_mean)**2)
    
    # Within groups sum of squares  
    SSW = sum((group_stats['n_j'] - 1) * group_stats['var_j'])
    
    # Degrees of freedom
    dfB = n_groups - 1
    dfW = total_n - n_groups
    
    # Mean squares
    MSB = SSB / dfB
    MSW = SSW / dfW
    
    # Average group size (for unbalanced design)
    n_groups_total = len(group_stats)
    k = (total_n - sum(group_stats['n_j']**2) / total_n) / (n_groups_total - 1)
    
    # ICC calculation
    icc = (MSB - MSW) / (MSB + (k - 1) * MSW)
    
    # F-statistic and p-value
    f_stat = MSB / MSW
    p_value = 1 - stats.f.cdf(f_stat, dfB, dfW)
    
    return {
        'ICC': max(0, icc),  # ICC can't be negative
        'F_statistic': f_stat,
        'p_value': p_value,
        'MSB': MSB,
        'MSW': MSW,
        'SSB': SSB,
        'SSW': SSW,
        'n_groups': n_groups,
        'avg_group_size': k
    }

def analyze_clustering_structure(df_rates):
    """
    Comprehensive analysis of clustering structure in neural data
    """
    results = {}
    
    print("=== CLUSTERING ANALYSIS FOR NEURAL FIRING RATE DATA ===\n")
    
    # 1. Overall ICCs
    print("1. OVERALL INTRACLASS CORRELATIONS")
    print("-" * 40)
    
    # Mouse-level ICC (ignoring neurons)
    mouse_icc = calculate_icc_oneway(df_rates['lograte'], df_rates['subject'])
    results['mouse_icc'] = mouse_icc
    print(f"Mouse-level ICC: {mouse_icc['ICC']:.4f}")
    print(f"  F({mouse_icc['n_groups']-1}, {len(df_rates)-mouse_icc['n_groups']}) = {mouse_icc['F_statistic']:.3f}, p = {mouse_icc['p_value']:.3e}")
    print(f"  Interpretation: {mouse_icc['ICC']*100:.1f}% of variance is between mice")
    
    # Neuron-level ICC
    neuron_icc = calculate_icc_oneway(df_rates['lograte'], df_rates['eid'])
    results['neuron_icc'] = neuron_icc
    print(f"\nNeuron-level ICC: {neuron_icc['ICC']:.4f}")
    print(f"  F({neuron_icc['n_groups']-1}, {len(df_rates)-neuron_icc['n_groups']}) = {neuron_icc['F_statistic']:.3f}, p = {neuron_icc['p_value']:.3e}")
    print(f"  Interpretation: {neuron_icc['ICC']*100:.1f}% of variance is between neurons")
    
    # 2. Condition-specific ICCs
    print("\n2. CONDITION-SPECIFIC ICCs")
    print("-" * 40)
    
    results['condition_iccs'] = {}
    for task in df_rates['task'].unique():
        for epoch in df_rates['epoch'].unique():
            subset = df_rates[(df_rates['task'] == task) & (df_rates['epoch'] == epoch)]
            if len(subset) > 0:
                # Mouse ICC within this condition
                mouse_icc_cond = calculate_icc_oneway(subset['lograte'], subset['subject'])
                neuron_icc_cond = calculate_icc_oneway(subset['lograte'], subset['eid'])
                
                condition = f"{task}_{epoch}"
                results['condition_iccs'][condition] = {
                    'mouse': mouse_icc_cond,
                    'neuron': neuron_icc_cond
                }
                
                print(f"\n{condition}:")
                print(f"  Mouse ICC: {mouse_icc_cond['ICC']:.4f}")
                print(f"  Neuron ICC: {neuron_icc_cond['ICC']:.4f}")
    
    # 3. Recording session analysis
    if 'control_recording' in df_rates.columns:
        print("\n3. RECORDING SESSION ANALYSIS")  
        print("-" * 40)
        
        # Create recording session identifier
        df_rates['recording_session'] = df_rates['subject'].astype(str) + '_' + df_rates['control_recording'].astype(str)
        session_icc = calculate_icc_oneway(df_rates['lograte'], df_rates['recording_session'])
        results['session_icc'] = session_icc
        
        print(f"Recording Session ICC: {session_icc['ICC']:.4f}")
        print(f"  Interpretation: {session_icc['ICC']*100:.1f}% of variance is between recording sessions")
        
        # Within vs between session neuron correlations
        session_neuron_stats = []
        for session in df_rates['recording_session'].unique():
            session_data = df_rates[df_rates['recording_session'] == session]
            if len(session_data['eid'].unique()) > 1:  # Need multiple neurons
                neuron_icc_within = calculate_icc_oneway(session_data['lograte'], session_data['eid'])
                session_neuron_stats.append({
                    'session': session,
                    'mouse': session.split('_')[0],
                    'control': session.split('_')[1],
                    'n_neurons': len(session_data['eid'].unique()),
                    'neuron_icc': neuron_icc_within['ICC']
                })
        
        if session_neuron_stats:
            session_df = pd.DataFrame(session_neuron_stats)
            results['session_neuron_stats'] = session_df
            print(f"\nAverage within-session neuron ICC: {session_df['neuron_icc'].mean():.4f}")
            print(f"Range: {session_df['neuron_icc'].min():.4f} - {session_df['neuron_icc'].max():.4f}")
    
    # 4. Variance decomposition
    print("\n4. VARIANCE DECOMPOSITION GUIDANCE")
    print("-" * 40)
    
    mouse_var_prop = mouse_icc['ICC']
    neuron_var_prop = neuron_icc['ICC']
    
    print(f"Variance Components (approximate):")
    print(f"  Between mice: {mouse_var_prop*100:.1f}%")
    print(f"  Between neurons: {neuron_var_prop*100:.1f}%") 
    print(f"  Within neuron (residual): {(1-max(mouse_var_prop, neuron_var_prop))*100:.1f}%")
    
    # Model recommendations
    print(f"\n5. MODELING RECOMMENDATIONS")
    print("-" * 40)
    
    if neuron_var_prop > mouse_var_prop:
        print("✓ RECOMMENDATION: Use neuron (eid) as primary grouping variable")
        print("  Neuron-level clustering is stronger than mouse-level")
        print("  Mouse effects will be partially captured through neuron correlation")
    else:
        print("✓ RECOMMENDATION: Use mouse (subject) as primary grouping variable") 
        print("  Mouse-level clustering is stronger than neuron-level")
        print("  Consider neuron as nested within mouse if using R")
    
    if 'session_icc' in results and results['session_icc']['ICC'] > 0.1:
        print("✓ Recording session effects are substantial - consider nested modeling")
    
    design_efficiency = 1 - max(mouse_var_prop, neuron_var_prop)
    if design_efficiency < 0.5:
        print("⚠ WARNING: High clustering reduces effective sample size")
        print("  Consider this when interpreting statistical power")
    
    return results

def plot_icc_comparison(results):
    """Create visualization of ICC analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Overall ICCs
    iccs = [results['mouse_icc']['ICC'], results['neuron_icc']['ICC']]
    if 'session_icc' in results:
        iccs.append(results['session_icc']['ICC'])
        labels = ['Mouse', 'Neuron', 'Recording\nSession']
    else:
        labels = ['Mouse', 'Neuron']
    
    axes[0,0].bar(labels, iccs, color=['skyblue', 'lightcoral', 'lightgreen'][:len(iccs)])
    axes[0,0].set_ylabel('Intraclass Correlation')
    axes[0,0].set_title('Overall ICCs by Grouping Level')
    axes[0,0].set_ylim(0, max(iccs) * 1.2)
    
    # Plot 2: Condition-specific mouse ICCs
    if 'condition_iccs' in results:
        conditions = list(results['condition_iccs'].keys())
        mouse_iccs = [results['condition_iccs'][c]['mouse']['ICC'] for c in conditions]
        
        axes[0,1].bar(range(len(conditions)), mouse_iccs, color='skyblue')
        axes[0,1].set_xticks(range(len(conditions)))
        axes[0,1].set_xticklabels(conditions, rotation=45)
        axes[0,1].set_ylabel('Mouse ICC')
        axes[0,1].set_title('Mouse ICC by Condition')
    
    # Plot 3: Condition-specific neuron ICCs  
    if 'condition_iccs' in results:
        neuron_iccs = [results['condition_iccs'][c]['neuron']['ICC'] for c in conditions]
        
        axes[1,0].bar(range(len(conditions)), neuron_iccs, color='lightcoral')
        axes[1,0].set_xticks(range(len(conditions)))
        axes[1,0].set_xticklabels(conditions, rotation=45)
        axes[1,0].set_ylabel('Neuron ICC') 
        axes[1,0].set_title('Neuron ICC by Condition')
    
    # Plot 4: Session-level analysis if available
    if 'session_neuron_stats' in results:
        session_df = results['session_neuron_stats']
        axes[1,1].scatter(session_df['n_neurons'], session_df['neuron_icc'], 
                         c=['blue' if x=='True' else 'red' for x in session_df['control']])
        axes[1,1].set_xlabel('Number of Neurons in Session')
        axes[1,1].set_ylabel('Within-Session Neuron ICC')
        axes[1,1].set_title('Neuron Clustering by Session')
        # Add legend
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Control')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Experimental')
        axes[1,1].legend(handles=[blue_patch, red_patch])
    
    plt.tight_layout()
    plt.show()

# Example usage:
# results = analyze_clustering_structure(df_rates)
# plot_icc_comparison(results)