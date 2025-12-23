"""
Custom WandB plots for Q-value geometry analysis
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def log_q_geometry_plots(info: dict, step: int):
    """Create and log custom Q-value geometry plots to WandB
    
    Args:
        info: Info dict from critic_loss containing Q statistics
        step: Current training step
    """
    
    # ===== 1. Box Plot Comparison =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    box_data = {
        'Q_ID': {
            'min': info['critic/q_id/min'],
            'p25': info['critic/q_id/p25'],
            'median': info['critic/q_id/median'],
            'p75': info['critic/q_id/p75'],
            'max': info['critic/q_id/max'],
        },
        'Q_Policy': {
            'min': info['critic/q_policy/min'],
            'p25': info['critic/q_policy/p25'],
            'median': info['critic/q_policy/median'],
            'p75': info['critic/q_policy/p75'],
            'max': info['critic/q_policy/max'],
        },
        'Q_Infeasible': {
            'min': info['critic/q_inf/min'],
            'p25': info['critic/q_inf/p25'],
            'median': info['critic/q_inf/median'],
            'p75': info['critic/q_inf/p75'],
            'max': info['critic/q_inf/max'],
        },
    }
    
    positions = [1, 2, 3]
    labels = ['Q_ID', 'Q_Policy', 'Q_Infeasible']
    colors = ['blue', 'green', 'red']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        data = box_data[label]
        pos = positions[i]
        
        # Draw box
        ax.plot([pos-0.2, pos+0.2], [data['p25'], data['p25']], 
                color=color, linewidth=2)
        ax.plot([pos-0.2, pos+0.2], [data['p75'], data['p75']], 
                color=color, linewidth=2)
        ax.plot([pos-0.2, pos-0.2], [data['p25'], data['p75']], 
                color=color, linewidth=2)
        ax.plot([pos+0.2, pos+0.2], [data['p25'], data['p75']], 
                color=color, linewidth=2)
        
        # Median line
        ax.plot([pos-0.2, pos+0.2], [data['median'], data['median']], 
                color='black', linewidth=3)
        
        # Whiskers
        ax.plot([pos, pos], [data['p75'], data['max']], 
                color=color, linestyle='--', linewidth=1)
        ax.plot([pos, pos], [data['p25'], data['min']], 
                color=color, linestyle='--', linewidth=1)
        
        # Caps
        ax.plot([pos-0.1, pos+0.1], [data['max'], data['max']], 
                color=color, linewidth=1)
        ax.plot([pos-0.1, pos+0.1], [data['min'], data['min']], 
                color=color, linewidth=1)
    
    # Q_min reference line
    q_min = info['critic/ref/q_min']
    ax.axhline(q_min, color='black', linestyle='--', 
               linewidth=2, label=f'Q_min={q_min:.1f}')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Q-value')
    ax.set_title(f'Q-value Distribution Comparison (Step {step})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    wandb.log({'custom/q_geometry_boxplot': wandb.Image(fig)}, step=step)
    plt.close(fig)
    
    # ===== 2. Histogram Overlay =====
    if 'critic/hist/q_id' in info:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        q_id_hist = np.array(info['critic/hist/q_id'])
        q_inf_hist = np.array(info['critic/hist/q_inf'])
        bins = np.array(info['critic/hist/bins'])
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Normalize histograms
        q_id_hist_norm = q_id_hist / (q_id_hist.sum() + 1e-8)
        q_inf_hist_norm = q_inf_hist / (q_inf_hist.sum() + 1e-8)
        
        ax.bar(bin_centers, q_id_hist_norm, width=(bins[1]-bins[0])*0.8,
               alpha=0.6, label='Q_ID', color='blue')
        ax.bar(bin_centers, q_inf_hist_norm, width=(bins[1]-bins[0])*0.8,
               alpha=0.6, label='Q_Infeasible', color='red')
        
        ax.axvline(q_min, color='black', linestyle='--', 
                   linewidth=2, label=f'Q_min={q_min:.1f}')
        ax.axvline(info['critic/q_id/mean'], color='blue', 
                   linestyle='-', linewidth=2, alpha=0.8, label='Q_ID mean')
        ax.axvline(info['critic/q_inf/mean'], color='red', 
                   linestyle='-', linewidth=2, alpha=0.8, label='Q_inf mean')
        
        ax.set_xlabel('Q-value')
        ax.set_ylabel('Density')
        ax.set_title(f'Q-value Distribution Overlay (Step {step})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        wandb.log({'custom/q_geometry_histogram': wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    # ===== 3. Range Visualization =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Q_ID', 'Q_Policy', 'Q_Infeasible']
    y_pos = np.arange(len(categories))
    
    ranges = [
        (info['critic/q_id/min'], info['critic/q_id/max']),
        (info['critic/q_policy/min'], info['critic/q_policy/max']),
        (info['critic/q_inf/min'], info['critic/q_inf/max']),
    ]
    
    means = [
        info['critic/q_id/mean'],
        info['critic/q_policy/mean'],
        info['critic/q_inf/mean'],
    ]
    
    colors = ['blue', 'green', 'red']
    
    for i, (category, (q_min_val, q_max_val), mean, color) in \
            enumerate(zip(categories, ranges, means, colors)):
        
        # Range bar
        ax.barh(i, q_max_val - q_min_val, left=q_min_val, 
                height=0.3, alpha=0.5, color=color)
        
        # Mean marker
        ax.plot(mean, i, 'o', markersize=10, color=color, 
                label=f'{category} mean')
        
        # Text annotations
        ax.text(q_max_val + 5, i, f'{q_max_val:.1f}', 
                va='center', fontsize=9)
        ax.text(q_min_val - 5, i, f'{q_min_val:.1f}', 
                va='center', ha='right', fontsize=9)
    
    # Q_min reference
    ax.axvline(q_min, color='black', linestyle='--', 
               linewidth=2, label=f'Q_min={q_min:.1f}')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel('Q-value')
    ax.set_title(f'Q-value Ranges (Step {step})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='x')
    
    wandb.log({'custom/q_geometry_ranges': wandb.Image(fig)}, step=step)
    plt.close(fig)
    
    # ===== 4. Violation Tracking =====
    wandb.log({
        'violations/q_id_below_qmin': info['critic/q_id/below_qmin_ratio'],
        'violations/q_id_negative': info['critic/q_id/negative_ratio'],
        'violations/q_inf_below_qmin': info['critic/q_inf/below_qmin_ratio'],
        'violations/q_inf_above_qmin': info['critic/q_inf/above_qmin_ratio'],
        'violations/q_policy_negative': info['critic/q_policy/negative_ratio'],
    }, step=step)
    
    # ===== 5. Separation Quality =====
    wandb.log({
        'separation/gap_mean': info['critic/gap/id_inf_mean'],
        'separation/gap_median': info['critic/gap/id_inf_median'],
        'separation/score': info['critic/gap/separation_score'],
    }, step=step)


def create_q_geometry_summary_table(info: dict) -> wandb.Table:
    """Create a WandB table summarizing Q-value geometry
    
    Args:
        info: Info dict from critic_loss
        
    Returns:
        WandB table object
    """
    
    columns = ['Metric', 'Q_ID', 'Q_Policy', 'Q_Infeasible', 'Q_min (ref)']
    
    data = [
        ['Mean', f"{info['critic/q_id/mean']:.2f}", 
         f"{info['critic/q_policy/mean']:.2f}", 
         f"{info['critic/q_inf/mean']:.2f}", 
         f"{info['critic/ref/q_min']:.2f}"],
        
        ['Median', f"{info['critic/q_id/median']:.2f}", 
         f"{info['critic/q_policy/median']:.2f}", 
         f"{info['critic/q_inf/median']:.2f}", '-'],
        
        ['Std', f"{info['critic/q_id/std']:.2f}", 
         f"{info['critic/q_policy/std']:.2f}", 
         f"{info['critic/q_inf/std']:.2f}", '-'],
        
        ['Min', f"{info['critic/q_id/min']:.2f}", 
         f"{info['critic/q_policy/min']:.2f}", 
         f"{info['critic/q_inf/min']:.2f}", '-'],
        
        ['Max', f"{info['critic/q_id/max']:.2f}", 
         f"{info['critic/q_policy/max']:.2f}", 
         f"{info['critic/q_inf/max']:.2f}", '-'],
        
        ['P25', f"{info['critic/q_id/p25']:.2f}", 
         f"{info['critic/q_policy/p25']:.2f}", 
         f"{info['critic/q_inf/p25']:.2f}", '-'],
        
        ['P75', f"{info['critic/q_id/p75']:.2f}", 
         f"{info['critic/q_policy/p75']:.2f}", 
         f"{info['critic/q_inf/p75']:.2f}", '-'],
        
        ['Range', f"{info['critic/q_id/range']:.2f}", 
         f"{info['critic/q_policy/range']:.2f}", 
         f"{info['critic/q_inf/range']:.2f}", '-'],
        
        ['Below Q_min (%)', 
         f"{info['critic/q_id/below_qmin_ratio']*100:.1f}", 
         f"{info['critic/q_policy/below_qmin_ratio']*100:.1f}", 
         f"{info['critic/q_inf/below_qmin_ratio']*100:.1f}", '-'],
        
        ['Negative (%)', 
         f"{info['critic/q_id/negative_ratio']*100:.1f}", 
         f"{info['critic/q_policy/negative_ratio']*100:.1f}", 
         f"{info['critic/q_inf/negative_ratio']*100:.1f}", '-'],
    ]
    
    table = wandb.Table(columns=columns, data=data)
    return table