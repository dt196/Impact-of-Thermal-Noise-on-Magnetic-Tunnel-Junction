import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# CONFIGURATION

# Input files (from Script 1 output)
DATA_DIR = r'D:\Thesis.T\switching -step 1\switching_processed_data'
RAW_DATA_FILE = f'{DATA_DIR}\\switching_raw_data.csv'
STATS_FILE = f'{DATA_DIR}\\switching_statistics.csv'

# Output directory for figures
FIGURE_DIR = r'D:\Thesis.T\switching -step 1\switching_figures'

# PLOTTING CONFIGURATION

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

MATERIAL_COLORS = {
    'CoFeB/MgO': '#1f77b4',
    'Co/Ni': '#ff7f0e',
    'Co/Pt': '#2ca02c'
}

MATERIAL_MARKERS = {
    'CoFeB/MgO': 'o',
    'Co/Ni': 's',
    'Co/Pt': '^'
}


# FUNCTIONS
def load_data():
    """Load processed data files"""
    
    print(f"LOADING DATA")
    
    try:
        df_raw = pd.read_csv(RAW_DATA_FILE)
        print(f" Loaded raw data: {len(df_raw)} rows")
        print(f"  File: {RAW_DATA_FILE}")
    except FileNotFoundError:
        print(f" ERROR: {RAW_DATA_FILE} not found!")
        print(f"   Run switching_01_extract_data_UPDATED.py first!")
        return None, None
    except Exception as e:
        print(f" ERROR loading raw data: {e}")
        return None, None
    
    try:
        df_stats = pd.read_csv(STATS_FILE)
        print(f" Loaded statistics: {len(df_stats)} rows")
        print(f"  File: {STATS_FILE}")
    except FileNotFoundError:
        print(f" ERROR: {STATS_FILE} not found!")
        return None, None
    except Exception as e:
        print(f" ERROR loading stats: {e}")
        return None, None
    
    return df_raw, df_stats


def create_figure_sw1_prob_vs_temp(df_stats):
    """Figure SW1: RIBBON PLOT with gradient fill """
        
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Clean white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for material in sorted(df_stats['Material'].unique()):
        mat_data = df_stats[df_stats['Material'] == material].copy()
        
        # Get mean and std for each temperature
        temp_stats = mat_data.groupby('Temperature_K')['Success_Probability'].agg(['mean', 'std'])
        temps = temp_stats.index.values
        means = temp_stats['mean'].values * 100
        stds = temp_stats['std'].values * 100
        
        # Calculate ribbon bounds (mean ± std)
        upper_bound = means + stds
        lower_bound = np.maximum(means - stds, 0)  # Don't go below 0
        
        # Plot gradient ribbon (multiple alpha layers for gradient effect)
        for alpha_level in [0.4, 0.25, 0.15, 0.08]:
            scale = alpha_level * 2.5
            upper_scaled = means + (stds * scale)
            lower_scaled = np.maximum(means - (stds * scale), 0)
            ax.fill_between(temps, lower_scaled, upper_scaled, 
                            color=MATERIAL_COLORS[material], 
                            alpha=alpha_level/2, linewidth=0)
        
        # Main bold line
        ax.plot(temps, means, color=MATERIAL_COLORS[material], 
                linewidth=4, label=material, zorder=5)
        
        # Large markers with border
        ax.scatter(temps, means, s=200, color=MATERIAL_COLORS[material], 
                  marker=MATERIAL_MARKERS[material],
                  edgecolors='black', linewidths=2, zorder=6)
        
        # Add value labels at each data point
        for t, v in zip(temps, means):
            ax.annotate(f'{v:.1f}%', 
                       xy=(t, v), xytext=(0, 12), 
                       textcoords='offset points',
                       ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', 
                               edgecolor=MATERIAL_COLORS[material],
                               alpha=0.85, linewidth=1.5))
        
        # Add subtle connecting lines between points
        ax.plot(temps, means, color=MATERIAL_COLORS[material], 
                linewidth=1, linestyle=':', alpha=0.4, zorder=4)
    
   
    ax.set_xlabel('Temperature (K)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Switching Probability (%)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_title('Switching Probability vs Temperature\n(Gradient Uncertainty Bands)', 
                 fontsize=17, fontweight='bold', pad=20)
    
    ax.legend(fontsize=13, loc='lower left', framealpha=0.95, 
              edgecolor='gray', shadow=True)
    
    # Refined grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylim([0, 105])
    ax.set_xlim([245, 405])
    
    # Reference lines
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.3)
    ax.text(247, 51, '50%', fontsize=10, color='gray', alpha=0.6)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    filename = f'{FIGURE_DIR}\\SW1_switching_prob_vs_temp.{FIGURE_FORMAT}'
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f" Saved: {filename}")


def create_figure_sw2_failure_comparison(df_raw):
    """Figure SW2: Failure rate comparison bar chart"""
        
    failure_rates = []
    for material in sorted(df_raw['Material'].unique()):
        mat_data = df_raw[df_raw['Material'] == material]
        failure_rate = (1 - mat_data['Switched'].mean()) * 100
        failure_rates.append({
            'Material': material,
            'Failure_Rate_%': failure_rate,
            'Success_Rate_%': mat_data['Switched'].mean() * 100
        })
    
    df_failure = pd.DataFrame(failure_rates)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_failure))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_failure['Success_Rate_%'], width, 
                   label='Success Rate', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_failure['Failure_Rate_%'], width,
                   label='Failure Rate', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Material', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Switching Success vs Failure Rate by Material', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_failure['Material'], fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    filename = f'{FIGURE_DIR}\\SW2_failure_rate_comparison.{FIGURE_FORMAT}'
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def create_figure_sw3_prob_vs_current(df_stats):
    """Figure SW3: Switching probability vs drive current - GROUPED BAR CHART"""
        
    materials = sorted(df_stats['Material'].unique())
    
    data_by_material = {}
    for material in materials:
        mat_data = df_stats[df_stats['Material'] == material].copy()
        curr_avg = mat_data.groupby('Current_mA')['Success_Probability'].mean() * 100
        data_by_material[material] = curr_avg
    
    all_currents = []
    for material in materials:
        mat_data = df_stats[df_stats['Material'] == material]
        all_currents.extend(mat_data['Current_mA'].unique())
    unique_currents = sorted(set(all_currents))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(unique_currents))
    width = 0.25
    
    for idx, material in enumerate(materials):
        values = []
        for curr in unique_currents:
            if curr in data_by_material[material].index:
                values.append(data_by_material[material][curr])
            else:
                values.append(0)
        
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width, 
                     label=material, 
                     color=MATERIAL_COLORS[material],
                     alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Drive Current (mA)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Switching Probability (%)', fontsize=14, fontweight='bold')
    ax.set_title('Switching Probability vs Drive Current\n(Averaged Across All Temperatures)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{curr:.1f}' for curr in unique_currents], fontsize=10)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    filename = f'{FIGURE_DIR}\\SW3_switching_prob_vs_current.{FIGURE_FORMAT}'
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def create_figure_sw4_heatmap(df_stats):
    """Figure SW4: Heatmap (Temperature vs Current)"""
    
    materials = sorted(df_stats['Material'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, material in enumerate(materials):
        mat_data = df_stats[df_stats['Material'] == material].copy()
        pivot_data = mat_data.pivot_table(
            values='Success_Percent',
            index='Temperature_K',
            columns='Current_mA',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=axes[idx], cbar_kws={'label': 'Success %'},
                   vmin=0, vmax=100)
        
        axes[idx].set_title(f'{material}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Current (mA)', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Temperature (K)' if idx == 0 else '', 
                            fontsize=12, fontweight='bold')
    
    plt.suptitle('Switching Probability Heatmap: Temperature vs Current',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = f'{FIGURE_DIR}\\SW4_heatmap_temp_current.{FIGURE_FORMAT}'
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def create_figure_sw5_theta_distribution(df_raw):
    """Figure SW5: PERFECT - Threshold legend positioned """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, material in enumerate(sorted(df_raw['Material'].unique())):
        mat_data = df_raw[df_raw['Material'] == material]
        
        axes[idx].hist(mat_data['Theta_Final_deg'], bins=50, 
                      color=MATERIAL_COLORS[material], alpha=0.7, edgecolor='black')
        
        # Add threshold line
        axes[idx].axvline(0, color='red', linestyle='--', linewidth=2)
        
        axes[idx].set_xlabel('Final Theta (degrees)', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{material}', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        switched = (mat_data['Theta_Final_deg'] < 0).sum()
        not_switched = (mat_data['Theta_Final_deg'] >= 0).sum()
        
        # Position text boxes
        if material in ['Co/Ni', 'Co/Pt']:
            # Top-right
            text_x = 0.95
            text_y = 0.95
            h_align = 'right'
            legend_y = 0.84
        else:
            # CoFeB/MgO - top-left
            text_x = 0.05
            text_y = 0.95
            h_align = 'left'
            legend_y = 0.84  # Same spacing below box
        
        # Add text box with stats
        axes[idx].text(text_x, text_y, f'Switched: {switched}\nNot switched: {not_switched}',
                      transform=axes[idx].transAxes, fontsize=10,
                      verticalalignment='top', horizontalalignment=h_align,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add "Switching Threshold" JUST BELOW the box (very close, not far!)
        axes[idx].text(text_x, legend_y, '-- Switching Threshold',
                      transform=axes[idx].transAxes, fontsize=9,
                      verticalalignment='top', horizontalalignment=h_align,
                      color='red', style='italic', weight='bold')
    
    plt.suptitle('Distribution of Final Magnetization Angle',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = f'{FIGURE_DIR}\\SW5_theta_distribution.{FIGURE_FORMAT}'
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def create_figure_sw6_failure_vs_temp(df_stats):
    """Figure SW6: STEP PLOT with markers"""
    
    print(f"\nCreating Figure SW6: Step Plot with Error Bars...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Clean white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for material in sorted(df_stats['Material'].unique()):
        mat_data = df_stats[df_stats['Material'] == material].copy()
        
        # Get failure statistics
        temp_stats = mat_data.groupby('Temperature_K')['Failure_Percent'].agg(['mean', 'std'])
        temps = temp_stats.index.values
        means = temp_stats['mean'].values
        stds = temp_stats['std'].values
        
        # Plot STEP plot (different from SW1!)
        ax.step(temps, means, where='mid', 
                color=MATERIAL_COLORS[material], 
                linewidth=3, label=material, zorder=5)
        
        # Add error bars at data points
        ax.errorbar(temps, means, yerr=stds,
                   fmt='none', ecolor=MATERIAL_COLORS[material], 
                   elinewidth=2, capsize=6, capthick=2, alpha=0.6)
        
        # Large markers at data points
        ax.scatter(temps, means, s=180, 
                  color=MATERIAL_COLORS[material],
                  marker=MATERIAL_MARKERS[material],
                  edgecolors='black', linewidths=2.5, zorder=7)
        
        # Add value labels at each point
        for t, v in zip(temps, means):
            ax.annotate(f'{v:.1f}%', 
                       xy=(t, v), xytext=(8, 8), 
                       textcoords='offset points',
                       ha='left', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', 
                               edgecolor=MATERIAL_COLORS[material],
                               alpha=0.9, linewidth=1.5))
    

    ax.set_xlabel('Temperature (K)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Failure Rate (%)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_title('Thermal-Induced Failure Rate vs Temperature\n(Step Plot with Error Bars)',
                 fontsize=17, fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='upper left', framealpha=0.95, 
              edgecolor='gray', shadow=True)
    
    # Refined grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    ax.set_ylim([0, max(80, df_stats['Failure_Percent'].max() + 5)])
    ax.set_xlim([245, 405])
    
    # Add subtle warning zone
    ax.axhspan(50, 100, alpha=0.05, color='red', zorder=0)
    ax.text(248, 52, 'High Risk Zone (>50%)', fontsize=10, 
            color='red', style='italic', alpha=0.5)
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    filename = f'{FIGURE_DIR}\\SW6_failure_vs_temp.{FIGURE_FORMAT}'
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {filename}")


def perform_statistical_analysis(df_raw):
    """Perform statistical tests"""
    
    print(f"STATISTICAL ANALYSIS")
    
    materials = sorted(df_raw['Material'].unique())
    
    print(f"\n1. ONE-WAY ANOVA TEST")
    groups = [df_raw[df_raw['Material'] == mat]['Switched'] for mat in materials]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   P-value: {p_value:.6f}")
    print(f"   Result: {'SIGNIFICANT' if p_value < 0.05 else 'Not significant'}")
    
    print(f"\n2. PAIRWISE T-TESTS")
    from itertools import combinations
    
    for mat1, mat2 in combinations(materials, 2):
        data1 = df_raw[df_raw['Material'] == mat1]['Switched']
        data2 = df_raw[df_raw['Material'] == mat2]['Switched']
        t_stat, p_val = stats.ttest_ind(data1, data2)
        
        print(f"\n   {mat1} vs {mat2}:")
        print(f"     P-value: {p_val:.6f}")
        print(f"     {'SIGNIFICANT' if p_val < 0.05 else 'Not significant'}")
    
    stats_output = f'{FIGURE_DIR}\\statistical_analysis.txt'
    with open(stats_output, 'w') as f:
        f.write("STATISTICAL ANALYSIS RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"1. ONE-WAY ANOVA\n")
        f.write(f"   F-statistic: {f_stat:.4f}\n")
        f.write(f"   P-value: {p_value:.6f}\n")
        f.write(f"2. PAIRWISE COMPARISONS\n")
        for mat1, mat2 in combinations(materials, 2):
            data1 = df_raw[df_raw['Material'] == mat1]['Switched']
            data2 = df_raw[df_raw['Material'] == mat2]['Switched']
            t_stat, p_val = stats.ttest_ind(data1, data2)
            f.write(f"   {mat1} vs {mat2}: p={p_val:.6f}\n")
    
    print(f"\n Statistics saved to: {stats_output}")


def main():
    """Main execution"""
  
    print("SWITCHING DATA VISUALIZATION")
    
    try:
        Path(FIGURE_DIR).mkdir(exist_ok=True, parents=True)
        print(f"\n Figure directory: {FIGURE_DIR}")
    except Exception as e:
        print(f" ERROR creating figure directory: {e}")
        return
    
    df_raw, df_stats = load_data()
    
    if df_raw is None or df_stats is None:
        print("\n ERROR: Cannot proceed!")
        return
    
    print(f"CREATING FIGURES")
    
    create_figure_sw1_prob_vs_temp(df_stats)      # RIBBON PLOT 
    create_figure_sw2_failure_comparison(df_raw)
    create_figure_sw3_prob_vs_current(df_stats)
    create_figure_sw4_heatmap(df_stats)
    create_figure_sw5_theta_distribution(df_raw)  # SPACING (y=0.92)
    create_figure_sw6_failure_vs_temp(df_stats)   # STEP PLOT 
    
    perform_statistical_analysis(df_raw)

    print(f"ALL FIGURES CREATED")
    print(f"\nGenerated files in: {FIGURE_DIR}")
    print(f"  1. SW1 - Ribbon plot with gradient")
    print(f"  2. SW2 - Failure comparison")
    print(f"  3. SW3 - Grouped bar chart")
    print(f"  4. SW4 - Heatmap")
    print(f"  5. SW5 - Threshold position")
    print(f"  6. SW6 - Step plot with error bars")
    print(f"  7. Statistical analysis")


if __name__ == "__main__":
    main()
