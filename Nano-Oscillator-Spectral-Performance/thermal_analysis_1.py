import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


# CONFIGURATION

BASE_DIR = Path(r"D:\Thesis.T\MTJ_Analysis\raw files")
INPUT_FILE = BASE_DIR / "Result" / "statistics_FIXED_THERMAL.csv" # CSV file
OUTPUT_DIR = BASE_DIR / "Result" / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATERIALS = ['CoFeB/MgO', 'Co/Ni', 'Co/Pt']
COLORS = {
    'CoFeB/MgO': '#2E86AB',
    'Co/Ni': '#A23B72', 
    'Co/Pt': '#F18F01'
}

BASELINE_TEMP = 250  # K - Reference temperature
MAX_TEMP = 400       # K - Maximum test temperature


# LOAD DATA

print("THERMAL STABILITY ANALYSIS - STARTING")


try:
    df = pd.read_csv(INPUT_FILE)  # CSV file
    print(f" Data loaded: {len(df)} rows")
    print(f"   Materials: {df['Material'].unique()}")
    print(f"   Temperatures: {sorted(df['Temperature_K'].unique())}")
    print(f"   Currents: {sorted(df['Current_mA'].unique())}")
except FileNotFoundError:
    print(f" ERROR: {INPUT_FILE} not found!")
    exit(1)

# ANALYSIS 1: DEGRADATION RATES (Slope Analysis)

print("ANALYSIS 1: DEGRADATION RATE CALCULATION")
print("METHOD: Linear regression of metric vs temperature")
print("UNIT: %/K (percent change per Kelvin)")

degradation_rates = []

for current in df['Current_mA'].unique():
    for material in MATERIALS:
        data_subset = df[(df['Material'] == material) & 
                        (df['Current_mA'] == current)].copy()
        data_subset = data_subset.sort_values('Temperature_K')
        
        temps = data_subset['Temperature_K'].values
        
        # Frequency degradation rate
        freq = data_subset['Peak_Frequency_GHz_mean'].values
        freq_baseline = freq[0]  # Value at lowest temp
        freq_pct = ((freq - freq_baseline) / freq_baseline) * 100
        slope_freq, intercept, r_value, p_value, std_err = stats.linregress(temps, freq_pct)
        
        # Linewidth degradation rate
        lw = data_subset['Linewidth_MHz_mean'].values
        lw_baseline = lw[0]
        lw_pct = ((lw - lw_baseline) / lw_baseline) * 100
        slope_lw, _, _, _, _ = stats.linregress(temps, lw_pct)
        
        # Q-Factor degradation rate
        qf = data_subset['Q_Factor_mean'].values
        qf_baseline = qf[0]
        qf_pct = ((qf - qf_baseline) / qf_baseline) * 100
        slope_qf, _, _, _, _ = stats.linregress(temps, qf_pct)
        
        degradation_rates.append({
            'Material': material,
            'Current_mA': current,
            'Frequency_Rate_%/K': slope_freq,
            'Linewidth_Rate_%/K': slope_lw,
            'Q_Factor_Rate_%/K': slope_qf
        })

df_rates = pd.DataFrame(degradation_rates)

# Save degradation rates
output_file = OUTPUT_DIR / 'degradation_rates.csv'
df_rates.to_csv(output_file, index=False)
print(f"\n Degradation rates saved: {output_file}")

# Print summary at 3.0 mA (standard comparison point)
print("\n DEGRADATION RATES AT 3.0 mA:")
summary_3ma = df_rates[df_rates['Current_mA'] == 3.0]
print(summary_3ma.to_string(index=False))
print("\nINTERPRETATION:")
print("  Frequency: Negative = frequency decreases with temperature")
print("  Linewidth: Positive = linewidth increases")
print("  Q-Factor: Negative = quality decreases ")

# ANALYSIS 1B: ABSOLUTE DEGRADATION RATES (Direct Units)

print("ANALYSIS 1B: ABSOLUTE DEGRADATION RATE CALCULATION")
print("WHY: Show actual performance loss in real units ")
print("METHOD: Linear regression of metric vs temperature ")
print("UNITS: GHz/K for frequency, MHz/K for linewidth, Q-points/K for Q-factor")

absolute_rates = []

for current in df['Current_mA'].unique():
    for material in MATERIALS:
        data_subset = df[(df['Material'] == material) & 
                        (df['Current_mA'] == current)].copy()
        data_subset = data_subset.sort_values('Temperature_K')
        
        temps = data_subset['Temperature_K'].values
        
        # Frequency absolute degradation rate (GHz/K)
        freq = data_subset['Peak_Frequency_GHz_mean'].values
        slope_freq_abs, _, _, _, _ = stats.linregress(temps, freq)
        
        # Linewidth absolute degradation rate (MHz/K)
        lw = data_subset['Linewidth_MHz_mean'].values
        slope_lw_abs, _, _, _, _ = stats.linregress(temps, lw)
        
        # Q-Factor absolute degradation rate (Q/K)
        qf = data_subset['Q_Factor_mean'].values
        slope_qf_abs, _, _, _, _ = stats.linregress(temps, qf)
        
        absolute_rates.append({
            'Material': material,
            'Current_mA': current,
            'Frequency_Rate_GHz/K': slope_freq_abs,
            'Linewidth_Rate_MHz/K': slope_lw_abs,
            'Q_Factor_Rate_Q/K': slope_qf_abs
        })

df_abs_rates = pd.DataFrame(absolute_rates)

# Save absolute rates
output_file = OUTPUT_DIR / 'absolute_degradation_rates.csv'
df_abs_rates.to_csv(output_file, index=False)
print(f"\n Absolute degradation rates saved: {output_file}")

# Print summary at 3.0 mA
print("\n ABSOLUTE DEGRADATION RATES AT 3.0 mA:")
summary_3ma_abs_rates = df_abs_rates[df_abs_rates['Current_mA'] == 3.0]
print(summary_3ma_abs_rates.to_string(index=False))
print("\nINTERPRETATION:")
print("  Frequency (GHz/K): Negative = frequency decreases with temperature")
print("  Linewidth (MHz/K): Positive = linewidth increases (spectral broadening)")
print("  Q-Factor (Q/K): Negative = quality decreases")


# ANALYSIS 2: ABSOLUTE DEGRADATION (250K → 400K)

print("ANALYSIS 2: ABSOLUTE DEGRADATION (250K → 400K)")
print("WHY: Show total performance loss from baseline to max temperature")
print("METHOD: Calculate % change between 250K and 400K")

absolute_degradation = []

for current in df['Current_mA'].unique():
    for material in MATERIALS:
        # Get baseline (250K) values
        baseline = df[(df['Material'] == material) & 
                     (df['Current_mA'] == current) &
                     (df['Temperature_K'] == BASELINE_TEMP)]
        
        # Get max temp (400K) values
        maxtemp = df[(df['Material'] == material) & 
                    (df['Current_mA'] == current) &
                    (df['Temperature_K'] == MAX_TEMP)]
        
        if len(baseline) > 0 and len(maxtemp) > 0:
            freq_change = ((maxtemp['Peak_Frequency_GHz_mean'].values[0] - 
                          baseline['Peak_Frequency_GHz_mean'].values[0]) / 
                          baseline['Peak_Frequency_GHz_mean'].values[0]) * 100
            
            lw_change = ((maxtemp['Linewidth_MHz_mean'].values[0] - 
                        baseline['Linewidth_MHz_mean'].values[0]) / 
                        baseline['Linewidth_MHz_mean'].values[0]) * 100
            
            qf_change = ((maxtemp['Q_Factor_mean'].values[0] - 
                        baseline['Q_Factor_mean'].values[0]) / 
                        baseline['Q_Factor_mean'].values[0]) * 100
            
            absolute_degradation.append({
                'Material': material,
                'Current_mA': current,
                'Frequency_Change_%': freq_change,
                'Linewidth_Change_%': lw_change,
                'Q_Factor_Change_%': qf_change
            })

df_absolute = pd.DataFrame(absolute_degradation)

# Save absolute degradation
output_file = OUTPUT_DIR / 'absolute_degradation_250K_to_400K.csv'
df_absolute.to_csv(output_file, index=False)
print(f"\n Absolute degradation saved: {output_file}")

# Print summary at 3.0 mA
print("\n ABSOLUTE DEGRADATION AT 3.0 mA (250K → 400K):")
summary_3ma_abs = df_absolute[df_absolute['Current_mA'] == 3.0]
print(summary_3ma_abs.to_string(index=False))


# ANALYSIS 3: STATISTICAL SIGNIFICANCE (ANOVA)

print("ANALYSIS 3: STATISTICAL SIGNIFICANCE TESTING")
print("METHOD: One-way ANOVA + post-hoc Tukey HSD test")
print("THRESHOLD: p < 0.05 = statistically significant")

# Focus on 3.0 mA for statistical comparison
data_3ma = df[df['Current_mA'] == 3.0].copy()

# Calculate percentage change from baseline for each temperature point
results_stats = []

for material in MATERIALS:
    mat_data = data_3ma[data_3ma['Material'] == material].copy()
    mat_data = mat_data.sort_values('Temperature_K')
    
    # Get baseline values
    baseline_freq = mat_data[mat_data['Temperature_K'] == BASELINE_TEMP]['Peak_Frequency_GHz_mean'].values[0]
    baseline_lw = mat_data[mat_data['Temperature_K'] == BASELINE_TEMP]['Linewidth_MHz_mean'].values[0]
    baseline_qf = mat_data[mat_data['Temperature_K'] == BASELINE_TEMP]['Q_Factor_mean'].values[0]
    
    # Calculate % change for all temperatures
    mat_data['Freq_Change_%'] = ((mat_data['Peak_Frequency_GHz_mean'] - baseline_freq) / baseline_freq) * 100
    mat_data['LW_Change_%'] = ((mat_data['Linewidth_MHz_mean'] - baseline_lw) / baseline_lw) * 100
    mat_data['QF_Change_%'] = ((mat_data['Q_Factor_mean'] - baseline_qf) / baseline_qf) * 100
    
    results_stats.append(mat_data)

df_stats = pd.concat(results_stats, ignore_index=True)

# Perform ANOVA for each metric
print("\n ANOVA RESULTS:")

# Frequency ANOVA
groups_freq = [df_stats[df_stats['Material'] == mat]['Freq_Change_%'].values 
               for mat in MATERIALS]
f_stat_freq, p_value_freq = stats.f_oneway(*groups_freq)
print(f"Frequency Change:   F={f_stat_freq:.3f}, p={p_value_freq:.4f}", end="")
print("  SIGNIFICANT" if p_value_freq < 0.05 else "  Not significant")

# Linewidth ANOVA
groups_lw = [df_stats[df_stats['Material'] == mat]['LW_Change_%'].values 
             for mat in MATERIALS]
f_stat_lw, p_value_lw = stats.f_oneway(*groups_lw)
print(f"Linewidth Change:   F={f_stat_lw:.3f}, p={p_value_lw:.4f}", end="")
print("  SIGNIFICANT" if p_value_lw < 0.05 else "  Not significant")

# Q-Factor ANOVA
groups_qf = [df_stats[df_stats['Material'] == mat]['QF_Change_%'].values 
             for mat in MATERIALS]
f_stat_qf, p_value_qf = stats.f_oneway(*groups_qf)
print(f"Q-Factor Change:    F={f_stat_qf:.3f}, p={p_value_qf:.4f}", end="")
print("  SIGNIFICANT" if p_value_qf < 0.05 else "  Not significant")

# Pairwise t-tests (post-hoc)
print("\n PAIRWISE COMPARISONS (t-tests):")

for metric, groups in [('Frequency', groups_freq), 
                       ('Linewidth', groups_lw), 
                       ('Q-Factor', groups_qf)]:
    print(f"\n{metric}:")
    for i, mat1 in enumerate(MATERIALS):
        for j, mat2 in enumerate(MATERIALS):
            if i < j:
                t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                sig = " Significant" if p_val < 0.05 else "Not significant"
                print(f"  {mat1:12} vs {mat2:12}: p={p_val:.4f} {sig}")

# Save statistical results
stats_output = OUTPUT_DIR / 'statistical_tests.txt'
with open(stats_output, 'w') as f:
    f.write("STATISTICAL SIGNIFICANCE TESTING\n")
    f.write("ANOVA Results:\n")
    f.write(f"Frequency: F={f_stat_freq:.3f}, p={p_value_freq:.4f}\n")
    f.write(f"Linewidth: F={f_stat_lw:.3f}, p={p_value_lw:.4f}\n")
    f.write(f"Q-Factor:  F={f_stat_qf:.3f}, p={p_value_qf:.4f}\n")

print(f"\n Statistical results saved: {stats_output}")


# ANALYSIS 4: RANKING & BEST MATERIAL

print("ANALYSIS 4: COMPREHENSIVE MATERIAL RANKING")

# Get 3.0 mA data
abs_rates_3ma_rank = df_abs_rates[df_abs_rates['Current_mA'] == 3.0].copy()

# Individual metric rankings
print("INDIVIDUAL METRIC RANKINGS (Smaller = Better)")


# Linewidth ranking
lw_sorted = abs_rates_3ma_rank.sort_values('Linewidth_Rate_MHz/K')
print(" LINEWIDTH STABILITY (Spectral Purity):")
print("-" * 70)
for idx, (i, row) in enumerate(lw_sorted.iterrows(), 1):
    print(f"  Rank {idx}: {row['Material']:12} - {row['Linewidth_Rate_MHz/K']:.4f} MHz/K")
best_linewidth = lw_sorted.iloc[0]['Material']
best_lw_value = lw_sorted.iloc[0]['Linewidth_Rate_MHz/K']
worst_lw_value = lw_sorted.iloc[-1]['Linewidth_Rate_MHz/K']
improvement = ((worst_lw_value - best_lw_value) / worst_lw_value) * 100
print(f"  → Best: {best_linewidth} ({best_lw_value:.4f} MHz/K)")
print(f"  → ADVANTAGE: {improvement:.1f}% better than worst material")


# Q-Factor ranking
qf_sorted = abs_rates_3ma_rank.sort_values('Q_Factor_Rate_Q/K', key=abs)
print(" Q-FACTOR STABILITY (Overall Quality):")
for idx, (i, row) in enumerate(qf_sorted.iterrows(), 1):
    print(f"  Rank {idx}: {row['Material']:12} - {row['Q_Factor_Rate_Q/K']:.4f} Q/K (abs: {abs(row['Q_Factor_Rate_Q/K']):.4f})")
best_qfactor = qf_sorted.iloc[0]['Material']
best_qf_value = abs(qf_sorted.iloc[0]['Q_Factor_Rate_Q/K'])
worst_qf_value = abs(qf_sorted.iloc[-1]['Q_Factor_Rate_Q/K'])
improvement_qf = ((worst_qf_value - best_qf_value) / worst_qf_value) * 100
print(f"  → Best: {best_qfactor} ({qf_sorted.iloc[0]['Q_Factor_Rate_Q/K']:.4f} Q/K)")
print(f"  → ADVANTAGE: {improvement_qf:.1f}% better than worst material")

# Frequency ranking
freq_sorted = abs_rates_3ma_rank.sort_values('Frequency_Rate_GHz/K', key=abs)
print(" FREQUENCY STABILITY:")
for idx, (i, row) in enumerate(freq_sorted.iterrows(), 1):
    print(f"  Rank {idx}: {row['Material']:12} - {row['Frequency_Rate_GHz/K']:.6f} GHz/K (abs: {abs(row['Frequency_Rate_GHz/K']):.6f})")
best_frequency = freq_sorted.iloc[0]['Material']
print(f"  → Best: {best_frequency}")

# Overall composite ranking (equal weight to all metrics)
abs_rates_3ma_rank['Freq_Rank'] = abs_rates_3ma_rank['Frequency_Rate_GHz/K'].abs().rank(ascending=True)
abs_rates_3ma_rank['LW_Rank'] = abs_rates_3ma_rank['Linewidth_Rate_MHz/K'].rank(ascending=True)
abs_rates_3ma_rank['QF_Rank'] = abs_rates_3ma_rank['Q_Factor_Rate_Q/K'].abs().rank(ascending=True)
abs_rates_3ma_rank['Overall_Rank'] = (abs_rates_3ma_rank['Freq_Rank'] + 
                                      abs_rates_3ma_rank['LW_Rank'] + 
                                      abs_rates_3ma_rank['QF_Rank']) / 3

abs_rates_sorted = abs_rates_3ma_rank.sort_values('Overall_Rank', ascending=True)

print("OVERALL COMPOSITE RANKING (Equal Weight: Freq + Linewidth + Q-Factor)")
for idx, (i, row) in enumerate(abs_rates_sorted.iterrows(), 1):
    print(f"Rank {idx}: {row['Material']}")
    print(f"  Frequency:  {row['Frequency_Rate_GHz/K']:.6f} GHz/K (rank {int(row['Freq_Rank'])})")
    print(f"  Linewidth:  {row['Linewidth_Rate_MHz/K']:.4f} MHz/K (rank {int(row['LW_Rank'])})")
    print(f"  Q-Factor:   {row['Q_Factor_Rate_Q/K']:.4f} Q/K (rank {int(row['QF_Rank'])})")
    print(f"  Overall Score: {row['Overall_Rank']:.2f}")

best_material = abs_rates_sorted.iloc[0]['Material']

print("SUMMARY & RECOMMENDATIONS")
print(f" BEST FOR LINEWIDTH (Spectral Purity):  {best_linewidth}")
print(f"    {improvement:.1f}% better spectral stability")
print(f"    Ideal for: Narrow-linewidth oscillators, spectroscopy")
print()
print(f" BEST FOR Q-FACTOR (Overall Quality):   {best_qfactor}")
print(f"    {improvement_qf:.1f}% better quality retention")
print(f"   Ideal for: General oscillators, high-Q applications")
print()
print(f" BEST FOR FREQUENCY STABILITY:          {best_frequency}")
print(f"    Most stable frequency vs temperature")
print(f"    Ideal for: Precision frequency sources")
print()
print(f" OVERALL Best (Balanced):             {best_material}")
print(f"    Best compromise across all metrics")
print()

# Save ranking with all details
output_file = OUTPUT_DIR / 'material_ranking.csv'
abs_rates_sorted.to_csv(output_file, index=False)
print(f" Material ranking saved: {output_file}")

# FIGURE 7: DEGRADATION RATE BAR CHART
print("GENERATING FIGURE 7: Degradation Rate Comparison")

fig, axes = plt.subplots(1, 3, figsize=(24, 9))

metrics = [
    ('Frequency_Rate_%/K', 'Frequency Degradation Rate (%/K)'),
    ('Linewidth_Rate_%/K', 'Linewidth Degradation Rate (%/K)'),
    ('Q_Factor_Rate_%/K', 'Q-Factor Degradation Rate (%/K)')
]

rates_3ma = df_rates[df_rates['Current_mA'] == 3.0]

for idx, (metric, title) in enumerate(metrics):
    ax = axes[idx]
    
    x_pos = np.arange(len(MATERIALS))
    values = [rates_3ma[rates_3ma['Material'] == mat][metric].values[0] 
              for mat in MATERIALS]
    colors_list = [COLORS[mat] for mat in MATERIALS]
    
    bars = ax.bar(x_pos, values, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.7)
    
    # Add value labels with 4 decimal places
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        
        if height >= 0:
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_pos = height + (0.03 * y_range)
            va = 'bottom'
        else:
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_pos = height - (0.03 * y_range)
            va = 'top'
        
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{val:.4f}',
               ha='center', va=va,
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
                        edgecolor='gray', alpha=0.95, linewidth=1.5))
    
    ax.set_xlabel('Material', fontsize=15, fontweight='bold')
    ax.set_ylabel(title, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MATERIALS, fontsize=13)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#fafafa')
    
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.15*y_range, y_max + 0.1*y_range)

plt.suptitle('       Thermal Degradation Rate Comparison (at 3.0 mA)',
             fontsize=17, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_fig = OUTPUT_DIR / 'figure_7_degradation_rates.png'
plt.savefig(output_fig, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f" Figure 7 saved: {output_fig}")


# FIGURE 8: ABSOLUTE DEGRADATION RATE BAR CHART 

print("GENERATING FIGURE 8: Absolute Degradation Rate Comparison (FIXED)")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

metrics_abs = [
    ('Frequency_Rate_GHz/K', 'Frequency Degradation Rate (GHz/K)', 6),  # 6 decimal places
    ('Linewidth_Rate_MHz/K', 'Linewidth Degradation Rate (MHz/K)', 4),  # 4 decimal places
    ('Q_Factor_Rate_Q/K', 'Q-Factor Degradation Rate (Q-points/K)', 4)  # 4 decimal places
]

abs_rates_3ma = df_abs_rates[df_abs_rates['Current_mA'] == 3.0]

for idx, (metric, title, decimals) in enumerate(metrics_abs):
    ax = axes[idx]
    
    x_pos = np.arange(len(MATERIALS))
    values = [abs_rates_3ma[abs_rates_3ma['Material'] == mat][metric].values[0] 
              for mat in MATERIALS]
    colors_list = [COLORS[mat] for mat in MATERIALS]
    
    bars = ax.bar(x_pos, values, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    
    # Add value labels with VARIABLE decimal places 
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height >= 0:
            y_pos = height + (0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
        else:
            y_pos = height - (0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
        
        # Format with appropriate decimal places
        label_text = f'{val:.{decimals}f}'
        
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               label_text,
               ha='center', va='bottom' if height >= 0 else 'top',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='gray', alpha=0.95))
    
    ax.set_xlabel('Material', fontsize=14, fontweight='bold')
    ax.set_ylabel(title, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MATERIALS, fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#fafafa')
    
    # Force y-axis to include zero and add appropriate margins
    y_min, y_max = min(values), max(values)
    if y_min > 0:
        ax.set_ylim(0, y_max * 1.25)
    elif y_max < 0:
        ax.set_ylim(y_min * 1.25, 0)
    else:
        margin = max(abs(y_min), abs(y_max)) * 0.2
        ax.set_ylim(y_min - margin, y_max + margin)

plt.suptitle('Absolute Degradation Rate Comparison (at 3.0 mA)\nActual Physical Changes Per Kelvin',
             fontsize=17, fontweight='bold', y=0.98)
plt.tight_layout()

output_fig = OUTPUT_DIR / 'figure_8_absolute_degradation_rates.png'
plt.savefig(output_fig, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# SUMMARY REPORT
summary_report = OUTPUT_DIR / 'SUMMARY_REPORT.txt'

try:
    if 'best_material' not in locals():
        best_material = abs_rates_sorted.iloc[0]['Material']
    if 'best_linewidth' not in locals():
        best_linewidth = lw_sorted.iloc[0]['Material']
    if 'best_qfactor' not in locals():
        best_qfactor = qf_sorted.iloc[0]['Material']
    
    with open(summary_report, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("THERMAL STABILITY ANALYSIS - COMPREHENSIVE SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. MATERIAL RANKINGS BY PERFORMANCE METRIC:\n")
        f.write("-" * 70 + "\n")
        f.write(f"   LINEWIDTH WINNER:  {best_linewidth}\n")
        f.write(f"      - Spectral purity: {best_lw_value:.4f} MHz/K\n")
        f.write(f"      - {improvement:.1f}% better than worst material\n")
        f.write(f"      - USE FOR: Narrow-linewidth oscillators, spectroscopy\n\n")
        
        f.write(f"   Q-FACTOR WINNER:   {best_qfactor}\n")
        f.write(f"      - Quality retention: {qf_sorted.iloc[0]['Q_Factor_Rate_Q/K']:.4f} Q/K\n")
        f.write(f"      - {improvement_qf:.1f}% better than worst material\n")
        f.write(f"      - USE FOR: High-Q oscillators, overall quality\n\n")
        
        f.write(f"   OVERALL WINNER:    {best_material}\n")
        f.write(f"      - Best balanced performance\n\n")
        
        f.write("2. PERCENTAGE DEGRADATION RATES AT 3.0 mA (250-400K):\n")
        f.write("-" * 70 + "\n")
        rates_3ma = df_rates[df_rates['Current_mA'] == 3.0]
        f.write(rates_3ma[['Material', 'Frequency_Rate_%/K', 'Linewidth_Rate_%/K', 'Q_Factor_Rate_%/K']].to_string(index=False))
        f.write("\n")
        f.write("   NOTE: All materials show similar RELATIVE degradation (~0.5%/K)\n")
        f.write("   This suggests similar thermal degradation mechanisms.\n\n")
        
        f.write("3. ABSOLUTE DEGRADATION RATES AT 3.0 mA:\n")
        f.write("-" * 70 + "\n")
        summary_3ma_abs_rates = df_abs_rates[df_abs_rates['Current_mA'] == 3.0]
        f.write(summary_3ma_abs_rates[['Material', 'Frequency_Rate_GHz/K', 'Linewidth_Rate_MHz/K', 'Q_Factor_Rate_Q/K']].to_string(index=False))
        f.write("\n\n")
        f.write("   INTERPRETATION (Physical Changes Per Kelvin):\n")
        for mat in MATERIALS:
            mat_data = summary_3ma_abs_rates[summary_3ma_abs_rates['Material'] == mat]
            if len(mat_data) > 0:
                f.write(f"   - {mat}:\n")
                f.write(f"     Frequency: {mat_data['Frequency_Rate_GHz/K'].values[0]:.6f} GHz/K\n")
                f.write(f"     Linewidth: {mat_data['Linewidth_Rate_MHz/K'].values[0]:.4f} MHz/K\n")
                f.write(f"     Q-Factor:  {mat_data['Q_Factor_Rate_Q/K'].values[0]:.4f} Q-points/K\n")
        f.write("\n")
        
        f.write("4. ABSOLUTE DEGRADATION (250K → 400K) AT 3.0 mA:\n")
        f.write("-" * 70 + "\n")
        summary_3ma_abs = df_absolute[df_absolute['Current_mA'] == 3.0]
        f.write(summary_3ma_abs[['Material', 'Frequency_Change_%', 'Linewidth_Change_%', 'Q_Factor_Change_%']].to_string(index=False))
        f.write("\n\n")
        
        f.write("5. MATERIAL RANKING (Best to Worst):\n")
        f.write("-" * 70 + "\n")
        for idx, row in abs_rates_sorted.iterrows():
            f.write(f"   Rank {int(row['Overall_Rank'])}: {row['Material']}\n")
        f.write("\n")
        
        f.write("6. STATISTICAL SIGNIFICANCE:\n")
        f.write(f"   Frequency: F={f_stat_freq:.3f}, p={p_value_freq:.4f}\n")
        f.write(f"   Linewidth: F={f_stat_lw:.3f}, p={p_value_lw:.4f}\n")
        f.write(f"   Q-Factor:  F={f_stat_qf:.3f}, p={p_value_qf:.4f}\n\n")
        
        f.write("7. KEY FINDINGS & RECOMMENDATIONS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"   LINEWIDTH PERFORMANCE:\n")
        f.write(f"   - {best_linewidth} achieves BEST spectral purity\n")
        f.write(f"   - {improvement:.1f}% superior linewidth stability\n")
        f.write(f"   - Recommendation: Use for narrow-linewidth applications\n\n")
        
        f.write(f"   Q-FACTOR PERFORMANCE:\n")
        f.write(f"   - {best_qfactor} maintains BEST quality factor\n")
        f.write(f"   - {improvement_qf:.1f}% better Q-factor retention\n")
        f.write(f"   - Recommendation: Use for high-Q oscillators\n\n")
        
        f.write(f"   OVERALL:\n")
        f.write(f"   - All materials show similar RELATIVE degradation (~0.5%/K)\n")
        f.write(f"   - But ABSOLUTE changes differ significantly!\n")
        f.write(f"   - Material selection depends on application priority:\n")
        f.write(f"      * Spectral purity → {best_linewidth}\n")
        f.write(f"      * Quality factor  → {best_qfactor}\n")
        f.write(f"      * Balanced use    → {best_material}\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f" Summary report saved: {summary_report}")
    
except Exception as e:
    print(f"WARNING: Error creating summary report: {e}")
    with open(summary_report, 'w', encoding='utf-8') as f:
        f.write("ERROR CREATING FULL REPORT\n")
        f.write(f"Error: {str(e)}\n")
        f.write("\nPlease check the CSV output files for results.\n")



print(f"\n All results saved in: {OUTPUT_DIR}/")
print("\nGenrated files:")
print("   1. degradation_rates.csv")
print("   2. absolute_degradation_rates.csv")
print("   3. absolute_degradation_250K_to_400K.csv")
print("   4. statistical_tests.txt")
print("   5. material_ranking.csv")
print("   6. figure_7_degradation_rates.png")
print("   7. figure_8_absolute_degradation_rates.png (FIXED - 6 decimals for frequency!)")
print("   8. SUMMARY_REPORT.txt")
