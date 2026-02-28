import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("CREATING 6 PLOTS")

# FILE PATHS - AUTOMATICALLY USES SCRIPT'S DIRECTORY
BASE_DIR = Path(__file__).parent  # Gets the directory where the script is
OUTPUT_DIR = BASE_DIR / "Result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(OUTPUT_DIR / "statistics_FIXED_THERMAL.csv")
print(f"\n Loaded {len(df)} data points")


# CONFIGURATION
COLORS = {
    'CoFeB/MgO': '#2E86AB',  # Blue
    'Co/Ni': '#A23B72',      # Purple
    'Co/Pt': '#F18F01'       # Orange
}

MATERIALS = ['CoFeB/MgO', 'Co/Ni', 'Co/Pt']
TEMPS = [250, 275, 300, 325, 350, 375, 400]
CURRENTS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


# PLOT 1: FREQUENCY - Line Plot with Error Bars
print("\n Creating Plot 1: Frequency (Line + Error Bars)")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, current in enumerate(CURRENTS):
    ax = axes[idx]
    
    for material in MATERIALS:
        data = df[(df['Material'] == material) & (df['Current_mA'] == current)]
        
        if len(data) > 0:
            x = data['Temperature_K'].values
            y = data['Peak_Frequency_GHz_mean'].values
            yerr = data['Peak_Frequency_GHz_std'].values
            
            # Main line
            ax.plot(x, y, marker='o', linewidth=3, markersize=12,
                   color=COLORS[material], label=material,
                   markerfacecolor=COLORS[material],
                   markeredgecolor='white', markeredgewidth=2)
            
            # Error bars
            ax.errorbar(x, y, yerr=yerr, fmt='none',
                       ecolor=COLORS[material], elinewidth=1.5,
                       capsize=4, capthick=1.5, alpha=0.5)
            
            # Labels CLOSE to points
            for xi, yi in zip(x, y):
                ax.text(xi, yi + 0.04, f'{yi:.2f}',
                       ha='center', va='bottom', fontsize=9,
                       color=COLORS[material], fontweight='bold')
    
    ax.set_xlabel('Temperature (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency (GHz)', fontsize=13, fontweight='bold')
    ax.set_title(f'I = {current:.1f} mA', fontsize=15, fontweight='bold', pad=10)
    
    # Fix: Move legend to upper left (away from center at 4.5mA)
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_xlim(240, 410)
    
    # Extra space at top for labels - INCREASED for safety
    y_data = []
    for mat in MATERIALS:
        data_temp = df[(df['Material'] == mat) & (df['Current_mA'] == current)]
        if len(data_temp) > 0:
            y_data.extend(data_temp['Peak_Frequency_GHz_mean'].values)
    if y_data:
        y_min = min(y_data) - 0.2
        y_max = max(y_data) + 0.45  # INCREASED from 0.35 to 0.45
        ax.set_ylim(y_min, y_max)
    
    ax.set_facecolor('#fafafa')

plt.suptitle('Oscillation Frequency vs Temperature\n(Error Bars Show Statistical Uncertainty)',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'v1_frequency_line_errorbar.png',
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    Saved: v1_frequency_line_errorbar.png")


# PLOT 2: LINEWIDTH - Gradient Area Chart

print("\n Creating Plot 2: Linewidth (Gradient Area Chart)")

# BIGGER figure size to prevent overlap
fig, axes = plt.subplots(2, 3, figsize=(26, 16))  # Increased further to prevent label overlap
axes = axes.flatten()

for idx, current in enumerate(CURRENTS):
    ax = axes[idx]
    
    # Create offset positions for each material to prevent overlap
    for mat_idx, material in enumerate(MATERIALS):
        data = df[(df['Material'] == material) & (df['Current_mA'] == current)]
        
        if len(data) > 0:
            # Apply LARGER horizontal offset - INCREASED to 15K for better separation
            x_offset = (mat_idx - 1) * 15.0 
            x = data['Temperature_K'].values + x_offset
            y = data['Linewidth_MHz_mean'].values
            
            # Gradient fill
            ax.fill_between(x, 0, y, color=COLORS[material], alpha=0.3)
            
            # Main line
            ax.plot(x, y, linewidth=4, color=COLORS[material],
                   label=material, marker='o', markersize=12,
                   markerfacecolor=COLORS[material],
                   markeredgecolor='white', markeredgewidth=2.5)
            
            # Value labels - positioned based on material, colored by material
            for xi, yi in zip(x, y):
                # Position based on material to prevent overlap - smaller offsets
                if material == 'CoFeB/MgO':
                    offset = 0.8  # Small offset above
                    va_pos = 'bottom'
                elif material == 'Co/Ni':
                    offset = 1.2  # Medium offset above
                    va_pos = 'bottom'
                else:  # Co/Pt
                    offset = -2.5  # Below the line
                    va_pos = 'top'
                
                ax.text(xi, yi + offset, f'{yi:.1f}',
                       ha='center', va=va_pos, fontsize=10,
                       color=COLORS[material],  # Use material color!
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4',
                               facecolor='white',
                               edgecolor='none',
                               alpha=0.8))
    
    ax.set_xlabel('Temperature (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Linewidth (MHz)', fontsize=14, fontweight='bold')
    ax.set_title(f'I = {current:.1f} mA', fontsize=16, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # DOUBLED x-axis range: 200 to 450 (from 232-422)
    ax.set_xlim(200, 450)
    
    ax.set_yticks(np.arange(0, 60, 20))  # 0, 20, 40
    
    # Fix: More space at top, limit to 55 max
    y_data_all = []
    for mat in MATERIALS:
        data_temp = df[(df['Material'] == mat) & (df['Current_mA'] == current)]
        if len(data_temp) > 0:
            y_data_all.extend(data_temp['Linewidth_MHz_mean'].values)
    
    if y_data_all:
        max_val = max(y_data_all)
        # Top 3 graphs (2.0, 2.5, 3.0 mA): 0 to 80
        # Bottom 3 graphs (3.5, 4.0, 4.5 mA): 0 to 75
        if current <= 3.0:
            ax.set_ylim(0, 80)
        else:
            ax.set_ylim(0, 75)
    else:
        if current <= 3.0:
            ax.set_ylim(0, 80)
        else:
            ax.set_ylim(0, 75)
    
    ax.set_facecolor('#fafafa')

plt.suptitle('Linewidth Thermal Broadening Analysis\n(Gradient Area Shows Broadening Intensity)',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'v2_linewidth_gradient_area.png',
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    Saved: v2_linewidth_gradient_area.png")


# PLOT 3: Q-FACTOR - Lollipop Chart
print("\n Creating Plot 3: Q-Factor (Lollipop Chart)")

fig, axes = plt.subplots(2, 3, figsize=(26, 16))  # Increased to prevent label overlap
axes = axes.flatten()

for idx, current in enumerate(CURRENTS):
    ax = axes[idx]
    
    # Create spacing for lollipops
    x_positions = {}
    base_x = np.array(TEMPS)
    
    for i, material in enumerate(MATERIALS):
        offset = (i - 1) * 8
        x_positions[material] = base_x + offset
    
    for material in MATERIALS:
        data = df[(df['Material'] == material) & (df['Current_mA'] == current)]
        
        if len(data) > 0:
            x = x_positions[material]
            y = data['Q_Factor_mean'].values
            
            # Stems
            for xi, yi in zip(x, y):
                ax.plot([xi, xi], [0, yi], linewidth=3,
                       color=COLORS[material], alpha=0.7, zorder=1)
            
            # Heads
            ax.scatter(x, y, s=250, color=COLORS[material],
                      edgecolor='white', linewidth=2.5,
                      label=material, zorder=3)
            
            # Labels
            for xi, yi in zip(x, y):
                ax.text(xi, yi + 8, f'{yi:.1f}',
                       ha='center', va='bottom', fontsize=9,
                       color='black', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor='none',
                               alpha=0.85))
    
    ax.set_xlabel('Temperature (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Q-Factor', fontsize=13, fontweight='bold')
    ax.set_title(f'I = {current:.1f} mA', fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks(base_x)
    ax.set_xticklabels(TEMPS, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    ax.set_xlim(235, 415)
    
    y_data_all = []
    for mat in MATERIALS:
        data_temp = df[(df['Material'] == mat) & (df['Current_mA'] == current)]
        if len(data_temp) > 0:
            y_data_all.extend(data_temp['Q_Factor_mean'].values)
    
    if y_data_all:
        max_val = max(y_data_all)
        ax.set_ylim(0, max_val * 1.20)  # 20% extra space for labels
    else:
        ax.set_ylim(0, None)
    
    ax.set_facecolor('#fafafa')

plt.suptitle('Q-Factor Degradation Analysis\n(Quality Factor Loss with Temperature)',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'v3_qfactor_lollipop.png',
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    Saved: v3_qfactor_lollipop.png")


# PLOT 4: HEATMAP - 2D Grid (All 3 Metrics)

print("\n Creating Plot 4: Heatmap")

fig, axes = plt.subplots(3, 3, figsize=(22, 18))

metrics = [
    ('Peak_Frequency_GHz_mean', 'Frequency (GHz)', 'YlGnBu'),
    ('Linewidth_MHz_mean', 'Linewidth (MHz)', 'YlOrRd'),
    ('Q_Factor_mean', 'Q-Factor', 'RdYlGn')
]

for col_idx, (metric, metric_name, cmap) in enumerate(metrics):
    for row_idx, material in enumerate(MATERIALS):
        ax = axes[row_idx, col_idx]
        
        # Create pivot table
        data_subset = df[df['Material'] == material]
        pivot = data_subset.pivot_table(
            values=metric,
            index='Temperature_K',
            columns='Current_mA'
        )
        
        # Create heatmap
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                      interpolation='nearest')
        
        # Set ticks
        ax.set_xticks(np.arange(len(CURRENTS)))
        ax.set_yticks(np.arange(len(TEMPS)))
        ax.set_xticklabels([f'{c:.1f}' for c in CURRENTS], fontsize=11)
        ax.set_yticklabels(TEMPS, fontsize=11)
        
        # Add value annotations
        for i in range(len(TEMPS)):
            for j in range(len(CURRENTS)):
                value = pivot.values[i, j]
                ax.text(j, i, f'{value:.1f}',
                       ha='center', va='center', fontsize=10,
                       color='black', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor='gray',
                               linewidth=0.5,
                               alpha=0.7))
        
        ax.set_title(f'{material}\n{metric_name}',
                    fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel('Current (mA)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric_name, fontsize=10, fontweight='bold')

plt.suptitle('Comprehensive Performance Heatmap\n(Red/Yellow = Problem Areas, Blue/Green = Optimal)',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'v4_heatmap_2d_grid.png',
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    Saved: v4_heatmap.png")


# PLOT 5: MATERIAL COMPARISON - Grouped Bars (at 3.0 mA)
print("\n Creating Plot 5: Material Comparison (Grouped Bars)")

# Taller figure to accommodate bottom legend
fig, axes = plt.subplots(1, 3, figsize=(22, 9))  

current_fixed = 3.0
metrics_plot = [
    ('Peak_Frequency_GHz_mean', 'Frequency (GHz)'),
    ('Linewidth_MHz_mean', 'Linewidth (MHz)'),
    ('Q_Factor_mean', 'Q-Factor')
]

for idx, (metric, ylabel) in enumerate(metrics_plot):
    ax = axes[idx]
    
    x_pos = np.arange(len(TEMPS))
    width = 0.25
    
    for i, material in enumerate(MATERIALS):
        data = df[(df['Material'] == material) & (df['Current_mA'] == current_fixed)]
        values = data[metric].values
        
        # Bars
        bars = ax.bar(x_pos + i * width, values, width,
                     label=material, color=COLORS[material],
                     alpha=0.85, edgecolor='black', linewidth=1.5)
        
        # Value labels on top
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10,
                   color='black', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white',
                           edgecolor='none',
                           alpha=0.85))
    
    ax.set_xlabel('Temperature (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(f'{ylabel} at I = {current_fixed} mA',
                fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(TEMPS, fontsize=12)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
             ncol=3, fontsize=12, frameon=True, shadow=True)
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_facecolor('#fafafa')

plt.suptitle(f'Direct Material Comparison at I = {current_fixed} mA',
             fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'v5_material_comparison_bars.png',
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    Saved: v5_material_comparison_bars.png")


# PLOT 6: DEGRADATION - Percentage Area (from 250K baseline)
print("\n Creating Plot 6: Thermal Degradation (Percentage Change)")

fig, axes = plt.subplots(1, 3, figsize=(22, 10))  # Increased from 9 to 10

metrics_deg = [
    ('Peak_Frequency_GHz_mean', 'Frequency Change (%)'),
    ('Linewidth_MHz_mean', 'Linewidth Increase (%)'),
    ('Q_Factor_mean', 'Q-Factor Reduction (%)')
]

for idx, (metric, ylabel) in enumerate(metrics_deg):
    ax = axes[idx]
    
    # Plot ALL materials - with horizontal offset for separation
    for mat_idx, material in enumerate(MATERIALS):
        data = df[(df['Material'] == material) & (df['Current_mA'] == 3.0)]
        
        if len(data) > 0:
            # DIFFERENT horizontal offsets for each graph type
            if idx == 0:  # Frequency - LARGER offset 
                x_offset = (mat_idx - 1) * 30.0  # 30K spacing for better separation
            else:  # Linewidth and Q-Factor - keep 24K
                x_offset = (mat_idx - 1) * 24.0  # 24K spacing
            
            baseline = data[data['Temperature_K'] == 250][metric].values[0]
            temps = data['Temperature_K'].values + x_offset  # With offset!
            values = data[metric].values
            
            # Calculate percentage change
            pct_change = ((values - baseline) / baseline) * 100
            
            # Area fill
            ax.fill_between(temps, 0, pct_change,
                           color=COLORS[material], alpha=0.25)
            
            # Line with markers - THINNER LINE (changed from 4 to 2.5)
            ax.plot(temps, pct_change, marker='D', linewidth=2.5, markersize=14,
                   color=COLORS[material], label=material,
                   markerfacecolor=COLORS[material],
                   markeredgecolor='white', markeredgewidth=2.5,
                   zorder=3)
            
            for i, (temp, pct) in enumerate(zip(temps, pct_change)):

                if material == 'Co/Pt':  # 3rd material 
                    if idx == 0:  # Frequency - position 
                        if abs(pct) < 0.01:  
                            offset = -0.03
                            ha_pos = 'left'
                        else:
                            offset = -0.03
                            ha_pos = 'right'
                    elif idx == 1:  # Linewidth
                        offset = 2.5  
                        ha_pos = 'center'
                    else:  # Q-Factor
                        
                        if abs(pct) < 0.01:
                            offset = -2.0  
                            va_pos = 'top'
                        else:
                            offset = 1.5
                            va_pos = 'bottom'
                        ha_pos = 'center'
                    va_pos = 'top' if idx == 0 else ('bottom' if idx != 2 or abs(pct) >= 0.01 else 'top')
                    
                elif material == 'Co/Ni': 
                    if idx == 0:  
                        if abs(pct) < 0.01:  
                            offset = -0.03
                            ha_pos = 'left'
                            va_pos = 'top'
                        else:
                            offset = 0.03  
                            ha_pos = 'center'
                            va_pos = 'bottom'  
                    elif idx == 1:  # Linewidth
                     
                        if abs(pct) < 0.01:
                            offset = 2.2  
                        else:
                            offset = 1.5
                        ha_pos = 'center'
                        va_pos = 'bottom'
                    else:  # Q-Factor
                      
                        if abs(pct) < 0.01:
                            offset = -1.8  
                            va_pos = 'top'
                        else:
                            offset = 1.2
                            va_pos = 'bottom'
                        ha_pos = 'center'
                    if idx != 0 and idx != 1:
                        va_pos = 'bottom' if abs(pct) >= 0.01 else 'top'
                    if material == 'Co/Pt':  # 3rd material 
                    if idx == 0:  # Frequency 
                        if abs(pct) < 0.01:  # This is 0.0%
                            offset = -0.03
                            ha_pos = 'left'
                        else:
                            offset = -0.03
                            ha_pos = 'right'
                    elif idx == 1:  # Linewidth
                        if abs(pct) < 0.01:  
                            offset = 2.2  
                            va_pos = 'bottom'
                        else:
                            offset = -1.8
                            va_pos = 'top'
                        ha_pos = 'center'
                    else:  # Q-Factor
                        if abs(pct) < 0.01:
                            offset = -2.0  
                            va_pos = 'top'
                        else:
                            offset = -1.5
                            va_pos = 'top'
                        ha_pos = 'center'
                    if idx != 1:
                        va_pos = 'top' if idx == 0 or idx == 2 else 'bottom'
                
                ax.text(temp, pct + offset, f'{pct:.1f}%',
                       ha=ha_pos, va=va_pos, fontsize=9,
                       color=COLORS[material],  
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor='none',
                               alpha=0.85))
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=1)
    
    ax.set_xlabel('Temperature (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(ylabel, fontsize=15, fontweight='bold', pad=10)
    
    # Fix: Move legend to bottom center (outside plot)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
             ncol=3, fontsize=12, frameon=True, shadow=True)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # X-limits 
    ax.set_xlim(210, 430)
    
    # Auto-scale y-axis 
    all_pct = []
    for mat in MATERIALS:
        data_temp = df[(df['Material'] == mat) & (df['Current_mA'] == 3.0)]
        if len(data_temp) > 0:
            baseline_temp = data_temp[data_temp['Temperature_K'] == 250][metric].values[0]
            temps_temp = data_temp['Temperature_K'].values
            values_temp = data_temp[metric].values
            pct_change_temp = ((values_temp - baseline_temp) / baseline_temp) * 100
            all_pct.extend(pct_change_temp)
    
    if all_pct:
        # First graph (Frequency)
        if idx == 0:
            ax.set_ylim(-1.70, 0.00)  # 0.00 to -1.70 as requested
            ax.set_yticks(np.arange(-1.50, 0.01, 0.25))  # Steps of 0.25: -1.50, -1.25, -1.00, -0.75, -0.50, -0.25, 0.00
        elif idx == 1:  # Linewidth - set to 0 to 80
            ax.set_ylim(0, 80)
        else:  # Q-Factor - set to 0 to -50
            ax.set_ylim(-50, 0)

    
    ax.set_facecolor('#fafafa')

plt.suptitle('Thermal Degradation Analysis (% Change from 250K Baseline at 3.0 mA)',
             fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'v6_degradation_percentage_area.png',
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    Saved: v6_degradation_percentage_area.png")

# SUMMARY
print("ALL 6 PLOTS CREATED SUCCESSFULLY")
print("="*80)

print("   1. v1_frequency_line_errorbar.png - Frequency stability")
print("   2. v2_linewidth_gradient_area.png - Linewidth broadening")
print("   3. v3_qfactor_lollipop.png - Q-factor degradation")
print("   4. v4_heatmap_2d_grid.png - Comprehensive overview")
print("   5. v5_material_comparison_bars.png - Material comparison")
print("   6. v6_degradation_percentage_area.png - Relative impact")

print(f"\n All files saved to: {OUTPUT_DIR}")
