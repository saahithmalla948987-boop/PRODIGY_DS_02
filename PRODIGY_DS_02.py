# PRODIGY INFOTECH - DATA SCIENCE INTERNSHIP
# Task 02: Data Cleaning and Exploratory Data Analysis (EDA)
# Dataset: Titanic Dataset from Kaggle

# ── Step 1: Install required libraries ──────────────────────────────────────
# pip install pandas matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Step 2: Load Titanic Dataset ─────────────────────────────────────────────
# Using the built-in Titanic dataset from seaborn (same as Kaggle data)
df = sns.load_dataset('titanic')

print("=" * 60)
print("       TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Step 3: Data Cleaning ────────────────────────────────────────────────────
print("\n📋 STEP 1: Initial Data Overview")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n  Columns: {list(df.columns)}")

print("\n📋 STEP 2: Missing Values BEFORE Cleaning")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# --- Fill missing Age with median ---
df['age'].fillna(df['age'].median(), inplace=True)

# --- Fill missing Embarked with mode ---
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# --- Drop 'deck' column (too many missing values ~77%) ---
df.drop(columns=['deck'], inplace=True)

# --- Remove duplicates ---
df.drop_duplicates(inplace=True)

print("\n✅ STEP 3: Missing Values AFTER Cleaning")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("  All critical missing values handled!")

print("\n📊 STEP 4: Basic Statistics")
print(df[['age', 'fare', 'pclass', 'sibsp', 'parch']].describe().round(2))

# ── Step 4: EDA Visualizations ───────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.suptitle('PRODIGY DS TASK-02\nTitanic Dataset - Exploratory Data Analysis',
             fontsize=20, fontweight='bold', color='#2c3e50', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Color palette
survived_colors = ['#e74c3c', '#2ecc71']  # red=died, green=survived

# --- Plot 1: Overall Survival Count ---
ax1 = fig.add_subplot(gs[0, 0])
survival_counts = df['survived'].value_counts()
bars = ax1.bar(['Did Not Survive', 'Survived'], survival_counts.values,
               color=survived_colors, edgecolor='white', linewidth=1.5, width=0.5)
ax1.set_title('Overall Survival Count', fontweight='bold', fontsize=12)
ax1.set_ylabel('Number of Passengers')
for bar, val in zip(bars, survival_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha='center', fontweight='bold', fontsize=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
total = survival_counts.sum()
ax1.set_ylim(0, max(survival_counts.values) * 1.15)
survival_rate = survival_counts[1] / total * 100
ax1.text(0.5, 0.92, f'Survival Rate: {survival_rate:.1f}%',
         transform=ax1.transAxes, ha='center', fontsize=10,
         color='#27ae60', fontweight='bold')

# --- Plot 2: Survival by Gender ---
ax2 = fig.add_subplot(gs[0, 1])
gender_survival = df.groupby('sex')['survived'].mean() * 100
bars2 = ax2.bar(gender_survival.index, gender_survival.values,
                color=['#3498db', '#e91e8c'], edgecolor='white', linewidth=1.5, width=0.4)
ax2.set_title('Survival Rate by Gender', fontweight='bold', fontsize=12)
ax2.set_ylabel('Survival Rate (%)')
ax2.set_ylim(0, 100)
for bar, val in zip(bars2, gender_survival.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Plot 3: Survival by Passenger Class ---
ax3 = fig.add_subplot(gs[0, 2])
class_survival = df.groupby('pclass')['survived'].mean() * 100
class_colors = ['#f39c12', '#95a5a6', '#cd7f32']
bars3 = ax3.bar(['1st Class', '2nd Class', '3rd Class'], class_survival.values,
                color=class_colors, edgecolor='white', linewidth=1.5, width=0.5)
ax3.set_title('Survival Rate by Class', fontweight='bold', fontsize=12)
ax3.set_ylabel('Survival Rate (%)')
ax3.set_ylim(0, 100)
for bar, val in zip(bars3, class_survival.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# --- Plot 4: Age Distribution ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df[df['survived'] == 0]['age'].dropna(), bins=25, alpha=0.7,
         color='#e74c3c', label='Did Not Survive', edgecolor='white')
ax4.hist(df[df['survived'] == 1]['age'].dropna(), bins=25, alpha=0.7,
         color='#2ecc71', label='Survived', edgecolor='white')
ax4.set_title('Age Distribution by Survival', fontweight='bold', fontsize=12)
ax4.set_xlabel('Age')
ax4.set_ylabel('Count')
ax4.legend()
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# --- Plot 5: Fare Distribution by Class ---
ax5 = fig.add_subplot(gs[1, 1])
class_labels = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
for pclass, color in zip([1, 2, 3], ['#f39c12', '#95a5a6', '#cd7f32']):
    data = df[df['pclass'] == pclass]['fare']
    ax5.hist(data, bins=20, alpha=0.6, color=color,
             label=class_labels[pclass], edgecolor='white')
ax5.set_title('Fare Distribution by Class', fontweight='bold', fontsize=12)
ax5.set_xlabel('Fare (£)')
ax5.set_ylabel('Count')
ax5.set_xlim(0, 300)
ax5.legend()
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# --- Plot 6: Embarkation Port ---
ax6 = fig.add_subplot(gs[1, 2])
embark_survival = df.groupby('embark_town')['survived'].mean() * 100
port_colors = ['#9b59b6', '#1abc9c', '#e67e22']
bars6 = ax6.bar(embark_survival.index, embark_survival.values,
                color=port_colors, edgecolor='white', linewidth=1.5, width=0.5)
ax6.set_title('Survival Rate by Embarkation Port', fontweight='bold', fontsize=12)
ax6.set_ylabel('Survival Rate (%)')
ax6.set_ylim(0, 80)
for bar, val in zip(bars6, embark_survival.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

# --- Plot 7: Heatmap - Correlation Matrix ---
ax7 = fig.add_subplot(gs[2, 0:2])
numeric_cols = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].corr()
sns.heatmap(numeric_cols, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax7, linewidths=0.5,
            annot_kws={'size': 11, 'weight': 'bold'})
ax7.set_title('Correlation Heatmap', fontweight='bold', fontsize=12)
ax7.tick_params(axis='x', rotation=30)

# --- Plot 8: Survival by Gender and Class ---
ax8 = fig.add_subplot(gs[2, 2])
pivot = df.groupby(['pclass', 'sex'])['survived'].mean().unstack() * 100
x = np.arange(3)
width = 0.35
ax8.bar(x - width/2, pivot['female'], width, label='Female',
        color='#e91e8c', alpha=0.85, edgecolor='white')
ax8.bar(x + width/2, pivot['male'], width, label='Male',
        color='#3498db', alpha=0.85, edgecolor='white')
ax8.set_title('Survival by Class & Gender', fontweight='bold', fontsize=12)
ax8.set_xticks(x)
ax8.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
ax8.set_ylabel('Survival Rate (%)')
ax8.legend()
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)

plt.savefig('titanic_eda.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✅ Chart saved as 'titanic_eda.png'")

# ── Step 5: Key Insights ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🔍 KEY INSIGHTS FROM TITANIC EDA")
print("=" * 60)
print(f"  1. Overall survival rate        : {df['survived'].mean()*100:.1f}%")
print(f"  2. Female survival rate         : {df[df['sex']=='female']['survived'].mean()*100:.1f}%")
print(f"  3. Male survival rate           : {df[df['sex']=='male']['survived'].mean()*100:.1f}%")
print(f"  4. 1st Class survival rate      : {df[df['pclass']==1]['survived'].mean()*100:.1f}%")
print(f"  5. 3rd Class survival rate      : {df[df['pclass']==3]['survived'].mean()*100:.1f}%")
print(f"  6. Average age of survivors     : {df[df['survived']==1]['age'].mean():.1f} years")
print(f"  7. Average age of non-survivors : {df[df['survived']==0]['age'].mean():.1f} years")
print(f"  8. Average fare (survivors)     : £{df[df['survived']==1]['fare'].mean():.2f}")
print(f"  9. Average fare (non-survivors) : £{df[df['survived']==0]['fare'].mean():.2f}")
print("\n  📌 CONCLUSION: Women, 1st class passengers, and")
print("     those who paid higher fares had better survival chances.")
