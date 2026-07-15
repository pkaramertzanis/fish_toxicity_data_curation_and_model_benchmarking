from string import ascii_letters

import logger
log = logger.setup_applevel_logger(file_name ='logs/REACH_flatten.log')

# mixed model analysis of experimental variability in LC50 data
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('tkagg')
%matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import upsetplot

# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr',False)
# maximum number of columns
pd.set_option("display.max_columns",50)
# maximum number of rows
pd.set_option("display.max_rows",5000)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write, needs to be False due to the UpSet library
pd.options.mode.copy_on_write = False

datasets = pd.read_pickle(rf'data/fish_chronic/processed/REACH_long_term_fish_measurement.pickle')

# remove excluded rows
msk = datasets['exclusion reasons'].isnull()
datasets = datasets.loc[msk]

# upset plot with scenarios, keep only scenarios with suffix a and b
msk = datasets['scenario'].str.split('_').str[1].isin(['1a', '1b', '2a', '2b', '4a', '4b'])
tmp = datasets.loc[msk]
FLS_scenarios = tmp.query('scenario.str.startswith("FLS")', engine='python')['scenario'].dropna().drop_duplicates().sort_values().to_list()
LCT_scenarios = tmp.query('scenario.str.startswith("LCT")', engine='python')['scenario'].dropna().drop_duplicates().sort_values().to_list()
JGT_scenarios = tmp.query('scenario.str.startswith("JGT")', engine='python')['scenario'].dropna().drop_duplicates().sort_values().to_list()
SDT_scenarios = tmp.query('scenario.str.startswith("SDT")', engine='python')['scenario'].dropna().drop_duplicates().sort_values().to_list()
ESF_scenarios = tmp.query('scenario.str.startswith("ESF")', engine='python')['scenario'].dropna().drop_duplicates().sort_values().to_list()
data = tmp.assign(dummy=1).pivot_table(index='dtxsid', columns='scenario', values='dummy', aggfunc='sum').map(pd.notnull).groupby(FLS_scenarios+LCT_scenarios+JGT_scenarios+SDT_scenarios+ESF_scenarios).size()
fig = plt.figure()
upset = upsetplot.UpSet(data, show_counts=True, element_size=30, intersection_plot_elements=10, totals_plot_elements=10)
upset.style_subsets(min_degree=2, facecolor="gray", label='multiple assays') # default is gray
for scenario in FLS_scenarios:
    upset.style_subsets(present=scenario, absent=LCT_scenarios+JGT_scenarios+SDT_scenarios+ESF_scenarios, facecolor="black", label='FLS')
for scenario in LCT_scenarios:
    upset.style_subsets(present=scenario, absent=FLS_scenarios+JGT_scenarios+SDT_scenarios+ESF_scenarios, facecolor="orange", label='LCT')
for scenario in JGT_scenarios:
    upset.style_subsets(present=scenario, absent=FLS_scenarios+LCT_scenarios+SDT_scenarios+ESF_scenarios, facecolor="olive", label='JGT')
for scenario in SDT_scenarios:
    upset.style_subsets(present=scenario, absent=FLS_scenarios+LCT_scenarios+JGT_scenarios+ESF_scenarios, facecolor="#4682B4", label='SDT') # steel blue
for scenario in ESF_scenarios:
    upset.style_subsets(present=scenario, absent=FLS_scenarios+LCT_scenarios+JGT_scenarios+SDT_scenarios, facecolor="#B22222", label='ESF') # dark red
# upset.style_subsets(min_degree=2, facecolor="gray", label='multiple scenarios')
plot_object = upset.plot(fig=fig, )
intersections_ax = plot_object["intersections"]
intersections_ax.set_ylabel("number of structures", fontsize=12)
totals_ax = plot_object["totals"]
totals_ax.set_xlabel("number of structures", fontsize=12)
plot_object['totals'].grid(False)
# Increase tick label fonts
for ax in plot_object.values():
    ax.tick_params(labelsize=12)
for txt in totals_ax.texts:
    txt.set_fontsize(12)
for txt in intersections_ax.texts:
    txt.set_fontsize(12)
leg = intersections_ax.get_legend()
if leg is not None:
    for child in leg.get_texts():
        if isinstance(child, plt.Text):
            child.set_fontsize(12)
fig.savefig(rf'figures/REACH_long_term_fish_toxicity_NOEC_EC10_upset_plot_scenario_a_b.png', dpi=600)


# keep only the rows corresponding to scenarios with suffix a and b
msk = datasets['scenario'].str.split('_').str[1].isin(['1a', '1b', '2a', '2b', '4a', '4b'])
datasets_filtered = datasets[msk].copy()


# for each study keep the minimum lower bound of the effect concentration (most toxic) per study and scenario, e.g. if we have different effect levels for growth and reproduction in the same study
datasets_filtered = (datasets_filtered
                     .groupby(['dtxsid', 'UUID (endpoint study record) (echachem)', 'scenario'])
                     ['effect concentration (mg/L, lower bound)'].agg('min')
                     .reset_index())

# assign an index to the dtxsid so that the most toxic is at index 0
y_pos_dict = (datasets_filtered
       .groupby('dtxsid')['effect concentration (mg/L, lower bound)'].min()
       .sort_values(ascending=True)
       .reset_index().reset_index().rename({'index': 'y_pos'}, axis='columns')
       .set_index('dtxsid')['y_pos']
       .to_dict())
datasets_filtered['y_pos'] = datasets_filtered['dtxsid'].map(y_pos_dict)

fig = plt.figure(figsize=(12, 6))
# create a GridSpec with width ratios: left = 80%, right = 20%
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])  # 4:1 = 80% : 20%
# create subplots from the GridSpec
ax_left = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1])
for label, scenario_suffix, zorder, color in zip(['scenarios 1a, 1b', 'scenarios 2a, 2b, 4a, 4b'], [['1a', '1b'],['2a','2b', '4a','4b']], [2, 1], ['black', 'orange']):
    msk = datasets_filtered['scenario'].str.split('_').str[1].isin(scenario_suffix)
    # lower bound
    effect_level = np.log10(datasets_filtered.loc[msk, 'effect concentration (mg/L, lower bound)'])
    y_pos = datasets_filtered.loc[msk, 'y_pos']
    ax_left.scatter(
        effect_level,
        y_pos,
        label=label,
        alpha=0.7,
        zorder=zorder,
        marker="o",
        facecolor=color,
        edgecolor=color,
        linewidth=0.5
    )
ax_left.legend(loc="upper left", frameon=False)
# handle spines
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)
ax_left.spines['bottom'].set_position(('outward', 10))
ax_left.spines['left'].set_position(('outward', 10))
ax_left.set_xlabel("log₁₀(LC50 [mg/L])")
ax_left.set_ylabel("substance index")
ax_left.grid(True, linestyle="--", alpha=0.3)
ax_left.set_yticks([])
# use the right axis to plot the range of effect concentrations per substance as horizontal bars
range_df = (datasets_filtered
            .melt(id_vars=['dtxsid', 'y_pos'], value_vars=['effect concentration (mg/L, lower bound)']))
range_df = (range_df
            .loc[~range_df['value'].isin([np.inf, -np.inf])]
            .groupby(['y_pos', 'dtxsid'])['value']
            .agg(['min', 'max'])
            .assign(range=lambda df: np.log10(df['max']) - np.log10(df['min'])))
ax_right.scatter(range_df['range'], range_df.index.get_level_values(0),
        alpha=1.,
        marker="o",
        facecolor='gray',
        edgecolor='gray',
        linewidth=0.5,
        s=5
    )
ax_right.spines['top'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['bottom'].set_position(('outward', 10))
ax_right.set_xlabel("log₁₀(LC50 [mg/L]) range")
ax_right.set_yticks([])
plt.tight_layout()
plt.show()

fig.savefig(rf'figures/REACH_long_term_fish_toxicity_NOEC_EC10_dot_plot.png', dpi=600)
