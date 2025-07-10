# import modules
import numpy as np
import pandas as pd
from scipy.stats import ranksums
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

#load the evaluation metrics results of all three frameworks across metrics
result = {
    "models": ["Proposed", "FREED++", "ParetoDrug", "REINVENT4"],
    "Uniqueness1": [0.999, 0.740, 0.999, 1.000],
    "Uniqueness2": [0.975, 0.719, 0.996, 1.000],
    "Uniqueness3": [0.963, 0.804, 1.000, 1.000],
    "Diversity1": [0.726, 0.480, 0.704, 0.853],
    "Diversity2": [0.728, 0.520, 0.743, 0.849],
    "Diversity3": [0.765, 0.520, 0.746, 0.857],
    "Filters1": [0.849, 0.982, 0.772, 0.810],
    "Filters2": [0.868, 0.598, 0.889, 0.799],
    "Filters3": [0.822, 0.967, 0.946, 0.820],
    "DS1": [-9.397, -12.825, -8.583, -6.748],
    "DS2": [-8.816, -11.359, -6.200, -6.035],
    "DS3": [-9.122, -11.629, -6.714, -6.699],
    "SAS1": [2.868, 3.161, 3.000, 2.887],
    "SAS2": [2.582, 3.382, 2.714, 2.838],
    "SAS3": [2.879, 3.804, 2.786, 2.816]
}

#convert data to DataFrames that is easier for calculation
result_df = pd.DataFrame(result)
result_df.set_index("models", inplace=True)

#Create function to calculate p-value from rank-sum tests and save data in the format that can be used to create heatmap
def create_p_values_dict(data_df, metrics):
    baseline_models = data_df.index

    p_values_dict = {metric: pd.DataFrame(index=baseline_models, columns=baseline_models) for metric in metrics}

    for metric in metrics:
        for model1 in baseline_models:
            for model2 in baseline_models:
                #statistic, p_value = stats.wilcoxon(model1_scores, model2_scores, alternative='two-sided')
                stat, p_val = ranksums(
                    data_df.loc[model1, [f"{metric}1", f"{metric}2", f"{metric}3"]],
                    data_df.loc[model2, [f"{metric}1", f"{metric}2", f"{metric}3"]])
                p_values_dict[metric].at[model1, model2] = p_val

    for metric in metrics:
        p_values_dict[metric] = p_values_dict[metric].astype(float)

    p_values_binary_dict = {metric: p_values_dict[metric].applymap(lambda x: 1 if x < 0.05 else 0) for metric in metrics}
    
    return p_values_binary_dict

#create p-values binary dictionaries
metrics = ['Uniqueness', 'Diversity', 'Filters', 'DS', 'SAS']
result_p_values_binary_dict = create_p_values_dict(result_df, metrics)
p_values_binary_dict = [result_p_values_binary_dict]

#set the font to Time for the heatmap plots
plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18}

#adjust the figure layout to improve the spacing between plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

#define the colormap and normalization
cmap = ListedColormap(['#8a8686', '#a84242'])
norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1], ncolors=2)

metric_titles = ['Uniqueness', 'Diversity', 'Filters', 'DS', 'SAS']

for metric_index, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[metric_index // 3, metric_index % 3]
        sns.heatmap(p_values_binary_dict[metric], annot=False, cmap=cmap, norm=norm, cbar=False, linewidths=1, linecolor='white', ax=ax)
        ax.set_title(f"{title}", fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_aspect('equal', adjustable='box')

#custom legend
legend_handles = [
    plt.Rectangle((0,0),1,1, color='#a84242', label='p-value < 0.05'),
    plt.Rectangle((0,0),1,1, color='#8a8686', label='p-value >= 0.05'),
]

axes[1, 2].axis('off')
fig.legend(handles=legend_handles, loc='lower center', ncol=2)
plt.tight_layout(rect=[0, 0.07, 1, 1])