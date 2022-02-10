from utils.general_utils import load_data_old

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.plot_utils import color_mapping


def plot_by_rating(data, col, title):
    sns.lineplot(x='month', y=col, hue='rating', data=data)
    plt.title(title)
    plt.grid(True)
    plt.xticks(ticks=data.month.unique(), rotation=70)
    plt.show()


def plot_by_rating_territory(data, col, ax, territory):
    data = data[data.territory == territory]
    mapping = color_mapping()
    colors = [mapping[rating] for rating in data.rating.unique()]
    sns.lineplot(data=data, x='month', y=col, hue='rating', ax=ax, palette=colors)
    ax.grid(True)
    ax.set_title(territory)
    ax.set_xticklabels(data.month.unique(), rotation=70)


def subplot_by_rating(data, col, title):
    fig, axs = plt.subplots(nrows=2, figsize=(12, 12))
    plot_by_rating_territory(data=data, col=col, ax=axs[0], territory='EMEA/APAC')
    plot_by_rating_territory(data=data, col=col, ax=axs[1], territory='Americas')
    plt.suptitle(title)
    plt.show()

# - Preparing data by territory for plotting
eval_df = load_data_old('evaluation.sql')
summary_eval_df = eval_df.groupby(['month', 'territory'], as_index=False).agg({'count': 'sum', 'n_upsells': 'sum'})
summary_eval_df['cr'] = summary_eval_df['n_upsells'] / summary_eval_df['count']
summary_eval_df['rating'] = 'total'
summary_eval_df = summary_eval_df[eval_df.columns]
eval_df = pd.concat([eval_df, summary_eval_df], axis=0)
eval_df['month'] = pd.to_datetime(eval_df['month']).dt.date
eval_df = eval_df.sort_values('month')

# - Preparing data without territory distinction for plotting
eval_df_total = eval_df.groupby(['month', 'rating'], as_index=False).agg({'count': 'sum', 'n_upsells': 'sum'})
eval_df_total = eval_df_total.query('rating != "total"')
eval_df_total['cr'] = eval_df_total['n_upsells'] / eval_df_total['count']
summary_eval_df_total = eval_df_total.groupby('month', as_index=False).agg({'count': 'sum', 'n_upsells': 'sum'})
summary_eval_df_total['cr'] = summary_eval_df_total['n_upsells'] / summary_eval_df_total['count']
summary_eval_df_total['rating'] = 'total'
summary_eval_df_total = summary_eval_df_total[eval_df_total.columns]
eval_df_total = pd.concat([eval_df_total, summary_eval_df_total], axis=0)
eval_df_total['month'] = pd.to_datetime(eval_df_total['month']).dt.date
eval_df_total = eval_df_total.sort_values('month')


# --- By territory
# - Number of Upsells
subplot_by_rating(eval_df, col='n_upsells', title='Number of Upsells Tracking')

# - Conversion Rate
subplot_by_rating(eval_df, col='cr', title='Conversion Rate Tracking')

# --- Total
# - Number of Upsells
plot_by_rating(data=eval_df_total, col='n_upsells', title='Number of Upsells Tracking')

# - Conversion Rate
plot_by_rating(data=eval_df_total, col='cr', title='Conversion Rate Tracking')


