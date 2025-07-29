"""
This visualizes the results of the caption evaluation analysis by generating
a series of bar plots from pre-computed summary and similarity CSV files.

It reads 'summary.csv' to create plots for summary statistics, including the average
caption length and the average number of defect keywords, grouped by the captioning
system (e.g., BLIP, Florence2) and data domain (e.g., real_io, syn_nio).

It reads 'pairwise_similarity.csv' to generate individual bar plots for each semantic
similarity metric (TFIDF, STS, BLEU, ROUGE, etc.). These plots compare the performance
of each captioning system for both IO and NIO image types.

All generated plots are saved as PNG files to the specified output directory,
'../../outputs/plots/captions/'.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = '../../outputs/plots/captions/'
os.makedirs(output_dir, exist_ok=True)

try:
    summary_df = pd.read_csv('../../outputs/reports/summary.csv')

    # average caption length
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x='system', y='avg_caption_length', hue='domain')
    plt.title('Average Caption Length by System and Domain')
    plt.xlabel('System')
    plt.ylabel('Average Caption Length')
    plt.tight_layout()
    plt.savefig(f'{output_dir}summary_avg_caption_length.png')
    plt.close()

    # average defect keywords
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x='system', y='avg_defect_keywords', hue='domain')
    plt.title('Average Defect Keywords by System and Domain')
    plt.xlabel('System')
    plt.ylabel('Average Defect Keywords')
    plt.tight_layout()
    plt.savefig(f'{output_dir}summary_avg_defect_keywords.png')
    plt.close()

except FileNotFoundError:
    print("Not found in the current directory")

try:
    similarity_df = pd.read_csv('../../outputs/reports/pairwise_similarity.csv')

    metrics = [
        'TFIDF_sim', 'STS_sim', 'BLEU_4', 'ROUGE_L_f1', 'METEOR',
        'BERTScore_F1', 'CLIPScore', 'DefectKeyword_F1'
    ]

    similarity_melted = similarity_df.melt(
        id_vars=['system', 'type'],
        value_vars=metrics,
        var_name='metric',
        value_name='score'
    )

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        metric_data = similarity_melted[similarity_melted['metric'] == metric]
        sns.barplot(data=metric_data, x='system', y='score', hue='type')
        plt.title(f'{metric} by System and Type')
        plt.ylabel(metric)
        plt.xlabel('System')
        plt.tight_layout()
        plt.savefig(f'{output_dir}similarity_{metric}.png')
        plt.close()

except FileNotFoundError:
    print("not found in the current directory")
