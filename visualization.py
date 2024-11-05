import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_style('whitegrid')
sns.set_palette('Set2')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14})

def plot_results(results, title):
    df = pd.DataFrame(results)

    if title == 'Grid Search Results':
        batch_sizes = df['batch_size'].unique()

        plt.figure(figsize=(16, 12))
        for i, batch_size in enumerate(batch_sizes, 1):
            plt.subplot(2, 2, i)
            batch_df = df[df['batch_size'] == batch_size]
            pivot_table = batch_df.pivot(index='learning_rate', columns='dropout_rate', values='accuracy')
            sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Accuracy'}, fmt='.4f')
            plt.title(f'Grid Search Accuracy for Batch Size = {batch_size}')
            plt.xlabel('Dropout Rate')
            plt.ylabel('Learning Rate')

        plt.tight_layout()
        plt.show()

    elif title == 'Random Search Results':
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='learning_rate', y='accuracy', hue='dropout_rate', style='batch_size', s=100, palette='deep')
        plt.title('Accuracy by Learning Rate with Dropout Rate and Batch Size Variations')
        plt.xlabel('Learning Rate')
        plt.ylabel('Accuracy')
        plt.legend(title='Dropout Rate / Batch Size')
        plt.grid(True)
        plt.show()
    
    else:
        print("Invalid title. Please use 'Grid Search Results' or 'Random Search Results'.")

