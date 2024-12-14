import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

class PerformanceVisualizer:
    @staticmethod
    def generate_category_performance_plot(metrics_file='reports/performance_metrics.json'):
        """
        Generate comprehensive performance visualization
        
        Args:
            metrics_file (str): Path to performance metrics JSON
        """
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(
            metrics['feature_extraction_rates'], 
            orient='index', 
            columns=['Extraction Rate']
        )
        
        # Set up the plot
        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")
        
        # Create bar plot
        ax = sns.barplot(
            x=df.index, 
            y='Extraction Rate', 
            data=df, 
            palette='viridis'
        )
        
        # Customize plot
        plt.title('Feature Extraction Performance by Category', fontsize=16)
        plt.xlabel('Product Categories', fontsize=12)
        plt.ylabel('Extraction Rate (%)', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(df['Extraction Rate']):
            ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('reports/category_performance.png')
        plt.close()

    @staticmethod
    def generate_trend_analysis_plot(metrics_file='reports/trend_metrics.json'):
        """
        Generate trend analysis visualization
        
        Args:
            metrics_file (str): Path to trend metrics JSON
        """
        # Implementation of trend analysis plot
        pass

# Execution
if __name__ == '__main__':
    visualizer = PerformanceVisualizer()
    visualizer.generate_category_performance_plot()
