import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


def analyze_training_results(log_path: str, output_dir: str):
    """
    Analyze training results and save visualizations and reports.
    
    Args:
        log_path: Path to the metrics.jsonl file
        output_dir: Directory to save analysis results
    """
    metrics_file = os.path.join(log_path, "metrics.jsonl")
    
    # Check if file exists
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file not found at {metrics_file}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_json(metrics_file, lines=True)
        print(f"Loaded {len(df)} training steps")
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return False
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Initial loss:     {df['train_mean_nll'].iloc[0]:.4f}")
    print(f"Final loss:       {df['train_mean_nll'].iloc[-1]:.4f}")
    print(f"Minimum loss:     {df['train_mean_nll'].min():.4f}")
    print(f"Loss reduction:   {df['train_mean_nll'].iloc[0] - df['train_mean_nll'].iloc[-1]:.4f}")
    print(f"Total time:       {df['time/total'].sum()/60:.2f} minutes")
    print(f"Total tokens:     {df['num_tokens'].sum():,}")
    if 'epoch' in df.columns:
        print(f"Total epochs:     {df['epoch'].max() + 1}")
    
    # Generate visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Training loss
    axes[0, 0].plot(df['step'], df['train_mean_nll'], marker='o', linewidth=2, 
                    markersize=4, color='#2E86AB')
    axes[0, 0].set_xlabel('Step', fontsize=12)
    axes[0, 0].set_ylabel('NLL Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Over Steps', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=df['train_mean_nll'].min(), color='red', linestyle='--', 
                       alpha=0.5, label=f'Min: {df["train_mean_nll"].min():.4f}')
    axes[0, 0].legend(fontsize=10)
    
    # Learning rate
    axes[0, 1].plot(df['step'], df['learning_rate'], color='#F18F01', linewidth=2)
    axes[0, 1].set_xlabel('Step', fontsize=12)
    axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Loss per epoch
    if 'epoch' in df.columns:
        epoch_means = df.groupby('epoch')['train_mean_nll'].mean()
        bars = axes[1, 0].bar(epoch_means.index, epoch_means.values, 
                              color='#6A4C93', alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Average NLL Loss', fontsize=12)
        axes[1, 0].set_title('Average Loss per Epoch', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, v in zip(bars, epoch_means.values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Time per step
    axes[1, 1].plot(df['step'], df['time/step'], marker='o', linewidth=2, 
                    markersize=4, color='#C73E1D')
    axes[1, 1].set_xlabel('Step', fontsize=12)
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_title('Time per Step', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=df['time/step'].mean(), color='blue', linestyle='--', 
                       alpha=0.5, label=f'Mean: {df["time/step"].mean():.2f}s')
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'training_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    plt.close()
    
    # Save text report
    report_file = os.path.join(output_dir, 'training_report.txt')
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("Loss Statistics:\n")
        f.write(f"  Initial loss:     {df['train_mean_nll'].iloc[0]:.4f}\n")
        f.write(f"  Final loss:       {df['train_mean_nll'].iloc[-1]:.4f}\n")
        f.write(f"  Minimum loss:     {df['train_mean_nll'].min():.4f}\n")
        f.write(f"  Loss reduction:   {df['train_mean_nll'].iloc[0] - df['train_mean_nll'].iloc[-1]:.4f}\n\n")
        
        f.write("Training Time:\n")
        f.write(f"  Total time:       {df['time/total'].sum():.2f} seconds ({df['time/total'].sum()/60:.2f} minutes)\n")
        f.write(f"  Avg step time:    {df['time/step'].mean():.2f} seconds\n\n")
        
        f.write("Data Statistics:\n")
        f.write(f"  Total steps:      {len(df)}\n")
        f.write(f"  Total tokens:     {df['num_tokens'].sum():,}\n")
        if 'epoch' in df.columns:
            f.write(f"  Total epochs:     {df['epoch'].max() + 1}\n")
            f.write("\nPer-Epoch Statistics:\n")
            epoch_stats = df.groupby('epoch')['train_mean_nll'].agg(['mean', 'min', 'max', 'count']).round(4)
            f.write(epoch_stats.to_string())
    
    print(f"Report saved to: {report_file}")
    print("\nAnalysis complete!")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analyze_results.py <log_path> <output_dir>")
        sys.exit(1)
    
    analyze_training_results(sys.argv[1], sys.argv[2])
