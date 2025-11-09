import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 配置部分 - 修改这里的路径
# ============================================================
log_path = "/tmp/tinker-examples/sl_basic/metrics.jsonl"

# ============================================================
# 1. 检查文件是否存在
# ============================================================
print("="*60)
print("CHECKING LOG FILE")
print("="*60)

if not os.path.exists(log_path):
    print(f"❌ File not found: {log_path}")
    print("\nSearching for metrics files...")
    os.system("find /tmp/tinker-examples -name 'metrics.jsonl' 2>/dev/null")
    os.system("find /srv/nlprx-lab/share6/jhe478/tinker-cookbook -name 'metrics.jsonl' 2>/dev/null | head -10")
    exit(1)

file_size = os.path.getsize(log_path)
print(f"✓ File found: {log_path}")
print(f"✓ File size: {file_size} bytes")

if file_size == 0:
    print("❌ File is empty!")
    exit(1)

# ============================================================
# 2. 读取数据
# ============================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

try:
    df = pd.read_json(log_path, lines=True)
    print(f"✓ Successfully loaded {len(df)} rows")
    print(f"\nAvailable columns: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error reading file: {e}")
    exit(1)

# ============================================================
# 3. 数据预览
# ============================================================
print("\n" + "="*60)
print("DATA PREVIEW")
print("="*60)
print(df.head())

# ============================================================
# 4. 训练统计
# ============================================================
print("\n" + "="*60)
print("TRAINING STATISTICS")
print("="*60)

print("\n📉 Loss Statistics:")
print(f"  Initial loss:        {df['train_mean_nll'].iloc[0]:.4f}")
print(f"  Final loss:          {df['train_mean_nll'].iloc[-1]:.4f}")
print(f"  Minimum loss:        {df['train_mean_nll'].min():.4f}")
print(f"  Maximum loss:        {df['train_mean_nll'].max():.4f}")
print(f"  Loss reduction:      {df['train_mean_nll'].iloc[0] - df['train_mean_nll'].iloc[-1]:.4f}")
print(f"  Improvement:         {(1 - df['train_mean_nll'].iloc[-1]/df['train_mean_nll'].iloc[0])*100:.1f}%")
print(f"  Loss std deviation:  {df['train_mean_nll'].std():.4f}")

print("\n📊 Per-Epoch Statistics:")
if 'epoch' in df.columns:
    epoch_stats = df.groupby('epoch')['train_mean_nll'].agg(['mean', 'min', 'max', 'count']).round(4)
    print(epoch_stats)
    print(f"\nTotal epochs completed: {df['epoch'].max() + 1}")

print("\n⏱️ Training Time:")
print(f"  Total training time: {df['time/total'].sum():.2f} seconds ({df['time/total'].sum()/60:.2f} minutes)")
print(f"  Average step time:   {df['time/step'].mean():.2f} seconds")
print(f"  Fastest step:        {df['time/step'].min():.2f} seconds")
print(f"  Slowest step:        {df['time/step'].max():.2f} seconds")

print("\n📦 Data Statistics:")
print(f"  Total steps:         {len(df)}")
print(f"  Sequences per step:  {df['num_sequences'].iloc[0]}")
print(f"  Avg tokens per step: {df['num_tokens'].mean():.0f}")
print(f"  Total tokens seen:   {df['num_tokens'].sum():,}")
print(f"  Total loss tokens:   {df['num_loss_tokens'].sum():,}")

print("\n📈 Learning Rate:")
print(f"  Initial LR:          {df['learning_rate'].iloc[0]:.2e}")
print(f"  Final LR:            {df['learning_rate'].iloc[-1]:.2e}")

# ============================================================
# 5. 训练评估
# ============================================================
print("\n" + "="*60)
print("TRAINING ASSESSMENT")
print("="*60)

# 检查损失水平
if df['train_mean_nll'].iloc[-1] < 0.8:
    print("  ✅ Final loss is good (< 0.8)")
elif df['train_mean_nll'].iloc[-1] < 1.0:
    print("  ⚠️ Final loss is acceptable (0.8-1.0)")
else:
    print("  ❌ Final loss is high (> 1.0)")

# 检查学习程度
loss_reduction = df['train_mean_nll'].iloc[0] - df['train_mean_nll'].iloc[-1]
if loss_reduction > 0.15:
    print("  ✅ Good learning progress (loss reduced > 0.15)")
elif loss_reduction > 0.05:
    print("  ⚠️ Moderate learning (loss reduced 0.05-0.15)")
else:
    print("  ❌ Limited learning (loss reduced < 0.05)")

# 检查收敛情况
recent_losses = df['train_mean_nll'].tail(5)
if recent_losses.is_monotonic_increasing:
    print("  ⚠️ Warning: Loss increasing in final steps (possible overfitting)")
elif recent_losses.std() < 0.01:
    print("  ✅ Loss has converged (stable in final steps)")
else:
    print("  ℹ️ Loss still decreasing (could train longer)")

# ============================================================
# 6. 可视化
# ============================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 图1: 训练损失曲线
axes[0, 0].plot(df['step'], df['train_mean_nll'], marker='o', linewidth=2, markersize=4, color='#2E86AB')
axes[0, 0].set_xlabel('Step', fontsize=12)
axes[0, 0].set_ylabel('NLL Loss', fontsize=12)
axes[0, 0].set_title('Training Loss Over Steps', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=df['train_mean_nll'].min(), color='red', linestyle='--', alpha=0.5, 
                   label=f'Min: {df["train_mean_nll"].min():.4f}')
axes[0, 0].axhline(y=df['train_mean_nll'].iloc[-1], color='green', linestyle='--', alpha=0.5,
                   label=f'Final: {df["train_mean_nll"].iloc[-1]:.4f}')
axes[0, 0].legend(fontsize=10)

# 图2: 学习率变化
axes[0, 1].plot(df['step'], df['learning_rate'], color='#F18F01', linewidth=2)
axes[0, 1].set_xlabel('Step', fontsize=12)
axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# 图3: 每个epoch的平均损失
if 'epoch' in df.columns:
    epoch_means = df.groupby('epoch')['train_mean_nll'].mean()
    bars = axes[1, 0].bar(epoch_means.index, epoch_means.values, color='#6A4C93', alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Average NLL Loss', fontsize=12)
    axes[1, 0].set_title('Average Loss per Epoch', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for i, (bar, v) in enumerate(zip(bars, epoch_means.values)):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 图4: 训练时间分析
axes[1, 1].plot(df['step'], df['time/step'], label='Step Time', marker='o', linewidth=2, 
                markersize=4, color='#C73E1D')
axes[1, 1].set_xlabel('Step', fontsize=12)
axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
axes[1, 1].set_title('Time per Step', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=df['time/step'].mean(), color='blue', linestyle='--', alpha=0.5, 
                   label=f'Mean: {df["time/step"].mean():.2f}s')
axes[1, 1].legend(fontsize=10)

plt.tight_layout()
output_file = 'training_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_file}")

# ============================================================
# 7. 保存详细报告
# ============================================================
print("\n" + "="*60)
print("SAVING DETAILED REPORT")
print("="*60)

report_file = 'training_report.txt'
with open(report_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("TRAINING REPORT\n")
    f.write("="*60 + "\n\n")
    
    f.write("Loss Statistics:\n")
    f.write(f"  Initial loss:        {df['train_mean_nll'].iloc[0]:.4f}\n")
    f.write(f"  Final loss:          {df['train_mean_nll'].iloc[-1]:.4f}\n")
    f.write(f"  Minimum loss:        {df['train_mean_nll'].min():.4f}\n")
    f.write(f"  Loss reduction:      {loss_reduction:.4f} ({(1 - df['train_mean_nll'].iloc[-1]/df['train_mean_nll'].iloc[0])*100:.1f}%)\n\n")
    
    f.write("Training Time:\n")
    f.write(f"  Total time:          {df['time/total'].sum():.2f} seconds ({df['time/total'].sum()/60:.2f} minutes)\n")
    f.write(f"  Average step time:   {df['time/step'].mean():.2f} seconds\n\n")
    
    f.write("Data Statistics:\n")
    f.write(f"  Total steps:         {len(df)}\n")
    f.write(f"  Total epochs:        {df['epoch'].max() + 1}\n")
    f.write(f"  Total tokens seen:   {df['num_tokens'].sum():,}\n\n")
    
    if 'epoch' in df.columns:
        f.write("Per-Epoch Statistics:\n")
        f.write(epoch_stats.to_string())
        f.write("\n")

print(f"✓ Report saved to: {report_file}")

# ============================================================
# 8. 显示图表
# ============================================================
print("\n" + "="*60)
print("DISPLAYING PLOTS")
print("="*60)
print("Close the plot window to continue...")

plt.show()

print("\n✅ Analysis complete!")
print(f"\nGenerated files:")
print(f"  - {output_file}")
print(f"  - {report_file}")