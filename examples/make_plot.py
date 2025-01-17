import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create quality folder if it doesn't exist
os.makedirs('quality', exist_ok=True)

# Read the CSV file
df = pd.read_csv('experiment_results_averaged.csv')

# Plot and save PSNR
plt.figure(figsize=(8, 6))
for num_points in df['num_points'].unique():
    data = df[df['num_points'] == num_points]
    plt.plot(data['iterations'], data['psnr_mean'], marker='o', label=f'{num_points} points')
plt.xlabel('Iterations')
plt.ylabel('PSNR')
plt.title('PSNR vs Iterations\n(Higher is better)')
# Add arrow pointing upward with text
plt.annotate('Better', xy=(0.90, 0.5), xytext=(0.90, 0.3),
             xycoords='axes fraction', arrowprops=dict(arrowstyle='->'), 
             ha='center', va='center')
plt.legend()
plt.grid(True)
plt.savefig('quality/psnr.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot and save LPIPS
plt.figure(figsize=(8, 6))
for num_points in df['num_points'].unique():
    data = df[df['num_points'] == num_points]
    plt.plot(data['iterations'], data['lpips_mean'], marker='o', label=f'{num_points} points')
plt.xlabel('Iterations')
plt.ylabel('LPIPS')
plt.title('LPIPS vs Iterations\n(Lower is better)')
# Add arrow pointing downward with text
plt.annotate('Better', xy=(0.90, 0.3), xytext=(0.90, 0.5),
             xycoords='axes fraction', arrowprops=dict(arrowstyle='->'), 
             ha='center', va='center')
plt.legend()
plt.grid(True)
plt.savefig('quality/lpips.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot and save SSIM
plt.figure(figsize=(8, 6))
for num_points in df['num_points'].unique():
    data = df[df['num_points'] == num_points]
    plt.plot(data['iterations'], data['ssim_mean'], marker='o', label=f'{num_points} points')
plt.xlabel('Iterations')
plt.ylabel('SSIM')
plt.title('SSIM vs Iterations\n(Higher is better)')
# Add arrow pointing upward with text
plt.annotate('Better', xy=(0.90, 0.5), xytext=(0.90, 0.3),
             xycoords='axes fraction', arrowprops=dict(arrowstyle='->'), 
             ha='center', va='center')
plt.legend()
plt.grid(True)
plt.savefig('quality/ssim.png', dpi=300, bbox_inches='tight')
plt.close()
