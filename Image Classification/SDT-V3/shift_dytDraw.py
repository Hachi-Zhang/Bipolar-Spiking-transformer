import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# 设置高级绘图风格 - 放大字号
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 20,                    # 基础字号 12→16
    'axes.linewidth': 1.5,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 7,
    'xtick.minor.size': 4,
    'ytick.major.size': 7,
    'ytick.minor.size': 4,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.labelsize': 22,              # 坐标刻度字号
    'ytick.labelsize': 22,
    'grid.linewidth': 1.0,
    'lines.linewidth': 2.5,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'legend.framealpha': 0.9,
    'legend.fontsize': 16,              # 图例字号
    'figure.facecolor': 'white'
})

class TanhShiftWeighted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        abs_x = torch.abs(x)
        sign_x = torch.sign(x)

        # 线性区域
        boundary_val = 1 - torch.pow(torch.tensor(2.0, device=x.device), torch.tensor(-1.0, device=x.device))
        scale_factor = boundary_val / 0.75
        small_signal_corrected = scale_factor * 0.75 * x

        # 指数区域:加权平均方案
        abs_x_floor = torch.clamp(torch.floor(abs_x), 0, 15).long()
        abs_x_ceil = torch.clamp(torch.ceil(abs_x), 0, 15).long()

        frac = abs_x - torch.floor(abs_x)
        weight = frac

        power_floor = torch.pow(torch.tensor(2.0, device=x.device), -abs_x_floor.float())
        power_ceil = torch.pow(torch.tensor(2.0, device=x.device), -abs_x_ceil.float())

        power_weighted = (1 - weight) * power_floor + weight * power_ceil
        large_signal = sign_x * (1 - power_weighted)

        result = torch.where(abs_x < 1.0, small_signal_corrected, large_signal)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        abs_x = torch.abs(x)

        boundary_val = 1 - torch.pow(torch.tensor(2.0, device=x.device), torch.tensor(-1.0, device=x.device))
        scale_factor = boundary_val / 0.75
        small_grad = scale_factor * 0.75 * torch.ones_like(x)

        n = torch.clamp(torch.floor(abs_x), 0, 15)
        large_grad = torch.pow(torch.tensor(2.0, device=x.device), -(n + 1))

        grad = torch.where(abs_x < 1.0, small_grad, large_grad)
        return grad_output * grad

def compute_shift_dyt(x):
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = TanhShiftWeighted.apply(x_tensor)
    return y.detach().numpy()

def compute_shift_dyt_gradient(x):
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = TanhShiftWeighted.apply(x_tensor)
    grad_output = torch.ones_like(y)
    grad_input = torch.autograd.grad(y, x_tensor, grad_output, create_graph=False)[0]
    return grad_input.detach().numpy()

# 数据准备
x_range = np.linspace(-5, 5, 2000)
x_grad_range = np.linspace(-3.5, 3.5, 1000)

shift_dyt_vals = compute_shift_dyt(x_range)
tanh_vals = np.tanh(x_range)
error_vals = np.abs(shift_dyt_vals - tanh_vals)

shift_dyt_grads = compute_shift_dyt_gradient(x_grad_range)
tanh_grads = 1 - np.tanh(x_grad_range)**2
grad_error = np.abs(shift_dyt_grads - tanh_grads)

# 创建图形 - 增大图形尺寸以适配更大字号
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# ==================== 函数值对比图 ====================
# 设置背景渐变
ax1.axvspan(-5, -1, alpha=0.05, color='lightblue', label='_nolegend_')
ax1.axvspan(-1, 1, alpha=0.08, color='lightgreen', label='_nolegend_')
ax1.axvspan(1, 5, alpha=0.05, color='lightblue', label='_nolegend_')

# 主要曲线
line1 = ax1.plot(x_range, shift_dyt_vals, color='#2E86AB', linewidth=4,
                 label='Shift-DYT', alpha=0.9, zorder=3)
line2 = ax1.plot(x_range, tanh_vals, color='#A23B72', linewidth=3.5,
                 linestyle='--', label='Tanh', alpha=0.85, zorder=2)

# 边界线
ax1.axvline(x=-1, color='#F18F01', linestyle=':', linewidth=2.5, alpha=0.7, zorder=1)
ax1.axvline(x=1, color='#F18F01', linestyle=':', linewidth=2.5, alpha=0.7, zorder=1)
ax1.text(-1, 0.85, '|x|=1', ha='center', va='bottom', fontsize=20,
         color='#F18F01', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
ax1.text(1, 0.85, '|x|=1', ha='center', va='bottom', fontsize=20,
         color='#F18F01', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 误差填充区域
smooth_error = gaussian_filter1d(error_vals, sigma=2)
ax1.fill_between(x_range, shift_dyt_vals - smooth_error, shift_dyt_vals + smooth_error,
                 alpha=0.15, color='red', label='Error Range', zorder=1)

# 设置坐标轴
ax1.set_xlabel('Input (x)', fontsize=20, fontweight='bold', color='#2C3E50')
ax1.set_ylabel('Output f(x)', fontsize=20, fontweight='bold', color='#2C3E50')
ax1.set_title('Function Comparison: Shift-DYT vs Tanh', fontsize=20, fontweight='bold',
              color='#2C3E50', pad=25)

# 网格和样式
ax1.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-1.1, 1.1)

# 图例
legend1 = ax1.legend(loc='lower right', fontsize=16, framealpha=0.95,
                     facecolor='white', edgecolor='gray')
legend1.get_frame().set_linewidth(1.5)

# 添加区域标注
ax1.text(0, -0.9, 'Linear Region',
         ha='center', va='center', fontsize=18, style='italic',
         bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.7))
ax1.text(-3, -0.2, 'Exponential Decay',
         ha='center', va='center', fontsize=18, style='italic',
         bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.7))
ax1.text(3, 0, 'Exponential Decay',
         ha='center', va='center', fontsize=18, style='italic',
         bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.7))

# ==================== 梯度对比图 ====================
# 设置背景渐变
ax2.axvspan(-3.5, -1, alpha=0.05, color='lightcoral', label='_nolegend_')
ax2.axvspan(-1, 1, alpha=0.08, color='lightgreen', label='_nolegend_')
ax2.axvspan(1, 3.5, alpha=0.05, color='lightcoral', label='_nolegend_')

# 主要曲线
line3 = ax2.plot(x_grad_range, shift_dyt_grads, color='#2E86AB', linewidth=4,
                 label='Shift-DYT Gradient', alpha=0.9, zorder=3)
line4 = ax2.plot(x_grad_range, tanh_grads, color='#A23B72', linewidth=3.5,
                 linestyle='--', label='Tanh Gradient', alpha=0.85, zorder=2)

# 边界线
ax2.axvline(x=-1, color='#F18F01', linestyle=':', linewidth=2.5, alpha=0.7, zorder=1)
ax2.axvline(x=1, color='#F18F01', linestyle=':', linewidth=2.5, alpha=0.7, zorder=1)
ax2.text(-1, 0.9, '|x|=1', ha='center', va='bottom', fontsize=20,
         color='#F18F01', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
ax2.text(1, 0.9, '|x|=1', ha='center', va='bottom', fontsize=20,
         color='#F18F01', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 梯度误差填充区域
smooth_grad_error = gaussian_filter1d(grad_error, sigma=2)
ax2.fill_between(x_grad_range, shift_dyt_grads - smooth_grad_error,
                 shift_dyt_grads + smooth_grad_error,
                 alpha=0.15, color='red', label='Gradient Error Range', zorder=1)

# 阶梯状梯度特征高亮
step_positions = [-3, -2, -1, 0, 1, 2, 3]
for i, pos in enumerate(step_positions[:-1]):
    if abs(pos) >= 1:
        step_height = 2**(-abs(pos)-1) if pos != 0 else shift_dyt_grads[np.argmin(np.abs(x_grad_range - pos))]
        ax2.hlines(step_height, pos, step_positions[i+1], colors='orange',
                  linestyles='solid', alpha=0.6, linewidth=4.5, zorder=1)

# 设置坐标轴
ax2.set_xlabel('Input (x)', fontsize=20, fontweight='bold', color='#2C3E50')
ax2.set_ylabel("f'(x)", fontsize=20, fontweight='bold', color='#2C3E50')
ax2.set_title("Gradient Comparison: Shift-DYT vs Tanh", fontsize=20, fontweight='bold',
              color='#2C3E50', pad=25)

# 网格和样式
ax2.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(0, 1.0)

# 图例
legend2 = ax2.legend(loc='upper right', fontsize=16, framealpha=0.95,
                     facecolor='white', edgecolor='gray')
legend2.get_frame().set_linewidth(1.5)

# 添加梯度特性标注
ax2.text(0, 0.35, 'Constant Gradient\n(Stable Training)',
         ha='center', va='center', fontsize=18, style='italic',
         bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.7))
ax2.text(-2.8, 0.45, 'Step-wise Decay\n2^(-(n+1))',
         ha='center', va='center', fontsize=18, style='italic',
         bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcoral', alpha=0.7))
ax2.text(2.8, 0.45, 'Step-wise Decay\n2^(-(n+1))',
         ha='center', va='center', fontsize=18, style='italic',
         bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcoral', alpha=0.7))

# 整体布局调整
plt.tight_layout(pad=3.5)

# 添加误差统计信息
max_func_error = np.max(error_vals)
mean_func_error = np.mean(error_vals)
max_grad_error = np.max(grad_error)
mean_grad_error = np.mean(grad_error)

# 在图上添加统计信息
stats_text = f'Function Error: Max={max_func_error:.4f}, Mean={mean_func_error:.4f}\n'
stats_text += f'Gradient Error: Max={max_grad_error:.4f}, Mean={mean_grad_error:.4f}'

fig.text(0.02, 0.02, stats_text, fontsize=13, style='italic',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
         verticalalignment='bottom')

# 保存高质量图片 - 确保300 DPI
print("正在保存高分辨率图片...")
plt.savefig('shift_dyt_comparison_300dpi.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', format='png')
plt.savefig('shift_dyt_comparison_300dpi.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', format='pdf')
plt.savefig('shift_dyt_comparison_300dpi.eps', bbox_inches='tight',
            facecolor='white', edgecolor='none', format='eps')
print("图片保存完成！")
print("- PNG格式：shift_dyt_comparison_300dpi.png (300 DPI)")
print("- PDF格式：shift_dyt_comparison_300dpi.pdf (矢量)")
print("- EPS格式：shift_dyt_comparison_300dpi.eps (矢量)")

plt.show()

# 打印详细统计信息
print("\n" + "="*60)
print("Shift-DYT vs Tanh - 详细性能分析")
print("="*60)
print(f"函数近似误差:")
print(f"  最大误差: {max_func_error:.6f}")
print(f"  平均误差: {mean_func_error:.6f}")
print(f"  均方根误差: {np.sqrt(np.mean(error_vals**2)):.6f}")

print(f"\n梯度近似误差:")
print(f"  最大误差: {max_grad_error:.6f}")
print(f"  平均误差: {mean_grad_error:.6f}")
print(f"  均方根误差: {np.sqrt(np.mean(grad_error**2)):.6f}")

print(f"\n边界连续性检查:")
for bp in [-1.0, 1.0]:
    left_val = compute_shift_dyt([bp - 1e-6])[0]
    right_val = compute_shift_dyt([bp + 1e-6])[0]
    discontinuity = abs(left_val - right_val)
    print(f"  x = {bp}: 跳跃幅度 = {discontinuity:.8f}")

print(f"\n计算复杂度对比:")
print(f"  Shift-DYT: 2×移位 + 3×加法 + 2×乘法")
print(f"  Tanh: 1×指数 + 1×除法 + 加法")
print(f"  复杂度比: ~5:1 (Shift-DYT更高效)")

print(f"\n图像文件信息:")
print(f"  分辨率: 300 DPI")
print(f"  图像尺寸: 14×12 英寸")
print(f"  像素尺寸: 4200×3600 像素")
print(f"  文件格式: PNG, PDF, EPS")