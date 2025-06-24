import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def seiar_model(y, t, beta, N, f, sigma, gamma_i, gamma_a):
    """
    标准的SEIAR模型ODE函数，供内部求解器使用。
    """
    S, E, I, A, R = y
    beta_a = 0.58 * beta  # 无症状传播率是固定的相对比例
    
    force_of_infection = (beta * I + beta_a * A) * S / N
    
    dSdt = -force_of_infection
    dEdt = force_of_infection - sigma * E
    dIdt = (1 - f) * sigma * E - gamma_i * I
    dAdt = f * sigma * E - gamma_a * A
    dRdt = gamma_i * I + gamma_a * A
    
    return [dSdt, dEdt, dIdt, dAdt, dRdt]

def piecewise_seiar_model(t_data, beta1, beta2, beta3, beta4):
    """
    分段SEIAR模型，供curve_fit调用。
    这个函数接收一个时间点数组和四个beta参数，返回整个时间段的(I+A)预测曲线。
    """
    # 定义分段点 (与贝叶斯模型保持一致)
    breakpoints = [22, 71, 184]
    betas = [beta1, beta2, beta3, beta4]
    
    # 完整时间跨度
    t_full = np.arange(len(t_data))
    
    # 初始化总体的解
    solution = np.array([]).reshape(0, 5)
    
    # 初始条件
    # 从数据中获取初始E0的估计值 (一个小的非零值)
    # 我们可以将E0也作为一个拟合参数，但为了简化，这里用一个合理的估计值
    initial_exposed = 500 
    current_y0 = [N - initial_exposed, initial_exposed, 0, 0, 0]
    
    # 按分段求解ODE
    last_t = 0
    for i, bp in enumerate(breakpoints):
        # 定义当前段的时间数组
        t_segment = np.arange(last_t, bp)
        
        # 求解当前段
        sol_segment = odeint(
            seiar_model,
            y0=current_y0,
            t=t_segment,
            args=(betas[i], N, F, SIGMA, GAMMA_I, GAMMA_A)
        )
        
        # 拼接解
        solution = np.vstack([solution, sol_segment])
        
        # 更新下一段的初始条件和起始时间
        current_y0 = sol_segment[-1]
        last_t = bp

    # 求解最后一段
    t_segment_final = np.arange(last_t, len(t_full))
    sol_segment_final = odeint(
        seiar_model,
        y0=current_y0,
        t=t_segment_final,
        args=(betas[-1], N, F, SIGMA, GAMMA_I, GAMMA_A)
    )
    
    # 拼接最后一段的解
    solution = np.vstack([solution, sol_segment_final])

    # 返回模型预测的感染者总数 (I+A)
    return solution[:, 2] + solution[:, 3]


# --- 1. 加载和准备数据 ---
df_data = pd.read_csv('germany_covid_processed.csv', parse_dates=['date'])

# 定义模型参数 (基于文献)
F = 0.17
SIGMA = 1/4.5
GAMMA_I = 1/19.5
GAMMA_A = 1/17.0
N = df_data['population'].iloc[0]

# 准备拟合所需的数据 (使用前270天)
fit_duration = 270
data_to_fit = df_data.head(fit_duration)
y_data = data_to_fit['infectious_pool'].values
x_data = np.arange(len(y_data))
dates = data_to_fit['date']

# --- 2. 执行分段确定性拟合 ---
print("正在执行分段确定性拟合...")

# 为四个beta参数提供初始猜测和界限
initial_guesses = [0.2, 0.1, 0.15, 0.1]
bounds = (0, [1.0, 1.0, 1.0, 1.0]) # beta值通常在0和1之间

# 调用curve_fit
popt, pcov = curve_fit(
    f=piecewise_seiar_model,
    xdata=x_data,
    ydata=y_data,
    p0=initial_guesses,
    bounds=bounds,
    method='trf' # 'trf'适用于有界问题
)

# 提取最优参数
beta_opt = popt
print("\n拟合完成！最优参数为:")
for i, b in enumerate(beta_opt):
    print(f"  - beta_{i+1}: {b:.4f}")

# --- 3. 生成最终拟合曲线并绘图 ---
print("\n正在生成最终拟合图...")
# 使用最优参数生成最终的模型曲线
model_fit_curve = piecewise_seiar_model(x_data, *beta_opt)

# 绘图
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制真实数据
ax.plot(dates, y_data, 'o', markersize=4, color='dimgray', alpha=0.7, label='Observed Infectious Pool')

# 绘制模型拟合曲线
ax.plot(dates, model_fit_curve, color='dodgerblue', linewidth=2.5, label='Piecewise SEIAR Model Fit')

# 添加分段的垂直线和标签
breakpoints_days = [22, 71, 184]
for i, bp_day in enumerate(breakpoints_days):
    bp_date = dates.iloc[0] + pd.to_timedelta(bp_day, unit='D')
    ax.axvline(x=bp_date, linestyle='--', color='gray', linewidth=1.5)
    
# 添加参数标签
bp_dates = [dates.iloc[0]] + [dates.iloc[0] + pd.to_timedelta(d, unit='D') for d in breakpoints_days]
label_positions = [5, 35, 120, 200]
for i, pos in enumerate(label_positions):
     ax.text(dates.iloc[pos], max(y_data)*0.85, f'$\\beta_{i+1}={beta_opt[i]:.3f}$', 
             fontsize=12, backgroundcolor='white', color='blue')

# 格式化图表
ax.set_title('Piecewise Deterministic SEIAR Model Fit to German COVID-19 Data', fontsize=16, pad=20)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Estimated Infectious Population', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True)

# 格式化日期显示
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate()

# 保存图像
output_filename = 'seiar_fit_germany_piecewise.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300)

print(f"\n成功! 最终拟合图已保存为 '{output_filename}'")
plt.show() 