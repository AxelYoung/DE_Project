import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- 第一步：导入库并加载数据 ---

# 加载我们刚刚处理好的德国疫情数据
try:
    df = pd.read_csv('germany_covid_processed.csv')
    # 核心拟合数据：估算的现存总感染人数
    i_data = df['infectious_pool'].values
    t_data = np.arange(len(i_data))
    # 获取总人口 N
    N = df['population'].iloc[0]
except FileNotFoundError:
    print("错误：数据文件 'germany_covid_processed.csv' 未找到。")
    print("请确保您已经运行了 data_processing.py 来生成此文件。")
    exit()

# --- 全局参数和初始条件 ---
# 固定参数 (来自我们报告Part 2中的文献)
F = 0.17       # 无症状比例
SIGMA = 1/4.5    # 潜伏期转换率 (1/天)
GAMMA_I = 1/19.5 # 有症状者恢复率 (1/天)
GAMMA_A = 1/17.0   # 无症状者恢复率 (1/天)

# 初始条件
I0 = 0      # 假设在t=0时，还没有被诊断出的有症状者
A0 = 0      # 假设在t=0时，还没有无症状者
E0 = 1      # 关键：我们假设在t=0时，有1个暴露者进入人群，作为疫情的"火种"
R0 = 0
S0 = N - I0 - E0 - A0 - R0
Y0 = (S0, E0, I0, A0, R0) # 初始状态元组

# --- 第二步：定义SEIAR模型函数 ---

# 定义SEIAR模型的微分方程
# 我们将所有参数作为输入，以增加函数的通用性
def seiar_model(y, t, N, beta_I, f_param, sigma_param, gamma_i_param, gamma_a_param):
    """
    SEIAR模型的ODE系统。
    beta_I现在是一个参数，而不是全局变量。
    """
    S, E, I, A, R = y
    
    # 根据文献中的关系计算beta_A
    beta_A = 0.58 * beta_I
    
    # 微分方程
    dSdt = -beta_I * S * I / N - beta_A * S * A / N
    dEdt = beta_I * S * I / N + beta_A * S * A / N - sigma_param * E
    dIdt = (1 - f_param) * sigma_param * E - gamma_i_param * I
    dAdt = f_param * sigma_param * E - gamma_a_param * A
    dRdt = gamma_i_param * I + gamma_a_param * A
    
    return dSdt, dEdt, dIdt, dAdt, dRdt

# --- 第三步：创建全新的、功能更强大的"求解器"函数 ---

def fit_odeint_piecewise(t, beta_pre, beta_post, rho, E0_fit):
    """
    一个先进的求解器，用于curve_fit。
    它使用分段变化的beta，并同时拟合观测比例rho和初始暴露人数E0。
    """
    t_switch = 22  # 德国开始全国性封锁的大致日期 (2020-03-22)，对应数据中的第22天

    # 使用拟合参数更新初始条件
    # E0是待拟合的，I0, A0, R0从0开始
    y0_fit = (N - E0_fit, E0_fit, 0, 0, 0)

    # 分割时间数组为"封锁前"和"封锁后"
    t_pre = t[t < t_switch]
    t_post = t[t >= t_switch]

    # 创建一个空的numpy数组来存储合并后的感染人数
    infected_combined = np.array([])
    
    # 求解第一阶段 (封锁前)
    if len(t_pre) > 0:
        # 使用封锁前的beta进行求解
        res_pre = odeint(seiar_model, y0_fit, t_pre, args=(N, beta_pre, F, SIGMA, GAMMA_I, GAMMA_A))
        # 提取I+A部分
        infected_pre = res_pre[:, 2] + res_pre[:, 3]
        infected_combined = np.concatenate((infected_combined, infected_pre))
        # 第二阶段的初始条件是第一阶段的终点
        y0_post = res_pre[-1]
    else:
        # 如果拟合窗口完全在封锁后，则第二阶段的初始条件就是我们设定的y0_fit
        y0_post = y0_fit

    # 求解第二阶段 (封锁后)
    if len(t_post) > 0:
        # 使用封锁后的beta进行求解
        res_post = odeint(seiar_model, y0_post, t_post, args=(N, beta_post, F, SIGMA, GAMMA_I, GAMMA_A))
        # 提取I+A部分
        infected_post = res_post[:, 2] + res_post[:, 3]
        infected_combined = np.concatenate((infected_combined, infected_post))

    # 返回由观测比例rho缩放后的模型预测值
    return rho * infected_combined

# --- 第四步：执行拟合并可视化 ---

# 定义拟合窗口
fit_duration = 180
t_fit = t_data[:fit_duration]
i_fit = i_data[:fit_duration]

# 为4个待拟合参数提供初始猜测值和边界
# 参数顺序: [beta_pre, beta_post, rho, E0_fit]
initial_guesses = [0.6, 0.1, 0.2, 100]
bounds = (
    [0.1, 0.01, 0.01, 1],      # 下界
    [2.0, 1.0, 1.0, 50000]     # 上界
)

# 执行拟合
try:
    popt, pcov = curve_fit(
        fit_odeint_piecewise,
        t_fit, 
        i_fit, 
        p0=initial_guesses, 
        bounds=bounds,
        maxfev=3000  # 增加最大迭代次数以帮助收敛
    )
    
    # 提取最优参数
    beta_pre_opt, beta_post_opt, rho_opt, E0_fit_opt = popt
    
    print("拟合成功！")
    print(f"封锁前传播率 (beta_pre): {beta_pre_opt:.4f}")
    print(f"封锁后传播率 (beta_post): {beta_post_opt:.4f}")
    print(f"病例观测比例 (rho): {rho_opt:.4f}")
    print(f"初始暴露人数 (E0_fit): {E0_fit_opt:.2f}")

    # --- 可视化 ---
    
    # 使用最优参数，重新运行求解器以生成最终的拟合曲线
    fitted_curve = fit_odeint_piecewise(t_fit, *popt)
    
    # 创建图形
    plt.figure(figsize=(14, 9))
    # 绘制所有真实数据点（整个时间序列）
    plt.plot(t_data, i_data, 'o', label='Real Data (Estimated Total Infectious)', markersize=4, alpha=0.5)
    # 绘制在拟合窗口内的拟合曲线
    plt.plot(t_fit, fitted_curve, 'r-', label='Fitted SEIAR Model', linewidth=3)
    
    # 使用垂直线标记封锁日期
    t_switch = 22
    plt.axvline(x=t_switch, color='k', linestyle='--', label=f'Lockdown Start (Day {t_switch})')
    
    # 在图表上注释最优参数值
    param_text = (
        f'Fitted Parameters:\n'
        f'$β_{{pre}}$ = {beta_pre_opt:.3f}\n'
        f'$β_{{post}}$ = {beta_post_opt:.3f}\n'
        f'$ρ$ (rho) = {rho_opt:.3f}\n'
        f'$E_0$ = {E0_fit_opt:.0f}'
    )
    plt.text(0.65, 0.7, param_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    # 格式化图表
    plt.title('SEIAR Model with Piecewise Beta Fit to German COVID-19 Data', fontsize=16)
    plt.xlabel('Days Since 2020-03-01', fontsize=12)
    plt.ylabel('Infectious Population', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=len(t_data))
    
    # 保存图像
    plt.savefig('seiar_fit_germany_piecewise.png', dpi=300)
    print("\n拟合结果图已保存为 'seiar_fit_germany_piecewise.png'")
    
    # 显示图像
    plt.show()

except RuntimeError as e:
    print(f"错误：无法完成拟合。错误信息: {e}")
    print("这可能是由于初始猜测值或边界不佳。请尝试调整 `initial_guesses` 和 `bounds`。")

except Exception as e:
    print(f"发生了一个未知错误: {e}")
