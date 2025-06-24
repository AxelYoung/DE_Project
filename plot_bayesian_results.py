import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 解决中文显示问题
# 尝试查找并设置中文字体
try:
    # 在macOS上，'Heiti TC' 是一个常用的中文字体
    # 在Windows上, 'SimHei' 或 'Microsoft YaHei' 更常见
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    print("绘图字体已设置为 Heiti TC。")
except Exception:
    print("警告：未能找到 Heiti TC 字体。图中中文可能无法正常显示。")

def seiar_model_scipy(y, t, beta, N, F, SIGMA, GAMMA_I, GAMMA_A):
    """
    SEIAR ODE模型函数，供scipy.integrate.odeint使用。
    """
    S, E, I, A, R = y
    beta_A = 0.58 * beta
    
    force_of_infection = (beta * I + beta_A * A) * S / N
    
    dSdt = -force_of_infection
    dEdt = force_of_infection - SIGMA * E
    dIdt = (1 - F) * SIGMA * E - GAMMA_I * I
    dAdt = F * SIGMA * E - GAMMA_A * A
    dRdt = GAMMA_I * I + GAMMA_A * A
    
    return [dSdt, dEdt, dIdt, dAdt, dRdt]

def plot_results():
    """
    主函数，用于加载数据和结果，并生成最终的拟合图。
    """
    # --- 1. 加载数据和参数 ---
    try:
        df_data = pd.read_csv('germany_covid_processed.csv', parse_dates=['date'])
        df_summary = pd.read_csv('bayesian_summary.csv', index_col=0)
    except FileNotFoundError as e:
        print(f"错误: 缺少必要文件: {e.filename}")
        print("请确保 'germany_covid_processed.csv' 和 'bayesian_summary.csv' 都在当前目录下。")
        return

    # 提取后验均值作为我们的点估计参数
    params = df_summary['mean']
    beta_1, beta_2, beta_3, beta_4 = params['beta_1'], params['beta_2'], params['beta_3'], params['beta_4']
    rho_1, rho_2, rho_3, rho_4 = params['rho_1'], params['rho_2'], params['rho_3'], params['rho_4']
    E0 = params['E0']
    
    # 模型固定参数
    F = 0.17
    SIGMA = 1/4.5
    GAMMA_I = 1/19.5
    GAMMA_A = 1/17.0
    N = df_data['population'].iloc[0]

    # 准备拟合所需的数据
    fit_duration = 270
    dates = df_data['date'].iloc[:fit_duration]
    i_data = df_data['infectious_pool'].values[:fit_duration]

    # --- 2. 使用提取的参数分段求解ODE ---
    
    # 定义分段点和时间数组
    breakpoints = [22, 71, 184]
    t_segs = [
        np.arange(0, breakpoints[0]),
        np.arange(breakpoints[0], breakpoints[1]),
        np.arange(breakpoints[1], breakpoints[2]),
        np.arange(breakpoints[2], fit_duration)
    ]
    betas = [beta_1, beta_2, beta_3, beta_4]
    
    # 初始条件
    y0 = [N - E0, E0, 0, 0, 0]
    
    all_solutions = []
    current_y0 = y0
    
    # 循环求解每个分段
    for i, t_seg in enumerate(t_segs):
        sol = odeint(
            seiar_model_scipy,
            y0=current_y0,
            t=t_seg,
            args=(betas[i], N, F, SIGMA, GAMMA_I, GAMMA_A)
        )
        all_solutions.append(sol)
        # 下一段的初始条件是当前段的最后状态
        current_y0 = sol[-1]
        
    # 合并所有解
    model_output = np.concatenate(all_solutions, axis=0)
    I_model = model_output[:, 2]
    A_model = model_output[:, 3]
    
    # --- 3. 计算可观测病例数 (mu) ---
    t_full = np.arange(fit_duration)
    # 根据分段点构建分段的rho值
    # 使用更直接的布尔索引来构建分段的rho值，以提高代码的清晰度和稳健性
    rho_t = np.zeros(fit_duration)
    rho_t[t_full < breakpoints[0]] = rho_1
    rho_t[(t_full >= breakpoints[0]) & (t_full < breakpoints[1])] = rho_2
    rho_t[(t_full >= breakpoints[1]) & (t_full < breakpoints[2])] = rho_3
    rho_t[t_full >= breakpoints[2]] = rho_4
                         
    # 这就是模型预测的、与真实数据可比的每日感染人数
    mu_predicted = rho_t * (I_model + A_model)

    # --- 4. 绘图 ---
    print("正在生成最终拟合图...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制真实数据
    ax.plot(dates, i_data, 'o', markersize=4, color='dimgray', alpha=0.7, label='Observed Data (7-day smoothed)')
    
    # 绘制模型拟合曲线
    ax.plot(dates, mu_predicted, color='crimson', linewidth=2.5, label='SEIAR Model Prediction (Mean Posterior)')
    
    # 添加分段的垂直线和标签
    for bp_day in breakpoints:
        bp_date = dates.iloc[0] + pd.to_timedelta(bp_day, unit='D')
        ax.axvline(x=bp_date, linestyle='--', color='gray', linewidth=1.5)

    ax.text(dates.iloc[5], max(i_data)*0.8, f'$\\beta_1={beta_1:.3f}$\n$\\rho_1={rho_1:.3f}$', fontsize=12, backgroundcolor='white')
    ax.text(dates.iloc[35], max(i_data)*0.8, f'$\\beta_2={beta_2:.3f}$\n$\\rho_2={rho_2:.3f}$', fontsize=12, backgroundcolor='white')
    ax.text(dates.iloc[120], max(i_data)*0.8, f'$\\beta_3={beta_3:.3f}$\n$\\rho_3={rho_3:.3f}$', fontsize=12, backgroundcolor='white')
    ax.text(dates.iloc[200], max(i_data)*0.6, f'$\\beta_4={beta_4:.3f}$\n$\\rho_4={rho_4:.3f}$', fontsize=12, backgroundcolor='white')

    # 格式化图表
    ax.set_title('Bayesian SEIAR Model Fit to German COVID-19 Data (4 Segments)', fontsize=18, pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Estimated Infectious Population', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # 格式化日期显示
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('bayesian_fit_to_data.png', dpi=300)
    print("Success! Final fit plot saved as 'bayesian_fit_to_data.png'")
    plt.show()


if __name__ == "__main__":
    plot_results() 