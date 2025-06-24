import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph import Apply
import matplotlib.pyplot as plt

print(f"PyMC Version: {pm.__version__}")
print(f"ArviZ Version: {az.__version__}")

# --- 1. 数据加载和准备 ---

try:
    df = pd.read_csv('germany_covid_processed.csv', parse_dates=['date'])
    # 我们将拟合整个第一波和第二波初期 (约9个月)
    fit_duration = 270 
    i_data = df['infectious_pool'].values[:fit_duration]
    t_data = np.arange(fit_duration)
    N = df['population'].iloc[0]
except FileNotFoundError:
    print("错误：数据文件 'germany_covid_processed.csv' 未找到。")
    exit()

# --- 2. 定义ODE系统 (PyTensor版本) ---

# 固定参数
F = 0.17       # 无症状比例
SIGMA = 1/4.5    # 潜伏期转换率 (1/天)
GAMMA_I = 1/19.5 # 有症状者恢复率 (1/天)
GAMMA_A = 1/17.0   # 无症状者恢复率 (1/天)

# 这是PyMC v5+ 中集成外部ODE求解器（如scipy）的标准方式
# 我们需要将scipy的odeint封装成一个PyTensor的Op
class OdeintOp(Op):
    def __init__(self, func, t):
        self._func = func
        self._t = t

    def make_node(self, y0, *theta):
        # 输入：y0是初始条件，theta是ODE的参数
        y0 = pt.as_tensor_variable(y0)
        theta = [pt.as_tensor_variable(arg) for arg in theta]
        # 输出：ODE的解
        out = pt.tensor(dtype=y0.dtype, broadcastable=(False, False))
        return Apply(self, [y0] + theta, [out])

    def perform(self, node, inputs, output_storage):
        from scipy.integrate import odeint
        y0 = inputs[0]
        theta = inputs[1:]
        # odeint需要一个只接受(y, t, *args)的函数
        def grad_func(y, t, *args):
            return self._func(y, t, *args)
        # 求解ODE
        result = odeint(grad_func, y0, self._t, args=tuple(theta))
        output_storage[0][0] = result.astype(y0.dtype)

# SEIAR模型的微分方程，现在可以被OdeintOp使用
def seiar_model_scipy(y, t, beta):
    S, E, I, A, R = y
    # 根据文献，无症状者的传播率是 symptomatic 的 0.58 倍
    beta_A = 0.58 * beta
    
    # 正确的感染力计算 (force of infection)
    force_of_infection = (beta * I + beta_A * A) * S / N
    
    dSdt = -force_of_infection
    dEdt = force_of_infection - SIGMA * E
    dIdt = (1 - F) * SIGMA * E - GAMMA_I * I
    dAdt = F * SIGMA * E - GAMMA_A * A
    dRdt = GAMMA_I * I + GAMMA_A * A
    
    return [dSdt, dEdt, dIdt, dAdt, dRdt]


# --- 3. 构建PyMC贝叶斯模型 ---

# 定义分段的时间点 (根据您的建议)
# 新增一个断点在9月初 (大约第184天)
# 段1: 0-21, 段2: 22-70, 段3: 71-183, 段4: 184-269
breakpoints = [22, 71, 184]
t_segs = [
    np.arange(0, breakpoints[0]),
    np.arange(breakpoints[0], breakpoints[1]),
    np.arange(breakpoints[1], breakpoints[2]),
    np.arange(breakpoints[2], fit_duration)
]

with pm.Model() as model:
    # --- 先验分布 (Priors) ---
    
    # 1. 传播率 beta (分四段)
    # beta_1 使用基于R0知识的强先验
    beta_1 = pm.Gamma('beta_1', alpha=10, beta=50) # 均值0.2, R0约3.5, 强约束
    # beta_2, beta_3, beta_4 的先验较为宽松
    beta_2 = pm.Gamma('beta_2', alpha=2, beta=4) # 均值0.5
    beta_3 = pm.Gamma('beta_3', alpha=2, beta=4) # 均值0.5
    beta_4 = pm.Gamma('beta_4', alpha=2, beta=4) # 新增段的先验
    betas = [beta_1, beta_2, beta_3, beta_4]
    
    # 2. 观测比例 rho (分四段)
    rho_1 = pm.Beta('rho_1', alpha=2, beta=8) # 先验认为比例较低
    rho_2 = pm.Beta('rho_2', alpha=4, beta=6) # 封锁后检测能力提升，比例可能稍高
    rho_3 = pm.Beta('rho_3', alpha=4, beta=6)
    rho_4 = pm.Beta('rho_4', alpha=4, beta=6) # 新增段的先验
    
    # 使用pytensor.switch实现分段rho (rho本身不在ODE内，是在观测时使用)
    rho_t = pt.switch(t_data < breakpoints[0], rho_1,
                      pt.switch(t_data < breakpoints[1], rho_2,
                                pt.switch(t_data < breakpoints[2], rho_3, rho_4)))
                      
    # 3. 初始暴露人数 E0 - 使用Gamma先验提高稳定性
    E0 = pm.Gamma('E0', alpha=2, beta=1/250) # 均值500
    
    # 4. 观测误差 (用于负二项分布)
    # alpha越大，分布越接近泊松分布(方差=均值)；alpha越小，方差越大(过度离散)
    noise_alpha = pm.Exponential('noise_alpha', 1.0)

    # --- ODE分段求解 ---
    y0 = pt.stack([N - E0, E0, 0, 0, 0])
    
    all_sols = []
    current_y0 = y0

    for i, t_seg in enumerate(t_segs):
        # 为每个分段创建一个ODE求解器实例
        ode_solver_seg = OdeintOp(seiar_model_scipy, t_seg)
        # 使用对应的beta求解
        sol_seg = ode_solver_seg(current_y0, betas[i])
        
        # 将当前段的解添加到列表中
        all_sols.append(sol_seg)
        
        # 更新下一段的初始条件
        # 我们需要确保在Theano/PyTensor的计算图中这是一个有效的操作
        # sol_seg[-1]会取出最后一个时间点的状态
        current_y0 = sol_seg[-1]

    # 将所有分段的解连接起来
    ode_solutions = pt.concatenate(all_sols, axis=0)
    
    # 提取模型预测的I和A
    I_model = ode_solutions[:, 2]
    A_model = ode_solutions[:, 3]
    
    # 计算模型预测的"可观测"病例数
    # 这里我们使用分段的rho_t来调整输出
    mu = rho_t * (I_model + A_model)

    # --- 似然 (Likelihood) ---
    
    # 我们使用负二项分布，它比泊松更能处理疫情数据的"过度离散"
    pm.NegativeBinomial('obs', mu=mu + 1e-6, alpha=noise_alpha, observed=i_data)


# --- 4. 运行MCMC采样 ---

if __name__ == "__main__":
    with model:
        print("开始MCMC采样... 这可能需要几分钟...")
        # JAX采样器通常更快
        try:
            idata = pm.sample(2000, tune=1000, chains=4, cores=1, target_accept=0.9, nuts_sampler="numpyro")
        except Exception:
            print("Numpyro采样失败，切换到默认NUTS采样器...")
            idata = pm.sample(2000, tune=1000, chains=4, cores=1, target_accept=0.9)

    # --- 5. 结果分析与可视化 ---
    
    # 打印参数的后验统计摘要
    summary = az.summary(idata, var_names=['beta_1', 'beta_2', 'beta_3', 'beta_4', 'rho_1', 'rho_2', 'rho_3', 'rho_4', 'E0', 'noise_alpha'])
    print("\n--- 参数后验分布摘要 ---")
    print(summary)
    
    # 保存摘要到CSV
    summary.to_csv("bayesian_summary.csv")
    print("\n摘要已保存到 'bayesian_summary.csv'")

    # 绘制后验分布图
    try:
        print("\n正在生成后验分布图...")
        az.plot_posterior(idata, var_names=['beta_1', 'beta_2', 'beta_3', 'beta_4', 'rho_1', 'rho_2', 'rho_3', 'rho_4', 'E0'])
        plt.tight_layout()
        plt.savefig("bayesian_posterior_plots.png", dpi=300)
        print("后验分布图已保存为 'bayesian_posterior_plots.png'")
        plt.show()
    except Exception as e:
        print(f"无法生成后验分布图: {e}")

    # 绘制轨迹图 (用于诊断)
    try:
        print("\n正在生成轨迹图...")
        az.plot_trace(idata, var_names=['beta_1', 'beta_2', 'beta_3', 'beta_4', 'rho_1', 'rho_2', 'rho_3', 'rho_4', 'E0'])
        plt.tight_layout()
        plt.savefig("bayesian_trace_plots.png", dpi=300)
        print("轨迹图已保存为 'bayesian_trace_plots.png'")
        plt.show()
    except Exception as e:
        print(f"无法生成轨迹图: {e}") 