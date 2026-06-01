import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

# ==================== 设置 matplotlib 支持中文 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 融合模型定义（带缩放） --------------------------
def model_predict_scaled(theta_scaled, x_norm, e_mag, x_min, x_max):
    """
    使用归一化位置的多项式-谐波模型
    theta_scaled = [a0_s, a1_s, a2_s, a3_s, k, b1, c1, b2, c2]
    x_norm  = 2*(x - x_min)/(x_max - x_min) - 1   (映射到[-1,1])
    返回预测误差
    """
    a0_s, a1_s, a2_s, a3_s, k, b1, c1, b2, c2 = theta_scaled
    poly = a0_s + a1_s * x_norm + a2_s * x_norm ** 2 + a3_s * x_norm ** 3
    linear = k * e_mag
    # 注意：谐波项使用原始位置 x (mm) 以保持物理周期性 (周期2mm)
    harm = (b1 * np.cos(np.pi * x_norm) + c1 * np.sin(np.pi * x_norm) +
            b2 * np.cos(2 * np.pi * x_norm) + c2 * np.sin(2 * np.pi * x_norm))
    return poly + linear + harm


def residuals_scaled(theta_scaled, x_norm, e_mag, y_true, x_min, x_max):
    return y_true - model_predict_scaled(theta_scaled, x_norm, e_mag, x_min, x_max)


# -------------------------- 主程序 ------------------------------------
def main():
    # 1. 读取数据
    df = pd.read_csv('data_modify.csv')
    x_raw = df.iloc[:, 0].values.astype(float)  # 原始位置 0 ~ -240 mm
    n_conditions = 9
    cond_names = [f'工况{i}' for i in range(1, n_conditions + 1)]

    # 汇总所有样本
    all_x = []  # 原始位置
    all_emag = []
    all_y = []
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        all_x.extend(x_raw)
        all_emag.extend(emag_i)
        all_y.extend(y_i)
    all_x = np.array(all_x)
    all_emag = np.array(all_emag)
    all_y = np.array(all_y)

    # 2. 特征缩放：将位置映射到 [-1, 1]
    x_min, x_max = all_x.min(), all_x.max()
    all_x_norm = 2 * (all_x - x_min) / (x_max - x_min) - 1

    # 3. 参数初值估计
    # 3.1 线性耦合系数 k (用全量数据线性回归)
    lr = LinearRegression()
    lr.fit(all_emag.reshape(-1, 1), all_y)
    k_init = lr.coef_[0]
    print(f"初始线性回归: k = {k_init:.6f}")

    # 3.2 多项式系数初值：先用三阶多项式拟合位置与实测误差（不引入磁栅和谐波）
    # 注意：这里使用归一化位置拟合
    poly_coefs = np.polyfit(all_x_norm, all_y, 3)  # 返回从高次到低次: a3, a2, a1, a0
    a0_init, a1_init, a2_init, a3_init = poly_coefs[3], poly_coefs[2], poly_coefs[1], poly_coefs[0]

    # 3.3 谐波系数初值：先计算残差（用多项式+线性部分），再对残差做傅里叶拟合
    temp_pred = (
                            a0_init + a1_init * all_x_norm + a2_init * all_x_norm ** 2 + a3_init * all_x_norm ** 3) + k_init * all_emag
    temp_res = all_y - temp_pred
    # 用二阶傅里叶级数拟合残差（周期基于归一化位置，实际对应原始位置周期2mm）
    from scipy.optimize import curve_fit
    def fourier2(x_norm, b1, c1, b2, c2):
        return b1 * np.cos(np.pi * x_norm) + c1 * np.sin(np.pi * x_norm) + b2 * np.cos(
            2 * np.pi * x_norm) + c2 * np.sin(2 * np.pi * x_norm)

    try:
        popt, _ = curve_fit(fourier2, all_x_norm, temp_res, p0=[0, 0, 0, 0])
        b1_init, c1_init, b2_init, c2_init = popt
    except:
        b1_init = c1_init = b2_init = c2_init = 0.0

    theta0_scaled = np.array([a0_init, a1_init, a2_init, a3_init, k_init,
                              b1_init, c1_init, b2_init, c2_init])
    print("初始参数(缩放后):", theta0_scaled)

    # 4. 使用 scipy.least_squares 优化 (基于LM)
    def obj_func(theta, x_norm, e_mag, y_true):
        return residuals_scaled(theta, x_norm, e_mag, y_true, x_min, x_max)

    result = least_squares(obj_func, theta0_scaled, args=(all_x_norm, all_emag, all_y),
                           method='lm', max_nfev=500, xtol=1e-10, ftol=1e-10)
    theta_opt_scaled = result.x
    print("\n优化收敛信息:", result.message)
    print("优化后参数(缩放后):", theta_opt_scaled)

    # 5. 将缩放后的参数转换为原始位置下的公式系数（用于输出定型公式）
    # 原始模型: poly_orig = A0 + A1*x + A2*x^2 + A3*x^3
    # 变换关系: x_norm = 2*(x - x_min)/(x_max - x_min) - 1 = alpha*x + beta
    # 其中 alpha = 2/(x_max - x_min), beta = -2*x_min/(x_max - x_min) - 1
    alpha = 2.0 / (x_max - x_min)
    beta = -2.0 * x_min / (x_max - x_min) - 1.0
    # poly_opt_scaled = a0_s + a1_s*x_norm + a2_s*x_norm^2 + a3_s*x_norm^3
    # 将 x_norm = alpha*x + beta 代入，展开得到原始多项式系数
    a0_s, a1_s, a2_s, a3_s = theta_opt_scaled[0:4]
    # 计算原始系数 (A3, A2, A1, A0)
    # 使用符号展开: (alpha*x + beta)^3 = alpha^3 x^3 + 3 alpha^2 beta x^2 + 3 alpha beta^2 x + beta^3
    A3 = a3_s * alpha ** 3
    A2 = a3_s * 3 * alpha ** 2 * beta + a2_s * alpha ** 2
    A1 = a3_s * 3 * alpha * beta ** 2 + a2_s * 2 * alpha * beta + a1_s * alpha
    A0 = a3_s * beta ** 3 + a2_s * beta ** 2 + a1_s * beta + a0_s
    # 线性系数 k 直接保留
    k_opt = theta_opt_scaled[4]

    # 谐波系数保持不变（因为使用原始位置，但注意：我们的谐波用了x_norm，应该用原始位置？）
    # 原模型期望谐波用原始位置 x (mm) 且周期为2mm，即 cos(pi*x) 等。
    # 但我们在优化时使用了 x_norm，因此需要将谐波参数转换回原始位置表达。
    # 实际上，若谐波项使用 cos(pi * x_norm)，而 x_norm = alpha*x + beta，则周期变化，不再是固定2mm。
    # 为了严格符合文档公式（谐波用原始位置），我们应直接使用原始位置 x 计算谐波项，不缩放。
    # 因此，下面重新构建模型：仅多项式部分缩放，谐波用原始x。
    # 为避免混乱，我们采用混合方式：多项式输入缩放位置，谐波输入原始位置。
    # 修改模型定义如下：

    # ------------------ 重新定义最终模型（多项式用归一化，谐波用原始x） ------------------
    def model_final(theta, x_raw, e_mag, x_min, x_max):
        a0_s, a1_s, a2_s, a3_s, k, b1, c1, b2, c2 = theta
        x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1
        poly = a0_s + a1_s * x_norm + a2_s * x_norm ** 2 + a3_s * x_norm ** 3
        linear = k * e_mag
        harm = (b1 * np.cos(np.pi * x_raw) + c1 * np.sin(np.pi * x_raw) +
                b2 * np.cos(2 * np.pi * x_raw) + c2 * np.sin(2 * np.pi * x_raw))
        return poly + linear + harm

    # 使用同样的优化结果 theta_opt_scaled 进行预测，但谐波参数直接使用（因为它们对应原始x）
    # 注意：我们优化时实际使用的是 x_norm 的谐波，所以目前 b1,c1,b2,c2 是基于 x_norm 的，需要转换。
    # 但转换复杂，且为了保持与文档完全一致，我们应该直接优化使用原始x的谐波项。
    # 因此，我们重新进行一次优化，使用正确的模型（多项式用归一化x_norm，谐波用原始x）。

    print("\n重新使用正确模型（多项式归一化，谐波原始位置）进行优化...")

    # 定义模型函数
    def model_correct(theta, x_raw, e_mag, x_min, x_max):
        a0_s, a1_s, a2_s, a3_s, k, b1, c1, b2, c2 = theta
        x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1
        poly = a0_s + a1_s * x_norm + a2_s * x_norm ** 2 + a3_s * x_norm ** 3
        linear = k * e_mag
        harm = (b1 * np.cos(np.pi * x_raw) + c1 * np.sin(np.pi * x_raw) +
                b2 * np.cos(2 * np.pi * x_raw) + c2 * np.sin(2 * np.pi * x_raw))
        return poly + linear + harm

    def model_correct_lyw(theta, x_raw, e_mag, x_min, x_max):
        a0_s, a1_s, a2_s, a3_s, k, b1, c1, b2, c2 = theta
        x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1
        poly = A0 + A1 * x_raw + A2 * x_raw ** 2 + A3 * x_raw ** 3
        linear = k * e_mag
        harm = (b1 * np.cos(np.pi * x_raw) + c1 * np.sin(np.pi * x_raw) +
                b2 * np.cos(2 * np.pi * x_raw) + c2 * np.sin(2 * np.pi * x_raw))
        return poly + linear + harm

    def residuals_correct(theta, x_raw, e_mag, y_true, x_min, x_max):
        return y_true - model_correct(theta, x_raw, e_mag, x_min, x_max)

    # 初值：多项式部分沿用之前（但注意谐波初值应为0，因为之前拟合的是基于x_norm的）
    theta0_correct = np.array([a0_init, a1_init, a2_init, a3_init, k_init,
                               0.0, 0.0, 0.0, 0.0])
    result_correct = least_squares(lambda theta: residuals_correct(theta, all_x, all_emag, all_y, x_min, x_max),
                                   theta0_correct, method='lm', max_nfev=500, xtol=1e-10, ftol=1e-10)
    theta_opt_correct = result_correct.x
    print("优化收敛信息:", result_correct.message)

    # 输出最终公式的原始多项式系数（展开后）
    a0_s, a1_s, a2_s, a3_s = theta_opt_correct[0:4]
    A3 = a3_s * alpha ** 3
    A2 = a3_s * 3 * alpha ** 2 * beta + a2_s * alpha ** 2
    A1 = a3_s * 3 * alpha * beta ** 2 + a2_s * 2 * alpha * beta + a1_s * alpha
    A0 = a3_s * beta ** 3 + a2_s * beta ** 2 + a1_s * beta + a0_s
    k_opt = theta_opt_correct[4]
    b1, c1, b2, c2 = theta_opt_correct[5:9]

    print("\n最终模型参数（原始位置多项式系数）:")
    print(f"多项式: {A0:.8f} + {A1:.8e}*x + {A2:.8e}*x^2 + {A3:.8e}*x^3")
    print(f"线性耦合: k = {k_opt:.8f}")
    print(f"谐波: b1={b1:.8f}, c1={c1:.8f}, b2={b2:.8f}, c2={c2:.8f}")

    # 6. 计算全域性能及每个工况的指标
    y_pred_all = model_correct_lyw(theta_opt_correct, all_x, all_emag, x_min, x_max)
    res_all = all_y - y_pred_all
    ss_res = np.sum(res_all ** 2)
    ss_tot = np.sum((all_y - np.mean(all_y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(res_all ** 2))
    max_res = np.max(np.abs(res_all))
    # print(f"\n全域性能指标-融合模型: R²={r2:.6f}, RMSE={rmse:.5f} mm, MaxRes={max_res:.5f} mm")

    print("\n========== 全域性能指标-融合模型 ==========")
    print(f"决定系数 R²      = {r2:.6f}")
    print(f"均方根误差 RMSE  = {rmse:.5f} mm")
    print(f"最大残差 MaxRes  = {max_res:.5f} mm\n")

    # 7. 生成 dataAnalysis.csv（每个工况6列）
    result_df = pd.DataFrame()
    result_df['Z轴位置(mm)'] = x_raw
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        y_pred_i = model_correct(theta_opt_correct, x_raw, emag_i, x_min, x_max)
        res_i = y_i - y_pred_i
        ss_res_i = np.sum(res_i ** 2)
        ss_tot_i = np.sum((y_i - np.mean(y_i)) ** 2)
        r2_i = 1 - ss_res_i / ss_tot_i if ss_tot_i != 0 else 0
        rmse_i = np.sqrt(np.mean(res_i ** 2))

        result_df[f'工况{i + 1}磁栅误差(mm)'] = emag_i
        result_df[f'工况{i + 1}激光实测(mm)'] = y_i
        result_df[f'工况{i + 1}模型拟合误差(mm)'] = y_pred_i
        result_df[f'工况{i + 1}残差(mm)'] = res_i
        result_df[f'工况{i + 1}_R²'] = np.full(len(x_raw), r2_i)
        result_df[f'工况{i + 1}_RMSE(mm)'] = np.full(len(x_raw), rmse_i)

    ordered_cols = ['Z轴位置(mm)']
    for i in range(1, n_conditions + 1):
        ordered_cols.extend([
            f'工况{i}磁栅误差(mm)',
            f'工况{i}激光实测(mm)',
            f'工况{i}模型拟合误差(mm)',
            f'工况{i}残差(mm)',
            f'工况{i}_R²',
            f'工况{i}_RMSE(mm)'
        ])
    result_df = result_df[ordered_cols]
    result_df.to_csv('dataFusionAnalysis.csv', index=False, float_format='%.6f')
    print("\n已生成 dataFusionAnalysis.csv，包含融合模型各工况分析结果。")

    # 8. 输出各工况性能表
    print("\n各工况独立性能指标（融合模型）:")
    print(f"{'工况':<6} {'R²':<12} {'RMSE (mm)':<14} {'MaxRes (mm)':<12}")
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        y_pred_i = model_correct(theta_opt_correct, x_raw, emag_i, x_min, x_max)
        res_i = y_i - y_pred_i
        ss_res_i = np.sum(res_i ** 2)
        ss_tot_i = np.sum((y_i - np.mean(y_i)) ** 2)
        r2_i = 1 - ss_res_i / ss_tot_i if ss_tot_i != 0 else 0
        rmse_i = np.sqrt(np.mean(res_i ** 2))
        max_i = np.max(np.abs(res_i))
        print(f"{cond_names[i]:<6} {r2_i:<12.6f} {rmse_i:<14.6f} {max_i:<12.6f}")

    # 9. 绘制各工况曲线（含残差）
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        y_pred_i = model_correct(theta_opt_correct, x_raw, emag_i, x_min, x_max)
        res_i = y_i - y_pred_i
        ax = axes[i]
        ax.plot(x_raw, emag_i, '--', color='gray', linewidth=1.5, label='磁栅误差')
        ax.plot(x_raw, y_i, '-', color='blue', linewidth=2, label='激光实测误差')
        ax.plot(x_raw, y_pred_i, '-', color='red', linewidth=2, label='融合模型拟合')
        ax.plot(x_raw, res_i, '-', color='green', linewidth=1.2, label='残差')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_title(f'{cond_names[i]} (融合模型)', fontsize=12)
        ax.set_xlabel('Z 轴位置 (mm)', fontsize=10)
        ax.set_ylabel('误差 / 残差 (mm)', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('datas/fusion_fitting_with_residuals.png', dpi=300)
    plt.show()




if __name__ == "__main__":
    main()