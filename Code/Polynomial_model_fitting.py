import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

# ==================== 设置 matplotlib 支持中文 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 简化模型：多项式 + 线性耦合（无谐波）--------------------------
def model_predict(theta, x_raw, e_mag, x_min, x_max):
    """
    theta = [a0_s, a1_s, a2_s, a3_s, k]   (多项式系数使用归一化位置)
    多项式输入归一化位置，线性耦合直接使用 e_mag
    """
    a0_s, a1_s, a2_s, a3_s, k = theta
    x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1   # 映射到 [-1,1]
    poly = a0_s + a1_s * x_norm + a2_s * x_norm**2 + a3_s * x_norm**3
    linear = k * e_mag
    return poly + linear


def residuals(theta, x_raw, e_mag, y_true, x_min, x_max):
    return y_true - model_predict(theta, x_raw, e_mag, x_min, x_max)


# ---------------------------- 主程序 ------------------------------------
def main():
    # 1. 读取数据
    df = pd.read_csv('data_modify.csv')          # 请根据实际文件名调整
    x_raw = df.iloc[:, 0].values.astype(float)   # Z轴位置 0 ~ -240 mm
    n_conditions = 9
    cond_names = [f'工况{i}' for i in range(1, n_conditions + 1)]

    # 汇总所有样本
    all_x = []
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

    # 2. 特征缩放参数
    x_min, x_max = all_x.min(), all_x.max()
    all_x_norm = 2 * (all_x - x_min) / (x_max - x_min) - 1

    # 3. 参数初值估计
    # 3.1 线性耦合系数 k (全量线性回归)
    lr = LinearRegression()
    lr.fit(all_emag.reshape(-1, 1), all_y)
    k_init = lr.coef_[0]
    print(f"初始线性回归: k = {k_init:.6f}")

    # 3.2 多项式系数初值：用三阶多项式拟合归一化位置与实测误差
    poly_coefs = np.polyfit(all_x_norm, all_y, 3)   # 返回 a3, a2, a1, a0
    a0_init, a1_init, a2_init, a3_init = poly_coefs[3], poly_coefs[2], poly_coefs[1], poly_coefs[0]

    theta0 = np.array([a0_init, a1_init, a2_init, a3_init, k_init])
    print("初始参数 (多项式系数为归一化形式):", theta0)

    # 4. 使用 scipy.least_squares 优化 (LM算法)
    def obj_func(theta, x_norm, e_mag, y_true):
        # 注意：这里为了方便，重新构造预测函数（使用归一化 x_norm）
        a0_s, a1_s, a2_s, a3_s, k = theta
        poly = a0_s + a1_s * x_norm + a2_s * x_norm**2 + a3_s * x_norm**3
        return y_true - (poly + k * e_mag)

    result = least_squares(obj_func, theta0, args=(all_x_norm, all_emag, all_y),
                           method='lm', max_nfev=500, xtol=1e-10, ftol=1e-10)
    theta_opt = result.x
    print("\n优化收敛信息:", result.message)
    print("优化后参数 (归一化多项式系数):", theta_opt)

    # 5. 将归一化多项式系数转换为原始位置 x (mm) 的多项式系数
    alpha = 2.0 / (x_max - x_min)          # x_norm = alpha * x + beta
    beta = -2.0 * x_min / (x_max - x_min) - 1.0
    a0_s, a1_s, a2_s, a3_s, k_opt = theta_opt
    # 展开: (alpha x + beta)^3 = alpha^3 x^3 + 3 alpha^2 beta x^2 + 3 alpha beta^2 x + beta^3
    A3 = a3_s * alpha**3
    A2 = a3_s * 3 * alpha**2 * beta + a2_s * alpha**2
    A1 = a3_s * 3 * alpha * beta**2 + a2_s * 2 * alpha * beta + a1_s * alpha
    A0 = a3_s * beta**3 + a2_s * beta**2 + a1_s * beta + a0_s

    print("\n最终模型参数（原始位置多项式系数）:")
    print(f"多项式: {A0:.8f} + {A1:.8e}*x + {A2:.8e}*x^2 + {A3:.8e}*x^3")
    print(f"线性耦合: k = {k_opt:.8f}")

    # 6. 全域性能评估
    # 定义最终预测函数（使用原始 x_raw）
    # 这是使用原始位置多项式系数*原始Z位置得到的预测值，得到的效果是一样的
    def final_predict_LYW(x_raw, e_mag):
        # x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1
        poly = A0 + A1 * x_raw + A2 * x_raw**2 + A3 * x_raw**3
        return poly + k_opt * e_mag

    # 这是原始的预测
    def final_predict(x_raw, e_mag):
        x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1
        poly = a0_s + a1_s * x_norm + a2_s * x_norm**2 + a3_s * x_norm**3
        return poly + k_opt * e_mag

    y_pred_all = final_predict(all_x, all_emag)
    res_all = all_y - y_pred_all
    ss_res = np.sum(res_all**2)
    ss_tot = np.sum((all_y - np.mean(all_y))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(res_all**2))
    max_res = np.max(np.abs(res_all))

    print("\n========== 全域性能指标 (多项式+线性耦合模型) ==========")
    print(f"决定系数 R²      = {r2:.6f}")
    print(f"均方根误差 RMSE  = {rmse:.5f} mm")
    print(f"最大残差 MaxRes  = {max_res:.5f} mm\n")

    # 7. 生成 CSV 文件 (每个工况6列)
    result_df = pd.DataFrame()
    result_df['Z轴位置(mm)'] = x_raw
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        y_pred_i = final_predict(x_raw, emag_i)
        res_i = y_i - y_pred_i

        ss_res_i = np.sum(res_i**2)
        ss_tot_i = np.sum((y_i - np.mean(y_i))**2)
        r2_i = 1 - ss_res_i / ss_tot_i if ss_tot_i != 0 else 0
        rmse_i = np.sqrt(np.mean(res_i**2))

        result_df[f'工况{i+1}磁栅误差(mm)'] = emag_i
        result_df[f'工况{i+1}激光实测(mm)'] = y_i
        result_df[f'工况{i+1}模型拟合误差(mm)'] = y_pred_i
        result_df[f'工况{i+1}残差(mm)'] = res_i
        result_df[f'工况{i+1}_R²'] = np.full(len(x_raw), r2_i)
        result_df[f'工况{i+1}_RMSE(mm)'] = np.full(len(x_raw), rmse_i)

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
    result_df.to_csv('dataPolyLinearAnalysis.csv', index=False, float_format='%.6f')
    print("已生成 dataPolyLinearAnalysis.csv，包含多项式+线性耦合模型各工况分析结果。")

    # 8. 输出各工况独立性能
    print("\n各工况独立性能指标 (多项式+线性耦合模型):")
    print(f"{'工况':<6} {'R²':<12} {'RMSE (mm)':<14} {'MaxRes (mm)':<12}")
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        y_pred_i = final_predict(x_raw, emag_i)
        res_i = y_i - y_pred_i
        ss_res_i = np.sum(res_i**2)
        ss_tot_i = np.sum((y_i - np.mean(y_i))**2)
        r2_i = 1 - ss_res_i / ss_tot_i if ss_tot_i != 0 else 0
        rmse_i = np.sqrt(np.mean(res_i**2))
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
        y_pred_i = final_predict(x_raw, emag_i)
        res_i = y_i - y_pred_i

        ax = axes[i]
        ax.plot(x_raw, emag_i, '--', color='gray', linewidth=1.5, label='磁栅误差')
        ax.plot(x_raw, y_i, '-', color='blue', linewidth=2, label='激光实测误差')
        ax.plot(x_raw, y_pred_i, '-', color='red', linewidth=2, label='多项式+线性拟合')
        ax.plot(x_raw, res_i, '-', color='green', linewidth=1.2, label='残差')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_title(f'{cond_names[i]} (多项式+线性模型)', fontsize=12)
        ax.set_xlabel('Z 轴位置 (mm)', fontsize=10)
        ax.set_ylabel('误差 / 残差 (mm)', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('datas/poly_linear_fitting_with_residuals.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()