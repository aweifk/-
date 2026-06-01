import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


# ---------- 模型：多项式和谐波均使用归一化位置 ----------
def model_predict(theta, x_raw, e_mag, x_min, x_max):
    a0, a1, a2, a3, k, b1, c1, b2, c2 = theta
    x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1  # 映射到 [-1,1]
    poly = a0 + a1 * x_norm + a2 * x_norm ** 2 + a3 * x_norm ** 3
    linear = k * e_mag
    # 谐波也使用 x_norm，周期对应原始位置 2mm
    harm = (b1 * np.cos(np.pi * x_norm) + c1 * np.sin(np.pi * x_norm) +
            b2 * np.cos(2 * np.pi * x_norm) + c2 * np.sin(2 * np.pi * x_norm))
    return poly + linear + harm


def residuals(theta, x_raw, e_mag, y_true, x_min, x_max):
    return y_true - model_predict(theta, x_raw, e_mag, x_min, x_max)


def main():
    df = pd.read_csv('data_modify.csv')
    x_raw = df.iloc[:, 0].values.astype(float)
    n_conditions = 9
    cond_names = [f'工况{i}' for i in range(1, n_conditions + 1)]

    # 汇总全部样本
    all_x, all_emag, all_y = [], [], []
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

    x_min, x_max = all_x.min(), all_x.max()
    x_norm_all = 2 * (all_x - x_min) / (x_max - x_min) - 1

    # ----- 初值估计 -----
    lr = LinearRegression()
    lr.fit(all_emag.reshape(-1, 1), all_y)
    k_init = lr.coef_[0]

    # 多项式初值：用三阶多项式拟合 x_norm 与 y
    coefs = np.polyfit(x_norm_all, all_y, 3)  # 返回 a3,a2,a1,a0
    a0_init, a1_init, a2_init, a3_init = coefs[3], coefs[2], coefs[1], coefs[0]

    # 谐波初值设为 0
    theta0 = np.array([a0_init, a1_init, a2_init, a3_init, k_init, 0.0, 0.0, 0.0, 0.0])

    # 边界约束（防止参数爆炸）
    lower = [-0.5, -0.5, -0.5, -0.5, 0.0, -0.05, -0.05, -0.05, -0.05]
    upper = [0.5, 0.5, 0.5, 0.5, 1.0, 0.05, 0.05, 0.05, 0.05]

    # 优化
    result = least_squares(
        lambda t: residuals(t, all_x, all_emag, all_y, x_min, x_max),
        theta0, bounds=(lower, upper), method='trf',
        max_nfev=500, ftol=1e-10, xtol=1e-10
    )
    theta_opt = result.x
    print("优化收敛信息:", result.message)
    print("优化后参数 (归一化多项式系数):", theta_opt)

    # 将归一化多项式系数转换为原始位置多项式系数（用于公式展示）
    alpha = 2.0 / (x_max - x_min)  # 约 0.008333...
    beta = -2.0 * x_min / (x_max - x_min) - 1.0  # x_min=-240 → beta = 1
    a0_s, a1_s, a2_s, a3_s = theta_opt[:4]
    A3 = a3_s * alpha ** 3
    A2 = a3_s * 3 * alpha ** 2 * beta + a2_s * alpha ** 2
    A1 = a3_s * 3 * alpha * beta ** 2 + a2_s * 2 * alpha * beta + a1_s * alpha
    A0 = a3_s * beta ** 3 + a2_s * beta ** 2 + a1_s * beta + a0_s
    k_opt = theta_opt[4]
    b1, c1, b2, c2 = theta_opt[5:9]

    print("\n原始位置多项式系数:")
    print(f"  A0 = {A0:.8e}, A1 = {A1:.8e}, A2 = {A2:.8e}, A3 = {A3:.8e}")
    print(f"  k = {k_opt:.8f}")
    print(f"  b1 = {b1:.8f}, c1 = {c1:.8f}, b2 = {b2:.8f}, c2 = {c2:.8f}")

    # 全域性能评估
    y_pred_all = model_predict(theta_opt, all_x, all_emag, x_min, x_max)
    res_all = all_y - y_pred_all
    ss_res = np.sum(res_all ** 2)
    ss_tot = np.sum((all_y - np.mean(all_y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(res_all ** 2))
    max_res = np.max(np.abs(res_all))
    print(f"\n全域性能: R² = {r2:.6f}, RMSE = {rmse:.5f} mm, MaxRes = {max_res:.5f} mm")

    # 生成 dataAnalysis.csv（每工况6列）
    result_df = pd.DataFrame()
    result_df['Z轴位置(mm)'] = x_raw
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        y_pred_i = model_predict(theta_opt, x_raw, emag_i, x_min, x_max)
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

    cols = ['Z轴位置(mm)']
    for i in range(1, n_conditions + 1):
        cols.extend([f'工况{i}磁栅误差(mm)', f'工况{i}激光实测(mm)',
                     f'工况{i}模型拟合误差(mm)', f'工况{i}残差(mm)',
                     f'工况{i}_R²', f'工况{i}_RMSE(mm)'])
    result_df = result_df[cols]
    result_df.to_csv('dataAnalysis.csv', index=False, float_format='%.6f')
    print("\n已保存 dataAnalysis.csv")

    # 绘制9个子图
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values
        y_i = df[col_y].values
        y_pred_i = model_predict(theta_opt, x_raw, emag_i, x_min, x_max)
        res_i = y_i - y_pred_i
        ax = axes[i]
        ax.plot(x_raw, emag_i, '--', color='gray', linewidth=1.5, label='磁栅误差')
        ax.plot(x_raw, y_i, '-', color='blue', linewidth=2, label='激光实测')
        ax.plot(x_raw, y_pred_i, '-', color='red', linewidth=2, label='融合拟合')
        ax.plot(x_raw, res_i, '-', color='green', linewidth=1.2, label='残差')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(cond_names[i])
        ax.set_xlabel('Z轴位置 (mm)')
        ax.set_ylabel('误差 / 残差 (mm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.5)
        ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('datas/fusion_fitting.png', dpi=300)
    plt.show()

    # 输出各工况性能
    print("\n各工况独立性能:")
    print(f"{'工况':<6} {'R²':<12} {'RMSE (mm)':<14} {'MaxRes (mm)':<12}")
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values
        y_i = df[col_y].values
        y_pred_i = model_predict(theta_opt, x_raw, emag_i, x_min, x_max)
        res_i = y_i - y_pred_i
        ss_res_i = np.sum(res_i ** 2)
        ss_tot_i = np.sum((y_i - np.mean(y_i)) ** 2)
        r2_i = 1 - ss_res_i / ss_tot_i if ss_tot_i != 0 else 0
        rmse_i = np.sqrt(np.mean(res_i ** 2))
        max_i = np.max(np.abs(res_i))
        print(f"{cond_names[i]:<6} {r2_i:<12.6f} {rmse_i:<14.6f} {max_i:<12.6f}")


if __name__ == "__main__":
    main()