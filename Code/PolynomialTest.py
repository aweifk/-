import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

# ==================== 设置 matplotlib 支持中文 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

N_CONDITIONS = 9


def normalize_z(z, x_min, x_max):
    return 2 * (z - x_min) / (x_max - x_min) - 1


def fit_emag_poly_norm(x_raw, emag, x_min, x_max):
    x_norm = normalize_z(x_raw, x_min, x_max)
    return np.polyfit(x_norm, emag, 3)


def predict_emag_poly_norm(z_new, coefs, x_min, x_max):
    z_norm = normalize_z(z_new, x_min, x_max)
    return np.polyval(coefs, z_norm)


def fit_laser_linear(x_raw, y):
    lr = LinearRegression()
    lr.fit(x_raw.reshape(-1, 1), y)
    return lr


def generate_testdata_from_ref(ref_df, verbose=True):
    x_raw = ref_df.iloc[:, 0].values.astype(float)
    x_min, x_max = x_raw.min(), x_raw.max()
    z_new = np.arange(0, -241, -1)

    if verbose:
        print(f"Z 轴范围: [{x_min}, {x_max}] mm，模拟网格: 1 mm 步长，共 {len(z_new)} 点")

    test_df = pd.DataFrame({'Z轴位置(mm)': z_new})
    emag_coefs_list = []
    laser_models = []
    for i in range(N_CONDITIONS):
        col_emag = ref_df.columns[1 + 2 * i]
        col_y = ref_df.columns[2 + 2 * i]
        emag_i = ref_df[col_emag].values.astype(float)
        y_i = ref_df[col_y].values.astype(float)

        coefs_emag = fit_emag_poly_norm(x_raw, emag_i, x_min, x_max)
        emag_sim = predict_emag_poly_norm(z_new, coefs_emag, x_min, x_max)
        emag_coefs_list.append(coefs_emag)

        lr = fit_laser_linear(x_raw, y_i)
        y_sim = lr.predict(z_new.reshape(-1, 1))
        laser_models.append(lr)

        test_df[f'工况{i + 1}磁栅误差E(mm)'] = emag_sim
        test_df[f'工况{i + 1}激光实测E(mm)'] = y_sim

    if verbose:
        print("\n各工况磁栅误差拟合（归一化三阶多项式 e_mag ~ Z）:")
        print(f"{'工况':<6} {'a3':<14} {'a2':<14} {'a1':<14} {'a0':<14}")
        print("-" * 62)
        for i, coefs_emag in enumerate(emag_coefs_list):
            a3, a2, a1, a0 = coefs_emag
            print(f"工况{i + 1:<4} {a3:<14.6e} {a2:<14.6e} {a1:<14.6e} {a0:<14.6f}")

        print("\n各工况激光实测拟合（线性回归 y ~ Z）:")
        print(f"{'工况':<6} {'斜率 a':<16} {'截距 b':<16}")
        print("-" * 40)
        for i, lr in enumerate(laser_models):
            print(f"工况{i + 1:<4} {lr.coef_[0]:<16.8e} {lr.intercept_:<16.8f}")

    return test_df


def collect_samples(df):
    x_raw = df.iloc[:, 0].values.astype(float)
    all_x, all_emag, all_y = [], [], []
    for i in range(N_CONDITIONS):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        all_x.extend(x_raw)
        all_emag.extend(emag_i)
        all_y.extend(y_i)
    return x_raw, np.array(all_x), np.array(all_emag), np.array(all_y)


def theta_to_raw_coeffs(theta_opt, x_min, x_max):
    alpha = 2.0 / (x_max - x_min)
    beta = -2.0 * x_min / (x_max - x_min) - 1.0
    a0_s, a1_s, a2_s, a3_s, k_opt = theta_opt
    A3 = a3_s * alpha**3
    A2 = a3_s * 3 * alpha**2 * beta + a2_s * alpha**2
    A1 = a3_s * 3 * alpha * beta**2 + a2_s * 2 * alpha * beta + a1_s * alpha
    A0 = a3_s * beta**3 + a2_s * beta**2 + a1_s * beta + a0_s
    return A0, A1, A2, A3, k_opt, a0_s, a1_s, a2_s, a3_s


def fit_and_analyze(df, csv_path, fig_path, verbose=True):
    x_raw, all_x, all_emag, all_y = collect_samples(df)
    cond_names = [f'工况{i}' for i in range(1, N_CONDITIONS + 1)]

    x_min, x_max = all_x.min(), all_x.max()
    all_x_norm = normalize_z(all_x, x_min, x_max)

    lr = LinearRegression()
    lr.fit(all_emag.reshape(-1, 1), all_y)
    k_init = lr.coef_[0]
    poly_coefs = np.polyfit(all_x_norm, all_y, 3)
    a0_init, a1_init, a2_init, a3_init = poly_coefs[3], poly_coefs[2], poly_coefs[1], poly_coefs[0]
    theta0 = np.array([a0_init, a1_init, a2_init, a3_init, k_init])

    if verbose:
        print(f"\n{'=' * 20} testdata 融合拟合预测 {'=' * 20}")
        print(f"初始线性回归: k = {k_init:.6f}")
        print("初始参数 (多项式系数为归一化形式):", theta0)

    def obj_func(theta, x_norm, e_mag, y_true):
        a0_s, a1_s, a2_s, a3_s, k = theta
        poly = a0_s + a1_s * x_norm + a2_s * x_norm**2 + a3_s * x_norm**3
        return y_true - (poly + k * e_mag)

    result = least_squares(
        obj_func, theta0, args=(all_x_norm, all_emag, all_y),
        method='lm', max_nfev=500, xtol=1e-10, ftol=1e-10,
    )
    theta_opt = result.x

    if verbose:
        print("\n优化收敛信息:", result.message)
        print("优化后参数 (归一化多项式系数):", theta_opt)

    A0, A1, A2, A3, k_opt, a0_s, a1_s, a2_s, a3_s = theta_to_raw_coeffs(theta_opt, x_min, x_max)

    if verbose:
        print("\n最终模型参数（原始位置多项式系数）:")
        print(f"多项式: {A0:.8f} + {A1:.8e}*x + {A2:.8e}*x^2 + {A3:.8e}*x^3")
        print(f"线性耦合: k = {k_opt:.8f}")

    def final_predict(x_vals, e_mag):
        x_norm = normalize_z(x_vals, x_min, x_max)
        poly = a0_s + a1_s * x_norm + a2_s * x_norm**2 + a3_s * x_norm**3
        return poly + k_opt * e_mag

    y_pred_all = final_predict(all_x, all_emag)
    res_all = all_y - y_pred_all
    ss_res = np.sum(res_all**2)
    ss_tot = np.sum((all_y - np.mean(all_y))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(res_all**2))
    max_res = np.max(np.abs(res_all))

    if verbose:
        print("\n========== 全域性能指标 (多项式+线性耦合模型) ==========")
        print(f"决定系数 R^2     = {r2:.6f}")
        print(f"均方根误差 RMSE  = {rmse:.5f} mm")
        print(f"最大残差 MaxRes  = {max_res:.5f} mm\n")

    result_df = pd.DataFrame()
    result_df['Z轴位置(mm)'] = x_raw
    for i in range(N_CONDITIONS):
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

        result_df[f'工况{i + 1}磁栅误差(mm)'] = emag_i
        result_df[f'工况{i + 1}激光实测(mm)'] = y_i
        result_df[f'工况{i + 1}模型拟合误差(mm)'] = y_pred_i
        result_df[f'工况{i + 1}残差(mm)'] = res_i
        result_df[f'工况{i + 1}_R²'] = np.full(len(x_raw), r2_i)
        result_df[f'工况{i + 1}_RMSE(mm)'] = np.full(len(x_raw), rmse_i)

    ordered_cols = ['Z轴位置(mm)']
    for i in range(1, N_CONDITIONS + 1):
        ordered_cols.extend([
            f'工况{i}磁栅误差(mm)',
            f'工况{i}激光实测(mm)',
            f'工况{i}模型拟合误差(mm)',
            f'工况{i}残差(mm)',
            f'工况{i}_R²',
            f'工况{i}_RMSE(mm)',
        ])
    result_df = result_df[ordered_cols]
    result_df.to_csv(csv_path, index=False, float_format='%.6f')

    if verbose:
        print(f"已生成 {csv_path}，包含多项式+线性耦合模型各工况分析结果。")
        print("\n各工况独立性能指标 (多项式+线性耦合模型):")
        print(f"{'工况':<6} {'R^2':<12} {'RMSE (mm)':<14} {'MaxRes (mm)':<12}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()
    for i in range(N_CONDITIONS):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        y_pred_i = final_predict(x_raw, emag_i)
        res_i = y_i - y_pred_i

        if verbose:
            ss_res_i = np.sum(res_i**2)
            ss_tot_i = np.sum((y_i - np.mean(y_i))**2)
            r2_i = 1 - ss_res_i / ss_tot_i if ss_tot_i != 0 else 0
            rmse_i = np.sqrt(np.mean(res_i**2))
            max_i = np.max(np.abs(res_i))
            print(f"{cond_names[i]:<6} {r2_i:<12.6f} {rmse_i:<14.6f} {max_i:<12.6f}")

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

    os.makedirs(os.path.dirname(fig_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    if verbose:
        print(f"\n已保存图像: {fig_path}")
    plt.show()


def main():
    print("=" * 20 + " 阶段一：生成 testdata.csv " + "=" * 20)
    ref_df = pd.read_csv('data_modify.csv')
    test_df = generate_testdata_from_ref(ref_df, verbose=True)
    test_df.to_csv('testdata.csv', index=False, float_format='%.4f')
    print(f"\n已生成 testdata.csv，共 {len(test_df)} 个 Z 位置点，9 个工况。")

    print("\n" + "=" * 20 + " 阶段二：testdata 融合拟合预测 " + "=" * 20)
    fit_and_analyze(
        test_df,
        'testdataPolyLinearAnalysis.csv',
        'datas/test_poly_linear_fitting_with_residuals.png',
    )


if __name__ == '__main__':
    main()
