import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 设置 matplotlib 支持中文 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------- 线性耦合模型闭式求解 -------------------------
def linear_least_squares(emag_all, y_all):
    """
    最小二乘闭式解求线性模型参数 K, B
    公式:
        K = (n*sum(E_mag*E_real) - sum(E_mag)*sum(E_real)) / (n*sum(E_mag^2) - (sum(E_mag))^2)
        B = (sum(E_real) - K*sum(E_mag)) / n
    参数:
        emag_all: 一维数组，所有样本的磁栅误差
        y_all: 一维数组，所有样本的激光实测误差
    返回:
        K, B
    """
    n = len(emag_all)
    sum_emag = np.sum(emag_all)
    sum_y = np.sum(y_all)
    sum_emag_y = np.sum(emag_all * y_all)
    sum_emag2 = np.sum(emag_all ** 2)

    K = (n * sum_emag_y - sum_emag * sum_y) / (n * sum_emag2 - sum_emag ** 2)
    B = (sum_y - K * sum_emag) / n
    return K, B


# ------------------------------ 主程序 ---------------------------------
def main():
    # 1. 读取数据
    df = pd.read_csv('data_modify.csv')
    x_pos = df.iloc[:, 0].values.astype(float)  # Z轴位置(mm)

    n_conditions = 9
    cond_names = [f'工况{i}' for i in range(1, n_conditions + 1)]

    # 汇总所有样本用于训练线性模型
    all_emag = []
    all_y = []
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]  # 磁栅误差列
        col_y = df.columns[2 + 2 * i]  # 激光实测列
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        all_emag.extend(emag_i)
        all_y.extend(y_i)

    all_emag = np.array(all_emag)
    all_y = np.array(all_y)

    # 2. 求解全局线性参数 K, B
    K, B = linear_least_squares(all_emag, all_y)
    print("========== 磁栅-激光线性耦合映射算法 ==========")
    print(f"全局线性模型: E_real = {K:.6f} * E_mag + {B:.6f}")
    print(f"参数保留4位小数: K = {K:.4f}, B = {B:.4f}\n")

    # 3. 计算每个工况的拟合值、残差、R²、RMSE
    # 准备存储结果的DataFrame
    result_df = pd.DataFrame()
    result_df['Z轴位置(mm)'] = x_pos

    # 存储每个工况的性能指标（用于控制台输出）
    perf_list = []

    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)

        # 线性模型预测
        y_pred_i = K * emag_i + B
        res_i = y_i - y_pred_i  # 残差 = 实测 - 拟合

        # 计算该工况的 R² 和 RMSE
        ss_res = np.sum(res_i ** 2)
        ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
        r2_i = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        rmse_i = np.sqrt(np.mean(res_i ** 2))
        max_res_i = np.max(np.abs(res_i))

        # 存储性能指标
        perf_list.append({
            '工况': cond_names[i],
            'R²': r2_i,
            'RMSE (mm)': rmse_i,
            'MaxRes (mm)': max_res_i
        })

        # 向 result_df 添加该工况的6列
        result_df[f'工况{i + 1}磁栅误差(mm)'] = emag_i
        result_df[f'工况{i + 1}激光实测(mm)'] = y_i
        result_df[f'工况{i + 1}模型拟合误差(mm)'] = y_pred_i
        result_df[f'工况{i + 1}残差(mm)'] = res_i
        result_df[f'工况{i + 1}_R²'] = np.full(len(x_pos), r2_i)
        result_df[f'工况{i + 1}_RMSE(mm)'] = np.full(len(x_pos), rmse_i)

    # 调整列顺序
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

    # 保存到 dataLineAnalysis.csv (数值保留6位小数)
    result_df.to_csv('dataLineAnalysis.csv', index=False, float_format='%.6f')
    print("已生成 dataLineAnalysis.csv，包含线性模型对各工况的分析结果。\n")

    # 4. 输出各工况性能对比表
    print("各工况独立性能指标（线性耦合映射模型）:")
    print(f"{'工况':<6} {'R²':<12} {'RMSE (mm)':<14} {'MaxRes (mm)':<12}")
    for p in perf_list:
        print(f"{p['工况']:<6} {p['R²']:<12.6f} {p['RMSE (mm)']:<14.6f} {p['MaxRes (mm)']:<12.6f}")

    # 5. 绘制各工况的折线图（磁栅误差、激光实测、线性拟合、残差）
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()

    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]
        col_y = df.columns[2 + 2 * i]
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        y_pred_i = K * emag_i + B
        res_i = y_i - y_pred_i

        ax = axes[i]
        ax.plot(x_pos, emag_i, '--', color='gray', linewidth=1.5, label='磁栅误差')
        ax.plot(x_pos, y_i, '-', color='blue', linewidth=2, label='激光实测误差')
        ax.plot(x_pos, y_pred_i, '-', color='red', linewidth=2, label='线性模型拟合误差')
        ax.plot(x_pos, res_i, '-', color='green', linewidth=1.2, label='残差 (实测-拟合)')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)

        ax.set_title(f'{cond_names[i]} (线性模型)', fontsize=12)
        ax.set_xlabel('Z 轴位置 (mm)', fontsize=10)
        ax.set_ylabel('误差 / 残差 (mm)', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig('datas/linear_fitting_with_residuals.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()