import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 设置 matplotlib 支持中文 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 1. 读取数据
    df = pd.read_csv('data_modify.csv')
    x_pos = df.iloc[:, 0].values.astype(float)  # Z轴位置 (mm)
    n_conditions = 9
    cond_names = [f'工况{i}' for i in range(1, n_conditions + 1)]

    # 提取所有工况的磁栅误差和激光实测
    emag_list = []
    y_list = []
    for i in range(n_conditions):
        col_emag = df.columns[1 + 2 * i]  # 工况i磁栅误差列
        col_y = df.columns[2 + 2 * i]  # 工况i激光实测列
        emag_i = df[col_emag].values.astype(float)
        y_i = df[col_y].values.astype(float)
        emag_list.append(emag_i)
        y_list.append(y_i)

    # 基准工况（工况1）
    base_emag = emag_list[0]
    base_y = y_list[0]

    # 2. 计算每个工况的预测值、残差及性能指标
    result_df = pd.DataFrame()
    result_df['Z轴位置(mm)'] = x_pos

    # 用于汇总全域数据（工况2~9）
    all_y_true = []
    all_y_pred = []

    perf_list = []  # 存储每个工况的性能

    for i in range(n_conditions):
        emag_i = emag_list[i]
        y_true_i = y_list[i]

        if i == 0:  # 工况1：预测值 = 实测值，残差为0
            y_pred_i = y_true_i.copy()
            res_i = np.zeros_like(y_true_i)
        else:
            # 预测公式：工况1激光实测 + (当前工况磁栅误差 - 工况1磁栅误差)
            y_pred_i = base_y + (emag_i - base_emag)
            res_i = y_true_i - y_pred_i
            # 累积到全域数据（仅工况2~9）
            all_y_true.extend(y_true_i)
            all_y_pred.extend(y_pred_i)

        # 计算工况性能指标
        ss_res = np.sum(res_i ** 2)
        ss_tot = np.sum((y_true_i - np.mean(y_true_i)) ** 2) if i == 0 else np.sum((y_true_i - np.mean(y_true_i)) ** 2)
        if ss_tot == 0:
            r2_i = 1.0
        else:
            r2_i = 1 - ss_res / ss_tot
        rmse_i = np.sqrt(np.mean(res_i ** 2))
        max_res_i = np.max(np.abs(res_i))

        perf_list.append({
            '工况': cond_names[i],
            'R²': r2_i,
            'RMSE (mm)': rmse_i,
            'MaxRes (mm)': max_res_i
        })

        # 将列添加到 result_df
        result_df[f'工况{i + 1}磁栅误差(mm)'] = emag_i
        result_df[f'工况{i + 1}激光实测(mm)'] = y_true_i
        result_df[f'工况{i + 1}预测值(mm)'] = y_pred_i
        result_df[f'工况{i + 1}残差(mm)'] = res_i
        result_df[f'工况{i + 1}_R²'] = np.full(len(x_pos), r2_i)
        result_df[f'工况{i + 1}_RMSE(mm)'] = np.full(len(x_pos), rmse_i)

    # 3. 全域性能（工况2~9的合并数据）
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    res_all = all_y_true - all_y_pred
    ss_res_all = np.sum(res_all ** 2)
    ss_tot_all = np.sum((all_y_true - np.mean(all_y_true)) ** 2)
    r2_all = 1 - ss_res_all / ss_tot_all
    rmse_all = np.sqrt(np.mean(res_all ** 2))
    max_res_all = np.max(np.abs(res_all))

    # 4. 控制台输出
    print("========== 基于磁栅误差差值的预测模型 ==========")
    print("预测公式: 预测值 = 工况1激光实测 + (工况i磁栅误差 - 工况1磁栅误差)")
    print("\n各工况独立性能指标（差分模型）:")
    print(f"{'工况':<6} {'R²':<12} {'RMSE (mm)':<14} {'MaxRes (mm)':<12}")
    for p in perf_list:
        print(f"{p['工况']:<6} {p['R²']:<12.6f} {p['RMSE (mm)']:<14.6f} {p['MaxRes (mm)']:<12.6f}")

    print("\n========== 全域性能指标 (工况2~9合并) ==========")
    print(f"决定系数 R²      = {r2_all:.6f}")
    print(f"均方根误差 RMSE  = {rmse_all:.5f} mm")
    print(f"最大残差 MaxRes  = {max_res_all:.5f} mm")

    # 5. 导出 CSV
    # 调整列顺序：Z轴位置，然后工况1的6列，工况2的6列...
    ordered_cols = ['Z轴位置(mm)']
    for i in range(1, n_conditions + 1):
        ordered_cols.extend([
            f'工况{i}磁栅误差(mm)',
            f'工况{i}激光实测(mm)',
            f'工况{i}预测值(mm)',
            f'工况{i}残差(mm)',
            f'工况{i}_R²',
            f'工况{i}_RMSE(mm)'
        ])
    result_df = result_df[ordered_cols]
    result_df.to_csv('dataDifferenceAnalysis.csv', index=False, float_format='%.6f')
    print("\n已生成 dataDifferenceAnalysis.csv")

    # 6. 绘制9个工况的子图（含磁栅误差、激光实测、预测值、残差）
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()

    for i in range(n_conditions):
        emag_i = emag_list[i]
        y_true_i = y_list[i]
        if i == 0:
            y_pred_i = y_true_i.copy()
            res_i = np.zeros_like(y_true_i)
        else:
            y_pred_i = base_y + (emag_i - base_emag)
            res_i = y_true_i - y_pred_i

        ax = axes[i]
        ax.plot(x_pos, emag_i, '--', color='gray', linewidth=1.5, label='磁栅误差')
        ax.plot(x_pos, y_true_i, '-', color='blue', linewidth=2, label='激光实测')
        ax.plot(x_pos, y_pred_i, '-', color='red', linewidth=2, label='预测值')
        ax.plot(x_pos, res_i, '-', color='green', linewidth=1.2, label='残差')
        ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)

        ax.set_title(f'{cond_names[i]} (差分模型)', fontsize=12)
        ax.set_xlabel('Z 轴位置 (mm)', fontsize=10)
        ax.set_ylabel('误差 / 残差 (mm)', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_xaxis()  # 使Z轴从0到-240显示

    plt.tight_layout()
    plt.savefig('datas/difference_prediction.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()