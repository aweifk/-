import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def build_features(Z, F, E_meas, degree=3):
    """构造多项式特征：Z, Z^2, Z^3, F, E_meas"""
    X = np.column_stack([Z**i for i in range(1, degree+1)] + [F, E_meas])
    # 添加截距项将在 LinearRegression 中自动处理
    return X

def train_and_predict(df, Z_col, f0_col, f_speeds, hot_state_name, degree=3):
    """
    对单个热状态进行训练和预测
    df: 原始数据框
    Z_col: Z位置列名
    f0_col: 该热状态下的 F0 列名
    f_speeds: 列表，包含速度值及对应的列名，例如 [(2000, '冷机-F2000'), ...]
    hot_state_name: 用于输出和绘图
    """
    Z = df[Z_col].values.astype(float)
    y_true = df[f0_col].values.astype(float)  # F0 真值 (25个点)

    # 收集所有训练样本 (每个 Z 位置下，每个 F 速度为一个样本)
    X_list = []
    y_list = []
    for speed, col_name in f_speeds:
        E_meas = df[col_name].values.astype(float)
        # 构造特征
        X_feat = build_features(Z, np.full_like(Z, speed), E_meas, degree)
        X_list.append(X_feat)
        y_list.append(y_true)  # 同一个 Z 下的 F0 真值重复 4 次

    X_train = np.vstack(X_list)
    y_train = np.hstack(y_list)

    # 训练线性模型
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # 预测所有训练样本
    y_pred = model.predict(X_train)

    # 计算指标
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    max_res = np.max(np.abs(y_train - y_pred))

    # 输出模型参数
    coef_names = [f'Z^{i}' for i in range(1, degree+1)] + ['F', 'E_meas']
    print(f"\n========== {hot_state_name} 模型 ==========")
    print(f"截距: {model.intercept_:.6f}")
    for name, coef in zip(coef_names, model.coef_):
        print(f"{name}: {coef:.6f}")
    print(f"R² = {r2:.6f}, RMSE = {rmse:.5f} mm, MaxRes = {max_res:.5f} mm")

    # 将预测结果按 Z 位置重新组织（用于绘图）
    # 每个 Z 有 4 个预测值（对应不同 F），我们取平均或保留全部
    # 为绘图方便，计算每个 Z 位置下预测值的平均值，并绘制与真值的对比
    n_points = len(Z)
    pred_by_z = y_pred.reshape(len(f_speeds), n_points).T  # (25, 4)
    y_pred_avg = np.mean(pred_by_z, axis=1)  # 每个 Z 位置的平均预测值

    return model, {
        'Z': Z,
        'y_true': y_true,
        'y_pred_avg': y_pred_avg,
        'y_pred_all': y_pred,
        'r2': r2,
        'rmse': rmse,
        'max_res': max_res,
        'f_speeds': f_speeds,
        'pred_by_z': pred_by_z
    }

def plot_results(results_dict, hot_state_names):
    """绘制每个热状态的对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, name in enumerate(hot_state_names):
        res = results_dict[name]
        Z = res['Z']
        y_true = res['y_true']
        y_pred_avg = res['y_pred_avg']
        f_speeds = res['f_speeds']
        pred_by_z = res['pred_by_z']  # (25, 4)

        ax = axes[idx]
        # 绘制真实 F0 曲线
        ax.plot(Z, y_true, 'o-', color='black', linewidth=2, label='真实 F0')
        # 绘制各个 F 速度下的测量值
        for i, (speed, col_name) in enumerate(f_speeds):
            # 获取该速度下的测量值（从原始数据中重新提取，避免重复）
            # 实际在 results 中未保存原始测量值，这里重新读取？简单起见，我们直接用 pred_by_z 的列？不对，pred_by_z 是预测值。
            # 需要在训练时保存原始测量值，为此我们稍作修改，在训练函数中返回 E_meas 矩阵
            # 下面简化：我们只绘制平均预测值，不绘制每个F的测量值，避免复杂。
            pass
        # 绘制平均预测值
        ax.plot(Z, y_pred_avg, 's--', color='red', linewidth=1.5, label='平均预测值')
        # 绘制每个样本点的预测值（不同F）散点
        for i in range(pred_by_z.shape[1]):
            ax.scatter(Z, pred_by_z[:, i], s=20, alpha=0.6, label=f'预测值 (F={f_speeds[i][0]})' if i==0 else "")
        ax.set_xlabel('Z 轴位置 (mm)')
        ax.set_ylabel('磁栅误差 (mm)')
        ax.set_title(f'{name}\nR²={res["r2"]:.4f}, RMSE={res["rmse"]:.4f}mm')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_xaxis()  # 使 Z=0 在右侧更直观，根据习惯可选
    plt.tight_layout()
    plt.savefig('F0_prediction_comparison.png', dpi=300)
    plt.show()

    # 残差图
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()
    for idx, name in enumerate(hot_state_names):
        res = results_dict[name]
        y_true = res['y_true']
        y_pred_avg = res['y_pred_avg']
        residuals = y_true - y_pred_avg
        ax = axes2[idx]
        ax.scatter(y_pred_avg, residuals, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('预测值 (mm)')
        ax.set_ylabel('残差 (mm)')
        ax.set_title(f'{name} 残差图')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('F0_prediction_residuals.png', dpi=300)
    plt.show()

def save_predictions(df, results_dict, hot_state_names, output_csv='F0_predictions.csv'):
    """保存每个Z位置下的真实值、各F速度预测值、平均预测值等"""
    Z = df['Z位置'].values
    out_df = pd.DataFrame({'Z位置(mm)': Z})
    for name in hot_state_names:
        res = results_dict[name]
        out_df[f'{name}_真实F0'] = res['y_true']
        out_df[f'{name}_平均预测F0'] = res['y_pred_avg']
        # 保存每个F速度下的预测值
        for i, (speed, _) in enumerate(res['f_speeds']):
            out_df[f'{name}_预测F{speed}'] = res['pred_by_z'][:, i]
    out_df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"\n预测结果已保存至 {output_csv}")

def main():
    # 1. 读取数据
    df = pd.read_csv('datasF.csv')
    # 列名检查（根据实际文件调整）
    print("数据列名:", df.columns.tolist())
    # 重新命名可能存在的空格问题（手动修正）
    df.columns = df.columns.str.strip()
    Z_col = 'Z位置'

    # 2. 定义热状态及对应的列名
    # 每个热状态包含 F0 列和四个速度列
    hot_states = {
        '冷机': {
            'f0': '冷机-F0',
            'f_speeds': [(2000, '冷机-F2000'), (5000, '冷机-F5000'), (8000, '冷机-F8000'), (10000, '冷机-F10000')]
        },
        '热机15分钟': {
            'f0': '热机15分钟-F0',
            'f_speeds': [(2000, '热机15分钟-F2000'), (5000, '热机15分钟-F5000'), (8000, '热机15分钟-F8000'), (10000, '热机15分钟-F10000')]
        },
        '热机30分钟': {
            'f0': '热机30分钟-F0',
            'f_speeds': [(2000, '热机30分钟-F2000'), (5000, '热机30分钟-F5000'), (8000, '热机30分钟-F8000'), (10000, '热机30分钟-F10000')]
        },
        '热机45分钟': {
            'f0': '热机45分钟-F0',
            'f_speeds': [(2000, '热机45分钟-F2000'), (5000, '热机45分钟-F5000'), (8000, '热机45分钟-F8000'), (10000, '热机45分钟-F10000')]
        }
    }

    results = {}
    for name, info in hot_states.items():
        model, res_dict = train_and_predict(
            df, Z_col, info['f0'], info['f_speeds'], name, degree=3
        )
        results[name] = res_dict

    # 3. 绘图
    plot_results(results, list(hot_states.keys()))

    # 4. 保存预测结果
    save_predictions(df, results, list(hot_states.keys()), 'F0_predictions.csv')

    # 5. 可选：输出每个热状态下的模型参数到单独文件
    # 已在训练过程中打印

if __name__ == "__main__":
    main()