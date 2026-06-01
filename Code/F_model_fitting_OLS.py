"""
F_model_fitting_V3.py
使用普通最小二乘回归，没有Z与进给速度F的交互项
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_dataset(df):
    """构建训练数据集：每个样本为 (Z, F, E_meas) -> y (对应热状态的 F0 真值)"""
    Z = df['Z位置'].values.astype(float)

    # 定义热状态（注意：不再使用时间 t）
    hot_states = ['冷机', '热机15分钟', '热机30分钟', '热机45分钟']

    # 不同进给速度及其列名后缀
    speeds = [(2000, 'F2000'), (5000, 'F5000'), (8000, 'F8000'), (10000, 'F10000')]

    records = []
    for state_name in hot_states:
        f0_col = f'{state_name}-F0'
        y_true = df[f0_col].values.astype(float)
        for speed, suffix in speeds:
            col_name = f'{state_name}-{suffix}'
            e_meas = df[col_name].values.astype(float)
            for i, z in enumerate(Z):
                records.append({
                    'Z': z,
                    'F': speed,
                    'E_meas': e_meas[i],
                    'state': state_name,  # 仅用于后期分组评估，不参与训练
                    'y_true': y_true[i]
                })
    return pd.DataFrame(records)


def train_model(df_train, use_ridge=False, alpha=1.0, degree=3):
    """
    训练统一模型（不使用时间 t）
    degree: Z 的多项式次数
    """
    # 构造特征矩阵：1, Z, Z^2, ..., Z^degree, F, E_meas
    X_list = []
    for _, row in df_train.iterrows():
        z = row['Z']
        features = [1.0]  # 截距项
        for d in range(1, degree + 1):
            features.append(z ** d)
        features.append(row['F'])
        features.append(row['E_meas'])
        X_list.append(features)

    X = np.array(X_list)
    y = df_train['y_true'].values

    # 使用线性回归或岭回归
    if use_ridge:
        model = Ridge(alpha=alpha, fit_intercept=False)  # 已加常数项
    else:
        model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    return model, X, y


def predict_with_coef(X, coef, intercept=0.0):
    """
    使用原始系数 coef_ 进行线性预测（不调用 model.predict）。

    公式: y_hat = X @ coef + intercept
    本项目中 fit_intercept=False 且特征已含常数列 1，intercept 通常为 0。
    """
    coef = np.asarray(coef).ravel()
    return X @ coef + intercept     # 矩阵乘法


def verify_coef_vs_predict(model, X, rtol=1e-12, atol=1e-12):
    """
    验证 model.coef_ 手动预测与 model.predict 是否一致。

    Returns
    -------
    dict
        包含是否一致、最大绝对误差、最大相对误差、前若干条差异样本索引等。
    """
    coef = model.coef_
    intercept = getattr(model, 'intercept_', 0.0)
    if intercept is None:
        intercept = 0.0
    intercept = float(np.asarray(intercept).ravel()[0]) if np.size(intercept) else 0.0

    y_manual = predict_with_coef(X, coef, intercept)
    y_sklearn = model.predict(X)

    diff = y_manual - y_sklearn
    abs_diff = np.abs(diff)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))

    # 相对误差（避免除零）
    denom = np.maximum(np.abs(y_sklearn), 1e-15)
    rel_diff = abs_diff / denom
    max_rel = float(np.max(rel_diff))

    is_close = np.allclose(y_manual, y_sklearn, rtol=rtol, atol=atol)

    worst_idx = int(np.argmax(abs_diff))
    report = {
        'is_close': is_close,
        'max_abs_diff': max_abs,
        'mean_abs_diff': mean_abs,
        'max_rel_diff': max_rel,
        'n_samples': len(y_manual),
        'coef': coef,
        'intercept': intercept,
        'worst_index': worst_idx,
        'y_manual_at_worst': float(y_manual[worst_idx]),
        'y_sklearn_at_worst': float(y_sklearn[worst_idx]),
        'X_at_worst': X[worst_idx],
    }

    print("\n========== coef_ 手动预测 vs model.predict 一致性验证 ==========")
    print(f"样本数: {report['n_samples']}")
    print(f"intercept_ = {intercept:.16e}  (fit_intercept=False 时应为 0)")
    print(f"coef_ 形状: {np.asarray(coef).shape}, 数值:\n  {np.asarray(coef)}")
    print(f"np.allclose 结果: {is_close}  (rtol={rtol}, atol={atol})")
    print(f"最大绝对误差 max|manual - predict|: {max_abs:.16e} mm")
    print(f"平均绝对误差 mean|manual - predict|: {mean_abs:.16e} mm")
    print(f"最大相对误差: {max_rel:.16e}")
    if not is_close:
        print(f"差异最大样本索引: {worst_idx}")
        print(f"  手动预测 y_manual = {report['y_manual_at_worst']:.16f}")
        print(f"  sklearn  y_sklearn = {report['y_sklearn_at_worst']:.16f}")
        print(f"  该样本特征 X = {report['X_at_worst']}")
    else:
        print("结论: model.coef_ 线性组合与 model.predict 完全一致（在数值容差内）。")
    print("手动预测公式: y_hat = X @ model.coef_ + model.intercept_")
    print("================================================================\n")

    return report


def evaluate_and_plot(model, df_train, hot_states, degree=3):
    """评估模型并绘制各热状态的结果"""
    # 重新构造特征以预测所有样本
    X_all = []
    for _, row in df_train.iterrows():
        z = row['Z']
        features = [1.0]
        for d in range(1, degree + 1):
            features.append(z ** d)
        features.append(row['F'])
        features.append(row['E_meas'])
        X_all.append(features)
    X_all = np.array(X_all)

    # 方法1：使用model.predict进行预测
    # y_pred = model.predict(X_all)

    # 方法二：使用model.coef_原始系数进行预测
    # 使用 model.coef_ 原始系数进行预测（等价于 y_hat = X @ coef_ + intercept_）
    intercept = getattr(model, 'intercept_', 0.0)
    if intercept is None:
        intercept = 0.0
    intercept = float(np.asarray(intercept).ravel()[0]) if np.size(intercept) else 0.0
    y_pred = predict_with_coef(X_all, model.coef_, intercept)

    # 验证：手动 coef_ 预测 与 model.predict 是否一致
    verify_coef_vs_predict(model, X_all)

    df_train['y_pred'] = y_pred

    # 整体性能（所有热状态合并）
    y_true_all = df_train['y_true'].values
    r2_all = r2_score(y_true_all, y_pred)
    rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred))
    max_res_all = np.max(np.abs(y_true_all - y_pred))
    print("========== 全域性能（所有热状态合并） ==========")
    print(f"R² = {r2_all:.6f}, RMSE = {rmse_all:.5f} mm, MaxRes = {max_res_all:.5f} mm\n")

    # 按热状态分组性能
    results = {}
    for state_name in hot_states:
        subset = df_train[df_train['state'] == state_name]
        y_true = subset['y_true'].values
        y_pred = subset['y_pred'].values
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        max_res = np.max(np.abs(y_true - y_pred))
        results[state_name] = {'r2': r2, 'rmse': rmse, 'max_res': max_res, 'subset': subset}
        print(f"{state_name}: R²={r2:.6f}, RMSE={rmse:.5f} mm, MaxRes={max_res:.5f} mm")

    # 绘图：每个热状态一张子图，包含真实F0曲线、预测值散点/平均值、各F速度测量值
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    Z_unique = np.sort(df_train['Z'].unique())

    for idx, state_name in enumerate(hot_states):
        ax = axes[idx]
        subset = df_train[df_train['state'] == state_name]
        # 真实F0曲线（每个Z一个真值）
        true_by_z = subset.groupby('Z')['y_true'].first().reindex(Z_unique)
        ax.plot(Z_unique, true_by_z, 'o-', color='black', linewidth=2, label='真实 F0')

        # 绘制各个进给速度下的测量值（输入 E_meas）
        speeds = [2000, 5000, 8000, 10000]
        for speed in speeds:
            speed_data = subset[subset['F'] == speed]
            if not speed_data.empty:
                ax.scatter(speed_data['Z'], speed_data['E_meas'], s=20, alpha=0.6,
                           label=f'测量值 F={speed}', marker='x')

        # 绘制每个Z下的所有预测值（散点）
        pred_by_z = subset.groupby('Z')['y_pred'].apply(list).reindex(Z_unique)
        for i, z in enumerate(Z_unique):
            preds = pred_by_z.iloc[i]
            if preds:
                ax.scatter([z] * len(preds), preds, s=30, alpha=0.7, color='red',
                           label='预测值' if i == 0 else "")

        # 绘制平均预测曲线
        avg_pred = subset.groupby('Z')['y_pred'].mean().reindex(Z_unique)
        ax.plot(Z_unique, avg_pred, 's--', color='red', linewidth=1.5, label='平均预测值')

        ax.set_xlabel('Z 轴位置 (mm)')
        ax.set_ylabel('磁栅误差 (mm)')
        ax.set_title(f'{state_name}\nR²={results[state_name]["r2"]:.4f}, RMSE={results[state_name]["rmse"]:.4f}mm')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('datas/F_model_All.png', dpi=300)
    plt.show()

    # 残差图：每个热状态
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()
    for idx, state_name in enumerate(hot_states):
        ax = axes2[idx]
        subset = df_train[df_train['state'] == state_name]
        y_true = subset['y_true'].values
        y_pred = subset['y_pred'].values
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.7)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('预测值 (mm)')
        ax.set_ylabel('残差 (mm)')
        ax.set_title(f'{state_name} 残差图')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('datas/F_model_residuals.png', dpi=300)
    plt.show()

    return results, df_train


def save_predictions(df_train, output_csv='datas/F_model_predictions.csv'):
    """保存每个样本的详细信息"""
    out_df = df_train.copy()
    out_df = out_df.sort_values(['state', 'Z', 'F'])
    out_df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"\n预测结果已保存至 {output_csv}")


def main():
    # 1. 读取数据
    df_raw = pd.read_csv('datasF.csv')
    df_raw.columns = df_raw.columns.str.strip()
    print("原始数据列名:", df_raw.columns.tolist())

    # 2. 构建统一训练数据集（不含时间 t）
    df_train = build_dataset(df_raw)
    print(f"总样本数: {len(df_train)}")

    # 3. 训练模型（不使用 t，无交互项）
    # degree=3 表示 Z 的三次多项式，可根据需要修改
    model, X, y = train_model(df_train, use_ridge=False, degree=3)

    # 输出模型系数
    feature_names = ['Intercept']
    for d in range(1, 4):
        feature_names.append(f'Z^{d}')
    feature_names += ['F', 'E_meas']
    print("\n模型系数:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"{name}: {coef:.6f}")

    # 4. 评估与绘图
    hot_states = ['冷机', '热机15分钟', '热机30分钟', '热机45分钟']
    results, df_train_pred = evaluate_and_plot(model, df_train, hot_states, degree=3)

    # 5. 保存预测结果
    save_predictions(df_train_pred, 'datas/F_model_predictions.csv')

    # 这个版本是使用所有热状态的值进行训练，Z的多项式 + 磁栅线性 + 进给线性
if __name__ == "__main__":
    main()