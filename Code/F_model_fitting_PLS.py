"""
F_model_fitting_PLS.py
使用偏最小二乘回归（PLS），解决多重共线性问题，提升模型稳定性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import json
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def build_dataset(df):
    """构建训练数据集：每个样本为 (Z, F, E_meas) -> y (对应热状态的 F0 真值)"""
    Z = df['Z位置'].values.astype(float)

    # 定义热状态
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
                    'state': state_name,
                    'y_true': y_true[i]
                })
    return pd.DataFrame(records)


def train_model(df_train, n_components=3, degree=3):
    """
    使用偏最小二乘回归（PLS）训练模型
    degree: Z 的多项式次数
    n_components: 提取的主成分个数
    """
    # 构造特征矩阵：Z, Z^2, ..., Z^degree, F, E_meas （无截距项，PLS内部会自动中心化）
    X_list = []
    for _, row in df_train.iterrows():
        z = row['Z']
        features = []
        for d in range(1, degree + 1):
            features.append(z ** d)
        features.append(row['F'])
        features.append(row['E_meas'])
        X_list.append(features)

    X = np.array(X_list)
    y = df_train['y_true'].values.reshape(-1, 1)  # PLS要求y是2D

    # 创建PLS模型，自动对X和Y进行中心化（scale=True也会缩放标准差）
    model = PLSRegression(n_components=n_components, scale=True)
    model.fit(X, y)

    return model, X, y.ravel()


def get_feature_names(degree=3):
    """返回与 build_feature 一致的特征名称列表。"""
    names = [f'Z^{d}' for d in range(1, degree + 1)]
    names += ['F', 'E_meas']
    return names


def build_single_feature_vector(z, f, e_meas, degree=3):
    """构造单样本特征 [Z, Z^2, ..., Z^degree, F, E_meas]。"""
    features = [z ** d for d in range(1, degree + 1)]
    features += [f, e_meas]
    return np.array(features, dtype=float)


def export_cpp_deployment_params(model, degree=3, output_json='datas/pls_cpp_params.json'):
    """
    导出 C++ 部署所需的全部参数，并在控制台打印。

    C++ 预测公式（与 sklearn PLSRegression.predict 一致）:
        y_hat = intercept + sum_i (x[i] - x_mean[i]) * coef[i]

    特征顺序（degree=3）:
        x[0]=Z, x[1]=Z^2, x[2]=Z^3, x[3]=F, x[4]=E_meas
    """
    feature_names = get_feature_names(degree)
    x_mean = np.asarray(model._x_mean, dtype=float).ravel()
    coef = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.asarray(model.intercept_, dtype=float).ravel()[0])

    if x_mean.shape[0] != len(feature_names) or coef.shape[0] != len(feature_names):
        raise ValueError(
            f"特征维度不一致: x_mean={x_mean.shape}, coef={coef.shape}, "
            f"期望 {len(feature_names)} 维"
        )

    params = {
        'model_type': 'PLSRegression',
        'degree': degree,
        'n_components': int(model.n_components),
        'scale': bool(model.scale),
        'feature_names': feature_names,
        'x_mean': x_mean.tolist(),
        'coef': coef.tolist(),
        'intercept': intercept,
        'predict_formula': 'y = intercept + sum((x[i] - x_mean[i]) * coef[i])',
        'feature_build_note': 'x[0]=Z, x[1]=Z^2, ... x[degree-1]=Z^degree, x[degree]=F, x[degree+1]=E_meas',
    }

    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("PLS 模型 C++ 部署参数（训练后导出）")
    print("=" * 70)
    print(f"特征维度 n_features = {len(feature_names)}")
    print(f"主成分数 n_components = {model.n_components}, scale = {model.scale}")
    print("\n【特征顺序】（C++ 数组 x[] 必须与之一致）")
    for i, name in enumerate(feature_names):
        print(f"  x[{i}] = {name}")

    print("\n【x_mean】（model._x_mean，预测前每个特征需减去对应均值）")
    print("  Python list:", x_mean.tolist())
    for i, (name, val) in enumerate(zip(feature_names, x_mean)):
        print(f"  x_mean[{i}] ({name:8s}) = {val:.16e}")

    print("\n【coef_】（model.coef_.ravel()，与 x_mean 按索引一一对应）")
    print("  Python list:", coef.tolist())
    for i, (name, val) in enumerate(zip(feature_names, coef)):
        print(f"  coef[{i}] ({name:8s}) = {val:.16e}")

    print("\n【intercept_】（model.intercept_，等于 _y_mean）")
    print(f"  intercept = {intercept:.16e}")

    print("\n【C++ 预测伪代码】")
    print("  double predict_pls(const double x[n_features]) {")
    print("      double y = intercept;")
    print("      for (int i = 0; i < n_features; ++i)")
    print("          y += (x[i] - x_mean[i]) * coef[i];")
    print("      return y;")
    print("  }")
    print("  // 构造特征: x[0]=Z, x[1]=Z*Z, x[2]=Z*Z*Z, x[3]=F, x[4]=E_meas")

    print("\n【C++ 可直接拷贝的常量数组】")
    print(f"  static const int PLS_N_FEATURES = {len(feature_names)};")
    print(f"  static const double PLS_INTERCEPT = {intercept:.16e};")
    print("  static const double PLS_X_MEAN[PLS_N_FEATURES] = {")
    for i, val in enumerate(x_mean):
        suffix = "," if i < len(x_mean) - 1 else ""
        print(f"      {val:.16e}{suffix}  // {feature_names[i]}")
    print("  };")
    print("  static const double PLS_COEF[PLS_N_FEATURES] = {")
    for i, val in enumerate(coef):
        suffix = "," if i < len(coef) - 1 else ""
        print(f"      {val:.16e}{suffix}  // {feature_names[i]}")
    print("  };")

    print(f"\n参数已保存至 JSON: {output_json}")
    print("=" * 70 + "\n")

    return params


def predict_with_coef(model, X):
    """
    使用 model.coef_ / model.intercept_ 手动预测，与 sklearn PLSRegression.predict 一致。

    关键（新版 sklearn）:
      predict 内部先对 X 做中心化: X_c = X - model._x_mean
      再计算: y_hat = X_c @ coef_.T + intercept_
      其中 intercept_ = model._y_mean，coef_ 已吸收 scale 标准化，predict 中不再除以 _x_std。

    错误写法（会导致全体样本偏差常数）:
      y_hat = X @ coef_ + intercept_   # 未中心化 X
    """
    X = np.asarray(X, dtype=float)
    X_centered = X - model._x_mean
    return (X_centered @ np.asarray(model.coef_).T + np.asarray(model.intercept_)).ravel()


def predict_with_coef_wrong_raw_x(model, X):
    """错误示范：直接 X @ coef + intercept（未中心化），用于对比验证。"""
    X = np.asarray(X, dtype=float)
    return (X @ np.asarray(model.coef_).ravel() + np.asarray(model.intercept_).ravel()[0])


def verify_coef_vs_predict(model, X, rtol=1e-12, atol=1e-12):
    """
    验证 model.coef_ 手动预测与 model.predict 是否一致（PLS 版本）。

    同时对比「未中心化 X」的错误写法，帮助定位与 intercept_ 量级相同的系统偏差。
    """
    X = np.asarray(X, dtype=float)

    y_sklearn = np.asarray(model.predict(X)).ravel()
    y_correct = predict_with_coef(model, X)
    y_wrong = predict_with_coef_wrong_raw_x(model, X)

    x_mean = np.asarray(model._x_mean).ravel()
    coef_ravel = np.asarray(model.coef_).ravel()
    bias_from_missing_center = float(x_mean @ coef_ravel)

    diff_correct = y_correct - y_sklearn
    diff_wrong = y_wrong - y_sklearn

    is_close = np.allclose(y_correct, y_sklearn, rtol=rtol, atol=atol)
    is_wrong_close = np.allclose(y_wrong, y_sklearn, rtol=rtol, atol=atol)

    intercept_val = float(np.asarray(model.intercept_).ravel()[0])
    y_mean_val = float(np.asarray(model._y_mean).ravel()[0])

    report = {
        'is_close': is_close,
        'is_wrong_close': is_wrong_close,
        'max_abs_diff_correct': float(np.max(np.abs(diff_correct))),
        'max_abs_diff_wrong': float(np.max(np.abs(diff_wrong))),
        'mean_abs_diff_wrong': float(np.mean(np.abs(diff_wrong))),
        'bias_from_missing_center': bias_from_missing_center,
        'n_samples': len(y_sklearn),
        'intercept_': intercept_val,
        '_y_mean': y_mean_val,
    }

    print("\n========== PLS: coef_ 手动预测 vs model.predict 一致性验证 ==========")
    print(f"样本数: {report['n_samples']}")
    print(f"scale={getattr(model, 'scale', 'N/A')}, n_components={model.n_components}")
    print(f"intercept_ = {intercept_val:.16e}  (= _y_mean，PLS 截距为 y 的均值)")
    print(f"_x_mean = {x_mean}")
    print(f"coef_.ravel() = {coef_ravel}")
    print(f"\n【正确】手动公式: y_hat = (X - _x_mean) @ coef_.T + intercept_")
    print(f"  np.allclose: {is_close}, max|diff| = {report['max_abs_diff_correct']:.16e} mm")
    print(f"\n【错误】未中心化: y_hat = X @ coef_ + intercept_")
    print(f"  np.allclose: {is_wrong_close}, max|diff| = {report['max_abs_diff_wrong']:.16e} mm")
    print(f"  mean|diff|  = {report['mean_abs_diff_wrong']:.16e} mm  (各样本几乎相同 → 系统偏差)")
    print(f"  理论偏差 x_mean @ coef = {bias_from_missing_center:.16e} mm")

    if is_close:
        print("\n结论: 使用 (X - _x_mean) @ coef_.T + intercept_ 与 model.predict 完全一致。")
    else:
        worst = int(np.argmax(np.abs(diff_correct)))
        print(f"\n【仍不一致】正确写法差异最大样本 idx={worst}")
        print(f"  manual={y_correct[worst]:.16f}, sklearn={y_sklearn[worst]:.16f}")

    if not is_wrong_close and np.allclose(
        report['max_abs_diff_wrong'], abs(bias_from_missing_center), rtol=1e-6, atol=1e-6
    ):
        print("\n原因分析: 旧代码未对 X 中心化。")
        print("  predict 内部: X -= _x_mean，再 X @ coef_.T + _y_mean")
        print("  旧手动公式:   X @ coef_ + intercept_")
        print("  两者相差常数: x_mean @ coef_ ≈ 你日志中的 ~0.0587 mm")
        print("  这不是 coef_ 错误，而是漏做了 X 减均值这一步。")

    print("=====================================================================\n")
    return report


def evaluate_and_plot(model, df_train, hot_states, degree=3):
    """评估模型并绘制各热状态的结果"""
    # 重新构造特征（无截距项）
    X_all = []
    for _, row in df_train.iterrows():
        z = row['Z']
        features = []
        for d in range(1, degree + 1):
            features.append(z ** d)
        features.append(row['F'])
        features.append(row['E_meas'])
        X_all.append(features)
    X_all = np.array(X_all)

    # 方法1：使用model.predict进行预测
    # y_pred = model.predict(X_all).ravel()  # 预测值

    # 方法2：使用 model.coef_ / intercept_ 手动预测（须先中心化 X）
    y_pred = predict_with_coef(model, X_all)

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

    # 绘图：每个热状态一张子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    Z_unique = np.sort(df_train['Z'].unique())

    for idx, state_name in enumerate(hot_states):
        ax = axes[idx]
        subset = df_train[df_train['state'] == state_name]
        # 真实F0曲线
        true_by_z = subset.groupby('Z')['y_true'].first().reindex(Z_unique)
        ax.plot(Z_unique, true_by_z, 'o-', color='black', linewidth=2, label='真实 F0')

        # 各个进给速度下的测量值
        speeds = [2000, 5000, 8000, 10000]
        for speed in speeds:
            speed_data = subset[subset['F'] == speed]
            if not speed_data.empty:
                ax.scatter(speed_data['Z'], speed_data['E_meas'], s=20, alpha=0.6,
                           label=f'测量值 F={speed}', marker='x')

        # 每个Z下的预测值散点
        pred_by_z = subset.groupby('Z')['y_pred'].apply(list).reindex(Z_unique)
        for i, z in enumerate(Z_unique):
            preds = pred_by_z.iloc[i]
            if preds:
                ax.scatter([z] * len(preds), preds, s=30, alpha=0.7, color='red',
                           label='预测值' if i == 0 else "")

        # 平均预测曲线
        avg_pred = subset.groupby('Z')['y_pred'].mean().reindex(Z_unique)
        ax.plot(Z_unique, avg_pred, 's--', color='red', linewidth=1.5, label='平均预测值')

        ax.set_xlabel('Z 轴位置 (mm)')
        ax.set_ylabel('磁栅误差 (mm)')
        ax.set_title(f'{state_name}\nR²={results[state_name]["r2"]:.4f}, RMSE={results[state_name]["rmse"]:.4f}mm')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('datas/F_model_PLS_All.png', dpi=300)
    plt.show()

    # 残差图
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
    plt.savefig('datas/F_model_PLS_residuals.png', dpi=300)
    plt.show()

    return results, df_train


def save_predictions(df_train, output_csv='datas/F_model_PLS_predictions.csv'):
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

    # 2. 构建统一训练数据集
    df_train = build_dataset(df_raw)
    print(f"总样本数: {len(df_train)}")

    # 3. 训练PLS模型（可调整主成分个数）
    degree = 3
    n_components = 3  # 通常取3即可，可根据交叉验证调整
    model, X, y = train_model(df_train, n_components=n_components, degree=degree)

    # 输出模型系数（对应原始特征，无截距）
    feature_names = get_feature_names(degree)
    print("\nPLS模型系数（对应原始特征）:")
    for name, c in zip(feature_names, model.coef_.ravel()):
        print(f"{name}: {c:.6f}")

    # 导出 C++ 部署参数：x_mean、coef_、intercept_
    export_cpp_deployment_params(model, degree=degree, output_json='datas/pls_cpp_params.json')

    # 4. 评估与绘图
    hot_states = ['冷机', '热机15分钟', '热机30分钟', '热机45分钟']
    results, df_train_pred = evaluate_and_plot(model, df_train, hot_states, degree=degree)

    # 5. 保存预测结果
    save_predictions(df_train_pred, 'datas/F_model_PLS_predictions.csv')

    # 打印提取的主成分个数
    print(f"\nPLS模型使用的主成分个数: {n_components}")


if __name__ == "__main__":
    main()