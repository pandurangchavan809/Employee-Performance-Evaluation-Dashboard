import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


def main():
    data_path = 'employee_data.csv'
    model_dir = 'model'
    out_dir = 'static'
    out_plot = os.path.join(out_dir, 'model_comparison.png')
    out_model = os.path.join(model_dir, 'performance_model.pkl')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    data = pd.read_csv(data_path)
    features = ['attendance', 'task_efficiency', 'teamwork', 'initiative', 'project_quality']
    X = data[features]
    y = data['performance_score']

    n_samples = len(data)
    cv = min(5, n_samples) if n_samples >= 2 else 2

    # candidate models
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR()
    }

    results = {}
    print(f"Training and evaluating {len(models)} models with cv={cv} (samples={n_samples})")
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            results[name] = (scores.mean(), scores.std())
            print(f"{name}: mean R^2 = {scores.mean():.4f}, std = {scores.std():.4f}")
        except Exception as e:
            print(f"{name} failed during cross_val_score: {e}")
            results[name] = (float('-inf'), 0.0)

    # Select best model by mean score
    best_name = max(results.keys(), key=lambda k: results[k][0])
    best_score, best_std = results[best_name]
    best_model = models[best_name]

    # Train best model on full dataset
    best_model.fit(X, y)
    joblib.dump(best_model, out_model)

    print('\nModel comparison summary:')
    for name, (mean_s, std_s) in results.items():
        marker = '<-- selected' if name == best_name else ''
        print(f"{name:15s} mean R^2 = {mean_s:.4f} ± {std_s:.4f} {marker}")

    print(f"\nSaved best model '{best_name}' to {out_model}")

    # Plot comparison
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color='tab:blue')
    ax.set_ylabel('Mean R^2 (CV)')
    ax.set_title('Model comparison (higher is better)')
    ax.axhline(0, color='gray', linewidth=0.8)
    for i, b in enumerate(bars):
        if names[i] == best_name:
            b.set_color('tab:orange')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(out_plot, dpi=150)
    print(f"Saved model comparison plot to {out_plot}")


if __name__ == '__main__':
    main()
