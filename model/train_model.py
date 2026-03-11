"""
PhishGuard AI - Advanced ML Training Pipeline
Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
Features: 20 URL-based features
Author: Vanshika Mittal| BTech/BCA Final Year Project
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, json, os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DATASET GENERATION
# ─────────────────────────────────────────────

def generate_dataset(n_samples=8000, random_state=42):
    np.random.seed(random_state)
    n_phishing = n_samples // 2
    n_legit    = n_samples - n_phishing

    def phishing_sample():
        return {
            'url_length':          np.random.randint(60, 200),
            'num_dots':            np.random.randint(3, 10),
            'num_hyphens':         np.random.randint(2, 8),
            'num_underscores':     np.random.randint(0, 5),
            'num_slash':           np.random.randint(4, 12),
            'num_question':        np.random.randint(1, 4),
            'num_equals':          np.random.randint(2, 8),
            'num_at':              np.random.randint(0, 2),
            'num_ampersand':       np.random.randint(1, 6),
            'num_digits':          np.random.randint(10, 40),
            'digit_ratio':         np.random.uniform(0.2, 0.6),
            'hostname_length':     np.random.randint(25, 80),
            'has_ip':              np.random.choice([0, 1], p=[0.3, 0.7]),
            'subdomain_count':     np.random.randint(2, 6),
            'has_https':           np.random.choice([0, 1], p=[0.7, 0.3]),
            'suspicious_words':    np.random.randint(2, 7),
            'path_length':         np.random.randint(30, 120),
            'num_path_components': np.random.randint(3, 8),
            'has_port':            np.random.choice([0, 1], p=[0.4, 0.6]),
            'tld_in_path':         np.random.choice([0, 1], p=[0.2, 0.8]),
            'label': 1
        }

    def legit_sample():
        return {
            'url_length':          np.random.randint(10, 55),
            'num_dots':            np.random.randint(1, 3),
            'num_hyphens':         np.random.randint(0, 2),
            'num_underscores':     np.random.randint(0, 1),
            'num_slash':           np.random.randint(1, 4),
            'num_question':        np.random.randint(0, 1),
            'num_equals':          np.random.randint(0, 2),
            'num_at':              0,
            'num_ampersand':       np.random.randint(0, 2),
            'num_digits':          np.random.randint(0, 8),
            'digit_ratio':         np.random.uniform(0.0, 0.12),
            'hostname_length':     np.random.randint(5, 22),
            'has_ip':              0,
            'subdomain_count':     np.random.randint(0, 2),
            'has_https':           np.random.choice([0, 1], p=[0.15, 0.85]),
            'suspicious_words':    np.random.randint(0, 2),
            'path_length':         np.random.randint(0, 35),
            'num_path_components': np.random.randint(0, 3),
            'has_port':            0,
            'tld_in_path':         0,
            'label': 0
        }

    data = [phishing_sample() for _ in range(n_phishing)] + \
           [legit_sample()    for _ in range(n_legit)]
    df = pd.DataFrame(data).sample(frac=1, random_state=random_state).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != 'label']
    noise = np.random.normal(0, 1.2, size=(len(df), len(feature_cols)))
    df[feature_cols] = (df[feature_cols].values + noise).clip(min=0)
    return df


# ─────────────────────────────────────────────
# 2. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def train_and_evaluate():
    print("=" * 65)
    print("   PhishGuard AI — Advanced ML Pipeline (4 Models)")
    print("=" * 65)

    df = generate_dataset(8000)
    feature_cols = [c for c in df.columns if c != 'label']
    X, y = df[feature_cols].values, df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=14,
                                           random_state=42, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                               max_depth=5, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(n_estimators=200, learning_rate=0.08,
                                  max_depth=6, random_state=42,
                                  eval_metric='logloss', verbosity=0))
        ]),
    }

    results, trained_models = {}, {}
    best_model_name, best_f1 = None, 0

    for name, model in models.items():
        print(f"\n▶  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy':  round(accuracy_score(y_test, y_pred)  * 100, 2),
            'precision': round(precision_score(y_test, y_pred) * 100, 2),
            'recall':    round(recall_score(y_test, y_pred)    * 100, 2),
            'f1':        round(f1_score(y_test, y_pred)        * 100, 2),
            'roc_auc':   round(roc_auc_score(y_test, y_prob)   * 100, 2),
            'cm':        confusion_matrix(y_test, y_pred).tolist(),
            'roc_curve': roc_curve(y_test, y_prob),
            'y_prob':    y_prob,
        }
        cv = cross_val_score(model, X, y, cv=5, scoring='f1')
        metrics['cv_f1_mean'] = round(cv.mean() * 100, 2)
        metrics['cv_f1_std']  = round(cv.std()  * 100, 2)

        results[name]       = metrics
        trained_models[name] = model

        print(f"   Accuracy : {metrics['accuracy']}%")
        print(f"   Precision: {metrics['precision']}%")
        print(f"   Recall   : {metrics['recall']}%")
        print(f"   F1-Score : {metrics['f1']}%")
        print(f"   ROC-AUC  : {metrics['roc_auc']}%")
        print(f"   CV  F1   : {metrics['cv_f1_mean']}% ± {metrics['cv_f1_std']}%")

        if metrics['f1'] > best_f1:
            best_f1, best_model_name = metrics['f1'], name

    print(f"\n✅  Best Model : {best_model_name}  (F1 = {best_f1}%)")

    # ── Save best model ──
    os.makedirs('model', exist_ok=True)
    with open('model/best_model.pkl', 'wb') as f:
        pickle.dump(trained_models[best_model_name], f)
    with open('model/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)

    # ── Feature importances (RF) ──
    for name in results:
        if 'Random Forest' in name:
            try:
                clf = trained_models[name].named_steps['clf']
                results[name]['feature_importances'] = clf.feature_importances_.tolist()
            except Exception:
                pass

    return results, feature_cols, y_test, best_model_name, trained_models


# ─────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────

C = {
    'bg': '#0D0F1A', 'panel': '#141829', 'panel2': '#0A0D18',
    'accent': '#00F5C8', 'danger': '#FF4C6B', 'safe': '#06D6A0',
    'warn': '#FFD166', 'blue': '#4B8EF1', 'purple': '#8B5CF6',
    'text': '#E8EAF6', 'muted': '#5C6080', 'grid': '#1E2240',
}

MODEL_COLORS = ['#00F5C8', '#8B5CF6', '#FF4C6B', '#FFD166']


def sax(ax, title=''):
    ax.set_facecolor(C['panel'])
    ax.tick_params(colors=C['text'], labelsize=8)
    ax.xaxis.label.set_color(C['text'])
    ax.yaxis.label.set_color(C['text'])
    for sp in ax.spines.values():
        sp.set_edgecolor(C['grid'])
    ax.grid(True, color=C['grid'], linewidth=0.4, alpha=0.6)
    if title:
        ax.set_title(title, color=C['accent'], fontsize=10,
                     fontweight='bold', pad=8, fontfamily='monospace')


def plot_all(results, feature_cols):
    names  = list(results.keys())
    short  = {'Logistic Regression': 'Logit', 'Random Forest': 'RF',
               'Gradient Boosting': 'GBM',   'XGBoost': 'XGB'}
    slabels = [short[n] for n in names]

    fig = plt.figure(figsize=(24, 16), facecolor=C['bg'])
    fig.suptitle('PhishGuard AI — ML Model Analysis Dashboard',
                 color=C['accent'], fontsize=20, fontweight='bold',
                 fontfamily='monospace', y=0.98)

    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.38,
                          left=0.05, right=0.97, top=0.93, bottom=0.05)

    # A — Metric Bar Chart
    ax1 = fig.add_subplot(gs[0, :2])
    sax(ax1, 'Model Performance Metrics')
    mlist  = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    mlabels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    bcolors = [C['accent'], C['purple'], C['warn'], C['danger'], C['safe']]
    x, w   = np.arange(len(names)), 0.14
    for i, (m, lbl, clr) in enumerate(zip(mlist, mlabels, bcolors)):
        vals = [results[n][m] for n in names]
        bars = ax1.bar(x + i * w, vals, w, label=lbl, color=clr, alpha=0.85)
        for b in bars:
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                     f'{b.get_height():.1f}', ha='center', va='bottom',
                     color=C['text'], fontsize=6)
    ax1.set_xticks(x + w * 2); ax1.set_xticklabels(slabels, color=C['text'])
    ax1.set_ylim(70, 103); ax1.set_ylabel('Score (%)', color=C['text'])
    ax1.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'],
               edgecolor=C['grid'], loc='lower right', ncol=5)

    # B — ROC Curves
    ax2 = fig.add_subplot(gs[0, 2:])
    sax(ax2, 'ROC Curves (All Models)')
    for name, clr in zip(names, MODEL_COLORS):
        fpr, tpr, _ = results[name]['roc_curve']
        ax2.plot(fpr, tpr, color=clr, lw=2.5,
                 label=f"{short[name]} (AUC={results[name]['roc_auc']:.1f}%)")
    ax2.plot([0,1],[0,1],'--', color=C['muted'], lw=1.5, label='Random Guess')
    ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
    ax2.legend(fontsize=8, facecolor=C['panel'], labelcolor=C['text'], edgecolor=C['grid'])

    # C — Confusion Matrices (all 4)
    for idx, (name, clr) in enumerate(zip(names, MODEL_COLORS)):
        ax = fig.add_subplot(gs[1, idx])
        cm = np.array(results[name]['cm'])
        ax.imshow(cm, cmap='RdYlGn', vmin=0, vmax=cm.max())
        ax.set_facecolor(C['panel'])
        for sp in ax.spines.values(): sp.set_edgecolor(clr)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Legit','Phish'], color=C['text'])
        ax.set_yticklabels(['Legit','Phish'], color=C['text'])
        ax.set_xlabel('Predicted', color=C['text'])
        ax.set_ylabel('Actual', color=C['text'])
        ax.set_title(f'CM — {short[name]}', color=clr, fontsize=10, fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='black', fontsize=12, fontweight='bold')

    # D — Feature Importance
    ax4 = fig.add_subplot(gs[2, :2])
    sax(ax4, 'Top 10 Feature Importance — Random Forest')
    fi = results.get('Random Forest', {}).get('feature_importances')
    if fi:
        fi = np.array(fi)
        top_idx  = np.argsort(fi)[-10:]
        top_vals = fi[top_idx]
        top_names = [feature_cols[i] for i in top_idx]
        bar_clrs  = [C['accent'] if v > 0.06 else C['purple'] for v in top_vals]
        ax4.barh(top_names, top_vals, color=bar_clrs, alpha=0.85)
        ax4.set_xlabel('Importance Score', color=C['text'])

    # E — CV F1 Scores
    ax5 = fig.add_subplot(gs[2, 2:])
    sax(ax5, 'Cross-Validation F1 Scores (5-Fold)')
    cv_means = [results[n]['cv_f1_mean'] for n in names]
    cv_stds  = [results[n]['cv_f1_std']  for n in names]
    bars = ax5.bar(slabels, cv_means, color=MODEL_COLORS, alpha=0.85, width=0.5)
    ax5.errorbar(slabels, cv_means, yerr=cv_stds,
                 fmt='none', color=C['text'], capsize=8, linewidth=2)
    for bar, val, std in zip(bars, cv_means, cv_stds):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + std + 0.4,
                 f'{val:.1f}±{std:.1f}%', ha='center', va='bottom',
                 color=C['text'], fontsize=9, fontweight='bold')
    ax5.set_ylabel('F1 Score (%)', color=C['text'])
    ax5.set_ylim(max(0, min(cv_means) - 10), 103)

    os.makedirs('static', exist_ok=True)
    plt.savefig('static/analysis.png', dpi=130,
                bbox_inches='tight', facecolor=C['bg'])
    print("\n📊  Analysis plot saved → static/analysis.png")


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    results, feature_cols, y_test, best_model_name, trained_models = train_and_evaluate()
    plot_all(results, feature_cols)

    # Save results JSON (strip non-serialisable keys)
    summary = {}
    for name, r in results.items():
        summary[name] = {k: v for k, v in r.items()
                         if k not in ('roc_curve', 'y_prob', 'feature_importances')}

    with open('model/results.json', 'w') as f:
        json.dump({'results': summary, 'best_model': best_model_name}, f, indent=2)

    print("\n✅  Training complete — model + plots + results saved!")
    print(f"    Best model : {best_model_name}")
    print(f"    Saved to   : model/best_model.pkl")
