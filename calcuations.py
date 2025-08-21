import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

seed = 4

plt.style.use('default')
sns.set_palette("husl")

def compute_mmd(X, Y, gamma=1.0):
    """Compute Maximum Mean Discrepancy using RBF kernel"""
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    mmd_squared = XX.mean() + YY.mean() - 2 * XY.mean()
    return np.sqrt(max(mmd_squared, 0))

def compute_wasserstein2(X, Y, sample_size=1000):
    """
    Compute Wasserstein-2 distance between two distributions
    Uses PCA for dimensionality reduction and sampling for efficiency
    """
    if len(X) > sample_size:
        idx_X = np.random.choice(len(X), sample_size, replace=False)
        X = X[idx_X]
    if len(Y) > sample_size:
        idx_Y = np.random.choice(len(Y), sample_size, replace=False)
        Y = Y[idx_Y]

    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X_combined = np.vstack([X, Y])
        X_combined_pca = pca.fit_transform(X_combined)
        X = X_combined_pca[:len(X)]
        Y = X_combined_pca[len(X):]

    cost_matrix = pairwise_distances(X, Y, metric='euclidean')

    min_size = min(len(X), len(Y))
    if len(X) != len(Y):
        if len(X) > len(Y):
            idx = np.random.choice(len(X), min_size, replace=False)
            X = X[idx]
        else:
            idx = np.random.choice(len(Y), min_size, replace=False)
            Y = Y[idx]
        cost_matrix = pairwise_distances(X, Y, metric='euclidean')

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    wasserstein2 = np.mean(cost_matrix[row_ind, col_ind] ** 2)

    return np.sqrt(wasserstein2)

def compute_ks_multivariate(X, Y, num_projections=100):
    """
    Compute multivariate KS test using random projections
    Projects high-dimensional data onto random 1D directions
    """
    np.random.seed(seed)  

    scaler = StandardScaler()
    X_combined = np.vstack([X, Y])
    X_combined_scaled = scaler.fit_transform(X_combined)
    X_scaled = X_combined_scaled[:len(X)]
    Y_scaled = X_combined_scaled[len(X):]

    p_values = []
    ks_stats = []

    for _ in range(num_projections):
        direction = np.random.randn(X_scaled.shape[1])
        direction = direction / np.linalg.norm(direction)

        X_proj = X_scaled @ direction
        Y_proj = Y_scaled @ direction

        ks_stat, p_val = ks_2samp(X_proj, Y_proj)
        ks_stats.append(ks_stat)
        p_values.append(p_val)

    avg_ks_stat = np.mean(ks_stats)
    avg_p_value = np.mean(p_values)

    return avg_ks_stat, avg_p_value

def bootstrap_mmd(X_real, X_synth, num_bootstrap=1000, sample_limit=3000):
    """Bootstrap MMD test with null hypothesis testing"""
    np.random.seed(seed)

    if sample_limit and len(X_real) > sample_limit:
        idx_real = np.random.choice(len(X_real), sample_limit, replace=False)
        idx_synth = np.random.choice(len(X_synth), sample_limit, replace=False)
        X_real = X_real[idx_real]
        X_synth = X_synth[idx_synth]

    scaler = StandardScaler().fit(X_real)
    X_real = scaler.transform(X_real)
    X_synth = scaler.transform(X_synth)

    X_combined = np.vstack([X_real, X_synth])
    dists = pairwise_distances(X_combined, metric='euclidean')
    median_dist = np.median(dists)
    gamma = 1.0 / (2 * (median_dist ** 2 + 1e-8))  

    observed_mmd = compute_mmd(X_real, X_synth, gamma)

    n = len(X_real)
    bootstrap_stats = []
    for _ in tqdm(range(num_bootstrap), desc="Bootstrapping MMD"):
        perm = np.random.permutation(len(X_combined))
        X1 = X_combined[perm[:n]]
        X2 = X_combined[perm[n:]]
        stat = compute_mmd(X1, X2, gamma)
        bootstrap_stats.append(stat)

    p_val = (np.sum(np.array(bootstrap_stats) >= observed_mmd) + 1) / (num_bootstrap + 1)
    return observed_mmd, p_val, bootstrap_stats

def plot_distribution_comparison(X_real, X_synth, feature_name, save_plots=False):
    """
    Create comprehensive visualization of distribution differences
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Distribution Comparison: {feature_name} Features', fontsize=16, fontweight='bold')

    print(f"    📊 Creating PCA projection...")
    pca = PCA(n_components=2)
    X_combined = np.vstack([X_real, X_synth])
    X_combined_pca = pca.fit_transform(X_combined)
    X_real_pca = X_combined_pca[:len(X_real)]
    X_synth_pca = X_combined_pca[len(X_real):]

    axes[0, 0].scatter(X_real_pca[:, 0], X_real_pca[:, 1], alpha=0.6, label='Real', s=20)
    axes[0, 0].scatter(X_synth_pca[:, 0], X_synth_pca[:, 1], alpha=0.6, label='Synthetic', s=20)
    axes[0, 0].set_title('PCA 2D Projection')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    print(f"    📊 Creating t-SNE projection...")
    sample_size = min(1000, len(X_real), len(X_synth))
    if len(X_real) > sample_size:
        idx_real = np.random.choice(len(X_real), sample_size, replace=False)
        idx_synth = np.random.choice(len(X_synth), sample_size, replace=False)
        X_real_sample = X_real[idx_real]
        X_synth_sample = X_synth[idx_synth]
    else:
        X_real_sample = X_real
        X_synth_sample = X_synth

    X_sample_combined = np.vstack([X_real_sample, X_synth_sample])

    if X_sample_combined.shape[1] > 50:
        pca_pre = PCA(n_components=50)
        X_sample_combined = pca_pre.fit_transform(X_sample_combined)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sample_combined)//4))
    X_tsne = tsne.fit_transform(X_sample_combined)
    X_real_tsne = X_tsne[:len(X_real_sample)]
    X_synth_tsne = X_tsne[len(X_real_sample):]

    axes[0, 1].scatter(X_real_tsne[:, 0], X_real_tsne[:, 1], alpha=0.6, label='Real', s=20)
    axes[0, 1].scatter(X_synth_tsne[:, 0], X_synth_tsne[:, 1], alpha=0.6, label='Synthetic', s=20)
    axes[0, 1].set_title('t-SNE 2D Projection')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    print(f"    📊 Creating feature magnitude distribution...")
    real_norms = np.linalg.norm(X_real, axis=1)
    synth_norms = np.linalg.norm(X_synth, axis=1)

    axes[0, 2].hist(real_norms, bins=50, alpha=0.7, label='Real', density=True)
    axes[0, 2].hist(synth_norms, bins=50, alpha=0.7, label='Synthetic', density=True)
    axes[0, 2].set_title('Feature Vector Magnitude Distribution')
    axes[0, 2].set_xlabel('L2 Norm')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    print(f"    📊 Creating top features comparison...")
    feature_vars = np.var(X_combined, axis=0)
    top_features = np.argsort(feature_vars)[-10:]

    real_top = X_real[:, top_features].mean(axis=0)
    synth_top = X_synth[:, top_features].mean(axis=0)

    x_pos = np.arange(len(top_features))
    width = 0.35

    axes[1, 0].bar(x_pos - width/2, real_top, width, label='Real', alpha=0.8)
    axes[1, 0].bar(x_pos + width/2, synth_top, width, label='Synthetic', alpha=0.8)
    axes[1, 0].set_title('Top 10 Most Varying Features (Mean)')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'F{i}' for i in top_features])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    print(f"    📊 Creating pairwise distance distribution...")
    sample_size_dist = min(500, len(X_real), len(X_synth))
    if len(X_real) > sample_size_dist:
        idx_real = np.random.choice(len(X_real), sample_size_dist, replace=False)
        idx_synth = np.random.choice(len(X_synth), sample_size_dist, replace=False)
        X_real_dist = X_real[idx_real]
        X_synth_dist = X_synth[idx_synth]
    else:
        X_real_dist = X_real
        X_synth_dist = X_synth

    real_distances = pdist(X_real_dist, metric='euclidean')
    synth_distances = pdist(X_synth_dist, metric='euclidean')

    axes[1, 1].hist(real_distances, bins=50, alpha=0.7, label='Real', density=True)
    axes[1, 1].hist(synth_distances, bins=50, alpha=0.7, label='Synthetic', density=True)
    axes[1, 1].set_title('Pairwise Distance Distribution')
    axes[1, 1].set_xlabel('Euclidean Distance')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    print(f"    📊 Creating dimensionality statistics...")
    real_stats = np.array([
        np.mean(X_real, axis=1),  
        np.std(X_real, axis=1),  
        np.max(X_real, axis=1),   
        np.min(X_real, axis=1)     
    ]).T

    synth_stats = np.array([
        np.mean(X_synth, axis=1),
        np.std(X_synth, axis=1),
        np.max(X_synth, axis=1),
        np.min(X_synth, axis=1)
    ]).T

    data_for_box = []
    labels = []

    for i, stat_name in enumerate(['Mean', 'Std', 'Max', 'Min']):
        data_for_box.extend([real_stats[:, i], synth_stats[:, i]])
        labels.extend([f'Real\n{stat_name}', f'Synth\n{stat_name}'])

    axes[1, 2].boxplot(data_for_box, labels=labels)
    axes[1, 2].set_title('Per-Sample Statistics Distribution')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig(f'{feature_name}_distribution_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()

    print(f"\n    📈 Summary Statistics for {feature_name}:")
    print(f"    Real data: Shape {X_real.shape}, Mean: {X_real.mean():.4f}, Std: {X_real.std():.4f}")
    print(f"    Synthetic data: Shape {X_synth.shape}, Mean: {X_synth.mean():.4f}, Std: {X_synth.std():.4f}")
    print(f"    Mean difference: {abs(X_real.mean() - X_synth.mean()):.4f}")
    print(f"    Std difference: {abs(X_real.std() - X_synth.std()):.4f}")

def plot_test_results(all_results, save_plots=False):
    """
    Create visualization of test results across all feature types
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Statistical Test Results Across Feature Types', fontsize=16, fontweight='bold')

    feature_names = list(all_results.keys())
    test_names = ['MMD', 'Wasserstein-2', 'KS']

    stats_data = []
    for feature in feature_names:
        for test in test_names:
            stats_data.append(all_results[feature][test]['statistic'])

    stats_array = np.array(stats_data).reshape(len(feature_names), len(test_names))

    im1 = axes[0].imshow(stats_array, cmap='viridis', aspect='auto')
    axes[0].set_title('Test Statistics Heatmap')
    axes[0].set_xlabel('Test Type')
    axes[0].set_ylabel('Feature Type')
    axes[0].set_xticks(range(len(test_names)))
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_xticklabels(test_names)
    axes[0].set_yticklabels(feature_names)

    for i in range(len(feature_names)):
        for j in range(len(test_names)):
            text = axes[0].text(j, i, f'{stats_array[i, j]:.3f}',
                              ha="center", va="center", color="white", fontweight='bold')

    plt.colorbar(im1, ax=axes[0], label='Statistic Value')

    pval_data = []
    for feature in feature_names:
        for test in test_names:
            pval_data.append(all_results[feature][test]['p_value'])

    pval_array = np.array(pval_data).reshape(len(feature_names), len(test_names))

    im2 = axes[1].imshow(pval_array, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.1)
    axes[1].set_title('P-values Heatmap')
    axes[1].set_xlabel('Test Type')
    axes[1].set_ylabel('Feature Type')
    axes[1].set_xticks(range(len(test_names)))
    axes[1].set_yticks(range(len(feature_names)))
    axes[1].set_xticklabels(test_names)
    axes[1].set_yticklabels(feature_names)

    for i in range(len(feature_names)):
        for j in range(len(test_names)):
            p_val = pval_array[i, j]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            text = axes[1].text(j, i, f'{p_val:.3f}\n{significance}',
                              ha="center", va="center", color="black", fontweight='bold')

    plt.colorbar(im2, ax=axes[1], label='P-value')

    significance_counts = np.zeros((len(feature_names), 2)) 
    for i, feature in enumerate(feature_names):
        sig_count = 0
        for test in test_names:
            if all_results[feature][test]['p_value'] < 0.05:
                sig_count += 1
        significance_counts[i, 0] = sig_count
        significance_counts[i, 1] = len(test_names) - sig_count

    x = np.arange(len(feature_names))
    width = 0.35

    axes[2].bar(x, significance_counts[:, 0], width, label='Significant (p < 0.05)', color='red', alpha=0.7)
    axes[2].bar(x, significance_counts[:, 1], width, bottom=significance_counts[:, 0],
                label='Not Significant (p ≥ 0.05)', color='blue', alpha=0.7)

    axes[2].set_title('Significance Summary')
    axes[2].set_xlabel('Feature Type')
    axes[2].set_ylabel('Number of Tests')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(feature_names)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig('test_results_summary.png', dpi=300, bbox_inches='tight')

    plt.show()
def bootstrap_wasserstein2(X_real, X_synth, num_bootstrap=500, sample_limit=1000):
    """Bootstrap Wasserstein-2 test with null hypothesis testing"""
    np.random.seed(seed)

    if sample_limit and len(X_real) > sample_limit:
        idx_real = np.random.choice(len(X_real), sample_limit, replace=False)
        idx_synth = np.random.choice(len(X_synth), sample_limit, replace=False)
        X_real = X_real[idx_real]
        X_synth = X_synth[idx_synth]

    scaler = StandardScaler().fit(X_real)
    X_real = scaler.transform(X_real)
    X_synth = scaler.transform(X_synth)

    observed_w2 = compute_wasserstein2(X_real, X_synth)

    X_combined = np.vstack([X_real, X_synth])
    n = len(X_real)
    bootstrap_stats = []

    for _ in tqdm(range(num_bootstrap), desc="Bootstrapping W2"):
        perm = np.random.permutation(len(X_combined))
        X1 = X_combined[perm[:n]]
        X2 = X_combined[perm[n:]]
        stat = compute_wasserstein2(X1, X2)
        bootstrap_stats.append(stat)

    p_val = (np.sum(np.array(bootstrap_stats) >= observed_w2) + 1) / (num_bootstrap + 1)
    return observed_w2, p_val, bootstrap_stats

def run_comprehensive_tests(X_real, X_synth, feature_name,
                          mmd_bootstrap=2000, w2_bootstrap=1000,
                          ks_projections=500, sample_limit=2000):
    """
    Run all three tests (MMD, Wasserstein-2, KS) on a feature set
    """
    print(f"\n🔍 Running comprehensive tests for {feature_name} features...")

    results = {}

    print(f"  📊 MMD Test...")
    mmd_val, mmd_p, _ = bootstrap_mmd(X_real, X_synth,
                                    num_bootstrap=mmd_bootstrap,
                                    sample_limit=sample_limit)
    results['MMD'] = {'statistic': mmd_val, 'p_value': mmd_p}

    print(f"  📊 Wasserstein-2 Test...")
    w2_val, w2_p, _ = bootstrap_wasserstein2(X_real, X_synth,
                                            num_bootstrap=w2_bootstrap,
                                            sample_limit=sample_limit)
    results['Wasserstein-2'] = {'statistic': w2_val, 'p_value': w2_p}

    print(f"  📊 Kolmogorov-Smirnov Test...")
    ks_stat, ks_p = compute_ks_multivariate(X_real, X_synth,
                                          num_projections=ks_projections)
    results['KS'] = {'statistic': ks_stat, 'p_value': ks_p}

    return results

if __name__ == "__main__":
    feature_sets = {
        "N-gram": (X_real, X_synth),
        "POS": (X_real_pos, X_synth_pos),
        "Embeddings": (real_abstracts_emb, synth_abstracts_emb)
    }

    all_results = {}

    for feature_name, (X_r, X_s) in feature_sets.items():
        results = run_comprehensive_tests(X_r, X_s, feature_name)
        all_results[feature_name] = results

    plot_test_results(all_results, save_plots=False)

    print("\n" + "="*80)
    print("📊 COMPREHENSIVE DISTRIBUTION TESTING RESULTS")
    print("="*80)

    print(f"\n{'Feature Type':<12} {'Test':<15} {'Statistic':<12} {'P-value':<10} {'Significant':<12}")
    print("-" * 70)

    for feature_name, feature_results in all_results.items():
        for test_name, test_results in feature_results.items():
            stat = test_results['statistic']
            p_val = test_results['p_value']
            significant = "Yes" if p_val < 0.05 else "No"

            print(f"{feature_name:<12} {test_name:<15} {stat:<12.4f} {p_val:<10.5f} {significant:<12}")

    print("\n" + "="*80)
    print("📝 INTERPRETATION GUIDE:")
    print("="*80)
    print("• Lower p-values (< 0.05) indicate significant differences between distributions")
    print("• Higher statistics generally indicate larger differences between distributions")
    print("• MMD: Maximum Mean Discrepancy - measures difference in distribution means")
    print("• Wasserstein-2: Earth Mover's Distance - measures cost of transforming one distribution to another")
    print("• KS: Kolmogorov-Smirnov - measures maximum difference between cumulative distributions")
    print("="*80)
