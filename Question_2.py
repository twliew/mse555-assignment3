import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Part A: Build the clustering model and apply K-means for the cumulative session trajectories
# This section loads scored session data, constructs client trajectories, and clusters clients
# by their cumulative progress patterns.
def load_and_prepare_data():
    """Load scored notes and prepare cumulative trajectories"""
    # Load the scored data
    df = pd.read_csv('output/q1/scored_notes.csv')

    # Pivot to get trajectories: rows = clients, columns = sessions
    trajectories = df.pivot(index='client_id', columns='session', values='score')

    # Fill any missing sessions with 0 
    trajectories = trajectories.fillna(0)

    # Create cumulative trajectories 
    cumulative_trajectories = trajectories.cumsum(axis=1)

    return trajectories, cumulative_trajectories

def apply_clustering(cumulative_trajectories, n_clusters=4):
    """Apply K-means clustering to the cumulative trajectories"""
    # Standardize the data for better clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cumulative_trajectories)

    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(scaled_data, clusters)

    return clusters, kmeans, scaler, silhouette_avg

def create_spaghetti_plots(trajectories, cumulative_trajectories, clusters, n_clusters=4):
    """Create spaghetti plots showing individual trajectories and cluster means"""

    # Set up the plotting area
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Colors for clusters
    colors = sns.color_palette("husl", n_clusters)

    # Plot 1: Raw session scores
    for i in range(n_clusters):
        cluster_clients = trajectories.index[clusters == i]
        cluster_data = trajectories.loc[cluster_clients]

        # Plot individual trajectories (spaghetti)
        for client in cluster_clients:
            axes[0].plot(range(1, 12), cluster_data.loc[client], color=colors[i], alpha=0.3, linewidth=1)

        # Plot cluster mean
        mean_trajectory = cluster_data.mean()
        axes[0].plot(range(1, 12), mean_trajectory, color=colors[i], linewidth=3, label=f'Cluster {i+1} (n={len(cluster_clients)})')

    axes[0].set_title('Session Progress Scores by Cluster', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Session Number')
    axes[0].set_ylabel('Progress Score (0-3)')
    axes[0].set_xticks(range(1, 12))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative trajectories
    for i in range(n_clusters):
        cluster_clients = cumulative_trajectories.index[clusters == i]
        cluster_data = cumulative_trajectories.loc[cluster_clients]

        # Plot individual trajectories (spaghetti)
        for client in cluster_clients:
            axes[1].plot(range(1, 12), cluster_data.loc[client], color=colors[i], alpha=0.3, linewidth=1)

        # Plot cluster mean
        mean_trajectory = cluster_data.mean()
        axes[1].plot(range(1, 12), mean_trajectory, color=colors[i], linewidth=3, label=f'Cluster {i+1}')

    axes[1].set_title('Cumulative Progress Trajectories by Cluster', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Session Number')
    axes[1].set_ylabel('Cumulative Progress Score')
    axes[1].set_xticks(range(1, 12))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/q2/clustering_spaghetti_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def analyze_clusters(cumulative_trajectories, clusters, n_clusters=4):
    """Analyze and summarize cluster characteristics"""
    print(f"\n=== Clustering Analysis (K={n_clusters}) ===")

    cluster_summary = []

    for i in range(n_clusters):
        cluster_clients = cumulative_trajectories.index[clusters == i]
        cluster_data = cumulative_trajectories.loc[cluster_clients]

        # Calculate statistics
        final_scores = cluster_data.iloc[:, -1]  # Last session cumulative score
        mean_final = final_scores.mean()
        std_final = final_scores.std()
        min_final = final_scores.min()
        max_final = final_scores.max()

        # Calculate trajectory characteristics
        initial_score = cluster_data.iloc[:, 0].mean()  # Average score in session 1
        growth_rate = (final_scores - cluster_data.iloc[:, 0]).mean()  # Average total growth

        cluster_summary.append({
            'cluster': i+1,
            'n_clients': len(cluster_clients),
            'mean_final_score': mean_final,
            'std_final_score': std_final,
            'min_final_score': min_final,
            'max_final_score': max_final,
            'initial_score': initial_score,
            'growth_rate': growth_rate
        })

        print(f"\nCluster {i+1} (n={len(cluster_clients)} clients):")
        print(f"mean_final={mean_final:.2f}, std_final={std_final:.2f}, min_final={min_final}, max_final={max_final}")
        print(f"initial_score={initial_score:.2f}, growth_rate={growth_rate:.2f}")

    return pd.DataFrame(cluster_summary)

def compute_stopping_points(cumulative_trajectories):
    """Compute stopping points t*_i for each client using 90% of total progress rule"""
    stopping_points = {}

    for client_id in cumulative_trajectories.index:
        trajectory = cumulative_trajectories.loc[client_id]
        total_progress = trajectory.iloc[-1]  # Final cumulative score
        threshold = 0.9 * total_progress

        # Find earliest session where cumulative progress >= 90% of total
        stopping_point = None
        for session in range(1, len(trajectory) + 1):
            if trajectory.iloc[session - 1] >= threshold:
                stopping_point = session
                break

        # If never reaches 90%, set to max sessions
        if stopping_point is None:
            stopping_point = len(trajectory)

        stopping_points[client_id] = stopping_point

    return pd.Series(stopping_points)

def compute_empirical_cdf(values, max_q):
    """Compute empirical CDF for values up to max_q"""
    cdf = {}
    sorted_values = sorted(values)

    for q in range(1, max_q + 1):
        # Count how many values <= q
        count_le_q = sum(1 for v in sorted_values if v <= q)
        cdf[q] = count_le_q / len(sorted_values)

    return cdf

# Part B: Derive the optimal reassessment policy from the cluster-specific progress distributions
# This section computes each client's stopping point and evaluates the expected sessions saved
# for each potential reassessment session Q.
def find_optimal_reassessment_policy(cumulative_trajectories, clusters, n_clusters=4, T_max=12):
    """Find optimal reassessment policy for each cluster using newsvendor-style audit model"""

    # Compute stopping points for all clients
    stopping_points = compute_stopping_points(cumulative_trajectories)

    print(f"\n=== Optimal Reassessment Policy Analysis (T_max={T_max}) ===")

    policy_results = []

    for c in range(n_clusters):
        cluster_clients = cumulative_trajectories.index[clusters == c]
        cluster_stopping_points = stopping_points.loc[cluster_clients]

        print(f"\nCluster {c+1} (n={len(cluster_clients)} clients):")
        print(f"Stopping points distribution: {cluster_stopping_points.value_counts().sort_index()}")

        # Compute empirical CDF
        empirical_cdf = compute_empirical_cdf(cluster_stopping_points.values, T_max)

        # Compute expected savings for each Q
        savings_by_q = {}
        max_savings = -1
        optimal_q = None

        print("\nQ | F_c(Q) | E[Savings]")
        print("-" * 25)

        for q in range(1, T_max + 1):
            f_c_q = empirical_cdf[q]
            expected_savings = f_c_q * (T_max - q)
            savings_by_q[q] = expected_savings

            if expected_savings > max_savings:
                max_savings = expected_savings
                optimal_q = q

            print(f"{q:2d} | {f_c_q:.2f} | {expected_savings:.2f}")

        policy_results.append({
            'cluster': c+1,
            'n_clients': len(cluster_clients),
            'optimal_q': optimal_q,
            'max_expected_savings': max_savings,
            'stopping_points': cluster_stopping_points.values.tolist(),
            'savings_by_q': savings_by_q
        })

        print(f"\nOptimal reassessment session: Q* = {optimal_q}")

    return policy_results, stopping_points

# Part C: Select K using policy outcomes and clustering quality
# This section evaluates candidate K values, compares silhouette scores and unique policies,
# and chooses the most actionable clustering solution.
def evaluate_k_values(cumulative_trajectories, k_values=[3, 4, 5, 6]):
    """Evaluate different K values and their resulting policies"""
    print("\n=== Evaluating Different K Values ===")

    k_results = []

    for k in k_values:
        print(f"\n--- Evaluating K={k} ---")

        # Apply clustering
        clusters, kmeans, scaler, silhouette_avg = apply_clustering(cumulative_trajectories, k)

        # Get policy results
        policy_results, stopping_points = find_optimal_reassessment_policy(
            cumulative_trajectories, clusters, k, T_max=12
        )

        # Extract optimal Q values
        optimal_qs = [result['optimal_q'] for result in policy_results]
        unique_qs = len(set(optimal_qs))

        # Calculate average savings
        avg_savings = np.mean([result['max_expected_savings'] for result in policy_results])

        k_results.append({
            'k': k,
            'silhouette_score': silhouette_avg,
            'optimal_qs': optimal_qs,
            'unique_policies': unique_qs,
            'avg_savings': avg_savings,
            'policy_results': policy_results
        })

        print(f"K={k}: Silhouette={silhouette_avg:.3f}, Unique policies={unique_qs}, Avg savings={avg_savings:.2f}")
        print(f"Optimal Q values: {optimal_qs}")

    return k_results

def select_optimal_k(k_results):
    """Select the optimal K based on policy distinctness and actionability"""
    print("\n=== Selecting Optimal K ===")

    # Criteria for selection:
    # 1. High silhouette score (good clustering quality)
    # 2. Maximum number of unique policies (distinct reassessment timing)
    # 3. Reasonable cluster sizes (not too many tiny clusters)
    # 4. Actionable Q values (spread across different sessions)

    best_k = None
    best_score = -1

    for result in k_results:
        # Score based on: silhouette + (unique_policies * 0.5) + (avg_savings * 0.1)
        # Prioritize unique policies and clustering quality
        score = result['silhouette_score'] + (result['unique_policies'] * 0.3) + (result['avg_savings'] * 0.05)

        print(f"K={result['k']}: Score={score:.3f} (Silhouette={result['silhouette_score']:.3f}, "
              f"Unique={result['unique_policies']}, AvgSavings={result['avg_savings']:.2f})")

        if score > best_score:
            best_score = score
            best_k = result['k']

    return best_k, k_results

# Part D: Required plots for the final chosen K
# This section generates the three required visualizations for the selected clustering policy.
def create_stopping_point_histograms(cumulative_trajectories, clusters, stopping_points, n_clusters=5):
    """Create Plot 1: t* distributions - histograms of stopping points for each cluster"""
    fig, axes = plt.subplots(1, n_clusters, figsize=(15, 4), sharex=True, sharey=True)

    for c in range(n_clusters):
        cluster_clients = cumulative_trajectories.index[clusters == c]
        cluster_stopping_points = stopping_points.loc[cluster_clients]

        # Create histogram
        counts, bins, patches = axes[c].hist(cluster_stopping_points, bins=range(1, 14), alpha=0.7,
                                           edgecolor='black', align='left')

        axes[c].set_title(f'Cluster {c+1}\n(n={len(cluster_clients)})', fontsize=12, fontweight='bold')
        axes[c].set_xlabel('Stopping Session (t*)')
        if c == 0:
            axes[c].set_ylabel('Number of Clients')
        axes[c].grid(True, alpha=0.3)

        # Add text with mean stopping point
        mean_stop = cluster_stopping_points.mean()
        axes[c].axvline(mean_stop, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[c].text(0.05, 0.95, f'Mean: {mean_stop:.1f}',
                    transform=axes[c].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Plot 1: Stopping Point (t*) Distributions by Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/q2/plot1_stopping_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def create_savings_vs_q_plot(policy_results, n_clusters=5):
    """Create Plot 2: Expected sessions saved vs Q for each cluster"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors = sns.color_palette("husl", n_clusters)

    for c, result in enumerate(policy_results):
        cluster_num = result['cluster']
        savings_by_q = result['savings_by_q']
        optimal_q = result['optimal_q']
        n_clients = result['n_clients']

        # Plot the savings curve
        q_values = list(savings_by_q.keys())
        savings_values = list(savings_by_q.values())

        ax.plot(q_values, savings_values, 'o-', color=colors[c], linewidth=2, markersize=6,
               label=f'Cluster {cluster_num} (n={n_clients})')

        # Mark optimal Q with vertical dashed line
        ax.axvline(x=optimal_q, color=colors[c], linestyle='--', alpha=0.7, linewidth=2)
        ax.text(optimal_q + 0.1, max(savings_values) * 0.9, f'Q*={optimal_q}',
               color=colors[c], fontsize=10, fontweight='bold')

    ax.set_xlabel('Reassessment Session (Q)', fontsize=12)
    ax.set_ylabel('Expected Sessions Saved', fontsize=12)
    ax.set_title('Plot 2: Expected Sessions Saved vs Reassessment Timing (Q)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/q2/plot2_savings_vs_q.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def create_optimized_vs_baseline_comparison(policy_results, stopping_points, clusters, cumulative_trajectories, n_clusters=5):
    """Create Plot 3: Optimized Q* vs mean(t*) baseline comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    cluster_labels = []
    optimized_savings = []
    baseline_savings = []

    total_optimized_savings = 0
    total_baseline_savings = 0
    total_clients = 0

    for c in range(n_clusters):
        cluster_clients = cumulative_trajectories.index[clusters == c]
        cluster_stopping_points = stopping_points.loc[cluster_clients]
        result = policy_results[c]

        n_clients = len(cluster_clients)
        optimal_q = result['optimal_q']
        max_savings = result['max_expected_savings']

        # Compute naive baseline: round(mean(t*))
        mean_stopping_point = cluster_stopping_points.mean()
        baseline_q = round(mean_stopping_point)

        # Compute baseline savings: F_c(baseline_q) * (12 - baseline_q)
        empirical_cdf = compute_empirical_cdf(cluster_stopping_points.values, 12)
        baseline_savings_val = empirical_cdf[baseline_q] * (12 - baseline_q)

        cluster_labels.append(f'Cluster {c+1}\n(n={n_clients})')
        optimized_savings.append(max_savings)
        baseline_savings.append(baseline_savings_val)

        total_optimized_savings += max_savings * n_clients
        total_baseline_savings += baseline_savings_val * n_clients
        total_clients += n_clients

        print(f"Cluster {c+1}: Mean t* = {mean_stopping_point:.2f}, Baseline Q = {baseline_q}, "
              f"Optimized Q = {optimal_q}, Baseline savings = {baseline_savings_val:.2f}, "
              f"Optimized savings = {max_savings:.2f}")

    # Create grouped bar chart
    x = np.arange(len(cluster_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_savings, width, label='Baseline (mean t*)',
                   alpha=0.8, color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, optimized_savings, width, label='Optimized Q*',
                   alpha=0.8, color='lightgreen', edgecolor='black')

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Expected Sessions Saved per Client', fontsize=12)
    ax.set_title('Plot 3: Optimized Policy vs Mean(t*) Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('output/q2/plot3_optimized_vs_baseline.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate overall improvement
    overall_improvement = total_optimized_savings - total_baseline_savings
    improvement_per_client = overall_improvement / total_clients

    print("\nOverall Results:")
    print(f"Total optimized savings: {total_optimized_savings:.2f}")
    print(f"Total baseline savings: {total_baseline_savings:.2f}")
    print(f"Improvement per client: {improvement_per_client:.3f}")

    return fig, overall_improvement, improvement_per_client

# Part E: Generate the summary table for the final K and recommended policy
# This section summarizes each cluster's size, optimal reassessment session, and expected savings.
def generate_summary_table(policy_results, n_clusters=5, T_max=12):
    """Generate and save a summary table for the final K choice."""
    rows = []
    total_saved = 0.0
    total_clients = 0

    for result in policy_results:
        n_clients = result['n_clients']
        q_star = result['optimal_q']
        avg_saved = result['max_expected_savings']
        pct_saved = avg_saved / T_max * 100

        rows.append({
            'cluster': result['cluster'],
            'size': n_clients,
            'q_star': q_star,
            'e_saved_per_child': avg_saved,
            'pct_saved': pct_saved
        })

        total_saved += avg_saved * n_clients
        total_clients += n_clients

    overall_saved_per_child = total_saved / total_clients
    overall_pct_saved = overall_saved_per_child / T_max * 100

    summary_df = pd.DataFrame(rows)
    summary_df.loc[len(summary_df.index)] = {
        'cluster': 'Total',
        'size': total_clients,
        'q_star': '',
        'e_saved_per_child': overall_saved_per_child,
        'pct_saved': overall_pct_saved
    }

    summary_df.to_csv('output/q2/summary_table.csv', index=False)

    print("\n=== Final Summary Table ===")
    print(summary_df.to_string(index=False, formatters={
        'e_saved_per_child': '{:.2f}'.format,
        'pct_saved': '{:.1f}%'.format
    }))

    return summary_df


def main():
    """Main function to run the clustering analysis and select optimal K"""
    print("Loading and preparing data...")
    trajectories, cumulative_trajectories = load_and_prepare_data()

    print(f"Data shape: {cumulative_trajectories.shape}")
    print(f"Sessions: 1-{cumulative_trajectories.shape[1]}")
    print(f"Clients: {cumulative_trajectories.shape[0]}")

    # Evaluate different K values
    k_values_to_test = [3, 4, 5, 6]
    k_results = evaluate_k_values(cumulative_trajectories, k_values_to_test)

    # Select optimal K
    optimal_k, k_results = select_optimal_k(k_results)

    print(f"\nSelected optimal K = {optimal_k}")

    # Get the results for the optimal K
    optimal_result = next(r for r in k_results if r['k'] == optimal_k)
    clusters, _, _, _ = apply_clustering(cumulative_trajectories, optimal_k)
    policy_results = optimal_result['policy_results']

    # Create output directory
    import os
    os.makedirs('output/q2', exist_ok=True)

    # Analyze clusters for optimal K
    cluster_analysis = analyze_clusters(cumulative_trajectories, clusters, optimal_k)

    # Get stopping points for optimal K
    stopping_points = compute_stopping_points(cumulative_trajectories)

    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'client_id': cumulative_trajectories.index,
        'cluster': clusters + 1,  # 1-indexed for readability
        'stopping_point': stopping_points
    })
    cluster_df.to_csv('output/q2/cluster_assignments.csv', index=False)

    # Save cluster analysis
    cluster_analysis.to_csv('output/q2/cluster_analysis.csv', index=False)

    # Save policy results
    policy_df = pd.DataFrame([{
        'cluster': r['cluster'],
        'n_clients': r['n_clients'],
        'optimal_q': r['optimal_q'],
        'max_expected_savings': r['max_expected_savings']
    } for r in policy_results])
    policy_df.to_csv('output/q2/reassessment_policy.csv', index=False)

    # Save detailed policy analysis
    with open('output/q2/policy_details.txt', 'w') as f:
        f.write(f"=== Optimal Reassessment Policy Details (K={optimal_k}) ===\n\n")
        for result in policy_results:
            f.write(f"Cluster {result['cluster']} (n={result['n_clients']} clients):\n")
            f.write(f"Optimal reassessment session: Q* = {result['optimal_q']}\n")
            f.write(f"Stopping points: {sorted(result['stopping_points'])}\n")
            f.write("Expected savings by Q:\n")
            for q, savings in result['savings_by_q'].items():
                f.write(f"  Q={q}: {savings:.2f}\n")
            f.write("\n\n")

        # Add K selection justification
        f.write("=== K Selection Analysis ===\n\n")
        f.write(f"Selected K = {optimal_k} based on:\n")
        f.write("- Maximum number of distinct reassessment policies\n")
        f.write("- Good clustering quality (silhouette score)\n")
        f.write("- Actionable policy recommendations\n\n")
        f.write("Comparison of K values:\n")
        for result in k_results:
            f.write(f"K={result['k']}: {result['unique_policies']} unique policies, "
                   f"silhouette={result['silhouette_score']:.3f}, avg savings={result['avg_savings']:.2f}\n")

    # Create spaghetti plots for optimal K
    print("\nGenerating spaghetti plots...")
    create_spaghetti_plots(trajectories, cumulative_trajectories, clusters, optimal_k)

    # Create the three required plots
    print("\nGenerating required plots...")

    # Plot 1: Stopping point distributions
    print("Creating Plot 1: Stopping point distributions...")
    create_stopping_point_histograms(cumulative_trajectories, clusters, stopping_points, optimal_k)

    # Plot 2: Expected savings vs Q
    print("Creating Plot 2: Expected savings vs Q...")
    create_savings_vs_q_plot(policy_results, optimal_k)

    # Plot 3: Optimized vs baseline comparison
    print("Creating Plot 3: Optimized vs baseline comparison...")
    _, overall_improvement, improvement_per_client = create_optimized_vs_baseline_comparison(
        policy_results, stopping_points, clusters, cumulative_trajectories, optimal_k
    )

    # Part F: Implications for the case and final interpretation
    # This section prints the final outcome and highlights how the cluster-specific policy
    # supports targeted reassessment decisions.
    print("Creating summary table...")
    generate_summary_table(policy_results, optimal_k)

    print(f"\nOverall improvement: {overall_improvement:.2f} sessions saved across all clients")
    print(f"Improvement per client: {improvement_per_client:.3f} sessions")

    print(f"\nOptimal K selection and policy analysis complete (K={optimal_k})!")
    print("Output files saved to output/q2/:")
    print("- cluster_assignments.csv")
    print("- cluster_analysis.csv")
    print("- reassessment_policy.csv")
    print("- policy_details.txt")
    print("- clustering_spaghetti_plots.png")
    print("- plot1_stopping_distributions.png")
    print("- plot2_savings_vs_q.png")
    print("- plot3_optimized_vs_baseline.png")

if __name__ == "__main__":
    main()