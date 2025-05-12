import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from PCA import MyPCA
from ClusteringAlgorithm.GMM import MyGMM
from ClusteringAlgorithm.KMeans import MyKMeans
import matplotlib.pyplot as plt
import os
from datetime import datetime

def transform_features(data_path, output_path=None, skew_threshold=1.0,
                       poly_degree=2, poly_threshold=0.1,
                       visualize=True, interaction_terms=True, use_poly=True):
    """
    Automatically transform features based on their characteristics.

    Args:
        data_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the transformed data. If None, will use 'transformed_[original_name].csv'
        skew_threshold (float): Threshold for considering a feature highly skewed
        poly_degree (int): Maximum degree for polynomial features
        poly_threshold (float): Correlation threshold for creating polynomial features
        visualize (bool): Whether to create visualization of original vs. transformed features
        interaction_terms (bool): Whether to create interaction terms between features
        use_poly (bool): Whether to create polynomial features

    Returns:
        pd.DataFrame: Transformed dataframe
    """
    # Set default output path
    if output_path is None:
        output_path = f"transformed_{data_path.split('/')[-1]}"

    # Load the data
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    print(f"Original data shape: {df.shape}")

    # Find the target column if it exists
    target_col = None
    if 'group' in df.columns:
        target_col = 'group'
        labels = df[target_col].copy()
        df_features = df.drop(columns=[target_col])
    else:
        df_features = df.copy()

    # Get numeric columns
    numeric_cols = df_features.select_dtypes(include=['number']).columns.tolist()
    print(f"Number of numeric features: {len(numeric_cols)}")

    # Calculate skewness for each feature
    skewness = df_features[numeric_cols].apply(lambda x: stats.skew(x.dropna()))

    # Identify highly skewed features
    highly_skewed = skewness[abs(skewness) > skew_threshold].index.tolist()
    print(f"Number of highly skewed features: {len(highly_skewed)}")

    # Create a new dataframe for transformed features
    transformed_df = pd.DataFrame(index=df_features.index)
    transformation_log = {}

    # Process each feature
    for col in numeric_cols:
        # Get the feature data
        x = df_features[col].values

        # Check if the feature is highly skewed
        if col in highly_skewed:
            # Check if the feature has non-positive values (can't apply log directly)
            if np.min(x) <= 0:
                # Shift data to make it positive
                shift = abs(np.min(x)) + 1.0
                x_transformed = np.log1p(x + shift)
                transformation_log[col] = f"log1p(x + {shift})"
            else:
                # Apply log transformation
                x_transformed = np.log1p(x)
                transformation_log[col] = "log1p(x)"

            # Add the transformed feature
            transformed_df[f"{col}"] = x_transformed
        else:
            # Keep the original feature
            transformed_df[f"{col}"] = x
            transformation_log[col] = "original"

    # Only create polynomial features if use_poly is True
    if use_poly:
        # Create polynomial features for a subset of important features
        # First, identify important features using correlation with target or variance
        if target_col is not None:
            # Use correlation with target to find important features
            important_features = []
            for col in transformed_df.columns:
                if abs(np.corrcoef(transformed_df[col], pd.get_dummies(labels).iloc[:, 0])[0, 1]) > poly_threshold:
                    important_features.append(col)
        else:
            # Use variance as a measure of importance
            variances = transformed_df.var().sort_values(ascending=False)
            important_features = variances.index[:int(len(transformed_df.columns) * poly_threshold)].tolist()

        print(f"Number of features selected for polynomial transformation: {len(important_features)}")

        # Generate polynomial features
        for col in important_features:
            x = transformed_df[col].values
            for degree in range(2, poly_degree + 1):
                transformed_df[f"{col}_pow{degree}"] = x ** degree
                transformation_log[f"{col}_pow{degree}"] = f"{col}^{degree}"

        # Create interaction terms between important features if requested
        if interaction_terms and len(important_features) >= 2:
            print("Generating interaction terms...")
            for i in range(len(important_features)):
                for j in range(i + 1, len(important_features)):
                    col1, col2 = important_features[i], important_features[j]
                    new_col = f"{col1}_mul_{col2}"
                    transformed_df[new_col] = transformed_df[col1] * transformed_df[col2]
                    # transformation_log[new_col] = f"{col1} * {col2}"

    # Standardize all features
    print("Standardizing features...")
    for col in transformed_df.columns:
        mean = transformed_df[col].mean()
        std = transformed_df[col].std()
        if std > 0:  # Avoid division by zero
            transformed_df[col] = (transformed_df[col] - mean) / std

    # Add back the target column if it exists
    if target_col is not None:
        transformed_df[target_col] = labels

    # Save the transformed data
    transformed_df.to_csv(output_path, index=False)
    print(f"Transformed data saved to {output_path}")
    print(f"Final data shape: {transformed_df.shape}")

    # Print transformation summary
    print("\nTransformation Summary:")
    # for col, transform in transformation_log.items():
    #     print(f"{col}: {transform}")

    # Create visualizations if requested
    if visualize:
        sample_cols = min(5, len(highly_skewed))
        if sample_cols > 0:
            plt.figure(figsize=(15, 3 * sample_cols))
            for i, col in enumerate(highly_skewed[:sample_cols]):
                # Original distribution
                plt.subplot(sample_cols, 2, 2 * i + 1)
                plt.hist(df_features[col].dropna(), bins=30)
                plt.title(f"Original: {col}")

                # Transformed distribution
                plt.subplot(sample_cols, 2, 2 * i + 2)
                plt.hist(transformed_df[col].dropna(), bins=30)
                plt.title(f"Transformed: {col}")

            plt.tight_layout()
            plt.savefig("feature_transformations.png")
            plt.close()
            print("Visualizations saved to 'feature_transformations.png'")

    return transformed_df


def run_experiment(dataset_path, output_dir='./', skew_threshold=1.0,
                  component_range=None, k=2, visualize=True, algorithm='KMeans', use_poly=True):
    """
    Run a complete experiment with feature transformation, PCA, and clustering.

    Args:
        dataset_path (str): Path to the input dataset CSV
        output_dir (str): Directory to save outputs
        skew_threshold (float): Threshold for considering a feature highly skewed
        component_range (list): List of n_components values to test
        k (int): Number of clusters
        visualize (bool): Whether to create visualizations
    """
    import os
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Define file paths
    transformed_path = os.path.join(experiment_dir, "transformed_data.csv")
    results_path = os.path.join(experiment_dir, "experiment_results.csv")
    log_path = os.path.join(experiment_dir, "experiment_log.txt")

    # Set up logging
    with open(log_path, 'w') as log_file:
        def log(message):
            print(message)
            log_file.write(message + '\n')
            log_file.flush()

        log(f"=== Experiment started at {timestamp} ===")
        log(f"Dataset: {dataset_path}")

        # Load and transform features
        log("\n--- Feature Transformation ---")
        transformed_data = transform_features(
            dataset_path,
            output_path=transformed_path,
            skew_threshold=skew_threshold,
            visualize=visualize,
            use_poly=use_poly
        )

        # Extract features and target
        if 'group' in transformed_data.columns:
            y = transformed_data['group'].copy()
            X = transformed_data.drop(columns=['group'])
        else:
            y = None
            X = transformed_data

        log(f"Transformed data shape: {X.shape}")

        # Determine component range if not provided
        if component_range is None:
            max_components = min(X.shape[0], X.shape[1])
            component_range = [10, 20, 30, 50, 70, 100, 150]
            component_range = [c for c in component_range if c <= max_components]

        # Prepare results storage
        results = []

        # Run PCA with different n_components
        log("\n--- PCA and Clustering Experiments ---")
        for n_components in component_range:
            log(f"\nTesting with n_components={n_components}")

            # Perform PCA
            pca = MyPCA(n_components=n_components)
            X_pca = pca.fit_transform(X)

            # Record basic PCA stats
            evr = sum(pca.EVR)
            log(f"Explained variance ratio: {evr:.4f}")

            if algorithm == 'KMeans':
                # Implement KMeans clustering
                kmeans = MyKMeans(n_clusters=k, max_iter=100, random_state=42, track_history=True)
                kmeans.fit(X_pca)

                # Lấy kết quả
                labels = kmeans.labels_
                centroids = kmeans.centroids
            elif algorithm == "GMM":
                gmm = MyGMM(n_components=k, max_iter=100, random_state=42)
                gmm.fit(X_pca)

                # Lấy kết quả
                labels = gmm.labels_

            # Evaluate clustering if we have true labels
            if y is not None:
                # Convert labels to numeric if they're categorical
                if not pd.api.types.is_numeric_dtype(y):
                    label_map = {label: i for i, label in enumerate(y.unique())}
                    y_numeric = y.map(label_map)
                else:
                    y_numeric = y

                # Calculate accuracy (after finding best label mapping)
                from scipy.optimize import linear_sum_assignment

                # Create confusion matrix
                conf_matrix = np.zeros((k, k))
                for i in range(len(labels)):
                    conf_matrix[labels[i], y_numeric.iloc[i]] += 1

                # Find optimal assignment
                row_ind, col_ind = linear_sum_assignment(-conf_matrix)

                # Remap cluster labels
                remapped_labels = np.zeros_like(labels)
                for i in range(k):
                    remapped_labels[labels == row_ind[i]] = col_ind[i]

                # Calculate accuracy
                accuracy = np.sum(remapped_labels == y_numeric) / len(y_numeric)
                log(f"Clustering accuracy: {accuracy:.4f}")

                # Calculate F1 score
                from sklearn.metrics import f1_score
                f1 = f1_score(y_numeric, remapped_labels, average='weighted')
                log(f"F1 score: {f1:.4f}")

                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_pca, labels)
                    log(f"Silhouette score: {silhouette:.4f}")
                else:
                    silhouette = 0
                    log("Silhouette score: Khong the tinh vi chi có 1 cum duoc tao ra.")

                # Store results
                results.append({
                    'n_components': n_components,
                    'explained_variance_ratio': evr,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'silhouette_score': silhouette
                })

            # Visualize 2D projection for the first experiment
            if visualize and n_components >= 2:
                plt.figure(figsize=(16, 6))
                X_pca = np.asarray(X_pca)
                # Plot clustering results
                plt.subplot(1, 2, 1)
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
                plt.title(f"Clustering Results (n_components={n_components})")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.colorbar()

                # Plot true labels if available
                if y is not None:
                    plt.subplot(1, 2, 2)
                    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='viridis', alpha=0.7)
                    plt.title("True Labels")
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.colorbar()

                plt.tight_layout()
                plt.savefig(os.path.join(experiment_dir, f"pca_n{n_components}.png"))
                plt.close()

        # Save results to CSV
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_path, index=False)
            log(f"\nResults saved to {results_path}")

            # Find the best configuration
            best_accuracy_idx = results_df['accuracy'].idxmax()
            best_config = results_df.iloc[best_accuracy_idx]
            log(f"\nBest configuration:")
            log(f"n_components: {best_config['n_components']}")
            log(f"Accuracy: {best_config['accuracy']:.4f}")
            log(f"F1 score: {best_config['f1_score']:.4f}")
            log(f"Silhouette score: {best_config['silhouette_score']:.4f}")

            # Plot metrics vs n_components
            if visualize and len(component_range) > 1:
                plt.figure(figsize=(15, 10))

                # Plot explained variance vs components
                plt.subplot(2, 2, 1)
                plt.plot(results_df['n_components'], results_df['explained_variance_ratio'], 'o-')
                plt.title('Explained Variance vs. Components')
                plt.xlabel('Number of Components')
                plt.ylabel('Cumulative Explained Variance')

                # Plot accuracy vs components
                plt.subplot(2, 2, 2)
                plt.plot(results_df['n_components'], results_df['accuracy'], 'o-')
                plt.title('Accuracy vs. Components')
                plt.xlabel('Number of Components')
                plt.ylabel('Accuracy')

                # Plot F1 score vs components
                plt.subplot(2, 2, 3)
                plt.plot(results_df['n_components'], results_df['f1_score'], 'o-')
                plt.title('F1 Score vs. Components')
                plt.xlabel('Number of Components')
                plt.ylabel('F1 Score')

                # Plot silhouette score vs components
                plt.subplot(2, 2, 4)
                plt.plot(results_df['n_components'], results_df['silhouette_score'], 'o-')
                plt.title('Silhouette Score vs. Components')
                plt.xlabel('Number of Components')
                plt.ylabel('Silhouette Score')

                plt.tight_layout()
                plt.savefig(os.path.join(experiment_dir, "metrics_vs_components.png"))
                plt.close()

        log(f"\n=== Experiment completed ===")
        return experiment_dir

def run_multi_threshold_experiment(dataset_path, output_dir='./',
                                   skew_thresholds=[0.5, 0.75, 1.0, 1.25, 1.5],
                                   component_range=[10, 20, 30, 50, 100],
                                   k=2, algorithms=["KMeans", "GMM"],
                                   poly_features=[False, True]):
    """
    Run experiments with multiple skew thresholds, algorithms, and polynomial features option.

    Args:
        dataset_path (str): Path to dataset
        output_dir (str): Output directory
        skew_thresholds (list): List of skew thresholds to test
        component_range (list): List of n_components values
        k (int): Number of clusters
        algorithms (list): List of clustering algorithms to use
        poly_features (list): List of boolean values whether to use polynomial features or not
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(output_dir, f"multi_threshold_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Store results for all thresholds and algorithms
    all_results = []

    # Run experiments for each threshold, algorithm, and poly_feature option
    for threshold in skew_thresholds:
        for algorithm in algorithms:
            for use_poly in poly_features:
                exp_dir = run_experiment(
                    dataset_path,
                    output_dir=experiment_dir,
                    skew_threshold=threshold,
                    component_range=component_range,
                    k=k,
                    visualize=True,
                    algorithm=algorithm,
                    use_poly=use_poly
                )

                # Read results
                results_df = pd.read_csv(os.path.join(exp_dir, "experiment_results.csv"))
                results_df['skew_threshold'] = threshold
                results_df['algorithm'] = algorithm
                results_df['poly_features'] = use_poly
                all_results.append(results_df)

    # Combine all results
    combined_results = pd.concat(all_results)

    # Save combined results
    combined_results.to_csv(os.path.join(experiment_dir, 'combined_results.csv'), index=False)

    # Create visualizations for each algorithm with polynomial feature comparison
    for algorithm in algorithms:
        alg_results = combined_results[combined_results['algorithm'] == algorithm]

        # Visualizations comparing with and without polynomial features
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'Results for {algorithm} - Polynomial Features Comparison', fontsize=16)

        # 1. Line plot: Accuracy vs Components for different thresholds and poly options
        ax1 = plt.subplot(221)
        for threshold in skew_thresholds:
            for use_poly in poly_features:
                data = alg_results[(alg_results['skew_threshold'] == threshold) &
                                   (alg_results['poly_features'] == use_poly)]
                poly_label = "With Poly" if use_poly else "Without Poly"
                linestyle = '-' if use_poly else '--'
                ax1.plot(data['n_components'], data['accuracy'],
                         marker='o', linestyle=linestyle,
                         label=f'Threshold={threshold}, {poly_label}')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Accuracy vs Components - {algorithm}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Box plot: Accuracy distribution by polynomial features
        ax2 = plt.subplot(222)
        sns.boxplot(data=alg_results, x='poly_features', y='accuracy', ax=ax2)
        ax2.set_xticklabels(['Without Poly', 'With Poly'])
        ax2.set_title(f'Accuracy Distribution by Polynomial Features - {algorithm}')

        # 3. Heatmap: Components vs Threshold for non-poly
        ax3 = plt.subplot(223)
        non_poly_data = alg_results[alg_results['poly_features'] == False]
        pivot_non_poly = non_poly_data.pivot(
            index='skew_threshold',
            columns='n_components',
            values='accuracy'
        )
        sns.heatmap(pivot_non_poly, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3)
        ax3.set_title(f'Accuracy Heatmap WITHOUT Polynomial Features - {algorithm}')

        # 4. Heatmap: Components vs Threshold for poly
        ax4 = plt.subplot(224)
        poly_data = alg_results[alg_results['poly_features'] == True]
        pivot_poly = poly_data.pivot(
            index='skew_threshold',
            columns='n_components',
            values='accuracy'
        )
        sns.heatmap(pivot_poly, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title(f'Accuracy Heatmap WITH Polynomial Features - {algorithm}')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig(os.path.join(experiment_dir, f'{algorithm}_poly_comparison.png'))
        plt.close()

    # Create comparison visualizations between algorithms and polynomial features
    plt.figure(figsize=(20, 10))
    plt.suptitle('Algorithm and Polynomial Features Comparison', fontsize=16)

    # 1. Bar plot: Mean accuracy by Algorithm and Poly Features
    ax1 = plt.subplot(121)
    alg_poly_comparison = combined_results.groupby(['algorithm', 'poly_features'])['accuracy'].mean().reset_index()
    alg_poly_comparison['poly_features'] = alg_poly_comparison['poly_features'].map(
        {False: 'Without Poly', True: 'With Poly'})
    sns.barplot(data=alg_poly_comparison, x='algorithm', y='accuracy', hue='poly_features', ax=ax1)
    ax1.set_title('Mean Accuracy by Algorithm and Polynomial Features')
    ax1.set_ylabel('Mean Accuracy')
    ax1.set_xlabel('Algorithm')

    # 2. Box plot: Accuracy distribution for each algorithm and poly feature
    ax2 = plt.subplot(122)
    combined_results_plot = combined_results.copy()
    combined_results_plot['algorithm_poly'] = combined_results_plot.apply(
        lambda row: f"{row['algorithm']} {'Poly' if row['poly_features'] else 'No-Poly'}", axis=1
    )
    sns.boxplot(data=combined_results_plot, x='algorithm_poly', y='accuracy', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_title('Accuracy Distribution by Algorithm and Polynomial Features')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Algorithm & Polynomial Feature')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(os.path.join(experiment_dir, 'algorithm_poly_comparison.png'))
    plt.close()

    # Find best configurations
    best_overall_idx = combined_results['accuracy'].idxmax()
    best_overall_config = combined_results.iloc[best_overall_idx]

    # Find best for each algorithm with and without polynomial features
    best_configs = {}
    for algorithm in algorithms:
        for use_poly in poly_features:
            filtered_df = combined_results[(combined_results['algorithm'] == algorithm) &
                                           (combined_results['poly_features'] == use_poly)]
            if not filtered_df.empty:
                best_idx = filtered_df['accuracy'].idxmax()
                best_config = filtered_df.iloc[best_idx]
                key = f"{algorithm}_{'poly' if use_poly else 'no_poly'}"
                best_configs[key] = best_config

    print("\nBest Overall Configuration:")
    print(f"Algorithm: {best_overall_config['algorithm']}")
    print(f"Polynomial Features: {'Yes' if best_overall_config['poly_features'] else 'No'}")
    print(f"Skew Threshold: {best_overall_config['skew_threshold']}")
    print(f"Number of Components: {best_overall_config['n_components']}")
    print(f"Accuracy: {best_overall_config['accuracy']:.4f}")
    print(f"F1 Score: {best_overall_config['f1_score']:.4f}")
    print(f"Silhouette Score: {best_overall_config['silhouette_score']:.4f}")

    print("\nBest Configurations by Algorithm and Polynomial Features:")
    for config_name, config in best_configs.items():
        print(f"\n{config_name}:")
        print(f"Skew Threshold: {config['skew_threshold']}")
        print(f"Number of Components: {config['n_components']}")
        print(f"Accuracy: {config['accuracy']:.4f}")
        print(f"F1 Score: {config['f1_score']:.4f}")
        print(f"Silhouette Score: {config['silhouette_score']:.4f}")

    return experiment_dir, combined_results