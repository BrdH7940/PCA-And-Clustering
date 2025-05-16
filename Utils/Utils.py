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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment
import warnings
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json

warnings.filterwarnings("ignore")

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

# 1-layer KAN Encoder
class KANEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.4)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.a = nn.Parameter(torch.ones(hidden_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.c = nn.Parameter(torch.ones(hidden_dim))
        self.d = nn.Parameter(torch.zeros(hidden_dim))

    def kan_activation(self, x):
        return self.a * torch.tanh(self.b * x + self.c) + self.d

    def forward(self, x):
        x = F.linear(x, self.weights.T, self.bias)
        x = x + torch.rand_like(x) * 0.2
        return self.kan_activation(x)

# 1-layer Tanh Decoder
class TanhDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = x + torch.rand_like(x) * 0.2
        return x

# Autoencoder
class AdversarialAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, noise_factor=0.1):
        super().__init__()
        self.encoder = KANEncoder(input_dim, hidden_dim)
        self.decoder = TanhDecoder(hidden_dim, input_dim)
        self.noise_factor = noise_factor

    def forward(self, x, add_noise=True):
        if add_noise:
            x_noisy = x + self.noise_factor * torch.randn_like(x)
        else:
            x_noisy = x
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Linear Classifier - FIXED
class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)

def train_autoencoder(model, loader, optimizer_enc, optimizer_dec, device):
    model.train()
    total_loss = 0
    for data, _ in loader:
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        data = data.to(device)
        _, decoded = model(data)
        loss = F.mse_loss(decoded, data)
        loss.backward()
        optimizer_enc.step()
        optimizer_dec.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# FIXED: Use BCEWithLogitsLoss instead of BCELoss
def train_classifier(encoder, classifier, loader, optimizer, device):
    encoder.eval()
    classifier.train()
    total_loss = 0
    # Changed from BCELoss to BCEWithLogitsLoss
    bce_with_logits = nn.BCEWithLogitsLoss()

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            embeddings = encoder(data)
        logits = classifier(embeddings)

        # BCEWithLogitsLoss applies sigmoid internally, so we don't need to
        loss = bce_with_logits(logits, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def pseudo_label_dataset(encoder, dataset, device, n_clusters=2):
    loader = DataLoader(dataset, batch_size=256)
    encoder.eval()
    embeddings = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            emb = encoder(data)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    pca = MyPCA(n_components=2)
    embeddings = pca.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

# Additional debugging function
def check_tensor_validity(tensor, name):
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN values")
        return False
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf values")
        return False
    return True

# Training loop with additional error checking
def safe_train_classifier(encoder, classifier, loader, optimizer, device):
    encoder.eval()
    classifier.train()
    total_loss = 0
    bce_with_logits = nn.BCEWithLogitsLoss()

    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)

        # Check input validity
        if not check_tensor_validity(data, f"batch_{batch_idx}_data"):
            continue
        if not check_tensor_validity(labels, f"batch_{batch_idx}_labels"):
            continue

        optimizer.zero_grad()
        with torch.no_grad():
            embeddings = encoder(data)
            if not check_tensor_validity(embeddings, f"batch_{batch_idx}_embeddings"):
                continue

        logits = classifier(embeddings)
        if not check_tensor_validity(logits, f"batch_{batch_idx}_logits"):
            continue

        loss = bce_with_logits(logits, labels.float())
        if not check_tensor_validity(loss, f"batch_{batch_idx}_loss"):
            continue

        loss.backward()

        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0

def best_map(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return {j: i for i, j in zip(*ind)}

def train_kan_autoencoder(X, y,
                         # Model parameters
                         hidden_dim=256,
                         noise_factor=0.1,

                         # Training parameters
                         n_warmup=70,
                         n_cycles=50,
                         unsup_epochs=15,
                         sup_epochs=25,
                         batch_size=128,

                         # Optimization parameters
                         lr_encoder=0.0024,
                         lr_decoder=0.001,
                         lr_classifier=0.001,

                         # Semi-supervised parameters
                         n_clusters=2,
                         sup_ratio=0.25,  # fraction of data used for supervised training

                         # Other parameters
                         device=None,
                         save_dir='./saved_models',
                         experiment_name=None,
                         save_best=True,
                         plot_loss=True,
                         verbose=True,
                         random_seed=42):
    """
    Train KAN Autoencoder with semi-supervised learning

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    y : array-like, shape (n_samples,)
        Target labels (for evaluation, not used in training)
    hidden_dim : int
        Dimension of the hidden representation
    noise_factor : float
        Noise factor for denoising autoencoder
    n_warmup : int
        Number of warmup epochs (unsupervised only)
    n_cycles : int
        Number of alternating cycles
    unsup_epochs : int
        Number of unsupervised epochs per cycle
    sup_epochs : int
        Number of supervised epochs per cycle
    batch_size : int
        Batch size for training
    lr_encoder, lr_decoder, lr_classifier : float
        Learning rates
    n_clusters : int
        Number of clusters for pseudo-labeling
    sup_ratio : float
        Ratio of data used for supervised training
    device : str or None
        Device to use ('cuda' or 'cpu')
    save_dir : str
        Directory to save models
    experiment_name : str or None
        Name for the experiment
    save_best : bool
        Whether to save the best model
    plot_loss : bool
        Whether to plot training losses
    verbose : bool
        Whether to print training progress
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict : Training results including model, history, and best metrics
    """

    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Setup experiment directory
    if experiment_name is None:
        experiment_name = datetime.now().strftime("kan_autoencoder_%Y%m%d_%H%M%S")

    exp_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create tensors and dataset
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = X.shape[1]
    model = AdversarialAutoencoder(input_dim, hidden_dim, noise_factor).to(device)

    # Initialize optimizers
    optimizer_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr_encoder)
    optimizer_dec = torch.optim.Adam(model.decoder.parameters(), lr=lr_decoder)

    # Training history
    history = {
        'losses': [],
        'cycle_accuracies': [],
        'cycle_f1_scores': [],
        'best_accuracy': 0.0,
        'best_f1': 0.0,
        'best_cycle': 0
    }

    # Best model state
    best_model_state = None

    if verbose:
        print(f"Starting training with {n_warmup} warmup epochs and {n_cycles} cycles")
        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}")
        print(f"Device: {device}")

    # Phase 1: Warmup (Unsupervised training)
    if verbose:
        print("\n=== Warmup Phase ===")

    for epoch in range(n_warmup):
        loss = train_autoencoder(model, train_loader, optimizer_enc, optimizer_dec, device)
        history['losses'].append(loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f'Warmup Epoch {epoch+1}/{n_warmup}, Loss: {loss:.4f}')

    # Phase 2: Alternating unsupervised and semi-supervised training
    if verbose:
        print("\n=== Alternating Phase ===")

    for cycle in range(n_cycles):
        # Unsupervised phase
        for epoch in range(unsup_epochs):
            loss = train_autoencoder(model, train_loader, optimizer_enc, optimizer_dec, device)
            history['losses'].append(loss)

        # Pseudo-labeling
        classifier = LinearClassifier(hidden_dim).to(device)
        cluster_labels = pseudo_label_dataset(model.encoder, dataset, device, n_clusters)

        # Create supervised dataset
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        sup_indices = indices[:int(len(indices) * sup_ratio)]
        sup_labels = cluster_labels[sup_indices]
        sup_X = X_tensor[sup_indices]
        sup_y = torch.FloatTensor(sup_labels)
        sup_dataset = TensorDataset(sup_X, sup_y)
        sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)

        # Supervised phase
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=lr_classifier)
        for epoch in range(sup_epochs):
            loss = safe_train_classifier(model.encoder, classifier, sup_loader, optimizer_cls, device)

        # Evaluate current cycle
        classifier.eval()
        with torch.no_grad():
            all_embeddings = model.encoder(X_tensor.to(device))
            logits = classifier(all_embeddings)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy().astype(int)
            true_labels = y_tensor.cpu().numpy().astype(int)
            acc = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, average='weighted')

        history['cycle_accuracies'].append(acc)
        history['cycle_f1_scores'].append(f1)

        # Update best model
        if acc > history['best_accuracy']:
            history['best_accuracy'] = acc
            history['best_f1'] = f1
            history['best_cycle'] = cycle

            # Save best model state
            if save_best:
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'scaler': scaler,
                    'accuracy': acc,
                    'f1_score': f1,
                    'cycle': cycle,
                    'hyperparameters': {
                        'hidden_dim': hidden_dim,
                        'noise_factor': noise_factor,
                        'n_warmup': n_warmup,
                        'n_cycles': n_cycles,
                        'lr_encoder': lr_encoder,
                        'lr_decoder': lr_decoder,
                        'lr_classifier': lr_classifier,
                        'batch_size': batch_size,
                        'sup_ratio': sup_ratio,
                        'n_clusters': n_clusters
                    }
                }

        if verbose and (cycle + 1) % 5 == 0:
            print(f'Cycle {cycle+1}/{n_cycles} - Accuracy: {acc:.4f}, F1: {f1:.4f}')

    # Final evaluation with clustering
    if verbose:
        print("\n=== Final Evaluation ===")

    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(X_tensor.to(device)).cpu().numpy()

    # Apply PCA for better clustering (optional)
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings)

    # Final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    cluster_labels = kmeans.fit_predict(embeddings_pca)

    # Align clusters to ground truth
    mapping = best_map(y, cluster_labels)
    aligned_preds = np.array([mapping[c] for c in cluster_labels])
    final_acc = accuracy_score(y, aligned_preds)
    final_f1 = f1_score(y, aligned_preds, average='weighted')

    history['final_accuracy'] = final_acc
    history['final_f1'] = final_f1

    if verbose:
        print(f'Final Clustering Accuracy: {final_acc:.4f}')
        print(f'Final Clustering F1 Score: {final_f1:.4f}')
        print(f'Best Supervised Accuracy: {history["best_accuracy"]:.4f} (Cycle {history["best_cycle"]+1})')

    # Save results
    if save_best and best_model_state is not None:
        # Save model
        model_path = os.path.join(exp_dir, 'best_model.pth')
        torch.save(best_model_state, model_path)

        # Save history
        history_path = os.path.join(exp_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_json = {}
            for key, value in history.items():
                if isinstance(value, np.ndarray):
                    history_json[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    history_json[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    history_json[key] = int(value)
                else:
                    history_json[key] = value
            json.dump(history_json, f, indent=2)

        if verbose:
            print(f"Best model saved to: {model_path}")
            print(f"Training history saved to: {history_path}")

    # Plot training curves
    if plot_loss:
        plt.figure(figsize=(15, 5))

        # Plot reconstruction loss
        plt.subplot(1, 3, 1)
        plt.plot(history['losses'])
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # Plot accuracy over cycles
        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(history['cycle_accuracies']) + 1), history['cycle_accuracies'], 'b-o')
        plt.title('Accuracy per Cycle')
        plt.xlabel('Cycle')
        plt.ylabel('Accuracy')
        plt.grid(True)

        # Plot F1 score over cycles
        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(history['cycle_f1_scores']) + 1), history['cycle_f1_scores'], 'r-o')
        plt.title('F1 Score per Cycle')
        plt.xlabel('Cycle')
        plt.ylabel('F1 Score')
        plt.grid(True)

        plt.tight_layout()

        if save_best:
            plot_path = os.path.join(exp_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Training curves saved to: {plot_path}")

        plt.show()

    # Prepare return dictionary
    results = {
        'model': model,
        'best_classifier': classifier if 'classifier' in locals() else None,
        'scaler': scaler,
        'history': history,
        'experiment_dir': exp_dir,
        'best_model_path': os.path.join(exp_dir, 'best_model.pth') if save_best else None,
        'final_embeddings': embeddings,
        'cluster_labels': cluster_labels,
        'aligned_predictions': aligned_preds
    }

    return results


def load_best_model(model_path, input_dim, device=None):
    """
    Load the best trained model

    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    input_dim : int
        Input dimension for the model
    device : str or None
        Device to load the model on

    Returns:
    --------
    dict : Dictionary containing model, classifier, scaler, and metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load saved state
    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruct model
    hidden_dim = checkpoint['hyperparameters']['hidden_dim']
    noise_factor = checkpoint['hyperparameters']['noise_factor']

    model = AdversarialAutoencoder(input_dim, hidden_dim, noise_factor)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Reconstruct classifier
    classifier = LinearClassifier(hidden_dim)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier.to(device)
    classifier.eval()

    return {
        'model': model,
        'classifier': classifier,
        'scaler': checkpoint['scaler'],
        'accuracy': checkpoint['accuracy'],
        'f1_score': checkpoint['f1_score'],
        'cycle': checkpoint['cycle'],
        'hyperparameters': checkpoint['hyperparameters']
    }


# Example usage function
def optimize_hyperparameters(X, y, param_grid, n_trials=10, device=None):
    """
    Simple hyperparameter optimization using grid search with random sampling

    Parameters:
    -----------
    X, y : array-like
        Data and labels
    param_grid : dict
        Dictionary of hyperparameters to try
    n_trials : int
        Number of random trials
    device : str or None
        Device to use

    Returns:
    --------
    dict : Best parameters and results
    """
    from itertools import product
    import random

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = list(product(*values))

    # Sample random combinations
    selected_combinations = random.sample(combinations, min(n_trials, len(combinations)))

    best_acc = 0
    best_params = None
    best_results = None

    for i, params in enumerate(selected_combinations):
        param_dict = dict(zip(keys, params))

        print(f"\nTrial {i+1}/{len(selected_combinations)}")
        print(f"Parameters: {param_dict}")

        # Run training with current parameters
        results = train_kan_autoencoder(
            X, y,
            **param_dict,
            device=device,
            verbose=False,
            save_best=False,
            plot_loss=False
        )

        acc = results['history']['best_accuracy']
        print(f"Best accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = param_dict
            best_results = results

    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_acc:.4f}")

    return {
        'best_params': best_params,
        'best_accuracy': best_acc,
        'best_results': best_results
    }