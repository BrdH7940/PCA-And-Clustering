from Utils.Utils import *
from ClusteringAlgorithm.KMeans import MyKMeans
from PCA import MyPCA

def plot_explained_variance_ratio(pca, component_ax_kwargs=None, cumulative_ax_kwargs=None, customizations=None, legend_location=(0.85, 0.7)):
    """
    Vẽ biểu đồ tỷ lệ phương sai theo số thành phần chính cho mô hình PCA đã được huấn luyện.
    
    Tham số:
    ----------
    pca : PCA
        Mô hình PCA đã được huấn luyện
    fig_kwargs : dict
        Tham số cho hình vẽ
    component_ax_kwargs : dict
        Tham số cho biểu đồ thành phần riêng lẻ
    cumulative_ax_kwargs : dict
        Tham số cho biểu đồ tích lũy
    customizations : dict
        Tham số tùy chỉnh cho trục
    legend_location: Tuple(float, float)
        Vị trí chú thích trên biểu đồ
        
    Trả về:
    -------
    Tuple(Figure, Axes)
        Đối tượng hình vẽ và trục tọa độ
    """
    fig_kwargs = fig_kwargs or {"figsize": (9, 6)}

    if component_ax_kwargs is None:
        component_ax_kwargs = {
            "color": "purple",
            "linewidth": 2,
            "label": "Individual Component",
        }

    if cumulative_ax_kwargs is None:
        cumulative_ax_kwargs = {
            "color": "orange",
            "linewidth": 2,
            "label": "Cumulative",
        }
        
    if customizations is None:
        customizations = {
            "title": "Tỷ lệ phương sai giải thích theo số lượng thành phần chính",
            "xlabel": "Số lượng thành phần chính",
            "ylabel": "Tỷ lệ phương sai giải thích",
        }

    # Tạo biểu đồ
    fig, ax = plt.subplots(**fig_kwargs)

    # Plot explained variance ratio by component
    x_range = range(1, pca.n_components + 1)
    ax.plot(x_range, pca.EVR, **component_ax_kwargs)

    # Plot cumulative explained variance
    cumulative_sum = np.cumsum(pca.EVR)
    ax.plot(x_range, cumulative_sum, **cumulative_ax_kwargs)

    # Tùy chỉnh trục và tiêu đề
    ax.set(**customizations)
    fig.legend(bbox_to_anchor=legend_location)
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range)

    return fig, ax

def create_explained_variance_df(pca):
    """
    Tạo DataFrame để hiển thị thông tin của biểu đồ.
    
    Tham số:
    ----------
    pca : PCA
        Mô hình PCA đã được huấn luyện
        
    Trả về:
    -------
    pd.DataFrame
        DataFrame chứa phương sai (theo từng thành phần và tích lũy)
    """
    results_dict = {
        "Thành phần riêng lẻ - Individual Component": pca.EVR,
        "Thành phần tích lũy - Cumulative": np.cumsum(pca.EVR),
    }
    index = range(1, pca.n_components + 1)
    df = pd.DataFrame(results_dict, index=index)
    df.index.name = "# Thành phần"
    return df


def plot_accuracy_by_pc(X_scaled, true_labels, max_components=10):
    """
    Vẽ biểu đồ độ chính xác theo số thành phần chính và xác định số thành phần tối ưu.
    
    Tham số:
    ----------
    X_scaled : np.array
        Dữ liệu đã được chuẩn hóa
    true_labels : array
        Nhãn thực tế
    max_components : int, optional
        Số thành phần chính tối đa cần thử, mặc định là 10
        
    Trả về:
    -------
    int
        Số thành phần chính tối ưu
    """
    accuracies = []
    true_labels = np.array(true_labels)

    for n_components in range(1, max_components + 1):
        # Giảm chiều bằng MyPCA
        pca = MyPCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Phân cụm bằng MyKMeans
        kmeans = MyKMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_pca)
        predicted_labels = kmeans.predict(X_pca)

        # Tìm hoán vị nhãn tốt nhất
        max_accuracy = 0
        for perm in permutations([0, 1]):
            permuted = np.array([perm[label] for label in predicted_labels])
            acc = np.mean(permuted == true_labels)
            max_accuracy = max(max_accuracy, acc)

        accuracies.append(max_accuracy)
        print(f"    🔹Số thành phần: {n_components}, Accuracy: {max_accuracy:.2%}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_components + 1), accuracies, marker='o', linewidth=2)
    plt.title('Độ chính xác theo số thành phần chính')
    plt.xlabel('Số thành phần chính')
    plt.ylabel('Độ chính xác')
    plt.xticks(range(1, max_components + 1))
    plt.grid(True)
    plt.show()

    optimal_n = np.argmax(accuracies) + 1
    print(f"\n🔸 Số thành phần chính tối ưu: {optimal_n}")
    print(f"🔸 Độ chính xác tối đa ứng với {optimal_n} thành phần chính: {max(accuracies):.2%}")

    return optimal_n
