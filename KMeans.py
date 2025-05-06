from Utils import *

class MyKMeans:
    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4, random_state=None, track_history=False):
        self.n_clusters = n_clusters            # Số nhóm cần phân
        self.n_init = n_init                    # Số lần chọn lại để ra chọn kết quả tốt nhất
        self.max_iter = max_iter                # Số vòng lặp tối đa
        self.tol = tol                          # Ngưỡng dừng nếu thay đổi quá nhỏ
        self.random_state = random_state        # Giúp chạy lại ra kết quả giống nhau
        self.track_history = track_history      # Có lưu lại lịch sử hay không
        self.centroids = None                   # Các điểm trung tâm cuối cùng
        self.labels_ = None                     # Nhóm mà mỗi điểm dữ liệu thuộc về
        self.inertia_ = None                    # Tổng khoảng cách giữa dữ liệu và trung tâm nhóm
        
        # Theo dõi tiến trình
        self.history_centroids = []             # Lưu các vị trí trung tâm theo từng vòng
        self.history_inertia = []               # Lưu sai số theo từng vòng

    def init_centroids(self, X):
        # Chọn ngẫu nhiên một vài điểm ban đầu làm trung tâm nhóm
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def compute_inertia(self, X, centroids, labels):
        # Tính tổng khoảng cách từ điểm đến trung tâm nhóm của nó
        return np.sum((np.linalg.norm(X - centroids[labels], axis=1)) ** 2)

    def fit(self, X):
        X = np.array(X)
        best_inertia = float('inf')     # Sai số nhỏ nhất từng thấy
        best_centroids = None
        best_labels = None
        best_history_centroids = []
        best_history_inertia = []

        # Giữ cho việc chọn ngẫu nhiên luôn giống nhau nếu đặt trước
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Thử lại nhiều lần để chọn được cách chia tốt nhất
        for _ in range(self.n_init):
            centroids = self.init_centroids(X)  # Chọn ngẫu nhiên trung tâm ban đầu
            history_centroids = []
            history_inertia = []

            for _ in range(self.max_iter):
                # Bước 1: Gán từng điểm vào nhóm gần nó nhất
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)

                # Bước 2: Tính lại trung tâm mỗi nhóm
                new_centroids = np.array([
                    X[labels == i].mean(axis=0) if np.any(labels == i) else X[np.random.choice(X.shape[0])]
                    for i in range(self.n_clusters)
                ])

                # Lưu lại quá trình nếu cần
                if self.track_history:
                    history_centroids.append(centroids.copy())
                    inertia = self.compute_inertia(X, centroids, labels)
                    history_inertia.append(inertia)

                # Kiểm tra xem các trung tâm có thay đổi nhiều không
                if np.all(np.linalg.norm(centroids - new_centroids, axis=1) < self.tol):
                    break

                centroids = new_centroids  # Cập nhật

            # Tính sai số sau khi đã hội tụ
            inertia = self.compute_inertia(X, centroids, labels)

            # Nếu lần này tốt hơn trước, thì lưu lại
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_history_centroids = history_centroids
                best_history_inertia = history_inertia

        # Ghi nhận kết quả tốt nhất
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        if self.track_history:
            self.history_centroids = best_history_centroids
            self.history_inertia = best_history_inertia

        return self

    def predict(self, X):
        X = np.array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_