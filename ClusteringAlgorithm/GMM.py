from Utils.Utils import *
class MyGMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, random_state=None):
        # Khởi tạo tham số
        self.n_components = n_components        # Số cụm (số thành phần Gaussian)
        self.max_iter = max_iter                
        self.tol = tol                          # Ngưỡng hội tụ (tolerance)
        self.random_state = random_state        # Seed cho reproducibility
        self.means_ = None                      # Trung bình của các Gaussian
        self.covariances_ = None                # Ma trận hiệp phương sai của các Gaussian
        self.weights_ = None                    # Trọng số (xác suất) của các Gaussian
        self.labels_ = None                     # Nhãn dự đoán sau khi phân cụm

    def _initialize(self, X):
        # Khởi tạo ban đầu các tham số
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # Chọn ngẫu nhiên các tâm ban đầu từ dữ liệu
        self.means_ = X[rng.choice(n_samples, self.n_components, replace=False)]
        
        # Khởi tạo ma trận hiệp phương sai giống nhau cho mỗi cụm
        self.covariances_ = np.array([np.cov(X, rowvar=False)] * self.n_components)
        
        # Khởi tạo trọng số đều nhau
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _e_step(self, X):
        # Bước E: Tính trách nhiệm (responsibilities)
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))
        
        # Tính xác suất cho từng điểm dữ liệu thuộc về từng Gaussian
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
        
        # Chuẩn hóa để tổng mỗi hàng bằng 1 (soft assignment)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp = resp / resp_sum
        return resp

    def _m_step(self, X, resp):
        # Bước M: Cập nhật tham số mô hình
        n_samples, n_features = X.shape
        Nk = resp.sum(axis=0)  # Tổng trọng số (responsibility) cho từng cụm
        
        # Cập nhật trọng số
        self.weights_ = Nk / n_samples
        
        # Cập nhật trung bình mới cho mỗi cụm
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]
        
        # Cập nhật ma trận hiệp phương sai
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (resp[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
            
            # Thêm một lượng nhỏ để tránh ma trận hiệp phương sai bị suy biến
            self.covariances_[k] += 1e-6 * np.eye(n_features)

    def _gaussian(self, X, mean, cov):
        # Tính giá trị phân phối Gaussian đa biến
        n = X.shape[1]
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        
        # Hệ số chuẩn hóa
        norm = 1.0 / np.sqrt((2 * np.pi) ** n * det_cov)
        
        # Tính phần mũ (exponential) trong công thức Gaussian
        exp = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
        return norm * exp

    def fit(self, X):
        # Huấn luyện mô hình GMM
        X = np.array(X)
        self._initialize(X)
        log_likelihood = None
        
        for _ in range(self.max_iter):
            # Bước E
            resp = self._e_step(X)
            
            # Bước M
            self._m_step(X, resp)
            
            # Tính log-likelihood để kiểm tra hội tụ
            ll_new = np.sum(np.log(np.sum([
                self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
                for k in range(self.n_components)
            ], axis=0)))
            
            # Kiểm tra điều kiện hội tụ
            if log_likelihood is not None and abs(ll_new - log_likelihood) < self.tol:
                break
            log_likelihood = ll_new
        
        # Gán nhãn (label) cho từng điểm dữ liệu
        self.labels_ = np.argmax(resp, axis=1)
        return self

    def fit_predict(self, X):
        # Huấn luyện và trả về nhãn cụm
        self.fit(X)
        return self.labels_

    def predict(self, X):
        # Dự đoán nhãn cụm cho dữ liệu mới
        X = np.array(X)
        resp = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
        return np.argmax(resp, axis=1)
