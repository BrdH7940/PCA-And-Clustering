import numpy as np

class MyPCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.principal_components = None
        self.EVR = None
        self.mean_ = None

    def fit(self, X):
        if self.n_components > X.shape[1]:
            print("⚠️ Cảnh báo: Số components > Số features")

        # Lưu mean để dùng trong transform
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Tính ma trận hiệp phương sai
        S = X_centered.T @ X_centered / X_centered.shape[0]

        # Tính trị riêng và vector riêng
        eigenvalues, eigenvectors = np.linalg.eigh(S)

        # Sắp xếp theo trị riêng giảm dần
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]

        # Chọn n_components
        self.principal_components = sorted_eigenvectors[:, :self.n_components]

        # Tính EVR, CEVR
        total_variance = np.sum(sorted_eigenvalues)
        EVR = sorted_eigenvalues / total_variance
        self.EVR = EVR[:self.n_components]

        CEVR = np.cumsum(self.EVR)

        # In kết quả
        print("--- Kết quả Fit ---")
        print(f"Số thành phần chính được chọn: {self.n_components}")
        print("Tỷ lệ phương sai giải thích (EVR):")
        for i, ratio in enumerate(self.EVR):
            print(f"  PC{i+1}: {ratio:.4f}")
        print("\nTỷ lệ phương sai giải thích tích lũy (CEVR):")
        for i, cum_ratio in enumerate(CEVR):
            print(f"  PC1 đến PC{i+1}: {cum_ratio:.4f}")
        print(f"\nTổng phương sai được giải thích: {CEVR[-1]:.4f}")
        print("--------------------")

        return self

    def transform(self, X):
        if self.principal_components is None:
            raise ValueError("Bạn cần gọi fit() trước khi gọi transform()")
        X_centered = X - self.mean_
        return X_centered @ self.principal_components

    def fit_transform(self, X):
        return self.fit(X).transform(X)
