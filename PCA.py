import numpy as np
import pandas as pd
import seaborn as sns
from Utils import *

class MyPCA:
    def __init__(self, n_components: int) -> None:
        super().__init__()
        self.n_components = n_components
        self.principal_components = None

    def fit(self, X):
        if self.n_components > X.shape[1]:
            print("Cảnh báo: Số components > Số features")

        X = X - X.mean(axis = 0)
        S = X.T @ X / X.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(S)

        # Retrieve n_components vectors with highest eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]
        self.principal_components = sorted_eigenvectors[:, :self.n_components]

        # Calculate EVR, CEVR
        total_variance = np.sum(sorted_eigenvalues)
        EVR = sorted_eigenvalues / total_variance
        CEVR = np.cumsum(EVR)

        # Print
        print("--- Kết quả Fit ---")
        print(f"Số thành phần chính được chọn: {self.n_components}")
        print(f"Tỷ lệ phương sai giải thích (EVR) cho {self.n_components} thành phần đầu:")

        EVR_selected = EVR[:self.n_components]
        for i, ratio in enumerate(EVR_selected):
             print(f"  PC{i+1}: {ratio:.4f}")

        print(f"\nTỷ lệ phương sai giải thích tích lũy (CEVR) cho {self.n_components} thành phần đầu:")

        CEVR_selected = CEVR[:self.n_components]
        for i, cum_ratio in enumerate(CEVR_selected):
            print(f"  PC1 đến PC{i+1}: {cum_ratio:.4f}")
        print(f"\nTổng phương sai được giải thích bởi {self.n_components} thành phần: {CEVR_selected[-1]:.4f}")
        print("--------------------")

    def transform(self, X):
        if self.principal_components is None:
            print('Chạy fit() trước đi')
            return
        
        print(type(self.principal_components))
        X = X - X.mean(axis = 0)
        
        return X @ self.principal_components