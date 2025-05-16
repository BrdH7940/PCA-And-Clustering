# PCA-And-Clustering

Lab 2 - Mathematics for AI - HCMUS

# Dự án: Giảm chiều dữ liệu bằng PCA và Phân cụm K-Means

Dự án này tập trung vào việc áp dụng kỹ thuật Phân tích Thành phần Chính (PCA) để giảm số chiều của bộ dữ liệu y sinh ABIDE II, sau đó sử dụng thuật toán K-Means Clustering (và các thuật toán phân cụm khác như GMM) để gom nhóm các đối tượng (bệnh nhân) dựa trên các đặc trưng hình thái não đã được giảm chiều. Mục tiêu là cluster các data points trong dataset.

## Mục tiêu chính

- **Triển khai PCA thủ công:** Xây dựng một lớp PCA từ đầu bằng Python (không sử dụng thư viện `sklearn` cho phần cốt lõi của PCA) để hiểu sâu hơn về cơ chế hoạt động của thuật toán.
- **Tiền xử lý dữ liệu:** Áp dụng các bước tiền xử lý cần thiết cho bộ dữ liệu ABIDE II, bao gồm xử lý giá trị thiếu, mã hóa biến hạng mục, và chuẩn hóa dữ liệu.
- **Giảm chiều dữ liệu:** Sử dụng PCA đã triển khai để giảm số lượng lớn các đặc trưng của bộ dữ liệu ABIDE II xuống một số lượng thành phần chính nhỏ hơn mà vẫn giữ được phần lớn phương sai của dữ liệu.
- **Phân cụm (Clustering):**
  - Áp dụng thuật toán K-Means Clustering trên dữ liệu đã giảm chiều để phân các đối tượng vào các cụm.
  - Thử nghiệm với các thuật toán phân cụm khác như Gaussian Mixture Models (GMM) để so sánh hiệu suất.
- **Đánh giá và Phân tích:**
  - Đánh giá chất lượng của các cụm được hình thành bằng cách so sánh với nhãn thực tế (cột 'group').
  - Khảo sát ảnh hưởng của số lượng thành phần chính đến hiệu suất phân cụm.
  - Trực quan hóa kết quả và phân tích các insights.

## Cấu trúc Thư mục Dự án

Dưới đây là mô tả về cấu trúc thư mục và các file chính trong dự án:
```
.
├── ClusteringAlgorithm/ # Chứa các module triển khai thuật toán phân cụm
│ ├── GMM.py # Triển khai thuật toán Gaussian Mixture Model (GMM)
│ └── KMeans.py # Triển khai thuật toán K-Means Clustering
│
├── Dataset/ # Chứa các file dữ liệu được sử dụng và tạo ra
│ ├── ABIDE2_transformed.csv # Dữ liệu ABIDE II sau biến đổi
│ ├── ABIDE2.csv # File dữ liệu ABIDE II gốc
│ ├── encoded_data.csv # Dữ liệu sau khi embedded
│
├── Notebooks/ # Chứa các Notebooks cho việc thử nghiệm và phát triển
│ ├── ImprovedPCA.ipynb # Notebook thử nghiệm/phát triển các cải tiến cho PCA
│ ├── KAN.ipynb # Notebook liên quan đến Kolmogorov-Arnold Networks
│ ├── PolyPCA.ipynb # Notebook thử nghiệm PCA kết hợp với Polynomial Features
│ ├── Spectral Clustering.ipynb # Notebook thử nghiệm thuật toán Spectral Clustering
│ └── Test.ipynb # Notebook dùng cho các thử nghiệm chung chung
│
├── Utils/ # Chứa các module tiện ích hỗ trợ
│ ├── **pycache**/
│ ├── **init**.py # Khởi tạo package Utils
│ ├── Plots.py # Chứa các hàm để vẽ biểu đồ
│ └── Utils.py # Chứa các hàm tiện ích chung
│
├── Main.ipynb # Jupyter Notebook chính, thực thi luồng công việc của dự án
└── PCA.py # Module chứa triển khai lớp MyPCA thủ công
```

## Cách chạy

1.  Mở và chạy `Main.ipynb` trong môi trường Jupyter Notebook/JupyterLab để thực thi toàn bộ quy trình.
2.  Các notebook khác trong thư mục `Notebooks/` có thể được chạy độc lập để xem các thử nghiệm khác.
