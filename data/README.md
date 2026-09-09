# data/

Thư mục này **không chứa dữ liệu thật** trong repository — chỉ giữ cấu trúc
thư mục (bằng các file `.gitkeep`) để người dùng biết cần đặt/sinh dữ liệu ở
đâu khi chạy lại pipeline. Xem lý do tại mục "Vì sao không commit dữ liệu"
bên dưới.

```
data/
├── raw/            # dữ liệu tài chính công ty-năm gốc (đầu vào)
├── processed/      # output của Spark pipeline (đầu ra bước 1)
└── graph_data/      # cạnh (edges) dùng để import vào NebulaGraph
```

## 1. `raw/` — Dữ liệu đầu vào

Đặt dữ liệu tài chính công ty-năm gốc vào đây trước khi chạy
`notebooks/01_spark_pipeline/Spark_Pipeline.ipynb`.

- Nếu bạn tái hiện đúng dữ liệu của bài báo nền ACRF-RNN (IJCAI 2025),
  vui lòng lấy dữ liệu trực tiếp từ repository chính thức của tác giả:
  https://github.com/XNetLab/ACRF-RNN, và tuân thủ điều khoản sử dụng/chia
  sẻ lại dữ liệu của họ. Dự án này **không đính kèm sẵn** dữ liệu đó.
- Nếu dùng nguồn khác (báo cáo tài chính công khai, CSMAR, WRDS,
  SEC EDGAR...), đặt file thô vào `raw/` theo định dạng mà
  `Spark_Pipeline.ipynb` đang đọc (xem cell đầu notebook để biết tên cột/
  định dạng file kỳ vọng).

## 2. `processed/` — Output của Spark pipeline

Sinh ra tự động khi chạy xong `notebooks/01_spark_pipeline/Spark_Pipeline.ipynb`.
Cấu trúc thực tế:

```
processed/
├── train.parquet
├── val.parquet
├── test_2018.parquet
├── test_2019.parquet
├── test_2020.parquet
├── scaler_model/        # scaler đã fit trên tập train, dùng lại khi infer
├── processed_flat        # bảng phẳng company-year, phục vụ baseline RF/XGBoost
└── statistics             # thống kê mô tả dùng để kiểm tra chất lượng dữ liệu
```

`train.parquet` / `val.parquet` dùng cho giai đoạn huấn luyện + chọn
epoch/ngưỡng (tuned trên validation 2017); `test_2018/2019/2020.parquet`
dùng để đánh giá cuối cùng, không được dùng trong quá trình tuning.

**Cách tái tạo:** chạy toàn bộ `notebooks/01_spark_pipeline/Spark_Pipeline.ipynb`
với `raw/` đã có dữ liệu đầu vào. Nếu chỉ cần chạy lại mô hình, tải
[`processed.zip` từ release v1.0.0](https://github.com/2nguyen3/financial-fraud-detection-ta-acrf-rnn/releases/download/v1.0.0/processed.zip)
và giải nén vào thư mục `data/processed/`.

## 3. `graph_data/` — Cạnh quan hệ liên công ty

Chứa các file CSV mô tả cạnh (edges) giữa các công ty theo từng năm, dùng để
import vào NebulaGraph bằng
`database/scripts/import_relationship_edges_to_nebula.py`. Có 2 loại đồ thị:

- **Cosine Graph** — cạnh dựa trên độ tương đồng cosine giữa vector đặc trưng
  tài chính của các công ty trong cùng năm.
- **Attention Graph** — cạnh dựa trên trọng số attention mà mô hình
  TA-ACRF-RNN học được giữa các công ty.

Các file cạnh đầy đủ không được commit vào repository. Để tạo `graph_data/`,
cần chạy notebook huấn luyện `03_ta_acrf_rnn` hoặc các script sinh Cosine
Graph sau khi đã có dữ liệu trong `processed/`.

## Vì sao không commit dữ liệu thật vào Git

1. **Dung lượng** — `processed/` (nhiều file `.parquet` qua các năm) và
   `graph_data/` đầy đủ vượt quá mức hợp lý cho một Git repository thông
   thường.
2. **Bản quyền** — nếu dữ liệu đầu vào bắt nguồn từ dataset riêng của bài
   báo nền ACRF-RNN, việc phát tán lại công khai (kể cả sau khi xử lý) có
   thể vi phạm điều khoản chia sẻ dữ liệu của nhóm tác giả gốc. Xem thêm
   `LICENSE` và `CITATION.cff` ở thư mục gốc.

Nếu cần bản dữ liệu đã xử lý sẵn để tiện tái lập kết quả (và bạn xác nhận
được phép chia sẻ lại), có thể đóng gói `processed/` thành `processed.zip`
và đính kèm ở GitHub Releases của repo, thay vì commit trực tiếp.
