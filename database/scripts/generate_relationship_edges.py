from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "financial_all.csv"
OUTPUT_DIR = BASE_DIR / "graph_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mỗi công ty trong từng năm được nối với TOP_K công ty có vector tài chính tương đồng nhất.
# TOP_K = 20 giúp graph đủ biểu diễn quan hệ liên công ty nhưng không quá dày.
TOP_K = 20


def prepare_feature_matrix(year_df, feature_cols):
    """
    Chuẩn hóa ma trận đặc trưng trong phạm vi từng mốc thời gian.

    Mục tiêu:
    - Ép toàn bộ đặc trưng về numeric.
    - Loại bỏ NaN, inf, -inf.
    - Chuẩn hóa z-score theo từng mốc thời gian.
    - Clip giá trị cực đoan để tránh overflow.
    - Chuẩn hóa vector để tính cosine similarity ổn định.
    """
    X_df = year_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    medians = X_df.median(axis=0, skipna=True)
    medians = medians.fillna(0.0)
    X_df = X_df.fillna(medians)

    X = X_df.to_numpy(dtype=np.float64)

    # Làm sạch lần 1 trước khi chuẩn hóa
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std == 0] = 1.0

    X = (X - mean) / std

    # Làm sạch lần 2 sau z-score
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Giới hạn giá trị cực đoan để phép nhân ma trận ổn định.
    X = np.clip(X, -10.0, 10.0)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.nan_to_num(norms, nan=1.0, posinf=1.0, neginf=1.0)
    norms[norms == 0] = 1.0

    X_norm = X / norms

    # Làm sạch lần 3 trước khi tính cosine similarity
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return X_norm


def cosine_top_k_edges(df, feature_cols, top_k=5):
    similar_edges = []
    years = sorted(df["year"].unique())

    for year in years:
        year_df = df[df["year"] == year].copy()
        year_df = year_df.sort_values("company_id").reset_index(drop=True)

        company_ids = year_df["company_id"].astype(str).tolist()
        labels = year_df["label"].astype(int).tolist()

        X_norm = prepare_feature_matrix(year_df, feature_cols)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sim_matrix = X_norm @ X_norm.T

        sim_matrix = np.nan_to_num(
            sim_matrix,
            nan=-np.inf,
            posinf=-np.inf,
            neginf=-np.inf
        )
        
        np.fill_diagonal(sim_matrix, -np.inf)

        for i, src_company in enumerate(company_ids):
            top_indices = np.argsort(sim_matrix[i])[-top_k:][::-1]

            for j in top_indices:
                dst_company = company_ids[j]
                similarity = float(sim_matrix[i, j])

                src_id = f"{src_company}_{year}"
                dst_id = f"{dst_company}_{year}"

                similar_edges.append({
                    "src": src_id,
                    "dst": dst_id,
                    "year": int(year),
                    "similarity": round(similarity, 6),
                    "src_company_id": src_company,
                    "dst_company_id": dst_company,
                    "src_label": int(labels[i]),
                    "dst_label": int(labels[j]),
                })

    return pd.DataFrame(similar_edges)


def build_stable_edges(similar_df):
    """
    Sinh quan hệ stable_similar.

    Lưu ý:
    Dữ liệu không có năm 2016, do đó stable_similar được hiểu là quan hệ
    được duy trì qua hai mốc thời gian liên tiếp trong tập dữ liệu,
    không nhất thiết là hai năm lịch liên tiếp tuyệt đối.
    """
    stable_edges = []

    available_years = sorted(similar_df["year"].unique())
    year_pairs = list(zip(available_years[:-1], available_years[1:]))

    temp_df = similar_df.copy()
    temp_df["pair_key"] = (
        temp_df["src_company_id"].astype(str)
        + "->"
        + temp_df["dst_company_id"].astype(str)
    )

    year_pair_map = {
        year: temp_df[temp_df["year"] == year].set_index("pair_key")
        for year in available_years
    }

    for year_from, year_to in year_pairs:
        df_from = year_pair_map[year_from]
        df_to = year_pair_map[year_to]

        common_pairs = sorted(set(df_from.index) & set(df_to.index))

        for pair_key in common_pairs:
            row_from = df_from.loc[pair_key]
            row_to = df_to.loc[pair_key]

            src_company = str(row_from["src_company_id"])
            dst_company = str(row_from["dst_company_id"])

            sim_from = float(row_from["similarity"])
            sim_to = float(row_to["similarity"])
            avg_similarity = (sim_from + sim_to) / 2

            stable_edges.append({
                "src": src_company,
                "dst": dst_company,
                "year_from": int(year_from),
                "year_to": int(year_to),
                "similarity_from": round(sim_from, 6),
                "similarity_to": round(sim_to, 6),
                "avg_similarity": round(avg_similarity, 6),
                "src_label_to": int(row_to["src_label"]),
                "dst_label_to": int(row_to["dst_label"]),
                "both_fraud_to": int(row_to["src_label"] == 1 and row_to["dst_label"] == 1),
                "edge_rank": int(year_from * 10000 + year_to),
            })

    return pd.DataFrame(stable_edges)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["company_id"] = df["company_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["label"] = df["label"].astype(int)

    exclude_cols = {"company_id", "year", "label", "sj_label"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print("===== THÔNG TIN DỮ LIỆU =====")
    print(f"Số dòng financial_all: {len(df)}")
    print(f"Số công ty: {df['company_id'].nunique()}")
    print(f"Số mốc thời gian: {df['year'].nunique()}")
    print(f"Các mốc thời gian: {sorted(df['year'].unique().tolist())}")
    print(f"Số đặc trưng dùng tính tương đồng: {len(feature_cols)}")
    print(f"TOP_K tương đồng mỗi công ty/mốc thời gian: {TOP_K}")

    similar_df = cosine_top_k_edges(df, feature_cols, TOP_K)
    stable_df = build_stable_edges(similar_df)

    similar_path = OUTPUT_DIR / "similar_in_year_edge.csv"
    stable_path = OUTPUT_DIR / "stable_similar_edge.csv"
    summary_path = OUTPUT_DIR / "relationship_summary.csv"

    similar_df.to_csv(similar_path, index=False, encoding="utf-8")
    stable_df.to_csv(stable_path, index=False, encoding="utf-8")

    summary_df = pd.DataFrame([
        {"metric": "top_k", "value": TOP_K},
        {"metric": "num_companies", "value": df["company_id"].nunique()},
        {"metric": "num_time_points", "value": df["year"].nunique()},
        {"metric": "time_points", "value": ", ".join(map(str, sorted(df["year"].unique().tolist())))},
        {"metric": "num_financial_rows", "value": len(df)},
        {"metric": "num_feature_columns", "value": len(feature_cols)},
        {"metric": "num_similar_in_year_edges", "value": len(similar_df)},
        {"metric": "num_stable_similar_edges", "value": len(stable_df)},
    ])
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n===== KẾT QUẢ SINH EDGE LIÊN CÔNG TY =====")
    print(f"Số similar_in_year edge: {len(similar_df)}")
    print(f"Số stable_similar edge: {len(stable_df)}")

    print("\n===== FILE ĐÃ TẠO =====")
    print(similar_path)
    print(stable_path)
    print(summary_path)

    print("\n===== MẪU similar_in_year =====")
    print(similar_df.head(10))

    print("\n===== MẪU stable_similar =====")
    print(stable_df.head(10))

    assert len(similar_df) > 0, "Không sinh được similar_in_year edge."
    assert len(stable_df) > 0, "Không sinh được stable_similar edge."

    print("\nKẾT QUẢ: Sinh quan hệ liên công ty theo thời gian thành công.")


if __name__ == "__main__":
    main()