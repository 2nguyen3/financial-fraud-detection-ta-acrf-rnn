from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

FINANCIAL_ALL = BASE_DIR / "data" / "financial_all.csv"
OUTPUT_DIR = BASE_DIR / "graph_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not FINANCIAL_ALL.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {FINANCIAL_ALL}")

    df = pd.read_csv(FINANCIAL_ALL)

    required_cols = {"company_id", "year", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")

    df["company_id"] = df["company_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["label"] = df["label"].astype(int)

    # 1. Vertex company
    company_vertex = (
        df[["company_id"]]
        .drop_duplicates()
        .sort_values("company_id")
        .reset_index(drop=True)
    )

    # 2. Vertex company_year
    company_year_vertex = df[["company_id", "year", "label"]].copy()
    company_year_vertex["company_year_id"] = (
        company_year_vertex["company_id"].astype(str)
        + "_"
        + company_year_vertex["year"].astype(str)
    )

    company_year_vertex = company_year_vertex[
        ["company_year_id", "company_id", "year", "label"]
    ].sort_values(["year", "company_id"])

    # 3. Edge has_record: company -> company_year
    has_record_edge = company_year_vertex.copy()
    has_record_edge["src"] = has_record_edge["company_id"]
    has_record_edge["dst"] = has_record_edge["company_year_id"]

    has_record_edge = has_record_edge[
        ["src", "dst", "year", "label"]
    ].sort_values(["year", "src"])

    # Ghi file
    company_vertex.to_csv(
        OUTPUT_DIR / "company_vertex.csv",
        index=False,
        encoding="utf-8"
    )

    company_year_vertex.to_csv(
        OUTPUT_DIR / "company_year_vertex.csv",
        index=False,
        encoding="utf-8"
    )

    has_record_edge.to_csv(
        OUTPUT_DIR / "has_record_edge.csv",
        index=False,
        encoding="utf-8"
    )

    # Kiểm tra
    print("===== KẾT QUẢ SINH DỮ LIỆU GRAPH =====")
    print(f"Số company vertex: {len(company_vertex)}")
    print(f"Số company_year vertex: {len(company_year_vertex)}")
    print(f"Số has_record edge: {len(has_record_edge)}")

    print("\n===== FILE ĐÃ TẠO =====")
    print(OUTPUT_DIR / "company_vertex.csv")
    print(OUTPUT_DIR / "company_year_vertex.csv")
    print(OUTPUT_DIR / "has_record_edge.csv")

    # Validate nhanh
    assert len(company_vertex) == 491, "Số company vertex phải là 491."
    assert len(company_year_vertex) == 4910, "Số company_year vertex phải là 4910."
    assert len(has_record_edge) == 4910, "Số has_record edge phải là 4910."

    duplicated_company_year = company_year_vertex["company_year_id"].duplicated().sum()
    assert duplicated_company_year == 0, "company_year_id bị trùng."

    print("\nKẾT QUẢ: Dữ liệu graph cơ bản hợp lệ.")


if __name__ == "__main__":
    main()
