import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

# Đường dẫn đúng theo kết quả bạn vừa gửi:
# processed/processed/processed_flat.json
PROCESSED_JSON = BASE_DIR / "processed" / "processed" / "processed_flat.json"

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_flat(json_path: Path) -> pd.DataFrame:
    if not json_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {json_path}\n"
            "Hãy kiểm tra lại đường dẫn processed_flat.json."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    columns = obj["columns"]
    records = []

    for year_block in obj["data"]:
        year = int(year_block["year"])
        symbols = year_block["symbols"]
        rows = year_block["rows"]

        if len(symbols) != len(rows):
            raise ValueError(f"Năm {year}: số lượng symbols không khớp số lượng rows.")

        for symbol, row in zip(symbols, rows):
            if len(row) != len(columns):
                raise ValueError(
                    f"Năm {year}, công ty {symbol}: số cột trong row không khớp columns."
                )

            record = {
                "company_id": str(symbol),
                "year": year,
            }

            for col_name, value in zip(columns, row):
                record[col_name] = value

            records.append(record)

    df = pd.DataFrame(records)

    df["company_id"] = df["company_id"].astype(str)
    df["year"] = df["year"].astype(int)

    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    return df


def create_company_table(df: pd.DataFrame) -> pd.DataFrame:
    company = (
        df[["company_id"]]
        .drop_duplicates()
        .sort_values("company_id")
        .reset_index(drop=True)
    )

    company["symbol"] = company["company_id"]

    return company[["company_id", "symbol"]]


def create_year_dim_table() -> pd.DataFrame:
    rows = []

    for year in [2010, 2011, 2012, 2013, 2014, 2015]:
        rows.append({
            "year": year,
            "dataset_role": "train",
            "node_name": "db_node_1",
            "note": "Dữ liệu lịch sử phục vụ huấn luyện"
        })

    rows.append({
        "year": 2017,
        "dataset_role": "validation",
        "node_name": "db_node_1",
        "note": "Dữ liệu kiểm định theo thiết lập bài báo"
    })

    for year in [2018, 2019, 2020]:
        rows.append({
            "year": year,
            "dataset_role": "test",
            "node_name": "db_node_2",
            "note": "Dữ liệu kiểm thử theo từng năm"
        })

    return pd.DataFrame(rows)


def split_financial_data(df: pd.DataFrame):
    node1_years = {2010, 2011, 2012, 2013, 2014, 2015, 2017}
    node2_years = {2018, 2019, 2020}

    financial_node1 = df[df["year"].isin(node1_years)].copy()
    financial_node2 = df[df["year"].isin(node2_years)].copy()

    financial_node1 = financial_node1.sort_values(["year", "company_id"])
    financial_node2 = financial_node2.sort_values(["year", "company_id"])

    return financial_node1, financial_node2


def validate_outputs(df, company, year_dim, node1, node2):
    print("===== KIỂM TRA DỮ LIỆU =====")
    print(f"Tổng số dòng financial_statement: {len(df)}")
    print(f"Số công ty: {company['company_id'].nunique()}")
    print(f"Số năm: {df['year'].nunique()}")
    print(f"Các năm: {sorted(df['year'].unique().tolist())}")

    print("\n===== THÔNG TIN CỘT =====")
    print(f"Tổng số cột financial_statement: {len(df.columns)}")
    print(f"10 cột đầu: {df.columns[:10].tolist()}")
    print(f"10 cột cuối: {df.columns[-10:].tolist()}")

    print("\n===== PHÂN MẢNH =====")
    print(f"db_node_1: {len(node1)} dòng")
    print(f"db_node_2: {len(node2)} dòng")
    print(f"Tổng 2 node: {len(node1) + len(node2)} dòng")

    print("\n===== YEAR_DIM =====")
    print(year_dim)

    print("\n===== PHÂN BỐ NHÃN THEO NODE =====")
    if "label" in df.columns:
        print("db_node_1:")
        print(node1["label"].value_counts().sort_index())
        print("db_node_2:")
        print(node2["label"].value_counts().sort_index())
    else:
        print("Không tìm thấy cột label trong dữ liệu.")

    duplicated = df.duplicated(subset=["company_id", "year"]).sum()
    print(f"\nSố dòng trùng theo (company_id, year): {duplicated}")

    assert len(df) == 4910, "Tổng số dòng phải là 4910."
    assert company["company_id"].nunique() == 491, "Số công ty phải là 491."
    assert len(node1) == 3437, "db_node_1 phải có 3437 dòng."
    assert len(node2) == 1473, "db_node_2 phải có 1473 dòng."
    assert len(node1) + len(node2) == len(df), "Tổng hai node không khớp dữ liệu gốc."
    assert duplicated == 0, "Có dòng trùng theo (company_id, year)."

    print("\nKẾT QUẢ: Dữ liệu hợp lệ, có thể dùng để nạp vào PostgreSQL.")


def main():
    df = load_processed_flat(PROCESSED_JSON)

    company = create_company_table(df)
    year_dim = create_year_dim_table()
    financial_node1, financial_node2 = split_financial_data(df)

    company.to_csv(OUTPUT_DIR / "company.csv", index=False, encoding="utf-8")
    year_dim.to_csv(OUTPUT_DIR / "year_dim.csv", index=False, encoding="utf-8")
    financial_node1.to_csv(OUTPUT_DIR / "financial_node1.csv", index=False, encoding="utf-8")
    financial_node2.to_csv(OUTPUT_DIR / "financial_node2.csv", index=False, encoding="utf-8")

    # File này dùng để kiểm tra tổng thể, không nhất thiết nạp vào database.
    df.to_csv(OUTPUT_DIR / "financial_all.csv", index=False, encoding="utf-8")

    validate_outputs(df, company, year_dim, financial_node1, financial_node2)

    print("\n===== FILE ĐÃ TẠO =====")
    for file in [
        "company.csv",
        "year_dim.csv",
        "financial_node1.csv",
        "financial_node2.csv",
        "financial_all.csv",
    ]:
        print(OUTPUT_DIR / file)


if __name__ == "__main__":
    main()
