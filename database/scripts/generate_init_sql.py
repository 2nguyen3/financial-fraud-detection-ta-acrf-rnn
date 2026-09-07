from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "data" / "financial_node1.csv"
OUTPUT_SQL = BASE_DIR / "sql" / "init.sql"


def safe_column_name(col: str) -> str:
    """
    Giữ tên cột gốc nếu hợp lệ.
    Nếu tên cột có ký tự đặc biệt hoặc bắt đầu bằng số,
    ta sẽ bọc bằng dấu nháy kép trong PostgreSQL.
    """
    return '"' + col.replace('"', '""') + '"'


def infer_sql_type(col: str) -> str:
    if col == "company_id":
        return "VARCHAR(50)"
    if col == "year":
        return "INT"
    if col == "label":
        return "INT"
    if col == "sj_label":
        return "DOUBLE PRECISION"
    return "DOUBLE PRECISION"


def main():
    df_head = pd.read_csv(CSV_PATH, nrows=1)
    columns = df_head.columns.tolist()

    if "company_id" not in columns:
        raise ValueError("Không tìm thấy cột company_id trong financial_node1.csv")

    if "year" not in columns:
        raise ValueError("Không tìm thấy cột year trong financial_node1.csv")

    if "label" not in columns:
        raise ValueError("Không tìm thấy cột label trong financial_node1.csv")

    # Giữ nguyên thứ tự cột như trong file CSV để khi COPY dữ liệu không bị lệch cột
    ordered_cols = columns

    financial_cols_sql = []
    for col in ordered_cols:
        col_name = safe_column_name(col)
        col_type = infer_sql_type(col)

        if col in ["company_id", "year", "label"]:
            financial_cols_sql.append(f"    {col_name} {col_type} NOT NULL")
        else:
            financial_cols_sql.append(f"    {col_name} {col_type}")

    financial_cols_block = ",\n".join(financial_cols_sql)

    sql = f"""DROP TABLE IF EXISTS financial_statement;
DROP TABLE IF EXISTS year_dim;
DROP TABLE IF EXISTS company;

CREATE TABLE company (
    company_id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL
);

CREATE TABLE year_dim (
    year INT PRIMARY KEY,
    dataset_role VARCHAR(30) NOT NULL,
    node_name VARCHAR(50) NOT NULL,
    note TEXT
);

CREATE TABLE financial_statement (
{financial_cols_block},

    PRIMARY KEY (company_id, year),

    CONSTRAINT fk_financial_company
        FOREIGN KEY (company_id)
        REFERENCES company(company_id),

    CONSTRAINT fk_financial_year
        FOREIGN KEY (year)
        REFERENCES year_dim(year)
);

CREATE INDEX idx_financial_year
    ON financial_statement(year);

CREATE INDEX idx_financial_label
    ON financial_statement(label);

CREATE INDEX idx_financial_company
    ON financial_statement(company_id);
"""

    OUTPUT_SQL.write_text(sql, encoding="utf-8")

    print(f"Đã tạo file SQL: {OUTPUT_SQL}")
    print(f"Số cột trong financial_statement: {len(ordered_cols)}")
    print("10 cột đầu:", ordered_cols[:10])
    print("10 cột cuối:", ordered_cols[-10:])


if __name__ == "__main__":
    main()
