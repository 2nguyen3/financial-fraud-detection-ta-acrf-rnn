import argparse
import warnings

import pandas as pd
import psycopg2


warnings.filterwarnings("ignore", category=UserWarning)


DB_NODE_1 = {
    "host": "localhost",
    "port": 5433,
    "database": "fraud_db",
    "user": "fraud_user",
    "password": "fraud_pass",
}

DB_NODE_2 = {
    "host": "localhost",
    "port": 5434,
    "database": "fraud_db",
    "user": "fraud_user",
    "password": "fraud_pass",
}


NODE_1_YEARS = {2010, 2011, 2012, 2013, 2014, 2015, 2017}
NODE_2_YEARS = {2018, 2019, 2020}


def connect(db_config):
    return psycopg2.connect(**db_config)


def query_to_dataframe(db_config, sql, params=None):
    with connect(db_config) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_node_by_year(year):
    """
    Điều phối truy vấn theo phân mảnh ngang:
    - db_node_1 lưu các năm 2010, 2011, 2012, 2013, 2014, 2015, 2017
    - db_node_2 lưu các năm 2018, 2019, 2020
    """
    if year in NODE_1_YEARS:
        return "db_node_1", DB_NODE_1

    if year in NODE_2_YEARS:
        return "db_node_2", DB_NODE_2

    valid_years = sorted(NODE_1_YEARS | NODE_2_YEARS)
    raise ValueError(
        f"Năm {year} không tồn tại trong phân mảnh dữ liệu. "
        f"Các năm hợp lệ: {valid_years}"
    )


def print_dataframe(df):
    if df.empty:
        print("Không có dữ liệu phù hợp.")
    else:
        print(df.to_string(index=False))


def query_count_by_node():
    print("\n===== TRUY VẤN: SỐ DÒNG TRÊN TỪNG NODE =====")

    sql = "SELECT COUNT(*) AS financial_count FROM financial_statement;"

    node1_df = query_to_dataframe(DB_NODE_1, sql)
    node2_df = query_to_dataframe(DB_NODE_2, sql)

    node1_count = int(node1_df.loc[0, "financial_count"])
    node2_count = int(node2_df.loc[0, "financial_count"])

    result = pd.DataFrame(
        [
            {"node": "db_node_1", "financial_count": node1_count},
            {"node": "db_node_2", "financial_count": node2_count},
            {"node": "total", "financial_count": node1_count + node2_count},
        ]
    )

    print_dataframe(result)


def query_by_year(year, limit):
    print(f"\n===== TRUY VẤN: DỮ LIỆU THEO NĂM {year} =====")

    node_name, db_config = get_node_by_year(year)

    sql = """
        SELECT company_id, year, label
        FROM financial_statement
        WHERE year = %s
        ORDER BY company_id
        LIMIT %s;
    """

    df = query_to_dataframe(db_config, sql, params=(year, limit))

    print(f"Năm {year} được điều phối đến node: {node_name}")
    print_dataframe(df)


def query_label_distribution_all_nodes():
    print("\n===== TRUY VẤN: PHÂN BỐ NHÃN TOÀN HỆ THỐNG =====")

    sql = """
        SELECT label, COUNT(*) AS count
        FROM financial_statement
        GROUP BY label
        ORDER BY label;
    """

    node1_df = query_to_dataframe(DB_NODE_1, sql)
    node1_df["node"] = "db_node_1"

    node2_df = query_to_dataframe(DB_NODE_2, sql)
    node2_df["node"] = "db_node_2"

    combined = pd.concat([node1_df, node2_df], ignore_index=True)

    total_by_label = (
        combined
        .groupby("label", as_index=False)["count"]
        .sum()
        .sort_values("label")
    )
    total_by_label["node"] = "total"

    result = pd.concat(
        [
            combined[["node", "label", "count"]],
            total_by_label[["node", "label", "count"]],
        ],
        ignore_index=True,
    )

    print_dataframe(result)


def query_company_history(company_id):
    print(f"\n===== TRUY VẤN: LỊCH SỬ CÔNG TY {company_id} =====")

    sql = """
        SELECT company_id, year, label
        FROM financial_statement
        WHERE company_id = %s
        ORDER BY year;
    """

    node1_df = query_to_dataframe(DB_NODE_1, sql, params=(company_id,))
    node1_df["source_node"] = "db_node_1"

    node2_df = query_to_dataframe(DB_NODE_2, sql, params=(company_id,))
    node2_df["source_node"] = "db_node_2"

    combined = pd.concat([node1_df, node2_df], ignore_index=True)
    combined = combined.sort_values("year").reset_index(drop=True)

    print_dataframe(combined)


def query_fraud_companies_by_year(year, limit):
    print(f"\n===== TRUY VẤN: CÔNG TY GIAN LẬN TRONG NĂM {year} =====")

    node_name, db_config = get_node_by_year(year)

    sql = """
        SELECT company_id, year, label
        FROM financial_statement
        WHERE year = %s AND label = 1
        ORDER BY company_id
        LIMIT %s;
    """

    df = query_to_dataframe(db_config, sql, params=(year, limit))

    print(f"Năm {year} được điều phối đến node: {node_name}")
    print_dataframe(df)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Công cụ truy vấn dữ liệu phân tán trên PostgreSQL."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "count",
        help="Đếm số dòng dữ liệu tài chính trên từng node và toàn hệ thống."
    )

    year_parser = subparsers.add_parser(
        "year",
        help="Truy vấn dữ liệu theo năm, tự động điều phối đến đúng node."
    )
    year_parser.add_argument("--year", required=True, type=int, help="Năm cần truy vấn.")
    year_parser.add_argument("--limit", type=int, default=10, help="Số dòng hiển thị.")

    subparsers.add_parser(
        "labels",
        help="Tính phân bố nhãn trên từng node và toàn hệ thống."
    )

    company_parser = subparsers.add_parser(
        "company",
        help="Truy vấn lịch sử dữ liệu của một công ty trên toàn hệ thống."
    )
    company_parser.add_argument("--company", required=True, help="Mã công ty cần truy vấn.")

    fraud_parser = subparsers.add_parser(
        "fraud",
        help="Truy vấn danh sách công ty gian lận trong một năm."
    )
    fraud_parser.add_argument("--year", required=True, type=int, help="Năm cần truy vấn.")
    fraud_parser.add_argument("--limit", type=int, default=10, help="Số dòng hiển thị.")

    all_parser = subparsers.add_parser(
        "all",
        help="Chạy toàn bộ nhóm truy vấn minh chứng với tham số truyền vào."
    )
    all_parser.add_argument("--year", required=True, type=int, help="Năm dùng cho truy vấn theo năm.")
    all_parser.add_argument("--company", required=True, help="Mã công ty dùng cho truy vấn lịch sử.")
    all_parser.add_argument("--limit", type=int, default=10, help="Số dòng hiển thị.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "count":
        query_count_by_node()

    elif args.command == "year":
        query_by_year(args.year, args.limit)

    elif args.command == "labels":
        query_label_distribution_all_nodes()

    elif args.command == "company":
        query_company_history(args.company)

    elif args.command == "fraud":
        query_fraud_companies_by_year(args.year, args.limit)

    elif args.command == "all":
        query_count_by_node()
        query_by_year(args.year, args.limit)
        query_label_distribution_all_nodes()
        query_company_history(args.company)
        query_fraud_companies_by_year(args.year, args.limit)

    else:
        raise ValueError(f"Lệnh không hợp lệ: {args.command}")


if __name__ == "__main__":
    main()