import argparse
import os
import re

import pandas as pd


DEFAULT_INPUT = "graph_data/attention_relation_edge.csv"
DEFAULT_OUTPUT = "graph_data/stable_attention_relation_edge.csv"


def parse_company_id_from_node(node_id):
    """
    Ví dụ:
    101_2018 -> 101
    """
    text = str(node_id)
    match = re.match(r"^(.+?)_\d{4}$", text)

    if match:
        return int(match.group(1))

    return int(text)


def ensure_company_columns(df):
    """
    Đảm bảo dataframe có src_company_id và dst_company_id.
    Nếu file đã có sẵn thì dùng trực tiếp.
    Nếu chưa có thì parse từ src, dst.
    """
    if "src_company_id" not in df.columns:
        if "src" not in df.columns:
            raise ValueError("Thiếu cả src_company_id và src, không thể xác định công ty nguồn.")
        df["src_company_id"] = df["src"].apply(parse_company_id_from_node)

    if "dst_company_id" not in df.columns:
        if "dst" not in df.columns:
            raise ValueError("Thiếu cả dst_company_id và dst, không thể xác định công ty đích.")
        df["dst_company_id"] = df["dst"].apply(parse_company_id_from_node)

    return df


def validate_required_columns(df):
    required = [
        "src",
        "dst",
        "year",
        "attention_weight",
        "attention_weight_raw",
        "dst_mean_attention",
        "src_company_id",
        "dst_company_id",
        "src_label",
        "dst_label",
        "both_fraud",
        "edge_rank",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"File attention_relation_edge.csv thiếu các cột bắt buộc: {missing}")


def build_stable_attention_edges(df):
    """
    Tạo stable attention edge nếu cùng một cặp công ty src -> dst xuất hiện
    trong attention_relation ở hai năm liên tiếp.

    Ví dụ:
    101_2018 -> 487_2018
    101_2019 -> 487_2019

    Sẽ tạo:
    101 -> 487
    year_from = 2018
    year_to = 2019
    attention_from = ...
    attention_to = ...
    avg_attention = ...
    """

    df = df.copy()

    numeric_cols = [
        "year",
        "attention_weight",
        "attention_weight_raw",
        "dst_mean_attention",
        "src_company_id",
        "dst_company_id",
        "src_label",
        "dst_label",
        "both_fraud",
        "edge_rank",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["year", "src_company_id", "dst_company_id"])
    df["year"] = df["year"].astype(int)
    df["src_company_id"] = df["src_company_id"].astype(int)
    df["dst_company_id"] = df["dst_company_id"].astype(int)

    # Sắp xếp để so sánh cùng cặp công ty qua các năm liên tiếp
    df = df.sort_values(
        by=["src_company_id", "dst_company_id", "year", "edge_rank"],
        ascending=[True, True, True, True],
    )

    rows = []

    grouped = df.groupby(["src_company_id", "dst_company_id"], as_index=False)

    for (src_company_id, dst_company_id), group in grouped:
        group = group.sort_values("year").reset_index(drop=True)

        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i + 1]

            year_from = int(current_row["year"])
            year_to = int(next_row["year"])

            # Chỉ lấy quan hệ ổn định giữa hai năm liên tiếp
            if year_to - year_from != 1:
                continue

            attention_from = float(current_row["attention_weight"])
            attention_to = float(next_row["attention_weight"])

            attention_raw_from = float(current_row["attention_weight_raw"])
            attention_raw_to = float(next_row["attention_weight_raw"])

            dst_mean_attention_from = float(current_row["dst_mean_attention"])
            dst_mean_attention_to = float(next_row["dst_mean_attention"])

            edge_rank_from = int(current_row["edge_rank"])
            edge_rank_to = int(next_row["edge_rank"])

            src_label_from = int(current_row["src_label"])
            dst_label_from = int(current_row["dst_label"])
            src_label_to = int(next_row["src_label"])
            dst_label_to = int(next_row["dst_label"])

            both_fraud_from = int(current_row["both_fraud"])
            both_fraud_to = int(next_row["both_fraud"])

            method_from = str(current_row.get("method", ""))
            method_to = str(next_row.get("method", ""))

            rows.append(
                {
                    "src": str(src_company_id),
                    "dst": str(dst_company_id),
                    "year_from": year_from,
                    "year_to": year_to,

                    "attention_from": attention_from,
                    "attention_to": attention_to,
                    "avg_attention": (attention_from + attention_to) / 2.0,
                    "attention_delta": attention_to - attention_from,

                    "attention_raw_from": attention_raw_from,
                    "attention_raw_to": attention_raw_to,
                    "avg_attention_raw": (attention_raw_from + attention_raw_to) / 2.0,
                    "attention_raw_delta": attention_raw_to - attention_raw_from,

                    "dst_mean_attention_from": dst_mean_attention_from,
                    "dst_mean_attention_to": dst_mean_attention_to,
                    "avg_dst_mean_attention": (dst_mean_attention_from + dst_mean_attention_to) / 2.0,

                    "edge_rank_from": edge_rank_from,
                    "edge_rank_to": edge_rank_to,
                    "avg_edge_rank": (edge_rank_from + edge_rank_to) / 2.0,

                    "src_label_from": src_label_from,
                    "dst_label_from": dst_label_from,
                    "src_label_to": src_label_to,
                    "dst_label_to": dst_label_to,

                    "same_label_from": int(src_label_from == dst_label_from),
                    "same_label_to": int(src_label_to == dst_label_to),
                    "same_label_stable": int((src_label_from == dst_label_from) and (src_label_to == dst_label_to)),

                    "both_fraud_from": both_fraud_from,
                    "both_fraud_to": both_fraud_to,
                    "both_fraud_stable": int(both_fraud_from == 1 and both_fraud_to == 1),

                    "method_from": method_from,
                    "method_to": method_to,
                }
            )

    stable_df = pd.DataFrame(rows)

    if stable_df.empty:
        return stable_df

    stable_df = stable_df.sort_values(
        by=["year_from", "year_to", "avg_attention"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return stable_df


def print_summary(attention_df, stable_df):
    print("===== GENERATE STABLE ATTENTION RELATION EDGES =====")
    print(f"Input attention edges       : {len(attention_df)}")
    print(f"Input years                 : {sorted(attention_df['year'].dropna().astype(int).unique().tolist())}")

    if stable_df.empty:
        print("Stable attention edges      : 0")
        print("Cảnh báo: Không tìm thấy cặp attention src -> dst lặp lại ở hai năm liên tiếp.")
        return

    print(f"Stable attention edges      : {len(stable_df)}")
    print("")

    print("===== EDGE COUNT BY YEAR PAIR =====")
    print(
        stable_df.groupby(["year_from", "year_to"])
        .size()
        .reset_index(name="edge_count")
        .to_string(index=False)
    )

    print("")
    print("===== LABEL RATIOS =====")
    same_label_ratio = stable_df["same_label_stable"].mean()
    fraud_fraud_ratio = stable_df["both_fraud_stable"].mean()

    print(f"same_label_stable_ratio     : {same_label_ratio:.6f}")
    print(f"fraud_fraud_stable_ratio    : {fraud_fraud_ratio:.6f}")

    print("")
    print("===== ATTENTION SUMMARY =====")
    print(f"avg_attention               : {stable_df['avg_attention'].mean():.6f}")
    print(f"min_avg_attention           : {stable_df['avg_attention'].min():.6f}")
    print(f"max_avg_attention           : {stable_df['avg_attention'].max():.6f}")

    print("")
    print("===== TOP 10 STABLE ATTENTION EDGES =====")
    display_cols = [
        "src",
        "dst",
        "year_from",
        "year_to",
        "attention_from",
        "attention_to",
        "avg_attention",
        "edge_rank_from",
        "edge_rank_to",
        "src_label_to",
        "dst_label_to",
        "both_fraud_stable",
    ]

    print(stable_df[display_cols].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Generate stable_attention_relation_edge.csv from attention_relation_edge.csv"
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Đường dẫn file attention_relation_edge.csv",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Đường dẫn file stable_attention_relation_edge.csv",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Không tìm thấy input file: {args.input}")

    attention_df = pd.read_csv(args.input)
    attention_df = ensure_company_columns(attention_df)
    validate_required_columns(attention_df)

    stable_df = build_stable_attention_edges(attention_df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    stable_df.to_csv(args.output, index=False)

    print_summary(attention_df, stable_df)
    print("")
    print(f"Đã ghi output: {args.output}")


if __name__ == "__main__":
    main()
