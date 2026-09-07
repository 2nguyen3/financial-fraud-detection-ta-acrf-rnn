import argparse
import math
import os

import pandas as pd
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9669
DEFAULT_USER = "root"
DEFAULT_PASSWORD = "nebula"
DEFAULT_SPACE = "fraud_graph"

DEFAULT_INPUT = "graph_data/stable_attention_relation_edge.csv"


EDGE_COLUMNS = [
    "year_from",
    "year_to",
    "attention_from",
    "attention_to",
    "avg_attention",
    "attention_delta",
    "attention_raw_from",
    "attention_raw_to",
    "avg_attention_raw",
    "attention_raw_delta",
    "dst_mean_attention_from",
    "dst_mean_attention_to",
    "avg_dst_mean_attention",
    "edge_rank_from",
    "edge_rank_to",
    "avg_edge_rank",
    "src_label_from",
    "dst_label_from",
    "src_label_to",
    "dst_label_to",
    "same_label_from",
    "same_label_to",
    "same_label_stable",
    "both_fraud_from",
    "both_fraud_to",
    "both_fraud_stable",
    "method_from",
    "method_to",
]


INT_COLUMNS = {
    "year_from",
    "year_to",
    "edge_rank_from",
    "edge_rank_to",
    "src_label_from",
    "dst_label_from",
    "src_label_to",
    "dst_label_to",
    "same_label_from",
    "same_label_to",
    "same_label_stable",
    "both_fraud_from",
    "both_fraud_to",
    "both_fraud_stable",
}


FLOAT_COLUMNS = {
    "attention_from",
    "attention_to",
    "avg_attention",
    "attention_delta",
    "attention_raw_from",
    "attention_raw_to",
    "avg_attention_raw",
    "attention_raw_delta",
    "dst_mean_attention_from",
    "dst_mean_attention_to",
    "avg_dst_mean_attention",
    "avg_edge_rank",
}


STRING_COLUMNS = {
    "method_from",
    "method_to",
}


def escape_ngql_string(value):
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    return f'"{text}"'


def to_ngql_value(value, column):
    if pd.isna(value):
        return "NULL"

    if column in INT_COLUMNS:
        return str(int(float(value)))

    if column in FLOAT_COLUMNS:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return "NULL"
        return repr(number)

    if column in STRING_COLUMNS:
        return escape_ngql_string(value)

    return escape_ngql_string(value)


def validate_dataframe(df):
    required = ["src", "dst"] + EDGE_COLUMNS
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc trong stable_attention_relation_edge.csv: {missing}")

    if df.empty:
        raise ValueError("File stable_attention_relation_edge.csv không có dữ liệu.")


def connect_nebula(host, port, user, password, space):
    config = Config()
    config.max_connection_pool_size = 10

    pool = ConnectionPool()
    ok = pool.init([(host, port)], config)

    if not ok:
        raise RuntimeError("Không thể khởi tạo connection pool tới NebulaGraph.")

    session = pool.get_session(user, password)
    result = session.execute(f"USE {space};")

    if not result.is_succeeded():
        raise RuntimeError(f"Không thể USE space {space}: {result.error_msg()}")

    return pool, session


def build_insert_query(batch_df):
    columns_text = ", ".join(EDGE_COLUMNS)
    values_parts = []

    for _, row in batch_df.iterrows():
        src = str(row["src"])
        dst = str(row["dst"])

        edge_values = [to_ngql_value(row[col], col) for col in EDGE_COLUMNS]
        edge_values_text = ", ".join(edge_values)

        values_parts.append(f'"{src}"->"{dst}":({edge_values_text})')

    values_text = ",\n".join(values_parts)

    query = (
        f"INSERT EDGE stable_attention_relation({columns_text}) VALUES\n"
        f"{values_text};"
    )

    return query


def import_edges(session, df, batch_size):
    total_rows = len(df)
    success_batches = 0
    failed_batches = 0
    inserted_rows = 0

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_df = df.iloc[start:end]

        query = build_insert_query(batch_df)
        result = session.execute(query)

        if result.is_succeeded():
            success_batches += 1
            inserted_rows += len(batch_df)
            print(f"OK batch {start + 1}-{end}/{total_rows}")
        else:
            failed_batches += 1
            print(f"FAILED batch {start + 1}-{end}/{total_rows}")
            print(result.error_msg())
            print(query)
            raise RuntimeError("Import stable_attention_relation thất bại.")

    return {
        "total_rows": total_rows,
        "inserted_rows": inserted_rows,
        "success_batches": success_batches,
        "failed_batches": failed_batches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import stable_attention_relation_edge.csv into NebulaGraph."
    )

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=int, default=50)

    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--space", default=DEFAULT_SPACE)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Không tìm thấy input file: {args.input}")

    df = pd.read_csv(args.input)
    validate_dataframe(df)

    print("===== IMPORT STABLE_ATTENTION_RELATION TO NEBULAGRAPH =====")
    print(f"Input file : {args.input}")
    print(f"Rows       : {len(df)}")
    print(f"Batch size : {args.batch_size}")
    print(f"Host       : {args.host}:{args.port}")
    print(f"Space      : {args.space}")
    print("")

    pool = None
    session = None

    try:
        pool, session = connect_nebula(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            space=args.space,
        )

        stats = import_edges(
            session=session,
            df=df,
            batch_size=args.batch_size,
        )

        print("")
        print("===== IMPORT SUMMARY =====")
        for key, value in stats.items():
            print(f"{key}: {value}")

    finally:
        if session is not None:
            session.release()
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    main()
