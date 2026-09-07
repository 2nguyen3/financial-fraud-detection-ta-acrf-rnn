import csv
import time
from pathlib import Path

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config


CSV_PATH = Path("graph_data/attention_relation_edge.csv")

NEBULA_HOST = "127.0.0.1"
NEBULA_PORT = 9669
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
NEBULA_SPACE = "fraud_graph"

BATCH_SIZE = 100


def escape_string(value: str) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def to_int(value):
    if value is None or value == "":
        return 0
    return int(float(value))


def to_float(value):
    if value is None or value == "":
        return 0.0
    return float(value)


def build_insert_statement(rows):
    props = (
        "`year`, "
        "attention_weight, "
        "attention_weight_raw, "
        "dst_mean_attention, "
        "src_company_id, "
        "dst_company_id, "
        "src_label, "
        "dst_label, "
        "both_fraud, "
        "edge_rank, "
        "`method`"
    )

    values_sql = []

    for row in rows:
        src = escape_string(row["src"])
        dst = escape_string(row["dst"])

        year = to_int(row["year"])
        attention_weight = to_float(row["attention_weight"])
        attention_weight_raw = to_float(row["attention_weight_raw"])
        dst_mean_attention = to_float(row["dst_mean_attention"])

        src_company_id = to_int(row["src_company_id"])
        dst_company_id = to_int(row["dst_company_id"])
        src_label = to_int(row["src_label"])
        dst_label = to_int(row["dst_label"])
        both_fraud = to_int(row["both_fraud"])
        edge_rank = to_int(row["edge_rank"])
        method = escape_string(row["method"])

        values_sql.append(
            f'"{src}"->"{dst}":('
            f'{year}, '
            f'{attention_weight}, '
            f'{attention_weight_raw}, '
            f'{dst_mean_attention}, '
            f'{src_company_id}, '
            f'{dst_company_id}, '
            f'{src_label}, '
            f'{dst_label}, '
            f'{both_fraud}, '
            f'{edge_rank}, '
            f'"{method}"'
            f')'
        )

    return f"INSERT EDGE attention_relation({props}) VALUES " + ", ".join(values_sql) + ";"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {CSV_PATH}")

    config = Config()
    config.max_connection_pool_size = 10

    connection_pool = ConnectionPool()
    ok = connection_pool.init([(NEBULA_HOST, NEBULA_PORT)], config)

    if not ok:
        raise RuntimeError("Không thể khởi tạo kết nối NebulaGraph.")

    session = connection_pool.get_session(NEBULA_USER, NEBULA_PASSWORD)

    try:
        result = session.execute(f"USE {NEBULA_SPACE};")
        if not result.is_succeeded():
            raise RuntimeError(f"USE space lỗi: {result.error_msg()}")

        print(f"Đã kết nối NebulaGraph space: {NEBULA_SPACE}")
        print(f"Đang import file: {CSV_PATH}")

        total_rows = 0
        success_batches = 0
        failed_batches = 0
        batch = []

        with CSV_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                batch.append(row)

                if len(batch) >= BATCH_SIZE:
                    stmt = build_insert_statement(batch)
                    result = session.execute(stmt)

                    if result.is_succeeded():
                        total_rows += len(batch)
                        success_batches += 1
                    else:
                        failed_batches += 1
                        print("Lỗi batch:")
                        print(result.error_msg())
                        print(stmt[:1000])

                    batch = []

            if batch:
                stmt = build_insert_statement(batch)
                result = session.execute(stmt)

                if result.is_succeeded():
                    total_rows += len(batch)
                    success_batches += 1
                else:
                    failed_batches += 1
                    print("Lỗi batch cuối:")
                    print(result.error_msg())
                    print(stmt[:1000])

        print("=" * 80)
        print("IMPORT ATTENTION RELATION DONE")
        print("=" * 80)
        print("total_inserted_rows:", total_rows)
        print("success_batches:", success_batches)
        print("failed_batches:", failed_batches)

        time.sleep(3)

        check_stmt = """
        MATCH ()-[e:attention_relation]->()
        RETURN count(e) AS edge_count;
        """
        result = session.execute(check_stmt)

        print("\nKiểm tra số cạnh attention_relation:")
        if result.is_succeeded():
            print(result)
        else:
            print("Query kiểm tra lỗi:", result.error_msg())

    finally:
        session.release()
        connection_pool.close()


if __name__ == "__main__":
    main()
