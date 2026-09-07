from pathlib import Path

import pandas as pd
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool


BASE_DIR = Path(__file__).resolve().parents[1]
GRAPH_DATA_DIR = BASE_DIR / "graph_data"

NEBULA_HOST = "127.0.0.1"
NEBULA_PORT = 9669
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
NEBULA_SPACE = "fraud_graph"


def escape_str(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def execute(session, query):
    result = session.execute(query)
    if not result.is_succeeded():
        raise RuntimeError(
            f"Lỗi khi chạy nGQL:\n{query}\nError: {result.error_msg()}"
        )
    return result


def main():
    similar_path = GRAPH_DATA_DIR / "similar_in_year_edge.csv"
    stable_path = GRAPH_DATA_DIR / "stable_similar_edge.csv"

    for path in [similar_path, stable_path]:
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

    config = Config()
    config.max_connection_pool_size = 10

    connection_pool = ConnectionPool()
    ok = connection_pool.init([(NEBULA_HOST, NEBULA_PORT)], config)
    if not ok:
        raise RuntimeError("Không kết nối được NebulaGraph.")

    session = connection_pool.get_session(NEBULA_USER, NEBULA_PASSWORD)

    try:
        execute(session, f"USE {NEBULA_SPACE};")

        similar_df = pd.read_csv(similar_path)
        stable_df = pd.read_csv(stable_path)

        print("===== IMPORT similar_in_year EDGE =====")
        for _, row in similar_df.iterrows():
            src = escape_str(row["src"])
            dst = escape_str(row["dst"])
            year = int(row["year"])
            similarity = float(row["similarity"])
            src_label = int(row["src_label"])
            dst_label = int(row["dst_label"])

            ngql = (
                f'INSERT EDGE similar_in_year(year, similarity, src_label, dst_label) '
                f'VALUES "{src}"->"{dst}":({year}, {similarity}, {src_label}, {dst_label});'
            )
            execute(session, ngql)

        print(f"Đã import similar_in_year edge: {len(similar_df)}")

        print("\n===== IMPORT stable_similar EDGE =====")
        for _, row in stable_df.iterrows():
            # Ép src/dst về dạng mã công ty nguyên, tránh lỗi pandas chuyển 1 thành 1.0.
            src = str(int(row["src"]))
            dst = str(int(row["dst"]))
            rank = int(row["edge_rank"])

            year_from = int(row["year_from"])
            year_to = int(row["year_to"])
            similarity_from = float(row["similarity_from"])
            similarity_to = float(row["similarity_to"])
            avg_similarity = float(row["avg_similarity"])
            src_label_to = int(row["src_label_to"])
            dst_label_to = int(row["dst_label_to"])
            both_fraud_to = int(row["both_fraud_to"])

            ngql = (
                f'INSERT EDGE stable_similar(year_from, year_to, similarity_from, similarity_to, avg_similarity, src_label_to, dst_label_to, both_fraud_to) '
                f'VALUES "{src}"->"{dst}"@{rank}:({year_from}, {year_to}, {similarity_from}, {similarity_to}, {avg_similarity}, {src_label_to}, {dst_label_to}, {both_fraud_to});'
            )
            execute(session, ngql)

        print(f"Đã import stable_similar edge: {len(stable_df)}")

        print("\nKẾT QUẢ: Import quan hệ liên công ty vào NebulaGraph thành công.")

    finally:
        session.release()
        connection_pool.close()


if __name__ == "__main__":
    main()
