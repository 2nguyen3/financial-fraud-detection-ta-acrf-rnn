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
    """Escape chuỗi để đưa vào nGQL."""
    if pd.isna(value):
        return ""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def execute(session, query):
    result = session.execute(query)
    if not result.is_succeeded():
        raise RuntimeError(
            f"Lỗi khi chạy nGQL:\n{query}\n"
            f"Error: {result.error_msg()}"
        )
    return result


def main():
    company_path = GRAPH_DATA_DIR / "company_vertex.csv"
    company_year_path = GRAPH_DATA_DIR / "company_year_vertex.csv"
    edge_path = GRAPH_DATA_DIR / "has_record_edge.csv"

    for path in [company_path, company_year_path, edge_path]:
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

        company_df = pd.read_csv(company_path, dtype=str)
        company_year_df = pd.read_csv(company_year_path, dtype=str)
        edge_df = pd.read_csv(edge_path, dtype=str)

        print("===== IMPORT COMPANY VERTEX =====")
        for _, row in company_df.iterrows():
            company_id = escape_str(row["company_id"])
            ngql = (
                f'INSERT VERTEX company(company_id) '
                f'VALUES "{company_id}":("{company_id}");'
            )
            execute(session, ngql)
        print(f"Đã import company vertex: {len(company_df)}")

        print("\n===== IMPORT COMPANY_YEAR VERTEX =====")
        for _, row in company_year_df.iterrows():
            company_year_id = escape_str(row["company_year_id"])
            company_id = escape_str(row["company_id"])
            year = int(row["year"])
            label = int(row["label"])

            ngql = (
                f'INSERT VERTEX company_year(company_year_id, company_id, year, label) '
                f'VALUES "{company_year_id}":("{company_year_id}", "{company_id}", {year}, {label});'
            )
            execute(session, ngql)
        print(f"Đã import company_year vertex: {len(company_year_df)}")

        print("\n===== IMPORT HAS_RECORD EDGE =====")
        for _, row in edge_df.iterrows():
            src = escape_str(row["src"])
            dst = escape_str(row["dst"])
            year = int(row["year"])
            label = int(row["label"])

            ngql = (
                f'INSERT EDGE has_record(year, label) '
                f'VALUES "{src}"->"{dst}":({year}, {label});'
            )
            execute(session, ngql)
        print(f"Đã import has_record edge: {len(edge_df)}")

        print("\nKẾT QUẢ: Import dữ liệu graph vào NebulaGraph thành công.")

    finally:
        session.release()
        connection_pool.close()


if __name__ == "__main__":
    main()
