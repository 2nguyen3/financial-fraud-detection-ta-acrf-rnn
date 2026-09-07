import argparse
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool


NEBULA_HOST = "127.0.0.1"
NEBULA_PORT = 9669
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
NEBULA_SPACE = "fraud_graph"


def connect():
    config = Config()
    config.max_connection_pool_size = 10

    connection_pool = ConnectionPool()
    ok = connection_pool.init([(NEBULA_HOST, NEBULA_PORT)], config)

    if not ok:
        raise RuntimeError("Không thể kết nối đến NebulaGraph.")

    session = connection_pool.get_session(NEBULA_USER, NEBULA_PASSWORD)

    result = session.execute(f"USE {NEBULA_SPACE};")
    if not result.is_succeeded():
        raise RuntimeError(f"Không thể USE space {NEBULA_SPACE}: {result.error_msg()}")

    return connection_pool, session


def execute_query(session, query):
    result = session.execute(query)

    if not result.is_succeeded():
        raise RuntimeError(
            f"Lỗi khi chạy truy vấn:\n{query}\n\nError: {result.error_msg()}"
        )

    print(result)


def build_company_year_id(company_id, year):
    return f"{company_id}_{year}"


# ============================================================
# 1. Query timeline theo công ty
# ============================================================

def query_timeline(company_id):
    return f"""
GO FROM "{company_id}" OVER has_record
YIELD dst(edge) AS company_year_id,
      properties(edge).year AS year,
      properties(edge).label AS label;
"""


# ============================================================
# 2. Query similar_in_year
# ============================================================

def query_similar(company_id, year):
    company_year_id = build_company_year_id(company_id, year)

    return f"""
GO FROM "{company_year_id}" OVER similar_in_year
YIELD dst(edge) AS similar_company_year,
      properties(edge).similarity AS similarity,
      properties(edge).dst_label AS dst_label;
"""


def query_similar_fraud(company_id, year):
    company_year_id = build_company_year_id(company_id, year)

    return f"""
GO FROM "{company_year_id}" OVER similar_in_year
WHERE properties(edge).dst_label == 1
YIELD dst(edge) AS similar_company_year,
      properties(edge).similarity AS similarity,
      properties(edge).dst_label AS dst_label;
"""


def query_neighbor(company_id, year):
    company_year_id = build_company_year_id(company_id, year)

    return f"""
GO FROM "{company_year_id}" OVER similar_in_year
YIELD src(edge) AS source_company_year,
      dst(edge) AS target_company_year,
      properties(edge).similarity AS similarity,
      properties(edge).src_label AS src_label,
      properties(edge).dst_label AS dst_label;
"""


def query_risk_neighbor(company_id, year, min_similarity):
    """
    Tìm các công ty chưa gian lận nhưng tương đồng với một công ty gian lận.
    Điều kiện rủi ro:
    - src_label = 1: công ty nguồn đang xét có nhãn gian lận.
    - dst_label = 0: công ty liên quan chưa bị gắn nhãn gian lận.
    - similarity >= min_similarity: mức tương đồng tài chính đạt ngưỡng yêu cầu.
    """
    company_year_id = build_company_year_id(company_id, year)

    return f"""
GO FROM "{company_year_id}" OVER similar_in_year
YIELD src(edge) AS source_company_year,
      dst(edge) AS risk_neighbor,
      properties(edge).similarity AS similarity,
      properties(edge).src_label AS src_label,
      properties(edge).dst_label AS dst_label
| YIELD $-.source_company_year AS source_company_year,
        $-.risk_neighbor AS risk_neighbor,
        $-.similarity AS similarity,
        $-.src_label AS src_label,
        $-.dst_label AS dst_label
  WHERE $-.src_label == 1
    AND $-.dst_label == 0
    AND $-.similarity >= {float(min_similarity)};
"""


def query_expand_neighbor(company_id, year, steps, min_similarity):
    """
    Mở rộng vùng lân cận 1-n bước từ một công ty-năm trên edge similar_in_year.
    """
    if steps < 1:
        raise ValueError("steps phải >= 1")

    company_year_id = build_company_year_id(company_id, year)

    return f"""
GO 1 TO {int(steps)} STEPS FROM "{company_year_id}" OVER similar_in_year
WHERE properties(edge).similarity >= {float(min_similarity)}
YIELD src(edge) AS source_node,
      dst(edge) AS target_node,
      properties(edge).similarity AS similarity,
      properties(edge).src_label AS src_label,
      properties(edge).dst_label AS dst_label;
"""


# ============================================================
# 3. Query stable_similar
# ============================================================

def query_stable(company_id):
    return f"""
GO FROM "{company_id}" OVER stable_similar
YIELD dst(edge) AS similar_company,
      properties(edge).year_from AS year_from,
      properties(edge).year_to AS year_to,
      properties(edge).similarity_from AS similarity_from,
      properties(edge).similarity_to AS similarity_to,
      properties(edge).avg_similarity AS avg_similarity,
      properties(edge).both_fraud_to AS both_fraud_to;
"""


def query_stable_fraud(company_id):
    return f"""
GO FROM "{company_id}" OVER stable_similar
WHERE properties(edge).both_fraud_to == 1
YIELD dst(edge) AS similar_company,
      properties(edge).year_from AS year_from,
      properties(edge).year_to AS year_to,
      properties(edge).avg_similarity AS avg_similarity,
      properties(edge).both_fraud_to AS both_fraud_to;
"""


def query_stable_risk(company_id):
    """
    Tìm các quan hệ stable_similar mà ở mốc năm sau có ít nhất một bên liên quan đến gian lận.
    Điều kiện rủi ro: src_label_to = 1 OR dst_label_to = 1.
    """
    return f"""
GO FROM "{company_id}" OVER stable_similar
WHERE properties(edge).src_label_to == 1
   OR properties(edge).dst_label_to == 1
YIELD dst(edge) AS similar_company,
      properties(edge).year_from AS year_from,
      properties(edge).year_to AS year_to,
      properties(edge).similarity_from AS similarity_from,
      properties(edge).similarity_to AS similarity_to,
      properties(edge).avg_similarity AS avg_similarity,
      properties(edge).src_label_to AS src_label_to,
      properties(edge).dst_label_to AS dst_label_to,
      properties(edge).both_fraud_to AS both_fraud_to;
"""


# ============================================================
# 4. Query attention_relation
# ============================================================

def query_attention(company_id, year):
    """
    Lấy Top-K attention neighbors của một công ty-năm.
    Edge: attention_relation.
    """
    company_year_id = build_company_year_id(company_id, year)

    return f"""GO FROM \"{company_year_id}\" OVER attention_relation YIELD src(edge) AS source_company_year, dst(edge) AS attention_neighbor, properties(edge).`year` AS year, properties(edge).attention_weight AS attention_weight, properties(edge).attention_weight_raw AS attention_weight_raw, properties(edge).dst_mean_attention AS dst_mean_attention, properties(edge).edge_rank AS edge_rank, properties(edge).src_label AS src_label, properties(edge).dst_label AS dst_label, properties(edge).both_fraud AS both_fraud, properties(edge).`method` AS method | ORDER BY $-.edge_rank ASC;"""


def query_attention_stats():
    """
    Thống kê số cạnh attention_relation theo năm.
    """
    return """MATCH ()-[e:attention_relation]->() RETURN properties(e).`year` AS year, count(e) AS edge_count ORDER BY year;"""


def query_attention_fraud(year=None, limit=20):
    """
    Tìm các cạnh attention_relation mà cả src và dst đều có nhãn gian lận.
    """
    year_filter = ""
    if year is not None:
        year_filter = f" AND properties(e).`year` == {int(year)}"

    return f"""MATCH (src)-[e:attention_relation]->(dst) WHERE properties(e).both_fraud == 1{year_filter} RETURN id(src) AS src, id(dst) AS dst, properties(e).`year` AS year, properties(e).attention_weight AS attention_weight, properties(e).attention_weight_raw AS attention_weight_raw, properties(e).edge_rank AS edge_rank, properties(e).src_label AS src_label, properties(e).dst_label AS dst_label LIMIT {int(limit)};"""


def query_attention_top(year, limit=20):
    """
    Lấy các cạnh attention_relation có attention_weight cao nhất trong một năm.
    """
    return f"""MATCH (src)-[e:attention_relation]->(dst) WHERE properties(e).`year` == {int(year)} RETURN id(src) AS src, id(dst) AS dst, properties(e).attention_weight AS attention_weight, properties(e).attention_weight_raw AS attention_weight_raw, properties(e).edge_rank AS edge_rank, properties(e).src_label AS src_label, properties(e).dst_label AS dst_label, properties(e).both_fraud AS both_fraud ORDER BY attention_weight DESC LIMIT {int(limit)};"""


def query_attention_year_summary():
    """
    Thống kê tổng quan attention graph theo từng năm.
    Lưu ý: MATCH không tự trả năm có 0 fraud-fraud edge nếu dùng WHERE, nên phần fraud-fraud nên xem riêng.
    """
    return """MATCH ()-[e:attention_relation]->() RETURN properties(e).`year` AS year, count(e) AS edge_count, avg(properties(e).attention_weight) AS avg_attention_weight, min(properties(e).attention_weight) AS min_attention_weight, max(properties(e).attention_weight) AS max_attention_weight ORDER BY year;"""


# ============================================================
# 5. Query stable_attention_relation
# ============================================================

def query_stable_attention(company_id, limit=20):
    """
    Lấy các quan hệ chú ý ổn định của một công ty.
    Edge: stable_attention_relation.
    Node ở cấp công ty, ví dụ: 101 -> 487.
    """
    return f"""GO FROM "{company_id}" OVER stable_attention_relation YIELD dst(edge) AS stable_attention_company, properties(edge).year_from AS year_from, properties(edge).year_to AS year_to, properties(edge).attention_from AS attention_from, properties(edge).attention_to AS attention_to, properties(edge).avg_attention AS avg_attention, properties(edge).attention_delta AS attention_delta, properties(edge).edge_rank_from AS edge_rank_from, properties(edge).edge_rank_to AS edge_rank_to, properties(edge).src_label_to AS src_label_to, properties(edge).dst_label_to AS dst_label_to, properties(edge).same_label_stable AS same_label_stable, properties(edge).both_fraud_stable AS both_fraud_stable, properties(edge).method_from AS method_from, properties(edge).method_to AS method_to | ORDER BY $-.avg_attention DESC | LIMIT {int(limit)};"""


def query_stable_attention_risk(company_id, limit=20):
    """
    Tìm các quan hệ stable_attention_relation có liên quan đến gian lận.
    Điều kiện:
    - src_label_to = 1 hoặc dst_label_to = 1
    """
    return f"""GO FROM "{company_id}" OVER stable_attention_relation WHERE properties(edge).src_label_to == 1 OR properties(edge).dst_label_to == 1 YIELD dst(edge) AS stable_attention_company, properties(edge).year_from AS year_from, properties(edge).year_to AS year_to, properties(edge).attention_from AS attention_from, properties(edge).attention_to AS attention_to, properties(edge).avg_attention AS avg_attention, properties(edge).edge_rank_from AS edge_rank_from, properties(edge).edge_rank_to AS edge_rank_to, properties(edge).src_label_to AS src_label_to, properties(edge).dst_label_to AS dst_label_to, properties(edge).same_label_stable AS same_label_stable, properties(edge).both_fraud_stable AS both_fraud_stable | ORDER BY $-.avg_attention DESC | LIMIT {int(limit)};"""


def query_stable_attention_stats():
    """
    Thống kê tổng quan stable_attention_relation theo cặp năm.
    """
    return """MATCH ()-[e:stable_attention_relation]->() RETURN properties(e).year_from AS year_from, properties(e).year_to AS year_to, count(e) AS edge_count, avg(properties(e).avg_attention) AS avg_attention, sum(properties(e).same_label_stable) AS same_label_edges, sum(properties(e).both_fraud_stable) AS fraud_fraud_edges ORDER BY year_from, year_to;"""


def query_stable_attention_top(limit=20):
    """
    Lấy các cạnh stable_attention_relation có avg_attention cao nhất.
    """
    return f"""MATCH (src)-[e:stable_attention_relation]->(dst) RETURN id(src) AS src, id(dst) AS dst, properties(e).year_from AS year_from, properties(e).year_to AS year_to, properties(e).attention_from AS attention_from, properties(e).attention_to AS attention_to, properties(e).avg_attention AS avg_attention, properties(e).edge_rank_from AS edge_rank_from, properties(e).edge_rank_to AS edge_rank_to, properties(e).same_label_stable AS same_label_stable, properties(e).both_fraud_stable AS both_fraud_stable ORDER BY avg_attention DESC LIMIT {int(limit)};"""

# ============================================================
# 6. CLI parser
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Công cụ truy vấn linh hoạt NebulaGraph cho quan hệ liên công ty theo thời gian."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    timeline = subparsers.add_parser(
        "timeline",
        help="Lấy chuỗi bản ghi theo thời gian của một công ty."
    )
    timeline.add_argument("--company", required=True, help="Mã công ty, ví dụ: 1")

    similar = subparsers.add_parser(
        "similar",
        help="Tìm các công ty tương đồng với một công ty trong một năm."
    )
    similar.add_argument("--company", required=True, help="Mã công ty, ví dụ: 101")
    similar.add_argument("--year", required=True, type=int, help="Năm, ví dụ: 2018")

    similar_fraud = subparsers.add_parser(
        "similar-fraud",
        help="Tìm các công ty tương đồng có nhãn gian lận."
    )
    similar_fraud.add_argument("--company", required=True, help="Mã công ty, ví dụ: 101")
    similar_fraud.add_argument("--year", required=True, type=int, help="Năm, ví dụ: 2018")

    stable = subparsers.add_parser(
        "stable",
        help="Tìm quan hệ tương đồng ổn định qua các mốc thời gian."
    )
    stable.add_argument("--company", required=True, help="Mã công ty, ví dụ: 101")

    stable_fraud = subparsers.add_parser(
        "stable-fraud",
        help="Tìm quan hệ tương đồng ổn định có liên quan đến nhãn gian lận."
    )
    stable_fraud.add_argument("--company", required=True, help="Mã công ty, ví dụ: 101")

    neighbor = subparsers.add_parser(
        "neighbor",
        help="Trích xuất vùng lân cận graph của một công ty tại một năm."
    )
    neighbor.add_argument("--company", required=True, help="Mã công ty, ví dụ: 101")
    neighbor.add_argument("--year", required=True, type=int, help="Năm, ví dụ: 2018")

    risk_neighbor = subparsers.add_parser(
        "risk-neighbor",
        help="Tìm công ty chưa gian lận nhưng tương đồng với công ty gian lận."
    )
    risk_neighbor.add_argument("--company", required=True, help="Mã công ty, ví dụ: 101")
    risk_neighbor.add_argument("--year", required=True, type=int, help="Năm, ví dụ: 2018")
    risk_neighbor.add_argument(
        "--min-similarity",
        type=float,
        default=0.45,
        help="Ngưỡng similarity tối thiểu, mặc định là 0.45"
    )

    expand_neighbor = subparsers.add_parser(
        "expand-neighbor",
        help="Mở rộng vùng lân cận nhiều bước từ một công ty-năm."
    )
    expand_neighbor.add_argument("--company", required=True, help="Mã công ty, ví dụ: 101")
    expand_neighbor.add_argument("--year", required=True, type=int, help="Năm, ví dụ: 2018")
    expand_neighbor.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Số bước mở rộng, mặc định là 2"
    )
    expand_neighbor.add_argument(
        "--min-similarity",
        type=float,
        default=0.45,
        help="Ngưỡng similarity tối thiểu, mặc định là 0.45"
    )

    stable_risk = subparsers.add_parser(
        "stable-risk",
        help="Tìm quan hệ stable_similar có liên quan đến gian lận."
    )
    stable_risk.add_argument("--company", required=True, help="Mã công ty, ví dụ: 55")

    attention = subparsers.add_parser(
        "attention",
        help="Lấy Top-K attention neighbors của một công ty trong một năm."
    )
    attention.add_argument("--company", required=True, help="Mã công ty, ví dụ: 1")
    attention.add_argument("--year", required=True, type=int, help="Năm, ví dụ: 2018")

    attention_stats = subparsers.add_parser(
        "attention-stats",
        help="Thống kê số cạnh attention_relation theo năm."
    )

    attention_fraud = subparsers.add_parser(
        "attention-fraud",
        help="Tìm các cạnh attention_relation mà cả src và dst đều gian lận."
    )
    attention_fraud.add_argument("--year", type=int, default=None, help="Lọc theo năm, ví dụ: 2018")
    attention_fraud.add_argument("--limit", type=int, default=20, help="Số dòng trả về")

    attention_top = subparsers.add_parser(
        "attention-top",
        help="Lấy các cạnh attention_relation có attention_weight cao nhất trong một năm."
    )
    attention_top.add_argument("--year", required=True, type=int, help="Năm, ví dụ: 2018")
    attention_top.add_argument("--limit", type=int, default=20, help="Số dòng trả về")

    attention_year_summary = subparsers.add_parser(
        "attention-year-summary",
        help="Thống kê tổng quan attention graph theo từng năm."
    )

    stable_attention = subparsers.add_parser(
        "stable-attention",
        help="Lấy các quan hệ chú ý ổn định của một công ty."
    )
    stable_attention.add_argument("--company", required=True, help="Mã công ty, ví dụ: 241")
    stable_attention.add_argument("--limit", type=int, default=20, help="Số dòng trả về")

    stable_attention_risk = subparsers.add_parser(
        "stable-attention-risk",
        help="Tìm các quan hệ chú ý ổn định có liên quan đến gian lận."
    )
    stable_attention_risk.add_argument("--company", required=True, help="Mã công ty, ví dụ: 241")
    stable_attention_risk.add_argument("--limit", type=int, default=20, help="Số dòng trả về")

    stable_attention_stats = subparsers.add_parser(
        "stable-attention-stats",
        help="Thống kê tổng quan stable_attention_relation theo cặp năm."
    )

    stable_attention_top = subparsers.add_parser(
        "stable-attention-top",
        help="Lấy các cạnh stable_attention_relation có avg_attention cao nhất."
    )
    stable_attention_top.add_argument("--limit", type=int, default=20, help="Số dòng trả về")

    raw = subparsers.add_parser(
        "raw",
        help="Chạy một truy vấn nGQL bất kỳ."
    )
    raw.add_argument("--query", required=True, help="Câu truy vấn nGQL đặt trong dấu nháy.")

    return parser


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "timeline":
        query = query_timeline(args.company)

    elif args.command == "similar":
        query = query_similar(args.company, args.year)

    elif args.command == "similar-fraud":
        query = query_similar_fraud(args.company, args.year)

    elif args.command == "stable":
        query = query_stable(args.company)

    elif args.command == "stable-fraud":
        query = query_stable_fraud(args.company)

    elif args.command == "neighbor":
        query = query_neighbor(args.company, args.year)

    elif args.command == "risk-neighbor":
        query = query_risk_neighbor(args.company, args.year, args.min_similarity)

    elif args.command == "expand-neighbor":
        query = query_expand_neighbor(
            args.company,
            args.year,
            args.steps,
            args.min_similarity,
        )

    elif args.command == "stable-risk":
        query = query_stable_risk(args.company)

    elif args.command == "attention":
        query = query_attention(args.company, args.year)

    elif args.command == "attention-stats":
        query = query_attention_stats()

    elif args.command == "attention-fraud":
        query = query_attention_fraud(args.year, args.limit)

    elif args.command == "attention-top":
        query = query_attention_top(args.year, args.limit)

    elif args.command == "attention-year-summary":
        query = query_attention_year_summary()

    elif args.command == "stable-attention":
        query = query_stable_attention(args.company, args.limit)

    elif args.command == "stable-attention-risk":
        query = query_stable_attention_risk(args.company, args.limit)

    elif args.command == "stable-attention-stats":
        query = query_stable_attention_stats()

    elif args.command == "stable-attention-top":
        query = query_stable_attention_top(args.limit)

    elif args.command == "raw":
        query = args.query

    else:
        raise ValueError(f"Lệnh không hợp lệ: {args.command}")

    print("===== nGQL QUERY =====")
    print(query.strip())

    print("\n===== RESULT =====")

    connection_pool, session = connect()

    try:
        execute_query(session, query)
    finally:
        session.release()
        connection_pool.close()


if __name__ == "__main__":
    main()
