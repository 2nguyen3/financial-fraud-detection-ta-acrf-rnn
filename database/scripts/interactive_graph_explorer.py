import argparse
import os
import re
import webbrowser
from collections import deque

from pyvis.network import Network
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9669
DEFAULT_USER = "root"
DEFAULT_PASSWORD = "nebula"
DEFAULT_SPACE = "fraud_graph"


def clean_value(value):
    """
    Chuyển dữ liệu trả về từ NebulaGraph sang kiểu Python dễ xử lý.

    Một số phiên bản nebula-python trả value dạng:
    - Value( sVal=b'101_2018' )
    - Value( fVal=0.407066 )
    - Value( iVal=1 )

    Hàm này sẽ bóc các dạng đó thành:
    - string
    - float
    - int
    """
    if value is None:
        return None

    text = str(value).strip()

    # Parse string: Value( sVal=b'101_2018' )
    m = re.search(r"sVal=b[\'\"]([^\'\"]+)[\'\"]", text)
    if m:
        return m.group(1)

    # Parse float: Value( fVal=0.407066 )
    m = re.search(r"fVal=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text)
    if m:
        return float(m.group(1))

    # Parse int: Value( iVal=1 )
    m = re.search(r"iVal=([-+]?[0-9]+)", text)
    if m:
        return int(m.group(1))

    # Parse bool nếu có
    m = re.search(r"bVal=(true|false|True|False)", text)
    if m:
        return m.group(1).lower() == "true"

    # Nếu là chuỗi có quote bình thường
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]

    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]

    # Nếu là số dạng text bình thường
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        pass

    # Thử các method của ValueWrapper nếu có
    for method in ["as_string", "as_double", "as_int", "as_bool"]:
        try:
            v = getattr(value, method)()
            if v is not None and v != "":
                return v
        except Exception:
            pass

    return text


def normalize_key(key):
    """
    Chuẩn hóa tên cột trả về từ NebulaGraph.
    Một số phiên bản nebula-python có thể trả key ở dạng bytes/object.
    """
    if isinstance(key, bytes):
        return key.decode("utf-8")

    text = str(key).strip()

    if text.startswith("b'") and text.endswith("'"):
        return text[2:-1]

    if text.startswith('b"') and text.endswith('"'):
        return text[2:-1]

    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]

    return text


def result_to_dicts(result):
    """
    Chuyển ResultSet của NebulaGraph thành list[dict].
    """
    keys = [normalize_key(k) for k in list(result.keys())]
    rows = []

    for row in result.rows():
        item = {}

        values = row.values
        if callable(values):
            values = values()

        for idx, key in enumerate(keys):
            item[key] = clean_value(values[idx])

        rows.append(item)

    return rows


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


def run_query(session, query):
    result = session.execute(query)

    if not result.is_succeeded():
        raise RuntimeError(
            "NebulaGraph query lỗi:\n"
            f"{query}\n"
            f"Error: {result.error_msg()}"
        )

    return result_to_dicts(result)


def parse_company_year(node_id):
    """
    Ví dụ: 101_2018 -> company_id=101, year=2018.
    """
    text = str(node_id)
    match = re.match(r"^(.+?)_(\d{4})$", text)

    if match:
        return match.group(1), int(match.group(2))

    return text, None


def get_neighbors(session, node_id, min_similarity, top_k, edge_type="similar_in_year"):
    """
    Lấy neighbor từ NebulaGraph.

    edge_type:
    - similar_in_year: dùng similarity cosine cũ.
    - attention_relation: dùng attention_weight đã khử hub theo node đích.
    """
    if edge_type == "attention_relation":
        query = f"""GO FROM \"{node_id}\" OVER attention_relation YIELD src(edge) AS source_node, dst(edge) AS target_node, properties(edge).attention_weight AS similarity, properties(edge).attention_weight_raw AS attention_weight_raw, properties(edge).dst_mean_attention AS dst_mean_attention, properties(edge).edge_rank AS edge_rank, properties(edge).src_label AS src_label, properties(edge).dst_label AS dst_label, properties(edge).both_fraud AS both_fraud, properties(edge).`year` AS year;"""
    else:
        query = f"""GO FROM \"{node_id}\" OVER similar_in_year YIELD src(edge) AS source_node, dst(edge) AS target_node, properties(edge).similarity AS similarity, properties(edge).src_label AS src_label, properties(edge).dst_label AS dst_label;"""

    rows = run_query(session, query)

    cleaned = []
    for row in rows:
        try:
            score = float(row["similarity"])
        except Exception:
            continue

        if score < min_similarity:
            continue

        item = {
            "source_node": str(row["source_node"]),
            "target_node": str(row["target_node"]),
            "similarity": score,
            "src_label": int(row["src_label"]),
            "dst_label": int(row["dst_label"]),
            "edge_type": edge_type,
        }

        for optional_key in ["attention_weight_raw", "dst_mean_attention", "edge_rank", "both_fraud", "year"]:
            if optional_key in row:
                item[optional_key] = row[optional_key]

        cleaned.append(item)

    cleaned.sort(key=lambda x: x["similarity"], reverse=True)
    return cleaned[:top_k]

def build_explorer_graph(session, center_node, steps, top_k, min_similarity, max_nodes, edge_type="similar_in_year"):
    nodes = {}
    edges = {}

    queue = deque()
    queue.append((center_node, 0))

    nodes[center_node] = {
        "id": center_node,
        "label": None,
        "depth": 0,
        "is_center": True,
    }

    visited_expand = set()

    while queue:
        current_node, depth = queue.popleft()

        if depth >= steps:
            continue

        if current_node in visited_expand:
            continue

        visited_expand.add(current_node)

        neighbors = get_neighbors(
            session=session,
            node_id=current_node,
            min_similarity=min_similarity,
            top_k=top_k,
            edge_type=edge_type,
        )

        for edge in neighbors:
            src = edge["source_node"]
            dst = edge["target_node"]

            if src not in nodes:
                nodes[src] = {
                    "id": src,
                    "label": edge["src_label"],
                    "depth": depth,
                    "is_center": src == center_node,
                }
            else:
                if nodes[src]["label"] is None:
                    nodes[src]["label"] = edge["src_label"]

            if dst not in nodes:
                if len(nodes) >= max_nodes:
                    continue

                nodes[dst] = {
                    "id": dst,
                    "label": edge["dst_label"],
                    "depth": depth + 1,
                    "is_center": False,
                }

                queue.append((dst, depth + 1))
            else:
                if nodes[dst]["label"] is None:
                    nodes[dst]["label"] = edge["dst_label"]

            edge_key = (src, dst)
            edges[edge_key] = edge

    return nodes, edges


def node_style(node):
    label_value = node.get("label")
    depth = node.get("depth", 0)

    if node.get("is_center"):
        color = "#f33e5d"
        size = 34
        border_width = 5
    elif label_value == 1:
        color = "#ff8a80"
        size = max(18, 28 - depth * 3)
        border_width = 3
    elif label_value == 0:
        color = "#0b9286"
        size = max(16, 25 - depth * 3)
        border_width = 2
    else:
        color = "#b0bec5"
        size = max(14, 22 - depth * 3)
        border_width = 2

    return color, size, border_width


def render_html(nodes, edges, center_node, output_path, steps, top_k, min_similarity, edge_type):
    net = Network(
        height="850px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=True,
        notebook=False,
    )

    net.repulsion(
        node_distance=280,
        central_gravity=0.06,
        spring_length=260,
        spring_strength=0.025,
        damping=0.82,
    )

    for node_id, node in nodes.items():
        company_id, year = parse_company_year(node_id)
        color, size, border_width = node_style(node)

        label_text = str(node_id)
        fraud_text = "Không rõ"
        if node.get("label") == 1:
            fraud_text = "Gian lận"
        elif node.get("label") == 0:
            fraud_text = "Không gian lận"

        title = (
            f"Node: {node_id}\n"
            f"Company: {company_id}\n"
            f"Year: {year}\n"
            f"Label: {fraud_text}\n"
            f"Depth: {node.get('depth')}"
        )

        net.add_node(
            node_id,
            label=label_text,
            title=title,
            color={
                "background": color,
                "border": "#101727",
                "highlight": {
                    "background": "#ffd166",
                    "border": "#f33e5d",
                },
            },
            size=size,
            borderWidth=border_width,
        )

    for (src, dst), edge in edges.items():
        sim = float(edge["similarity"])
        width = 1.0 + min(abs(sim), 8.0) * 0.35

        if edge.get("edge_type") == "attention_relation":
            title = (
                f"Source: {src}\n"
                f"Target: {dst}\n"
                f"Attention weight: {sim:.6f}\n"
                f"Raw attention: {edge.get('attention_weight_raw')}\n"
                f"Dst mean attention: {edge.get('dst_mean_attention')}\n"
                f"Edge rank: {edge.get('edge_rank')}\n"
                f"Year: {edge.get('year')}\n"
                f"Src label: {edge['src_label']}\n"
                f"Dst label: {edge['dst_label']}"
            )
        else:
            title = (
                f"Source: {src}\n"
                f"Target: {dst}\n"
                f"Similarity: {sim:.6f}\n"
                f"Src label: {edge['src_label']}\n"
                f"Dst label: {edge['dst_label']}"
            )

        net.add_edge(
            src,
            dst,
            value=sim,
            width=width,
            title=title,
            arrows="",
            color={
                "color": "rgba(120, 144, 156, 0.55)",
                "highlight": "#f33e5d",
            },
        )

    net.set_options("""
var options = {
  "nodes": {
    "font": {
      "size": 13,
      "face": "arial",
      "color": "#1f2937"
    },
    "shape": "dot"
  },
  "edges": {
    "smooth": {
      "type": "dynamic"
    },
    "font": {
      "size": 12,
      "align": "middle"
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "hideEdgesOnDrag": false,
    "navigationButtons": true,
    "keyboard": true
  },
  "physics": {
    "enabled": true,
    "solver": "repulsion",
    "stabilization": {
      "enabled": true,
      "iterations": 700,
      "fit": true
    }
  }
}
""")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net.write_html(output_path, notebook=False, open_browser=False)

    inject_summary_panel(
        output_path=output_path,
        center_node=center_node,
        node_count=len(nodes),
        edge_count=len(edges),
        steps=steps,
        top_k=top_k,
        min_similarity=min_similarity,
        edge_type=edge_type,
    )


def inject_summary_panel(output_path, center_node, node_count, edge_count, steps, top_k, min_similarity, edge_type):
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()

    panel = f"""
<div style="
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 9999;
    width: 360px;
    padding: 14px 16px;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #d0d7de;
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(16, 23, 39, 0.16);
    font-family: Arial, sans-serif;
    color: #101727;
">
  <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">
    Interactive Fraud / Attention Graph Explorer
  </div>
  <div style="font-size: 13px; line-height: 1.55;">
    <b>Center node:</b> {center_node}<br>
    <b>Expansion steps:</b> {steps}<br>
    <b>Top neighbors per node:</b> {top_k}<br>
    <b>Edge type:</b> {edge_type}<br>
    <b>Minimum score:</b> {min_similarity}<br>
    <b>Nodes:</b> {node_count}<br>
    <b>Edges:</b> {edge_count}
  </div>
  <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 10px 0;">
  <div style="font-size: 12px; line-height: 1.5;">
    <span style="display:inline-block;width:10px;height:10px;background:#f33e5d;border-radius:50%;"></span>
    Node trung tâm<br>
    <span style="display:inline-block;width:10px;height:10px;background:#ff8a80;border-radius:50%;"></span>
    Công ty có nhãn gian lận<br>
    <span style="display:inline-block;width:10px;height:10px;background:#0b9286;border-radius:50%;"></span>
    Công ty không gian lận
  </div>
</div>
"""

    html = html.replace("<body>", "<body>\n" + panel)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive graph explorer for fraud relationship graph in NebulaGraph."
    )

    parser.add_argument("--company", required=True, help="Company id, ví dụ: 101")
    parser.add_argument("--year", required=True, type=int, help="Year, ví dụ: 2018")
    parser.add_argument("--steps", type=int, default=2, help="Số bước mở rộng graph")
    parser.add_argument("--top-k", type=int, default=8, help="Số neighbor lấy từ mỗi node")
    parser.add_argument("--min-similarity", type=float, default=0.45, help="Ngưỡng similarity/attention_weight tối thiểu")
    parser.add_argument("--edge-type", choices=["similar_in_year", "attention_relation"], default="similar_in_year", help="Loại cạnh dùng để mở rộng graph")
    parser.add_argument("--max-nodes", type=int, default=80, help="Số node tối đa để tránh graph quá rối")
    parser.add_argument("--output", default=None, help="Đường dẫn file HTML output")
    parser.add_argument("--open", action="store_true", help="Mở file HTML sau khi tạo")

    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--space", default=DEFAULT_SPACE)

    args = parser.parse_args()

    center_node = f"{args.company}_{args.year}"

    if args.output is None:
        args.output = f"outputs/graph_explorer/company_{args.company}_{args.year}_{args.edge_type}_steps{args.steps}_top{args.top_k}.html"

    print("===== INTERACTIVE GRAPH EXPLORER =====")
    print(f"Center node       : {center_node}")
    print(f"Steps             : {args.steps}")
    print(f"Top-k             : {args.top_k}")
    print(f"Min score         : {args.min_similarity}")
    print(f"Edge type         : {args.edge_type}")
    print(f"Max nodes         : {args.max_nodes}")
    print(f"Output            : {args.output}")

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

        nodes, edges = build_explorer_graph(
            session=session,
            center_node=center_node,
            steps=args.steps,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            max_nodes=args.max_nodes,
            edge_type=args.edge_type,
        )

        render_html(
            nodes=nodes,
            edges=edges,
            center_node=center_node,
            output_path=args.output,
            steps=args.steps,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
            edge_type=args.edge_type,
        )

        print("")
        print("===== KẾT QUẢ =====")
        print(f"Số node: {len(nodes)}")
        print(f"Số edge: {len(edges)}")
        print(f"Đã tạo file HTML: {args.output}")

        if args.open:
            webbrowser.open("file://" + os.path.abspath(args.output))

    finally:
        if session is not None:
            session.release()
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    main()
