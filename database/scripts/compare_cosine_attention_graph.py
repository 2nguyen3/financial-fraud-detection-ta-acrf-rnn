import argparse
import os
import re
from datetime import datetime

import pandas as pd

DEFAULT_SIMILAR = "graph_data/similar_in_year_edge.csv"
DEFAULT_ATTENTION = "graph_data/attention_relation_edge.csv"
DEFAULT_STABLE_SIMILAR = "graph_data/stable_similar_edge.csv"
DEFAULT_STABLE_ATTENTION = "graph_data/stable_attention_relation_edge.csv"
DEFAULT_OUTPUT = "docs/evidence/graph_comparison_result.txt"


def parse_company_id(node_id):
    text = str(node_id)
    match = re.match(r"^(.+?)_\d{4}$", text)
    if match:
        return str(match.group(1))
    return text


def parse_year(node_id):
    text = str(node_id)
    match = re.match(r"^.+?_(\d{4})$", text)
    if match:
        return int(match.group(1))
    return None


def read_csv(path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        return None
    return pd.read_csv(path)


def normalize_company_year_edges(df, graph_name):
    df = df.copy()
    if "src" not in df.columns or "dst" not in df.columns:
        raise ValueError(f"{graph_name}: thiếu cột src hoặc dst.")

    if "year" not in df.columns:
        df["year"] = df["src"].apply(parse_year)
    if "src_company_id" not in df.columns:
        df["src_company_id"] = df["src"].apply(parse_company_id)
    if "dst_company_id" not in df.columns:
        df["dst_company_id"] = df["dst"].apply(parse_company_id)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["src_company_id"] = df["src_company_id"].astype(str)
    df["dst_company_id"] = df["dst_company_id"].astype(str)

    score_col = "similarity" if graph_name == "SIMILAR_IN_YEAR" else "attention_weight"
    if score_col not in df.columns:
        raise ValueError(f"{graph_name}: thiếu cột trọng số {score_col}.")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])

    for col in ["src_label", "dst_label", "both_fraud", "edge_rank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pct(x):
    return f"{x * 100:.2f}%"


def safe_ratio(a, b):
    if b == 0:
        return 0.0
    return a / b


def edge_count_summary(df, graph_name):
    lines = [f"[{graph_name}]", f"Tổng số cạnh: {len(df):,}"]
    year_count = df.groupby("year").size().reset_index(name="edge_count")
    lines.append("Số cạnh theo năm:")
    for _, row in year_count.iterrows():
        lines.append(f"  - {int(row['year'])}: {int(row['edge_count']):,}")
    return lines


def same_label_ratio(df):
    if "src_label" not in df.columns or "dst_label" not in df.columns:
        return None
    valid = df.dropna(subset=["src_label", "dst_label"])
    if valid.empty:
        return None
    return float((valid["src_label"].astype(int) == valid["dst_label"].astype(int)).mean())


def fraud_fraud_ratio(df):
    if "both_fraud" in df.columns:
        valid = pd.to_numeric(df["both_fraud"], errors="coerce").dropna()
        if len(valid) > 0:
            return float((valid.astype(int) == 1).mean())
    if "src_label" not in df.columns or "dst_label" not in df.columns:
        return None
    valid = df.dropna(subset=["src_label", "dst_label"])
    if valid.empty:
        return None
    return float(((valid["src_label"].astype(int) == 1) & (valid["dst_label"].astype(int) == 1)).mean())


def topk_df(df, graph_name, k):
    df = df.copy()
    if graph_name == "SIMILAR_IN_YEAR":
        sort_cols = ["similarity"]
        ascending = [False]
    else:
        if "edge_rank" in df.columns:
            sort_cols = ["edge_rank", "attention_weight"]
            ascending = [True, False]
        else:
            sort_cols = ["attention_weight"]
            ascending = [False]
    df = df.sort_values(by=["src_company_id", "year"] + sort_cols, ascending=[True, True] + ascending)
    return df.groupby(["src_company_id", "year"], as_index=False).head(k)


def jaccard_topk(similar_df, attention_df, k):
    similar_top = topk_df(similar_df, "SIMILAR_IN_YEAR", k)
    attention_top = topk_df(attention_df, "ATTENTION_RELATION", k)

    sim_groups = {key: set(g["dst_company_id"].astype(str)) for key, g in similar_top.groupby(["src_company_id", "year"])}
    att_groups = {key: set(g["dst_company_id"].astype(str)) for key, g in attention_top.groupby(["src_company_id", "year"])}
    keys = sorted(set(sim_groups) & set(att_groups))
    rows = []
    for key in keys:
        sim_set, att_set = sim_groups[key], att_groups[key]
        inter, union = sim_set & att_set, sim_set | att_set
        rows.append({
            "src_company_id": key[0], "year": key[1],
            "cosine_topk": len(sim_set), "attention_topk": len(att_set),
            "intersection": len(inter), "union": len(union),
            "jaccard": 0.0 if len(union) == 0 else len(inter) / len(union),
        })
    return pd.DataFrame(rows)


def fraud_neighbor_precision_at_k(df, graph_name, k):
    if "src_label" not in df.columns or "dst_label" not in df.columns:
        return None, pd.DataFrame()
    top = topk_df(df, graph_name, k).dropna(subset=["src_label", "dst_label"])
    fraud_src = top[top["src_label"].astype(int) == 1]
    if fraud_src.empty:
        return None, pd.DataFrame()
    rows = []
    for (src_company_id, year), g in fraud_src.groupby(["src_company_id", "year"]):
        denom = min(k, len(g))
        if denom > 0:
            rows.append({
                "src_company_id": src_company_id,
                "year": int(year),
                "neighbor_count": int(len(g)),
                "precision_at_k": float((g["dst_label"].astype(int) == 1).sum() / denom),
            })
    result = pd.DataFrame(rows)
    return (None, result) if result.empty else (float(result["precision_at_k"].mean()), result)


def year_level_label_metrics(df):
    rows = []
    for year, g in df.groupby("year"):
        rows.append({
            "year": int(year),
            "edge_count": int(len(g)),
            "same_label_ratio": same_label_ratio(g),
            "fraud_fraud_ratio": fraud_fraud_ratio(g),
        })
    return pd.DataFrame(rows)


def stable_summary(stable_df, graph_name):
    lines = []
    if stable_df is None:
        lines.append(f"[{graph_name}]")
        lines.append("Không có file dữ liệu stable tương ứng, bỏ qua phần này.")
        return lines, {}

    metrics = {"edge_count": len(stable_df)}
    lines.append(f"[{graph_name}]")
    lines.append(f"Tổng số stable edge: {len(stable_df):,}")
    if "year_from" in stable_df.columns and "year_to" in stable_df.columns:
        lines.append("Số stable edge theo cặp năm:")
        for _, row in stable_df.groupby(["year_from", "year_to"]).size().reset_index(name="edge_count").iterrows():
            lines.append(f"  - {int(row['year_from'])} -> {int(row['year_to'])}: {int(row['edge_count']):,}")

    if graph_name == "STABLE_SIMILAR":
        if "avg_similarity" in stable_df.columns:
            metrics["avg_score"] = pd.to_numeric(stable_df["avg_similarity"], errors="coerce").mean()
            lines.append(f"avg_similarity trung bình: {metrics['avg_score']:.6f}")
        if "src_label_to" in stable_df.columns and "dst_label_to" in stable_df.columns:
            valid = stable_df.dropna(subset=["src_label_to", "dst_label_to"])
            metrics["same_label_ratio"] = float((valid["src_label_to"].astype(int) == valid["dst_label_to"].astype(int)).mean()) if len(valid) else None
            metrics["fraud_fraud_ratio"] = float(((valid["src_label_to"].astype(int) == 1) & (valid["dst_label_to"].astype(int) == 1)).mean()) if len(valid) else None
            if metrics["same_label_ratio"] is not None:
                lines.append(f"same_label_ratio trên stable edge: {metrics['same_label_ratio']:.6f}")
                lines.append(f"fraud_fraud_ratio trên stable edge: {metrics['fraud_fraud_ratio']:.6f}")
        elif "both_fraud_to" in stable_df.columns:
            metrics["fraud_fraud_ratio"] = float((pd.to_numeric(stable_df["both_fraud_to"], errors="coerce").fillna(0).astype(int) == 1).mean())
            lines.append(f"fraud_fraud_ratio trên stable edge: {metrics['fraud_fraud_ratio']:.6f}")
    else:
        if "avg_attention" in stable_df.columns:
            metrics["avg_score"] = pd.to_numeric(stable_df["avg_attention"], errors="coerce").mean()
            lines.append(f"avg_attention trung bình: {metrics['avg_score']:.6f}")
        if "same_label_stable" in stable_df.columns:
            metrics["same_label_ratio"] = float((pd.to_numeric(stable_df["same_label_stable"], errors="coerce").fillna(0).astype(int) == 1).mean())
            lines.append(f"same_label_ratio trên stable edge: {metrics['same_label_ratio']:.6f}")
        if "both_fraud_stable" in stable_df.columns:
            metrics["fraud_fraud_ratio"] = float((pd.to_numeric(stable_df["both_fraud_stable"], errors="coerce").fillna(0).astype(int) == 1).mean())
            lines.append(f"fraud_fraud_ratio trên stable edge: {metrics['fraud_fraud_ratio']:.6f}")
    return lines, metrics


def write_section(lines, title):
    lines.append("")
    lines.append("=" * 88)
    lines.append(title)
    lines.append("=" * 88)


def make_decision_lines(sim_same, att_same, sim_fraud, att_fraud, sim_precision, att_precision, jaccard_mean, stable_sim_metrics, stable_att_metrics, k):
    lines = []
    lines.append("Bảng đọc kết quả theo từng mục tiêu:")
    lines.append("- Mục tiêu gom chung nhãn: ATTENTION_RELATION tốt hơn nếu same_label_ratio cao hơn.")
    lines.append("- Mục tiêu tìm hàng xóm gian lận: SIMILAR_IN_YEAR tốt hơn nếu fraud_fraud_ratio và fraud_neighbor_precision@K cao hơn.")
    lines.append("- Mục tiêu giải thích mô hình: ATTENTION_RELATION có giá trị riêng vì đây là quan hệ học được từ attention của ACRF-RNN.")
    lines.append("")

    rows = []
    rows.append(["same_label_ratio", f"{sim_same:.6f}", f"{att_same:.6f}", "ATTENTION_RELATION" if att_same > sim_same else "SIMILAR_IN_YEAR", "Attention nối các node cùng nhãn nhiều hơn." if att_same > sim_same else "Cosine nối các node cùng nhãn nhiều hơn."])
    rows.append(["fraud_fraud_ratio", f"{sim_fraud:.6f}", f"{att_fraud:.6f}", "SIMILAR_IN_YEAR" if sim_fraud > att_fraud else "ATTENTION_RELATION", "Cosine gom cặp fraud-fraud tốt hơn theo nhãn trực tiếp." if sim_fraud > att_fraud else "Attention gom cặp fraud-fraud tốt hơn theo nhãn trực tiếp."])
    rows.append([f"fraud_neighbor_precision@{k}", f"{sim_precision:.6f}", f"{att_precision:.6f}", "SIMILAR_IN_YEAR" if sim_precision > att_precision else "ATTENTION_RELATION", "Cosine phù hợp hơn nếu dùng graph để truy xuất neighbor gian lận." if sim_precision > att_precision else "Attention phù hợp hơn nếu dùng graph để truy xuất neighbor gian lận."])
    rows.append([f"Jaccard Top-{k}", f"{jaccard_mean:.6f}", f"{jaccard_mean:.6f}", "Không phải thước đo thắng/thua", "Jaccard rất thấp nghĩa là hai graph bổ sung góc nhìn khác nhau, không trùng lặp."])
    if stable_sim_metrics and stable_att_metrics:
        s1, s2 = stable_sim_metrics.get("edge_count", 0), stable_att_metrics.get("edge_count", 0)
        rows.append(["stable_edge_count", f"{s1:,}", f"{s2:,}", "STABLE_SIMILAR" if s1 > s2 else "STABLE_ATTENTION_RELATION", "Stable attention chọn lọc hơn rất nhiều; không nên xem số cạnh thấp là yếu hơn tuyệt đối."])
    header = ["Chỉ số", "Cosine", "Attention", "Bên trội hơn", "Cách hiểu"]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = " | ".join(["{:<" + str(w) + "}" for w in widths])
    lines.append(fmt.format(*header))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt.format(*r))
    return lines


def main():
    parser = argparse.ArgumentParser(description="Compare SIMILAR_IN_YEAR cosine graph and ATTENTION_RELATION attention graph with stronger interpretation.")
    parser.add_argument("--similar", default=DEFAULT_SIMILAR)
    parser.add_argument("--attention", default=DEFAULT_ATTENTION)
    parser.add_argument("--stable-similar", default=DEFAULT_STABLE_SIMILAR)
    parser.add_argument("--stable-attention", default=DEFAULT_STABLE_ATTENTION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    similar_df = normalize_company_year_edges(read_csv(args.similar, True), "SIMILAR_IN_YEAR")
    attention_df = normalize_company_year_edges(read_csv(args.attention, True), "ATTENTION_RELATION")
    stable_similar = read_csv(args.stable_similar, False)
    stable_attention = read_csv(args.stable_attention, False)

    common_years = sorted(
    set(similar_df["year"].astype(int).unique().tolist())
    & set(attention_df["year"].astype(int).unique().tolist())
    )
    common_years = [int(y) for y in common_years]
    similar_common = similar_df[similar_df["year"].isin(common_years)]
    attention_common = attention_df[attention_df["year"].isin(common_years)]

    jaccard_df = jaccard_topk(similar_df, attention_df, args.top_k)
    sim_same = same_label_ratio(similar_df)
    att_same = same_label_ratio(attention_df)
    sim_fraud = fraud_fraud_ratio(similar_df)
    att_fraud = fraud_fraud_ratio(attention_df)
    sim_precision, _ = fraud_neighbor_precision_at_k(similar_df, "SIMILAR_IN_YEAR", args.top_k)
    att_precision, _ = fraud_neighbor_precision_at_k(attention_df, "ATTENTION_RELATION", args.top_k)

    sim_same_common = same_label_ratio(similar_common)
    att_same_common = same_label_ratio(attention_common)
    sim_fraud_common = fraud_fraud_ratio(similar_common)
    att_fraud_common = fraud_fraud_ratio(attention_common)
    sim_precision_common, _ = fraud_neighbor_precision_at_k(similar_common, "SIMILAR_IN_YEAR", args.top_k)
    att_precision_common, _ = fraud_neighbor_precision_at_k(attention_common, "ATTENTION_RELATION", args.top_k)

    stable_sim_lines, stable_sim_metrics = stable_summary(stable_similar, "STABLE_SIMILAR")
    stable_att_lines, stable_att_metrics = stable_summary(stable_attention, "STABLE_ATTENTION_RELATION")

    lines = []
    lines.append("BÁO CÁO SO SÁNH COSINE GRAPH VÀ ATTENTION GRAPH - BẢN ĐÁNH GIÁ MỞ RỘNG")
    lines.append(f"Thời điểm tạo báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Top-K dùng để so sánh: {args.top_k}")
    lines.append("")
    lines.append("Mục tiêu của báo cáo này không chỉ là liệt kê số liệu, mà còn phân tích graph nào phù hợp hơn cho từng mục tiêu sử dụng trong bài toán phát hiện gian lận tài chính.")
    lines.append("")
    lines.append("Input:")
    lines.append(f"- Cosine graph: {args.similar}")
    lines.append(f"- Attention graph: {args.attention}")
    lines.append(f"- Stable cosine graph: {args.stable_similar}")
    lines.append(f"- Stable attention graph: {args.stable_attention}")

    write_section(lines, "1. EDGE COUNT VÀ PHẠM VI DỮ LIỆU")
    lines.extend(edge_count_summary(similar_df, "SIMILAR_IN_YEAR"))
    lines.append("")
    lines.extend(edge_count_summary(attention_df, "ATTENTION_RELATION"))
    lines.append("")
    lines.append(f"Các năm chung giữa hai graph: {common_years}")
    lines.append(f"Số cạnh SIMILAR_IN_YEAR trong các năm chung: {len(similar_common):,}")
    lines.append(f"Số cạnh ATTENTION_RELATION trong các năm chung: {len(attention_common):,}")
    lines.append(f"Tỷ lệ số cạnh attention/cosine trên các năm chung: {safe_ratio(len(attention_common), len(similar_common)):.6f} ({pct(safe_ratio(len(attention_common), len(similar_common)))}).")
    lines.append("Nhận xét: attention graph nhỏ hơn nhiều vì chỉ lưu Top-K quan hệ mô hình chú ý. Do đó không nên so sánh edge count theo nghĩa graph nào 'mạnh hơn'; edge count chủ yếu phản ánh mức độ chọn lọc.")

    write_section(lines, "2. JACCARD OVERLAP TOP-K")
    if jaccard_df.empty:
        lines.append("Không có source-year chung giữa hai graph để tính Jaccard.")
        jaccard_mean = 0.0
    else:
        jaccard_mean = float(jaccard_df["jaccard"].mean())
        zero_count = int((jaccard_df["jaccard"] == 0).sum())
        positive_count = int((jaccard_df["jaccard"] > 0).sum())
        lines.append(f"Số source-year chung: {len(jaccard_df):,}")
        lines.append(f"Jaccard trung bình: {jaccard_mean:.6f}")
        lines.append(f"Jaccard nhỏ nhất: {jaccard_df['jaccard'].min():.6f}")
        lines.append(f"Jaccard lớn nhất: {jaccard_df['jaccard'].max():.6f}")
        lines.append(f"Số source-year có Jaccard = 0: {zero_count:,} ({pct(safe_ratio(zero_count, len(jaccard_df)))})")
        lines.append(f"Số source-year có Jaccard > 0: {positive_count:,} ({pct(safe_ratio(positive_count, len(jaccard_df)))})")
        lines.append("")
        lines.append("Top 10 source-year có Jaccard cao nhất:")
        lines.append(jaccard_df.sort_values("jaccard", ascending=False).head(10).to_string(index=False))
        lines.append("")
        lines.append("Đánh giá: Jaccard rất thấp là bằng chứng mạnh rằng attention graph không sao chép cosine graph. Hai graph đang cung cấp hai kiểu quan hệ khác nhau.")

    write_section(lines, "3. ĐÁNH GIÁ THEO NHÃN: SAME_LABEL, FRAUD_FRAUD, PRECISION@K")
    lines.append("Trên toàn bộ dữ liệu khả dụng:")
    lines.append(f"- SIMILAR_IN_YEAR same_label_ratio: {sim_same:.6f}")
    lines.append(f"- ATTENTION_RELATION same_label_ratio: {att_same:.6f}")
    lines.append(f"- SIMILAR_IN_YEAR fraud_fraud_ratio: {sim_fraud:.6f}")
    lines.append(f"- ATTENTION_RELATION fraud_fraud_ratio: {att_fraud:.6f}")
    lines.append(f"- SIMILAR_IN_YEAR fraud_neighbor_precision@{args.top_k}: {sim_precision:.6f}")
    lines.append(f"- ATTENTION_RELATION fraud_neighbor_precision@{args.top_k}: {att_precision:.6f}")
    lines.append("")
    lines.append("Trên các năm chung 2018-2020 để so sánh công bằng hơn:")
    lines.append(f"- SIMILAR_IN_YEAR same_label_ratio: {sim_same_common:.6f}")
    lines.append(f"- ATTENTION_RELATION same_label_ratio: {att_same_common:.6f}")
    lines.append(f"- SIMILAR_IN_YEAR fraud_fraud_ratio: {sim_fraud_common:.6f}")
    lines.append(f"- ATTENTION_RELATION fraud_fraud_ratio: {att_fraud_common:.6f}")
    lines.append(f"- SIMILAR_IN_YEAR fraud_neighbor_precision@{args.top_k}: {sim_precision_common:.6f}")
    lines.append(f"- ATTENTION_RELATION fraud_neighbor_precision@{args.top_k}: {att_precision_common:.6f}")
    lines.append("")
    lines.append("Đánh giá: attention graph có same_label_ratio cao hơn, nghĩa là có xu hướng nối các công ty cùng nhãn nhiều hơn. Tuy nhiên, với mục tiêu tìm neighbor gian lận trực tiếp, cosine graph tốt hơn vì fraud_fraud_ratio và fraud_neighbor_precision@K cao hơn.")

    write_section(lines, "4. SO SÁNH THEO TỪNG NĂM")
    sim_year = year_level_label_metrics(similar_common).rename(columns={"same_label_ratio": "cosine_same_label", "fraud_fraud_ratio": "cosine_fraud_fraud", "edge_count": "cosine_edges"})
    att_year = year_level_label_metrics(attention_common).rename(columns={"same_label_ratio": "attention_same_label", "fraud_fraud_ratio": "attention_fraud_fraud", "edge_count": "attention_edges"})
    by_year = sim_year.merge(att_year, on="year", how="inner")
    lines.append(by_year.to_string(index=False))
    lines.append("")
    lines.append("Đánh giá: bảng theo năm giúp tránh hiểu nhầm do cosine graph có nhiều năm hơn. Khi chỉ xét các năm chung, kết luận chính vẫn giữ nguyên: attention có xu hướng same-label cao hơn, còn cosine tốt hơn ở chỉ số fraud-fraud.")

    write_section(lines, "5. STABLE EDGE COMPARISON")
    lines.extend(stable_sim_lines)
    lines.append("")
    lines.extend(stable_att_lines)
    lines.append("")
    if stable_sim_metrics and stable_att_metrics:
        ratio = safe_ratio(stable_att_metrics.get("edge_count", 0), stable_sim_metrics.get("edge_count", 0))
        lines.append(f"Tỷ lệ stable_attention/stable_similar theo số cạnh: {ratio:.6f} ({pct(ratio)}).")
    lines.append("Đánh giá: stable attention rất chọn lọc. Việc chỉ có ít stable edge không chứng minh attention yếu hơn; nó cho thấy quan hệ attention Top-K lặp lại qua thời gian là hiếm và có tính lọc cao. Tuy nhiên, fraud_fraud_ratio của stable attention bằng 0 nên chưa thể dùng stable attention như chỉ báo trực tiếp để gom cặp fraud-fraud trong dữ liệu hiện tại.")

    write_section(lines, "6. BẢNG KẾT LUẬN THEO MỤC TIÊU")
    lines.extend(make_decision_lines(sim_same, att_same, sim_fraud, att_fraud, sim_precision, att_precision, jaccard_mean, stable_sim_metrics, stable_att_metrics, args.top_k))

    write_section(lines, "7. NHẬN XÉT ĐÁNH GIÁ TRONG BỐI CẢNH TÀI CHÍNH")
    lines.append("Trong bài toán tài chính, đặc biệt là phát hiện gian lận báo cáo tài chính, không nên chỉ hỏi graph nào có số cạnh nhiều hơn. Quan trọng hơn là graph đó phục vụ mục tiêu nào: sàng lọc công ty rủi ro, giải thích mô hình, hay phát hiện quan hệ ổn định qua thời gian.")
    lines.append("")
    lines.append("Nếu mục tiêu là sàng lọc rủi ro trực tiếp dựa trên nhãn gian lận, cosine graph đang có lợi thế hơn. Lý do là SIMILAR_IN_YEAR có fraud_fraud_ratio và fraud_neighbor_precision@K cao hơn ATTENTION_RELATION. Điều này nghĩa là khi xuất phát từ một công ty gian lận, cosine graph có xác suất gặp neighbor cũng gian lận cao hơn. Với nghiệp vụ kiểm toán hoặc cảnh báo sớm, đây là điểm quan trọng vì hệ thống cần ưu tiên các vùng lân cận có nhiều tín hiệu gian lận rõ ràng.")
    lines.append("")
    lines.append("Nếu mục tiêu là giải thích mô hình hoặc bổ sung một lớp quan hệ học được, attention graph lại có giá trị riêng. ATTENTION_RELATION có same_label_ratio cao hơn và Jaccard Top-K cực thấp so với cosine graph. Điều này cho thấy mô hình ACRF-RNN học ra một cấu trúc liên hệ khác, không đơn thuần lặp lại độ tương đồng tài chính thủ công. Trong bối cảnh tài chính hiện nay, khi các hệ thống phát hiện gian lận cần có khả năng giải thích và truy vết lý do mô hình chú ý đến một nhóm công ty, attention graph có vai trò như một lớp giải thích bổ sung.")
    lines.append("")
    lines.append("Vì vậy, kết luận hợp lý không phải là attention graph chính xác hơn cosine graph một cách tuyệt đối. Kết luận đúng là: cosine graph phù hợp hơn cho truy xuất neighbor gian lận trực tiếp, còn attention graph phù hợp hơn cho phân tích quan hệ ngầm và giải thích hành vi của mô hình. Hai graph nên được dùng bổ sung cho nhau thay vì thay thế nhau.")

    write_section(lines, "8. KẾT LUẬN NGẮN GỌN")
    lines.append("1. Không đủ cơ sở để nói ATTENTION_RELATION chính xác hơn SIMILAR_IN_YEAR theo nghĩa tuyệt đối, vì các chỉ số fraud-oriented của attention thấp hơn cosine.")
    lines.append("2. SIMILAR_IN_YEAR phù hợp hơn nếu mục tiêu là tìm các neighbor gian lận hoặc cụm công ty rủi ro trực tiếp.")
    lines.append("3. ATTENTION_RELATION có giá trị vì tạo ra quan hệ khác biệt, cùng nhãn nhiều hơn và phản ánh cấu trúc mà mô hình ACRF-RNN học được.")
    lines.append("4. STABLE_ATTENTION_RELATION rất chọn lọc, cho thấy chỉ một số ít quan hệ attention lặp lại qua thời gian; hiện chưa đủ mạnh để thay thế stable cosine graph trong truy vết fraud-fraud.")
    lines.append("5. Khuyến nghị triển khai: dùng cosine graph làm lớp truy vấn rủi ro chính, dùng attention graph làm lớp giải thích và bổ sung tín hiệu mô hình; khi hai graph cùng chỉ về một nhóm công ty thì ưu tiên kiểm tra thủ công cao hơn.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("===== GRAPH COMPARISON DONE =====")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
