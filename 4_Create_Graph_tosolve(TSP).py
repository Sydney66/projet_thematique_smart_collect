# -*- coding: utf-8 -*-
"""
只加载第一组解并构建 TSP 加权无向图（不做任何后续评估/求解）。
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

# -----------------------
# 路径配置（按需修改）
# -----------------------
GRAPHML_DENSIFIED = "outputs/maps/marseille_1er_densified_network.graphml"
GRAPHML_RAW = "outputs/maps/marseille_1er_network.graphml"
SOL_JSON = "outputs/data/random_solutions_prepared.json"
# 若你的CSV名不固定，就用通配；若固定请填死
CSV_PATTERN = "outputs/data/marseille_1er_graph_nodes_projected.csv"

# -----------------------
# 工具函数
# -----------------------
import os
import networkx as nx
import osmnx as ox

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return x

def _safe_int(x):
    try:
        # 允许 "4.0" 这类字符串转成 4
        return int(float(x))
    except Exception:
        return x

def pick_graph(graphml_main: str, graphml_fallback: str) -> nx.Graph:
    """优先加载插值后的图，失败再用原始图；包含 dtype 兜底修复。"""
    path = graphml_main if os.path.exists(graphml_main) else graphml_fallback
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图文件：\n- {graphml_main}\n- {graphml_fallback}")

    try:
        G = ox.load_graphml(path)  # 首选：让 OSMnx 处理它自己的序列化格式
        return G.to_undirected()
    except ValueError as e:
        print(f"[warn] ox.load_graphml 解析失败，将回退到 networkx.read_graphml 并进行 dtype 修复：{e}")

    # 回退：直接用 networkx 读取，不做类型强制，然后手动清洗常见字段
    G = nx.read_graphml(path)

    # 规范 node 属性的数值类型（按需添加你项目的自定义字段）
    node_float_keys = ["x", "y", "lon", "lat"]
    node_int_keys_maybe = ["street_count"]  # 常见触发点：“4.0”→int失败
    # 如果你给节点添加了自定义字段（例如 candidate_id 等），也可以在此自定义修复：
    # node_int_keys_maybe += ["candidate_id", "candidate_id_new"]

    for _, data in G.nodes(data=True):
        for k in node_float_keys:
            if k in data:
                data[k] = _safe_float(data[k])
        for k in node_int_keys_maybe:
            if k in data:
                data[k] = _safe_int(data[k])

    # 规范 edge 的长度字段
    edge_len_keys = ["length", "length_m", "distance", "weight"]
    for _, _, _, data in G.edges(keys=True, data=True):
        for k in edge_len_keys:
            if k in data:
                data[k] = _safe_float(data[k])

    return G.to_undirected()


import os, json

def load_solution_indices(json_path: str, n: int = 1) -> list:
    """
    读取前 n 组解（默认第0组），合并为一个去重后的候选点编号列表（保持原顺序）。
    :param json_path: JSON 文件路径
    :param n: 要合并的解组数量（从第0组开始）
    :return: 合并并去重的编号列表
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到解文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        sols = json.load(f)

    if not sols:
        raise ValueError("解列表为空")

    if n > len(sols):
        n = len(sols)  # 防止超出范围

    def extract_indices(solution):
        if isinstance(solution, dict):
            for key in ("solution", "indices", "nodes", "ids"):
                if key in solution:
                    return list(solution[key])
            flat = []
            for v in solution.values():
                if isinstance(v, (list, tuple, set)):
                    flat.extend(v)
            if flat:
                return flat
            raise ValueError("无法提取编号")
        return list(solution)

    # 合并前 n 组解
    combined = []
    for i in range(n):
        combined.extend(extract_indices(sols[i]))

    # 去重但保持原顺序
    seen = set()
    unique_combined = [x for x in combined if not (x in seen or seen.add(x))]

    return unique_combined

def load_candidates_df(csv_pattern: str) -> pd.DataFrame:
    """读取 output/data 下的一张候选表；如多张，取第一张匹配含关键列的。"""
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        raise FileNotFoundError(f"未在 {csv_pattern} 匹配到任何 CSV")
    # 依次尝试，挑出包含关键列的一张
    for p in csv_files:
        df = pd.read_csv(p)
        cols = set(c.lower() for c in df.columns)
        needed_any_id = {"candidate_id", "candidate_id_new", "id", "index"}
        if cols & needed_any_id:
            return df
    # 没有发现明显的编号列也先返回第一张，后面抛错提示列名
    return pd.read_csv(csv_files[0])

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """规范列名，返回含: id_col, lon, lat, x, y, osmid（若有）"""
    # 找编号列
    id_candidates = ["node_index"]
    id_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        # 尝试忽略大小写
        low = {c.lower(): c for c in df.columns}
        for c in id_candidates:
            if c in low:
                id_col = low[c]
                break
    if id_col is None:
        raise KeyError(f"未找到编号列，期望之一: {id_candidates}")

    # 经纬度/x/y 列
    lon_col = next((c for c in ["lon", "longitude", "x"] if c in df.columns), None)
    lat_col = next((c for c in ["lat", "latitude", "y"] if c in df.columns), None)

    # 若有 x/y 与 lon/lat 并存，优先用 lon/lat，x/y 作为备份
    x_col = next((c for c in ["x", "lon", "longitude"] if c in df.columns), None)
    y_col = next((c for c in ["y", "lat", "latitude"] if c in df.columns), None)

    # osmid 列可选
    osmid_col = next((c for c in ["osmid", "osmid_node", "node", "node_id"] if c in df.columns), None)

    return df.rename(columns={
        id_col: "cid",
        lon_col or "": "lon",
        lat_col or "": "lat",
        x_col or "": "x",
        y_col or "": "y",
        osmid_col or "": "osmid",
    })

def filter_subset(df: pd.DataFrame, indices: list) -> pd.DataFrame:
    """按照第一组解的编号过滤，并以编号排序。"""
    if "cid" not in df.columns:
        raise KeyError("内部错误：缺少标准化后的列 cid")
    sub = df[df["cid"].isin(indices)].copy()
    if sub.empty:
        raise ValueError("根据第一组解的编号在CSV里没有匹配到任何行")
    sub.sort_values(by="cid", inplace=True)
    sub.reset_index(drop=True, inplace=True)
    return sub

def map_points_to_nodes(G: nx.Graph, sub: pd.DataFrame) -> np.ndarray:
    """把点映射到路网节点：
       1) 若有 osmid 且存在于图中，直接用；
       2) 否则优先用经纬度 lon/lat 最近节点；
       3) 若无经纬度则尝试 x/y（OSMnx存的是 x=lon, y=lat）。"""
    # 情况1：osmid
    if "osmid" in sub.columns and sub["osmid"].notna().any():
        node_ids = []
        nodes_set = set(G.nodes)
        for v in sub["osmid"].tolist():
            vv = v
            # 尝试把 '123,456' 这种列表字符串解析为第一个，或把浮点转整
            if isinstance(v, str) and "," in v:
                vv = v.split(",")[0].strip()
            try:
                # osmid 可能是numpy类型/浮点/字符串
                if isinstance(vv, float) and vv.is_integer():
                    vv = int(vv)
                else:
                    vv = int(vv)
            except Exception:
                # 保持原值（有些图的osmid为字符串）
                pass
            if vv in nodes_set:
                node_ids.append(vv)
            else:
                # 回退到最近节点
                node_ids = None
                break
        if node_ids is not None:
            return np.array(node_ids)

    # 情况2：lon/lat
    if {"lon", "lat"}.issubset(sub.columns) and sub["lon"].notna().all() and sub["lat"].notna().all():
        return ox.distance.nearest_nodes(G, X=sub["lon"].to_numpy(), Y=sub["lat"].to_numpy())

    # 情况3：x/y -> 作为经纬度使用
    if {"x", "y"}.issubset(sub.columns) and sub["x"].notna().all() and sub["y"].notna().all():
        return ox.distance.nearest_nodes(G, X=sub["x"].to_numpy(), Y=sub["y"].to_numpy())

    raise ValueError("无法将点映射到路网节点：缺少 osmid 或 lon/lat 或 x/y 信息")

def edge_weight_key(G: nx.Graph) -> str:
    """判断边的距离属性键：优先 'length'（米），否则尝试 'length_m' 或其他"""
    # 随便取一条边看属性
    for u, v, data in G.edges(data=True):
        for k in ("length", "length_m", "distance", "weight"):
            if k in data:
                return k
        break
    # 若没找到，默认 'length'
    return "length"

def build_distance_matrix(G: nx.Graph, node_ids: np.ndarray, wkey: str) -> np.ndarray:
    """基于路网最短路构建距离矩阵。"""
    n = len(node_ids)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                d = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight=wkey)
                dist[i, j] = float(d)
            except nx.NetworkXNoPath:
                dist[i, j] = np.inf
    return dist

def build_tsp_graph(dist: np.ndarray, sub: pd.DataFrame) -> nx.Graph:
    """把距离矩阵转成 TSP 用的加权无向图；节点携带原始属性便于后处理/可视化。"""
    Gt = nx.Graph()
    n = dist.shape[0]
    # 节点属性
    for i in range(n):
        attrs = {"candidate_id": int(sub.loc[i, "cid"])}
        for k in ("lon", "lat", "x", "y", "osmid"):
            if k in sub.columns:
                val = sub.loc[i, k]
                try:
                    attrs[k] = float(val) if k in ("lon", "lat", "x", "y") else val
                except Exception:
                    attrs[k] = val
        Gt.add_node(i, **attrs)
    # 边权
    for i in range(n):
        for j in range(i + 1, n):
            w = dist[i, j]
            if np.isfinite(w):
                Gt.add_edge(i, j, weight=float(w))
            else:
                # 如需强制完全图，可改为:
                # Gt.add_edge(i, j, weight=1e12)
                pass
    return Gt

# -----------------------
# 主流程
# -----------------------
if __name__ == "__main__":
    size = 10
    # 1) 路网
    G = pick_graph(GRAPHML_DENSIFIED, GRAPHML_RAW)
    wkey = edge_weight_key(G)
    print(f"Loaded graph with |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, weight_key='{wkey}'")

    # 2) 第一组解编号
    indices = load_solution_indices(SOL_JSON,size)
    print(f"First solution size: {len(indices)}")

    # 3) 候选点表（自动识别列名）
    df_raw = load_candidates_df(CSV_PATTERN)
    df = normalize_columns(df_raw)

    # 4) 过滤本次用到的点，并映射到路网节点
    sub = filter_subset(df, indices)
    node_ids = map_points_to_nodes(G, sub)
    print(f"Mapped {len(node_ids)} points to graph nodes.")

    # 5) 路网最短路距离矩阵
    dist = build_distance_matrix(G, node_ids, wkey=wkey)
    print("Distance matrix shape:", dist.shape,
          "| finite ratio:", np.isfinite(dist).mean())

    # 6) 构建 TSP 图（仅构建）
    G_tsp = build_tsp_graph(dist, sub)
    print(f"TSP graph built: |V|={G_tsp.number_of_nodes()}, |E|={G_tsp.number_of_edges()} (finite edges only)")

    # 可选：保存
    nx.write_gexf(G_tsp, f"outputs/data/tsp_graph_{size}_solution.gexf")

