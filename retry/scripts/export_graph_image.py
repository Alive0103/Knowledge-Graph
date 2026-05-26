"""
导出全量知识图谱可视化图片（快速版）
使用谱布局 (spectral) + 少量力导向微调，大幅加速。
"""

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw

from alignment.dbp15k import DBP15KDataset


def main():
    data_dir = Path(__file__).resolve().parent.parent / "recovered" / "alignment" / "DBP15K" / "zh_en"
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "knowledge_graph_full.png"

    print(f"Loading dataset from {data_dir} ...")
    dataset = DBP15KDataset(data_dir)

    # --- 构建图 ---
    print("Building graph ...")
    G = nx.Graph()
    for kg in ("1", "2"):
        for eid in dataset.entities[kg]:
            G.add_node(f"{kg}:{eid}", kg=kg)
    for kg in ("1", "2"):
        for t in dataset.triples[kg]:
            G.add_edge(f"{kg}:{t.head_id}", f"{kg}:{t.tail_id}")

    # 加入对齐边，把中英文侧连接起来
    for split in ("test", "valid", "ref_ent_ids"):
        for pair in dataset.alignments.get(split, []):
            G.add_edge(f"1:{pair.left_id}", f"2:{pair.right_id}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # --- 布局：只处理最大连通分量，其余随机放置 ---
    print("Computing layout ...")
    t0 = time.time()

    # 取最大连通分量
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    largest_cc = components[0]
    print(f"  Largest component: {len(largest_cc)} nodes, total components: {len(components)}")

    # 对最大连通分量用 spring_layout，iterations 少（10次足够出结构）
    sub = G.subgraph(largest_cc)
    pos = nx.spring_layout(sub, k=2.0 / math.sqrt(len(largest_cc)), iterations=25, seed=42)

    # 其余小分量放在边缘
    rng = np.random.RandomState(42)
    for comp in components[1:]:
        cx = rng.uniform(-1.2, 1.2)
        cy = rng.uniform(-1.2, 1.2)
        for i, n in enumerate(comp):
            angle = 2 * math.pi * i / max(len(comp), 1)
            r = 0.01 * math.sqrt(len(comp))
            pos[n] = np.array([cx + r * math.cos(angle), cy + r * math.sin(angle)])

    print(f"  Layout done in {time.time() - t0:.1f}s")

    # --- 渲染 ---
    IMG_SIZE = 8192
    PADDING = 300
    DRAW_SIZE = IMG_SIZE - 2 * PADDING

    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    def to_px(x, y):
        px = int(PADDING + (x - x_min) / x_range * DRAW_SIZE)
        py = int(PADDING + (y - y_min) / y_range * DRAW_SIZE)
        return px, py

    degrees = dict(G.degree())
    deg_vals = np.array(list(degrees.values()), dtype=float)
    deg_min, deg_max = deg_vals.min(), deg_vals.max()
    deg_range = deg_max - deg_min or 1
    print(f"  Degree range: {int(deg_min)} - {int(deg_max)}")

    print(f"Rendering {IMG_SIZE}x{IMG_SIZE} image ...")
    t0 = time.time()

    # 边层 (RGBA)
    img = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (255, 255, 255, 255))
    edge_layer = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge_layer)

    total_edges = G.number_of_edges()
    for i, (u, v) in enumerate(G.edges()):
        if i % 20000 == 0:
            print(f"  Edges: {i}/{total_edges}")
        x1, y1 = to_px(*pos[u])
        x2, y2 = to_px(*pos[v])
        edge_draw.line([(x1, y1), (x2, y2)], fill=(160, 160, 160, 20), width=1)

    img = Image.alpha_composite(img, edge_layer)
    del edge_layer, edge_draw

    # 节点层
    node_draw = ImageDraw.Draw(img)
    COLOR_ZH = (220, 60, 60)
    COLOR_EN = (50, 100, 220)

    # 先画小节点，大节点覆盖在上面
    sorted_nodes = sorted(G.nodes(), key=lambda n: degrees[n])
    for n in sorted_nodes:
        x, y = to_px(*pos[n])
        d = degrees[n]
        norm = math.sqrt((d - deg_min) / deg_range) if deg_range > 0 else 0.5
        radius = int(2 + norm * 16)
        color = COLOR_ZH if G.nodes[n]["kg"] == "1" else COLOR_EN
        node_draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color,
        )

    print(f"  Render done in {time.time() - t0:.1f}s")

    final = img.convert("RGB")
    final.save(str(output_path), quality=95)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
