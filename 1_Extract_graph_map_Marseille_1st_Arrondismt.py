import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置要提取的区域（马赛第1区）
# Set the area of interest (1st arrondissement of Marseille)
place_name = "1er arrondissement, Marseille, France"

# 下载该区域的道路网络（仅限可行驶车辆的道路）
# Download the street network (driveable roads only)
G = ox.graph_from_place(place_name, network_type="drive")

# 创建绘图画布
# Create the matplotlib figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制道路网络
# Plot the street network
ox.plot_graph(
    G,
    ax=ax,
    node_size=0,
    edge_color="black",
    edge_linewidth=0.5,
    show=False
)

# 设置标题（不调用 legend 避免警告）
# Set the title (omit legend to avoid warnings)
plt.title("Marseille 1st District - Road Network")

# 生成输出目录（项目目录/outputs/maps）
# Create output folder (project_dir/outputs/maps)
project_dir = Path(__file__).resolve().parent
out_dir = project_dir / "outputs" / "maps"
out_dir.mkdir(parents=True, exist_ok=True)

# 定义固定文件名（不带时间戳）
# Define fixed filenames (no timestamp)
img_path = out_dir / "marseille_1er_road_network.png"
graphml_path = out_dir / "marseille_1er_road_network.graphml"

# 保存地图为 PNG（300 DPI）
# Save the figure as PNG (300 DPI)
fig.savefig(img_path, dpi=300, bbox_inches="tight", facecolor="white")

# 同时保存道路网络为 GraphML 文件
# Save the road network as GraphML file
ox.save_graphml(G, filepath=graphml_path)

# 如需显示绘图结果，可取消下一行注释
# Uncomment the line below to show the plot:
plt.show()

# 关闭图像释放内存
# Close the figure to free memory
plt.close(fig)

# 打印保存路径和统计信息
# Print save paths and network stats
print(f"Map image saved at: {img_path}")
print(f"GraphML network saved at: {graphml_path}")
print(f"节点数 (Number of nodes): {len(G.nodes)}")
print(f"边数 (Number of edges): {len(G.edges)}")
