# 项目说明 / Project Description

按顺序操作，足以复现这个项目。  
1 提取道路地图网格  
2 将道路网格中间均匀插入格点后，返回所有格点的坐标和序列号。  
3 生成随机格点Json文件。  
4 自定义尺寸大小，生成gexf图。  
5 自定义尺寸大小，将格点在地图上标出。

Follow the steps in order to fully reproduce this project.  
1 Extract the road map grid.  
2 After uniformly inserting grid points into the road grid, return the coordinates and serial numbers (IDs) of all grid points.  
3 Generate a JSON file of random grid points.  
4 Generate a GEXF graph with custom dimensions.  
5 Mark the grid points on the map with custom dimensions.

---

## 调取已生成图的方法 / How to Load the Generated Graph

```python
import networkx as nx

# 读取已保存的 GEXF 图
# Load the saved GEXF graph
G_tsp = nx.read_gexf("outputs/data/tsp_graph_1_solution.gexf")

# 检查节点和边
# Check the number of nodes and edges
print("节点数量:", len(G_tsp.nodes))  # Number of nodes
print("边数量:", len(G_tsp.edges))    # Number of edges
