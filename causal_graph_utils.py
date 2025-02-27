import os
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

class CausalGraphUtils:
    def load_feature_mapping(self, mapping_file):
        """
        加载特征映射文件
        Args:
            mapping_file: 特征映射文件路径
        Returns:
            dict: 特征映射字典
        """
        feature_mapping = {}
        try:
            with open(mapping_file, 'r') as f:
                for line in f:
                    x_name, original_name = line.strip().split('\t')
                    feature_mapping[x_name] = original_name
            print("加载特征映射成功:")
            for x_name, original_name in feature_mapping.items():
                print(f"{x_name} -> {original_name}")
        except Exception as e:
            print(f"加载特征映射失败: {e}")
            feature_mapping = None
        return feature_mapping

    def generate_causal_graph(self, causal_matrix, filename, output_dir=None, threshold=0.1, figsize=(10,8), 
                             positive_color='#2ecc71', negative_color='#e74c3c',
                             node_color='#3498db', show_labels=True,
                             title="Causal Graph with Absolute Threshold"):
        """
        生成优化后的一步时延因果图
        Args:
            causal_matrix: 因果矩阵
            filename: 输出文件名
            output_dir: 数据处理输出目录，包含feature_mapping.txt
            threshold: 阈值
            ...其他参数保持不变...
        """
        # 尝试加载特征映射
        if output_dir is None:
            output_dir = '/home/hz/projects/CUTS/UNN-main/CUTS/'
        
        mapping_file = os.path.join(output_dir, 'feature_mapping.txt')
        feature_mapping = self.load_feature_mapping(mapping_file)
        
        # 如果没有找到映射文件，使用默认的X0, X1等
        if feature_mapping is None:
            feature_mapping = {f'X{i}': f'X{i}' for i in range(len(causal_matrix))}
        
        # 参数校验
        assert all(len(row) == len(causal_matrix) for row in causal_matrix), "必须为方阵"
        assert all(causal_matrix[i][j] == 0 for i in range(len(causal_matrix)) 
                  for j in range(i+1, len(causal_matrix))), "必须为下三角矩阵"

        N = len(causal_matrix)
        G = nx.DiGraph()

        # 找出有效节点（有边相连的节点）
        active_nodes = set()
        for target in range(N):
            for source in range(N):
                weight = causal_matrix[target][source]
                if abs(weight) > threshold:
                    active_nodes.add(source)
                    active_nodes.add(target)

        # 使用原始特征名称创建节点
        t_minus_1_nodes = [f'{feature_mapping[f"X{i}"]}_t-1' for i in active_nodes]
        t_nodes = [f'{feature_mapping[f"X{i}"]}_t' for i in active_nodes]
        G.add_nodes_from(t_minus_1_nodes + t_nodes)

        # 添加边
        edges = []
        for target in active_nodes:
            for source in active_nodes:
                weight = causal_matrix[target][source]
                if abs(weight) > threshold:
                    weight = float(weight) if isinstance(weight, np.bool_) else weight
                    edges.append((
                        f'{feature_mapping[f"X{source}"]}_t-1',
                        f'{feature_mapping[f"X{target}"]}_t',
                        {'weight': round(weight, 3)}
                    ))
        G.add_edges_from(edges)

        # 优化节点布局
        active_list = sorted(list(active_nodes))
        pos = {
            **{f'{feature_mapping[f"X{i}"]}_t-1': (0, len(active_list)-1-active_list.index(i)) 
               for i in active_nodes},
            **{f'{feature_mapping[f"X{i}"]}_t': (2, len(active_list)-1-active_list.index(i)) 
               for i in active_nodes}
        }

        # 绘图
        plt.figure(figsize=figsize, dpi=100)
        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=1000,  # 增大节点大小以适应更长的文字
            node_color=[node_color] * (len(active_nodes) * 2),
            edgecolors='white',
            linewidths=1
        )
        
        edge_data = G.edges(data=True)
        edge_colors = []
        for u, v, d in edge_data:
            weight = d['weight']
            edge_colors.append(positive_color if weight > 0 else negative_color)
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_data,
            edge_color=edge_colors,
            width=1.5,
            arrowsize=10,
            alpha=0.7,
            min_source_margin=20,
            min_target_margin=20
        )
        
        # 使用更小的字体以适应更长的文字
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='normal')
        if show_labels:
            edge_labels = {(u, v): f'{d["weight"]:+.2f}' for u, v, d in edge_data}
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=8,
                label_pos=0.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        plt.title(title + f"\n(Threshold: |weight| > {threshold})", fontsize=12, pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_dir + filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return G, plt.gcf()

    def make_lower_triangular(self, matrix):
        """
        将输入矩阵转换为下三角矩阵，保留原本下三角数据，上三角数据置0
        
        参数：
        matrix : numpy.ndarray 或 torch.Tensor 输入矩阵
        
        返回：
        lower_triangular_matrix : 与输入相同类型的下三角矩阵
        """
        # 复制输入矩阵以避免修改原始数据
        if isinstance(matrix, torch.Tensor):
            result = matrix.clone()
            # 将上三角部分（不包括对角线）置为0
            mask = torch.triu(torch.ones_like(result), diagonal=1)
            result = result * (1 - mask)
        else:
            result = np.copy(matrix)
            # 将上三角部分（不包括对角线）置为0
            result[np.triu_indices(result.shape[0], k=1)] = 0
            
        return result 