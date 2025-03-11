import numpy as np
from causal_graph_utils import CausalGraphUtils
import os
import matplotlib.pyplot as plt

# 创建 CausalGraphUtils 实例
graph_utils = CausalGraphUtils()

def generate_causal_graphs(log_dir):
    # 从data子文件夹读取特征映射文件
    feature_mapping_path = os.path.join(log_dir, 'data', 'feature_mapping.txt')
    try:
        with open(feature_mapping_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"成功加载特征映射: {feature_mapping_path}")
    except Exception as e:
        print(f"加载特征映射失败: {str(e)}")
        feature_names = None
    
    # 从data目录读取数据
    data_dir = os.path.join(log_dir, 'data')
    coef_path = os.path.join(data_dir, 'discovered_graph_coef.npz')
    thres_path = os.path.join(data_dir, 'thresholded_graph.npz')
    
    coef_data = np.load(coef_path)
    thresholded_data = np.load(thres_path)
    
    # 获取矩阵
    coef_matrix = coef_data['coef_matrix']
    thresholded_matrix = thresholded_data['thresholded_matrix']

    # 将矩阵转换为下三角矩阵
    lower_triangular_coef = graph_utils.make_lower_triangular(coef_matrix)

    # 生成因果图并保存到日志目录
    graph_utils.generate_causal_graph(
        causal_matrix=coef_matrix,
        filename="weighted_causal_graph.png",
        output_dir=log_dir,  # 直接保存到日志主目录
        threshold=0.5
    )
    plt.close()