import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import random  # 添加到文件开头的导入部分
import networkx as nx
import matplotlib.pyplot as plt

class AnomalyDataProcessor:
    def __init__(self, data_file, label_file, output_dir):
        """
        初始化数据处理器
        Args:
            data_file: 原始数据文件路径
            label_file: 标签文件路径
            output_dir: 输出目录
        """
        self.data_df = pd.read_csv(data_file)
        self.label_df = pd.read_csv(label_file)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def remove_multihop_links(self, multihop_prefixes):
        """
        去除多跳链路
        Args:
            multihop_prefixes: 需要移除的多跳链路前缀列表
        """
        filtered_cols = [col for col in self.data_df.columns 
                        if not any(col.startswith(prefix) for prefix in multihop_prefixes)]
        self.data_df = self.data_df[filtered_cols]
        self.label_df = self.label_df[filtered_cols]
        
        print(f"移除多跳链路后剩余特征数: {len(filtered_cols)}")
        return filtered_cols

    def detect_anomaly_segments(self, min_consecutive=15):
        """
        检测连续异常片段
        Args:
            min_consecutive: 最小连续异常数
        Returns:
            list of tuples: [(start_idx, end_idx, affected_metrics)]
        """
        anomaly_segments = []
        
        for col in self.label_df.columns:
            if col == 'Time':
                continue
                
            count = 0
            start_idx = -1
            
            for i in range(len(self.label_df)):
                if self.label_df[col][i] == 1:
                    if count == 0:
                        start_idx = i
                    count += 1
                else:
                    if count >= min_consecutive:
                        anomaly_segments.append((start_idx, i-1, col))
                    count = 0
                    
            if count >= min_consecutive:
                anomaly_segments.append((start_idx, len(self.label_df)-1, col))
        
        # 按开始时间排序
        anomaly_segments.sort(key=lambda x: x[0])
        
        print(f"检测到 {len(anomaly_segments)} 个异常片段")
        for start, end, metric in anomaly_segments:
            print(f"指标 {metric}: {start}-{end} (长度: {end-start+1})")
            
        return anomaly_segments

    def extract_root_cause_segment(self, segment_idx, anomaly_segments, padding=20):
        """
        提取根因分析片段（前padding + 异常片段 + 后padding）
        Args:
            segment_idx: 要处理的异常片段索引
            anomaly_segments: 异常片段列表
            padding: 前后填充长度
        """
        if segment_idx >= len(anomaly_segments):
            raise ValueError("无效的片段索引")
            
        start, end, current_metric = anomaly_segments[segment_idx]
        
        # 计算扩展后的范围
        padded_start = max(0, start - padding)
        padded_end = min(len(self.data_df) - 1, end + padding)
        
        # 提取数据和对应的标签
        segment_data = self.data_df.iloc[padded_start:padded_end+1]
        segment_labels = self.label_df.iloc[padded_start:padded_end+1]
        
        # 找出在这个时间段内所有出现异常的指标
        anomaly_cols = set()
        for s, e, metric in anomaly_segments:
            # 检查是否与当前片段有重叠
            if not (e < padded_start or s > padded_end):
                anomaly_cols.add(metric)
        
        # 转换为列表并排序，确保结果可重现
        anomaly_cols = sorted(list(anomaly_cols))
        
        print(f"\n选中的异常片段: {start}-{end} ({end-start+1} 个点)")
        print(f"扩展后的范围: {padded_start}-{padded_end} ({padded_end-padded_start+1} 个点)")
        print(f"涉及的异常指标数量: {len(anomaly_cols)}")
        print("异常指标列表:")
        for col in anomaly_cols:
            print(f"- {col}")
        
        # 创建特征名称映射
        feature_mapping = {f'X{i}': col for i, col in enumerate(anomaly_cols)}
        print("\n特征映射关系:")
        for x_name, original_name in feature_mapping.items():
            print(f"{x_name} -> {original_name}")
        
        # 保存特征映射到文件
        mapping_file = os.path.join(self.output_dir, 'feature_mapping.txt')
        with open(mapping_file, 'w') as f:
            for x_name, original_name in feature_mapping.items():
                f.write(f"{x_name}\t{original_name}\n")
        
        # 只保留异常标签的列
        segment_data = segment_data[['Time'] + anomaly_cols]
        segment_labels = segment_labels[anomaly_cols]
        
        return segment_data, segment_labels, (padded_start, padded_end)

    def extract_normal_segments(self, selected_cols, window_size=50):
        """
        提取正常数据段
        Args:
            selected_cols: 需要考虑的列
            window_size: 窗口大小
        """
        # 找出所有列都为0的位置
        all_normal = (self.label_df[selected_cols] == 0).all(axis=1)
        
        normal_segments = []
        count = 0
        start_idx = -1
        
        for i in range(len(all_normal)):
            if all_normal[i]:
                if count == 0:
                    start_idx = i
                count += 1
            else:
                if count >= window_size:
                    normal_segments.append((start_idx, i-1))
                count = 0
                
        if count >= window_size:
            normal_segments.append((start_idx, len(all_normal)-1))
            
        return normal_segments

    def create_npy_files(self, root_cause_data, normal_segments, selected_cols, window_size=50):
        """
        创建NPY文件和特征映射文件，同时保存根因时间戳
        """
        # 只使用实际存在于root_cause_data中的列，并分开处理Time列
        available_cols = [col for col in root_cause_data.columns if col != 'Time']
        timestamps = root_cause_data['Time'].values  # 保存根因片段的时间戳
        
        # 创建特征映射并保存
        feature_mapping = {f'X{i}': col for i, col in enumerate(available_cols)}
        mapping_file = os.path.join(self.output_dir, 'feature_mapping.txt')
        with open(mapping_file, 'w') as f:
            for x_name, original_name in feature_mapping.items():
                f.write(f"{x_name}\t{original_name}\n")
        
        print("\n特征映射关系:")
        for x_name, original_name in feature_mapping.items():
            print(f"{x_name} -> {original_name}")
        print(f"特征映射已保存到: {mapping_file}")
        
        # 处理根因片段
        root_cause_array = root_cause_data[available_cols].values
        root_cause_array = root_cause_array.reshape(-1, len(available_cols), 1)  # (T, 特征数量, 1)
        
        # 处理正常片段
        normal_windows = []
        normal_lengths = []  # 用于存储每个正常段的长度
        for start, end in normal_segments:
            segment_length = end - start + 1  # 计算正常段的长度
            normal_lengths.append(segment_length)  # 存储长度
            
            num_windows = segment_length // window_size
            for i in range(num_windows):
                window_start = start + i * window_size
                window_end = window_start + window_size
                window_data = self.data_df.iloc[window_start:window_end][available_cols].values
                normal_windows.append(window_data.reshape(window_size, -1, 1))  # (window_size, 特征数量, 1)

        normal_array = np.vstack(normal_windows)  # (n*T, 特征数量, 1)

        # 保存为npy文件
        np.save(f"{self.output_dir}/root_cause_segment.npy", root_cause_array)  # 形状为 (T, 特征数量, 1)
        np.save(f"{self.output_dir}/normal_segments.npy", normal_array)  # 形状为 (n*T, 特征数量, 1)
        np.save(f"{self.output_dir}/root_cause_timestamps.npy", timestamps)  # 保存根因片段的时间戳
        
        print(f"\n使用的特征列: {available_cols}")
        print(f"根因片段数据已保存，形状: {root_cause_array.shape}")
        print(f"根因片段时间戳已保存，形状: {timestamps.shape}")
        print(f"正常片段数据已保存，形状: {normal_array.shape}")
        
        # 输出正常段数量和每个正常段的长度
        print(f"\n检测到的正常段数量: {len(normal_segments)}")
        print("每个正常段的长度:", normal_lengths)
        
        # 验证保存的数据
        loaded_root_cause = np.load(f"{self.output_dir}/root_cause_segment.npy")
        loaded_normal = np.load(f"{self.output_dir}/normal_segments.npy")
        loaded_timestamps = np.load(f"{self.output_dir}/root_cause_timestamps.npy")
        
        print("\n验证加载的数据形状:")
        print(f"加载的根因片段形状: {loaded_root_cause.shape}")
        print(f"加载的正常片段形状: {loaded_normal.shape}")
        print(f"加载的根因时间戳形状: {loaded_timestamps.shape}")
        
        return feature_mapping

def main():
    # 设置路径
    data_file = '/home/hz/projects/CUTS/UNN-main/CUTS/data_10_26/test_d/data/merged_output_normalized_2.csv'
    output_dir = f'/home/hz/projects/CUTS/UNN-main/CUTS/data_10_26/test_d/data_processed'
    label_file = '/home/hz/projects/CUTS/UNN-main/CUTS/data_10_26/10_26_anomaly_res/duodian/label_d_2.csv'
    
    # 初始化处理器
    processor = AnomalyDataProcessor(data_file, label_file, output_dir)
    
    # 去除多跳链路
    multihop_prefixes = ['F2S1', 'F2F4', 'F2C1', 'C1S1', 'S1C1', 'C1F2', 'F4F2', 'S1F2']
    remaining_cols = processor.remove_multihop_links(multihop_prefixes)
    
    # 检测异常片段
    anomaly_segments = processor.detect_anomaly_segments(min_consecutive=15)
    
    if not anomaly_segments:
        raise ValueError("没有找到异常片段")
    
    # 随机选择一个异常片段
    random_idx = random.randint(0, len(anomaly_segments) - 1)
    print(f"\n总共找到 {len(anomaly_segments)} 个异常片段")
    print(f"随机选择第 {random_idx + 1} 个异常片段")
    
    # 提取根因片段
    root_cause_data, root_cause_labels, (start, end) = processor.extract_root_cause_segment(
        segment_idx=random_idx,  # 使用随机选择的索引
        anomaly_segments=anomaly_segments,
        padding=20
    )
    
    # 提取正常片段
    selected_cols = [col for col in remaining_cols if col != 'Time']
    normal_segments = processor.extract_normal_segments(selected_cols, window_size=50)
    
    # 创建NPY文件
    processor.create_npy_files(root_cause_data, normal_segments, selected_cols, window_size=50)

if __name__ == "__main__":
    main()