import numpy as np
import os
import pandas as pd
from datetime import datetime
import time

def find_root_cause(anomaly_labels, anomaly_scores, timestamps=None, feature_names=None):
    """
    根因分析：找出最早出现异常的变量节点
    
    参数:
    anomaly_labels: numpy.ndarray, 形状为 [n_samples, n_features], 异常标签 (0/1)
    anomaly_scores: numpy.ndarray, 形状为 [n_samples, n_features], 异常分数
    timestamps: numpy.ndarray, 形状为 [n_samples], 时间戳数组，如果为None则使用索引
    feature_names: list, 特征名称列表，如果为None则使用索引
    
    返回:
    dict: 包含根因分析结果的字典
    """
    # 参数检查
    if anomaly_labels.shape != anomaly_scores.shape:
        raise ValueError("异常标签和异常分数的形状必须相同")
    
    n_samples, n_features = anomaly_labels.shape
    
    # 如果没有提供时间戳，使用索引
    if timestamps is None:
        timestamps = np.arange(n_samples)
    
    # 如果没有提供特征名称，使用索引
    if feature_names is None:
        feature_names = [f"特征 {i}" for i in range(n_features)]
    
    # 找出第一个异常点
    first_anomaly_idx = None
    first_anomaly_time = float('inf')
    first_anomaly_features = []
    
    for t in range(n_samples):
        # 检查当前时间点是否有异常
        if np.any(anomaly_labels[t]):
            # 如果这是第一个发现的异常时间点
            if first_anomaly_idx is None or timestamps[t] < first_anomaly_time:
                first_anomaly_idx = t
                first_anomaly_time = timestamps[t]
                # 记录在这个时间点异常的特征
                first_anomaly_features = [(i, anomaly_scores[t, i]) 
                                         for i in range(n_features) 
                                         if anomaly_labels[t, i] == 1]
    
    if first_anomaly_idx is None:
        return {"error": "未发现异常"}
    
    # 按异常分数排序特征
    first_anomaly_features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # 提取根因特征
    root_causes = []
    for feature_idx, score in first_anomaly_features:
        root_causes.append({
            "feature_idx": feature_idx,
            "feature_name": feature_names[feature_idx],
            "score": score
        })
    
    # 确定主要根因（分数最高的特征）
    main_root_cause = root_causes[0]["feature_name"] if root_causes else "未知"
    
    # 格式化时间戳
    if isinstance(first_anomaly_time, (int, float)) and first_anomaly_time > 1000000000:
        # 可能是UNIX时间戳
        time_str = datetime.fromtimestamp(first_anomaly_time).strftime('%Y-%m-%d %H:%M:%S')
    else:
        time_str = str(first_anomaly_time)
    
    return {
        "first_anomaly_idx": first_anomaly_idx,
        "first_anomaly_time": first_anomaly_time,
        "time_str": time_str,
        "root_causes": root_causes,
        "main_root_cause": main_root_cause
    }

def format_root_cause_report(result):
    """
    格式化根因分析报告
    
    参数:
    result: dict, 根因分析结果
    
    返回:
    str: 格式化的报告
    """
    if "error" in result:
        return f"错误: {result['error']}"
    
    report = []
    report.append("=== 根因（最早异常时间点）===")
    report.append(f"时间点: {result['first_anomaly_idx']} (时间戳: {result['first_anomaly_time']})")
    
    # 添加异常特征
    for cause in result["root_causes"]:
        report.append(f"特征 {cause['feature_idx']} ({cause['feature_name']}): {cause['score']:.4f}")
    
    report.append("")
    report.append(f"根因指标为: {result['main_root_cause']}")
    
    return "\n".join(report)

def load_feature_mapping(mapping_file):
    """
    加载特征映射文件
    
    参数:
    mapping_file: str, 特征映射文件路径
    
    返回:
    list: 特征名称列表
    """
    try:
        with open(mapping_file, 'r') as f:
            # 假设每行格式为 "X0\t实际特征名"
            lines = f.readlines()
            feature_names = []
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    feature_names.append(parts[1])  # 使用实际特征名
                else:
                    feature_names.append(parts[0])  # 使用X0等标识符
        return feature_names
    except Exception as e:
        print(f"加载特征映射失败: {e}")
        return None

def analyze_root_cause(log_dir):
    """
    分析根因并生成报告
    
    参数:
    log_dir: str, 日志目录
    
    返回:
    str: 根因分析报告
    """
    # 加载数据
    anomaly_dir = os.path.join(log_dir, 'anomaly_detection')
    data_dir = os.path.join(log_dir, 'data')
    
    # 加载异常标签和分数
    try:
        anomaly_labels = np.load(os.path.join(anomaly_dir, 'anomaly_labels.npy'))
        anomaly_scores = np.load(os.path.join(anomaly_dir, 'anomaly_scores.npy'))
    except Exception as e:
        return f"加载异常数据失败: {e}"
    
    # 加载特征映射
    feature_mapping_path = os.path.join(data_dir, 'feature_mapping.txt')
    feature_names = load_feature_mapping(feature_mapping_path)
    
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(anomaly_scores.shape[1])]
    
    # 生成时间戳（如果没有实际时间戳，使用索引）
    timestamps = np.arange(anomaly_labels.shape[0])
    
    # 执行根因分析
    result = find_root_cause(anomaly_labels, anomaly_scores, timestamps, feature_names)
    
    # 格式化报告
    report = format_root_cause_report(result)
    
    # 保存报告
    report_path = os.path.join(anomaly_dir, 'root_cause_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"根因分析报告已保存到: {report_path}")
    
    return report

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
        report = analyze_root_cause(log_dir)
        print(report)
    else:
        print("请提供日志目录路径") 