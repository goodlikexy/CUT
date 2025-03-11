import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import matplotlib.pyplot as plt
import os
from os.path import join as opj
from sklearn.ensemble import IsolationForest
from scipy import stats

class StatisticalAnomalyDetector:
    """基于统计特征的异常检测器"""
    def __init__(self, threshold_std=3, window_size=5):
        self.threshold_std = threshold_std
        self.window_size = window_size
    
    def fit_detect(self, data):
        """
        使用统计方法检测异常
        Args:
            data: shape [T, N] 或 [T, N, D]
        Returns:
            anomaly_scores: 异常分数
            anomaly_labels: 异常标签
        """
        # 确保数据是2D的 [T, N]
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], -1)
        
        T, N = data.shape
        anomaly_scores = np.zeros((T, N))
        anomaly_labels = np.zeros((T, N))
        
        # 对每个变量分别计算
        for n in range(N):
            series = data[:, n]
            
            # 计算移动均值和标准差
            mean = np.mean(series)
            std = np.std(series)
            
            # 计算Z分数
            z_scores = np.abs((series - mean) / std)
            
            # 使用Z分数作为异常分数
            anomaly_scores[:, n] = z_scores
            
            # 标记异常点
            anomaly_labels[:, n] = (z_scores > self.threshold_std).astype(int)
            
        return anomaly_scores, anomaly_labels

class AmplitudeAnomalyDetector:
    """基于幅值变化的异常检测器"""
    def __init__(self, threshold_ratio=3.0, use_window=False, window_size=5):
        self.threshold_ratio = threshold_ratio
        self.use_window = use_window
        self.window_size = window_size
    
    def fit_detect(self, data):
        """
        使用幅值变化检测异常
        Args:
            data: shape [T, N] 或 [T, N, D]
        Returns:
            anomaly_scores: 异常分数
            anomaly_labels: 异常标签
        """
        # 确保数据是2D的 [T, N]
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], -1)
        
        T, N = data.shape
        anomaly_scores = np.zeros((T, N))
        anomaly_labels = np.zeros((T, N))
        
        # 对每个变量分别计算
        for n in range(N):
            series = data[:, n]  # 不取绝对值，保留正负号
            
            # 计算基准值（使用前20个点的均值和标准差）
            base_mean = np.mean(series[:20])
            base_std = np.std(series[:20])  # 添加标准差计算
            
            if self.use_window:
                # 使用滑动窗口平滑数据
                window_mean = np.convolve(series, 
                                        np.ones(self.window_size)/self.window_size, 
                                        mode='same')
                # 计算相对变化率作为异常分数
                relative_change = (window_mean - base_mean) / base_std
            else:
                # 直接计算每个点的相对变化
                relative_change = (series - base_mean) / base_std
            
            # 使用相对变化的绝对值作为异常分数
            anomaly_scores[:, n] = np.abs(relative_change)
            
            # 标记异常点（变化超过阈值，不管是变大还是变小）
            anomaly_labels[:, n] = (anomaly_scores[:, n] > self.threshold_ratio).astype(int)
        
        return anomaly_scores, anomaly_labels

def detect_anomalies(data, method="amplitude", **kwargs):
    """
    异常检测主函数
    Args:
        data: 输入数据
        method: 检测方法 ["statistical", "amplitude"]
        **kwargs: 其他参数
    """
    if method == "statistical":
        detector = StatisticalAnomalyDetector(**kwargs)
        return detector.fit_detect(data)
    
    elif method == "amplitude":
        detector = AmplitudeAnomalyDetector(**kwargs)
        return detector.fit_detect(data)
    
    else:
        raise ValueError(f"不支持的检测方法: {method}")

def plot_anomalies(data, anomaly_labels, save_path=None):
    """
    可视化每个变量的异常检测结果
    """
    T, N, D = data.shape
    fig, axes = plt.subplots(N, 1, figsize=(15, 4*N))
    if N == 1:
        axes = [axes]
    
    for n in range(N):
        ax = axes[n]
        time_series = data[:, n, 0]
        anomaly_points = anomaly_labels[:, n]
        
        # 绘制原始时间序列
        ax.plot(time_series, label='原始数据', alpha=0.7)
        
        # 标记异常点
        anomaly_idx = np.where(anomaly_points)[0]
        ax.scatter(anomaly_idx, time_series[anomaly_idx], 
                  color='red', label='异常点', marker='x', s=100)
        
        ax.set_title(f'变量 {n+1} 的异常检测结果')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close() 