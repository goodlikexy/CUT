import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join as opj
import argparse

def process_anomaly_detection(data_path, outdir, method, **kwargs):
    """处理异常检测结果并生成可视化和CSV文件"""
    os.makedirs(outdir, exist_ok=True)

    try:
        # 加载原始异常检测数据
        if os.path.exists(data_path):
            original_data = np.load(data_path)
            # 确保数据是二维的 [时间点, 变量]
            if len(original_data.shape) == 3:
                original_data = original_data.reshape(original_data.shape[0], -1)
            
            # 创建原始数据的DataFrame
            time_points = range(len(original_data))
            original_df = pd.DataFrame(original_data, index=time_points)
            original_df.columns = [f'variable{i+1}' for i in range(original_data.shape[1])]
            original_df.index.name = 'timestamp'
            
            # 保存原始数据为CSV
            original_df.to_csv(opj(outdir, 'original_data.csv'))
            print(f"原始数据已保存到 {opj(outdir, 'original_data.csv')}")
            print("\n原始数据预览:")
            print(original_df.head())
        
        # 加载异常检测结果
        anomaly_scores = np.load(opj(outdir, 'anomaly_scores.npy'))
        anomaly_labels = np.load(opj(outdir, 'anomaly_labels.npy'))
        
        # 确保数据类型正确
        anomaly_scores = anomaly_scores.astype(float)
        anomaly_labels = anomaly_labels.astype(int)
        
        # 打印原始数据信息
        print("\n异常检测结果信息:")
        print("异常分数数组形状:", anomaly_scores.shape)
        print("异常分数数组类型:", anomaly_scores.dtype)
        print("异常分数范围:", np.min(anomaly_scores), "到", np.max(anomaly_scores))
        print("异常标签数组形状:", anomaly_labels.shape)
        print("异常标签数组类型:", anomaly_labels.dtype)
        
        # 尝试加载变量异常数据（如果存在）
        try:
            variable_anomalies = np.load(opj(outdir, 'variable_anomalies.npy'))
            print("变量异常数组形状:", variable_anomalies.shape)
            if len(variable_anomalies.shape) == 2:
                T, N = variable_anomalies.shape
                anomaly_scores = variable_anomalies.astype(float)  # 确保是浮点数
                anomaly_labels = (variable_anomalies > 0).astype(int)  # 转换为0/1标签
                print(f"使用variable_anomalies数据: {T}个时间点, {N}个变量")
        except:
            print("未找到variable_anomalies.npy文件或加载失败")
            # 如果是一维数组，检查是否需要重塑
            if len(anomaly_scores.shape) == 1:
                print("检测到一维数组，尝试重构多变量数据...")
                T = len(anomaly_scores)
                # 假设有5个变量（根据您之前的设置）
                N = 5
                # 尝试将数据重塑为多变量形式
                try:
                    anomaly_scores = anomaly_scores.reshape(T//N, N).astype(float)
                    anomaly_labels = anomaly_labels.reshape(T//N, N).astype(int)
                    print(f"重塑数据为: {T//N}个时间点, {N}个变量")
                except:
                    print("重塑失败，保持为单变量数据")
                    anomaly_scores = anomaly_scores.reshape(-1, 1).astype(float)
                    anomaly_labels = anomaly_labels.reshape(-1, 1).astype(int)
                    N = 1
                    T = len(anomaly_scores)

        # 获取最终的数据维度
        T, N = anomaly_scores.shape
        print(f"\n最终数据维度: {T}个时间点, {N}个变量")

        # 创建宽格式DataFrame - 异常标签
        labels_dict = {'timestamp': range(T)}
        for n in range(N):
            labels_dict[f'variable{n+1}'] = anomaly_labels[:, n]
        df_labels = pd.DataFrame(labels_dict)

        # 创建宽格式DataFrame - 异常分数
        scores_dict = {'timestamp': range(T)}
        for n in range(N):
            scores_dict[f'variable{n+1}'] = anomaly_scores[:, n]
        df_scores = pd.DataFrame(scores_dict)

        # 保存为CSV，确保使用float格式
        df_labels.to_csv(opj(outdir, 'anomaly_labels_wide.csv'), index=False)
        df_scores.to_csv(opj(outdir, 'anomaly_scores_wide.csv'), index=False, float_format='%.6f')

        # 打印预览
        print("\n异常标签文件预览 (anomaly_labels_wide.csv):")
        print(df_labels.head())
        print("\n异常分数文件预览 (anomaly_scores_wide.csv):")
        print(df_scores.head())

        # 可视化
        plt.figure(figsize=(15, 4*N))
        for n in range(N):
            plt.subplot(N, 1, n+1)
            # 绘制异常分数
            plt.plot(df_scores[f'variable{n+1}'], label=f'Variable {n+1}')
            # 标记异常点
            anomaly_indices = df_labels.index[df_labels[f'variable{n+1}'] == 1]
            plt.scatter(anomaly_indices, df_scores.iloc[anomaly_indices, n+1], 
                       color='red', marker='x', s=100, label='Anomaly')
            plt.title(f'Variable {n+1} Anomaly Detection')
            plt.legend()

        plt.tight_layout()
        plt.savefig(opj(outdir, 'anomaly_visualization.png'))
        plt.close()

        # 打印统计信息
        print("\n异常检测统计信息:")
        for n in range(N):
            var_name = f'variable{n+1}'
            anomaly_count = df_labels[var_name].sum()
            anomaly_ratio = anomaly_count / T * 100
            print(f"{var_name}:")
            print(f"  异常点数量: {anomaly_count}")
            print(f"  异常比例: {anomaly_ratio:.2f}%")

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='处理异常检测结果并生成可视化')
    parser.add_argument('--data_path', type=str, 
                       default="data_10_26/test_d/data_processed/root_cause_segment.npy",
                       help='原始数据的路径')
    parser.add_argument('--outdir', type=str, 
                       default='results/causal_anomaly_detection_latest/anomaly_detection_results',
                       help='输出目录路径')
    parser.add_argument('--method', type=str, 
                       choices=['amplitude', 'statistical'],
                       default='amplitude',
                       help='异常检测方法')
    parser.add_argument('--threshold_ratio', type=float, 
                       default=2.0,
                       help='异常检测阈值（用于amplitude方法）')
    parser.add_argument('--threshold_std', type=float, 
                       default=3.0,
                       help='异常检测阈值（用于statistical方法）')
    args = parser.parse_args()
    
    # 根据方法选择阈值参数
    kwargs = {}
    if args.method == 'amplitude':
        kwargs['threshold_ratio'] = args.threshold_ratio
    elif args.method == 'statistical':
        kwargs['threshold_std'] = args.threshold_std
    
    process_anomaly_detection(args.data_path, args.outdir, 
                            method=args.method, 
                            **kwargs)

if __name__ == "__main__":
    main()