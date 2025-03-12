import logging
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import subprocess

import networkx as nx
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import argparse
from omegaconf import OmegaConf
from copy import deepcopy
import torch
from torch import dropout, nn

from utils.cuts_parts import *
from utils.gumbel_softmax import gumbel_softmax
from utils.misc import plot_causal_matrix, reproduc, plot_causal_matrix_in_training, calc_and_log_metrics, log_time_series, prepross_data
from utils.batch_generater import batch_generater
from utils.opt_type import CUTSopt
from utils.logger import MyLogger
from utils.data_interpolate import interp_multivar_data
from utils.load_data import simulate_var_from_links, simulate_var, simulate_lorenz_96_process, load_netsim_data
from utils.anomaly_detection import AmplitudeAnomalyDetector, StatisticalAnomalyDetector, detect_anomalies

from datetime import datetime

import os
from einops import rearrange



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import chi2


class CUTS(object):
    def __init__(self, args: CUTSopt.CUTSargs, log, device="cuda"):
        self.log: MyLogger = log
        self.args = args
        self.device = device

        if self.args.data_pred.model == "multi_mlp":
            self.fitting_model = MultiMLP(self.args.input_step * self.args.n_nodes * self.args.data_dim,
                                          self.args.data_pred.mlp_hid,
                                          self.args.data_dim * self.args.data_pred.pred_step,
                                          self.args.data_pred.mlp_layers,
                                          self.args.n_nodes).to(self.device)
        elif self.args.data_pred.model == "multi_lstm":
            self.fitting_model = MultiLSTM(self.args.n_nodes * self.args.data_dim,
                                          self.args.data_pred.mlp_hid,
                                          self.args.data_dim * self.args.data_pred.pred_step,
                                          self.args.data_pred.mlp_layers,
                                          self.args.n_nodes).to(self.device)
        else:
            raise NotImplementedError

        self.data_pred_loss = nn.MSELoss()
        self.data_pred_optimizer = torch.optim.Adam(self.fitting_model.parameters(),
                                                    lr=self.args.data_pred.lr_data_start,
                                                    weight_decay=self.args.data_pred.weight_decay)
        
        
        if "every" in self.args.fill_policy:
            lr_schedule_length = int(self.args.fill_policy.split("_")[-1])
        else:
            lr_schedule_length = self.args.total_epoch
            
        gamma = (self.args.data_pred.lr_data_end / self.args.data_pred.lr_data_start) ** (1 / lr_schedule_length)
        self.data_pred_scheduler = torch.optim.lr_scheduler.StepLR(
            self.data_pred_optimizer, step_size=1, gamma=gamma)
        
        if hasattr(self.args, "disable_graph") and self.args.disable_graph:
            print("Using full graph and disable graph discovery...")
            self.graph = nn.Parameter(torch.ones([self.args.n_nodes, self.args.n_nodes, self.args.input_step]).to(self.device) * 1000)
        else:
            self.graph = nn.Parameter(torch.ones([self.args.n_nodes, self.args.n_nodes, self.args.input_step]).to(self.device) * 0)
        # self.graph = nn.Parameter(torch.zeros([self.args.n_nodes, self.args.n_nodes, self.args.input_step]).to(self.device))
        self.graph_optimizer = torch.optim.Adam([self.graph], lr=self.args.graph_discov.lr_graph_start)
        gamma = (self.args.graph_discov.lr_graph_end / self.args.graph_discov.lr_graph_start) ** (1 / self.args.total_epoch)
        self.graph_scheduler = torch.optim.lr_scheduler.StepLR(self.graph_optimizer, step_size=1, gamma=gamma)

        end_tau, start_tau = self.args.graph_discov.end_tau, self.args.graph_discov.start_tau
        self.gumbel_tau_gamma = (end_tau / start_tau) ** (1 / self.args.total_epoch)
        self.gumbel_tau = start_tau
        self.start_tau = start_tau
        
        end_lmd, start_lmd = self.args.graph_discov.lambda_s_end, self.args.graph_discov.lambda_s_start
        self.lambda_gamma = (end_lmd / start_lmd) ** (1 / self.args.total_epoch)
        self.lambda_s = start_lmd
        
        # 添加异常检测器
        self.anomaly_detector = None

    def latent_data_pred(self, x, y, observ_mask):
        
        def sample_graph(sample_matrix, batch_size, prob=True):
            sample_matrix = torch.sigmoid(
                sample_matrix[None, :, :, :].expand(batch_size, -1, -1, -1))
            if prob:
                return torch.bernoulli(sample_matrix)
            else:
                return sample_matrix
        
        bs, n, m, t, d = x.shape
        self.fitting_model.train()
        self.data_pred_optimizer.zero_grad()
        
        # graph_no_self = self.graph.clone()
        # for i in range(graph_no_self.shape[0]):
        #     graph_no_self[i,i,:] = torch.ones_like(graph_no_self[i,i,:]) * -1000
        if hasattr(self.args.data_pred, "disable_graph") and \
            self.args.data_pred.disable_graph:
                sampled_graph = torch.ones_like(self.graph)[None].expand(bs, -1, -1, -1)
        else:
            sampled_graph = sample_graph(self.graph, bs, self.args.data_pred.prob)
            
        y_pred = self.fitting_model(x, sampled_graph)

        loss = self.data_pred_loss(y * observ_mask, y_pred * observ_mask) / torch.mean(observ_mask)
        loss.backward()
        self.data_pred_optimizer.step()
        return y_pred, loss

    def graph_discov(self, x, y, observ_mask):

        def sigmoid_gumbel_sample(graph, batch_size, tau=1):
            prob = torch.sigmoid(graph[None, :, :, :, None].expand(batch_size, -1, -1, -1, -1))
            logits = torch.concat([prob, (1-prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau)[:, :, :, :, 0]
            return samples

        # self.fitting_model.eval()
        self.graph_optimizer.zero_grad()
        prob_graph = torch.sigmoid(self.graph[None, :, :])
        sample_graph = sigmoid_gumbel_sample(self.graph, self.args.batch_size, tau=self.gumbel_tau)

        y_pred = self.fitting_model(x, sample_graph)
        
        gs = prob_graph.shape
        loss_sparsity = torch.norm(prob_graph, p=1) / (gs[0] * gs[1] * gs[2])
        loss_data = self.data_pred_loss(y * observ_mask, y_pred * observ_mask) / torch.mean(observ_mask)
        loss = loss_sparsity * self.lambda_s + loss_data
        loss.backward()
        self.graph_optimizer.step()
        return loss, loss_sparsity, loss_data



    def train(self, data, observ_mask, original_data, true_cm=None):

        original_data = torch.from_numpy(original_data).float().to(self.device)
        observ_mask = torch.from_numpy(observ_mask).float().to(self.device)
        data = torch.from_numpy(data).float().to(self.device)
        
        if self.args.supervision_policy == "masked":
            print("Using masked supervision for data prediction...")
        elif self.args.supervision_policy == "full":
            print("Using full supervision for data prediction......")
            observ_mask = torch.ones_like(observ_mask)
        elif "masked_before" in self.args.supervision_policy:
            print(f"Using masked supervision for data prediction ({self.args.supervision_policy:s})......")

        latent_pred_step = 0
        graph_discov_step = 0
        pbar = tqdm.tqdm(total=self.args.total_epoch)
        data_interp = deepcopy(data)
        original_mask = deepcopy(observ_mask)
        auc = 0
        for epoch_i in range(self.args.total_epoch):
            if "every" in self.args.fill_policy:
                update_every = int(self.args.fill_policy.split("_")[-1])
                if (epoch_i+1) % update_every == 0:
                    data = data_pred
                    print("Update data!")
                    # self.graph_optimizer.param_groups[0]['lr'] = self.args.graph_discov.lr_graph_start
                    self.data_pred_optimizer.param_groups[0]['lr'] = self.args.data_pred.lr_data_start
                    observ_mask = torch.ones_like(original_mask)
            elif "rate" in self.args.fill_policy:
                update_rate = float(self.args.fill_policy.split("_")[1])
                update_after = int(self.args.fill_policy.split("_")[3])
                if epoch_i+1 > update_after:
                    if epoch_i == update_after:
                        print("Data update started!")
                    data = data * (1 - update_rate) + data_pred * update_rate
            else:
                # no data update
                pass
            
            if "masked_before" in self.args.supervision_policy:
                masked_before = int(self.args.supervision_policy.split("_")[2])
                if epoch_i == masked_before:
                    print("Using full supervision for data prediction......")
                    observ_mask = torch.ones_like(original_mask)
                    self.gumbel_tau = self.start_tau
            
            # Data Prediction
            if hasattr(self.args, "data_pred"):
                if hasattr(self.args, "sample_period"):
                    sample_period = self.args.sample_period
                else:
                    sample_period = 1
                ## 
                batch_gen = batch_generater(data, observ_mask, # !!!!! TO-DO
                                            bs=self.args.batch_size, 
                                            n_nodes=self.args.n_nodes, 
                                            input_step=self.args.input_step, 
                                            pred_step=self.args.data_pred.pred_step,
                                            sample_period=sample_period)
                batch_gen = list(batch_gen)
                
                data_pred = deepcopy(data) # masked data points are predicted
                data_pred_all = deepcopy(data)
                for x, y, t, mask in batch_gen:
                    latent_pred_step += self.args.batch_size
                    y_pred, loss = self.latent_data_pred(x, y, mask)
                    data_pred[t] = (y_pred*(1-mask) + y*mask).clone().detach()[:,:,0]
                    data_pred_all[t] = y_pred.clone().detach()[:,:,0]
                    self.log.log_metrics({"latent_data_pred/pred_loss": loss.item()}, latent_pred_step)
                    pbar.set_postfix_str(f"S1 loss={loss.item():.2f}, spr=IDLE, auc={auc:.4f}")

                current_data_pred_lr = self.graph_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_data_pred_lr}, latent_pred_step)
                self.data_pred_scheduler.step()
                mse_pred_to_original = self.data_pred_loss(original_data, data_pred)
                mse_interp_to_original = self.data_pred_loss(original_data, data_interp)
                
                self.log.log_metrics({"latent_data_pred/mse_pred_to_original": mse_pred_to_original,
                                      "latent_data_pred/mse_interp_to_original": mse_interp_to_original}, latent_pred_step)
            
            # Graph Discovery
            if hasattr(self.args, "graph_discov"):
                # batch_gen = batch_generater(data, observ_mask, 
                #                             bs=self.args.batch_size, 
                #                             n_nodes=self.args.n_nodes, 
                #                             input_step=self.args.input_step, 
                #                             pred_step=self.args.data_pred.pred_step, 
                #                             sample_period=period)
                for x, y, t, mask in batch_gen:
                    graph_discov_step += self.args.batch_size
                    if hasattr(self.args, "disable_graph") and self.args.disable_graph:
                        pass
                    else:
                        loss, loss_sparsity, loss_data = self.graph_discov(x, y, mask)
                        self.log.log_metrics({"graph_discov/sparsity_loss": loss_sparsity.item(),
                                            "graph_discov/data_loss": loss_data.item(),
                                            "graph_discov/total_loss": loss.item()}, graph_discov_step)
                        pbar.set_postfix_str(f"S2 loss={loss_data.item():.2f}, spr={loss_sparsity.item():.2f}, auc={auc:.4f}")
                    
                self.graph_scheduler.step()
                current_graph_disconv_lr = self.graph_optimizer.param_groups[0]['lr']
                self.log.log_metrics({"graph_discov/lr": current_graph_disconv_lr}, graph_discov_step)
                self.log.log_metrics({"graph_discov/tau": self.gumbel_tau}, graph_discov_step)
                self.gumbel_tau *= self.gumbel_tau_gamma

            pbar.update(1)
            self.lambda_s *= self.lambda_gamma
                     
            calc, val = self.args.causal_thres.split("_")
            if calc == "value":
                threshold = float(val)
            else:
                raise NotImplementedError
            
            time_coef_mat = self.graph.detach().cpu().numpy()
            plot_roc = False
            if (epoch_i+1) % self.args.show_graph_every == 0:
                avg_mask = np.mean(observ_mask.cpu().numpy(), axis=(0,2))
                if np.min(avg_mask) < 1:
                    time_series_idx = int(np.argwhere(avg_mask < 1)[0])
                else:
                    time_series_idx = 0
                log_time_series(original_data.cpu()[-100:,time_series_idx], 
                                data_interp.cpu()[-100:,time_series_idx], 
                                data_pred_all.cpu()[-100:,time_series_idx], log=self.log, log_step=latent_pred_step)
                plot_causal_matrix_in_training(time_coef_mat, self.log, graph_discov_step, threshold=threshold)
                plot_roc = True
            
            # Show TPR FPR AUC ROC
            if true_cm is not None:
                time_prob_mat = torch.sigmoid(self.graph).detach().cpu().numpy()      
                auc = calc_and_log_metrics(time_prob_mat, true_cm, self.log, graph_discov_step, threshold=threshold, plot_roc=plot_roc)
                

    def detect_anomalies(self, data, method="amplitude", **kwargs):
        """
        执行异常检测
        Args:
            data: 输入数据
            method: 使用的异常检测方法 ("amplitude" 或 "statistical")
            **kwargs: 其他参数，包括：
                - window_size: 滑动窗口大小
                - threshold_ratio: 幅值检测的阈值（用于amplitude方法）
                - threshold_std: 标准差倍数（用于statistical方法）
        """
        print(f"使用 {method} 方法进行异常检测...")
        
        from utils.anomaly_detection import detect_anomalies as detect_func
        return detect_func(data, method=method, **kwargs)


def prepare_data(opt):
    if opt.name == "custom_data":
        # 根据文件扩展名选择合适的加载方法
        file_ext = os.path.splitext(opt.param.path)[1].lower()
        if file_ext == '.csv':
            import pandas as pd
            data = pd.read_csv(opt.param.path).values
        elif file_ext == '.npy':
            data = np.load(opt.param.path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        # 确保数据格式为 [T, N, D]
        if len(data.shape) == 2:  # 如果数据是2维的，添加一个维度
            data = data.reshape(data.shape[0], data.shape[1], 1)
        
        true_cm = None  # 或者加载您的因果矩阵
    elif opt.name == "var":
        data, true_cm = simulate_var_from_links(**opt.param)
    elif opt.name == "lorenz_96":
        data, true_cm = simulate_lorenz_96_process(**opt.param)
    elif opt.name == "zeros": # for debug
        data = np.zeros([opt.param.T, opt.param.N, 1])
    elif opt.name == "netsim":
        data, true_cm = load_netsim_data(**opt.param)
    else:
        raise NotImplementedError

    T, N, D = data.shape
    print("Data shape: ", data.shape)
    data = prepross_data(data)
    
    mask = np.ones_like(data)
    if hasattr(opt.pre_sample, "period") or hasattr(opt.pre_sample, "random_period"):
        if hasattr(opt.pre_sample, "period"):
            assert N == len(opt.pre_sample.period), "opt.pre_sample.period length not matched"
            period = opt.pre_sample.period
            print("Using sampling periods: ", period)
        elif hasattr(opt.pre_sample, "random_period"):
            np.random.seed(opt.pre_sample.random_period.seed)
            period = np.random.choice(opt.pre_sample.random_period.choices, N, p=opt.pre_sample.random_period.prob)
            print("Generated presampling periods: ", period)
        mask *= 0
        for i in range(N):
            period_i = period[i]
            mask[::period_i, i] += 1
            
    elif hasattr(opt.pre_sample, "random_missing"):
        np.random.seed(opt.pre_sample.random_missing.seed)
        p = opt.pre_sample.random_missing.missing_prob
        missing_var = opt.pre_sample.random_missing.missing_var
        if isinstance(missing_var, str) and missing_var=="all":
            mask = np.random.choice([0,1], size=mask.shape, p=[p,1-p])
        else:
            for var_i in missing_var:
                mask[:,var_i] = np.random.choice([0,1], size=mask[:,var_i].shape, p=[p,1-p])
        print(f"Generated random missing with missing_prob: {p:.4f}")
    else:
        raise NotImplementedError
        

    sampled_data = data * mask
    interp_data = interp_multivar_data(sampled_data, mask, interp=opt.init_fill)
    return interp_data, mask, true_cm, data


def main(opt: CUTSopt, device="cuda"):
    reproduc(**opt.reproduc)
    timestamp = datetime.now().strftime("_%Y_%m%d_%H%M%S_%f")
    opt.task_name += timestamp
    proj_path = opj(opt.dir_name, opt.task_name)
    log = MyLogger(log_dir=proj_path, **opt.log)
    log.log_opt(opt)

    # 第一阶段：使用正常数据进行因果发现
    print("阶段1：使用正常数据进行因果发现...")
    data_dir = opj(log.log_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 复制原始数据文件到日志目录
    original_normal_path = "data_10_26/test_d/data_processed/normal_segments.npy"
    original_anomaly_path = "data_10_26/test_d/data_processed/root_cause_segment.npy"
    original_feature_mapping = "data_10_26/test_d/data_processed/feature_mapping.txt"
    
    import shutil
    # 复制正常数据
    shutil.copy2(original_normal_path, opj(data_dir, 'normal_segments.npy'))
    # 复制异常数据
    shutil.copy2(original_anomaly_path, opj(data_dir, 'root_cause_segment.npy'))
    # 复制特征映射文件
    shutil.copy2(original_feature_mapping, opj(data_dir, 'feature_mapping.txt'))
    
    # 从日志目录读取正常数据
    opt.data.param.path = opj(data_dir, 'normal_segments.npy')
    normal_data, normal_mask, true_cm, original_normal_data = prepare_data(opt.data)
    
    if true_cm is not None:
        sub_cg = plot_causal_matrix(
            true_cm, 
            figsize=[1.5*normal_data.shape[1], 1*normal_data.shape[1]])
        log.log_figures(name="True Graph", figure=sub_cg, iters=0)
    
    if hasattr(opt, "cuts"):
        cuts = CUTS(opt.cuts, log, device=device)
        # 使用正常数据训练模型和发现因果关系
        cuts.train(normal_data, normal_mask, original_normal_data, true_cm)
        
        # 第二阶段：使用异常数据进行异常检测
        print("\n阶段2：使用异常数据进行异常检测...")
        # 从日志目录读取异常数据
        opt.data.param.path = opj(data_dir, 'root_cause_segment.npy')
        anomaly_data, anomaly_mask, _, original_anomaly_data = prepare_data(opt.data)
        
        # 执行异常检测
        if hasattr(opt.cuts, "anomaly_detection"):
            print("开始进行异常检测...")
            method = getattr(opt.cuts.anomaly_detection, "method", "amplitude")
            window_size = getattr(opt.cuts.anomaly_detection, "window_size", 5)
            
            # 根据不同方法构建参数字典
            kwargs = {"window_size": window_size}
            
            if method == "amplitude":
                threshold_ratio = getattr(opt.cuts.anomaly_detection, "threshold_ratio", 3.0)
                use_window = getattr(opt.cuts.anomaly_detection, "use_window", False)
                kwargs.update({
                    "threshold_ratio": threshold_ratio,
                    "use_window": use_window
                })
                print(f"使用幅值检测方法 (threshold_ratio={threshold_ratio}, window_size={window_size}, use_window={use_window})")
            
            elif method == "statistical":
                threshold_std = getattr(opt.cuts.anomaly_detection, "threshold_std", 3.0)
                kwargs.update({
                    "threshold_std": threshold_std
                })
                print(f"使用统计方法进行异常检测 (threshold_std={threshold_std}, window_size={window_size})")
            
            # 执行异常检测
            anomaly_scores, anomaly_labels = cuts.detect_anomalies(
                anomaly_data,
                method=method,
                **kwargs
            )
            
            # 保存异常检测结果
            anomaly_dir = opj(log.log_dir, 'anomaly_detection')
            os.makedirs(anomaly_dir, exist_ok=True)
            
            # 保存数据文件
            np.save(opj(anomaly_dir, 'anomaly_scores.npy'), anomaly_scores)
            np.save(opj(anomaly_dir, 'anomaly_labels.npy'), anomaly_labels)
            
            # 调用anomaly_labels.py处理结果
            print("\n处理异常检测结果并生成可视化...")
            try:
                cmd = [
                    'python', 
                    'anomaly_labels.py',
                    '--data_path', opt.data.param.path,
                    '--outdir', anomaly_dir,
                    '--method', method
                ]
                
                # 根据方法添加相应的参数
                if method == "amplitude":
                    cmd.extend(['--threshold_ratio', str(kwargs['threshold_ratio'])])
                elif method == "statistical":
                    cmd.extend(['--threshold_std', str(kwargs['threshold_std'])])
                
                subprocess.run(cmd, check=True)
                print("异常检测结果处理完成")
            except subprocess.CalledProcessError as e:
                print(f"处理异常检测结果时出错: {str(e)}")
            
            print(f"异常检测完成。结果已保存到: {anomaly_dir}")
            
            # 生成因果图
            try:
                from generate_causal_graphs import generate_causal_graphs
                print("\n生成因果图...")
                generate_causal_graphs(log.log_dir)
                print("因果图生成完成，已保存到日志目录")
            except Exception as e:
                print(f"生成因果图时出错: {str(e)}")

    # 第三阶段：根因分析
    print("阶段3：根因分析...")
    from root_cause_analysis import analyze_root_cause
    
    # 执行根因分析
    root_cause_report = analyze_root_cause(log.log_dir)
    print("\n根因分析结果:")
    print(root_cause_report)
    
    # 生成因果图
    print("生成因果图...")
    from generate_causal_graphs import generate_causal_graphs
    generate_causal_graphs(log.log_dir)
    
    print(f"所有处理完成。结果保存在: {log.log_dir}")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser(description='Batch Compress')
    parser.add_argument('-opt', type=str, default=opj(opd(__file__),
                        'cuts_anomaly.yaml'), help='yaml file path')
    parser.add_argument('-g', help='availabel gpu list', default='0', type=str)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-log', action='store_true')
    args = parser.parse_args()

    if args.g == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = "mps"
    elif args.g == "cpu":
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.g
        device = "cuda"
    print(f"Using device: {device}")
    main(OmegaConf.load(args.opt), device=device)

