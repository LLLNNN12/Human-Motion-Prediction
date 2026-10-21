"""
train_xgboost.py
使用LightGBM学习AuxFormer模型预测的残差 - 22关节消融实验版本（去除掩码和去噪）
只训练前10帧的残差模型（660个模型：66维×10帧）

修改内容：
1. 去除类别特征（动作语义one-hot编码）
2. 利用LightGBM自带的特征重要性分析进行特征筛选，保留60%特征（比例可配置）
3. 保留原有全部功能

特性：
1. 增强特征工程：包含预测统计、输入序列、速度、加速度、关键帧、关节距离、频域等特征（去除动作类别）
2. 特征重要性筛选：先用代表性子模型评估特征重要性，再用筛选后的特征训练全部模型
3. 只拟合前10帧：660个模型（22关节×10帧×3坐标）
4. GPU加速：自动检测并尝试使用CUDA加速LightGBM
5. CPU限制：如果GPU不可用，限制CPU使用率在85%
6. 早停机制：防止过拟合
7. 特定帧MPJPE：计算第2、4、8、10帧的MPJPE
8. 适配消融实验版本AuxFormer（去除掩码和去噪机制）
"""
import sys
import torch
import numpy as np
import pandas as pd
import os
import pickle
import time
import glob
import multiprocessing
import lightgbm as lgb
from model import AuxFormer
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.fft import fft, dct

# 22关节配置
N_JOINTS = 22
N_FEATURES = 66  # 22 * 3

# 只训练前10帧
TRAIN_FRAMES = 10
TRAIN_OUTPUTS = N_FEATURES * TRAIN_FRAMES  # 660

# 15种动作类别（保留用于数据解析，但不再作为特征）
ACTION_NAMES = [
    'directions', 'discussion', 'eating', 'greeting', 'phoning',
    'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',
    'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether'
]

# 骨骼连接定义（21对）
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11),
    (11, 12), (12, 13),
    (11, 14), (14, 15), (15, 16), (16, 17),
    (11, 18), (18, 19), (19, 20), (20, 21)
]

# 对称关节对
SYMMETRIC_PAIRS = [
    (1, 5), (2, 6), (3, 7), (4, 8),
    (14, 18), (15, 19), (16, 20), (17, 21)
]

# 角度计算的关节三元组
ANGLE_TRIPLETS = [
    (14, 15, 16), (18, 19, 20), (5, 6, 7), (1, 2, 3),
]

# ==================== 特征筛选默认比例 ====================
DEFAULT_FEATURE_KEEP_RATIO = 0.6  # 保留60%的特征


def get_optimal_thread_count(target_usage=0.85):
    """计算保持CPU使用率在目标范围内需要的线程数"""
    total_threads = multiprocessing.cpu_count()
    optimal_threads = max(1, int(total_threads * target_usage))
    return optimal_threads


def check_lightgbm_gpu_support():
    """检查LightGBM是否支持GPU加速"""
    print("  正在检测LightGBM GPU支持...")

    np.random.seed(42)
    X_test = np.random.rand(100, 10).astype(np.float32)
    y_test = np.random.rand(100).astype(np.float32)

    gpu_configs = [
        {
            'device': 'cuda',
            'params': {
                'objective': 'regression',
                'device': 'cuda',
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.1,
            }
        },
        {
            'device': 'gpu',
            'params': {
                'objective': 'regression',
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.1,
            }
        },
    ]

    for config in gpu_configs:
        try:
            test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)
            booster = lgb.train(
                config['params'],
                test_data,
                num_boost_round=3,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            _ = booster.predict(X_test)
            del booster, test_data

            print(f"  ✓ LightGBM {config['device'].upper()} 加速可用!")
            return True, config['device'], config['params']

        except Exception as e:
            error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            print(f"  ✗ {config['device'].upper()} 不可用: {error_msg}")
            continue

    return False, None, None


class MocapDatasetForLightGBM:
    """专门用于LightGBM的22关节数据加载器（带动作标签）- 消融实验版本"""

    def __init__(self, data_dir, input_n=10, output_n=30, scaler=None):
        self.input_n = input_n
        self.output_n = output_n
        self.n_joints = N_JOINTS
        self.n_features = N_FEATURES

        if scaler is None:
            raise ValueError("必须提供scaler参数！")
        self.scaler = scaler

        excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))

        if not excel_files:
            raise ValueError(f"在目录 {data_dir} 中没有找到Excel文件")

        print(f"  找到 {len(excel_files)} 个Excel文件")

        all_sequences = []
        self.file_actions = []

        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)
                filename = os.path.basename(file_path)
                action = self._parse_action_from_filename(filename)

                if df.shape[1] < self.n_features:
                    continue

                position_data = df.iloc[:, :self.n_features].values
                sequence_data = self.preprocess_sequence(position_data)

                if len(sequence_data) >= input_n + output_n:
                    all_sequences.append(sequence_data)
                    self.file_actions.append(action)

            except Exception as e:
                continue

        if not all_sequences:
            raise ValueError("没有有效的数据文件")

        print(f"  成功加载 {len(all_sequences)} 个序列")

        normalized_sequences = []
        for seq in all_sequences:
            normalized_seq = self.scaler.transform(seq)
            normalized_sequences.append(normalized_seq)

        self.sequences = normalized_sequences
        self.samples, self.sample_actions = self.generate_samples()
        self.format_data()

        print(f"  总样本数: {len(self.samples)}")

    def _parse_action_from_filename(self, filename):
        """从文件名解析动作类型"""
        filename_lower = filename.lower()
        ordered_actions = [
            'walkingtogether', 'walkingdog', 'sittingdown', 'takingphoto',
            'directions', 'discussion', 'eating', 'greeting', 'phoning',
            'posing', 'purchases', 'sitting', 'smoking', 'waiting', 'walking'
        ]
        for action in ordered_actions:
            if action in filename_lower:
                return action
        return 'unknown'

    def preprocess_sequence(self, data):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data[:, :self.n_features] = data[:, :self.n_features] / 1000.0
        return data

    def generate_samples(self):
        samples = []
        sample_actions = []
        seq_len = self.input_n + self.output_n

        for seq_idx, sequence in enumerate(self.sequences):
            action = self.file_actions[seq_idx]
            for i in range(len(sequence) - seq_len + 1):
                sample = sequence[i:i + seq_len]
                samples.append(sample)
                sample_actions.append(action)

        return samples, sample_actions

    def format_data(self):
        formatted_samples = []

        for sample in self.samples:
            position_data = sample[:, :self.n_features]
            position_data = position_data.reshape(len(sample), self.n_joints, 3)
            formatted_samples.append(position_data)

        self.samples = formatted_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        action = self.sample_actions[idx]

        input_seq = sample[:self.input_n]
        output_seq = sample[self.input_n:]

        input_seq = torch.FloatTensor(input_seq).transpose(0, 1)
        output_seq = torch.FloatTensor(output_seq).transpose(0, 1)

        vel_seq = torch.zeros_like(input_seq)
        vel_seq[:, 1:] = input_seq[:, 1:] - input_seq[:, :-1]
        vel_seq[:, 0] = vel_seq[:, 1]

        return input_seq, vel_seq, output_seq, output_seq, idx, action


def extract_model_features_enhanced(model, dataset, device, batch_size=16):
    """从AuxFormer模型中提取预测值和增强特征"""
    print("  正在提取模型特征（增强版）...")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model.eval()
    all_predictions = []
    all_ground_truth = []
    all_input_features = []
    all_velocities = []
    all_accelerations = []
    all_actions = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            loc, vel, loc_end, _, _, actions = data
            loc = loc.to(device)
            loc_end = loc_end.to(device)
            vel = vel.to(device)

            all_traj = torch.cat([loc, torch.zeros_like(loc_end)], dim=2)
            prediction = model.predict(all_traj)

            all_predictions.append(prediction.cpu().numpy())
            all_ground_truth.append(loc_end.cpu().numpy())
            all_input_features.append(loc.cpu().numpy())
            all_velocities.append(vel.cpu().numpy())

            acc = torch.zeros_like(vel)
            acc[:, :, 1:] = vel[:, :, 1:] - vel[:, :, :-1]
            all_accelerations.append(acc.cpu().numpy())
            all_actions.extend(actions)

            if batch_idx % 50 == 0:
                print(f"    已处理 {batch_idx}/{len(dataloader)} 批次")

    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    input_features = np.concatenate(all_input_features, axis=0)
    velocities = np.concatenate(all_velocities, axis=0)
    accelerations = np.concatenate(all_accelerations, axis=0)

    print(f"  提取完成: 预测形状={predictions.shape}, 真实值形状={ground_truth.shape}")

    return predictions, ground_truth, input_features, velocities, accelerations, all_actions


def get_feature_names():
    """获取所有特征的分类名称（已去除动作类别特征）"""
    feature_names = []

    # ========== 特征组1: 预测统计特征 ==========
    for stat in ['mean', 'std', 'max', 'min']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"pred_global_{stat}_{axis}")

    for stat in ['mean', 'std']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"pred_short_{stat}_{axis}")

    for stat in ['mean', 'std']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"pred_mid_{stat}_{axis}")

    for stat in ['mean', 'std']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"pred_long_{stat}_{axis}")

    for q in ['q25', 'q75', 'iqr']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"pred_{q}_{axis}")

    for moment in ['skew', 'kurtosis']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"pred_{moment}_{axis}")

    # ========== 特征组2: 输入序列特征 ==========
    for stat in ['mean', 'std']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"input_global_{stat}_{axis}")

    for n in [1, 3, 5, 7, 10]:
        for j in range(N_JOINTS):
            for axis in ['x', 'y', 'z']:
                feature_names.append(f"input_last{n}_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"input_range_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"input_exp_weighted_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"input_first_last_diff_j{j}_{axis}")

    # ========== 特征组3: 速度特征 ==========
    for stat in ['mean', 'std']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"vel_global_{stat}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"vel_last1_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"vel_last3_j{j}_{axis}")

    for trend in ['last1_last3', 'last1_last5', 'last1_first1']:
        for j in range(N_JOINTS):
            for axis in ['x', 'y', 'z']:
                feature_names.append(f"vel_trend_{trend}_j{j}_{axis}")

    for axis in ['x', 'y', 'z']:
        feature_names.append(f"vel_cv_{axis}")

    for j in range(N_JOINTS):
        feature_names.append(f"vel_magnitude_j{j}")

    # ========== 特征组4: 加速度特征 ==========
    for stat in ['mean', 'std']:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"acc_global_{stat}_{axis}")

    for n in [1, 3, 5]:
        for j in range(N_JOINTS):
            for axis in ['x', 'y', 'z']:
                feature_names.append(f"acc_last{n}_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"acc_trend_j{j}_{axis}")

    for j in range(N_JOINTS):
        feature_names.append(f"acc_magnitude_j{j}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"jerk_last_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"acc_sign_changes_j{j}_{axis}")

    # ========== 特征组5: 关键帧预测特征 ==========
    for frame in [1, 2, 3, 4, 5, 6, 8, 10]:
        for j in range(N_JOINTS):
            for axis in ['x', 'y', 'z']:
                feature_names.append(f"pred_frame{frame}_j{j}_{axis}")

    for pair in ['1_2', '2_3', '2_4', '4_8', '8_10']:
        for j in range(N_JOINTS):
            for axis in ['x', 'y', 'z']:
                feature_names.append(f"pred_delta_{pair}_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"pred_continuity_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"pred_long_trend_j{j}_{axis}")

    # ========== 特征组6: 关节距离特征 ==========
    for j1, j2 in BONE_CONNECTIONS:
        for frame in ['last1', 'last3', 'last5']:
            for axis in ['x', 'y', 'z']:
                feature_names.append(f"bone_{j1}_{j2}_{frame}_{axis}")

    # ========== 特征组7: 频域特征 ==========
    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"fft_low_j{j}_{axis}")
            feature_names.append(f"fft_mid_j{j}_{axis}")
            feature_names.append(f"fft_high_j{j}_{axis}")
            feature_names.append(f"fft_dominant_j{j}_{axis}")

    for j in range(N_JOINTS):
        for axis in ['x', 'y', 'z']:
            for k in range(3):
                feature_names.append(f"dct_{k}_j{j}_{axis}")

    # ========== 【已去除】特征组8: 动作语义特征 ==========
    # 不再添加 action one-hot 特征

    # ========== 特征组9: 运动特性特征 ==========
    feature_names.append("motion_amplitude_total")

    for j in range(N_JOINTS):
        feature_names.append(f"motion_amplitude_j{j}")

    feature_names.append("motion_complexity")
    feature_names.append("pose_change_rate")

    for axis in ['x', 'y', 'z']:
        feature_names.append(f"com_position_{axis}")

    for axis in ['x', 'y', 'z']:
        feature_names.append(f"com_velocity_{axis}")

    # ========== 特征组10: 预测质量特征 ==========
    feature_names.append("pred_smoothness")
    feature_names.append("bone_length_variance")
    feature_names.append("continuity_score")
    feature_names.append("velocity_reasonability")
    feature_names.append("trajectory_curvature")

    # ========== 特征组11: 关节角度特征 ==========
    for triplet_idx, (j1, j2, j3) in enumerate(ANGLE_TRIPLETS):
        for stat in ['mean', 'std', 'max', 'min']:
            feature_names.append(f"angle_{j1}_{j2}_{j3}_{stat}")

    # ========== 特征组12: 对称性特征 ==========
    for pair_idx, (left, right) in enumerate(SYMMETRIC_PAIRS):
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"symmetry_{left}_{right}_{axis}")

    # ========== 特征组13: 自相关特征 ==========
    for lag in [1, 2, 3]:
        for axis in ['x', 'y', 'z']:
            feature_names.append(f"autocorr_lag{lag}_{axis}")

    return feature_names


def build_lightgbm_features(predictions, input_features, velocities, accelerations,
                            actions, predict_length):
    """
    构造LightGBM的增强特征矩阵 - 统一的特征构建函数
    【已修改】去除了动作类别one-hot特征（特征组8）

    Args:
        predictions: 预测结果 (N, joints, frames, 3)
        input_features: 输入特征 (N, joints, frames, 3)
        velocities: 速度 (N, joints, frames, 3)
        accelerations: 加速度 (N, joints, frames, 3)
        actions: 动作标签列表或单个动作字符串（保留参数但不生成特征）
        predict_length: 预测长度

    Returns:
        feature_matrix: 特征矩阵
    """
    N = predictions.shape[0]
    feature_list = []

    # 处理actions参数 - 统一为列表格式（保留用于兼容性，但不生成特征）
    if isinstance(actions, str):
        actions = [actions] * N

    # ========== 特征组1: AuxFormer预测统计特征 ==========
    pred_mean = predictions.mean(axis=(1, 2))
    pred_std = predictions.std(axis=(1, 2))
    pred_max = predictions.max(axis=(1, 2))
    pred_min = predictions.min(axis=(1, 2))
    feature_list.extend([pred_mean, pred_std, pred_max, pred_min])

    short_term_frames = min(10, predictions.shape[2])
    pred_short = predictions[:, :, :short_term_frames, :]
    pred_short_mean = pred_short.mean(axis=(1, 2))
    pred_short_std = pred_short.std(axis=(1, 2))
    feature_list.extend([pred_short_mean, pred_short_std])

    if predictions.shape[2] > 10:
        mid_end = min(20, predictions.shape[2])
        pred_mid = predictions[:, :, 10:mid_end, :]
        pred_mid_mean = pred_mid.mean(axis=(1, 2))
        pred_mid_std = pred_mid.std(axis=(1, 2))
    else:
        pred_mid_mean = np.zeros((N, 3))
        pred_mid_std = np.zeros((N, 3))
    feature_list.extend([pred_mid_mean, pred_mid_std])

    if predictions.shape[2] > 20:
        pred_long = predictions[:, :, 20:, :]
        pred_long_mean = pred_long.mean(axis=(1, 2))
        pred_long_std = pred_long.std(axis=(1, 2))
    else:
        pred_long_mean = np.zeros((N, 3))
        pred_long_std = np.zeros((N, 3))
    feature_list.extend([pred_long_mean, pred_long_std])

    pred_flat = predictions.reshape(N, -1, 3)
    pred_q25 = np.percentile(pred_flat, 25, axis=1)
    pred_q75 = np.percentile(pred_flat, 75, axis=1)
    pred_iqr = pred_q75 - pred_q25
    feature_list.extend([pred_q25, pred_q75, pred_iqr])

    pred_skew = np.zeros((N, 3))
    pred_kurtosis = np.zeros((N, 3))
    for i in range(N):
        for axis in range(3):
            data = pred_flat[i, :, axis]
            pred_skew[i, axis] = stats.skew(data) if np.std(data) > 1e-8 else 0
            pred_kurtosis[i, axis] = stats.kurtosis(data) if np.std(data) > 1e-8 else 0
    feature_list.extend([pred_skew, pred_kurtosis])

    # ========== 特征组2: 输入序列特征 ==========
    input_mean = input_features.mean(axis=(1, 2))
    input_std = input_features.std(axis=(1, 2))
    feature_list.extend([input_mean, input_std])

    input_last1 = input_features[:, :, -1, :].reshape(N, -1)
    input_last3_mean = input_features[:, :, -3:, :].mean(axis=2).reshape(N, -1)
    input_last5_mean = input_features[:, :, -5:, :].mean(axis=2).reshape(N, -1)
    input_last7_mean = input_features[:, :, -7:, :].mean(axis=2).reshape(N, -1)
    input_last10_mean = input_features.mean(axis=2).reshape(N, -1)
    feature_list.extend([input_last1, input_last3_mean, input_last5_mean,
                         input_last7_mean, input_last10_mean])

    input_range = input_features.max(axis=2) - input_features.min(axis=2)
    input_range = input_range.reshape(N, -1)
    feature_list.append(input_range)

    decay_weights = np.exp(np.linspace(-2, 0, input_features.shape[2]))
    decay_weights = decay_weights / decay_weights.sum()
    input_exp_weighted = np.einsum('njta,t->nja', input_features, decay_weights).reshape(N, -1)
    feature_list.append(input_exp_weighted)

    input_first_last_diff = (input_features[:, :, -1, :] - input_features[:, :, 0, :]).reshape(N, -1)
    feature_list.append(input_first_last_diff)

    # ========== 特征组3: 速度特征 ==========
    vel_mean = velocities.mean(axis=(1, 2))
    vel_std = velocities.std(axis=(1, 2))
    feature_list.extend([vel_mean, vel_std])

    vel_last1 = velocities[:, :, -1, :].reshape(N, -1)
    feature_list.append(vel_last1)

    vel_last3_mean = velocities[:, :, -3:, :].mean(axis=2).reshape(N, -1)
    feature_list.append(vel_last3_mean)

    vel_trend_1_3 = (velocities[:, :, -1, :] - velocities[:, :, -3, :]).reshape(N, -1)
    vel_trend_1_5 = (velocities[:, :, -1, :] - velocities[:, :, -5, :]).reshape(N, -1)
    vel_trend_1_first = (velocities[:, :, -1, :] - velocities[:, :, 0, :]).reshape(N, -1)
    feature_list.extend([vel_trend_1_3, vel_trend_1_5, vel_trend_1_first])

    vel_cv = np.zeros((N, 3))
    for i in range(N):
        for axis in range(3):
            mean_val = np.mean(np.abs(velocities[i, :, :, axis]))
            if mean_val > 1e-8:
                vel_cv[i, axis] = np.std(velocities[i, :, :, axis]) / mean_val
    feature_list.append(vel_cv)

    vel_magnitude = np.sqrt(np.sum(velocities[:, :, -1, :] ** 2, axis=2))
    feature_list.append(vel_magnitude)

    # ========== 特征组4: 加速度特征 ==========
    acc_mean = accelerations.mean(axis=(1, 2))
    acc_std = accelerations.std(axis=(1, 2))
    feature_list.extend([acc_mean, acc_std])

    acc_last1 = accelerations[:, :, -1, :].reshape(N, -1)
    acc_last3_mean = accelerations[:, :, -3:, :].mean(axis=2).reshape(N, -1)
    acc_last5_mean = accelerations[:, :, -5:, :].mean(axis=2).reshape(N, -1)
    feature_list.extend([acc_last1, acc_last3_mean, acc_last5_mean])

    acc_trend = (accelerations[:, :, -1, :] - accelerations[:, :, -3, :]).reshape(N, -1)
    feature_list.append(acc_trend)

    acc_magnitude = np.sqrt(np.sum(accelerations[:, :, -1, :] ** 2, axis=2))
    feature_list.append(acc_magnitude)

    jerk = np.zeros_like(accelerations)
    jerk[:, :, 1:, :] = accelerations[:, :, 1:, :] - accelerations[:, :, :-1, :]
    jerk_last = jerk[:, :, -1, :].reshape(N, -1)
    feature_list.append(jerk_last)

    acc_sign_changes = np.sum(np.diff(np.sign(accelerations), axis=2) != 0, axis=2).reshape(N, -1)
    feature_list.append(acc_sign_changes)

    # ========== 特征组5: 关键帧预测特征 ==========
    key_frames = [0, 1, 2, 3, 4, 5, 7, 9]
    for frame_idx in key_frames:
        if frame_idx < predictions.shape[2]:
            pred_frame = predictions[:, :, frame_idx, :].reshape(N, -1)
            feature_list.append(pred_frame)

    frame_pairs = [(0, 1), (1, 2), (1, 3), (3, 7), (7, 9)]
    for f1, f2 in frame_pairs:
        if f2 < predictions.shape[2]:
            pred_delta = (predictions[:, :, f2, :] - predictions[:, :, f1, :]).reshape(N, -1)
            feature_list.append(pred_delta)

    pred_continuity = (predictions[:, :, 0, :] - input_features[:, :, -1, :]).reshape(N, -1)
    feature_list.append(pred_continuity)

    if predictions.shape[2] > 1:
        pred_long_trend = (predictions[:, :, -1, :] - predictions[:, :, 0, :]).reshape(N, -1)
    else:
        pred_long_trend = np.zeros((N, N_FEATURES))
    feature_list.append(pred_long_trend)

    # ========== 特征组6: 关节距离特征 ==========
    for j1, j2 in BONE_CONNECTIONS:
        bone_diff_last1 = input_features[:, j1, -1, :] - input_features[:, j2, -1, :]
        feature_list.append(bone_diff_last1)
        bone_diff_last3 = input_features[:, j1, -3, :] - input_features[:, j2, -3, :]
        feature_list.append(bone_diff_last3)
        bone_diff_last5 = input_features[:, j1, -5, :] - input_features[:, j2, -5, :]
        feature_list.append(bone_diff_last5)

    # ========== 特征组7: 频域特征 ==========
    input_for_fft = input_features.transpose(0, 1, 3, 2)
    fft_features = []
    for j in range(N_JOINTS):
        for axis in range(3):
            signal = input_for_fft[:, j, axis, :]
            fft_result = np.abs(fft(signal, axis=1))
            n_freqs = fft_result.shape[1] // 2

            low_freq = np.sum(fft_result[:, :n_freqs//3] ** 2, axis=1)
            mid_freq = np.sum(fft_result[:, n_freqs//3:2*n_freqs//3] ** 2, axis=1)
            high_freq = np.sum(fft_result[:, 2*n_freqs//3:n_freqs] ** 2, axis=1)
            total_energy = low_freq + mid_freq + high_freq + 1e-8

            fft_features.append(low_freq / total_energy)
            fft_features.append(mid_freq / total_energy)
            fft_features.append(high_freq / total_energy)

            dominant_freq = np.argmax(fft_result[:, 1:n_freqs], axis=1) + 1
            fft_features.append(dominant_freq.astype(np.float32))

    fft_features = np.stack(fft_features, axis=1)
    feature_list.append(fft_features)

    dct_features = []
    for j in range(N_JOINTS):
        for axis in range(3):
            signal = input_for_fft[:, j, axis, :]
            dct_result = dct(signal, type=2, axis=1, norm='ortho')
            dct_features.append(dct_result[:, :3])
    dct_features = np.concatenate(dct_features, axis=1)
    feature_list.append(dct_features)

    # ========== 【已去除】特征组8: 动作语义特征 ==========
    # 不再生成 action one-hot 特征

    # ========== 特征组9: 运动特性特征 ==========
    motion_amplitude_total = np.sum(np.abs(input_first_last_diff.reshape(N, N_JOINTS, 3)), axis=(1, 2))
    feature_list.append(motion_amplitude_total.reshape(N, 1))

    motion_amplitude_per_joint = np.sqrt(np.sum(
        (input_features[:, :, -1, :] - input_features[:, :, 0, :]) ** 2, axis=2))
    feature_list.append(motion_amplitude_per_joint)

    vel_normalized = velocities[:, :, -1, :] / (np.linalg.norm(velocities[:, :, -1, :], axis=2, keepdims=True) + 1e-8)
    motion_complexity = 1 - np.abs(np.mean(vel_normalized, axis=1)).mean(axis=1)
    feature_list.append(motion_complexity.reshape(N, 1))

    pose_diff = np.diff(input_features, axis=2)
    pose_change_rate_per_sample = np.mean(np.sqrt(np.sum(pose_diff ** 2, axis=3)), axis=(1, 2))
    feature_list.append(pose_change_rate_per_sample.reshape(N, 1))

    com_position = input_features[:, 0, -1, :]
    com_velocity = velocities[:, 0, -1, :]
    feature_list.extend([com_position, com_velocity])

    # ========== 特征组10: 预测质量特征 ==========
    pred_diff = np.diff(predictions, axis=2)
    if pred_diff.shape[2] > 1:
        pred_diff2 = np.diff(pred_diff, axis=2)
        pred_smoothness = np.mean(np.sqrt(np.sum(pred_diff2 ** 2, axis=3)), axis=(1, 2))
    else:
        pred_smoothness = np.zeros(N)
    feature_list.append(pred_smoothness.reshape(N, 1))

    bone_length_var = []
    for j1, j2 in BONE_CONNECTIONS[:5]:
        bone_vec = predictions[:, j1, :, :] - predictions[:, j2, :, :]
        bone_lengths = np.sqrt(np.sum(bone_vec ** 2, axis=2))
        bone_length_var.append(np.std(bone_lengths, axis=1))
    bone_length_variance = np.mean(np.stack(bone_length_var, axis=1), axis=1)
    feature_list.append(bone_length_variance.reshape(N, 1))

    continuity_score = np.sqrt(np.sum(pred_continuity.reshape(N, N_JOINTS, 3) ** 2, axis=(1, 2)))
    feature_list.append(continuity_score.reshape(N, 1))

    pred_vel_first = predictions[:, :, 1, :] - predictions[:, :, 0, :] if predictions.shape[2] > 1 else np.zeros((N, N_JOINTS, 3))
    input_vel_last = velocities[:, :, -1, :]
    velocity_reasonability = np.mean(np.sqrt(np.sum((pred_vel_first - input_vel_last) ** 2, axis=2)), axis=1)
    feature_list.append(velocity_reasonability.reshape(N, 1))

    if predictions.shape[2] > 2:
        v1 = predictions[:, :, 1:-1, :] - predictions[:, :, :-2, :]
        v2 = predictions[:, :, 2:, :] - predictions[:, :, 1:-1, :]
        cross = np.cross(v1, v2)
        curvature = np.mean(np.sqrt(np.sum(cross ** 2, axis=3)) / (np.sqrt(np.sum(v1 ** 2, axis=3)) ** 3 + 1e-8), axis=(1, 2))
    else:
        curvature = np.zeros(N)
    feature_list.append(curvature.reshape(N, 1))

    # ========== 特征组11: 关节角度特征 ==========
    angle_features = []
    for j1, j2, j3 in ANGLE_TRIPLETS:
        v1 = input_features[:, j1, :, :] - input_features[:, j2, :, :]
        v2 = input_features[:, j3, :, :] - input_features[:, j2, :, :]

        dot = np.sum(v1 * v2, axis=2)
        norm1 = np.sqrt(np.sum(v1 ** 2, axis=2))
        norm2 = np.sqrt(np.sum(v2 ** 2, axis=2))
        cos_angle = dot / (norm1 * norm2 + 1e-8)
        angles = np.arccos(np.clip(cos_angle, -1, 1))

        angle_features.extend([
            np.mean(angles, axis=1),
            np.std(angles, axis=1),
            np.max(angles, axis=1),
            np.min(angles, axis=1)
        ])

    angle_features = np.stack(angle_features, axis=1)
    feature_list.append(angle_features)

    # ========== 特征组12: 对称性特征 ==========
    symmetry_features = []
    for left, right in SYMMETRIC_PAIRS:
        sym_diff = input_features[:, left, -1, :] - input_features[:, right, -1, :]
        symmetry_features.append(sym_diff)
    symmetry_features = np.concatenate(symmetry_features, axis=1)
    feature_list.append(symmetry_features)

    # ========== 特征组13: 自相关特征 ==========
    autocorr_features = []
    input_flat = input_features.mean(axis=1)
    for lag in [1, 2, 3]:
        for axis in range(3):
            signal = input_flat[:, :, axis]
            autocorr = np.array([np.corrcoef(signal[i, :-lag], signal[i, lag:])[0, 1]
                                if len(signal[i]) > lag else 0 for i in range(N)])
            autocorr = np.nan_to_num(autocorr)
            autocorr_features.append(autocorr)
    autocorr_features = np.stack(autocorr_features, axis=1)
    feature_list.append(autocorr_features)

    # 合并所有特征
    feature_matrix = np.concatenate([f.reshape(N, -1) if len(f.shape) > 1 else f.reshape(N, 1)
                                     for f in feature_list], axis=1)

    # 处理NaN和Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_matrix


def select_features_by_importance(X_train, y_train_flat, X_val, y_val_flat,
                                  lgb_params, sample_weights,
                                  feature_keep_ratio=DEFAULT_FEATURE_KEEP_RATIO,
                                  n_probe_outputs=20,
                                  early_stopping_rounds=50):
    """
    利用LightGBM特征重要性进行特征筛选

    策略：随机选取 n_probe_outputs 个输出维度训练探测模型，
    汇总特征重要性（gain），保留 top feature_keep_ratio 比例的特征。

    Args:
        X_train: 训练特征矩阵
        y_train_flat: 训练残差 (N, n_outputs)
        X_val: 验证特征矩阵
        y_val_flat: 验证残差 (N, n_outputs)
        lgb_params: LightGBM参数
        sample_weights: 样本权重
        feature_keep_ratio: 保留特征比例 (0, 1]
        n_probe_outputs: 用于探测的输出维度数量
        early_stopping_rounds: 早停轮数

    Returns:
        selected_indices: 被选中的特征索引数组
        importance_scores: 各特征的汇总重要性分数
    """
    n_total_features = X_train.shape[1]
    n_outputs = y_train_flat.shape[1]

    print(f"\n  ===== 特征重要性筛选 =====")
    print(f"  总特征数: {n_total_features}")
    print(f"  保留比例: {feature_keep_ratio*100:.0f}%")
    print(f"  探测模型数: {n_probe_outputs}")

    # 随机选取代表性输出维度（均匀分布在关节/帧/坐标上）
    np.random.seed(42)
    probe_indices = np.random.choice(n_outputs, size=min(n_probe_outputs, n_outputs), replace=False)
    probe_indices.sort()

    # 汇总特征重要性
    importance_sum = np.zeros(n_total_features)

    print(f"  正在训练{len(probe_indices)}个探测模型...")
    for idx_count, out_idx in enumerate(probe_indices):
        y_tr = y_train_flat[:, out_idx]
        y_va = y_val_flat[:, out_idx]

        train_data = lgb.Dataset(X_train, label=y_tr, weight=sample_weights, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_va, reference=train_data, free_raw_data=False)

        try:
            booster = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=500,  # 探测阶段用较少轮数
                valid_sets=[val_data],
                valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            # 累加 gain 类型重要性
            imp = booster.feature_importance(importance_type='gain')
            importance_sum += imp
            del booster

        except Exception as e:
            print(f"    探测模型 {idx_count+1} 训练出错: {e}，跳过")
            continue

        if (idx_count + 1) % 5 == 0 or idx_count == len(probe_indices) - 1:
            print(f"    已完成 {idx_count+1}/{len(probe_indices)} 个探测模型")

    # 选取 top-k 特征
    n_keep = max(1, int(n_total_features * feature_keep_ratio))
    selected_indices = np.argsort(importance_sum)[::-1][:n_keep]
    selected_indices.sort()  # 保持原始顺序

    # 统计
    nonzero_count = np.sum(importance_sum > 0)
    print(f"\n  特征筛选结果:")
    print(f"    非零重要性特征数: {nonzero_count}/{n_total_features}")
    print(f"    保留特征数: {n_keep}/{n_total_features} ({n_keep/n_total_features*100:.1f}%)")
    print(f"    筛除特征数: {n_total_features - n_keep}")

    # 显示 top-10 特征
    feature_names = get_feature_names()
    top10_idx = np.argsort(importance_sum)[::-1][:10]
    print(f"\n  Top-10 重要特征:")
    for rank, fi in enumerate(top10_idx):
        fname = feature_names[fi] if fi < len(feature_names) else f"feature_{fi}"
        print(f"    {rank+1:2d}. {fname}: {importance_sum[fi]:.2f}")

    return selected_indices, importance_sum


def create_sample_weights(ground_truth, predictions, train_frames=TRAIN_FRAMES):
    """创建样本权重，对短期误差大的样本增加权重"""
    N = ground_truth.shape[0]

    residuals = ground_truth[:, :, :train_frames, :] - predictions[:, :, :train_frames, :]
    short_term_error = np.sqrt(np.sum(residuals ** 2, axis=(1, 2, 3)))

    weights = 1.0 + short_term_error / (np.mean(short_term_error) + 1e-8)
    weights = weights / np.mean(weights)

    return weights


def calculate_specific_frame_mpjpe(ground_truth, predictions, frame_idx):
    """计算特定帧的MPJPE"""
    if len(ground_truth.shape) == 4:
        gt_frame = ground_truth[:, :, frame_idx, :]
        pred_frame = predictions[:, :, frame_idx, :]
    else:
        gt_frame = ground_truth
        pred_frame = predictions

    joint_errors = np.sqrt(np.sum((gt_frame - pred_frame) ** 2, axis=2))
    mpjpe = np.mean(joint_errors)

    return mpjpe


def train_lightgbm_fusion(model_path, train_data_dir, val_data_dir, lgb_save_path,
                          predict_length=30, batch_size=16, early_stopping_rounds=50,
                          force_cpu=False, cpu_usage_target=0.85,
                          feature_keep_ratio=DEFAULT_FEATURE_KEEP_RATIO):
    """
    训练LightGBM融合模型 - 22关节消融实验版本（只训练前10帧）
    【新增】去除类别特征 + 特征重要性筛选

    Args:
        ... (原有参数)
        feature_keep_ratio: 特征保留比例，默认0.6（保留60%）
    """
    print("=" * 70)
    print("开始训练LightGBM融合模型（消融实验版本：去除掩码和去噪）")
    print("【新增】去除类别特征 + 特征重要性筛选")
    print("=" * 70)
    print(f"训练数据目录: {train_data_dir}")
    print(f"验证数据目录: {val_data_dir}")
    print(f"早停轮数: {early_stopping_rounds}")
    print(f"关节数量: {N_JOINTS}, 特征维度: {N_FEATURES}")
    print(f"训练帧数: {TRAIN_FRAMES}, 输出维度: {TRAIN_OUTPUTS}")
    print(f"特征保留比例: {feature_keep_ratio*100:.0f}%")

    # 1. 加载AuxFormer模型 - 消融实验版本
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print("\n1. 加载AuxFormer模型（消融实验版本：去除掩码和去噪）...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model_config = checkpoint.get('model_config', {})

    model = AuxFormer(
        in_dim=3,
        h_dim=model_config.get('h_dim', 128),
        past_timestep=10,
        future_timestep=30,
        mask_ratio=model_config.get('mask_ratio', 0.4),
        decoder_dim=128,
        num_heads=8,
        encoder_depth=model_config.get('encoder_depth', 6),
        decoder_depth=model_config.get('decoder_depth', 2),
        decoder_dim_per_head=16,
        same_head=False,
        n_agent=model_config.get('n_agent', N_JOINTS),
        add_joint_token=True,
        add_residual=True,
        denoise=False,
        denoise_mode='past',
        noise_dev=model_config.get('noise_dev', 0.6),
        part_noise=False,
        part_noise_ratio=0.3,
        mlp_dim=256,
        dim_per_head=16,
        regular_masking=False,
        range_mask_ratio=False,
        multi_output=True
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  模型加载成功（消融实验版本），使用设备: {device}")

    # 2. 获取scaler
    auxformer_scaler = checkpoint['scaler']
    print(f"  使用AuxFormer checkpoint中的scaler (特征数: {auxformer_scaler.n_features_in_})")

    # 3. 加载数据集
    print("\n2. 加载训练数据...")
    train_dataset = MocapDatasetForLightGBM(
        data_dir=train_data_dir,
        input_n=10,
        output_n=predict_length,
        scaler=auxformer_scaler
    )

    print("\n3. 加载验证数据...")
    val_dataset = MocapDatasetForLightGBM(
        data_dir=val_data_dir,
        input_n=10,
        output_n=predict_length,
        scaler=auxformer_scaler
    )

    print(f"\n数据集信息:")
    print(f"  训练样本数: {len(train_dataset)}")
    print(f"  验证样本数: {len(val_dataset)}")

    # 4. 提取特征
    print("\n4. 提取训练集增强特征...")
    train_preds, train_gt, train_input_feats, train_vel, train_acc, train_actions = extract_model_features_enhanced(
        model, train_dataset, device, batch_size
    )

    print("\n5. 提取验证集增强特征...")
    val_preds, val_gt, val_input_feats, val_vel, val_acc, val_actions = extract_model_features_enhanced(
        model, val_dataset, device, batch_size
    )

    # 5. 计算残差（只计算前10帧）
    print("\n6. 计算残差（前10帧）...")
    train_preds_cut = train_preds[:, :, :TRAIN_FRAMES, :]
    val_preds_cut = val_preds[:, :, :TRAIN_FRAMES, :]

    train_residuals = train_gt[:, :, :TRAIN_FRAMES, :] - train_preds_cut
    val_residuals = val_gt[:, :, :TRAIN_FRAMES, :] - val_preds_cut

    train_residuals_flat = train_residuals.reshape(len(train_residuals), -1)
    val_residuals_flat = val_residuals.reshape(len(val_residuals), -1)

    print(f"  训练残差形状: {train_residuals_flat.shape}")
    print(f"  验证残差形状: {val_residuals_flat.shape}")

    # 6. 构造特征（已去除类别特征）
    print("\n7. 构造LightGBM增强特征矩阵（已去除动作类别特征）...")
    X_train_full = build_lightgbm_features(train_preds, train_input_feats, train_vel, train_acc,
                                            train_actions, predict_length)
    X_val_full = build_lightgbm_features(val_preds, val_input_feats, val_vel, val_acc,
                                          val_actions, predict_length)

    print(f"  训练特征形状（筛选前）: {X_train_full.shape}")
    print(f"  验证特征形状（筛选前）: {X_val_full.shape}")

    feature_names = get_feature_names()
    print(f"  特征名称数量: {len(feature_names)}")
    if X_train_full.shape[1] != len(feature_names):
        print(f"  ⚠ 警告: 特征维度({X_train_full.shape[1]})与特征名称数量({len(feature_names)})不一致")

    # 7. 创建样本权重
    print("\n8. 创建样本权重...")
    sample_weights = create_sample_weights(train_gt, train_preds, TRAIN_FRAMES)
    print(f"  样本权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    # 8. 配置参数 - GPU检测
    print("\n9. 配置LightGBM训练参数...")

    use_gpu = False
    gpu_device = None
    gpu_params = None

    if not force_cpu:
        use_gpu, gpu_device, gpu_params = check_lightgbm_gpu_support()

    total_threads = multiprocessing.cpu_count()
    optimal_threads = get_optimal_thread_count(cpu_usage_target)

    print(f"\n系统信息:")
    print(f"  CPU总线程数: {total_threads}")
    print(f"  目标CPU使用率: {cpu_usage_target*100:.0f}%")
    print(f"  使用线程数: {optimal_threads}")

    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1,
        'force_col_wise': True,
    }

    device_used = 'cpu'

    if use_gpu and gpu_params:
        lgb_params.update(gpu_params)
        device_used = gpu_device
        print(f"\n✓ 将使用 {gpu_device.upper()} 加速训练")
    else:
        lgb_params['num_threads'] = optimal_threads
        lgb_params['n_jobs'] = optimal_threads
        print(f"\n将使用CPU训练（{optimal_threads}线程，约{cpu_usage_target*100:.0f}%使用率）")

    print(f"\n最终配置:")
    print(f"  device: {device_used}")
    print(f"  num_leaves: {lgb_params['num_leaves']}")
    print(f"  learning_rate: {lgb_params['learning_rate']}")
    print(f"  early_stopping_rounds: {early_stopping_rounds}")

    # ==================== 9. 特征重要性筛选 ====================
    print("\n10. 特征重要性筛选...")
    selected_indices, importance_scores = select_features_by_importance(
        X_train_full, train_residuals_flat,
        X_val_full, val_residuals_flat,
        lgb_params, sample_weights,
        feature_keep_ratio=feature_keep_ratio,
        n_probe_outputs=20,
        early_stopping_rounds=early_stopping_rounds
    )

    # 应用特征筛选
    X_train = X_train_full[:, selected_indices]
    X_val = X_val_full[:, selected_indices]

    print(f"\n  筛选后训练特征形状: {X_train.shape}")
    print(f"  筛选后验证特征形状: {X_val.shape}")

    # 10. 训练模型（只训练前10帧，使用筛选后的特征）
    print("\n11. 训练LightGBM模型（前10帧，使用筛选后特征）...")

    n_outputs = TRAIN_OUTPUTS  # 660
    print(f"  输出维度: {n_outputs} (22关节 × {TRAIN_FRAMES}帧 × 3坐标)")

    lgb_models = []
    training_times = []
    start_time_total = time.time()

    max_iterations = 1000
    batch_outputs = N_FEATURES

    for i in range(0, n_outputs, batch_outputs):
        end_idx = min(i + batch_outputs, n_outputs)
        print(f"\n  训练输出维度 {i+1}-{end_idx}/{n_outputs}...")

        for j in range(i, end_idx):
            y_train = train_residuals_flat[:, j]
            y_val = val_residuals_flat[:, j]

            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                weight=sample_weights,
                free_raw_data=False
            )
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                free_raw_data=False
            )

            start_time = time.time()

            try:
                booster = lgb.train(
                    lgb_params,
                    train_data,
                    num_boost_round=max_iterations,
                    valid_sets=[val_data],
                    valid_names=['valid'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                lgb_models.append(booster)

            except Exception as e:
                if use_gpu:
                    print(f"    ⚠ GPU训练出错，切换到CPU...")
                    cpu_params = lgb_params.copy()
                    cpu_params['device'] = 'cpu'
                    cpu_params.pop('gpu_platform_id', None)
                    cpu_params.pop('gpu_device_id', None)
                    cpu_params['num_threads'] = optimal_threads
                    cpu_params['n_jobs'] = optimal_threads

                    booster = lgb.train(
                        cpu_params,
                        train_data,
                        num_boost_round=max_iterations,
                        valid_sets=[val_data],
                        valid_names=['valid'],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                            lgb.log_evaluation(period=0)
                        ]
                    )
                    lgb_models.append(booster)
                    device_used = 'cpu'
                    use_gpu = False
                else:
                    raise e

            training_times.append(time.time() - start_time)

        avg_time = np.mean(training_times[-min(batch_outputs, len(training_times)):])
        remaining = (n_outputs - end_idx) * avg_time / 60
        print(f"    平均训练时间: {avg_time:.3f}秒/维度, 预计剩余: {remaining:.1f}分钟")

    total_training_time = time.time() - start_time_total
    print(f"\n✓ 训练完成，总耗时: {total_training_time:.2f}秒 ({total_training_time/60:.2f}分钟)")

    # 11. 验证集评估
    print("\n12. 验证集评估（特定帧MPJPE）...")

    val_residuals_pred = np.zeros_like(val_residuals_flat)
    for j, booster in enumerate(lgb_models):
        val_residuals_pred[:, j] = booster.predict(X_val)

    val_residuals_pred = val_residuals_pred.reshape(-1, N_JOINTS, TRAIN_FRAMES, 3)
    val_fusion_pred = val_preds_cut + val_residuals_pred

    from sklearn.metrics import mean_absolute_error

    mae_auxformer = mean_absolute_error(val_gt[:, :, :TRAIN_FRAMES, :].flatten(), val_preds_cut.flatten())
    mae_fusion = mean_absolute_error(val_gt[:, :, :TRAIN_FRAMES, :].flatten(), val_fusion_pred.flatten())

    print(f"\n验证集总体评估结果（前{TRAIN_FRAMES}帧）:")
    print(f"  AuxFormer MAE（消融版本）: {mae_auxformer:.6f}")
    print(f"  LightGBM融合 MAE: {mae_fusion:.6f}")
    improvement = ((mae_auxformer - mae_fusion) / mae_auxformer * 100)
    print(f"  改进: {improvement:.2f}%")

    # 特定帧MPJPE评估
    print(f"\n特定帧MPJPE评估:")
    print("-" * 60)
    target_frames = [2, 4, 8, 10]
    frame_results = {}

    for frame in target_frames:
        if frame <= TRAIN_FRAMES:
            frame_idx = frame - 1

            auxformer_mpjpe = calculate_specific_frame_mpjpe(val_gt[:, :, :TRAIN_FRAMES, :], val_preds_cut, frame_idx)
            fusion_mpjpe = calculate_specific_frame_mpjpe(val_gt[:, :, :TRAIN_FRAMES, :], val_fusion_pred, frame_idx)

            improvement_pct = ((auxformer_mpjpe - fusion_mpjpe) / auxformer_mpjpe * 100)

            frame_results[frame] = {
                'auxformer': auxformer_mpjpe,
                'fusion': fusion_mpjpe,
                'improvement': improvement_pct
            }

            print(f"  第{frame}帧 MPJPE:")
            print(f"    AuxFormer（消融）: {auxformer_mpjpe:.4f}")
            print(f"    融合模型:  {fusion_mpjpe:.4f}")
            print(f"    改进:      {improvement_pct:.2f}%")

    # 12. 保存模型
    print(f"\n13. 保存LightGBM模型...")
    os.makedirs(os.path.dirname(lgb_save_path) if os.path.dirname(lgb_save_path) else '.', exist_ok=True)

    fusion_checkpoint = {
        'lgb_models': lgb_models,
        'scaler': auxformer_scaler,
        'predict_length': predict_length,
        'train_frames': TRAIN_FRAMES,
        'train_mae_auxformer': mae_auxformer,
        'train_mae_fusion': mae_fusion,
        # ==================== 新增：特征筛选信息 ====================
        'selected_feature_indices': selected_indices,
        'feature_keep_ratio': feature_keep_ratio,
        'importance_scores': importance_scores,
        'n_total_features_before_selection': X_train_full.shape[1],
        'n_selected_features': len(selected_indices),
        # ============================================================
        'feature_info': {
            'feature_dim': X_train.shape[1],  # 筛选后的维度
            'feature_dim_before_selection': X_train_full.shape[1],  # 筛选前维度
            'output_dim': n_outputs,
            'n_joints': N_JOINTS,
            'n_features': N_FEATURES,
            'train_frames': TRAIN_FRAMES
        },
        'device_used': device_used,
        'training_time': total_training_time,
        'model_type': 'lightgbm_enhanced_10frames_no_mask_denoise_feature_selected',
        'early_stopping_rounds': early_stopping_rounds,
        'frame_results': frame_results,
        'cpu_threads_used': optimal_threads if device_used == 'cpu' else None,
        'feature_categories': [
            'pred_statistics', 'input_sequence', 'velocity', 'acceleration',
            'key_frames', 'bone_distances', 'frequency_domain',
            # 'action_semantic',  # 已去除
            'motion_characteristics', 'prediction_quality', 'joint_angles',
            'symmetry', 'autocorrelation'
        ],
        'ablation_type': 'no_mask_no_denoise',
        'removed_features': ['action_semantic (one-hot)'],
    }

    with open(lgb_save_path, 'wb') as f:
        pickle.dump(fusion_checkpoint, f)

    print(f"✓ LightGBM模型已保存: {lgb_save_path}")
    print(f"  筛选前特征数: {X_train_full.shape[1]}")
    print(f"  筛选后特征数: {X_train.shape[1]} ({feature_keep_ratio*100:.0f}%)")
    print("=" * 70)

    return lgb_save_path


if __name__ == "__main__":
    # 消融实验版本路径
    model_path = "models/mocap_model_22joints_no_mask_denoise.pth"
    train_data_dir = "dataset_22joints/train_data"
    val_data_dir = "dataset_22joints/val_data"
    lgb_save_path = "models/lightgbm_fusion_22joints_no_mask_denoise.pkl"

    train_lightgbm_fusion(
        model_path=model_path,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        lgb_save_path=lgb_save_path,
        predict_length=30,
        batch_size=16,
        early_stopping_rounds=50,
        force_cpu=False,
        cpu_usage_target=0.85,
        feature_keep_ratio=0.6  # 保留60%特征，可修改
    )