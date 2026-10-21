"""

使用LightGBM融合模型进行预测和评估 - 22关节预测任务版本（仅使用预测损失）
按15种动作类别评估第2、4、8、10帧的MPJPE，包含提升百分比

修改内容：
1. 去除类别特征（动作语义one-hot编码）
2. 加载训练时保存的特征筛选索引，预测时自动应用同样的特征筛选
3. 保留原有全部功能

特性：
1. 与原Transformer评估规则完全一致
2. 每个文件提取256个随机子序列（不重复）
3. 子序列长度40帧（前10帧输入，后30帧输出）
4. 按动作分类评估第2、4、8、10帧的MPJPE
5. 显示每种动作和平均的精度提升百分比
6. 增强特征工程与训练保持一致（使用统一的特征构建函数，已去除类别特征）
7. 只对前10帧进行LightGBM修正
8. 适配预测任务版本Transformer（仅使用预测损失）
9. 自动加载并应用训练时的特征重要性筛选索引
"""
import sys
import torch
import numpy as np
import pandas as pd
import os
import pickle
import glob
import random
import multiprocessing
from model import Transformer
from sklearn.metrics import mean_absolute_error
from scipy import stats
from scipy.fft import fft, dct

# 22关节配置
N_JOINTS = 22
N_FEATURES = 66

# 只预测前10帧
TRAIN_FRAMES = 10

# 评估帧配置
EVAL_FRAMES = [1, 3, 7, 9]  # 0-indexed: 第2、4、8、10帧
EVAL_FRAME_LABELS = [2, 4, 8, 10]

# 子序列参数
SUBSEQUENCE_LENGTH = 40
INPUT_LENGTH = 10
OUTPUT_LENGTH = 30
NUM_SUBSEQUENCES = 256

# 15种动作类别（保留用于数据解析，但不再作为特征）
ACTION_NAMES = [
    'directions', 'discussion', 'eating', 'greeting', 'phoning',
    'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',
    'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether'
]

# 骨骼连接定义
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


def get_optimal_thread_count(target_usage=0.85):
    total_threads = multiprocessing.cpu_count()
    optimal_threads = max(1, int(total_threads * target_usage))
    return optimal_threads


def get_action_from_filename(filename):
    """从文件名精确解析动作类型"""
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


def extract_random_subsequences(sequence_data, num_subsequences=256, subseq_length=40, seed=None):
    """从序列中随机提取不重复的子序列"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    total_length = len(sequence_data)

    if total_length < subseq_length:
        raise ValueError(f"序列长度({total_length})不足以提取长度为{subseq_length}的子序列")

    max_start_idx = total_length - subseq_length + 1

    if max_start_idx < num_subsequences:
        print(f"  警告: 可用的子序列起始位置({max_start_idx})少于需要的数量({num_subsequences})")
        num_subsequences = max_start_idx

    start_indices = random.sample(range(max_start_idx), num_subsequences)

    subsequences = []
    for start_idx in start_indices:
        subseq = sequence_data[start_idx:start_idx + subseq_length]
        subsequences.append(subseq)

    return subsequences


def preprocess_single_file(file_path, scaler):
    """预处理单个Excel文件"""
    try:
        df = pd.read_excel(file_path)

        if df.shape[1] < N_FEATURES:
            raise ValueError(f"数据列数不足，需要至少{N_FEATURES}列")

        position_data = df.iloc[:, :N_FEATURES].values
        full_data = np.nan_to_num(position_data, nan=0.0, posinf=0.0, neginf=0.0)
        full_data = full_data / 1000.0

        expected_features = scaler.n_features_in_
        if full_data.shape[1] != expected_features:
            if full_data.shape[1] > expected_features:
                full_data = full_data[:, :expected_features]
            else:
                padding = np.zeros((full_data.shape[0], expected_features - full_data.shape[1]))
                full_data = np.concatenate([full_data, padding], axis=1)

        normalized_data = scaler.transform(full_data)
        position_normalized = normalized_data[:, :N_FEATURES]
        sequence_data = position_normalized.reshape(-1, N_JOINTS, 3)

        return sequence_data

    except Exception as e:
        print(f"  处理失败: {e}")
        return None


def build_lightgbm_features(predictions, input_features, velocities, accelerations,
                            actions, predict_length):
    """
    构造LightGBM的增强特征矩阵 - 统一的特征构建函数
    【已修改】去除了动作类别one-hot特征（特征组8）
    训练和预测都使用这个函数，确保特征一致性

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

    # ========== 特征组1: Transformer预测统计特征 ==========
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


def predict_subsequences_with_lightgbm(transformer_model, lgb_models, subsequences, scaler,
                                        lgb_train_length, train_frames, action, device,
                                        selected_feature_indices=None):
    """
    对多个子序列使用Transformer + LightGBM融合模型进行批量预测
    只对前train_frames帧进行LightGBM修正
    【新增】支持特征筛选索引
    """
    TRAINED_OUTPUT_FRAMES = 30

    fusion_predictions = []
    transformer_predictions = []
    ground_truths = []

    transformer_model.eval()

    for subseq in subsequences:
        input_data = subseq[:INPUT_LENGTH]
        output_data = subseq[INPUT_LENGTH:]

        input_tensor = torch.FloatTensor(input_data).transpose(0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        fake_future = torch.zeros(1, N_JOINTS, TRAINED_OUTPUT_FRAMES, 3, device=device)
        all_traj = torch.cat([input_tensor, fake_future], dim=2)

        with torch.no_grad():
            transformer_pred = transformer_model.predict(all_traj)

        transformer_pred = transformer_pred.squeeze(0).transpose(0, 1).cpu().numpy()

        input_data_np = input_data
        velocity = np.zeros_like(input_data_np)
        velocity[1:] = input_data_np[1:] - input_data_np[:-1]
        velocity[0] = velocity[1]

        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        acceleration[0] = acceleration[1]

        # 转换维度为 (1, joints, frames, 3)
        transformer_pred_reshaped = transformer_pred.transpose(1, 0, 2)[np.newaxis, ...]
        input_data_reshaped = input_data.transpose(1, 0, 2)[np.newaxis, ...]
        velocity_reshaped = velocity.transpose(1, 0, 2)[np.newaxis, ...]
        acceleration_reshaped = acceleration.transpose(1, 0, 2)[np.newaxis, ...]

        # 使用统一的特征构建函数（已去除类别特征）
        X_features = build_lightgbm_features(
            transformer_pred_reshaped, input_data_reshaped, velocity_reshaped,
            acceleration_reshaped, action, lgb_train_length
        )

        # 【新增】应用特征筛选
        if selected_feature_indices is not None:
            X_features = X_features[:, selected_feature_indices]

        # LightGBM只预测前train_frames帧
        n_outputs = N_JOINTS * train_frames * 3
        residual_pred_flat = np.zeros((1, n_outputs))

        for j, booster in enumerate(lgb_models):
            residual_pred_flat[0, j] = booster.predict(X_features)[0]

        residual_pred = residual_pred_flat.reshape(1, N_JOINTS, train_frames, 3)[0]
        residual_pred = residual_pred.transpose(1, 0, 2)

        # 融合预测（只修正前train_frames帧）
        fusion_pred = transformer_pred.copy()
        fusion_pred[:train_frames] = transformer_pred[:train_frames] + residual_pred

        def denormalize_prediction(pred, length):
            pred_flat = pred.reshape(length, N_FEATURES)
            expected_features = scaler.n_features_in_

            if expected_features > N_FEATURES:
                zero_padding = np.zeros((length, expected_features - N_FEATURES))
                pred_full = np.concatenate([pred_flat, zero_padding], axis=1)
            else:
                pred_full = pred_flat

            pred_original = scaler.inverse_transform(pred_full)
            pred_original = pred_original[:, :N_FEATURES]
            pred_original = pred_original * 1000.0

            return pred_original

        fusion_pred_mm = denormalize_prediction(fusion_pred[:OUTPUT_LENGTH], OUTPUT_LENGTH)
        transformer_pred_mm = denormalize_prediction(transformer_pred[:OUTPUT_LENGTH], OUTPUT_LENGTH)
        output_original_mm = denormalize_prediction(output_data, OUTPUT_LENGTH)

        fusion_predictions.append(fusion_pred_mm)
        transformer_predictions.append(transformer_pred_mm)
        ground_truths.append(output_original_mm)

    return fusion_predictions, transformer_predictions, ground_truths


def calculate_mpjpe_per_frame(y_true, y_pred):
    """计算每一帧的MPJPE"""
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(-1, N_JOINTS, 3)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.reshape(-1, N_JOINTS, 3)

    joint_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2))
    mpjpe_per_frame = np.mean(joint_errors, axis=1)

    return mpjpe_per_frame


def evaluate_specific_frames(fusion_preds, transformer_preds, ground_truths, eval_frames=EVAL_FRAMES):
    """评估特定帧的MPJPE"""
    frame_results = {label: {'fusion': [], 'transformer': [], 'improvement': []}
                     for label in EVAL_FRAME_LABELS}

    for fusion_pred, transformer_pred, gt in zip(fusion_preds, transformer_preds, ground_truths):
        mpjpe_fusion = calculate_mpjpe_per_frame(gt, fusion_pred)
        mpjpe_transformer = calculate_mpjpe_per_frame(gt, transformer_pred)

        for i, frame_idx in enumerate(eval_frames):
            if frame_idx < len(mpjpe_fusion):
                frame_label = EVAL_FRAME_LABELS[i]

                fusion_mpjpe = mpjpe_fusion[frame_idx]
                transformer_mpjpe = mpjpe_transformer[frame_idx]

                if transformer_mpjpe > 0:
                    improvement = ((transformer_mpjpe - fusion_mpjpe) / transformer_mpjpe * 100)
                else:
                    improvement = 0.0

                frame_results[frame_label]['fusion'].append(fusion_mpjpe)
                frame_results[frame_label]['transformer'].append(transformer_mpjpe)
                frame_results[frame_label]['improvement'].append(improvement)

    return frame_results


def predict_and_evaluate_by_action_lightgbm(transformer_model_path, lgb_model_path, data_dir, output_dir,
                                             cpu_usage_target=0.85):
    """
    按动作类别评估LightGBM融合模型 - 预测任务版本
    【新增】自动加载并应用训练时的特征筛选索引
    """
    print("=" * 100)
    print("LightGBM融合模型 - 22关节动作预测评估报告（预测任务版本）")
    print("【已去除类别特征 + 特征重要性筛选】")
    print(f"每个文件提取{NUM_SUBSEQUENCES}个随机子序列（与原Transformer评估规则一致）")
    print(f"子序列长度: {SUBSEQUENCE_LENGTH}帧（前{INPUT_LENGTH}帧输入，后{OUTPUT_LENGTH}帧输出）")
    print(f"评估帧: 第{EVAL_FRAME_LABELS}帧的MPJPE（含提升百分比）")
    print("=" * 100)

    optimal_threads = get_optimal_thread_count(cpu_usage_target)
    print(f"CPU线程限制: {optimal_threads}")

    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return None

    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    if not excel_files:
        print(f"在 {data_dir} 中没有找到Excel文件")
        return None

    print(f"测试文件数: {len(excel_files)}")

    # 加载Transformer模型 - 预测任务版本
    print("\n加载Transformer模型（预测任务版本）...")
    checkpoint = torch.load(transformer_model_path, map_location='cpu', weights_only=False)

    model_config = checkpoint.get('model_config', {})

    transformer_model = Transformer(
        in_dim=3,
        h_dim=model_config.get('h_dim', 128),
        past_timestep=10,
        future_timestep=30,
        context_ratio=model_config.get('context_ratio', 0.4),
        decoder_dim=128,
        num_heads=8,
        encoder_depth=model_config.get('encoder_depth', 6),
        decoder_depth=model_config.get('decoder_depth', 2),
        decoder_dim_per_head=16,
        same_head=False,
        n_agent=model_config.get('n_agent', N_JOINTS),
        add_joint_token=True,
        add_residual=True,
        task_flag=False,
        history_setting='past',
        offset_scale=model_config.get('offset_scale', 0.6),
        partial_offset=False,
        partial_offset_ratio=0.3,
        mlp_dim=256,
        dim_per_head=16,
        regular_context=False,
        range_context_ratio=False,
        multi_output=True
    )

    transformer_model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model = transformer_model.to(device)
    print(f"✓ Transformer模型加载成功（预测任务版本），设备: {device}")

    # 加载LightGBM模型
    print("加载LightGBM融合模型...")
    with open(lgb_model_path, 'rb') as f:
        fusion_checkpoint = pickle.load(f)

    lgb_models = fusion_checkpoint['lgb_models']
    scaler = fusion_checkpoint['scaler']
    lgb_train_length = fusion_checkpoint.get('predict_length', 30)
    train_frames = fusion_checkpoint.get('train_frames', TRAIN_FRAMES)

    # 【新增】加载特征筛选索引
    selected_feature_indices = fusion_checkpoint.get('selected_feature_indices', None)
    feature_keep_ratio = fusion_checkpoint.get('feature_keep_ratio', None)
    n_selected = fusion_checkpoint.get('n_selected_features', None)
    n_total_before = fusion_checkpoint.get('n_total_features_before_selection', None)

    print(f"✓ LightGBM模型加载成功 ({len(lgb_models)}个子模型，修正前{train_frames}帧)")
    print(f"✓ 使用checkpoint中的scaler (特征数: {scaler.n_features_in_})")

    if selected_feature_indices is not None:
        print(f"✓ 已加载特征筛选索引: 保留{n_selected}/{n_total_before}个特征 "
              f"({feature_keep_ratio*100:.0f}%)")
    else:
        print("  注意: 未找到特征筛选索引，将使用全部特征")

    os.makedirs(output_dir, exist_ok=True)

    action_results = {action: {f: {'transformer': [], 'fusion': [], 'improvement': []} for f in EVAL_FRAME_LABELS}
                      for action in ACTION_NAMES}
    action_file_counts = {action: 0 for action in ACTION_NAMES}
    action_subseq_counts = {action: 0 for action in ACTION_NAMES}

    successful_predictions = 0
    failed_predictions = 0

    for i, excel_file in enumerate(excel_files, 1):
        print(f"\n[{i}/{len(excel_files)}] 处理文件: {excel_file}")

        action = get_action_from_filename(excel_file)
        file_path = os.path.join(data_dir, excel_file)

        try:
            sequence_data = preprocess_single_file(file_path, scaler)

            if sequence_data is None:
                failed_predictions += 1
                print(f"  ✗ 预处理失败")
                continue

            if len(sequence_data) < SUBSEQUENCE_LENGTH:
                failed_predictions += 1
                print(f"  ✗ 序列长度不足: {len(sequence_data)} < {SUBSEQUENCE_LENGTH}")
                continue

            try:
                subsequences = extract_random_subsequences(
                    sequence_data,
                    num_subsequences=NUM_SUBSEQUENCES,
                    subseq_length=SUBSEQUENCE_LENGTH,
                    seed=42 + i
                )
            except Exception as e:
                failed_predictions += 1
                print(f"  ✗ 子序列提取失败: {e}")
                continue

            print(f"  提取了{len(subsequences)}个子序列")

            try:
                fusion_preds, transformer_preds, ground_truths = predict_subsequences_with_lightgbm(
                    transformer_model, lgb_models, subsequences, scaler, lgb_train_length,
                    train_frames, action, device,
                    selected_feature_indices=selected_feature_indices  # 【新增】传入筛选索引
                )
            except Exception as e:
                failed_predictions += 1
                print(f"  ✗ 预测失败: {e}")
                import traceback
                traceback.print_exc()
                continue

            frame_results = evaluate_specific_frames(fusion_preds, transformer_preds, ground_truths)

            for frame_label, result in frame_results.items():
                action_results[action][frame_label]['fusion'].extend(result['fusion'])
                action_results[action][frame_label]['transformer'].extend(result['transformer'])
                action_results[action][frame_label]['improvement'].extend(result['improvement'])

            successful_predictions += 1
            action_file_counts[action] += 1
            action_subseq_counts[action] += len(subsequences)

            print(f"  ✓ 动作: {action}")
            print(f"    ", end="")
            for fl in EVAL_FRAME_LABELS:
                if frame_results[fl]['fusion']:
                    avg_fusion = np.mean(frame_results[fl]['fusion'])
                    avg_transformer = np.mean(frame_results[fl]['transformer'])
                    avg_improvement = np.mean(frame_results[fl]['improvement'])
                    print(f"第{fl}帧: {avg_transformer:.2f}→{avg_fusion:.2f}mm ({avg_improvement:+.2f}%) ", end="")
            print()

        except Exception as e:
            failed_predictions += 1
            print(f"  ✗ 处理失败: {e}")

    # 生成评估报告
    print("\n" + "=" * 100)
    print("评估结果汇总 - 各动作第2、4、8、10帧的MPJPE (mm)")
    print("Transformer（预测任务版本）vs LightGBM融合模型（含精度提升百分比）")
    if selected_feature_indices is not None:
        print(f"【特征筛选: 保留{n_selected}/{n_total_before}个特征 ({feature_keep_ratio*100:.0f}%)】")
    print("=" * 100)

    results_data = []
    all_frame_results = {f: {'transformer': [], 'fusion': [], 'improvement': []} for f in EVAL_FRAME_LABELS}

    header = f"{'动作':<18}{'文件数':<6}{'子序列数':<10}"
    for fl in EVAL_FRAME_LABELS:
        header += f"第{fl}帧(Transformer)".center(18)
        header += f"第{fl}帧(融合)".center(12)
        header += f"提升%".center(8)
    print(header)
    print("-" * 150)

    for action in ACTION_NAMES:
        row_data = {'动作': action, '文件数': action_file_counts[action], '子序列数': action_subseq_counts[action]}
        row_str = f"{action:<18}{action_file_counts[action]:<6}{action_subseq_counts[action]:<10}"

        for fl in EVAL_FRAME_LABELS:
            transformer_list = action_results[action][fl]['transformer']
            fusion_list = action_results[action][fl]['fusion']
            improvement_list = action_results[action][fl]['improvement']

            if transformer_list:
                avg_transformer = np.mean(transformer_list)
                avg_fusion = np.mean(fusion_list)
                avg_improvement = np.mean(improvement_list)

                row_str += f"{avg_transformer:>16.2f}  "
                row_str += f"{avg_fusion:>10.2f}  "
                row_str += f"{avg_improvement:>6.2f}  "

                row_data[f'第{fl}帧_Transformer'] = avg_transformer
                row_data[f'第{fl}帧_融合'] = avg_fusion
                row_data[f'第{fl}帧_提升%'] = avg_improvement

                all_frame_results[fl]['transformer'].extend(transformer_list)
                all_frame_results[fl]['fusion'].extend(fusion_list)
                all_frame_results[fl]['improvement'].extend(improvement_list)
            else:
                row_str += f"{'N/A':>16}  "
                row_str += f"{'N/A':>10}  "
                row_str += f"{'N/A':>6}  "
                row_data[f'第{fl}帧_Transformer'] = np.nan
                row_data[f'第{fl}帧_融合'] = np.nan
                row_data[f'第{fl}帧_提升%'] = np.nan

        print(row_str)
        results_data.append(row_data)

    print("-" * 150)
    total_subseqs = sum(action_subseq_counts.values())
    avg_row = {'动作': '平均', '文件数': successful_predictions, '子序列数': total_subseqs}
    avg_str = f"{'平均':<18}{successful_predictions:<6}{total_subseqs:<10}"

    for fl in EVAL_FRAME_LABELS:
        if all_frame_results[fl]['transformer']:
            avg_transformer = np.mean(all_frame_results[fl]['transformer'])
            avg_fusion = np.mean(all_frame_results[fl]['fusion'])
            avg_improvement = np.mean(all_frame_results[fl]['improvement'])

            avg_str += f"{avg_transformer:>16.2f}  "
            avg_str += f"{avg_fusion:>10.2f}  "
            avg_str += f"{avg_improvement:>6.2f}  "

            avg_row[f'第{fl}帧_Transformer'] = avg_transformer
            avg_row[f'第{fl}帧_融合'] = avg_fusion
            avg_row[f'第{fl}帧_提升%'] = avg_improvement
        else:
            avg_str += f"{'N/A':>16}  "
            avg_str += f"{'N/A':>10}  "
            avg_str += f"{'N/A':>6}  "
            avg_row[f'第{fl}帧_Transformer'] = np.nan
            avg_row[f'第{fl}帧_融合'] = np.nan
            avg_row[f'第{fl}帧_提升%'] = np.nan

    print(avg_str)
    results_data.append(avg_row)

    print("=" * 150)
    print(f"成功预测: {successful_predictions}/{len(excel_files)} 个文件")
    print(f"失败预测: {failed_predictions} 个文件")
    print(f"总子序列数: {total_subseqs}")

    print("\n各动作子序列数量:")
    for action in ACTION_NAMES:
        if action_subseq_counts[action] > 0:
            print(f"  {action}: {action_subseq_counts[action]}个子序列（{action_file_counts[action]}个文件 × {NUM_SUBSEQUENCES}）")

    # 保存结果
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, 'mpjpe_by_action_frames_lightgbm_prediction.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {csv_path}")

    report_path = os.path.join(output_dir, 'evaluation_report_lightgbm_prediction.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("LightGBM融合模型 - 22关节动作预测评估报告\n")
        f.write("（预测任务：仅使用预测损失，与原Transformer评估规则完全一致：256个随机子序列采样）\n")
        f.write("【已去除类别特征 + 特征重要性筛选】\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"评估方法: 每个文件提取{NUM_SUBSEQUENCES}个随机子序列\n")
        f.write(f"子序列长度: {SUBSEQUENCE_LENGTH}帧（前{INPUT_LENGTH}帧输入，后{OUTPUT_LENGTH}帧输出）\n")
        f.write(f"LightGBM修正帧数: 前{train_frames}帧\n")
        f.write(f"评估帧: 第{EVAL_FRAME_LABELS}帧\n")
        if selected_feature_indices is not None:
            f.write(f"特征筛选: 保留{n_selected}/{n_total_before}个特征 ({feature_keep_ratio*100:.0f}%)\n")
        f.write(f"测试文件数: {len(excel_files)}\n")
        f.write(f"成功预测: {successful_predictions}\n")
        f.write(f"失败预测: {failed_predictions}\n")
        f.write(f"总子序列数: {total_subseqs}\n\n")

        f.write("各动作第2、4、8、10帧的MPJPE (mm) - Transformer（预测）vs LightGBM融合\n")
        f.write("-" * 100 + "\n")

        f.write(f"{'动作':<18}{'文件数':<6}{'子序列数':<10}")
        for fl in EVAL_FRAME_LABELS:
            f.write(f"第{fl}帧(Transformer)".center(18))
            f.write(f"第{fl}帧(融合)".center(12))
            f.write(f"提升%".center(8))
        f.write("\n")
        f.write("-" * 100 + "\n")

        for action in ACTION_NAMES:
            f.write(f"{action:<18}{action_file_counts[action]:<6}{action_subseq_counts[action]:<10}")
            for fl in EVAL_FRAME_LABELS:
                transformer_list = action_results[action][fl]['transformer']
                if transformer_list:
                    f.write(f"{np.mean(transformer_list):>16.2f}  ")
                    f.write(f"{np.mean(action_results[action][fl]['fusion']):>10.2f}  ")
                    f.write(f"{np.mean(action_results[action][fl]['improvement']):>6.2f}  ")
                else:
                    f.write(f"{'N/A':>16}  ")
                    f.write(f"{'N/A':>10}  ")
                    f.write(f"{'N/A':>6}  ")
            f.write("\n")

        f.write("-" * 100 + "\n")
        f.write(f"{'平均':<18}{successful_predictions:<6}{total_subseqs:<10}")
        for fl in EVAL_FRAME_LABELS:
            if all_frame_results[fl]['transformer']:
                f.write(f"{np.mean(all_frame_results[fl]['transformer']):>16.2f}  ")
                f.write(f"{np.mean(all_frame_results[fl]['fusion']):>10.2f}  ")
                f.write(f"{np.mean(all_frame_results[fl]['improvement']):>6.2f}  ")
            else:
                f.write(f"{'N/A':>16}  ")
                f.write(f"{'N/A':>10}  ")
                f.write(f"{'N/A':>6}  ")
        f.write("\n")
        f.write("=" * 100 + "\n")

    print(f"详细报告已保存到: {report_path}")

    return {
        'action_results': action_results,
        'all_frame_results': all_frame_results,
        'successful_predictions': successful_predictions,
        'failed_predictions': failed_predictions,
        'action_file_counts': action_file_counts,
        'action_subseq_counts': action_subseq_counts
    }


if __name__ == "__main__":
    # 预测任务版本路径
    model_path = "models/mocap_model_22joints_prediction.pth"
    lgb_model_path = "models/lightgbm_fusion_22joints_prediction.pkl"
    test_data_dir = "dataset_22joints/test_data"
    output_dir = "evaluation_results_prediction/lightgbm_fusion"

    predict_and_evaluate_by_action_lightgbm(
        transformer_model_path=model_path,
        lgb_model_path=lgb_model_path,
        data_dir=test_data_dir,
        output_dir=output_dir,
        cpu_usage_target=0.85
    )
