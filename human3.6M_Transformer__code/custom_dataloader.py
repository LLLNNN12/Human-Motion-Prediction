"""
自定义动捕数据加载器 - 22关节版本
用于预测任务
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
from sklearn.preprocessing import StandardScaler


class MocapDataset(Dataset):
    def __init__(self, data_dir, input_n=10, output_n=30, split='train', scale=1000,
                 use_angles=False, scaler=None):
        """
        自定义动捕数据加载器 - 修改为22个关节(66列)

        Args:
            data_dir: Excel文件所在目录
            input_n: 输入序列长度(用于预测的历史帧数)
            output_n: 输出序列长度(要预测的帧数)
            split: 'train', 'val', 'test'
            scale: 数据缩放因子
            use_angles: 是否使用角度信息(22关节数据不使用)
            scaler: 预训练的StandardScaler(用于验证集和测试集)
        """
        self.input_n = input_n
        self.output_n = output_n
        self.scale = scale
        self.use_angles = use_angles
        self.split = split
        self.n_joints = 22  # 22个关节
        self.n_features = 66  # 22 * 3 = 66

        # 获取所有Excel文件
        excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))

        if not excel_files:
            raise ValueError(f"在目录 {data_dir} 中没有找到Excel文件")

        print(f"[{split.upper()}数据集] 找到 {len(excel_files)} 个Excel文件")

        # 加载和处理数据
        all_sequences = []
        self.data_info = []

        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)

                # 检查数据完整性(22个部位 * 3坐标 = 66列)
                if df.shape[1] < self.n_features:
                    print(
                        f"  - 警告: 文件 {os.path.basename(file_path)} 数据列数不足 ({df.shape[1]} < {self.n_features})，跳过此文件")
                    continue

                # 提取位置数据(前66列：22个部位 * 3坐标)
                position_data = df.iloc[:, :self.n_features].values
                sequence_data = position_data

                # 数据预处理
                sequence_data = self.preprocess_sequence(sequence_data)

                if len(sequence_data) >= input_n + output_n:
                    all_sequences.append(sequence_data)
                    self.data_info.append({
                        'file': os.path.basename(file_path),
                        'length': len(sequence_data),
                        'features': sequence_data.shape[1]
                    })
                else:
                    print(
                        f"  - 文件 {os.path.basename(file_path)} 序列太短 ({len(sequence_data)} < {input_n + output_n})，跳过")

            except Exception as e:
                print(f"  - 文件 {os.path.basename(file_path)} 加载失败: {e}")
                continue

        if not all_sequences:
            raise ValueError(f"在目录 {data_dir} 中没有找到有效的数据文件")

        print(f"\n[{split.upper()}数据集] 成功加载 {len(all_sequences)} 个序列")

        selected_sequences = all_sequences
        print(f"[{split.upper()}数据集] 将处理 {len(selected_sequences)} 个序列")

        # 数据标准化
        print(f"[{split.upper()}数据集] 正在进行数据标准化...")

        if split == 'train':
            self.scaler = StandardScaler()
            train_data = np.concatenate(selected_sequences, axis=0)
            self.scaler.fit(train_data)
            print(f"[{split.upper()}数据集] ✓ 使用训练集数据拟合标准化器，数据形状: {train_data.shape}")
        else:
            if scaler is None:
                raise ValueError(f"{split}集必须提供训练集的scaler参数！")
            self.scaler = scaler
            print(f"[{split.upper()}数据集] ✓ 使用训练集的标准化器转换{split}集数据")

        print(f"[{split.upper()}数据集] 标准化器特征数: {self.scaler.n_features_in_}")

        # 标准化当前数据集的序列
        normalized_sequences = []
        for seq in selected_sequences:
            normalized_seq = self.scaler.transform(seq)
            normalized_sequences.append(normalized_seq)

        self.sequences = normalized_sequences

        # 生成样本
        self.samples = self.generate_samples()

        # 重新整形数据以适应模型
        self.format_data()

        print(f"[{split.upper()}数据集] 总样本数: {len(self.samples)}")

    def preprocess_sequence(self, data):
        """预处理单个序列数据"""
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # 位置数据转换为米(前66列是位置数据)
        data[:, :self.n_features] = data[:, :self.n_features] / 1000.0  # mm转m
        return data

    def generate_samples(self):
        """从序列中生成训练样本"""
        samples = []
        seq_len = self.input_n + self.output_n

        for sequence in self.sequences:
            for i in range(len(sequence) - seq_len + 1):
                sample = sequence[i:i + seq_len]
                samples.append(sample)

        return samples

    def format_data(self):
        """将数据格式化为模型需要的格式"""
        formatted_samples = []

        for sample in self.samples:
            # 使用前66列(22个关节的xyz坐标)
            position_data = sample[:, :self.n_features]  # (timesteps, 66)
            # 重新整形为 (timesteps, 22, 3)
            position_data = position_data.reshape(len(sample), self.n_joints, 3)
            formatted_samples.append(position_data)

        self.samples = formatted_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]  # (timesteps, 22, 3)

        # 分割输入和输出
        input_seq = sample[:self.input_n]  # (input_n, 22, 3)
        output_seq = sample[self.input_n:]  # (output_n, 22, 3)

        # 转换为torch tensor
        input_seq = torch.FloatTensor(input_seq).transpose(0, 1)  # (22, input_n, 3)
        output_seq = torch.FloatTensor(output_seq).transpose(0, 1)  # (22, output_n, 3)

        # 计算速度
        vel_seq = torch.zeros_like(input_seq)
        vel_seq[:, 1:] = input_seq[:, 1:] - input_seq[:, :-1]
        vel_seq[:, 0] = vel_seq[:, 1]

        return input_seq, vel_seq, output_seq, output_seq, idx


class MocapDatasetWithAction(Dataset):
    """
    带动作标签的动捕数据集 - 用于按动作类别评估
    """

    def __init__(self, data_dir, input_n=10, output_n=30, split='train', scale=1000,
                 use_angles=False, scaler=None):
        self.input_n = input_n
        self.output_n = output_n
        self.scale = scale
        self.use_angles = use_angles
        self.split = split
        self.n_joints = 22
        self.n_features = 66

        # 15种动作类别
        self.action_names = [
            'directions', 'discussion', 'eating', 'greeting', 'phoning',
            'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',
            'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether'
        ]

        excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))

        if not excel_files:
            raise ValueError(f"在目录 {data_dir} 中没有找到Excel文件")

        print(f"[{split.upper()}数据集] 找到 {len(excel_files)} 个Excel文件")

        all_sequences = []
        self.data_info = []
        self.file_actions = []  # 存储每个文件对应的动作

        for file_path in excel_files:
            try:
                df = pd.read_excel(file_path)
                filename = os.path.basename(file_path)

                # 从文件名解析动作类型
                action = self._parse_action_from_filename(filename)

                if df.shape[1] < self.n_features:
                    print(f"  - 警告: 文件 {filename} 数据列数不足，跳过")
                    continue

                position_data = df.iloc[:, :self.n_features].values
                sequence_data = self.preprocess_sequence(position_data)

                if len(sequence_data) >= input_n + output_n:
                    all_sequences.append(sequence_data)
                    self.data_info.append({
                        'file': filename,
                        'length': len(sequence_data),
                        'features': sequence_data.shape[1],
                        'action': action
                    })
                    self.file_actions.append(action)

            except Exception as e:
                print(f"  - 文件 {os.path.basename(file_path)} 加载失败: {e}")
                continue

        if not all_sequences:
            raise ValueError(f"在目录 {data_dir} 中没有找到有效的数据文件")

        print(f"\n[{split.upper()}数据集] 成功加载 {len(all_sequences)} 个序列")

        # 统计每个动作的样本数
        action_counts = {}
        for action in self.file_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        print(f"[{split.upper()}数据集] 动作分布: {action_counts}")

        # 数据标准化
        if split == 'train':
            self.scaler = StandardScaler()
            train_data = np.concatenate(all_sequences, axis=0)
            self.scaler.fit(train_data)
        else:
            if scaler is None:
                raise ValueError(f"{split}集必须提供训练集的scaler参数！")
            self.scaler = scaler

        # 标准化
        normalized_sequences = []
        for seq in all_sequences:
            normalized_seq = self.scaler.transform(seq)
            normalized_sequences.append(normalized_seq)

        self.sequences = normalized_sequences

        # 生成样本（同时记录动作标签）
        self.samples, self.sample_actions = self.generate_samples_with_actions()
        self.format_data()

        print(f"[{split.upper()}数据集] 总样本数: {len(self.samples)}")

    def _parse_action_from_filename(self, filename):
        """从文件名解析动作类型"""
        filename_lower = filename.lower()
        for action in self.action_names:
            if action in filename_lower:
                return action
        return 'unknown'

    def preprocess_sequence(self, data):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data[:, :self.n_features] = data[:, :self.n_features] / 1000.0
        return data

    def generate_samples_with_actions(self):
        """生成样本并记录动作标签"""
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