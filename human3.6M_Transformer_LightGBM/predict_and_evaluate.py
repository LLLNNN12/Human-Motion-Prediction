"""
Transformer模型预测和评估脚本 - 预测任务版本
22关节版本
评估特定帧(第2、4、8、10帧)的MPJPE，按动作类别统计
每个文件提取256个随机不重复的子序列进行评估
"""
import torch
import numpy as np
import pandas as pd
import os
import random
from model import Transformer

# 15种动作类别
ACTION_NAMES = [
    'directions', 'discussion', 'eating', 'greeting', 'phoning',
    'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',
    'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether'
]

# 评估的帧索引 (第2、4、8、10帧，注意索引从0开始)
EVAL_FRAMES = [1, 3, 7, 9]  # 对应第2、4、8、10帧
EVAL_FRAME_LABELS = [2, 4, 8, 10]  # 显示用的帧号

# 模型参数
N_JOINTS = 22
N_FEATURES = 66

# 子序列参数
SUBSEQUENCE_LENGTH = 40  # 每个子序列40行
INPUT_LENGTH = 10  # 前10行作为输入
OUTPUT_LENGTH = 30  # 后30行作为输出
NUM_SUBSEQUENCES = 256  # 每个文件提取256个子序列


def get_action_from_filename(filename):
    """
    从文件名精确解析动作类型
    注意：walkingdog、walkingtogether不应归入walking；sittingdown不应归入sitting
    """
    filename_lower = filename.lower()

    # 按照从长到短的顺序匹配，避免误匹配
    ordered_actions = [
        'walkingtogether', 'walkingdog', 'sittingdown', 'takingphoto',
        'directions', 'discussion', 'eating', 'greeting', 'phoning',
        'posing', 'purchases', 'sitting', 'smoking', 'waiting', 'walking'
    ]

    for action in ordered_actions:
        if action in filename_lower:
            return action

    return 'unknown'


def calculate_mpjpe_per_frame(y_true, y_pred):
    """
    计算每一帧的MPJPE

    Args:
        y_true: (frames, 66) 或 (frames, 22, 3)
        y_pred: (frames, 66) 或 (frames, 22, 3)

    Returns:
        mpjpe_per_frame: (frames,) 每帧的MPJPE
    """
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(-1, N_JOINTS, 3)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.reshape(-1, N_JOINTS, 3)

    # 计算每个关节的欧氏距离
    joint_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2))  # (frames, 22)

    # 每帧的平均关节误差
    mpjpe_per_frame = np.mean(joint_errors, axis=1)  # (frames,)

    return mpjpe_per_frame


def setup_model_from_checkpoint(checkpoint_path):
    """从检查点加载模型 - 预测任务版本"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    print(f"加载模型检查点（预测任务版本）...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model_config = checkpoint.get('model_config', {})

    TRAINED_PAST_TIMESTEP = 10
    TRAINED_FUTURE_TIMESTEP = 30

    # 创建预测任务模型
    model = Transformer(
        in_dim=3,
        h_dim=model_config.get('h_dim', 128),
        past_timestep=TRAINED_PAST_TIMESTEP,
        future_timestep=TRAINED_FUTURE_TIMESTEP,
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
        task_flag=False,  # 预测任务：不使用额外处理
        history_setting='past',
        offset_scale=model_config.get('offset_scale', 0.6),
        partial_offset=False,  # 预测任务：不使用可选偏移
        partial_offset_ratio=0.3,
        mlp_dim=256,
        dim_per_head=16,
        regular_context=False,  # 预测任务：不使用上下文选择
        range_context_ratio=False,
        multi_output=True
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']

    ablation_type = model_config.get('ablation', 'prediction_only')
    print(f"✓ 模型加载成功！训练轮次: {checkpoint['epoch']}")
    print(f"  训练类型: {ablation_type}")
    print(f"  模型参数: 输入帧={TRAINED_PAST_TIMESTEP}, 输出帧={TRAINED_FUTURE_TIMESTEP}, 关节数={model_config.get('n_agent', N_JOINTS)}")

    return model, scaler


def extract_random_subsequences(sequence_data, num_subsequences=256, subseq_length=40, seed=None):
    """
    从序列中随机提取不重复的子序列

    Args:
        sequence_data: 完整序列数据 (total_length, 22, 3)
        num_subsequences: 要提取的子序列数量
        subseq_length: 每个子序列的长度
        seed: 随机种子

    Returns:
        list of subsequences: [(40, 22, 3), ...]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    total_length = len(sequence_data)

    if total_length < subseq_length:
        raise ValueError(f"序列长度({total_length})不足以提取长度为{subseq_length}的子序列")

    max_start_idx = total_length - subseq_length + 1

    if max_start_idx < num_subsequences:
        print(f"  警告: 可用的子序列起始位置({max_start_idx})少于需要的数量({num_subsequences})")
        print(f"  将提取所有可能的{max_start_idx}个子序列")
        num_subsequences = max_start_idx

    start_indices = random.sample(range(max_start_idx), num_subsequences)

    subsequences = []
    for start_idx in start_indices:
        subseq = sequence_data[start_idx:start_idx + subseq_length]
        subsequences.append(subseq)

    return subsequences


def preprocess_single_file(file_path, scaler):
    """预处理单个Excel文件，返回归一化的序列数据"""
    try:
        df = pd.read_excel(file_path)

        if df.shape[1] < N_FEATURES:
            raise ValueError(f"数据列数不足，需要至少{N_FEATURES}列，实际为{df.shape[1]}列")

        position_data = df.iloc[:, :N_FEATURES].values
        original_data = position_data.copy()

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

        return sequence_data, original_data

    except Exception as e:
        print(f"  处理失败: {e}")
        return None, None


def predict_subsequences(model, subsequences, scaler, device):
    """
    对多个子序列进行批量预测
    """
    predictions = []
    ground_truths = []

    model.eval()

    for subseq in subsequences:
        input_data = subseq[:INPUT_LENGTH]
        output_data = subseq[INPUT_LENGTH:]

        input_tensor = torch.FloatTensor(input_data).transpose(0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        fake_future = torch.zeros(1, N_JOINTS, OUTPUT_LENGTH, 3, device=device)
        all_traj = torch.cat([input_tensor, fake_future], dim=2)

        with torch.no_grad():
            prediction = model.predict(all_traj)

        prediction = prediction.squeeze(0).transpose(0, 1).cpu().numpy()

        prediction_flat = prediction.reshape(OUTPUT_LENGTH, N_FEATURES)

        expected_features = scaler.n_features_in_
        if expected_features > N_FEATURES:
            zero_padding = np.zeros((OUTPUT_LENGTH, expected_features - N_FEATURES))
            prediction_full_padded = np.concatenate([prediction_flat, zero_padding], axis=1)
        else:
            prediction_full_padded = prediction_flat

        prediction_original = scaler.inverse_transform(prediction_full_padded)
        prediction_original = prediction_original[:, :N_FEATURES]
        prediction_original = prediction_original * 1000.0

        output_flat = output_data.reshape(OUTPUT_LENGTH, N_FEATURES)

        if expected_features > N_FEATURES:
            zero_padding = np.zeros((OUTPUT_LENGTH, expected_features - N_FEATURES))
            output_full_padded = np.concatenate([output_flat, zero_padding], axis=1)
        else:
            output_full_padded = output_flat

        output_original = scaler.inverse_transform(output_full_padded)
        output_original = output_original[:, :N_FEATURES]
        output_original = output_original * 1000.0

        predictions.append(prediction_original)
        ground_truths.append(output_original)

    return predictions, ground_truths


def evaluate_specific_frames(predictions, ground_truths, eval_frames=EVAL_FRAMES):
    """
    评估多个预测的特定帧的MPJPE
    """
    frame_mpjpes = {label: [] for label in EVAL_FRAME_LABELS}

    for pred, gt in zip(predictions, ground_truths):
        mpjpe_per_frame = calculate_mpjpe_per_frame(gt, pred)

        for i, frame_idx in enumerate(eval_frames):
            if frame_idx < len(mpjpe_per_frame):
                frame_label = EVAL_FRAME_LABELS[i]
                frame_mpjpes[frame_label].append(mpjpe_per_frame[frame_idx])

    return frame_mpjpes


def predict_and_evaluate_by_action(model_path, data_dir, output_dir):
    """
    按动作类别预测和评估 - 预测任务版本

    Args:
        model_path: 模型路径
        data_dir: 测试数据目录（S5数据）
        output_dir: 输出目录
    """
    print("=" * 80)
    print(f"Transformer预测和评估 - 预测任务版本")
    print(f"每个文件提取{NUM_SUBSEQUENCES}个随机子序列")
    print(f"子序列长度: {SUBSEQUENCE_LENGTH}帧（前{INPUT_LENGTH}帧输入，后{OUTPUT_LENGTH}帧输出）")
    print(f"评估帧: 第{EVAL_FRAME_LABELS}帧的MPJPE")
    print("=" * 80)

    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return None

    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    if not excel_files:
        print(f"在 {data_dir} 中没有找到Excel文件")
        return None

    print(f"找到 {len(excel_files)} 个Excel文件\n")

    # 加载模型
    try:
        model, scaler = setup_model_from_checkpoint(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"使用设备: {device}\n")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    action_results = {action: {f: [] for f in EVAL_FRAME_LABELS} for action in ACTION_NAMES}

    successful_predictions = 0
    failed_predictions = 0
    action_file_counts = {action: 0 for action in ACTION_NAMES}

    for i, excel_file in enumerate(excel_files, 1):
        print(f"[{i}/{len(excel_files)}] 处理文件: {excel_file}")

        action = get_action_from_filename(excel_file)
        file_path = os.path.join(data_dir, excel_file)

        try:
            sequence_data, original_data = preprocess_single_file(file_path, scaler)

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
                predictions, ground_truths = predict_subsequences(
                    model, subsequences, scaler, device
                )
            except Exception as e:
                failed_predictions += 1
                print(f"  ✗ 预测失败: {e}")
                continue

            frame_mpjpes = evaluate_specific_frames(predictions, ground_truths)

            for frame_label, mpjpe_list in frame_mpjpes.items():
                action_results[action][frame_label].extend(mpjpe_list)

            successful_predictions += 1
            action_file_counts[action] += 1

            print(f"  ✓ 动作: {action}, 平均MPJPE: ", end="")
            for fl in EVAL_FRAME_LABELS:
                avg_mpjpe = np.mean(frame_mpjpes[fl]) if frame_mpjpes[fl] else 0
                print(f"第{fl}帧={avg_mpjpe:.2f}mm ", end="")
            print()

        except Exception as e:
            failed_predictions += 1
            print(f"  ✗ 处理失败: {e}")

    # 计算并输出结果
    print("\n" + "=" * 80)
    print("评估结果汇总 - 预测任务（仅使用预测损失）")
    print("各动作第2、4、8、10帧的MPJPE (mm)")
    print("=" * 80)

    results_data = []
    all_frame_mpjpes = {f: [] for f in EVAL_FRAME_LABELS}

    header = f"{'动作':<20}{'文件数':<8}"
    for fl in EVAL_FRAME_LABELS:
        header += f"第{fl}帧".center(12)
    print(header)
    print("-" * 80)

    for action in ACTION_NAMES:
        row_data = {'动作': action, '文件数': action_file_counts[action]}
        row_str = f"{action:<20}{action_file_counts[action]:<8}"

        for fl in EVAL_FRAME_LABELS:
            mpjpes = action_results[action][fl]
            if mpjpes:
                avg_mpjpe = np.mean(mpjpes)
                row_str += f"{avg_mpjpe:>10.2f}  "
                row_data[f'第{fl}帧_MPJPE'] = avg_mpjpe
                all_frame_mpjpes[fl].extend(mpjpes)
            else:
                row_str += f"{'N/A':>10}  "
                row_data[f'第{fl}帧_MPJPE'] = np.nan

        print(row_str)
        results_data.append(row_data)

    print("-" * 80)
    avg_row = {'动作': '平均', '文件数': successful_predictions}
    avg_str = f"{'平均':<20}{successful_predictions:<8}"

    for fl in EVAL_FRAME_LABELS:
        if all_frame_mpjpes[fl]:
            avg_mpjpe = np.mean(all_frame_mpjpes[fl])
            avg_str += f"{avg_mpjpe:>10.2f}  "
            avg_row[f'第{fl}帧_MPJPE'] = avg_mpjpe
        else:
            avg_str += f"{'N/A':>10}  "
            avg_row[f'第{fl}帧_MPJPE'] = np.nan

    print(avg_str)
    results_data.append(avg_row)

    print("=" * 80)
    print(f"成功预测: {successful_predictions}/{len(excel_files)} 个文件")
    print(f"失败预测: {failed_predictions} 个文件")

    print("\n各动作子序列数量:")
    for action in ACTION_NAMES:
        total_subseqs = len(action_results[action][EVAL_FRAME_LABELS[0]])
        if total_subseqs > 0:
            print(f"  {action}: {total_subseqs}个子序列（{action_file_counts[action]}个文件 × {NUM_SUBSEQUENCES}）")

    # 保存结果到CSV
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, 'mpjpe_by_action_frames_prediction.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {csv_path}")

    # 保存详细报告
    report_path = os.path.join(output_dir, 'evaluation_report_prediction.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Transformer 22关节动作预测评估报告\n")
        f.write("预测任务：仅使用预测损失\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"评估方法: 每个文件提取{NUM_SUBSEQUENCES}个随机子序列\n")
        f.write(f"子序列长度: {SUBSEQUENCE_LENGTH}帧（前{INPUT_LENGTH}帧输入，后{OUTPUT_LENGTH}帧输出）\n")
        f.write(f"评估帧: 第{EVAL_FRAME_LABELS}帧\n")
        f.write(f"测试文件数: {len(excel_files)}\n")
        f.write(f"成功预测: {successful_predictions}\n")
        f.write(f"失败预测: {failed_predictions}\n\n")

        f.write("各动作第2、4、8、10帧的MPJPE (mm)\n")
        f.write("-" * 80 + "\n")

        f.write(f"{'动作':<20}{'文件数':<8}")
        for fl in EVAL_FRAME_LABELS:
            f.write(f"第{fl}帧".center(12))
        f.write("\n")
        f.write("-" * 80 + "\n")

        for action in ACTION_NAMES:
            f.write(f"{action:<20}{action_file_counts[action]:<8}")
            for fl in EVAL_FRAME_LABELS:
                mpjpes = action_results[action][fl]
                if mpjpes:
                    f.write(f"{np.mean(mpjpes):>10.2f}  ")
                else:
                    f.write(f"{'N/A':>10}  ")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'平均':<20}{successful_predictions:<8}")
        for fl in EVAL_FRAME_LABELS:
            if all_frame_mpjpes[fl]:
                f.write(f"{np.mean(all_frame_mpjpes[fl]):>10.2f}  ")
            else:
                f.write(f"{'N/A':>10}  ")
        f.write("\n")
        f.write("=" * 80 + "\n")

        f.write("\n各动作子序列数量:\n")
        for action in ACTION_NAMES:
            total_subseqs = len(action_results[action][EVAL_FRAME_LABELS[0]])
            if total_subseqs > 0:
                f.write(f"  {action}: {total_subseqs}个子序列（{action_file_counts[action]}个文件 × {NUM_SUBSEQUENCES}）\n")

    print(f"详细报告已保存到: {report_path}")

    return {
        'action_results': action_results,
        'all_frame_mpjpes': all_frame_mpjpes,
        'successful_predictions': successful_predictions,
        'failed_predictions': failed_predictions,
        'action_file_counts': action_file_counts
    }


if __name__ == "__main__":
    model_path = "models/mocap_model_22joints_prediction.pth"
    test_data_dir = "dataset_22joints/test_data"
    output_dir = "evaluation_results_prediction"

    result = predict_and_evaluate_by_action(
        model_path=model_path,
        data_dir=test_data_dir,
        output_dir=output_dir
    )

    if result:
        print("\n评估完成！")
    else:
        print("\n评估失败！")
