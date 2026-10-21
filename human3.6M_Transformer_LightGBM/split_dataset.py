"""
数据集划分脚本 - 22关节版本
用于预测任务
"""
import os
import shutil
import random
import re


def split_dataset_by_subject_22joints(source_dir, output_base_dir, seed=42):
    """
    按Subject编号划分数据集 - 22个关节版本

    划分规则：
    - S11 作为验证集
    - S5 作为测试集
    - S1/6/7/8/9 作为训练集

    文件命名规则：
    S1/5/6/7/8/9/11 + _ + 动作名 + _ + 1/2 + 数字
    例如: S1_directions_1-1.xlsx, S5_walking_2-15.xlsx

    Args:
        source_dir: 原始数据文件夹路径
        output_base_dir: 输出根目录
        seed: 随机种子

    Returns:
        train_dir, val_dir, test_dir: 训练集、验证集、测试集目录
    """
    print("=" * 70)
    print("数据集划分（22关节版本）")
    print("=" * 70)
    print(f"  训练集: S1, S6, S7, S8, S9")
    print(f"  验证集: S11")
    print(f"  测试集: S5")
    print("=" * 70)

    random.seed(seed)

    # 获取所有Excel文件
    excel_files = [f for f in os.listdir(source_dir) if f.endswith('.xlsx')]
    print(f"\n总文件数: {len(excel_files)}")

    # 定义Subject分组
    train_subjects = ['S1', 'S6', 'S7', 'S8', 'S9']  # 训练集
    val_subjects = ['S11']  # 验证集
    test_subjects = ['S5']  # 测试集

    # 按Subject分类文件
    train_files = []
    val_files = []
    test_files = []
    unmatched_files = []

    # 解析文件名的模式: S{数字}_{动作}_{样本编号}
    pattern = r'^(S\d+)_.*'

    for filename in excel_files:
        match = re.match(pattern, filename, re.IGNORECASE)
        if match:
            subject = match.group(1).upper()
            if subject in train_subjects:
                train_files.append(filename)
            elif subject in val_subjects:
                val_files.append(filename)
            elif subject in test_subjects:
                test_files.append(filename)
            else:
                unmatched_files.append(filename)
        else:
            unmatched_files.append(filename)

    # 打乱训练集顺序
    random.shuffle(train_files)

    print(f"\n文件划分结果:")
    print(f"  训练集 (S1/6/7/8/9): {len(train_files)} 个文件")
    print(f"  验证集 (S11): {len(val_files)} 个文件")
    print(f"  测试集 (S5): {len(test_files)} 个文件")

    if unmatched_files:
        print(f"\n警告: {len(unmatched_files)} 个文件无法匹配Subject编号:")
        for f in unmatched_files[:5]:
            print(f"  - {f}")
        if len(unmatched_files) > 5:
            print(f"  ... 还有 {len(unmatched_files) - 5} 个文件")

    # 创建输出目录
    train_dir = os.path.join(output_base_dir, 'train_data')
    val_dir = os.path.join(output_base_dir, 'val_data')
    test_dir = os.path.join(output_base_dir, 'test_data')

    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # 复制文件
    print(f"\n开始复制文件...")

    def copy_files(file_list, target_dir, dataset_name):
        for i, filename in enumerate(file_list, 1):
            src = os.path.join(source_dir, filename)
            dst = os.path.join(target_dir, filename)
            shutil.copy2(src, dst)
            if i % 50 == 0 or i == len(file_list):
                print(f"  {dataset_name}: 已复制 {i}/{len(file_list)} 个文件", end='\r')
        print(f"  {dataset_name}: 已复制 {len(file_list)}/{len(file_list)} 个文件 ✓")

    copy_files(train_files, train_dir, "训练集")
    copy_files(val_files, val_dir, "验证集")
    copy_files(test_files, test_dir, "测试集")

    # 保存划分信息
    split_info_file = os.path.join(output_base_dir, 'subject_split_info.txt')
    with open(split_info_file, 'w', encoding='utf-8') as f:
        f.write("数据集划分信息 (22关节版本 - 按Subject编号)\n")
        f.write("=" * 50 + "\n")
        f.write(f"总文件数: {len(excel_files)}\n")
        f.write(f"训练集 (S1/6/7/8/9): {len(train_files)} 个文件\n")
        f.write(f"验证集 (S11): {len(val_files)} 个文件\n")
        f.write(f"测试集 (S5): {len(test_files)} 个文件\n")
        f.write(f"随机种子: {seed}\n\n")

        # 统计每个Subject的文件数
        subject_counts = {}
        for fn in excel_files:
            match = re.match(pattern, fn, re.IGNORECASE)
            if match:
                subject = match.group(1).upper()
                subject_counts[subject] = subject_counts.get(subject, 0) + 1

        f.write("各Subject文件数:\n")
        for subject in sorted(subject_counts.keys()):
            f.write(f"  {subject}: {subject_counts[subject]} 个文件\n")

        # 统计每个动作的文件数
        action_counts = {}
        action_pattern = r'^S\d+_([a-z]+)_.*'
        for fn in excel_files:
            match = re.match(action_pattern, fn, re.IGNORECASE)
            if match:
                action = match.group(1).lower()
                action_counts[action] = action_counts.get(action, 0) + 1

        f.write("\n各动作文件数:\n")
        for action in sorted(action_counts.keys()):
            f.write(f"  {action}: {action_counts[action]} 个文件\n")

    print(f"\n✓ 数据集划分完成!")
    print(f"  训练集目录: {train_dir}")
    print(f"  验证集目录: {val_dir}")
    print(f"  测试集目录: {test_dir}")
    print(f"  划分信息: {split_info_file}")
    print("=" * 70)

    return train_dir, val_dir, test_dir


def get_action_from_filename(filename):
    """
    从文件名解析动作类型

    Args:
        filename: 文件名，格式如 S1_directions_1-1.xlsx

    Returns:
        action: 动作名称
    """
    # 15种动作
    actions = [
        'directions', 'discussion', 'eating', 'greeting', 'phoning',
        'posing', 'purchases', 'sitting', 'sittingdown', 'smoking',
        'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether'
    ]

    filename_lower = filename.lower()
    for action in actions:
        if action in filename_lower:
            return action
    return 'unknown'


def analyze_dataset_by_action(data_dir):
    """
    按动作类型分析数据集

    Args:
        data_dir: 数据目录

    Returns:
        action_files: 动作到文件列表的字典
    """
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]

    action_files = {}
    for filename in excel_files:
        action = get_action_from_filename(filename)
        if action not in action_files:
            action_files[action] = []
        action_files[action].append(filename)

    print(f"\n数据集动作分布 ({data_dir}):")
    for action in sorted(action_files.keys()):
        print(f"  {action}: {len(action_files[action])} 个文件")

    return action_files


if __name__ == "__main__":
    # 使用示例
    source_directory = "./all_data"  # 原始数据文件夹
    output_directory = "./dataset_22joints"  # 输出目录

    # 执行按Subject编号划分
    train_dir, val_dir, test_dir = split_dataset_by_subject_22joints(
        source_dir=source_directory,
        output_base_dir=output_directory,
        seed=42
    )

    # 分析测试集的动作分布
    analyze_dataset_by_action(test_dir)
