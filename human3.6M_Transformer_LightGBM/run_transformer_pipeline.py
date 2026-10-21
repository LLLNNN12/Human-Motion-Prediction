"""
Transformer 22关节动作预测 - 预测任务一键运行脚本
仅使用预测损失
包含：数据集划分 -> 模型训练 -> 预测评估 全流程
"""

import os
import sys
import argparse
from datetime import datetime


def main():
    """主流程"""

    print("\n" + "=" * 80)
    print(" " * 10 + "Transformer 22关节动作预测系统 - 预测任务")
    print(" " * 10 + "仅使用预测损失")
    print("=" * 80)

    # ==================== 配置部分 ====================
    # 请根据您的实际情况修改以下路径

    # 原始数据文件夹（包含所有Excel文件）
    SOURCE_DATA_DIR = "all_data"

    # 数据集输出目录
    DATASET_OUTPUT_DIR = "dataset_22joints"

    # 模型保存目录
    MODEL_SAVE_DIR = "models"

    # 评估结果输出目录
    EVALUATION_OUTPUT_DIR = "evaluation_results_prediction"

    # 训练参数
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-4

    # 预测长度（需要至少10帧才能评估第2、4、8、10帧）
    PREDICT_LENGTH = 10

    # ==================== 步骤1: 数据集划分 ====================
    print("\n" + "=" * 80)
    print("步骤 1/3: 数据集划分")
    print("-" * 80)
    print("划分规则:")
    print("  - 训练集: S1, S6, S7, S8, S9")
    print("  - 验证集: S11")
    print("  - 测试集: S5")
    print("=" * 80)

    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"错误: 源数据目录不存在: {SOURCE_DATA_DIR}")
        print("请创建该目录并放入您的Excel文件，或修改 SOURCE_DATA_DIR 路径")
        return

    from split_dataset import split_dataset_by_subject_22joints

    try:
        train_dir, val_dir, test_dir = split_dataset_by_subject_22joints(
            source_dir=SOURCE_DATA_DIR,
            output_base_dir=DATASET_OUTPUT_DIR,
            seed=42
        )
        print(f"✓ 数据集划分完成")
    except Exception as e:
        print(f"✗ 数据集划分失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建模型保存目录和评估结果目录
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

    # 模型保存路径（预测任务版本）
    model_path = os.path.join(MODEL_SAVE_DIR, "mocap_model_22joints_prediction.pth")

    # ==================== 步骤2: 训练Transformer模型（预测版本）====================
    print("\n" + "=" * 80)
    print("步骤 2/3: 训练Transformer模型（预测任务版本）")
    print("-" * 80)
    print(f"训练参数:")
    print(f"  - 训练轮数: {EPOCHS}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 关节数量: 22")
    print(f"  - 训练配置: 仅保留预测损失")
    print("=" * 80)

    from train_mocap import train_mocap_model

    try:
        result_path = train_mocap_model(
            train_data_dir=train_dir,
            val_data_dir=val_dir,
            model_save_path=model_path,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=1e-4
        )

        if result_path:
            print(f"✓ Transformer模型训练完成（预测版本），保存在: {result_path}")
        else:
            print(f"✗ Transformer模型训练失败")
            return

    except Exception as e:
        print(f"✗ Transformer模型训练出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 步骤3: 预测和评估 ====================
    print("\n" + "=" * 80)
    print("步骤 3/3: 预测和评估（预测任务结果）")
    print("-" * 80)
    print("评估内容:")
    print("  - 15种动作的第2、4、8、10帧MPJPE")
    print("  - 15种动作平均的第2、4、8、10帧MPJPE")
    print("  - 预测任务：仅使用预测损失")
    print("=" * 80)

    from predict_and_evaluate import predict_and_evaluate_by_action

    try:
        eval_result = predict_and_evaluate_by_action(
            model_path=model_path,
            data_dir=test_dir,
            output_dir=EVALUATION_OUTPUT_DIR
        )

        if eval_result:
            print(f"✓ 预测和评估完成")
        else:
            print(f"✗ 预测和评估失败")

    except Exception as e:
        print(f"✗ 预测和评估出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 完成 ====================
    print("\n" + "=" * 80)
    print(" " * 15 + "预测任务全流程完成！")
    print("=" * 80)

    print(f"\n生成的文件和目录:")
    print(f"\n  1. 数据集: {DATASET_OUTPUT_DIR}/")
    print(f"     - 训练集: {train_dir}/")
    print(f"     - 验证集: {val_dir}/")
    print(f"     - 测试集: {test_dir}/")

    print(f"\n  2. 模型文件: {model_path}")

    print(f"\n  3. 评估结果: {EVALUATION_OUTPUT_DIR}/")
    print(f"     - mpjpe_by_action_frames_prediction.csv (各动作各帧MPJPE)")
    print(f"     - evaluation_report_prediction.txt (详细评估报告)")

    print("\n" + "=" * 80)
    print("预测任务说明:")
    print("  - 本模型训练时只计算运动预测损失")
    print("  - 只使用预测损失进行训练")
    print("  - 评估输出保留原有逐动作和逐帧统计规则")
    print("=" * 80 + "\n")


def run_training_only():
    """仅运行训练"""
    print("\n运行模式: 仅训练（预测任务）")

    SOURCE_DATA_DIR = "all_data"
    DATASET_OUTPUT_DIR = "dataset_22joints"
    MODEL_SAVE_DIR = "models"

    # 检查数据集是否已划分
    train_dir = os.path.join(DATASET_OUTPUT_DIR, 'train_data')
    val_dir = os.path.join(DATASET_OUTPUT_DIR, 'val_data')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("数据集未划分，先进行划分...")
        from split_dataset import split_dataset_by_subject_22joints
        train_dir, val_dir, test_dir = split_dataset_by_subject_22joints(
            source_dir=SOURCE_DATA_DIR,
            output_base_dir=DATASET_OUTPUT_DIR,
            seed=42
        )

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, "mocap_model_22joints_prediction.pth")

    from train_mocap import train_mocap_model

    result = train_mocap_model(
        train_data_dir=train_dir,
        val_data_dir=val_dir,
        model_save_path=model_path,
        epochs=50,
        batch_size=8,
        learning_rate=5e-4
    )

    if result:
        print(f"\n训练完成！模型保存在: {result}")


def run_evaluation_only():
    """仅运行评估"""
    print("\n运行模式: 仅评估（预测任务）")

    MODEL_SAVE_DIR = "models"
    DATASET_OUTPUT_DIR = "dataset_22joints"
    EVALUATION_OUTPUT_DIR = "evaluation_results_prediction"

    model_path = os.path.join(MODEL_SAVE_DIR, "mocap_model_22joints_prediction.pth")
    test_dir = os.path.join(DATASET_OUTPUT_DIR, 'test_data')

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练")
        return

    if not os.path.exists(test_dir):
        print(f"错误: 测试集目录不存在: {test_dir}")
        print("请先进行数据集划分")
        return

    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

    from predict_and_evaluate import predict_and_evaluate_by_action

    result = predict_and_evaluate_by_action(
        model_path=model_path,
        data_dir=test_dir,
        output_dir=EVALUATION_OUTPUT_DIR
    )

    if result:
        print("\n评估完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer 22关节动作预测系统 - 预测任务（仅使用预测损失）')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'train', 'eval'],
                        help='运行模式: full(完整流程), train(仅训练), eval(仅评估)')

    args = parser.parse_args()

    if args.mode == 'full':
        main()
    elif args.mode == 'train':
        run_training_only()
    elif args.mode == 'eval':
        run_evaluation_only()
