"""
run_xgboost_pipeline.py
LightGBM融合模型完整运行流程 - 22关节消融实验版本（去除掩码和去噪）

修改内容：
1. 去除类别特征（动作语义one-hot编码）
2. 新增FEATURE_KEEP_RATIO配置项，控制特征保留比例（默认0.6即60%）
3. 保留原有全部功能

功能：
1. 一键运行训练 + 评估全流程
2. 支持单独运行训练或评估
3. GPU/CPU自动检测
4. 生成详细评估报告
5. 适配消融实验版本AuxFormer（去除掩码和去噪机制）
6. 特征重要性筛选（可配置保留比例）

使用方法：
    python run_xgboost_pipeline.py              # 运行完整流程（训练+评估）
    python run_xgboost_pipeline.py --train      # 仅训练
    python run_xgboost_pipeline.py --eval       # 仅评估
    python run_xgboost_pipeline.py --help       # 显示帮助
"""
import sys
import os
import time
import numpy as np
import multiprocessing

# 配置常量
N_JOINTS = 22
N_FEATURES = 66
TRAIN_FRAMES = 10

# ==================== 特征筛选比例（可修改）====================
# 取值范围 (0, 1]，例如 0.6 表示保留60%特征，0.8 表示保留80%特征
FEATURE_KEEP_RATIO = 0.7
# ==============================================================


def check_requirements():
    """检查必要的依赖包"""
    print("\n检查依赖包...")

    required_packages = {
        'torch': 'PyTorch',
        'lightgbm': 'human3.6M-LightGBM代码4-10.txt',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'scipy': 'SciPy',
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} 未安装")
            missing.append(name)

    if missing:
        print(f"\n缺少依赖包: {', '.join(missing)}")
        print("请使用以下命令安装:")
        print("  pip install torch lightgbm pandas numpy scikit-learn scipy openpyxl")
        return False

    # 检查PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA可用 (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print(f"  → CUDA不可用，将使用CPU")
    except Exception as e:
        print(f"  ⚠ 无法检测CUDA: {e}")

    print("✓ 所有依赖检查通过")
    return True


def check_lightgbm_gpu():
    """检查LightGBM GPU支持"""
    print("\n检查LightGBM GPU加速...")

    try:
        import lightgbm as lgb

        np.random.seed(42)
        X_test = np.random.rand(100, 10).astype(np.float32)
        y_test = np.random.rand(100).astype(np.float32)

        # 尝试CUDA
        try:
            test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)
            params = {
                'objective': 'regression',
                'device': 'cuda',
                'verbosity': -1,
                'num_leaves': 15,
            }
            booster = lgb.train(params, test_data, num_boost_round=2,
                               callbacks=[lgb.log_evaluation(period=0)])
            del booster, test_data
            print("  ✓ human3.6M-LightGBM代码4-10.txt CUDA 加速可用!")
            return True, 'cuda'
        except Exception as e:
            print(f"  ✗ CUDA不可用: {str(e)[:60]}...")

        # 尝试OpenCL GPU
        try:
            test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)
            params = {
                'objective': 'regression',
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbosity': -1,
                'num_leaves': 15,
            }
            booster = lgb.train(params, test_data, num_boost_round=2,
                               callbacks=[lgb.log_evaluation(period=0)])
            del booster, test_data
            print("  ✓ human3.6M-LightGBM代码4-10.txt OpenCL GPU 加速可用!")
            return True, 'gpu'
        except Exception as e:
            print(f"  ✗ OpenCL GPU不可用: {str(e)[:60]}...")

        print("  → 将使用CPU训练（已限制线程数）")
        return False, None

    except ImportError:
        print("  ✗ LightGBM未安装")
        return False, None


def check_files(auxformer_model_path, train_data_dir, val_data_dir, test_data_dir):
    """检查必要的文件"""
    print("\n检查必要文件...")

    if not os.path.exists(auxformer_model_path):
        print(f"  ✗ AuxFormer模型不存在: {auxformer_model_path}")
        return False
    else:
        print(f"  ✓ AuxFormer模型存在（消融实验版本）")

    for name, path in [("训练", train_data_dir), ("验证", val_data_dir), ("测试", test_data_dir)]:
        if not os.path.exists(path):
            print(f"  ✗ {name}数据目录不存在: {path}")
            return False
        else:
            files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
            print(f"  ✓ {name}数据: {len(files)} 个Excel文件")

    return True


def main():
    """主函数 - 完整流程"""
    print("\n" + "=" * 80)
    print(" " * 10 + "LightGBM融合模型 - 22关节消融实验版本")
    print(" " * 10 + "（去除掩码和去噪机制）")
    print(" " * 10 + "【去除类别特征 + 特征重要性筛选】")
    print(" " * 10 + "一键运行全流程（GPU加速）")
    print(" " * 10 + "（评估规则与原AuxFormer完全一致：256个随机子序列采样）")
    print(" " * 10 + "（评估：15种动作 × 第2、4、8、10帧MPJPE + 精度提升%）")
    print(" " * 10 + f"（只训练前{TRAIN_FRAMES}帧，共{TRAIN_FRAMES * N_FEATURES}个模型）")
    print(" " * 10 + f"（特征保留比例: {FEATURE_KEEP_RATIO*100:.0f}%）")
    print("=" * 80)

    # ==================== 配置 ====================
    # 消融实验版本路径
    AUXFORMER_MODEL_PATH = "models/mocap_model_22joints_no_mask_denoise.pth"
    TRAIN_DATA_DIR = "dataset_22joints/train_data"
    VAL_DATA_DIR = "dataset_22joints/val_data"
    TEST_DATA_DIR = "dataset_22joints/test_data"
    LIGHTGBM_MODEL_PATH = "models/lightgbm_fusion_22joints_no_mask_denoise.pkl"
    EVALUATION_OUTPUT_DIR = "evaluation_results_no_mask_denoise/lightgbm_fusion"

    # LightGBM训练参数
    PREDICT_LENGTH = 30
    BATCH_SIZE = 16
    EARLY_STOPPING_ROUNDS = 50

    # CPU使用率控制
    CPU_USAGE_TARGET = 0.85
    FORCE_CPU = False

    # ==================== 检查环境 ====================
    if not check_requirements():
        return

    gpu_available, gpu_type = check_lightgbm_gpu()

    if gpu_available:
        print(f"\n✓ 将使用 {gpu_type.upper()} 加速LightGBM训练")
    else:
        total_threads = multiprocessing.cpu_count()
        used_threads = max(1, int(total_threads * CPU_USAGE_TARGET))
        print(f"\n→ 将使用CPU训练（{used_threads}/{total_threads}线程，约{CPU_USAGE_TARGET*100:.0f}%使用率）")

    if not check_files(AUXFORMER_MODEL_PATH, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR):
        print("\n请确保以下文件存在：")
        print(f"  1. AuxFormer模型（消融版本）: {AUXFORMER_MODEL_PATH}")
        print(f"  2. 训练数据目录: {TRAIN_DATA_DIR}")
        print(f"  3. 验证数据目录: {VAL_DATA_DIR}")
        print(f"  4. 测试数据目录: {TEST_DATA_DIR}")
        return

    os.makedirs(os.path.dirname(LIGHTGBM_MODEL_PATH), exist_ok=True)
    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

    # ==================== 步骤1: 训练 ====================
    print("\n" + "=" * 80)
    print("步骤 1/2: 训练LightGBM融合模型（消融实验版本）")
    print(f"  【去除类别特征 + 特征重要性筛选（保留{FEATURE_KEEP_RATIO*100:.0f}%）】")
    print("=" * 80)

    from train_lightgbm import train_lightgbm_fusion

    try:
        start_time = time.time()

        lgb_model_path = train_lightgbm_fusion(
            model_path=AUXFORMER_MODEL_PATH,
            train_data_dir=TRAIN_DATA_DIR,
            val_data_dir=VAL_DATA_DIR,
            lgb_save_path=LIGHTGBM_MODEL_PATH,
            predict_length=PREDICT_LENGTH,
            batch_size=BATCH_SIZE,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            force_cpu=FORCE_CPU,
            cpu_usage_target=CPU_USAGE_TARGET,
            feature_keep_ratio=FEATURE_KEEP_RATIO  # 【新增】传入特征保留比例
        )

        training_time = time.time() - start_time

        if lgb_model_path:
            print(f"\n✓ LightGBM训练完成!")
            print(f"  模型: {lgb_model_path}")
            print(f"  耗时: {training_time/60:.2f}分钟")
        else:
            print(f"\n✗ 训练失败")
            return

    except Exception as e:
        print(f"\n✗ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 步骤2: 预测和评估 ====================
    print("\n" + "=" * 80)
    print("步骤 2/2: 按动作类别预测和评估（消融实验版本）")
    print("（与原AuxFormer评估规则完全一致：每个文件提取256个随机子序列）")
    print("=" * 80)

    from predict_lightgbm import predict_and_evaluate_by_action_lightgbm

    try:
        start_time = time.time()

        result = predict_and_evaluate_by_action_lightgbm(
            auxformer_model_path=AUXFORMER_MODEL_PATH,
            lgb_model_path=LIGHTGBM_MODEL_PATH,
            data_dir=TEST_DATA_DIR,
            output_dir=EVALUATION_OUTPUT_DIR,
            cpu_usage_target=CPU_USAGE_TARGET
        )

        eval_time = time.time() - start_time
        print(f"\n✓ 评估完成! 耗时: {eval_time/60:.2f}分钟")

    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 生成最终汇总 ====================
    print("\n" + "=" * 80)
    print("生成最终汇总报告（消融实验版本）")
    print("=" * 80)

    if result:
        summary_file = os.path.join(EVALUATION_OUTPUT_DIR, "final_summary_no_mask_denoise.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("        LightGBM融合模型评估结果汇总\n")
            f.write("        22关节消融实验版本（去除掩码和去噪机制）\n")
            f.write("        【去除类别特征 + 特征重要性筛选】\n")
            f.write("        （只修正前10帧，与原AuxFormer评估规则完全一致：256个随机子序列采样）\n")
            f.write("        第2、4、8、10帧的MPJPE及精度提升百分比\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"配置信息:\n")
            f.write(f"  关节数量: {N_JOINTS}\n")
            f.write(f"  特征维度: {N_FEATURES}\n")
            f.write(f"  LightGBM修正帧数: {TRAIN_FRAMES}\n")
            f.write(f"  LightGBM模型数: {TRAIN_FRAMES * N_FEATURES}\n")
            f.write(f"  特征保留比例: {FEATURE_KEEP_RATIO*100:.0f}%\n")
            f.write(f"  去除类别特征: 是（动作one-hot编码）\n")
            f.write(f"  训练数据: {TRAIN_DATA_DIR}\n")
            f.write(f"  验证数据: {VAL_DATA_DIR}\n")
            f.write(f"  测试数据: {TEST_DATA_DIR}\n")
            f.write(f"  早停轮数: {EARLY_STOPPING_ROUNDS}\n")
            f.write(f"  CPU使用率限制: {CPU_USAGE_TARGET*100:.0f}%\n")
            f.write(f"  每个文件子序列数: 256\n")
            f.write(f"  子序列长度: 40帧（前10帧输入，后30帧输出）\n")
            f.write(f"  消融实验类型: 去除掩码和去噪机制\n\n")

            f.write("特征分类（共12类，已去除动作类别特征）:\n")
            feature_categories = [
                "pred_statistics", "input_sequence", "velocity", "acceleration",
                "key_frames", "bone_distances", "frequency_domain",
                "motion_characteristics", "prediction_quality", "joint_angles",
                "symmetry", "autocorrelation"
            ]
            for i, cat in enumerate(feature_categories, 1):
                f.write(f"  {i:2d}. {cat}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("15种动作平均MPJPE结果汇总:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'评估帧':<12}{'AuxFormer(消融)(mm)':<22}{'融合模型(mm)':<18}{'精度提升':<12}\n")
            f.write("-" * 80 + "\n")

            all_frame_results = result['all_frame_results']
            for frame_label in [2, 4, 8, 10]:
                if all_frame_results[frame_label]['auxformer']:
                    avg_auxformer = np.mean(all_frame_results[frame_label]['auxformer'])
                    avg_fusion = np.mean(all_frame_results[frame_label]['fusion'])
                    avg_improvement = np.mean(all_frame_results[frame_label]['improvement'])
                    f.write(f"第{frame_label}帧{'':<8}{avg_auxformer:<22.4f}{avg_fusion:<18.4f}{avg_improvement:<12.2f}%\n")

            f.write("=" * 80 + "\n")

            f.write(f"\n统计信息:\n")
            f.write(f"  成功预测: {result['successful_predictions']} 个文件\n")
            f.write(f"  失败预测: {result['failed_predictions']} 个文件\n")
            f.write(f"  总子序列数: {sum(result['action_subseq_counts'].values())}\n\n")

            f.write("各动作文件分布:\n")
            for action, count in result['action_file_counts'].items():
                if count > 0:
                    subseq_count = result['action_subseq_counts'][action]
                    f.write(f"  {action:<20}: {count} 个文件, {subseq_count} 个子序列\n")

        print(f"✓ 最终汇总已保存: {summary_file}")

    # ==================== 完成 ====================
    print("\n" + "=" * 80)
    print(" " * 20 + "全流程运行完成!")
    print(" " * 15 + "（消融实验版本：去除掩码和去噪）")
    print(" " * 10 + f"（去除类别特征 + 特征筛选保留{FEATURE_KEEP_RATIO*100:.0f}%）")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  1. LightGBM模型: {LIGHTGBM_MODEL_PATH}")
    print(f"  2. 评估结果目录: {EVALUATION_OUTPUT_DIR}")
    print(f"     - mpjpe_by_action_frames_lightgbm_no_mask_denoise.csv (详细结果)")
    print(f"     - evaluation_report_lightgbm_no_mask_denoise.txt (文本报告)")
    print(f"     - final_summary_no_mask_denoise.txt (最终汇总)")

    print("\n" + "=" * 80)
    print("评估指标说明:")
    print("  - MPJPE: Mean Per Joint Position Error (平均关节位置误差)")
    print("  - 单位: 毫米 (mm)")
    print("  - 评估帧: 第2帧、第4帧、第8帧、第10帧")
    print(f"  - LightGBM修正范围: 前{TRAIN_FRAMES}帧")
    print("  - 精度提升: (AuxFormer误差 - 融合模型误差) / AuxFormer误差 × 100%")
    print("  - 采样方法: 每个文件提取256个随机不重复子序列（与原AuxFormer一致）")
    print("  - 消融实验: 基础模型去除掩码和去噪机制")
    print(f"  - 特征筛选: 去除类别特征，保留{FEATURE_KEEP_RATIO*100:.0f}%重要特征")
    print("=" * 80 + "\n")


def run_training_only():
    """仅运行训练"""
    print("\n运行模式: 仅训练（消融实验版本）")
    print(f"特征保留比例: {FEATURE_KEEP_RATIO*100:.0f}%")

    AUXFORMER_MODEL_PATH = "models/mocap_model_22joints_no_mask_denoise.pth"
    TRAIN_DATA_DIR = "dataset_22joints/train_data"
    VAL_DATA_DIR = "dataset_22joints/val_data"
    LIGHTGBM_MODEL_PATH = "models/lightgbm_fusion_22joints_no_mask_denoise.pkl"

    if not check_requirements():
        return

    if not os.path.exists(AUXFORMER_MODEL_PATH):
        print(f"错误: AuxFormer模型（消融版本）不存在: {AUXFORMER_MODEL_PATH}")
        return

    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"错误: 训练数据目录不存在: {TRAIN_DATA_DIR}")
        return

    if not os.path.exists(VAL_DATA_DIR):
        print(f"错误: 验证数据目录不存在: {VAL_DATA_DIR}")
        return

    os.makedirs(os.path.dirname(LIGHTGBM_MODEL_PATH), exist_ok=True)

    from train_lightgbm import train_lightgbm_fusion

    result = train_lightgbm_fusion(
        model_path=AUXFORMER_MODEL_PATH,
        train_data_dir=TRAIN_DATA_DIR,
        val_data_dir=VAL_DATA_DIR,
        lgb_save_path=LIGHTGBM_MODEL_PATH,
        predict_length=30,
        batch_size=16,
        early_stopping_rounds=50,
        force_cpu=False,
        cpu_usage_target=0.85,
        feature_keep_ratio=FEATURE_KEEP_RATIO  # 【新增】
    )

    if result:
        print(f"\n训练完成！模型保存在: {result}")


def run_evaluation_only():
    """仅运行评估"""
    print("\n运行模式: 仅评估（消融实验版本，与原AuxFormer评估规则一致）")

    AUXFORMER_MODEL_PATH = "models/mocap_model_22joints_no_mask_denoise.pth"
    LIGHTGBM_MODEL_PATH = "models/lightgbm_fusion_22joints_no_mask_denoise.pkl"
    TEST_DATA_DIR = "dataset_22joints/test_data"
    EVALUATION_OUTPUT_DIR = "evaluation_results_no_mask_denoise/lightgbm_fusion"

    if not check_requirements():
        return

    if not os.path.exists(AUXFORMER_MODEL_PATH):
        print(f"错误: AuxFormer模型（消融版本）不存在: {AUXFORMER_MODEL_PATH}")
        return

    if not os.path.exists(LIGHTGBM_MODEL_PATH):
        print(f"错误: LightGBM模型不存在: {LIGHTGBM_MODEL_PATH}")
        print("请先运行训练步骤")
        return

    if not os.path.exists(TEST_DATA_DIR):
        print(f"错误: 测试数据目录不存在: {TEST_DATA_DIR}")
        return

    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)

    from predict_lightgbm import predict_and_evaluate_by_action_lightgbm

    result = predict_and_evaluate_by_action_lightgbm(
        auxformer_model_path=AUXFORMER_MODEL_PATH,
        lgb_model_path=LIGHTGBM_MODEL_PATH,
        data_dir=TEST_DATA_DIR,
        output_dir=EVALUATION_OUTPUT_DIR,
        cpu_usage_target=0.85
    )

    if result:
        print("\n评估完成！")


def print_help():
    """打印帮助信息"""
    print(f"""
LightGBM融合模型 - 22关节动作预测系统
消融实验版本（去除掩码和去噪机制）
【去除类别特征 + 特征重要性筛选（保留{FEATURE_KEEP_RATIO*100:.0f}%特征）】

用法:
    python run_xgboost_pipeline.py              运行完整流程（训练+评估）
    python run_xgboost_pipeline.py --train      仅训练LightGBM模型
    python run_xgboost_pipeline.py --eval       仅评估（需要已训练的模型）
    python run_xgboost_pipeline.py --help       显示此帮助信息

修改特征保留比例:
    打开 run_xgboost_pipeline.py，修改文件顶部的 FEATURE_KEEP_RATIO 变量
    例如: FEATURE_KEEP_RATIO = 0.6  # 保留60%特征
          FEATURE_KEEP_RATIO = 0.8  # 保留80%特征

    或者直接在 train_lightgbm.py 的 __main__ 中修改 feature_keep_ratio 参数

目录结构要求:
    models/
        mocap_model_22joints_no_mask_denoise.pth    # 预训练的AuxFormer模型（消融版本）
    dataset_22joints/
        train_data/                 # 训练数据（S1/6/7/8/9）
        val_data/                   # 验证数据（S11）
        test_data/                  # 测试数据（S5）

输出:
    models/
        lightgbm_fusion_22joints_no_mask_denoise.pkl    # 训练的LightGBM模型（含特征筛选索引）
    evaluation_results_no_mask_denoise/lightgbm_fusion/
        mpjpe_by_action_frames_lightgbm_no_mask_denoise.csv  # 详细结果
        evaluation_report_lightgbm_no_mask_denoise.txt       # 文本报告
        final_summary_no_mask_denoise.txt                    # 最终汇总

特性：
    - 消融实验版本（基础模型去除掩码和去噪机制）
    - 增强特征工程（12类特征，去除动作类别one-hot编码）
    - 特征重要性筛选（保留{FEATURE_KEEP_RATIO*100:.0f}%重要特征，比例可配置）
    - GPU加速（自动检测）
    - CPU使用率限制（默认85%）
    - 早停机制
    - 与原AuxFormer评估规则一致
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--train', '-t', 'train']:
            run_training_only()
        elif arg in ['--eval', '-e', 'eval', 'evaluate']:
            run_evaluation_only()
        elif arg in ['--help', '-h', 'help']:
            print_help()
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助")
    else:
        main()