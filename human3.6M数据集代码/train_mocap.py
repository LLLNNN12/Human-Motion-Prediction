"""
AuxFormer模型训练脚本 - 消融实验版本（去除掩码和去噪机制）
22关节版本
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import time

from model import AuxFormer
from custom_dataloader import MocapDataset


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


class LRScheduler:
    """学习率调度器"""
    def __init__(self, optimizer, patience=5, factor=0.5, min_lr=1e-7):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.counter = 0
        self.best_loss = None

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                for param_group in self.optimizer.param_groups:
                    new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                self.counter = 0
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_mocap_model(train_data_dir, val_data_dir, model_save_path,
                      epochs=50, batch_size=8, learning_rate=5e-4, weight_decay=1e-4):
    """
    训练AuxFormer模型 - 消融实验版本（去除掩码和去噪机制）
    只使用预测损失进行训练

    Args:
        train_data_dir: 训练数据目录
        val_data_dir: 验证数据目录
        model_save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        weight_decay: 权重衰减
    """
    print("=" * 80)
    print("AuxFormer模型训练 - 消融实验版本（去除掩码和去噪机制）")
    print("=" * 80)
    print("注意: 本版本只使用预测损失，不使用掩码重构损失和去噪损失")
    print("=" * 80)

    # 模型参数配置 - 22关节
    N_JOINTS = 22
    N_FEATURES = 66  # 22 * 3
    INPUT_N = 10
    OUTPUT_N = 30

    # 加载数据集
    print("\n1. 加载数据集...")
    try:
        train_dataset = MocapDataset(
            data_dir=train_data_dir,
            input_n=INPUT_N,
            output_n=OUTPUT_N,
            split='train',
            scaler=None
        )

        val_dataset = MocapDataset(
            data_dir=val_data_dir,
            input_n=INPUT_N,
            output_n=OUTPUT_N,
            split='val',
            scaler=train_dataset.scaler
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        print(f"  训练集样本数: {len(train_dataset)}")
        print(f"  验证集样本数: {len(val_dataset)}")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")

    except Exception as e:
        print(f"数据准备失败: {e}")
        return None

    # 创建模型 - 22关节（保持相同的模型结构，但训练时不使用掩码和去噪）
    print("\n2. 创建模型...")
    model = AuxFormer(
        in_dim=3,
        h_dim=128,
        past_timestep=INPUT_N,
        future_timestep=OUTPUT_N,
        mask_ratio=0.4,  # 保留参数但不使用
        decoder_dim=128,
        num_heads=8,
        encoder_depth=6,
        decoder_depth=2,
        decoder_dim_per_head=16,
        same_head=False,
        n_agent=N_JOINTS,  # 22个关节
        add_joint_token=True,
        add_residual=True,
        denoise=False,  # 去噪设为False
        denoise_mode='past',
        noise_dev=0.6,
        part_noise=False,  # 不使用部分噪声
        part_noise_ratio=0.3,
        mlp_dim=256,
        dim_per_head=16,
        regular_masking=False,  # 不使用掩码
        range_mask_ratio=False,  # 不使用范围掩码比例
        multi_output=True
    )

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数总数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  关节数量: {N_JOINTS}")
    print(f"  消融设置: 去除掩码和去噪机制")

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  使用设备: {device}")

    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay, betas=(0.9, 0.999))

    # 梯度裁剪
    max_grad_norm = 1.0

    # 早停和学习率调度
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)
    lr_scheduler = LRScheduler(optimizer, patience=8, factor=0.7)

    # 损失函数权重 - 只使用预测损失
    pred_weight = 1.0
    # mask_weight = 0  # 去除
    # denoise_weight = 0  # 去除

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\n3. 开始训练 (共{epochs}轮)...")
    print("-" * 80)

    for epoch in range(epochs):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0
        pred_loss_sum = 0
        num_batches = 0

        for batch_idx, data in enumerate(train_loader):
            try:
                loc, vel, loc_end, _, _ = data
                loc, vel, loc_end = loc.to(device), vel.to(device), loc_end.to(device)

                optimizer.zero_grad()

                # 准备输入数据
                all_traj = torch.cat([loc, loc_end], dim=2)

                # 模型前向传播
                if model.multi_output:
                    loc_pred_list, _, _, _, _ = model(all_traj)

                    # 多输出预测损失
                    pred_loss = 0
                    for loc_pred in loc_pred_list:
                        pred_loss += torch.mean(torch.norm(loc_pred - loc_end, dim=-1))
                    pred_loss = pred_loss / len(loc_pred_list)
                else:
                    loc_pred, _, _, _, _ = model(all_traj)
                    pred_loss = torch.mean(torch.norm(loc_pred - loc_end, dim=-1))

                # 总损失 - 只使用预测损失（消融实验：去除掩码和去噪损失）
                loss = pred_weight * pred_loss

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                train_loss += loss.item()
                pred_loss_sum += pred_loss.item()
                num_batches += 1

                if batch_idx % 20 == 0:
                    print(f'  训练批次 [{batch_idx:3d}/{len(train_loader)}] '
                          f'损失: {loss.item():.6f} '
                          f'(预测: {pred_loss.item():.4f})')

            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue

        avg_train_loss = train_loss / max(num_batches, 1)
        avg_pred_loss = pred_loss_sum / max(num_batches, 1)

        # 验证阶段
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for data in val_loader:
                try:
                    loc, vel, loc_end, _, _ = data
                    loc, vel, loc_end = loc.to(device), vel.to(device), loc_end.to(device)

                    # 预测模式
                    all_traj = torch.cat([loc, torch.zeros_like(loc_end)], dim=2)
                    loc_pred = model.predict(all_traj)

                    loss = torch.mean(torch.norm(loc_pred - loc_end, dim=-1))
                    val_loss += loss.item()
                    num_val_batches += 1

                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue

        avg_val_loss = val_loss / max(num_val_batches, 1)

        # 记录损失
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f'\n轮次 [{epoch + 1:3d}/{epochs}] - 时间: {epoch_time:.1f}s - 学习率: {current_lr:.2e}')
        print(f'  训练损失: {avg_train_loss:.6f} (预测: {avg_pred_loss:.4f})')
        print(f'  验证损失: {avg_val_loss:.6f}')

        # 学习率调度
        lr_scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

            # 创建模型保存目录
            os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_loss': best_loss,
                'scaler': train_dataset.scaler,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'model_config': {
                    'h_dim': 128,
                    'encoder_depth': 6,
                    'decoder_depth': 2,
                    'mask_ratio': 0.4,  # 保留信息但训练时不使用
                    'noise_dev': 0.6,
                    'n_agent': N_JOINTS,
                    'n_features': N_FEATURES,
                    'ablation': 'no_mask_no_denoise'  # 标记消融实验类型
                }
            }

            torch.save(checkpoint, model_save_path)
            print(f'  ✓ 保存最佳模型 - 验证损失: {best_loss:.6f}')

        # 早停检查
        if early_stopping(avg_val_loss):
            print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
            break

        print("-" * 80)

    print(f"\n训练完成！")
    print(f"最佳验证损失: {best_loss:.6f}")
    print(f"模型保存在: {model_save_path}")
    print(f"消融设置: 去除掩码和去噪机制")

    return model_save_path


if __name__ == "__main__":
    # 使用示例
    train_data_dir = "dataset_22joints/train_data"
    val_data_dir = "dataset_22joints/val_data"
    model_save_path = "models/mocap_model_22joints_no_mask_denoise.pth"

    # 开始训练
    result = train_mocap_model(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        model_save_path=model_save_path,
        epochs=50,
        batch_size=8,
        learning_rate=5e-4,
        weight_decay=1e-4
    )

    if result:
        print(f"\n训练成功完成！模型保存在: {result}")
    else:
        print("\n训练失败！")
