"""
AuxFormer模型 - 消融实验版本（去除掩码和去噪机制）
22关节版本
"""
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import random


class AuxFormer(nn.Module):
    """
    AuxFormer模型 - 消融实验版本
    去除了掩码重构和去噪机制，只保留预测功能
    """
    def __init__(self,
                 in_dim,
                 h_dim,
                 past_timestep,
                 future_timestep,
                 mask_ratio,
                 decoder_dim=64,
                 num_heads=8,
                 encoder_depth=3,
                 decoder_depth=1,
                 decoder_dim_per_head=32,
                 same_head=True,
                 range_mask_ratio=False,
                 dim_per_head=64,
                 mlp_head=False,
                 mask_past=False,
                 mask_range=20,
                 multi_output=False,
                 decoder_masking=False,
                 pred_all=False,
                 mlp_dim=64,
                 use_projector=False,
                 easier_given_dest=False,
                 noise_dev=0.,
                 part_noise=False,
                 denoise_mode='all',
                 part_noise_ratio=0.25,
                 add_joint_token=False,
                 n_agent=22,
                 concat_vel=False,
                 concat_acc=False,
                 dropout=0.,
                 only_recons_past=False,
                 two_mask_mode=False,
                 add_residual=False,
                 denoise=True,  # 保留参数但不使用
                 regular_masking=False,
                 multi_same_head=False,
                 range_noise_dev=False):
        super(AuxFormer, self).__init__()
        self.all_timesteps = past_timestep + future_timestep
        self.mask_ratio = mask_ratio
        self.range_mask_ratio = range_mask_ratio
        self.pred_num_masked = future_timestep
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))
        self.past_timestep = past_timestep
        self.future_timestep = future_timestep
        self.mask_past = mask_past
        self.mask_range = mask_range
        self.multi_output = multi_output
        self.decoder_masking = decoder_masking
        self.encoder_depth = encoder_depth
        self.patch_embed = nn.Linear(in_dim, h_dim)
        self.decoder_pos_embed = nn.Embedding(self.all_timesteps, decoder_dim)
        self.decoder_agent_embed = nn.Embedding(n_agent, decoder_dim)
        self.only_recons_past = only_recons_past

        self.pred_time = 1
        self.pred_all = pred_all
        self.concat_vel = concat_vel
        self.concat_acc = concat_acc
        if self.concat_vel:
            self.patch_embed = nn.Linear(6, h_dim)
        elif self.concat_acc:
            self.patch_embed = nn.Linear(9, h_dim)
        else:
            self.patch_embed = nn.Linear(3, h_dim)
        self.noise_dev = noise_dev
        self.part_noise = part_noise
        self.part_noise_ratio = part_noise_ratio
        self.denoise_mode = denoise_mode
        self.add_joint_token = add_joint_token
        self.mlp_head = mlp_head
        self.agent_embed = nn.Parameter(torch.randn(1, n_agent, 1, h_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.all_timesteps, h_dim))
        self.two_mask_mode = two_mask_mode
        self.add_residual = add_residual
        self.regular_masking = regular_masking
        self.multi_same_head = multi_same_head
        self.range_noise_dev = range_noise_dev

        if self.multi_output:
            self.head = []
            for i in range(self.encoder_depth):
                if mlp_head:
                    self.head.append(MLP(decoder_dim, 3 * self.pred_time, hidden_size=(128,)))
                else:
                    self.head.append(nn.Linear(decoder_dim, 3 * self.pred_time))
            self.head = nn.ModuleList(self.head)
        else:
            if mlp_head:
                self.head = MLP(decoder_dim, 3 * self.pred_time, hidden_size=(decoder_dim,))
            else:
                self.head = nn.Linear(decoder_dim, 3 * self.pred_time)
        self.same_head = same_head

        if not same_head:
            if mlp_head:
                self.aux_head = MLP(decoder_dim, 3 * self.pred_time, hidden_size=(decoder_dim,))
                self.aux_head_2 = MLP(decoder_dim, 3 * self.pred_time, hidden_size=(decoder_dim,))
            else:
                self.aux_head = nn.Linear(decoder_dim, 3 * self.pred_time)
                self.aux_head_2 = nn.Linear(decoder_dim, 3 * self.pred_time)

        self.encoder = []
        self.decoder = []
        for _ in range(self.encoder_depth):
            self.encoder.append(STTrans(self.all_timesteps, h_dim, depth=1, mlp_dim=mlp_dim, num_heads=num_heads,
                                        dim_per_head=dim_per_head, dropout=dropout))
            self.decoder.append(STTrans(self.all_timesteps, h_dim, depth=1, mlp_dim=mlp_dim, num_heads=num_heads,
                                        dim_per_head=dim_per_head, dropout=dropout))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, all_traj):
        """
        消融实验版本的前向传播 - 去除掩码和去噪机制
        只进行预测任务
        """
        B, N, T = all_traj.shape[0], all_traj.shape[1], all_traj.shape[2]
        all_traj = all_traj.view(B, N, T, 3)
        start = all_traj[:, :, self.past_timestep - 1:self.past_timestep]

        batch_ind = torch.arange(B)[:, None, None].cuda()
        agent_ind = torch.arange(N)[None, :, None].cuda()

        # 只使用ordinary_mask（不进行随机掩码）
        ordinary_mask = torch.zeros((B, N, T)).type_as(all_traj)
        ordinary_mask[:, :, :self.past_timestep] = 1.

        past_future_indices = torch.arange(T)[None].repeat(B, 1).cuda()
        past_future_indices = past_future_indices[:, None, :].repeat(1, N, 1)
        past_ind, future_ind = past_future_indices[:, :, :self.past_timestep], past_future_indices[
            :, :, self.past_timestep:]

        # 只进行预测，不做掩码重构和去噪
        if self.multi_output:
            out = self.mask_forward(all_traj, ordinary_mask, all_out=True)
            pred_future_coord_values = []
            for ind in range(len(out)):
                if self.multi_same_head:
                    head_ind = -1
                else:
                    head_ind = ind
                if self.mlp_head:
                    output = self.head[head_ind](out[ind][batch_ind, agent_ind, future_ind, :])
                else:
                    output = self.head[head_ind](out[ind][batch_ind, agent_ind, future_ind, :])
                output = output.view(B, N, -1, 3)
                if self.add_residual:
                    output += start
                pred_future_coord_values.append(output)
        else:
            decoded_tokens = self.mask_forward(all_traj, ordinary_mask)
            dec_future_tokens = decoded_tokens[batch_ind, agent_ind, future_ind, :]
            pred_future_coord_values = self.head(dec_future_tokens)
            pred_future_coord_values = pred_future_coord_values.view(B, N, -1, 3)
            if self.add_residual:
                pred_future_coord_values += start

        # 返回格式保持兼容，但掩码和去噪相关的输出为占位符
        # 这样可以保持与训练代码的接口兼容
        dummy_mask_pred = torch.zeros_like(all_traj)
        dummy_mask_gt = torch.zeros_like(all_traj)
        dummy_denoise_pred = torch.zeros(B, N, self.past_timestep, 3).type_as(all_traj)
        dummy_loss_mask = torch.zeros((B, N, T)).type_as(all_traj)

        return pred_future_coord_values, dummy_mask_pred, dummy_mask_gt, dummy_denoise_pred, dummy_loss_mask

    def mask_forward(self, all_trajs, mask, all_out=False):
        B, N, T = all_trajs.shape[0], all_trajs.shape[1], all_trajs.shape[2]

        if self.concat_acc:
            vel = torch.zeros_like(all_trajs)
            vel[:, :, 1:] = all_trajs[:, :, 1:] - all_trajs[:, :, :-1]
            acc = torch.zeros_like(vel)
            acc[:, :, 1:] = vel[:, :, 1:] - vel[:, :, :-1]
            all_traj_input = torch.cat([all_trajs, vel, acc], dim=-1)
        elif self.concat_vel:
            vel = torch.zeros_like(all_trajs)
            vel[:, :, 1:] = all_trajs[:, :, 1:] - all_trajs[:, :, :-1]
            all_traj_input = torch.cat([all_trajs, vel], dim=-1)
        else:
            all_traj_input = all_trajs

        batch_ind = torch.arange(B)[:, None, None].cuda()
        agent_ind = torch.arange(N)[None, :, None].cuda()

        inverse_mask = 1 - mask
        unmask_tokens = self.patch_embed(all_traj_input) * mask[:, :, :, None]
        unmask_tokens += self.pos_embed.repeat(B, N, 1, 1)

        if self.add_joint_token:
            unmask_tokens += self.agent_embed.repeat(B, 1, T, 1)

        unmask_tokens = unmask_tokens * mask[:, :, :, None]

        mask_s = mask.permute(0, 2, 1).contiguous().view(B * T, N)
        mask_s = torch.matmul(mask_s[:, :, None], mask_s[:, None, :])
        mask_t = mask.contiguous().view(B * N, T)
        mask_t = torch.matmul(mask_t[:, :, None], mask_t[:, None, :])

        unmask_tokens_pad = unmask_tokens
        mask_ind = torch.arange(T)[None, None, :].repeat(B, N, 1).cuda()

        out = []
        for l in range(self.encoder_depth):
            if l == 0:
                encoded_tokens = self.encoder[l](unmask_tokens_pad, mask_s, mask_t)

                enc_to_dec_tokens = encoded_tokens * mask[:, :, :, None]
                mask_tokens = self.mask_embed[None, None, None, :].repeat(B, N, T, 1)
                mask_tokens += self.decoder_pos_embed(mask_ind)

                if self.add_joint_token:
                    mask_tokens += self.decoder_agent_embed(agent_ind.repeat(B, 1, mask_tokens.shape[2]))

                mask_tokens = mask_tokens * inverse_mask[:, :, :, None]
                concat_tokens = enc_to_dec_tokens + mask_tokens

                dec_input_tokens = concat_tokens

                if self.decoder_masking:
                    decoded_tokens = self.decoder[l](dec_input_tokens, 1 - mask_s, 1 - mask_t)
                else:
                    decoded_tokens = self.decoder[l](dec_input_tokens)
                out.append(decoded_tokens)
            else:
                encoder_input = decoded_tokens

                encoder_input_pad = encoder_input * mask[:, :, :, None]
                encoder_output = self.encoder[l](encoder_input_pad, mask_s, mask_t)
                decoded_tokens = encoder_output * mask[:, :, :, None] + decoded_tokens * inverse_mask[:, :, :, None]
                if self.decoder_masking:
                    decoded_tokens = self.decoder[l](decoded_tokens, 1 - mask_s, 1 - mask_t)
                else:
                    decoded_tokens = self.decoder[l](decoded_tokens)
                out.append(decoded_tokens)
        if self.multi_output and all_out:
            return out
        else:
            return decoded_tokens

    def predict(self, all_traj):
        """预测函数保持不变"""
        B, N, T = all_traj.shape[0], all_traj.shape[1], all_traj.shape[2]
        all_traj = all_traj.view(B, N, T, 3)
        start = all_traj[:, :, self.past_timestep - 1:self.past_timestep]
        vel = torch.zeros_like(all_traj)
        vel[:, :, 1:] = all_traj[:, :, 1:] - all_traj[:, :, :-1]
        all_traj_with_vel = torch.cat([all_traj, vel], dim=-1)

        past_future_indices = torch.arange(T)[None].repeat(B, 1).cuda()
        past_future_indices = past_future_indices[:, None, :].repeat(1, N, 1)
        past_ind, future_ind = past_future_indices[:, :, :self.past_timestep], past_future_indices[
            :, :, self.past_timestep:]

        batch_ind = torch.arange(B)[:, None, None].cuda()
        agent_ind = torch.arange(N)[None, :, None].cuda()

        ordinary_mask = torch.zeros((B, N, T)).type_as(all_traj)
        ordinary_mask[:, :, :self.past_timestep] = 1.

        if self.multi_output:
            decoded_tokens = self.mask_forward(all_traj, ordinary_mask, all_out=True)[-1]
        else:
            decoded_tokens = self.mask_forward(all_traj, ordinary_mask)

        dec_future_tokens = decoded_tokens[batch_ind, agent_ind, future_ind, :]

        if self.multi_output:
            pred_future_coord_values = self.head[-1](dec_future_tokens)
        else:
            pred_future_coord_values = self.head(dec_future_tokens)

        pred_future_coord_values = pred_future_coord_values.view(B, N, -1, 3)
        if self.add_residual:
            pred_future_coord_values += start
        return pred_future_coord_values


class STTrans(nn.Module):
    def __init__(
            self, num_patches, h_dim, depth=3, num_heads=8, mlp_dim=128,
            pool='cls', dim_per_head=64, dropout=0., embed_dropout=0., multi_output=False
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=embed_dropout)
        self.multi_output = multi_output

        self.transformer_s = []
        self.transformer_t = []
        self.depth = depth
        for i in range(depth):
            self.transformer_t.append(Transformer(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
            self.transformer_s.append(Transformer(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
        self.transformer_t = nn.ModuleList(self.transformer_t)
        self.transformer_s = nn.ModuleList(self.transformer_s)

        self.pool = pool

    def forward(self, x, mask_s=None, mask_t=None):
        B, N = x.shape[0], x.shape[1]
        out = []
        for i in range(self.depth):
            x = x.contiguous().view(B * N, -1, x.shape[-1])
            x = self.transformer_t[i](x, mask_t)
            x = x.view(B, N, -1, x.shape[-1]).permute(0, 2, 1, 3)
            x = x.contiguous().view(-1, N, x.shape[-1])
            x = self.transformer_s[i](x, mask_s)
            x = x.view(B, -1, N, x.shape[-1]).permute(0, 2, 1, 3)
            out.append(x)
        if self.multi_output:
            return out
        else:
            return x


def to_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, l, d = x.shape

        qkv = self.to_qkv(x)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.chunk(3)

        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        if mask is not None:
            mask = mask[:, None, :, :].repeat(1, self.num_heads, 1, 1)
            attn = attn * mask
            attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + 1e-10)

        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).reshape(b, l, -1)
        out = self.out(z)
        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=1, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x, mask=mask)
            x = x + norm_ffn(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout / 3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x
