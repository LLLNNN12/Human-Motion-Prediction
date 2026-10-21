"""
Transformer模型 - 预测任务版本
22关节版本
"""
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import random


class Transformer(nn.Module):
    """
    Transformer模型 - 预测任务版本
    仅执行运动预测任务
    """
    def __init__(self,
                 in_dim,
                 h_dim,
                 past_timestep,
                 future_timestep,
                 context_ratio,
                 decoder_dim=64,
                 num_heads=8,
                 encoder_depth=3,
                 decoder_depth=1,
                 decoder_dim_per_head=32,
                 same_head=True,
                 range_context_ratio=False,
                 dim_per_head=64,
                 mlp_head=False,
                 select_past=False,
                 context_range=20,
                 multi_output=False,
                 decoder_context=False,
                 pred_all=False,
                 mlp_dim=64,
                 use_projector=False,
                 easier_given_dest=False,
                 offset_scale=0.,
                 partial_offset=False,
                 history_setting='all',
                 partial_offset_ratio=0.25,
                 add_joint_token=False,
                 n_agent=22,
                 concat_vel=False,
                 concat_acc=False,
                 dropout=0.,
                 only_history_output=False,
                 dual_context_mode=False,
                 add_residual=False,
                 task_flag=True,  # 保留参数但不使用
                 regular_context=False,
                 multi_same_head=False,
                 range_offset_scale=False):
        super(Transformer, self).__init__()
        self.all_timesteps = past_timestep + future_timestep
        self.context_ratio = context_ratio
        self.range_context_ratio = range_context_ratio
        self.pred_num_future = future_timestep
        self.future_embed = nn.Parameter(torch.randn(decoder_dim))
        self.past_timestep = past_timestep
        self.future_timestep = future_timestep
        self.select_past = select_past
        self.context_range = context_range
        self.multi_output = multi_output
        self.decoder_context = decoder_context
        self.encoder_depth = encoder_depth
        self.patch_embed = nn.Linear(in_dim, h_dim)
        self.decoder_pos_embed = nn.Embedding(self.all_timesteps, decoder_dim)
        self.decoder_agent_embed = nn.Embedding(n_agent, decoder_dim)
        self.only_history_output = only_history_output

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
        self.offset_scale = offset_scale
        self.partial_offset = partial_offset
        self.partial_offset_ratio = partial_offset_ratio
        self.history_setting = history_setting
        self.add_joint_token = add_joint_token
        self.mlp_head = mlp_head
        self.agent_embed = nn.Parameter(torch.randn(1, n_agent, 1, h_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.all_timesteps, h_dim))
        self.dual_context_mode = dual_context_mode
        self.add_residual = add_residual
        self.regular_context = regular_context
        self.multi_same_head = multi_same_head
        self.range_offset_scale = range_offset_scale

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
                self.branch_head_1 = MLP(decoder_dim, 3 * self.pred_time, hidden_size=(decoder_dim,))
                self.branch_head_2 = MLP(decoder_dim, 3 * self.pred_time, hidden_size=(decoder_dim,))
            else:
                self.branch_head_1 = nn.Linear(decoder_dim, 3 * self.pred_time)
                self.branch_head_2 = nn.Linear(decoder_dim, 3 * self.pred_time)

        self.encoder = []
        self.decoder = []
        for _ in range(self.encoder_depth):
            self.encoder.append(STTrans(self.all_timesteps, h_dim, depth=1, mlp_dim=mlp_dim, num_heads=num_heads,
                                        dim_per_head=dim_per_head, dropout=dropout))
            self.decoder.append(STTrans(self.all_timesteps, h_dim, depth=1, mlp_dim=mlp_dim, num_heads=num_heads,
                                        dim_per_head=dim_per_head, dropout=dropout))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load checkpoints using both current and earlier parameter names."""
        current_keys = self.state_dict()
        remapped = state_dict
        if 'future_embed' not in remapped:
            prior_token_keys = [
                key for key, value in remapped.items()
                if key not in current_keys
                and key.endswith('_embed')
                and tuple(value.shape) == tuple(self.future_embed.shape)
            ]
            if len(prior_token_keys) == 1:
                remapped = remapped.copy()
                remapped['future_embed'] = remapped.pop(prior_token_keys[0])

        # Registered prediction heads were renamed; infer former names from
        # unmatched checkpoint keys and their exact tensor shapes, without
        # changing any layer dimensions or parameter values.
        if not self.same_head:
            unknown_roots = {
                key.split('.', 1)[0] for key in remapped
                if key not in current_keys and '.' in key
            }
            for target_root in ('branch_head_1', 'branch_head_2'):
                target_keys = [
                    key for key in current_keys
                    if key.startswith(target_root + '.') and key not in remapped
                ]
                if not target_keys:
                    continue
                endings = [key[len(target_root):] for key in target_keys]
                candidates = [
                    root for root in unknown_roots
                    if root.endswith('_2') == target_root.endswith('_2')
                    and all(
                        root + ending in remapped
                        and tuple(remapped[root + ending].shape)
                            == tuple(current_keys[target_root + ending].shape)
                        for ending in endings
                    )
                ]
                if len(candidates) == 1:
                    legacy_root = candidates[0]
                    if remapped is state_dict:
                        remapped = remapped.copy()
                    for ending in endings:
                        remapped[target_root + ending] = remapped.pop(legacy_root + ending)
                    unknown_roots.remove(legacy_root)
        return super().load_state_dict(remapped, strict=strict, assign=assign)

    def forward(self, all_traj):
        """
        预测任务版本的前向传播 - 仅使用预测损失
        只进行预测任务
        """
        B, N, T = all_traj.shape[0], all_traj.shape[1], all_traj.shape[2]
        all_traj = all_traj.view(B, N, T, 3)
        start = all_traj[:, :, self.past_timestep - 1:self.past_timestep]

        batch_ind = torch.arange(B)[:, None, None].cuda()
        agent_ind = torch.arange(N)[None, :, None].cuda()

        # 使用历史帧标识张量，保留原有编码与预测流程
        observed_positions = torch.zeros((B, N, T)).type_as(all_traj)
        observed_positions[:, :, :self.past_timestep] = 1.

        past_future_indices = torch.arange(T)[None].repeat(B, 1).cuda()
        past_future_indices = past_future_indices[:, None, :].repeat(1, N, 1)
        past_ind, future_ind = past_future_indices[:, :, :self.past_timestep], past_future_indices[
            :, :, self.past_timestep:]

        # 进行未来帧预测
        if self.multi_output:
            out = self.encode_decode(all_traj, observed_positions, all_out=True)
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
            decoded_tokens = self.encode_decode(all_traj, observed_positions)
            dec_future_tokens = decoded_tokens[batch_ind, agent_ind, future_ind, :]
            pred_future_coord_values = self.head(dec_future_tokens)
            pred_future_coord_values = pred_future_coord_values.view(B, N, -1, 3)
            if self.add_residual:
                pred_future_coord_values += start

        # 返回格式保持兼容，但额外返回值保持原有占位结构
        # 这样可以保持与训练代码的接口兼容
        placeholder_output_1 = torch.zeros_like(all_traj)
        placeholder_output_2 = torch.zeros_like(all_traj)
        placeholder_output_3 = torch.zeros(B, N, self.past_timestep, 3).type_as(all_traj)
        placeholder_output_4 = torch.zeros((B, N, T)).type_as(all_traj)

        return pred_future_coord_values, placeholder_output_1, placeholder_output_2, placeholder_output_3, placeholder_output_4

    def encode_decode(self, all_trajs, context, all_out=False):
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

        future_positions = 1 - context
        observed_tokens = self.patch_embed(all_traj_input) * context[:, :, :, None]
        observed_tokens += self.pos_embed.repeat(B, N, 1, 1)

        if self.add_joint_token:
            observed_tokens += self.agent_embed.repeat(B, 1, T, 1)

        observed_tokens = observed_tokens * context[:, :, :, None]

        context_s = context.permute(0, 2, 1).contiguous().view(B * T, N)
        context_s = torch.matmul(context_s[:, :, None], context_s[:, None, :])
        context_t = context.contiguous().view(B * N, T)
        context_t = torch.matmul(context_t[:, :, None], context_t[:, None, :])

        observed_tokens_pad = observed_tokens
        time_indices = torch.arange(T)[None, None, :].repeat(B, N, 1).cuda()

        out = []
        for l in range(self.encoder_depth):
            if l == 0:
                encoded_tokens = self.encoder[l](observed_tokens_pad, context_s, context_t)

                enc_to_dec_tokens = encoded_tokens * context[:, :, :, None]
                future_tokens = self.future_embed[None, None, None, :].repeat(B, N, T, 1)
                future_tokens += self.decoder_pos_embed(time_indices)

                if self.add_joint_token:
                    future_tokens += self.decoder_agent_embed(agent_ind.repeat(B, 1, future_tokens.shape[2]))

                future_tokens = future_tokens * future_positions[:, :, :, None]
                concat_tokens = enc_to_dec_tokens + future_tokens

                dec_input_tokens = concat_tokens

                if self.decoder_context:
                    decoded_tokens = self.decoder[l](dec_input_tokens, 1 - context_s, 1 - context_t)
                else:
                    decoded_tokens = self.decoder[l](dec_input_tokens)
                out.append(decoded_tokens)
            else:
                encoder_input = decoded_tokens

                encoder_input_pad = encoder_input * context[:, :, :, None]
                encoder_output = self.encoder[l](encoder_input_pad, context_s, context_t)
                decoded_tokens = encoder_output * context[:, :, :, None] + decoded_tokens * future_positions[:, :, :, None]
                if self.decoder_context:
                    decoded_tokens = self.decoder[l](decoded_tokens, 1 - context_s, 1 - context_t)
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

        observed_positions = torch.zeros((B, N, T)).type_as(all_traj)
        observed_positions[:, :, :self.past_timestep] = 1.

        if self.multi_output:
            decoded_tokens = self.encode_decode(all_traj, observed_positions, all_out=True)[-1]
        else:
            decoded_tokens = self.encode_decode(all_traj, observed_positions)

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
            self.transformer_t.append(TransformerLayers(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
            self.transformer_s.append(TransformerLayers(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
        self.transformer_t = nn.ModuleList(self.transformer_t)
        self.transformer_s = nn.ModuleList(self.transformer_s)

        self.pool = pool

    def forward(self, x, context_s=None, context_t=None):
        B, N = x.shape[0], x.shape[1]
        out = []
        for i in range(self.depth):
            x = x.contiguous().view(B * N, -1, x.shape[-1])
            x = self.transformer_t[i](x, context_t)
            x = x.view(B, N, -1, x.shape[-1]).permute(0, 2, 1, 3)
            x = x.contiguous().view(-1, N, x.shape[-1])
            x = self.transformer_s[i](x, context_s)
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

    def forward(self, x, context=None):
        b, l, d = x.shape

        qkv = self.to_qkv(x)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.chunk(3)

        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        if context is not None:
            context = context[:, None, :, :].repeat(1, self.num_heads, 1, 1)
            attn = attn * context
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


class TransformerLayers(nn.Module):
    def __init__(self, dim, mlp_dim, depth=1, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x, context=context)
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
