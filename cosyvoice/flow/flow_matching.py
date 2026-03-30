# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM
from cosyvoice.utils.common import set_all_random_seed


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16 inference, so set dtype=spks.dtype
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + 0.5) * dphi_dt - 0.5 * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            # sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x.float()

    def solve_euler_BNS(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        b=mu.size(0)
        x=torch.tile(x, (2, 1, 1))
        x0=x.clone()
        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([b*2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([b*2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([b*2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([b*2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([b*2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([b*2, 80, x.size(2)], device=x.device, dtype=x.dtype)

        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask.tile(2, 1, 1)
            mu_in[:b] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[:b] = spks
            cond_in[:b] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [b, b], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            
            x = x + dt * dphi_dt.tile(2, 1, 1)
            t = t + dt
            # sol.append()
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t


        return x[:b].float()

    def solve_euler_mid(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            cache: cache dictionary for streaming inference
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        b = x.size(0)
        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([b * 2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([b * 2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([b * 2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([b * 2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([b * 2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([b * 2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        next_step_velocity = None

        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            if next_step_velocity is None:
                x_in[:] = x
                mask_in[:] = mask.tile(2, 1, 1)
                mu_in[:b] = mu
                t_in[:] = t.unsqueeze(0)
                spks_in[:b] = spks
                cond_in[:b] = cond
                dphi_dt = self.forward_estimator(
                    x_in, mask_in,
                    mu_in, t_in,
                    spks_in,
                    cond_in,
                    streaming
                )
                # NOTE if smaller than flow_cache_size, means last chunk, no need to cache

                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                dphi_dt = ((1.0 + 0.5) * dphi_dt - 0.5 * cfg_dphi_dt)
            else:
                dphi_dt=next_step_velocity
                
            mid_x=x + 0.5*dt*dphi_dt

            x_in[:] = mid_x
            t_in[:] = t.unsqueeze(0)
            t_in2=t_in+0.5*dt
            # dphi_dt, cache_step = self.forward_estimator(
            #     x_in, mask_in,
            #     mu_in, t_in2,
            #     spks_in,
            #     cond_in,
            #     cache_step
            # )
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in2,
                spks_in,
                cond_in,
                streaming
            )
            # NOTE if smaller than flow_cache_size, means last chunk, no need to cache

            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + 0.5) * dphi_dt - 0.5 * cfg_dphi_dt)
            next_step_velocity = dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
                
                
        
        return x.float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            with stream:
                estimator.set_input_shape('x', tuple(x.shape))
                estimator.set_input_shape('mask', tuple(mask.shape))
                estimator.set_input_shape('mu', tuple(mu.shape))
                estimator.set_input_shape('t', tuple(t.shape))
                estimator.set_input_shape('spks', tuple(spks.shape))
                estimator.set_input_shape('cond', tuple(cond.shape))
                # IMPORTANT:
                # Bind by explicit tensor names, not by engine index order.
                # Tensor order is not guaranteed to be stable across TensorRT versions/builds,
                # and mis-binding leads to silent corruption/NaNs (commonly observed in fp16).
                #
                # Also avoid in-place output into `x` unless the engine explicitly supports it.
                # Allocate a dedicated output buffer.
                x_c = x.contiguous()
                mask_c = mask.contiguous()
                mu_c = mu.contiguous()
                t_c = t.contiguous()
                spks_c = spks.contiguous()
                cond_c = cond.contiguous()
                out = torch.empty_like(x_c)

                estimator.set_tensor_address('x', x_c.data_ptr())
                estimator.set_tensor_address('mask', mask_c.data_ptr())
                estimator.set_tensor_address('mu', mu_c.data_ptr())
                estimator.set_tensor_address('t', t_c.data_ptr())
                estimator.set_tensor_address('spks', spks_c.data_ptr())
                estimator.set_tensor_address('cond', cond_c.data_ptr())
                estimator.set_tensor_address('estimator_out', out.data_ptr())
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return out

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)

        self.rand_noise = torch.randn([1, 80, 50 * 300])*0.9

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        set_all_random_seed(0)
        # z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature*0.9
        # z=z.tile(mu.size(0), 1, 1)
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler_BNS(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None


class CacheCausalConditionalCFM(ConditionalCFM):
    """Cache-based CausalConditionalCFM for CausalMaskedDiffWithXvec.

    Uses CausalConditionalDecoder (UNet) with forward_chunk + KV/conv cache.
    Supports batch B >= 1 (CFG doubling: 2*B).
    Ported from CosyVoice_streaming_HQ_v3.
    """

    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300]) * 0.9

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, cache={}):
        """Cache-based forward diffusion with batch support.

        Args:
            mu: (B, 80, T) encoder output
            mask: (B, 1, T)
            n_timesteps: ODE solver steps
            spks: (B, 80)
            cond: (B, 80, T)
            cache: dict with 'offset' + decoder KV/conv caches (per-timestep)
        Returns:
            mel: (B, 80, T)
            cache: updated cache dict
        """
        offset = cache.pop('offset')
        b = mu.size(0)
        z = self.rand_noise[:, :, :mu.size(2) + offset].to(mu.device).to(mu.dtype) * temperature
        z = z[:, :, offset:]
        if b > 1:
            z = z.expand(b, -1, -1).contiguous()
        offset += mu.size(2)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        mel, cache = self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, cache=cache)
        cache['offset'] = offset
        return mel, cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, cache):
        """Euler solver with decoder cache. Batch B >= 1 (CFG -> 2*B)."""
        b = x.size(0)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        x_in = torch.zeros([2 * b, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2 * b, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2 * b, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2 * b, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2 * b, 80, x.size(2)], device=x.device, dtype=x.dtype)
        flow_cache_size = cache['down_blocks_kv_cache'].shape[4]

        for step in range(1, len(t_span)):
            x_in[:b] = x
            x_in[b:] = x
            mask_in[:b] = mask
            mask_in[b:] = mask
            mu_in[:b] = mu
            t_in[:] = t
            spks_in[:b] = spks
            cond_in[:b] = cond

            cache_step = {k: v[step - 1] for k, v in cache.items()}
            dphi_dt, cache_step = self.forward_estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, cache_step
            )

            if flow_cache_size != 0 and x_in.shape[2] >= flow_cache_size:
                cache['down_blocks_conv_cache'][step - 1] = cache_step[0]
                cache['down_blocks_kv_cache'][step - 1] = cache_step[1][:, :, :, -flow_cache_size:]
                cache['mid_blocks_conv_cache'][step - 1] = cache_step[2]
                cache['mid_blocks_kv_cache'][step - 1] = cache_step[3][:, :, :, -flow_cache_size:]
                cache['up_blocks_conv_cache'][step - 1] = cache_step[4]
                cache['up_blocks_kv_cache'][step - 1] = cache_step[5][:, :, :, -flow_cache_size:]
                cache['final_blocks_conv_cache'][step - 1] = cache_step[6]

            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [b, b], dim=0)
            dphi_dt = ((1.0 + 0.5) * dphi_dt - 0.5 * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x.float(), cache

    def forward_estimator(self, x, mask, mu, t, spks, cond, cache):
        """Call CausalConditionalDecoder.forward_chunk with cache (PyTorch or TRT)."""
        if isinstance(self.estimator, torch.nn.Module):
            x, cache1, cache2, cache3, cache4, cache5, cache6, cache7 = \
                self.estimator.forward_chunk(x, mask, mu, t, spks, cond, **cache)
            cache = (cache1, cache2, cache3, cache4, cache5, cache6, cache7)
            return x, cache
        else:
            # ── TRT path for cache-based UNet decoder ──
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            torch.cuda.current_stream().synchronize()
            with torch.cuda.device(x.device), stream:
                seq_len = x.size(2)
                cache_keys = ['down_blocks_conv_cache', 'down_blocks_kv_cache',
                              'mid_blocks_conv_cache', 'mid_blocks_kv_cache',
                              'up_blocks_conv_cache', 'up_blocks_kv_cache',
                              'final_blocks_conv_cache']
                # Dynamic input shapes — ALL dynamic inputs must be set
                cfg_batch = x.size(0)
                estimator.set_input_shape('x', tuple(x.shape))
                estimator.set_input_shape('mask', tuple(mask.shape))
                estimator.set_input_shape('mu', tuple(mu.shape))
                estimator.set_input_shape('t', tuple(t.shape))
                estimator.set_input_shape('spks', tuple(spks.shape))
                estimator.set_input_shape('cond', tuple(cond.shape))
                for k in cache_keys:
                    estimator.set_input_shape(k, tuple(cache[k].shape))
                estimator.infer_shapes()
                # Allocate output tensors
                x_out = torch.empty((cfg_batch, 80, seq_len), dtype=x.dtype, device=x.device)
                cache_outs = []
                for k in cache_keys:
                    out_shape = tuple(estimator.get_tensor_shape(k + '_out'))
                    cache_outs.append(torch.empty(out_shape, dtype=cache[k].dtype, device=cache[k].device))
                # Bind inputs (keep contiguous refs alive)
                x_cont = x.contiguous()
                mask_cont = mask.contiguous()
                mu_cont = mu.contiguous()
                t_cont = t.contiguous()
                spks_cont = spks.contiguous()
                cond_cont = cond.contiguous()
                cache_conts = {k: cache[k].contiguous() for k in cache_keys}
                estimator.set_tensor_address('x', x_cont.data_ptr())
                estimator.set_tensor_address('mask', mask_cont.data_ptr())
                estimator.set_tensor_address('mu', mu_cont.data_ptr())
                estimator.set_tensor_address('t', t_cont.data_ptr())
                estimator.set_tensor_address('spks', spks_cont.data_ptr())
                estimator.set_tensor_address('cond', cond_cont.data_ptr())
                for k in cache_keys:
                    estimator.set_tensor_address(k, cache_conts[k].data_ptr())
                # Bind outputs
                estimator.set_tensor_address('estimator_out', x_out.data_ptr())
                for k, c_out in zip(cache_keys, cache_outs):
                    estimator.set_tensor_address(k + '_out', c_out.data_ptr())
                # Execute
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return x_out, tuple(cache_outs)
