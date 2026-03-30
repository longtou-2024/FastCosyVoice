# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li, Qihua, Shengqiang Li)
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
import queue
import random
import time
import threading
from typing import Dict, Optional, Callable, List, Generator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM,AutoModel,AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.cache_utils import Cache, DynamicCache
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path
import tqdm

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = self.speech_token_size
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID, 
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        # weighted_scores=weighted_scores.squeeze(0)
        # print('weighted_scores',weighted_scores.size())
        if weighted_scores.dim() == 1:
            num_trials, max_trials = 0, 100
            while True:
                top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
                if (not ignore_eos) or (top_ids < self.speech_token_size):
                    break
                num_trials += 1
                if num_trials > max_trials:
                    raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
            return top_ids

        if weighted_scores.dim() != 2:
            raise ValueError(f'weighted_scores must be 1D or 2D, got shape {tuple(weighted_scores.shape)}')
        batch_size = weighted_scores.size(0)
        if isinstance(ignore_eos, torch.Tensor):
            ignore_eos_list = ignore_eos.detach().cpu().reshape(-1).tolist()
        elif isinstance(ignore_eos, (list, tuple)):
            ignore_eos_list = list(ignore_eos)
        else:
            ignore_eos_list = [ignore_eos] * batch_size
        if len(ignore_eos_list) != batch_size:
            raise ValueError(f'ignore_eos batch size mismatch: expected {batch_size}, got {len(ignore_eos_list)}')

        if len(decoded_tokens) == 0:
            decoded_tokens_batch = [[] for _ in range(batch_size)]
        elif isinstance(decoded_tokens[0], list):
            decoded_tokens_batch = decoded_tokens
        else:
            decoded_tokens_batch = [decoded_tokens] * batch_size

        sampled_ids = []
        for i in range(batch_size):
            num_trials, max_trials = 0, 100
            # print('weighted_scores[i].shape',weighted_scores[i].size())
            # print('decoded_tokens_batch[i].shape',decoded_tokens_batch[i])
            while True:
                top_ids = self.sampling(weighted_scores[i], decoded_tokens_batch[i], sampling)
                if (not ignore_eos_list[i]) or (top_ids < self.speech_token_size):
                    break
                num_trials += 1
                if num_trials > max_trials:
                    raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
            sampled_ids.append(top_ids)
        return torch.tensor(sampled_ids, dtype=torch.int64, device=weighted_scores.device)

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False)
            if top_ids == self.eos_token:
                break
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)


# class Qwen2Encoder(torch.nn.Module):
#     def __init__(self, pretrain_path):
#         super().__init__()
#         self.model = AutoModel.from_pretrained(pretrain_path)

#     def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
#         T = xs.size(1)
#         masks = ~make_pad_mask(xs_lens, T)
#         outs = self.model(
#             inputs_embeds=xs,
#             attention_mask=masks,
#             use_cache=False,
#             output_hidden_states=False,
#             return_dict=True,
#         )
#         return outs.last_hidden_state, masks.unsqueeze(1)


class Qwen2Encoder3(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrain_path).eval()

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outs.last_hidden_state, masks.unsqueeze(1)

class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrain_path)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outs.last_hidden_state, masks.unsqueeze(1)
    #     self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    # def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
    #     T = xs.size(1)
    #     masks = ~make_pad_mask(xs_lens, T)
    #     outs = self.model(
    #         inputs_embeds=xs,
    #         attention_mask=masks,
    #         output_hidden_states=True,
    #         return_dict=True,
    #     )
    #     return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        if masks.dim() == 2:
            input_masks = masks
        else:
            input_masks = masks[:, -1, :]
        if cache is not None and not isinstance(cache, Cache):
            cache = DynamicCache.from_legacy_cache(cache)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache


# class Qwen2Encoder(torch.nn.Module):
#     def __init__(self, pretrain_path):
#         super().__init__()
#         self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

#     def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
#         T = xs.size(1)
#         masks = ~make_pad_mask(xs_lens, T)
#         outs = self.model(
#             inputs_embeds=xs,
#             attention_mask=masks,
#             output_hidden_states=True,
#             return_dict=True,
#         )
#         return outs.hidden_states[-1], masks.unsqueeze(1)

#     def forward_one_step(self, xs, masks, cache=None):
#         if masks.dim() == 2:
#             input_masks = masks
#         else:
#             input_masks = masks[:, -1, :]
#         if cache is not None and not isinstance(cache, Cache):
#             cache = DynamicCache.from_legacy_cache(cache)
#         outs = self.model(
#             inputs_embeds=xs,
#             attention_mask=input_masks,
#             output_hidden_states=True,
#             return_dict=True,
#             use_cache=True,
#             past_key_values=cache,
#         )
#         xs = outs.hidden_states[-1]
#         new_cache = outs.past_key_values
#         return xs, new_cache
    # def forward_one_step(self, xs, masks, cache=None):
    #     if masks.dim() == 2:
    #         input_masks = masks
    #     else:
    #         input_masks = masks[:, -1, :]
    #     if cache is not None and not isinstance(cache, Cache):
    #         cache = DynamicCache.from_legacy_cache(cache)
    #     outs = self.model(
    #         inputs_embeds=xs,
    #         attention_mask=input_masks,
    #         output_hidden_states=True,
    #         return_dict=True,
    #         use_cache=True,
    #         past_key_values=cache,
    #     )
    #     xs = outs.hidden_states[-1]
    #     new_cache = outs.past_key_values
    #     return xs, new_cache







class Qwen2LM(TransformerLM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = 0
        self.task_id = 1
        self.eos_token = speech_token_size
        self.fill_token = speech_token_size + 2

        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.vllm_output_queue = {}

    def _expand_to_batch(self, tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor.reshape(1).expand(batch_size)
        if tensor.size(0) == batch_size:
            return tensor
        if tensor.size(0) == 1:
            expand_sizes = [batch_size] + [-1] * (tensor.dim() - 1)
            return tensor.expand(*expand_sizes)
        raise ValueError(f'{name} batch size mismatch: expected 1 or {batch_size}, got {tensor.size(0)}')

    def _build_decode_input(self, parts_per_sample: List[List[torch.Tensor]], device: torch.device):
        lm_input = [torch.concat(parts, dim=0) for parts in parts_per_sample]
        lm_input_len = torch.tensor([item.size(0) for item in lm_input], dtype=torch.int32, device=device)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=0.0)
        return lm_input, lm_input_len

    def _gather_last_valid_logits(self, y_pred: torch.Tensor, input_lens: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(y_pred.size(0), device=y_pred.device)
        last_indices = (input_lens.to(device=y_pred.device, dtype=torch.long) - 1).clamp_min(0)
        return y_pred[batch_indices, last_indices]

    def _select_cache(self, cache, keep_indices: torch.Tensor):
        if cache is None:
            return None

        keep_indices = keep_indices.to(dtype=torch.long)
        if isinstance(cache, Cache):
            cache.batch_select_indices(keep_indices)
            return cache

        selected_cache = []
        for layer_cache in cache:
            if isinstance(layer_cache, tuple):
                selected_cache.append(tuple(
                    tensor.index_select(0, keep_indices) if torch.is_tensor(tensor) else tensor
                    for tensor in layer_cache
                ))
            else:
                selected_cache.append(layer_cache.index_select(0, keep_indices) if torch.is_tensor(layer_cache) else layer_cache)
        return DynamicCache.from_legacy_cache(tuple(selected_cache))

    def prepare_lm_input_target(self, sos_emb, text_token, text_token_emb, text_token_len, task_id_emb, speech_token, speech_token_emb, speech_token_len,caption_token,caption_lengths, instruct_token=None, instruct_token_emb=None, instruct_token_len=None):
        lm_target, lm_input = [], []
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        text_token_emb = unpad_sequence(text_token_emb, text_token_len.cpu(), batch_first=True)
        speech_token_emb = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)
        caption_token = unpad_sequence(caption_token, caption_lengths.cpu(), batch_first=True)

        # NOTE add instruct_token in CosyVoice3
        if instruct_token is not None and instruct_token_emb is not None and instruct_token_len is not None:
            instruct_token = unpad_sequence(instruct_token, instruct_token_len.cpu(), batch_first=True)
            instruct_token_emb = unpad_sequence(instruct_token_emb, instruct_token_len.cpu(), batch_first=True)
        else:
            instruct_token = [torch.empty(0).to(text_token[0])] * len(text_token)
            instruct_token_emb = [torch.empty(0, 896).to(text_token_emb[0])] * len(text_token)
            instruct_token_len = torch.zeros(len(text_token)).to(text_token_len)
        for i in range(len(text_token)):
            # bistream sequence
            if random.random() < 0.5 and speech_token_len[i] / text_token_len[i] > self.mix_ratio[1] / self.mix_ratio[0]:
                this_lm_target, this_lm_input = [IGNORE_ID], [sos_emb.squeeze(dim=0)]
                this_lm_target += [IGNORE_ID] * instruct_token_len[i]
                this_lm_input.append(instruct_token_emb[i])
                
                if caption_lengths[i] !=0:
                    this_lm_target+= [IGNORE_ID]*caption_lengths[i]
                    this_lm_input.append(caption_token[i])
                
                
                
                
                for j in range(((text_token_len[i] + 1) / self.mix_ratio[0]).ceil().int().item()):
                    this_text_token = text_token[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]].tolist()
                    this_speech_token = speech_token[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]].tolist()
                    if len(this_text_token) == self.mix_ratio[0]:
                        assert len(this_speech_token) == self.mix_ratio[1]
                        this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
                        this_lm_target += this_speech_token
                        this_lm_target.append(self.fill_token)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]])
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]])
                    else:
                        this_lm_target += [-1] * len(this_text_token)
                        this_lm_target += speech_token[i][j * self.mix_ratio[1]:].tolist()
                        this_lm_target.append(self.eos_token)
                        this_lm_input.append(text_token_emb[i][j * self.mix_ratio[0]:])
                        this_lm_input.append(task_id_emb.squeeze(dim=0))
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]:])
                this_lm_target, this_lm_input = torch.tensor(this_lm_target), torch.concat(this_lm_input, dim=0)
            # unistream sequence
            else:
                
                if caption_lengths[i] !=0:
                    this_lm_target = torch.tensor([IGNORE_ID] * (1 +caption_lengths[i] +  text_token_len[i]) + speech_token[i].tolist() + [self.eos_token])
                    this_lm_input = torch.concat([sos_emb.squeeze(dim=0), caption_token[i], text_token_emb[i], task_id_emb.squeeze(dim=0), speech_token_emb[i]], dim=0)
                else:                
                    this_lm_target = torch.tensor([IGNORE_ID] * (1 + instruct_token_len[i] + text_token_len[i]) + speech_token[i].tolist() + [self.eos_token])
                    this_lm_input = torch.concat([sos_emb.squeeze(dim=0), instruct_token_emb[i], text_token_emb[i], task_id_emb.squeeze(dim=0), speech_token_emb[i]], dim=0)
            lm_target.append(this_lm_target)
            lm_input.append(this_lm_input)
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID)
        return lm_target, lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        LM_latents = batch['LM_latents'].to(device)
        caption_lengths = batch['caption_lengths'].to(device)

        # 1. encode text_token
        text_token_emb = self.llm.model.embed_tokens(text_token)
        self.caption_
        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(sos_emb, text_token, text_token_emb, text_token_len, task_id_emb,
                                                                         speech_token, speech_token_emb, speech_token_len)
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target.to(device))
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def forward_dpo(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        reject_speech_token = batch['reject_speech_token'].to(device)
        reject_speech_token_len = batch['reject_speech_token_len'].to(device)

        # 1. encode text_token
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        # 3. sos and task_id
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 2. encode speech_token
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        reject_speech_token = unpad_sequence(reject_speech_token, reject_speech_token_len.cpu(), batch_first=True)
        speech_token_combined = speech_token + reject_speech_token
        speech_token_combined = pad_sequence(speech_token_combined, batch_first=True, padding_value=0)
        speech_token_combined_len = torch.concat([speech_token_len, reject_speech_token_len], dim=0)
        speech_token_combined_emb = self.speech_embedding(speech_token_combined)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(sos_emb, text_token.repeat(2, 1), text_token_emb.repeat(2, 1, 1), text_token_len.repeat(2),
                                                                         task_id_emb, speech_token_combined, speech_token_combined_emb, speech_token_combined_len)
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        chosen_logits = logits[:text_token.shape[0]]
        rejected_logits = logits[text_token.shape[0]:]
        chosen_lm_target = lm_target[:text_token.shape[0]]
        rejected_lm_target = lm_target[text_token.shape[0]:]
        loss = self.criterion_ce(chosen_logits, chosen_lm_target.to(device))
        acc = th_accuracy(chosen_logits.view(-1, self.speech_token_size + 3), chosen_lm_target, ignore_label=IGNORE_ID)

        # 5. calculate dpo logits
        chosen_lm_mask = chosen_lm_target == IGNORE_ID
        rejected_lm_mask = rejected_lm_target == IGNORE_ID
        chosen_logps = torch.gather(chosen_logits.log_softmax(dim=-1), dim=2, index=chosen_lm_target.masked_fill(chosen_lm_mask, 0).unsqueeze(dim=-1)).squeeze(dim=-1)
        rejected_logps = torch.gather(rejected_logits.log_softmax(dim=-1), dim=2, index=rejected_lm_target.masked_fill(rejected_lm_mask, 0).unsqueeze(dim=-1)).squeeze(dim=-1)
        chosen_logps = (chosen_logps * chosen_lm_mask).sum(dim=-1) / chosen_lm_mask.sum(dim=-1)
        rejected_logps = (rejected_logps * rejected_lm_mask).sum(dim=-1) / rejected_lm_mask.sum(dim=-1)
        return {'loss': loss, 'acc': acc, 'chosen_logps': chosen_logps, 'rejected_logps': rejected_logps}

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        batch_size = text.size(0)
        prompt_text = self._expand_to_batch(prompt_text, batch_size, 'prompt_text')
        prompt_text_len = self._expand_to_batch(prompt_text_len, batch_size, 'prompt_text_len')
        prompt_speech_token = self._expand_to_batch(prompt_speech_token, batch_size, 'prompt_speech_token')
        prompt_speech_token_len = self._expand_to_batch(prompt_speech_token_len, batch_size, 'prompt_speech_token_len')

        text = torch.concat([prompt_text, text], dim=1)
        text_len = text_len + prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 3. concat llm_input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1).expand(batch_size, -1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1).expand(batch_size, -1, -1)
        if prompt_speech_token.size(1) > 0 and torch.any(prompt_speech_token_len > 0):
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(batch_size, 0, self.llm_input_size, dtype=text.dtype, device=device)
        lm_parts = []
        for i in range(batch_size):
            lm_parts.append([
                sos_emb[i],
                text[i, :int(text_len[i].item())],
                task_id_emb[i],
                prompt_speech_token_emb[i, :int(prompt_speech_token_len[i].item())],
            ])
        lm_input, lm_input_len = self._build_decode_input(lm_parts, device)

        # 4. cal min/max_length
        decode_text_len = (text_len - prompt_text_len).to(torch.float32)
        min_len = torch.floor(decode_text_len * min_token_text_ratio).to(torch.int64).clamp_min(0)
        max_len = torch.ceil(decode_text_len * max_token_text_ratio).to(torch.int64).clamp_min(1)

        # 5. step by step decode
        for token in self.inference_wrapper(lm_input, lm_input_len, sampling, min_len, max_len, uuid):
            yield token

    @torch.inference_mode()
    def inference_wrapper(self, lm_input, lm_input_len, sampling, min_len, max_len, uuid, initial_cache=None):
        batch_size = lm_input.size(0)
        if hasattr(self, 'vllm'):
            if batch_size != 1:
                raise NotImplementedError('vLLM inference_wrapper currently supports batch_size=1 only')
            from vllm import SamplingParams, RequestOutput
            sampling_params = SamplingParams(top_k=sampling,
                                             stop_token_ids=self.stop_token_ids,
                                             min_tokens=int(min_len[0].item()),
                                             max_tokens=int(max_len[0].item()))
            with self.lock:
                self.vllm.add_request(uuid, {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16).to(lm_input.device)}, sampling_params)
                self.vllm_output_queue[uuid] = queue.Queue()
            out_tokens = []
            while True:
                with self.lock:
                    if self.vllm_output_queue[uuid].empty() is True:
                        request_outputs: List[RequestOutput] = self.vllm.step()
                        for request_output in request_outputs:
                            top_ids = list(request_output.outputs[0].token_ids)[-1]
                            self.vllm_output_queue[request_output.request_id].put(top_ids)
                if self.vllm_output_queue[uuid].empty() is False:
                    top_ids = self.vllm_output_queue[uuid].get()
                    if top_ids in self.stop_token_ids:
                        break
                    # in stream mode, yield token one by one
                    yield top_ids
                    out_tokens.append(top_ids)
                    if len(out_tokens) == int(max_len[0].item()):
                        break
                time.sleep(0.001)
            with self.lock:
                self.vllm_output_queue.pop(uuid)
        else:
            decoded_tokens = [[] for _ in range(batch_size)]
            active_indices = torch.arange(batch_size, device=lm_input.device, dtype=torch.long)
            active_input = lm_input
            active_input_len = lm_input_len.to(device=lm_input.device, dtype=torch.int64)
            cache = initial_cache
            max_steps = int(max_len.max().item())
            for i in range(max_steps):
                expired_mask = i >= max_len.index_select(0, active_indices)
                if torch.any(expired_mask):
                    keep_mask = ~expired_mask
                    if not torch.any(keep_mask):
                        break
                    if cache is not None:
                        cache = self._select_cache(cache, keep_mask.nonzero(as_tuple=False).squeeze(1))
                    active_indices = active_indices[keep_mask]
                    active_input = active_input[keep_mask]
                    active_input_len = active_input_len[keep_mask]
                # Build attention mask covering past (cached) + current positions
                past_len = cache.get_seq_length() if cache is not None else 0
                total_len = past_len + active_input.size(1)
                valid_mask = torch.arange(total_len, device=active_input.device).unsqueeze(0) < (past_len + active_input_len).unsqueeze(1)
                y_pred, cache = self.llm.forward_one_step(active_input, masks=valid_mask, cache=cache)
                last_hidden = self._gather_last_valid_logits(y_pred, active_input_len)
                logp = self.llm_decoder(last_hidden).log_softmax(dim=-1)
                ignore_eos = i < min_len.index_select(0, active_indices)
                active_histories = [decoded_tokens[idx] for idx in active_indices.tolist()]
                top_ids = self.sampling_ids(logp, active_histories, sampling, ignore_eos=ignore_eos)
                if not torch.is_tensor(top_ids):
                    top_ids = torch.tensor([top_ids], dtype=torch.int64, device=logp.device)
                stop_ids = self._stop_ids_tensor
                stop_mask = (top_ids.unsqueeze(1) == stop_ids.unsqueeze(0)).any(dim=1)

                if batch_size == 1:
                    if stop_mask[0]:
                        break
                    token = int(top_ids[0].item())
                    yield token
                    decoded_tokens[0].append(token)
                else:
                    step_tokens = torch.full((batch_size,), IGNORE_ID, dtype=torch.int32, device=top_ids.device)
                    finished_indices = active_indices[stop_mask]
                    live_tokens = top_ids[~stop_mask].to(torch.int32)
                    if finished_indices.numel() > 0:
                        step_tokens.index_fill_(0, finished_indices, IGNORE_ID - 1)
                    step_tokens.index_copy_(0, active_indices[~stop_mask], live_tokens)
                    if live_tokens.numel() > 0 or finished_indices.numel() > 0:
                        yield step_tokens
                    for batch_idx, token in zip(active_indices[~stop_mask].tolist(), live_tokens.tolist()):
                        decoded_tokens[batch_idx].append(token)

                if torch.all(stop_mask):
                    break

                keep_mask = ~stop_mask
                kept_indices = active_indices[keep_mask]
                if cache is not None and keep_mask.sum().item() != keep_mask.numel():
                    cache = self._select_cache(cache, keep_mask.nonzero(as_tuple=False).squeeze(1))
                active_indices = kept_indices
                active_input = self.speech_embedding(top_ids[keep_mask]).unsqueeze(1)
                active_input_len = torch.ones(active_input.size(0), dtype=torch.int64, device=active_input.device)

    @torch.inference_mode()
    def inference_bistream(
            self,
            text: Generator,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ) -> Generator[torch.Tensor, None, None]:

        device = prompt_text.device
        # 1. prepare input
        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_text.dtype).to(device)
        lm_input = torch.concat([sos_emb], dim=1)

        # 2. iterate text
        out_tokens = []
        cache = None
        # NOTE init prompt_text as text_cache as it is basically impossible prompt_speech_token/prompt_text < 15/5
        text_cache = self.llm.model.model.embed_tokens(prompt_text)
        next_fill_index = (int(prompt_speech_token.shape[1] / self.mix_ratio[1]) + 1) * self.mix_ratio[1] - prompt_speech_token.shape[1]
        for this_text in text:
            text_cache = torch.concat([text_cache, self.llm.model.model.embed_tokens(this_text)], dim=1)
            # prompt_speech_token_emb not empty, try append to lm_input
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = text_cache[:, :self.mix_ratio[0]], prompt_speech_token_emb[:, :self.mix_ratio[1]]
                    logging.info('append {} text token {} speech token'.format(lm_input_text.size(1), lm_input_speech.size(1)))
                    lm_input = torch.concat([lm_input, lm_input_text, lm_input_speech], dim=1)
                    text_cache, prompt_speech_token_emb = text_cache[:, self.mix_ratio[0]:], prompt_speech_token_emb[:, self.mix_ratio[1]:]
                else:
                    logging.info('not enough text token to decode, wait for more')
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.fill_token) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                    logging.info('get fill token, need to append more text token')
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, :self.mix_ratio[0]]
                        logging.info('append {} text token'.format(lm_input_text.size(1)))
                        if len(out_tokens) != 0 and out_tokens[-1] == self.fill_token:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        continue
                while True:
                    seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
                    y_pred, cache = self.llm.forward_one_step(lm_input,
                                                              masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                              cache=cache)
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.fill_token
                        next_fill_index += (self.mix_ratio[1] + 1)
                    else:
                        top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True)
                    if top_ids == self.fill_token:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info('fill_token index {} next fill_token index {}'.format(len(out_tokens), next_fill_index))
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.fill_token:
                            break
                        else:
                            raise ValueError('should not get token {}'.format(top_ids))
                    yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info('no more text token, decode until met eos')
        while True:
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False)
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.eos_token:
                    break
                else:
                    raise ValueError('should not get token {}'.format(top_ids))
            # in stream mode, yield token one by one
            yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)





class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.w1 = nn.Linear(dim, dim*4, bias=False)
        self.w2 = nn.Linear(dim*4, dim, bias=False)
        self.drouput= nn.Dropout(0.1)
        self.w3 = nn.Linear(dim, dim*4, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))  # type: ignore




class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)*0.01)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class MoeLayer(torch.nn.Module):
    def __init__(self, experts: List[torch.nn.Module], gate: torch.nn.Module,llm_input_size: int):
        super().__init__()
        assert len(experts) > 0
        self.experts = torch.nn.ModuleList(experts)
        self.gate=gate
        self.norm=RMSNorm(llm_input_size)
        self.drouput= nn.Dropout(0.1)
        self.proj=nn.Linear(2048,llm_input_size,bias=False)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = inputs.shape
        x_flat = inputs.contiguous().view(-1, input_dim)
        gate_logits = self.gate(x_flat)
        
        weights, selected_experts = gate_logits.topk(2, dim=1)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            batch_idx,nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(x_flat[batch_idx])
        return self.norm(self.proj(results.view_as(inputs)))

class CosyVoice3LM(Qwen2LM):
    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        self.sos = speech_token_size + 0
        self.eos_token = speech_token_size + 1
        self.task_id = speech_token_size + 2
        self.fill_token = speech_token_size + 3

        self.llm = llm
        
        self.llm_caption = MoeLayer(
        experts=[FeedForward(dim=2048, hidden_dim=2048) for _ in range(4)],
        gate=nn.Linear(2048, 4, bias=False),
        llm_input_size=llm_input_size)     
        

        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 200, bias=False)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 200,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 200, llm_input_size)
        # self.speech_token_extractor = SpeechTokenExtractor()

        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(200)]
        self.register_buffer('_stop_ids_tensor', torch.tensor(self.stop_token_ids, dtype=torch.int64))
        self.vllm_output_queue = {}

    # ── Prompt KV Cache ──────────────────────────────────────────────────
    @torch.inference_mode()
    def compute_prefix_cache(self, prompt_text, prompt_text_len, LM_latents):
        """Pre-compute KV cache for fixed prefix [SOS | LM_latents | prompt_text_emb].

        Call once per speaker, then pass the returned cache to inference().
        All tensors should already be on the correct device.
        """
        prompt_text_emb = self.llm.model.embed_tokens(prompt_text[:1]).to(prompt_text.device)
        LM_latents_processed = self.llm_caption(LM_latents[:1])

        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        plen = int(prompt_text_len[0].item()) if prompt_text_len.dim() >= 1 else int(prompt_text_len.item())
        prefix = torch.cat([sos_emb, LM_latents_processed, prompt_text_emb[:, :plen]], dim=1)

        valid_mask = torch.ones(1, prefix.size(1), dtype=torch.bool, device=prefix.device)
        _, cache = self.llm.forward_one_step(prefix, masks=valid_mask, cache=None)
        logging.info('[CosyVoice3LM] Prefix cache computed: %d positions', prefix.size(1))
        return cache

    @staticmethod
    def _clone_and_expand_cache(cache, batch_size):
        """Clone a batch_size=1 DynamicCache and expand to *batch_size*."""
        new_cache = DynamicCache()
        for layer_idx in range(len(cache.key_cache)):
            new_cache.update(
                cache.key_cache[layer_idx].expand(batch_size, -1, -1, -1).contiguous(),
                cache.value_cache[layer_idx].expand(batch_size, -1, -1, -1).contiguous(),
                layer_idx,
            )
        return new_cache

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        
        # speech_token, speech_token_len = self.speech_token_extractor.inference(batch['whisper_feat'].transpose(1, 2).to(device), batch['whisper_feat_len'].to(device), device)

        
        # NOTE should append instruct_token to sequence, not implemented yet
        # instruct_token = batch['instruct_token'].to(device)
        # instruct_token_len = batch['instruct_token_len'].to(device)
        caption_token = batch['LM_latents'].to(device)
        
        caption_lengths = batch['caption_lengths'].to(device)
        
        text_token_emb = self.llm.model.embed_tokens(text_token)

        # 1. encode text_token
        # text_token_emb = self.llm.model.model.embed_tokens(text_token)
        # instruct_token_emb = self.llm.model.model.embed_tokens(instruct_token)


        caption_token=self.llm_caption(caption_token)
        
        # 3. sos and task_id
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 2. encode speech_token
        speech_token_emb = self.speech_embedding(speech_token)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(sos_emb, text_token, text_token_emb, text_token_len, task_id_emb,
                                                                         speech_token, speech_token_emb, speech_token_len,caption_token,caption_lengths)
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 200), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            LM_latents: torch.Tensor,
            prefix_cache=None,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ) -> Generator[torch.Tensor, None, None]:
        torch.manual_seed(1986)
        torch.cuda.manual_seed_all(1986)
        device = text.device
        batch_size = text.size(0)
        prompt_speech_token = self._expand_to_batch(prompt_speech_token, batch_size, 'prompt_speech_token')
        prompt_speech_token_len = self._expand_to_batch(prompt_speech_token_len, batch_size, 'prompt_speech_token_len')

        task_id_emb = self.speech_embedding.weight[self.task_id].reshape(1, 1, -1).expand(batch_size, -1, -1)
        if prompt_speech_token.size(1) > 0 and torch.any(prompt_speech_token_len > 0):
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(batch_size, 0, self.llm_input_size, dtype=torch.float32, device=device)

        # if prefix_cache is not None:
        #     # ── Fast path: prefix [SOS | LM_latents | prompt_text_emb] is cached ──
        #     # `text` contains ONLY tts_text (not concatenated with prompt_text)
        #     tts_text_emb = self.llm.model.embed_tokens(text).to(device)
        #     lm_parts = []
        #     for i in range(batch_size):
        #         lm_parts.append([
        #             tts_text_emb[i, :int(text_len[i].item())],
        #             task_id_emb[i],
        #             prompt_speech_token_emb[i, :int(prompt_speech_token_len[i].item())],
        #         ])
        #     lm_input, lm_input_len = self._build_decode_input(lm_parts, device)
        #     initial_cache = self._clone_and_expand_cache(prefix_cache, batch_size)
        #     decode_text_len = text_len.to(torch.float32)
        # else:
            # ── Original path: no prefix cache ────────────────────────────────
        prompt_text = self._expand_to_batch(prompt_text, batch_size, 'prompt_text')
        prompt_text_len = self._expand_to_batch(prompt_text_len, batch_size, 'prompt_text_len')
        LM_latents = self._expand_to_batch(LM_latents, batch_size, 'LM_latents')
        # Embed prompt_text and text separately, then concat per-sample
        # to avoid padding contamination when prompt lengths differ
        prompt_text_emb = self.llm.model.embed_tokens(prompt_text).to(device)
        text_emb = self.llm.model.embed_tokens(text).to(device)
        LM_latents = self.llm_caption(LM_latents)
        sos_emb = self.speech_embedding.weight[self.sos].reshape(1, 1, -1).expand(batch_size, -1, -1)
        lm_parts = []

        for i in range(batch_size):
            pt_len_i = int(prompt_text_len[i].item())
            t_len_i = int(text_len[i].item())
            lm_parts.append([
                sos_emb[i],
                LM_latents[i],
                prompt_text_emb[i, :pt_len_i],
                text_emb[i, :t_len_i],
                task_id_emb[i],
                prompt_speech_token_emb[i, :int(prompt_speech_token_len[i].item())],
            ])
        lm_input, lm_input_len = self._build_decode_input(lm_parts, device)
        initial_cache = None
        decode_text_len = text_len.to(torch.float32)

        # 4. cal min/max_length
        min_len = torch.floor(decode_text_len * min_token_text_ratio).to(torch.int64).clamp_min(0)
        max_len = torch.ceil(decode_text_len * max_token_text_ratio).to(torch.int64).clamp_min(1)

        # 5. step by step decode
        for token in self.inference_wrapper(lm_input, lm_input_len, sampling, min_len, max_len, uuid,
                                            initial_cache=initial_cache):
            yield token
