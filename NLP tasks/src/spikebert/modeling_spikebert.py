# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bert.configuration_bert import BertConfig
from .spiking import SpikeLinear, QuantizeEmbedding, ElasticBiSpiking, AlphaInit
from .attention import shiftmax_exp2, valid_key_mask
from .quantization import MultiSpikeNorm, SignedSSAQuantizer
from .normalization import DynamicTanh, ShiftedDynamicTanh, build_normalization


logger = logging.get_logger(__name__)


class TernarySpikeSTE(torch.autograd.Function):
    """Stateless ternary quantizer with an STE on the interval [-1, 1]."""

    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return torch.round(inputs).clamp(-1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        pass_through = (inputs >= -1) & (inputs <= 1)
        return grad_output * pass_through.to(dtype=grad_output.dtype)


class StatelessTernarySpike(nn.Module):
    """Pure {-1, 0, 1} spike quantization without learnable state."""

    def forward(self, inputs):
        return TernarySpikeSTE.apply(inputs)


class IdentityInitializedLinear(nn.Linear):
    """Bias-free feedback projection whose initial transform is identity."""

    def __init__(self, hidden_size):
        super().__init__(hidden_size, hidden_size, bias=False)
        self._identity_initialized = True
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.weight)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "deepset/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01


BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = build_normalization(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
        

class BertSelfAttention(nn.Module):
    """SpikeLinear self-attention with softmax, signed SSA, or ShiftMax."""
    MODES = ("softmax", "ssa", "shiftmax_ln")

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        self.query = SpikeLinear(config.hidden_size, self.all_head_size, config=config)
        self.key = SpikeLinear(config.hidden_size, self.all_head_size, config=config)
        self.value = SpikeLinear(config.hidden_size, self.all_head_size, config=config)
        self.T = config.T
        self.attention_mode = getattr(config, "attention_mode", "softmax")
        self.attention_max_value = getattr(config, "attention_max_value", 16)
        if self.attention_mode not in self.MODES:
            raise ValueError(f"unknown attention_mode {self.attention_mode!r}")
        if self.attention_max_value not in (8, 16):
            raise ValueError("attention_max_value must be 8 or 16")
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        self.is_decoder = config.is_decoder
        self.act_quantizer_k = ElasticBiSpiking
        self.act_quantizer_v = ElasticBiSpiking
        self.clip_key = nn.ParameterList([AlphaInit(torch.tensor(1.0), requires_grad=False) for _ in range(config.T)])
        self.clip_value = nn.ParameterList([AlphaInit(torch.tensor(1.0), requires_grad=False) for _ in range(config.T)])
        self.input_bits = 2
        if self.attention_mode == "ssa":
            self.q_attention_quant = SignedSSAQuantizer()
            self.k_attention_quant = SignedSSAQuantizer()
            self.v_attention_quant = SignedSSAQuantizer()
            self.context_quant = SignedSSAQuantizer()
        elif self.attention_mode == "shiftmax_ln":
            self.q_attention_norm = build_normalization(config, self.all_head_size)
            self.k_attention_norm = build_normalization(config, self.all_head_size)
            self.v_attention_norm = build_normalization(config, self.all_head_size)
            self.q_attention_quant = MultiSpikeNorm(self.attention_max_value)
            self.k_attention_quant = MultiSpikeNorm(self.attention_max_value)
            self.context_quant = MultiSpikeNorm(self.attention_max_value)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(shape).permute(0, 1, 3, 2, 4)

    def _legacy_kv_spikes(self, key, value, time_step):
        if time_step is None:
            key = torch.cat([self.act_quantizer_k.apply(key[i:i + 1], self.clip_key[i], self.input_bits, True)
                             for i in range(self.T)], dim=0)
            value = torch.cat([self.act_quantizer_v.apply(value[i:i + 1], self.clip_value[i], self.input_bits, True)
                               for i in range(self.T)], dim=0)
        else:
            key = self.act_quantizer_k.apply(key, self.clip_key[time_step], self.input_bits, True)
            value = self.act_quantizer_v.apply(value, self.clip_value[time_step], self.input_bits, True)
        return key, value

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False, time_step=None):
        if encoder_hidden_states is not None or past_key_value is not None:
            raise NotImplementedError("cross-attention and cached decoding are not supported by SpikeBERT")
        query = self.query(hidden_states, time_step=time_step)
        key = self.key(hidden_states, time_step=time_step)
        value = self.value(hidden_states, time_step=time_step)
        if self.attention_mode == "softmax":
            key, value = self._legacy_kv_spikes(key, value, time_step)
        elif self.attention_mode == "ssa":
            query = self.q_attention_quant(query)
            key = self.k_attention_quant(key)
            value = self.v_attention_quant(value)
        else:
            query = self.q_attention_quant(self.q_attention_norm(query))
            key = self.k_attention_quant(self.k_attention_norm(key))
            value = self.v_attention_norm(value)

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        scores = query @ key.transpose(-1, -2)
        additive_mask = attention_mask.unsqueeze(0) if attention_mask is not None else None

        if self.attention_mode == "softmax":
            weights = scores / math.sqrt(self.attention_head_size)
            if additive_mask is not None:
                weights = weights + additive_mask
            weights = self.dropout(nn.functional.softmax(weights, dim=-1))
        elif self.attention_mode == "ssa":
            weights = scores * (2.0 / math.sqrt(self.attention_head_size))
            valid = valid_key_mask(additive_mask, weights)
            if valid is not None:
                weights = weights.masked_fill(~valid, 0.0)
        else:
            scores = (query * self.attention_max_value) @ (key * self.attention_max_value).transpose(-1, -2)
            valid = valid_key_mask(additive_mask, scores)
            if additive_mask is not None:
                scores = scores + additive_mask
            weights = shiftmax_exp2(scores)
            if valid is not None:
                weights = weights.masked_fill(~valid, 0.0)
            weights = self.dropout(weights)

        if head_mask is not None:
            weights = weights * head_mask
        context = weights @ value
        if self.attention_mode in ("ssa", "shiftmax_ln"):
            context = self.context_quant(context)
        context = context.permute(0, 1, 3, 2, 4).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return (context, weights) if output_attentions else (context,)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = SpikeLinear(config.hidden_size, config.hidden_size, symmetric=True, config=config)

        self.LayerNorm = build_normalization(config, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, time_step: Optional[int] = None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, time_step=time_step)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        time_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            time_step=time_step,
        )
        attention_output = self.output(self_outputs[0], hidden_states, time_step=time_step)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = SpikeLinear(config.hidden_size, config.intermediate_size, config=config)

        self.intermediate_act_fn = gelu
        

    def forward(self, hidden_states: torch.Tensor, time_step: Optional[int] = None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, time_step=time_step)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = SpikeLinear(config.intermediate_size, config.hidden_size, config=config)

        self.LayerNorm = build_normalization(config, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, time_step: Optional[int] = None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, time_step=time_step)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        time_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            time_step=time_step,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
                time_step=time_step,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            lambda chunk: self.feed_forward_chunk(chunk, time_step=time_step),
            self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        
        # print(layer_output.shape)

        return outputs

    def feed_forward_chunk(self, attention_output, time_step=None):
        intermediate_output = self.intermediate(attention_output, time_step=time_step)
        layer_output = self.output(intermediate_output, attention_output, time_step=time_step)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.T = config.T
        self.temporal_mode = getattr(config, "temporal_mode", "network")
        if self.temporal_mode not in ("network", "layerwise"):
            raise ValueError(f"unknown temporal_mode {self.temporal_mode!r}, expected 'network' or 'layerwise'")
        self.temporal_readout = getattr(config, "temporal_readout", "mean")
        if self.temporal_readout not in ("mean", "last"):
            raise ValueError(
                f"unknown temporal_readout {self.temporal_readout!r}, expected 'mean' or 'last'"
            )
        self.use_history = getattr(config, "use_history", False)
        self.history_type = getattr(config, "history_type", "direct")
        if self.history_type not in ("direct", "spike_linear"):
            raise ValueError(
                f"unknown history_type {self.history_type!r}, expected 'direct' or 'spike_linear'"
            )
        if self.history_type == "spike_linear" and (
            not self.use_history or self.temporal_mode != "network"
        ):
            raise ValueError(
                "history_type='spike_linear' requires use_history=True and temporal_mode='network'"
            )

        if self.use_history and self.history_type == "direct":
            self.history_alpha = nn.Parameter(
                torch.tensor(float(getattr(config, "history_alpha_init", 0.0)))
            )
        elif self.history_type == "spike_linear":
            self.feedback_layer_norm = build_normalization(config)
            self.feedback_spike = StatelessTernarySpike()
            self.feedback_linear = IdentityInitializedLinear(config.hidden_size)

    def _temporal_readout(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reduce a leading temporal dimension according to the configured readout."""
        if self.temporal_readout == "last":
            return hidden_states[-1]
        return hidden_states.mean(0)

    def _apply_temporal_history(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add learnable feedback from the preceding shared-weight time step."""
        if not self.use_history or hidden_states.shape[0] <= 1:
            return hidden_states
        outputs = [hidden_states[0]]
        for time_step in range(1, hidden_states.shape[0]):
            outputs.append(hidden_states[time_step] + self.history_alpha * outputs[-1])
        return torch.stack(outputs, dim=0)

    def _forward_layerwise(
        self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,
        past_key_values, use_cache, output_attentions, output_hidden_states,
    ):
        """Legacy path: all T steps go through one layer before the next, history is applied per layer."""
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        hidden_states = hidden_states.repeat(tuple([self.T] + torch.ones(len(hidden_states.size()), dtype=int).tolist())) # T B L D

        for i, layer_module in enumerate(self.layer):
            hidden_states = self._apply_temporal_history(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        final_hidden = self._temporal_readout(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (final_hidden,)

        return final_hidden, next_decoder_cache, all_hidden_states, all_self_attentions

    def _forward_network(
        self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,
        past_key_values, output_attentions, output_hidden_states, output_temporal_states=False,
    ):
        """Each time step runs the full layer stack; the previous step's final output feeds the next step's input."""
        num_layers = len(self.layer)
        layer_inputs = [[] for _ in range(num_layers)] if output_hidden_states else None
        layer_attentions = [[] for _ in range(num_layers)] if output_attentions else None

        outputs = []
        previous = None
        try:
            for t in range(self.T):
                current = hidden_states
                if self.use_history and previous is not None:
                    if self.history_type == "direct":
                        current = current + self.history_alpha * previous
                    else:
                        feedback_spikes = self.feedback_spike(self.feedback_layer_norm(previous))
                        current = current + self.feedback_linear(feedback_spikes)
                current = current.unsqueeze(0)
                for i, layer_module in enumerate(self.layer):
                    if output_hidden_states:
                        layer_inputs[i].append(current[0])
                    layer_head_mask = head_mask[i] if head_mask is not None else None
                    past_key_value = past_key_values[i] if past_key_values is not None else None
                    layer_outputs = layer_module(
                        current,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                        time_step=t,
                    )
                    current = layer_outputs[0]
                    if output_attentions:
                        layer_attentions[i].append(layer_outputs[1][0])
                previous = current.squeeze(0)
                outputs.append(previous)
        finally:
            for module in self.modules():
                if isinstance(module, SpikeLinear):
                    module.reset_state()

        temporal_outputs = torch.stack(outputs, dim=0)
        final_hidden = self._temporal_readout(temporal_outputs)

        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = tuple(torch.stack(steps, dim=0) for steps in layer_inputs) + (final_hidden,)
        all_self_attentions = None
        if output_attentions:
            all_self_attentions = tuple(torch.stack(steps, dim=0) for steps in layer_attentions)

        returned_temporal = temporal_outputs if output_temporal_states else None
        return final_hidden, None, all_hidden_states, all_self_attentions, returned_temporal

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        output_temporal_states: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        layer_args = (head_mask, encoder_hidden_states, encoder_attention_mask, past_key_values)
        if output_temporal_states and self.temporal_mode != "network":
            raise ValueError("output_temporal_states is only supported with temporal_mode='network'")
        temporal_hidden_states = None
        if self.temporal_mode == "network":
            if use_cache:
                raise NotImplementedError("use_cache is not supported with temporal_mode='network'")
            hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, temporal_hidden_states = self._forward_network(
                hidden_states, attention_mask, *layer_args, output_attentions, output_hidden_states,
                output_temporal_states,
            )
        else:
            hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions = self._forward_layerwise(
                hidden_states, attention_mask, *layer_args, use_cache, output_attentions, output_hidden_states
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    None,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentionsAndTemporal(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
            temporal_hidden_states=temporal_hidden_states,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = build_normalization(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if getattr(module, "_identity_initialized", False):
            nn.init.eye_(module.weight)
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (DynamicTanh, ShiftedDynamicTanh)):
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias"):
                module.bias.data.zero_()
            if hasattr(module, "shift"):
                module.shift.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


@dataclass
class BaseModelOutputWithPastAndCrossAttentionsAndTemporal(BaseModelOutputWithPastAndCrossAttentions):
    """BERT encoder output with optional full-loop states [T, B, L, D]."""

    temporal_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentionsAndTemporal(BaseModelOutputWithPoolingAndCrossAttentions):
    """BERT model output with optional full-loop sequence and pooled states."""

    temporal_hidden_states: Optional[torch.FloatTensor] = None
    temporal_pooler_output: Optional[torch.FloatTensor] = None


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_temporal_states: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_temporal_states=output_temporal_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        temporal_hidden_states = (
            encoder_outputs.temporal_hidden_states if return_dict and output_temporal_states else None
        )
        temporal_pooler_output = None
        if temporal_hidden_states is not None and self.pooler is not None:
            temporal_pooler_output = torch.stack(
                [self.pooler(step_hidden) for step_hidden in temporal_hidden_states], dim=0
            )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentionsAndTemporal(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            temporal_hidden_states=temporal_hidden_states,
            temporal_pooler_output=temporal_pooler_output,
        )


@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class SpikeBertForPreTraining(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



@dataclass
class SequenceClassifierOutputWithTemporalLogits(SequenceClassifierOutput):
    temporal_logits: Optional[torch.FloatTensor] = None


class SpikeBertForSequenceClassification(BertPreTrainedModel):
    """SpikeBERT encoder with a sequence classification/regression head."""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                output_temporal_logits=False):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if output_temporal_logits and self.training:
            raise ValueError("output_temporal_logits is inference-only")
        if output_temporal_logits and not return_dict:
            raise ValueError("output_temporal_logits requires return_dict=True")
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, output_temporal_states=output_temporal_logits)
        logits = self.classifier(self.dropout(outputs[1]))
        temporal_logits = None
        if output_temporal_logits:
            temporal_logits = torch.stack([self.classifier(self.dropout(x))
                                           for x in outputs.temporal_pooler_output], dim=0)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss = MSELoss()(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = BCEWithLogitsLoss()(logits, labels)
        if not return_dict:
            tail = (logits,) + outputs[2:]
            return ((loss,) + tail) if loss is not None else tail
        return SequenceClassifierOutputWithTemporalLogits(loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            temporal_logits=temporal_logits)
