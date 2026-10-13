"""Helpers for adding SpikeBERT fields to a Hugging Face BertConfig."""

def configure_spikebert(config, *, time_steps=4, temporal_mode="network", temporal_readout="mean", use_history=False, history_type="direct", history_alpha_init=0.0, attention_mode="softmax", attention_max_value=16, normalization_mode="layernorm", dyt_alpha_init=0.5):
    if attention_mode not in ("softmax", "ssa", "shiftmax_ln"):
        raise ValueError(f"unknown attention_mode {attention_mode!r}")
    if normalization_mode not in ("layernorm", "dyt", "shifted_dyt"):
        raise ValueError(f"unknown normalization_mode {normalization_mode!r}")
    if attention_max_value not in (8, 16):
        raise ValueError("attention_max_value must be 8 or 16")
    config.weight_bits, config.input_bits = 32, 2
    config.quantize_act, config.clip_val, config.T = True, 1.0, time_steps
    config.temporal_mode, config.temporal_readout = temporal_mode, temporal_readout
    config.use_history, config.history_type = use_history, history_type
    config.history_alpha_init = history_alpha_init
    config.attention_mode, config.attention_max_value = attention_mode, attention_max_value
    config.normalization_mode, config.dyt_alpha_init = normalization_mode, float(dyt_alpha_init)
    return config
