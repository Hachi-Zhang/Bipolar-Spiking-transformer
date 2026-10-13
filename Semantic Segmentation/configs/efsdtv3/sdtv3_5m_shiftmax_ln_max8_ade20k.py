_base_ = ["./sdtv3_5m_ade20k.py"]
model = dict(backbone=dict(attention_mode="shiftmax_ln", attention_max_value=8))
