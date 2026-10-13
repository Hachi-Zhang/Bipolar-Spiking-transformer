_base_ = ["./sdtv3_5m_ade20k.py"]
# SSA uses round(clamp(x, -4, 4)) / 4 for Q, K, V and attention output.
model = dict(backbone=dict(attention_mode="ssa"))
