import pytest
import torch
import yaml
from seq_cifar.config import load_config, total_epochs
from seq_cifar.training import WarmupCosineCooldown


def config_file(tmp_path, values):
    path=tmp_path/"c.yml"; path.write_text(yaml.safe_dump(values)); return path


def test_merge_and_total_epochs(tmp_path):
    c=load_config(config_file(tmp_path,{"epochs":400,"cooldown_epochs":10}),{"batch_size":2})
    assert c["batch_size"]==2 and total_epochs(c)==410


def test_scheduler_boundaries_and_restore(tmp_path):
    c=load_config(config_file(tmp_path,{"epochs":400,"warmup_epochs":20,"cooldown_epochs":10}))
    p=torch.nn.Parameter(torch.tensor(1.)); opt=torch.optim.AdamW([p],lr=c["lr"]); sched=WarmupCosineCooldown(opt,c)
    assert sched.step(0)==pytest.approx(c["warmup_lr"])
    assert sched.step(19)==pytest.approx(c["lr"])
    assert sched.step(399)==pytest.approx(c["min_lr"])
    for epoch in range(400,410): assert sched.step(epoch)==pytest.approx(c["min_lr"])
    sched.step(19); state=sched.state_dict(); opt2=torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))],lr=c["lr"]); restored=WarmupCosineCooldown(opt2,c); restored.load_state_dict(state)
    assert restored.epoch==19
    assert restored.step(20)==pytest.approx(sched.step(20))


def test_unknown_and_invalid_values(tmp_path):
    with pytest.raises(ValueError,match="unknown YAML"): load_config(config_file(tmp_path,{"typo_key":1}))
    with pytest.raises(ValueError,match="unknown model"): load_config(config_file(tmp_path,{"model":"bad"}))
    with pytest.raises(ValueError,match="sequence_length"): load_config(config_file(tmp_path,{"sequence_length":128}))
    with pytest.raises(ValueError,match="divisible"): load_config(config_file(tmp_path,{"embed_dim":48,"num_heads":8}))
