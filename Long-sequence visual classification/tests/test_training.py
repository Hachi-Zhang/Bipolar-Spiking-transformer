import yaml,torch
from seq_cifar.config import load_config
from seq_cifar.training import run_training

def tiny(path,out,epochs=1,resume=""):
 path.write_text(yaml.safe_dump(dict(model="TIMv2_e3_seq",embed_dim=16,num_heads=1,depths=3,step=1,sequence_length=1024,epochs=epochs,cooldown_epochs=0,warmup_epochs=0,batch_size=2,val_batch_size=2,workers=0,synthetic_data=True,max_train_batches=1,max_val_batches=1,output=str(out),device="cpu",mixup=.5,resume=resume)))
 return load_config(path)
def test_train_and_resume(tmp_path):
 out=tmp_path/"out";run_training(tiny(tmp_path/"a.yml",out));assert (out/"resolved_config.yml").is_file();assert (out/"metrics.csv").is_file();assert (out/"checkpoint-last.pt").is_file();assert (out/"checkpoint-best.pt").is_file()
 ck=torch.load(out/"checkpoint-last.pt",weights_only=False);assert ck["epoch"]==0;assert ck["next_epoch"]==1;assert ck["global_step"]==1;assert ck["format_version"]==2
 run_training(tiny(tmp_path/"b.yml",out,2,str(out/"checkpoint-last.pt")));resumed=torch.load(out/"checkpoint-last.pt",weights_only=False);assert resumed["epoch"]==1;assert resumed["next_epoch"]==2;assert resumed["global_step"]==2;assert resumed["format_version"]==2
