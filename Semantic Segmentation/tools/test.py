#!/usr/bin/env python3
import argparse, os
from pathlib import Path
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

def main():
    parser=argparse.ArgumentParser(description="Evaluate EFSDT segmentation")
    parser.add_argument("config"); parser.add_argument("checkpoint"); parser.add_argument("--work-dir")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--launcher", default="none", choices=["none", "pytorch", "slurm", "mpi"])
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args=parser.parse_args(); os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    cfg=Config.fromfile(args.config); cfg.launcher=args.launcher; cfg.load_from=args.checkpoint
    if args.cfg_options: cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir=args.work_dir or cfg.get("work_dir", str(Path("work_dirs") / Path(args.config).stem))
    Runner.from_cfg(cfg).test()
if __name__ == "__main__": main()
