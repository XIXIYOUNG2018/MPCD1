# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.optim.optimizer import Optimizer
from model import Graphormer
from data import GraphDataModule, get_dataset

from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,BaseFinetuning
# from pytorch_lightning.callbacks.finetuning import BaseFinetuning
import os
from pytorch_lightning.plugins import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class FinetuningCallback(BaseFinetuning):
    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        self.freeze(pl_module.clf_head)
        # pl_module.reg_head.fc_layers1[10]
        only_logd_head=[0,2,3,4,5,6,7,8,9,10,11,12] #freeze other reg head, when only logd task is trained
        for i in only_logd_head:
            self.freeze(pl_module.reg_head.fc_layers1[i])
            self.freeze(pl_module.reg_head.fc_layers2[i])
            self.freeze(pl_module.reg_head.fc_layers3[i])
            self.freeze(pl_module.reg_head.output_layer1[i])
        self.freeze(pl_module.mask_out_proj)
        self.freeze(pl_module.gap_out_proj)
    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        # return super().finetune_function(pl_module, epoch, optimizer, opt_idx)()
        pass


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Graphormer.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)
    # ------------
    # model
    # ------------
    if args.checkpoint_path != '':
        model = Graphormer.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    else:
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    if not args.test and not args.validate:
        print(model)
    


    
    print('total params:', sum(p.numel() for p in model.parameters()))

    # # ------------
    # training
    # ------------
    metric = 'valid_' + get_dataset(dm.dataset_name)['metric']
    dirpath = args.default_root_dir + f'/lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=30,
        mode=get_dataset(dm.dataset_name)['metric_mode'],
        save_last=True,
    )
    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)
    trainer = pl.Trainer(accelerator='gpu',devices=1,gpus=[0],precision=16,strategy=DDPStrategy(find_unused_parameters=True),accumulate_grad_batches=4,max_epochs=2000)

    finetunecallback=FinetuningCallback()
    trainer.callbacks.append(finetunecallback)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.test:
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.validate:
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:
        trainer.fit(model, datamodule=dm)
                   #ckpt_path='/home/ps/Documents/xxy/pred/Admethormer_finetune/exps/admet/reg/1/lightning_logs/checkpoints/last.ckpt')


if __name__ == '__main__':
    cli_main()





