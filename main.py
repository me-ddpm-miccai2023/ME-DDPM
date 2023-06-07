import argparse
import os
from pytorch_lightning import loggers as pl_loggers
from lib.datasets.biobank.pytorch import TrainDataset, ValidationDataset
from models.recon import Model
from pytorch_lightning import Trainer, seed_everything
from lib.defaults import image_size
seed_everything(1117, workers=True)

image_size_ = image_size if image_size != 256 else None
acc_rate = 16.0
no_central_lines = 8
model_ = lambda : Model(acc_rate=acc_rate, no_central_lines=no_central_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1']),
                        help='Whether to run in test mode')
    parser.add_argument(
        '--checkpoint_file', default=None, type=str,
        help="""Whether to use a checkpoint file to restore training or full
        running an evaluation/test""")
    parser.add_argument('--gpu', type=str, default='0',
                        help='Which GPUs to use')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    if opt.test:
        raise NotImplementedError
    if "," in opt.gpu:
        raise NotImplementedError
    
    if opt.checkpoint_file is not None and opt.checkpoint_file != '':
        assert(os.path.isfile(opt.checkpoint_file))
        print(f"Training from checkpoint file {opt.checkpoint_file}")
        model = Model.load_from_checkpoint(opt.checkpoint_file)
        model.acc_rate = acc_rate
    else:
        model = model_()


    tb_logger = pl_loggers.TensorBoardLogger("logs//")
    trainer = Trainer(logger=tb_logger, gpus=1,
                      default_root_dir='./checkpoints/')
    train = TrainDataset(batch_size=16, epochs=20000,
                         central_crop_size=image_size_, num_workers=16)
    val = ValidationDataset(
        batch_size=4, central_crop_size=image_size_, num_workers=16)
    trainer.fit(model, train, val)
