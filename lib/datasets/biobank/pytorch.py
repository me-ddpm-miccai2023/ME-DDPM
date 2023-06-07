from .dataset import Dataset as BDL
from torch.utils.data import Dataset as PyTDataset, DataLoader as PyDataLoader

class Dataset(PyTDataset):
    def __init__(
        self, type_='train',
        epochs=None,
        debug=False,
        central_crop_size=None,
        *args, **kwargs
    ):
        super().__init__()
        dataset = BDL(
            debug=debug,
            batch_size=1,
            Nslices=2,
            central_crop_size=central_crop_size,
            num_test_files=None,
            synthetic_phases=True,
            temporal=True
        )

        dataset.config.epochs = epochs
        self.dataset = dataset.py_gen(type_=type_)

        self.BDL = dataset
        

    def __len__(self):
        return self.BDL.generator.num_files

    def __getitem__(self, idx):
        # NB/ `idx` is not used
        example = self.dataset.__next__()
        return example

def DataLoader(
    type_,
    batch_size=16, 
    epochs=None,
    debug=False,
    central_crop_size=None,
    num_workers=16,
    pin_memory=True):

    return PyDataLoader(
        Dataset(
            type_=type_,
            epochs=epochs,
            debug=debug,
            central_crop_size=central_crop_size
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def TrainDataset(*args, **kwargs):
    return DataLoader('train', *args, **kwargs)

def TestDataset(*args, **kwargs):
    return DataLoader('test', *args, **kwargs)

def ValidationDataset(*args, **kwargs):
    return DataLoader('validation', *args, **kwargs)