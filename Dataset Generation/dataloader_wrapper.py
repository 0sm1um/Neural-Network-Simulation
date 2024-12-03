from torch.utils.data import DataLoader

class DataloaderWrapper:
    def __init__(self, dataset, batch_size, num_workers, shuffle):
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)