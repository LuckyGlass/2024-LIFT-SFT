from torch.utils.data import Sampler, DataLoader
from transformers import Trainer
from typing import List


class LIFTSFTSampler(Sampler):
    def __init__(self, batch_size: int, num_datapoints: List[int]):
        super().__init__()
        self.batch_ids = []
        cur_datapoint_id = 0
        for total_datapoints in num_datapoints:
            st_datapoint_id = cur_datapoint_id
            ed_datapoint_id = cur_datapoint_id + total_datapoints
            self.batch_ids += [list(range(st, min(st + batch_size, ed_datapoint_id))) for st in range(st_datapoint_id, ed_datapoint_id, batch_size)]
            cur_datapoint_id += total_datapoints
    
    def __iter__(self):
        for batch_ids in self.batch_ids:
            yield batch_ids
    
    def __len__(self):
        return len(self.batch_ids)


class LIFTSFTTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        
        from transformers.trainer_utils import seed_worker

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": seed_worker,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
            "batch_sampler": LIFTSFTSampler(self._train_batch_size, train_dataset.num_datapoints),
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
