import json
import os
import pickle
import torch
import tqdm
import transformers
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Sequence
from lift_args import LIFTDataArguments


BAMBOO_FORMAT = "Given a long text, and {num_events} events which take place in the long text, each indicated by number identifier [] that represents the shuffled order, (e.g. [0], [2], etc.). Reorder the events according to the original order of the events in the long text. The events should be listed in descending order using identifiers, and the first event in original order should be list first, and the output format should be [] > [], e.g., [0] > [2]. Only response the reorder results, do not say any word or explain.\n\nLong text:\n{content}\nEvents: {events}\n\nGive the reorder results, only give the ordered identifiers of the {num_events} events {answer_format}: "


class LIFTSFTDataset(Dataset):
    """Dataset for long-context time-line-reorder task SFT."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, len_segment: int, len_offset: int, block_size: int, model_max_length: int, cache_path: Optional[str]=None, ignore_index: int=-1):
        super().__init__()
        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                dict_data = pickle.load(f)
            self.input_ids = dict_data['input_ids']
            self.labels = dict_data['labels']
            self.num_datapoints = dict_data['num_datapoints']
        else:
            with open(data_path, 'r') as f:
                if os.path.splitext(data_path)[1] == '.json':
                    list_data_dict = json.load(f)
                elif os.path.splitext(data_path)[1] == '.jsonl':
                    list_data_dict = list(map(json.loads, f.readlines()))

            # Formatting for Bamboo-style timeline-reorder and tokenizing
            self.input_ids = []
            self.labels = []
            self.num_datapoints = []  # This is used to identify different samples
            self.sample_ids = []
            len_segment = len_segment * block_size
            len_offset = len_offset * block_size
            for sample in tqdm.tqdm(list_data_dict, desc="Tokenize"):
                context_ids = tokenizer(sample['input'] + tokenizer.eos_token, add_special_tokens=False, return_tensors='pt').input_ids.flatten()
                self.input_ids += [context_ids[s:s+len_segment] for s in range(0, context_ids.shape[-1], len_offset)]
                self.labels += [context_ids[s:s+len_segment] for s in range(0, context_ids.shape[-1], len_offset)]
                self.sample_ids += [len(self.num_datapoints) for _ in range(0, context_ids.shape[-1], len_offset)]
                self.num_datapoints.append(len(range(0, context_ids.shape[-1], len_offset)))
                for qa in sample['qa_pairs']:
                    prompt = BAMBOO_FORMAT.format_map(dict(
                        num_events=len(qa['summaries']),
                        content=sample['input'],
                        events='\n'.join(f"[{i}]: {event}" for i, event in enumerate(qa['summaries'], start=1)),
                        answer_format=' < '.join(['[]'] * len(qa['summaries']))
                    ))
                    messages = [
                        {'role': 'system', 'content': "You are a helpful assistant."},
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': ' < '.join(f'[{i}]' for i in qa['answers'])}
                    ]
                    input_length = tokenizer.apply_chat_template(messages[:-1], return_tensors='pt', add_generation_prompt=True, return_dict=True)['input_ids'].shape[-1]
                    input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=False, return_dict=True)['input_ids'].flatten()
                    output_length = input_ids.shape[-1] - input_length
                    if input_ids.shape[-1] > model_max_length:
                        input_ids = torch.concat((input_ids[:model_max_length//2], input_ids[-model_max_length//2:]), dim=-1)
                        input_length = len(input_ids) - output_length
                    labels = input_ids.clone()
                    labels[:input_length] = ignore_index

                    assert input_ids.shape[-1] <= model_max_length
                    self.input_ids.append(input_ids)
                    self.labels.append(labels)
                    self.sample_ids.append(len(self.num_datapoints))
                self.num_datapoints.append(len(sample['qa_pairs']))
            if cache_path is not None:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'input_ids': self.input_ids,
                        'labels': self.labels,
                        'num_datapoints': self.num_datapoints,
                    }, f)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print('?', i, self.sample_ids[i])
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    ignore_index: int = -1

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_lift_dataset(tokenizer: transformers.PreTrainedTokenizer, data_args: LIFTDataArguments, model_max_length: int) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LIFTSFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        len_segment=data_args.len_segment,
        len_offset=data_args.len_offset,
        block_size=data_args.block_size,
        model_max_length=model_max_length,
        cache_path=data_args.input_cache_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, ignore_index=data_args.ignore_index)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
