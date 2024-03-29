from typing import Sequence, Dict
import copy
import logging
from dataclasses import dataclass
import requests

from transformers import LlamaTokenizerFast
from torch.utils.data import Dataset
import torch
from lightning.pytorch import LightningDataModule

URL = "https://github.com/tatsu-lab/stanford_alpaca/raw/761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json"
IGNORE_INDEX = -1
PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n{input}\n\n"
    ),
    "prompt_no_input": (
        "{instruction}\n\n"
    ),
}


# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: LlamaTokenizerFast) -> Dict:
    outputs = tokenizer.batch_encode_plus(
        [s + t for s, t in zip(sources, targets)],
        return_tensors="pt",
        return_offsets_mapping=True,
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    mask = outputs['offset_mapping'][:, :, 1] < torch.as_tensor([len(s) for s in sources]).unsqueeze(1)
    labels = outputs["input_ids"].masked_fill(mask, IGNORE_INDEX)
    return dict(input_ids=outputs["input_ids"], labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_url: str, tokenizer: LlamaTokenizerFast):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = requests.get(data_url).json()

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        logging.warning(f"Tokenization done. Got {len(self.labels)} examples")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class DataModule(LightningDataModule):
    def __init__(self, tokenizer: LlamaTokenizerFast, batch_size: int, chunk_size: int = 256):
        super().__init__()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = chunk_size
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = SupervisedDataset(URL, self.tokenizer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
