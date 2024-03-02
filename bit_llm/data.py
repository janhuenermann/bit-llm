import lightning.pytorch as pl
from transformers import LlamaTokenizerFast
import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids  # [chunks, chunk_size]
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def chunk_tokens(tokens, chunk_size):
    input_ids = tokens["input_ids"][0]
    attention_mask = tokens["attention_mask"][0]
    max_len = (input_ids.shape[0] // chunk_size) * chunk_size
    chunked_input_ids = input_ids[:max_len].reshape(-1, chunk_size)
    chunked_attention_mask = attention_mask[:max_len].reshape(-1, chunk_size)
    return {
        "input_ids": chunked_input_ids,
        "attention_mask": chunked_attention_mask,
    }


def prepare_dataset(tokenizer: LlamaTokenizerFast, chunk_size: int):
    with open("data/train.txt", "r") as f:
        text = f.read()

    tokens = tokenizer(text, return_tensors="np", add_special_tokens=False)
    return TextDataset(**chunk_tokens(tokens, chunk_size=chunk_size))


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer: LlamaTokenizerFast, batch_size: int = 32, chunk_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = prepare_dataset(self.tokenizer, self.chunk_size)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None
