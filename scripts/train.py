from functools import partial
import os

import hydra
import torch
from lightning import pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizerFast

os.environ["CURL_CA_BUNDLE"] = ""


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


def prepare_dataset(tokenizer: LlamaTokenizerFast):
    with open("data/train.txt", "r") as f:
        text = f.read()

    tokens = tokenizer(text, return_tensors="np", add_special_tokens=False)
    return TextDataset(**chunk_tokens(tokens, chunk_size=32))


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_name: str, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name

    def setup(self, stage=None):
        if stage == "fit":
            self.tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)
            self.train_dataset = prepare_dataset(self.tokenizer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None


@hydra.main(config_path="../config", config_name="train", version_base="1.2")
def main(cfg):
    model_name = "HuggingFaceTB/cosmo-1b"
    base_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    model = hydra.utils.instantiate(cfg.model, should_init_weights=False)
    model.load_state_dict(base_model.state_dict(), strict=True)

    out = model(torch.randn((3, 128, 2048)))
    print(out.shape)
    print(cfg)

    trainer = pl.Trainer(
        accelerator="auto",
        max_steps=10_000,
    )

    data_module = DataModule(tokenizer_name=model_name, batch_size=cfg.batch_size)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
