import os
import io

import numpy as np
import hydra
from lightning import pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from bit_llm.data import DataModule
from bit_llm.callbacks.generation import GenerationCallback
from bit_llm.model import get_quantized_state_dict


os.environ["CURL_CA_BUNDLE"] = ""


@hydra.main(config_path="../config", config_name="train", version_base="1.2")
def main(cfg):
    torch.set_float32_matmul_precision("high")

    model_name = "HuggingFaceTB/cosmo-1b"
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    base_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    model = hydra.utils.instantiate(cfg.model, should_init_weights=False)
    model.load_state_dict(base_model.state_dict(), strict=True)

    model.cuda()

    with io.BytesIO() as f:
        np.savez(f, **get_quantized_state_dict(model))
        model_size_mb = f.tell() / (1000 ** 2)
        print(f"Model size: {model_size_mb:.2f} MB")

    trainer = pl.Trainer(
        accelerator="auto",
        max_steps=1000,
        precision="bf16-true",
        gradient_clip_val=5.0,
        logger=pl.loggers.WandbLogger(project="bit-llm"),
        log_every_n_steps=50,
        default_root_dir="~/logs",
        enable_checkpointing=False,
        callbacks=[
            GenerationCallback(tokenizer=tokenizer),
        ]
    )

    data_module = DataModule(tokenizer=tokenizer, batch_size=cfg.batch_size)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
