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


os.environ["CURL_CA_BUNDLE"] = ""


@hydra.main(config_path="../config", config_name="train", version_base="1.2")
def main(cfg):
    torch.set_float32_matmul_precision("high")

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    base_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    model = hydra.utils.instantiate(cfg.model, should_init_weights=False)
    model.load_state_dict(base_model.state_dict(), strict=True)

    model.half().eval().cuda()

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello, how are you?"}], return_tensors="pt"
    ).cuda()

    output_ids = model.generate(prompt_ids, max_len=50)

    print(output_ids)

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(output)

    # trainer = pl.Trainer(
    #     accelerator="auto",
    #     max_steps=10_000,
    #     precision="bf16-true",
    #     gradient_clip_val=1.0,
    #     logger=pl.loggers.WandbLogger(project="bit-llm"),
    #     log_every_n_steps=50,
    #     default_root_dir="~/logs",
    #     callbacks=[
    #         GenerationCallback(tokenizer=tokenizer),
    #     ],
    # )

    # data_module = DataModule(tokenizer=tokenizer, batch_size=cfg.batch_size)

    # trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
