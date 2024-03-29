from typing import Any
import lightning.pytorch as pl
from transformers import LlamaTokenizerFast

from bit_llm.model import Llama


class GenerationCallback(pl.Callback):
    def __init__(self, tokenizer: LlamaTokenizerFast):
        super().__init__()
        self.tokenizer = tokenizer
        self.counter = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.invoke(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Llama):
        self.invoke(trainer, pl_module)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int):
        self.counter = self.counter + 1
        if self.counter >= 500:
            self.invoke(trainer, pl_module)

    def invoke(self, trainer: pl.Trainer, pl_module: Llama):
        prompt = "What are the moons of Saturn? "
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(pl_module.device)

        generated = pl_module.generate(prompt_ids, max_len=50)
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        print("=======================")
        print(generated_text)
        print("=======================")

        self.counter = 0

        columns = ["prompt", "model"]
        data = [[prompt, generated_text]]
        pl_module.logger.log_text(key="samples", columns=columns, data=data)
