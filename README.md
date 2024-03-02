# bit-llm

This repository implements training of binarized large language models. All matmuls use weights with values {-1, 0, 1}.

- **Faster inference**. Matmuls can be implemented only using addition and subtraction, which are faster than multiplication and division.
- **Small model files.** The weights can be stored in `num of params/5` bytes, meaning a 2B parameter model can be stored in 400MB or a 7B in 1.4GB, making state-of-the-art models available to consumers.
