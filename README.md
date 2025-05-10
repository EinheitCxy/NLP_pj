# Reproduction and Extension of NEFTune

This project reproduces and extends the NEFTune (Noise-Embedded Fine-Tuning) algorithm, which introduces controlled noise into the embedding layer of LLMs to enhance generalization during instruction tuning.

## Background & Motivation

NEFTune improves model performance with a simple modification: injecting Gaussian noise into the embedding layer during fine-tuning.
But how much the altered embedding algorithm can change model performance is unknown for now.

Our project goals:

- Reproduce the original NEFTune experiments
- See how the noise insertion influences model performance
- Explore new hyperparameters (e.g., noise decay strategies)
- Propose and evaluate **Sparse-NEFT**, combining noise injection with dynamic weight sparsity

---

## Project Structure

### Core Reproduction

- Model: `LLaMA 3.2-1B-Instruct`
- Noise variants: `α ∈ {0, 5, 10}`
- Datasets: Alpaca, Evol-Instruct 70K, ShareGPT
- Evaluation: AlpacaEval and qualitative comparison

### Extended Experiments

- **Noise scheduling**: Linear decay, Step decay of alpha
- **Sparse-NEFT**: Integrating dynamic weight sparsity with NEFTune
- **Scaling**: Experiments extended to `LLaMA 3.2-3B-Instruct` if feasible

---

## Technical Setup

- **Frameworks**: Hugging Face `transformers`, `peft`, `accelerate`
- **Noise injection**: Implemented in the embedding forward pass
- **PEFT**: LoRA for efficient fine-tuning
- **System Requirements**:
  - CPU: 4–8 cores
  - GPU: 24GB VRAM (e.g., RTX 4090 or A100)
  - Storage: 10–20 GB

---

## Timeline

| Period         | Task Summary                                                                 | Process|
|----------------|------------------------------------------------------------------------------| -----|
| **4/20–5/6**   | Environment setup, codebase implementation, reproduction of core experiments |WIP|
| **5/6–5/27**   | Hyperparameter tuning (decay strategies), Sparse-NEFT, optional 3B scaling   ||
| **5/27–6/17**  | Final evaluations, result analysis, paper/report preparation                ||

---

## Limitations

- Evaluation via AlpacaEval may reflect GPT-4 bias
- Limited resources prevent sweeping all hyperparameters or testing on 70B models
- The mechanism behind NEFTune's effectiveness is not yet fully understood

---

## To be added

- [ ] `neftune_trainer.py`: Main training script
- [ ] `neftune_wrapper.py`: Noise injection module
- [ ] `configs/`: Training configurations
- [ ] `notebooks/`: For interactive analysis
- [ ] `results/`: Visualized experiment outputs

---

## Citation

Our research is based on following paper:
> https://arxiv.org/abs/2305.14980

---

## Contributions

Contributions and questions are welcome via issues or pull requests!
