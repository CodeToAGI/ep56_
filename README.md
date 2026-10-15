# EP56 — Model Compression Explained

**Pruning · Quantization · Knowledge Distillation · TensorRT (explained)**

Part of the [CodeToAGI Deep Learning Series](https://github.com/CodeToAGI/deep-learning-series) (72 episodes · Neurons to GPT).

## What this episode does

We take a real PyTorch classifier on scikit-learn digits (8×8 images) and measure three compression techniques:

| Technique              | What it does                          | Measured? |
|------------------------|---------------------------------------|-----------|
| Magnitude pruning      | Remove smallest weights + fine-tune   | Yes       |
| Dynamic int8 quant     | Real PyTorch `quantize_dynamic`       | Yes       |
| Fake k-bit + QAT       | Simulated 8/4/3/2-bit weights         | Yes       |
| Knowledge distillation | Small student learns from big teacher | Yes       |
| TensorRT               | NVIDIA compiler                       | Explained only |

Every number shown in the video comes from a real run (5 seeds). Nothing is typed by hand.

## Quick start

```bash
pip install torch scikit-learn numpy
python ep56_compression.py
