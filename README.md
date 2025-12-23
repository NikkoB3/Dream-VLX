## Dream-VL & Dream-VLA

[![Blog Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://hkunlp.github.io/blog/2025/dream-vlx)
[![Model: Dream-VL-7B](https://img.shields.io/badge/HuggingFace-Dream--VL--7B-yellow.svg)](https://huggingface.co/Dream-org/Dream-VL-7B)
[![Model: Dream-VLA-7B](https://img.shields.io/badge/HuggingFace-Dream--VLA--7B-yellow.svg)](https://huggingface.co/Dream-org/Dream-VLA-7B)

![](assets/VLA_mid.gif)

Building on the success of Dream 7B, we introduce Dream-VL and Dream-VLA, open VL and VLA models that fully unlock discrete diffusion’s advantages in **long-horizon planning**, **bidirectional reasoning**, and **parallel action generation** for multimodal tasks.

![](assets/overview.png)

Key Results:
- **Dream-VL**: Achieves state-of-the-art performance among diffusion VLMs, comparable to top-tier AR VLMs trained on open data, with superior performance on visual planning tasks requiring long-horizon reasoning.
- **Dream-VLA**: Establishes top-tier performance with 97.2% average on LIBERO, 71.4% on SimplerEnv–Bridge, and 60.5% on SimplerEnv–Fractal, surpassing leading models including GR00T-N1 and OpenVLA-OFT. Consistently outperforms AR baselines across diverse finetuning objectives.

## 🚀 News

- **2025-12-23**: [Dream-VL](https://huggingface.co/Dream-org/Dream-VL-7B) & [Dream-VLA](https://huggingface.co/Dream-org/Dream-VLA-7B) models and [blog](https://hkunlp.github.io/blog/2025/dream-vlx) released.


## 🧱 Repository Structure

> The exact structure may evolve; please refer to the repo for up-to-date details.

```text
Dream-VLX/
├── Dream-VL/          # Dream-VL training and evaluation (preparing)
├── Dream-VLA/         # Dream-VLA training and evaluation (preparing)
└── README.md          # This file
```

## Citation
```
@article{ye2025dreamvla,
  title={Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone},
  author={Ye, Jiacheng and Gong, Shansan and Gao, Jiahui and Fan, Junming and Wu, Shuang and Bi, Wei and Bai, Haoli and Shang, Lifeng and Kong, Lingpeng},
  journal={arXiv preprint},
  year={2025}
}
```