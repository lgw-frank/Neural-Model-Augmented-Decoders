# Neural-Model-Augmented Hybrid NMS-OSD Decoders for Near-ML Performance in Short Block Codes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2509.25580)

A practical implementation achieving optimal trade-offs among **FER performance**, **high throughput**, **low latency**, and **moderate complexity** for short block code decoding.

> **📝 Implementation Notice**
>
> This project is released under the **MIT License**. However, please note that specific algorithmic implementations within the codebase are protected by Chinese Patents:
> - **ZL 2024 1 1666758.9**
> - **ZL 2025 1 0795632.X**
>
> The open-source license **does not grant** rights for commercial use of these patented technologies. Separate commercial licensing is required.
>
> 本项目基于 **MIT 许可证**开源。请注意，代码中的特定算法实现受中国专利保护：
> - **ZL 2024 1 1666758.9**
> - **ZL 2025 1 0795632.X**
>
> 开源许可证**未授予**这些专利技术的商业使用权。商业用途需要单独授权。

**For details, please see:** [`LICENSE`](LICENSE) and [`PATENTS`](PATENTS)  
**详情请参阅：** [`LICENSE`](LICENSE) 和 [`PATENTS`](PATENTS) 文件

---

## 🚀 Implementation Status

**Source code will be released within one week**, following comprehensive documentation and code cleanup.

Associated research paper:  
**"Neural-Model-Augmented Hybrid NMS-OSD Decoders for Near-ML Performance in Short Block Codes"**  
*arXiv preprint available at: [arXiv:2509.25580](https://arxiv.org/abs/2509.25580)*

---

## 📁 Project Structure
├── LDPC_codes/ # 🎯 LDPC code implementations (training & inference)
├── BCH_codes/ # 📊 BCH code implementations (training & inference)
├── RS_codes/ # 🧠 Reed-Solomon code implementations (training & inference)
├── utils/ # 🔧 Utility functions (visualization, analysis)
├── LICENSE # 📄 MIT License
├── PATENTS # ⚖️ Patent disclosure (English)
├── PATENTS.zh-CN # ⚖️ 专利声明 (中文)
└── README.md # 📖 Project documentation

---

## 🎯 Key Features

- **Near-ML Performance**: Approaches maximum likelihood decoding for short block codes
- **Hybrid Architecture**: Combines Neural Min-Sum (NMS) and Ordered Statistics Decoding (OSD)
- **Practical Efficiency**: Optimized for throughput-latency-complexity trade-offs
- **Multi-Code Support**: Implementations for LDPC, BCH, and Reed-Solomon codes

---

## 📜 License

This project is open-sourced under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Important**: Commercial use of the patented technologies requires additional licensing.