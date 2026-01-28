---
license: odc-by
task_categories:
- text2text-generation
language:
- en
size_categories:
- 10K<n<100K
---

This repository contains the data presented in [SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines](https://huggingface.co/papers/2502.14739).


## Tutorials for submitting to the official leadboard
coming soon


## 📜 License

**SuperGPQA** is a composite dataset that includes both original content and portions of data derived from other sources. The dataset is made available under the **[Open Data Commons Attribution License (ODC-BY)](https://opendatacommons.org/licenses/by/)**, which asserts no copyright over the underlying content. 

This means that while the dataset itself is freely available for use, you are required to:  
- Give appropriate credit to the original creators of any third-party data included in SuperGPQA.  
- Comply with the respective licenses of the referenced datasets.  

For more details, please refer to the [ODC-BY License](https://opendatacommons.org/licenses/by/) and the licenses of the referenced datasets.

### 🔗 Referenced Datasets & Links

SuperGPQA includes a limited portion of data sourced from the following datasets:

- **[LawBench](https://lawbench.opencompass.org.cn/home)**
- **[MedMCQA](https://medmcqa.github.io/)**
- **[MedQA](https://github.com/jind11/MedQA)**
- **[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)**
- **[MMLU-CF](https://huggingface.co/datasets/microsoft/MMLU-CF)**
- **[ShoppingMMLU](https://github.com/KL4805/ShoppingMMLU)**
- **[UTMath](https://utmathhomepage.github.io/)**
- **[MusicTheoryBench](https://ezmonyi.github.io/ChatMusician/)**
- **[Omni-Math](https://omni-math.github.io/)**
- **[U-MATH](https://toloka.ai/math-benchmark)**
- **[Putnam-AXIOM](https://huggingface.co/Putnam-AXIOM)**
- **[Short-form Factuality](https://github.com/openai/simple-evals)**
- **[Chinese SimpleQA](https://openstellarteam.github.io/ChineseSimpleQA/)**
- **[AIME-AOPS](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions?srsltid=AfmBOooGfWIPceky_dYe-owTh_eTCSEqh2NCUi3FhVREjxJ5AeSjjhbo)**
- **[AIMO Validation AIME](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)**

## ⚠️ Usage Notice

1. **SuperGPQA is primarily composed of newly created data**, but it also incorporates a small fraction of transformed content from other datasets.
2. If using SuperGPQA, please **comply with the original dataset licenses** when dealing with any referenced content.
3. We encourage proper attribution—if you use SuperGPQA in your research, please **cite and acknowledge the original dataset contributors**.

## 📚 Citation

**BibTeX:**
```bibtex
@misc{pteam2025supergpqascalingllmevaluation,
      title={SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines}, 
      author={M-A-P Team and Xinrun Du and Yifan Yao and Kaijing Ma and Bingli Wang and Tianyu Zheng and Kang Zhu and Minghao Liu and Yiming Liang and Xiaolong Jin and Zhenlin Wei and Chujie Zheng and Kaixing Deng and Shuyue Guo and Shian Jia and Sichao Jiang and Yiyan Liao and Rui Li and Qinrui Li and Sirun Li and Yizhi Li and Yunwen Li and Dehua Ma and Yuansheng Ni and Haoran Que and Qiyao Wang and Zhoufutu Wen and Siwei Wu and Tianshun Xing and Ming Xu and Zhenzhu Yang and Zekun Moore Wang and Junting Zhou and Yuelin Bai and Xingyuan Bu and Chenglin Cai and Liang Chen and Yifan Chen and Chengtuo Cheng and Tianhao Cheng and Keyi Ding and Siming Huang and Yun Huang and Yaoru Li and Yizhe Li and Zhaoqun Li and Tianhao Liang and Chengdong Lin and Hongquan Lin and Yinghao Ma and Zhongyuan Peng and Zifan Peng and Qige Qi and Shi Qiu and Xingwei Qu and Yizhou Tan and Zili Wang and Chenqing Wang and Hao Wang and Yiya Wang and Yubo Wang and Jiajun Xu and Kexin Yang and Ruibin Yuan and Yuanhao Yue and Tianyang Zhan and Chun Zhang and Jingyang Zhang and Xiyue Zhang and Xingjian Zhang and Yue Zhang and Yongchi Zhao and Xiangyu Zheng and Chenghua Zhong and Yang Gao and Zhoujun Li and Dayiheng Liu and Qian Liu and Tianyu Liu and Shiwen Ni and Junran Peng and Yujia Qin and Wenbo Su and Guoyin Wang and Shi Wang and Jian Yang and Min Yang and Meng Cao and Xiang Yue and Zhaoxiang Zhang and Wangchunshu Zhou and Jiaheng Liu and Qunshu Lin and Wenhao Huang and Ge Zhang},
      year={2025},
      eprint={2502.14739},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14739}, 
}