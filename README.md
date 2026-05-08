
# BERT 中文情感分析实验

本项目使用 `bert-base-chinese` 完成中文文本情感二分类实验，包含数据读取、文本编码、BERT 微调、模型评估和结果保存。

## 文件说明

text
.
├── README.md
├── requirements.txt
└── BERT
    ├── run_bert_experiment.py
    ├── train.tsv
    ├── test.tsv
    └── Model_Parameter_save
        └── experiment_results.json
```

- `requirements.txt`：实验依赖
- `BERT/run_bert_experiment.py`：实验主程序
- `BERT/train.tsv`：训练数据
- `BERT/test.tsv`：测试数据
- `BERT/Model_Parameter_save/experiment_results.json`：实验结果

## 环境配置
bash
pip install -r requirements.txt

## 运行方式

bash
python BERT/run_bert_experiment.py

## 实验结果

本次实验验证集结果：

text
valid_loss = 0.4160
valid_acc = 0.8200
confusion_matrix = [[49, 6], [12, 33]]

## 说明

预训练模型 `bert-base-chinese` 和训练后的权重文件较大，未上传到仓库。运行时可通过 Hugging Face 下载模型，或在本地准备模型文件。
