# MAR

## 环境
cuda 12.8.1
transformers==5.3.0
torch==2.8.0

## 执行训练
1. 设置 train.sh 中的 BASE_MODEL, SMALL_MODEL, DATASET (训练集), LR 以及 torchrun 配置.
2. 执行 bash train.sh

## 测试
1. 设置 benchmark_mar_qwen3.py 中的测试集路径 (attn_medusa 仓库中的 data 目录)
2. 按照 benchmark_mar_qwen3.py 中的示例执行命令


