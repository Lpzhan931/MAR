# MAR

## 环境
cuda 12.8.1
transformers==5.3.0
torch==2.8.0

## 原版 MAR
```
mar_model.py
mar_train.py
mar_train.sh
mar_benchmark.py
```

训练: 
- 配置 mar_train.sh
- 执行 bash mar_train.sh

测试:
- 配置 mar_benchmark.py
- 执行 mar_benchmark.py (命令行参数参考文件开头注释)

## DecoderLayer + MAR
```
attn_mar_model.py
attn_mar_train.py
attn_mar_train.sh
attn_mar_benchmark.py
```

训练: 
- 配置 attn_mar_train.sh
- 执行 bash attn_mar_train.sh

测试:
- 配置 attn_mar_benchmark.py
- 执行 attn_mar_benchmark.py (命令行参数参考文件开头注释)


## 需要配置的参数
- BASE_MODEL
- SMALL_MODEL
- DATASET (训练集)
- LR
- torchrun 配置
- DATA_DIR (测试集路径, 在 *_benchmark.py 中配置)

