# 快速修复 - Qwen3-VL架构不匹配

## 🔴 错误
```
You are using a model of type `qwen3_vl` to instantiate `qwen2_vl`
RuntimeError: ignore_mismatched_sizes=False
大量MISMATCH错误
```

## 🔍 原因
**Qwen3-VL ≠ Qwen2-VL** （完全不同的架构！）

## ✅ 我已修复
修改了代码，现在**直接使用AutoModel + trust_remote_code=True**

```python
# 修复前 ❌
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(...)  # 架构不匹配！

# 修复后 ✅
from transformers import AutoModel
model = AutoModel.from_pretrained(
    ...,
    trust_remote_code=True  # 让HuggingFace自动加载正确的类
)
```

## 🧪 验证
```bash
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --limit 3
```

预期：
- ✓ "Using AutoModel with trust_remote_code=True"
- ✓ 不再有MISMATCH错误
- ✓ "Model loaded on cuda:0"

## 🚀 运行评估
```bash
./run_parallel.sh
```

## 💡 教训
对于**新模型**（version 3+），总是用：
```python
AutoModel.from_pretrained(..., trust_remote_code=True)
```

不要假设可以用旧版本的类！

详细说明见：`Qwen3VL_架构不匹配修复.md`
