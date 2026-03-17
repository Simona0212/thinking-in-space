# Qwen3-VL架构不匹配错误修复

## 🔴 严重错误

```
You are using a model of type `qwen3_vl` to instantiate a model of type `qwen2_vl`.
RuntimeError: You set `ignore_mismatched_sizes` to `False`

大量权重不匹配：
- model.language_model.layers.*.self_attn.k_proj.weight  | MISMATCH
- model.visual.blocks.*.attn.proj.weight                 | MISMATCH
...
```

## 🔍 问题根源

**Qwen3-VL ≠ Qwen2-VL**

虽然名字相似，但Qwen3-VL和Qwen2-VL是**完全不同的架构**：

| 特征 | Qwen2-VL | Qwen3-VL |
|------|---------|---------|
| 视觉编码器 | ViT-L/14 (1024维) | ViT (1280维) |
| 语言模型 | Qwen2-7B (4096维) | Qwen2.5 (2560维) |
| Attention | 标准 | 带k_norm/q_norm |
| MLP | fc1/fc2 | linear_fc1/linear_fc2 |
| Merger | 简单 | DeepStack + 多层 |

**结论**: 不能用`Qwen2VLForConditionalGeneration`加载Qwen3-VL！

## ✅ 正确的修复

### 修复前（错误）❌
```python
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct"  # ❌ 架构不匹配！
)
```

### 修复后（正确）✅
```python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    trust_remote_code=True,  # ✅ 让HuggingFace自动加载正确的类
    dtype=torch.bfloat16,
    device_map="cuda:0"
)
```

## 🔧 我已经应用的修复

修改了`Qwen3VLEvaluator.load_model()`：

```python
def load_model(self):
    from transformers import AutoProcessor, AutoModel

    # IMPORTANT: Qwen3-VL必须使用AutoModel + trust_remote_code
    # 不要使用Qwen2VLForConditionalGeneration！
    print("Using AutoModel with trust_remote_code=True (required for Qwen3-VL)")

    self.processor = AutoProcessor.from_pretrained(
        self.model_name,
        trust_remote_code=True
    )

    self.model = AutoModel.from_pretrained(
        self.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,  # 或 torch_dtype (自动fallback)
        device_map=self.device
    )
```

**关键点**：
1. ✅ 使用`AutoModel`而不是`Qwen2VLForConditionalGeneration`
2. ✅ 必须设置`trust_remote_code=True`
3. ✅ HuggingFace会自动下载并加载正确的Qwen3-VL类

## ⚠️ 为什么之前的fallback机制没用？

之前的代码尝试了：
```python
try:
    Qwen2VLForConditionalGeneration  # ❌ 架构不匹配
except ImportError:
    try:
        AutoModelForVision2Seq  # ❌ 也不匹配
    except:
        AutoModel  # ✅ 应该直接用这个！
```

但是`Qwen2VLForConditionalGeneration`能成功导入，所以fallback从未触发！

## 🧪 验证修复

```bash
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --limit 3
```

预期输出：
```
Loading Qwen/Qwen3-VL-4B-Instruct...
Using AutoModel with trust_remote_code=True (required for Qwen3-VL)
Downloading/Loading model...
Model loaded on cuda:0
✓ 不再有MISMATCH错误
✓ 不再有RuntimeError
```

## 📊 为什么会发生这个问题？

### 原因1：命名混淆
- HuggingFace上的模型叫`Qwen3-VL`
- 但model type在config中是`qwen3_vl`
- transformers库中最接近的类是`Qwen2VLForConditionalGeneration`
- 结果：自动选择了错误的类

### 原因2：transformers版本
- transformers可能还没有独立的`Qwen3VLForConditionalGeneration`类
- Qwen3-VL使用自定义代码（需要trust_remote_code=True）
- 只有AutoModel才能正确处理

## 💡 教训

对于**新模型**（尤其是带version 3+）：
1. ✅ **优先使用** `AutoModel` + `trust_remote_code=True`
2. ❌ **不要假设** 可以用旧版本的类（如Qwen2VL）
3. ✅ **检查config.json** 中的 `model_type` 字段
4. ✅ **看模型卡** 是否需要`trust_remote_code`

## 🔍 如何识别这类问题？

看到这些信号就要警惕：
- ⚠️ `You are using a model of type X to instantiate a model of type Y`
- ⚠️ 大量`MISMATCH`错误（shape不同）
- ⚠️ `UNEXPECTED` keys（新架构的layer）
- ⚠️ `MISSING` keys（缺少某些layer）

**解决方案**：改用`AutoModel` + `trust_remote_code=True`

## 📝 其他受影响的模型

类似问题可能出现在：
- ✅ Qwen3-VL (已修复)
- ⚠️ 任何version 3+的模型
- ⚠️ 带"改进版"、"增强版"标签的模型
- ⚠️ 模型卡上明确要求`trust_remote_code=True`的模型

## 🎯 通用修复模板

对于任何新模型：

```python
from transformers import AutoModel, AutoProcessor

# Step 1: 检查是否需要trust_remote_code
# (查看模型卡 README)

# Step 2: 使用AutoModel（最安全）
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,  # 关键！
    dtype=torch.bfloat16,
    device_map="auto"
)
```

## ✅ 检查清单

- [x] 修改Qwen3VLEvaluator使用AutoModel
- [x] 添加trust_remote_code=True
- [x] 移除Qwen2VL导入尝试
- [x] 语法检查通过
- [ ] 运行测试验证
- [ ] 确认不再有MISMATCH错误

## 🚀 现在可以测试了

```bash
# 测试Qwen3-VL Instruct
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --limit 3

# 测试Qwen3-VL Thinking
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Thinking" \
    --gpu_id 0 \
    --limit 3

# 如果成功，运行完整评估
./run_parallel.sh
```

## 🎉 成功标志

不再看到：
- ❌ "You are using a model of type qwen3_vl to instantiate qwen2_vl"
- ❌ "MISMATCH: Reinit due to size mismatch"
- ❌ "RuntimeError: You set ignore_mismatched_sizes to False"

而是看到：
- ✓ "Using AutoModel with trust_remote_code=True"
- ✓ "Model loaded on cuda:0"
- ✓ "Processing sample X: Success"

---

## 🆘 如果还是失败

### 选项1：检查transformers版本
```bash
pip install --upgrade transformers>=5.0.0
```

### 选项2：清除缓存重新下载
```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct
python evaluate_vsibench.py --model_name "Qwen/Qwen3-VL-4B-Instruct" --limit 1
```

### 选项3：检查是否真的需要trust_remote_code
```bash
# 查看模型配置
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('Qwen/Qwen3-VL-4B-Instruct', trust_remote_code=True)
print(f'Model type: {config.model_type}')
print(f'Auto map: {config.auto_map if hasattr(config, \"auto_map\") else \"None\"}')
"
```

## 📖 参考

- Qwen3-VL模型卡: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- trust_remote_code文档: https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModel.from_pretrained
