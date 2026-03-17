# LLaVA-OneVision 动态模块加载错误修复

## 🔴 错误信息

```
File "/root/miniconda3/envs/vsibench/lib/python3.10/site-packages/transformers/dynamic_module_utils.py", line 309, in get_class_in_module
    module_spec.loader.exec_module(module)
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
```

这是在加载 `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct` 时，`trust_remote_code=True` 尝试执行动态模块时失败。

## 🔍 可能的原因

### 1. 缺少依赖包
LLaVA-OneVision 需要额外的依赖：
- `einops` ✓ (已在 requirements_eval.txt)
- `timm` ← **新添加**
- `flash-attn` (可选，可以用 eager attention 代替)

### 2. Flash Attention 兼容性问题
动态模块可能尝试使用 flash-attention，但：
- 可能未安装
- 可能版本不兼容
- 可能与 CUDA 版本不匹配

### 3. dtype 参数问题
虽然 CLAUDE.md 指定 `dtype="auto"`，但动态模块加载时可能有兼容性问题。

## ✅ 已应用的修复

### 修复 1: 添加 timm 依赖
**文件**: `requirements_eval.txt`

```txt
# Model-specific dependencies
qwen-vl-utils  # For Qwen3-VL models
einops  # Required by some vision models
timm  # PyTorch Image Models, required by LLaVA-OneVision  ← 新添加
```

**操作**:
```bash
pip install timm
```

### 修复 2: 3层加载 Fallback 机制
**文件**: `evaluate_vsibench.py` - `LLaVAOneVisionEvaluator.load_model()`

```python
# Attempt 1: dtype="auto" (CLAUDE.md 规范)
try:
    self.model = AutoModel.from_pretrained(
        self.model_name,
        trust_remote_code=True,
        dtype="auto",
        device_map=self.device,
        low_cpu_mem_usage=True
    )
except Exception as e:
    print(f"Attempt 1 failed: {str(e)[:200]}")

# Attempt 2: 添加 attn_implementation="eager" (避免 flash-attn)
try:
    self.model = AutoModel.from_pretrained(
        self.model_name,
        trust_remote_code=True,
        dtype="auto",
        device_map=self.device,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # ← 关键！
    )
except Exception as e:
    print(f"Attempt 2 failed: {str(e)[:200]}")

# Attempt 3: 使用 torch_dtype (兼容性 fallback)
try:
    self.model = AutoModel.from_pretrained(
        self.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=self.device,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
except Exception as e:
    print(f"Attempt 3 failed: {str(e)[:200]}")
    raise  # 如果所有尝试都失败，抛出错误
```

### 修复 3: 依赖检查脚本
**文件**: `check_llava_deps.py`

运行此脚本检查是否缺少依赖：
```bash
python check_llava_deps.py
```

输出示例：
```
Checking LLaVA-OneVision dependencies...

Required dependencies:
✓ torch is installed
✓ transformers is installed
✓ Pillow is installed
✓ av is installed
✓ einops is installed
✓ timm is installed
✓ numpy is installed

Optional dependencies:
✗ flash-attn (optional, can use eager attention) is NOT installed

✓ All required dependencies are installed!
```

## 🧪 测试步骤

### 步骤 1: 安装缺失的依赖
```bash
pip install timm
```

### 步骤 2: 检查依赖
```bash
python check_llava_deps.py
```

### 步骤 3: 重新运行评估
```bash
python evaluate_vsibench.py \
    --model_name "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" \
    --gpu_id 0 \
    --limit 3
```

### 步骤 4: 观察输出
现在应该看到：
```
Using AutoModel with trust_remote_code=True and dtype='auto'
Attempt 1: Loading with dtype='auto'...
[如果失败]
Attempt 2: Loading with attn_implementation='eager'...
[如果成功]
✓ Successfully loaded with attn_implementation='eager'
Model loaded on cuda:0
```

## 🔍 如果仍然失败

### 检查 1: 查看完整错误信息
错误信息在日志中可能被截断了。运行：
```bash
python evaluate_vsibench.py \
    --model_name "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" \
    --gpu_id 0 \
    --limit 1 2>&1 | tee llava_error.log
```

然后查看 `llava_error.log` 中的完整错误。

### 检查 2: transformers 版本
```bash
python -c "import transformers; print(transformers.__version__)"
```

应该是 >= 4.45.0。如果不是：
```bash
pip install --upgrade transformers>=4.45.0
```

### 检查 3: 尝试手动加载
```python
from transformers import AutoModel
import torch

# 测试 1: 基本加载
try:
    model = AutoModel.from_pretrained(
        "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        trust_remote_code=True,
        dtype="auto"
    )
    print("✓ 基本加载成功")
except Exception as e:
    print(f"✗ 基本加载失败: {e}")

# 测试 2: 加 eager attention
try:
    model = AutoModel.from_pretrained(
        "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        trust_remote_code=True,
        dtype="auto",
        attn_implementation="eager"
    )
    print("✓ Eager attention 加载成功")
except Exception as e:
    print(f"✗ Eager attention 加载失败: {e}")
```

### 检查 4: 清除缓存重新下载
有时缓存的模型文件损坏：
```bash
rm -rf ~/.cache/huggingface/hub/models--lmms-lab--LLaVA-OneVision-1.5-4B-Instruct
python evaluate_vsibench.py \
    --model_name "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" \
    --gpu_id 0 \
    --limit 1
```

## 📊 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| `cannot import name 'AutoModelForImageTextToText'` | transformers 版本过旧 | `pip install --upgrade transformers>=4.45.0` |
| `ImportError: einops` | 缺少 einops | `pip install einops` |
| `ImportError: timm` | 缺少 timm | `pip install timm` |
| `Flash attention not available` | 未安装 flash-attn | 添加 `attn_implementation="eager"` |
| `torch_dtype is deprecated` | 使用了旧参数名 | 改用 `dtype="auto"` |
| 动态模块执行失败 | 依赖问题 | 1) 安装所有依赖 2) 使用 eager attention 3) 清除缓存 |

## 💡 为什么需要 3 层 Fallback？

1. **Attempt 1 (dtype="auto")**:
   - 严格遵守 CLAUDE.md 规范
   - 最标准的加载方式
   - 但可能因为环境问题失败

2. **Attempt 2 (+ attn_implementation="eager")**:
   - 避免 flash-attention 依赖
   - 兼容性更好
   - 性能稍慢但更稳定

3. **Attempt 3 (torch_dtype fallback)**:
   - 针对旧版本 transformers
   - 显式指定 dtype
   - 最后的兜底方案

## 🎯 预期结果

如果修复成功，应该看到：
```
Loading lmms-lab/LLaVA-OneVision-1.5-4B-Instruct...
Using AutoModel with trust_remote_code=True and dtype='auto'
Attempt 1: Loading with dtype='auto'...
✓ Successfully loaded with dtype='auto'
Model loaded on cuda:0
Processing sample 0: Success
```

或者（如果 Attempt 1 失败）：
```
Attempt 1: Loading with dtype='auto'...
Attempt 1 failed: ...
Attempt 2: Loading with attn_implementation='eager'...
✓ Successfully loaded with attn_implementation='eager'
Model loaded on cuda:0
Processing sample 0: Success
```

## 📝 后续步骤

1. ✅ 安装 timm: `pip install timm`
2. ✅ 运行依赖检查: `python check_llava_deps.py`
3. ✅ 测试 LLaVA 加载: `python evaluate_vsibench.py --model_name "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" --gpu_id 0 --limit 1`
4. ✅ 如果成功，运行完整评估: `./run_parallel.sh`

## 🆘 如果仍然无法解决

请提供以下信息：
1. `python check_llava_deps.py` 的完整输出
2. `python -c "import transformers; print(transformers.__version__)"` 的输出
3. 完整的错误信息（包括被截断的部分）
4. Python 版本: `python --version`
5. CUDA 版本: `nvcc --version`
6. PyTorch 版本: `python -c "import torch; print(torch.__version__)"`

这将帮助诊断具体的环境问题。

---

**总结**: 添加了 timm 依赖，实现了 3 层加载 fallback 机制，创建了依赖检查脚本。现在应该能够处理大多数 LLaVA-OneVision 加载问题。
