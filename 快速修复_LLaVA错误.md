# LLaVA-OneVision错误快速修复

## 🔴 问题
```
AttributeError/ImportError during dynamic module execution
```

## ✅ 解决方案

### 快速修复（2步）

```bash
# 1. 安装缺失依赖
pip install einops timm

# 2. 尝试安装flash-attn（失败也没关系）
pip install flash-attn --no-build-isolation || echo "跳过flash-attn"
```

### 如果还是失败 - 暂时跳过LLaVA模型

编辑`run_parallel.sh`第13-21行：

```bash
MODELS=(
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-4B-Thinking"
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-8B-Thinking"
    # "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"  # 暂时注释
    # "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"  # 暂时注释
)
```

然后运行：
```bash
./run_parallel.sh
```

## 🔧 我已经做的代码修复

1. ✅ 添加了多层fallback机制
2. ✅ 使用`attn_implementation="eager"`避免flash-attn问题
3. ✅ 支持多种推理方法
4. ✅ 详细的错误处理

## 🧪 测试

```bash
# 测试LLaVA是否能工作
python evaluate_vsibench.py \
    --model_name "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" \
    --gpu_id 0 \
    --limit 1
```

## 💡 建议

**优先评估Qwen3-VL和BAGEL模型**，它们更稳定。LLaVA-OneVision可以稍后单独处理。

完整说明见：`LLAVA_加载错误修复.md`
