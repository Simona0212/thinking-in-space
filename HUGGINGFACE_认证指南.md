# HuggingFace认证设置指南

## ❗ 问题

运行评估脚本时出现：
```
Error loading dataset: Token is required (`token=True`), but no token found.
```

所有评估任务失败，原因是VSI-Bench数据集需要HuggingFace账户认证。

## ✅ 解决方案（3种方法）

### 方法1: HuggingFace CLI登录 ⭐ 推荐

这是最简单和最安全的方法，token会被加密保存在本地。

```bash
# 1. 安装HuggingFace CLI（如果还没有）
pip install huggingface_hub

# 2. 登录
huggingface-cli login

# 3. 输入token（从下面的链接获取）
# Token: hf_xxxxxxxxxxxxxxxxxxxxx
```

**获取token**:
1. 访问 https://huggingface.co/settings/tokens
2. 点击 "Create new token"
3. 选择 "Read" 权限
4. 复制生成的token（格式：`hf_xxxxxxxxxxxx`）

**验证登录**:
```bash
huggingface-cli whoami
# 应该显示你的用户名
```

### 方法2: 环境变量

临时设置（当前终端会话有效）：
```bash
export HF_TOKEN="hf_your_token_here"
```

永久设置（添加到shell配置文件）：
```bash
# 对于bash
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc

# 对于zsh
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

### 方法3: 命令行参数

直接在运行时传递token：
```bash
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --hf_token "hf_your_token_here"
```

或在并行脚本中：
```bash
# 编辑run_parallel.sh，在第44行添加：
python $PYTHON_SCRIPT \
    --model_name "$model" \
    --gpu_id $gpu_id \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --hf_token "$HF_TOKEN" \
    > "$log_file" 2>&1
```

## 📋 完整设置步骤

### Step 1: 创建HuggingFace账户

如果还没有账户：
1. 访问 https://huggingface.co/join
2. 注册账户

### Step 2: 申请VSI-Bench访问权限

1. 访问 https://huggingface.co/datasets/nyu-visionx/VSI-Bench
2. 点击 "Request access" 或 "Access repository"
3. 等待批准（通常很快）

### Step 3: 创建访问token

1. 访问 https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 名称：`vsi-bench-eval`
4. 权限：选择 "Read"
5. 点击 "Generate"
6. **立即复制token**（只显示一次！）

### Step 4: 配置认证

选择上面的方法1、2或3之一。

### Step 5: 验证设置

```bash
# 测试数据集访问
python -c "from datasets import load_dataset; ds = load_dataset('nyu-visionx/VSI-Bench', split='test', token=True); print(f'✓ Success! Loaded {len(ds)} samples')"
```

预期输出：
```
✓ Success! Loaded 2000 samples
```

## 🔄 现在重新运行评估

### 单个模型测试
```bash
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --limit 10
```

### 并行评估所有模型
```bash
./run_parallel.sh
```

## 🐛 故障排除

### 问题1: "401 Unauthorized"
**原因**: Token无效或已过期
**解决**:
```bash
# 重新登录
huggingface-cli login --token hf_your_new_token
```

### 问题2: "403 Forbidden"
**原因**: 没有访问VSI-Bench的权限
**解决**:
1. 访问 https://huggingface.co/datasets/nyu-visionx/VSI-Bench
2. 确认已申请并获得访问权限

### 问题3: "Dataset not found"
**原因**: 数据集名称错误或网络问题
**解决**:
```bash
# 检查网络连接
ping huggingface.co

# 验证数据集存在
huggingface-cli repo info nyu-visionx/VSI-Bench --repo-type dataset
```

### 问题4: Token在环境变量中但仍然失败
**原因**: 可能是shell未加载环境变量
**解决**:
```bash
# 检查环境变量
echo $HF_TOKEN

# 如果为空，重新加载配置
source ~/.bashrc  # 或 ~/.zshrc

# 或直接在当前会话设置
export HF_TOKEN="hf_your_token_here"
```

## 📊 优先级总结

| 方法 | 优先级 | 优点 | 缺点 |
|------|-------|------|------|
| CLI登录 | ⭐⭐⭐⭐⭐ | 最安全、一次配置、加密存储 | 需要安装CLI |
| 环境变量 | ⭐⭐⭐⭐ | 灵活、易于CI/CD | 明文存储（需注意安全） |
| 命令行参数 | ⭐⭐⭐ | 临时使用方便 | 命令历史记录中可见 |

## 💡 安全提示

1. ✅ **不要**将token提交到Git仓库
2. ✅ **不要**在日志中打印token
3. ✅ 定期轮换token（每3-6个月）
4. ✅ 使用最小权限（Read即可）
5. ✅ 如果token泄露，立即在HuggingFace设置中撤销

## 🎯 快速检查清单

- [ ] 有HuggingFace账户
- [ ] 已申请VSI-Bench访问权限
- [ ] 创建了Read权限的token
- [ ] 使用方法1/2/3之一配置认证
- [ ] 运行验证命令成功
- [ ] 可以正常加载数据集

全部完成后，评估脚本应该能正常运行了！🎉
