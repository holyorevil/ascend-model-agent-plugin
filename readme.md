# ascend-model-agent-plugin 使用说明

## 快速开始

### 1. 插件安装

#### 单次使用
使用如下命令打开claude，该插件在本次claude会话周期内生效
```bash
# 系统级 skills 目录
claude --plugin-dir ./ascend-model-agent-plugin
```

#### 系统级安装
将 `ascend-model-agent-plugin` 目录复制到 Claude 的 skills 目录：

```bash
# 系统级 skills 目录
cp -r ascend-model-agent-plugin ~/.claude/skills/

# 执行 claude 目录
claude
```

### 2. 使用技能

Claude 会根据对话内容**自动触发**相应技能，无需手动选择。

## 技能详解

### 1. ai4s-basic（通用昇腾迁移）

**描述**：通用昇腾 NPU 模型迁移 Skill

**适用场景**：
- PyTorch / TensorFlow / vLLM 项目迁移到华为昇腾 NPU
- CUDA 代码转 NPU 代码
- 环境配置和依赖适配

**触发词**：
```
"昇腾迁移"
"NPU 适配"
"CUDA 转 NPU"
"模型移植到华为 NPU"
```

**核心流程**：
1. 环境检查与代码分析
2. 自动迁移注入 (`transfer_to_npu`)
3. 手动修改 CUDA 依赖
4. 分布式适配（nccl → hccl）
5. 第三方库适配
6. 验证与测试

**关键命令**：
```bash
# 入口脚本顶部添加
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 查看 NPU 状态
npu-smi info

# 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

---

### 2. adapt-agent（GPU 代码审查）

**描述**：GPU 到昇腾 NPU 适配审查专家

**适用场景**：
- 深度审查 GPU 代码仓库
- 生成完整适配报告
- 识别迁移堵点

**触发词**：
```
"审查代码"
"NPU 适配审查"
"生成适配报告"
"GPU 转 NPU 分析"
```

**输出**：
- 代码结构分析报告
- 堵点清单及迁移方案
- 适配脚本（`npu_compat.py`、`npu_ops.py`）
- 验证脚本（`verify_npu.sh`）
- Markdown 审查报告（`CodeReview_Results_YYYY-MM-DD.md`）

---

### 3. ascend-optimization（推理优化）

**描述**：torch_npu 推理性能优化

**适用场景**：
- 提升模型在 NPU 上的推理速度
- 使用融合算子替换原生算子
- Python 层性能优化

**触发词**：
```
"torch_npu 优化"
"NPU 推理性能"
"算子替换"
"融合算子"
```

**核心优化项**：

| 优化项 | 原始实现 | NPU 优化 | 典型收益 |
|-------|---------|---------|---------|
| RMSNorm | 手写 Python | `npu_rms_norm` | +5~10% |
| SwiGLU | SiLU + 乘法 | `npu_swiglu` | +5~10% |
| RoPE | 手写旋转编码 | `npu_rotary_mul` | +5~10% |
| Attention | Q@K^T→softmax→@V | `npu_fusion_attention` | +15~20% |

**精度验收标准**：
- Logits 余弦相似度 > 0.99
- PPL 平均相对差异 < 15%
- 使用 pretrained 权重对比

---

### 4. quant-by-modelslim（昇腾工具链）

**描述**：昇腾 NPU 推理工具链入口

**适用场景**：
- vLLM 在昇腾上的安装和运行
- msmodelslim 模型量化
- AISBench 精度评估
- NPU 故障排查

**触发词**：
```
"vLLM 昇腾"
"模型量化"
"msmodelslim"
"NPU 报错"
```

**前置检查**：
```bash
npu-smi info
```

**关键约束**：
- 所有命令必须通过 shell 脚本运行，并保存日志
- 使用 `ASCEND_RT_VISIBLE_DEVICES` 控制可见 NPU
- 支持可编辑安装调试

---

### 5. vllm-ascend-model-adapter（vLLM 适配）

**描述**：vLLM 在昇腾 NPU 上的适配和部署

**触发词**：
```
"vLLM 部署"
"昇腾 vLLM"
"vLLM 多卡"
```

---

### 6. model-series-vendor-detector（模型识别）

**描述**：根据模型名称识别系列和供应商

**触发词**：
```
"这是什么模型"
"模型系列"
"模型供应商"
```

**支持的系列**：GLM、Qwen、DeepSeek、MiniCPM、Llama、Gemma 等

---

### 7. hardware-check-principle（硬件检查）

**描述**：NPU 硬件状态检查

**触发词**：
```
"检查 NPU"
"npu-smi"
"硬件状态"
```

---

### 8. adapter-check-principle（适配器检查）

**描述**：适配器审查检查

**触发词**：
```
"检查适配器"
"适配器审查"
```

---

### 9. ascend-model-verification（模型验证）

**描述**：模型验证流程

**触发词**：
```
"验证模型"
"benchmark"
"精度测试"
```

---

### 10. repo-reader（仓库分析）

**描述**：代码仓库分析和读取

**触发词**：
```
"分析仓库"
"读取代码"
"代码结构"
```

## 典型工作流

### 工作流 1：全新模型迁移

```
用户：将 Qwen2.5-7B 迁移到昇腾 NPU

Claude 触发：ai4s-basic

1. 环境检查：npu-smi info
2. 代码分析：识别 CUDA 依赖
3. 自动迁移：添加 transfer_to_npu
4. 手动修复：处理不兼容 API
5. 验证运行：测试推理
```

### 工作流 2：GPU 代码审查

```
用户：审查这个 GPU 项目并生成适配报告

Claude 触发：adapt-agent

1. 克隆仓库
2. 全面扫描 CUDA 依赖
3. 识别所有堵点
4. 生成适配脚本
5. 输出审查报告 CodeReview_Results_2024-01-15.md
```

### 工作流 3：推理优化

```
用户：优化这个模型在 NPU 上的推理速度

Claude 触发：ascend-optimization

1. 创建 model_files/ 目录
2. 替换 RMSNorm → npu_rms_norm
3. 替换 SwiGLU → npu_swiglu
4. 替换 Attention → npu_fusion_attention
5. 运行精度对比验证
```

### 工作流 4：vLLM 部署

```
用户：在昇腾上部署 vLLM 服务

Claude 触发：quant-by-modelslim + vllm-ascend-model-adapter

1. 硬件检查：npu-smi info
2. 安装 vllm-ascend
3. 配置多卡环境
4. 启动服务
5. 故障排查
```
