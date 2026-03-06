# DeMo 到 SNN 分支改造指南（I/O 完全兼容）

本文档用于指导你在当前项目上新建一个 `snn` 分支，并以工程化方式将 DeMo 逐步改造成 SNN 版本。核心原则是：

- 不改数据接口
- 不改训练入口（Hydra + Lightning）
- 不改评测接口
- 模型输入输出与当前 DeMo **完全一致**
- SNN 相关实现统一基于 **spikingjelly**

当前策略补充（性能与速度优先）：

- full-SNN（V2/V3 全替换）不再作为默认主线
- 默认主线改为 `SNNModelForecastFast`（SNN temporal encoder + ANN scene/decoder）
- 优先保证掉点可控与训练吞吐，再做更激进的 SNN 替换

---

## 0. 硬性要求：使用 spikingjelly

从 `snn` 分支开始，所有脉冲神经网络组件必须优先使用 `spikingjelly`，包括但不限于：

- neuron（如 LIF/IF/ALIF 等）
- surrogate gradient
- 多步时序前向（multi-step）
- 与 activation-based 框架相关的功能组件

不建议自定义一套并行 neuron 栈，除非 `spikingjelly` 无法满足需求且有明确性能/功能证据。

---

## 1. 分支创建与基线冻结

先在当前可运行代码基础上创建分支：

```bash
git checkout -b snn
```

然后做一次基线记录（建议在你本地跑）：

```bash
PYTHONPATH=. HYDRA_FULL_ERROR=1 python train.py
```

记录内容建议包括：

- 使用的配置文件与关键超参（`gpus`、`batch_size`、`epochs`、`limit_*_batches`）
- 验证指标（`val_minFDE6`、`val_minADE6`、`val_MR` 等）
- 训练日志与 ckpt 路径

目标是后续每个 SNN 阶段都和这个基线做对照。

---

## 2. 兼容性红线（必须满足）

SNN 改造过程中，以下内容必须保持兼容：

1. `train.py` / `eval.py` 入口不变  
2. datamodule 输入字段不变（`data` 字典 key 不变）
3. `forward(data)` 输出字典 key 不变，至少保持：
   - `y_hat`
   - `pi`
   - `y_hat_others`
   - `dense_predict`
   - `new_y_hat`
   - `new_pi`
   - `scal`
   - `scal_new`
4. 输出 shape 与语义与 DeMo 一致（供现有 loss/metrics/evaluator 直接使用）

建议加一个“兼容性单测”：给定同一 batch，验证 SNN 模型输出 key 集合与每个 tensor shape 均匹配原模型。

---

## 3. 推荐改造顺序（S-DeMo 路线）

不要一次性全网替换，按以下顺序推进：

1. `V1`：先 SNN 化 agent 历史时序编码
2. `V2`：再 SNN 化 scene context 交互
3. `V3`：最后 SNN 化 time decoder（mode/state 双分支）

每个阶段都保证“可训练、可评测、可回退”。

---

## 4. 目录与代码落点建议

在 `src/model/layers/` 新增以下文件（建议）：

- `spike_neuron.py`：LIF/ALIF、surrogate gradient
- `spike_blocks.py`：spiking residual block、membrane norm、temporal pooling
- `spike_encoder.py`：连续输入到脉冲表示的编码
- `spike_interaction.py`：spiking scene interaction（graph/attention/hybrid）
- `spike_decoder.py`：spiking mode/state 解码

实现约束：

- 优先从 `spikingjelly.activation_based` 生态选型
- 新增层尽量封装为对 `spikingjelly` 的轻包装，避免重复造轮子

在 `src/model/` 新增：

- `model_forecast_snn_v1.py`
- `model_forecast_snn_v2.py`
- `model_forecast_snn_v3.py`

并在 trainer 的 `get_model()` 注册新模型类型，保持旧模型可并行共存。

---

## 5. 阶段实现细则

## 5.1 V1：SNN Agent Temporal Encoder（最小可行版）

替换 `ModelForecast` 中：

- `hist_embed_mlp`
- `hist_embed_mamba`

改为：

- `SpikeInputProjector`
- `SpikingTemporalEncoder`

其他模块（lane、scene block、time decoder）先不动。

目标：

- 保持接口不变
- 跑通训练
- 对比基线指标，确认仅 temporal SNN 化的收益/损失

## 5.2 V2：SNN Scene Context

替换 `self.blocks`（原 Transformer block）为 `SpikingInteractionBlock x L`。

建议支持可切换配置：

- `interaction_type: transformer | spiking_graph | hybrid`

优先做 `hybrid`（更稳），再尝试纯 spiking。

## 5.3 V3：SNN Time Decoder

将 `time_embedding_mlp + time_decoder` 改造成脉冲双分支：

- Intention 分支（mode）
- Dynamic state 分支（future rollout）

注意：

- 保留 `pi` 与轨迹连续输出
- 最终 readout 仍输出连续坐标（任务本质是连续回归）

---

## 6. 训练策略（SNN 友好）

建议在配置中逐步加入：

- 更长 warmup
- 更小学习率
- surrogate gradient clipping
- spike sparsity regularization
- membrane stability regularization

可使用两段训练：

1. 分模块训练（先 V1，再 V2，再 V3）
2. 端到端联合微调

如需 warm start，可复用现有 `pretrained_weights` 加载逻辑初始化共享线性层/embedding。

依赖要求：

- 在环境中安装 `spikingjelly`（建议固定版本，避免行为漂移）
- 在 `requirements.txt` 或单独环境文件中显式声明 `spikingjelly`

---

## 7. 配置管理建议（Hydra）

新增配置文件建议：

- `conf/model/model_forecast_snn_v1.yaml`
- `conf/model/model_forecast_snn_v2.yaml`
- `conf/model/model_forecast_snn_v3.yaml`

建议关键参数：

- `spike_steps`
- `neuron_type`（`lif`/`alif`）
- `surrogate_type`
- `v_th`、`tau`
- `spike_reg_weight`
- `mem_reg_weight`
- `interaction_type`

保持默认配置可一键切回原 DeMo，方便 A/B 对照。

---

## 8. 验收清单（每阶段都要过）

1. 训练可启动，无 shape/key 报错  
2. 输出字典 key 与 DeMo 完全一致  
3. loss 与 metrics 可正常记录  
4. `eval.py` 可直接跑  
5. 指标可对比基线并留档  
6. 推理速度与显存有记录（至少提供趋势）

---

## 9. 推荐分支提交节奏

建议按以下粒度提交，便于回滚：

1. `chore: add snn base layers and config scaffolding`
2. `feat: add SNNModelForecastV1 temporal encoder`
3. `feat: add SNNModelForecastV2 spiking interaction`
4. `feat: add SNNModelForecastV3 spiking decoder`
5. `exp: tune snn training strategy and regularization`
6. `docs: summarize demo vs s-demo results`

---

## 10. 你可以直接执行的最小起步动作

1. 新建 `snn` 分支  
2. 新增 `spike_neuron.py / spike_blocks.py / spike_encoder.py`  
3. 先实现 `SNNModelForecastV1`（仅替换历史时序编码）  
4. 在 Hydra 增加 `model_forecast_snn_v1.yaml`  
5. 跑通训练并做第一版对照报告

---

如果你希望，我下一步可以继续补一份 `SNNModelForecastV1` 的代码骨架（含类名、`forward` 输入输出模板、以及最小可运行的 neuron/block 实现接口）。
