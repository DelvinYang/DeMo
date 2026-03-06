# SNN Refine Plan (DeMo)

## 0. 目标与结论

当前目标：

- 性能损失控制在 5% 以内
- 训练速度至少提升 50%
- 同时体现 SNN 价值

基于现状（V1/V2/V3 全 SNN 化掉点且变慢），执行策略改为：

1. 暂停 full-SNN 主线
2. 转成 `Hybrid-SNN-Fast` 主线
3. 只在时间冗余高的模块使用 SNN
4. scene/context 与最终连续 readout 保持 ANN

核心判断：先追求可复现实验收益，再追求 SNN 纯度。

---

## 1. 新主线架构

目标模型：`SNNModelForecastFast`

保留 ANN：

- lane encoder
- scene context 主交互块（Transformer/Block）
- mode/state 最终 readout (`y_hat/pi/scal/new_y_hat/new_pi/scal_new`)

使用 SNN：

- agent 历史时序编码（现有 V1）
- 可选：future state rollout 的轻量时序 cell（仅一小段）

不做：

- 纯 spiking scene interaction（V2 的主路径）
- 纯 spiking decoder 全量替换（V3 的主路径）

---

## 2. 代码改造步骤（按顺序）

## Step A: 固定基线与评测协议

1. 基线固定为 `model_forecast`（当前 Raw DeMo）。
2. 使用相同数据比例、epoch、batch、seed 跑三次。
3. 记录：
   - `val_minFDE6`, `val_minADE6`, `val_MR`
   - 每 epoch 耗时、总耗时
   - samples/s 或 steps/s

交付物：`baseline_report.md`。

## Step B: 建立 `SNNModelForecastFast`（从 V1 出发）

修改建议：

1. 复制 `src/model/model_forecast_snn_v1.py` 为 `src/model/model_forecast_snn_fast.py`。
2. 保留 V1 的 `SpikingTemporalEncoder`。
3. 明确恢复 scene 为 ANN：
   - `self.blocks` 强制使用原 `Block`
4. 明确恢复 decoder 为 ANN：
   - 使用 `TimeDecoder`（原版）
5. 输出接口保持与 `ModelForecast` 完全一致。

Trainer 注册：

- 在 `src/model/trainer_forecast.py` 加入 `SNNModelForecastFast` 映射。

Hydra 配置：

- 新建 `conf/model/model_forecast_snn_fast.yaml`。

## Step C: 回退 V2/V3 的训练入口优先级

1. 不删除 V2/V3 文件，但不作为默认实验主线。
2. 所有对比实验优先：
   - `model_forecast` vs `model_forecast_snn_fast`
3. 仅当 `snn_fast` 达到目标后，再重启局部 V2/V3 ablation。

## Step D: 速度优先改造（必须做）

对所有 SNN 子模块统一要求：

1. `step_mode='m'`
2. 避免 `for t in range(T)` 的 Python 单步展开
3. 使用 multi-step 输入批处理
4. 后端优先 `cupy/triton`（可用时）

说明：如果后端仍是纯 torch，先不要宣称“训练速度优势”。

## Step E: 降低时间展开开销

新增配置项（在 `model_forecast_snn_fast.yaml`）：

- `spike_steps: 4`（起步）
- `hist_downsample: 5`（50 步 -> 10 macro steps）
- `spike_tau`
- `spike_v_threshold`

策略：

1. 历史序列先压缩，再做脉冲时序建模。
2. 不做“原始时间长度 × spike_steps”双重展开。

## Step F: 训练稳定性约束

建议加入：

- spike sparsity loss（弱权重）
- membrane stability loss（弱权重）
- 梯度裁剪保持开启
- warmup 适当拉长（SNN 早期更不稳）

优先保证精度，再加稀疏约束。

---

## 3. 实验矩阵（最小可执行）

固定条件：同数据、同训练轮数、同卡数、同 seed 集合。

1. `Raw DeMo`（baseline）
2. `SNN V1`（已有）
3. `SNN Fast`（本次主线）
4. `SNN V2/V3`（仅作为对照，不作为主线）

每组至少 3 个种子，输出：

- 平均与方差
- 相对基线掉点百分比
- 总训练时长与吞吐提升比例

---

## 4. 验收门槛

进入下一阶段前必须满足：

1. 精度损失 <= 5%
2. 训练耗时较当前 full-SNN 明显下降（优先）
3. 若要声称“优于 Raw DeMo”，必须给出严格对照结果

推荐目标顺序：

1. 先赢 full-SNN（速度和精度都更好）
2. 再逼近 Raw DeMo 精度
3. 最后再冲 Raw DeMo 吞吐

---

## 5. 文件级实施清单

必改：

1. `src/model/model_forecast_snn_fast.py`（新增）
2. `conf/model/model_forecast_snn_fast.yaml`（新增）
3. `src/model/trainer_forecast.py`（注册新模型）
4. `change_to_snn.md`（补充“full-SNN 暂停，fast 主线优先”）

可选改：

1. `src/model/layers/spike_encoder.py`（加 downsample + spike_steps）
2. `train.py`（打印 step/s, samples/s）

---

## 6. 一周执行计划

Day 1:

- 建好 `SNNModelForecastFast`
- 跑通单卡 sanity

Day 2:

- 多卡跑通
- 出 first metrics

Day 3:

- `spike_steps`/`hist_downsample` 网格

Day 4:

- 稳定性调参（tau/threshold/dropout/warmup）

Day 5:

- 完成 3-seed 对照报告

---

## 7. 当前执行建议（直接做）

立即执行：

1. 新建 `SNNModelForecastFast`（V1 + ANN scene + ANN decoder）
2. 降低 `spike_steps` 到 4-6，并加历史降采样
3. 使用 multi-step 前向，避免 Python 时间循环
4. 完整跑 3 组对照（Raw/V1/Fast）

在 `SNNModelForecastFast` 达标前，不再扩展 full-SNN 的 V2/V3。
