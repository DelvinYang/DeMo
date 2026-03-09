# 基于 SNN 的轨迹预测全流程（当前默认：`model_forecast_snn_fast`）

## 1. 总体流程

当前工程的完整链路是：

1. 原始 AV2 场景（`train/val/test/*.parquet`）
2. `preprocess.py` + `Av2Extractor` 预处理为 `.pt` 样本（`data/DeMo_processed/...`）
3. `Av2Dataset` 做局部坐标变换、历史/未来切分、特征构建
4. `Av2DataModule` + `collate_fn` 组织 batch
5. `Trainer` 调用 `SNNModelForecastFast` 前向预测
6. 计算多项 loss（回归 + 分类 + Laplace + SNN 正则）
7. AdamW + WarmupCosLR 训练，Lightning 记录/保存 checkpoint
8. `eval.py` 做验证或测试导出

---

## 2. 数据预处理（离线）

入口：`preprocess.py`

- 遍历 `data_root/{train,val,test}` 下全部场景 parquet
- `Av2Extractor` 读取轨迹与地图：
  - Agent: `x_positions`, `x_angles`, `x_velocity`, `x_valid_mask`, `x_attr`
  - Map: `lane_positions`, `lane_attr`, `is_intersections`
  - 元信息: `scenario_id`, `agent_ids`, `focal_idx`, `scored_idx`, `city`
- 每个场景保存为一个 `.pt`

输出目录默认：`data/DeMo_processed/{train,val,test}`。

---

## 3. 训练时样本构建（在线）

核心：`src/datamodule/av2_dataset.py`

### 3.1 单样本处理

- 以 focal agent（训练可扩展 scored agent）为中心建立局部坐标系
- 历史长度：`50` 帧；未来长度：`60` 帧（test 无未来 GT）
- 邻域半径：`150m`
- 生成模型输入关键字段：
  - Agent 历史增量：`x_positions_diff`, `x_velocity_diff`, `x_valid_mask`
  - Agent 静态属性：`x_attr`
  - Agent 几何：`x_centers`, `x_angles`
  - Lane：`lane_positions`, `lane_centers`, `lane_angles`, `lane_valid_mask`, `lane_attr`
  - 监督：`target`, `target_mask`（以及 diff 辅助量）

### 3.2 Batch 拼接

`collate_fn` 对变长 agent/lane 维度做 pad，并构造：

- `x_key_valid_mask = x_valid_mask.any(-1)`
- `lane_key_valid_mask = lane_valid_mask.any(-1)`

这些 mask 是后续 SNN 图选择与解码的重要条件。

---

## 4. 网络架构（`SNNModelForecastFast`）

实现文件：`src/model/model_forecast_snn_fast.py`

### 4.1 当前默认超参（`conf/model/model_forecast_snn_fast.yaml`）

- `embed_dim=128`
- `future_steps=60`
- `max_lane_tokens=24`
- `active_agents=16`
- `lane_tokens=24`
- `graph_depth=2`
- `hybrid_scene_depth=1`
- `hybrid_num_heads=4`
- `hybrid_mlp_ratio=2.0`
- `compressed_steps=10`
- `spike_depth=2`
- `spike_tau=2.5`
- `spike_v_threshold=0.9`
- `recent_frames=6`
- `recent_residual_weight=0.35`
- 结构化 lane 采样：`near_ego=12`, `near_goal=8`, `diverse=4`

### 4.2 主干结构

整体是：

`TemporalConvSNNEncoder -> EventSceneGraph -> LightGlobalContext -> 1层Hybrid Transformer -> TwoStageFastDecoder`

主干输入输出主形状（batch 记为 `B`）：

- agent token: `actor_feat [B, N, 128]`
- lane token: `lane_feat [B, M, 128]`（结构化裁剪后 `M<=24`）
- 融合编码: `x_encoder [B, (N+M), 128]`
- 多模态轨迹: `y_hat/new_y_hat [B, 6, 60, 2]`

#### A. TemporalConvSNNEncoder（时序 SNN 编码）

文件：`src/model/layers/temporal_conv_snn.py`

输入（每个 agent 历史序列特征，4 维）：

- `dx, dy, dvel, valid`

层级细节（默认配置）：

1. `Conv1d(4, 64, k=3, p=1) + GELU`
2. `Conv1d(64, 128, k=3, p=1) + GELU`
3. `AdaptiveAvgPool1d(output_size=10)`，时间维从 `50 -> 10`
4. `LayerNorm(128)`
5. `LIFNode x 2`（`tau=2.5`, `v_threshold=0.9`, `step_mode='m'`, surrogate=`ATan`）
6. `Linear(128, 128)` 输出时序特征
7. 取最后有效时刻特征作为 actor 表征，并计算 `spike_rate`
8. 最近帧残差支路：
   - `Linear(4,128) + GELU + Linear(128,128)`
   - 对最近 `6` 帧均值池化后，按 `0.35` 权重加回主分支

该模块额外输出两项 SNN 辅助损失：

- `spike_sparsity_loss`：脉冲激活稀疏度
- `membrane_stability_loss`：膜电位时序稳定项

#### B. 结构化 Lane 选择

在 `SNNModelForecastFast._select_structured_lanes` 中进行，不是简单最近 Top-K：

- 近 ego 的 lane
- 朝 ego 前向 goal 的 lane
- 与 ego 朝向差异较大的 diverse lane
- 剩余额度用近距补齐

这样在固定 token 预算下保留更多拓扑语义。

#### C. EventSceneGraph（稀疏场景交互）

文件：`src/model/layers/event_scene_graph.py`

层级细节（默认配置）：

1. active agent 选择：
   - 用 `spike_rate [B,N]` 在 valid agent 内做 Top-K（`K=16`）
2. lane 选择：
   - 计算 active agent 到 lane center 的 `cdist`
   - 每个 batch 选最近 `24` 个 lane
3. 节点拼接：
   - `nodes = [active_actor_nodes ; selected_lane_nodes]`
4. 图更新层（重复 `depth=2` 次）：
   - `Linear(128,128) -> GELU -> Linear(128,128)` + residual
   - `LayerNorm(128)`
5. scatter 回原 actor/lane token 位置，未选中的 token 保持原值

#### D. LightGlobalContext（全局补偿分支）

文件：`src/model/layers/light_global_context.py`

层级细节：

1. `actor_ctx = masked_mean(actor_feat)`
2. `lane_ctx = masked_mean(lane_feat)`
3. `global_ctx`：
   - `Linear(256,128) -> GELU -> Linear(128,128)`
4. token gate：
   - actor gate: `Linear(128,128) -> Sigmoid`
   - lane gate: `Linear(128,128) -> Sigmoid`
5. 回注：
   - `actor_feat += gate_actor * global_ctx`
   - `lane_feat += gate_lane * global_ctx`

#### E. Hybrid Scene Block

在拼接后的 `x_encoder [B,N+M,128]` 上执行：

1. 1 层轻量 Transformer Block（`num_heads=4`, `mlp_ratio=2.0`）
2. `LayerNorm(128)` 得到最终场景编码

#### F. TwoStageFastDecoder（两阶段快速解码）

文件：`src/model/layers/two_stage_fast_decoder.py`

关键子层：

1. scene 聚合：
   - `scene_feat = masked_mean(x_encoder)`
   - 再加 `global_ctx`
2. 模态查询（`K=6`）：
   - `mode_feat = scene_feat + Embedding(6,128)`
   - `mode_logits: Linear(128,128)->GELU->Linear(128,1)` 得 `pi`
3. Stage-1 粗轨迹：
   - `coarse_head: Linear(128,256)->GELU->Linear(256,12)`（`6*2`）
   - 线性插值 `6步 -> 60步`
   - `refine_head: Linear(128,256)->GELU->Linear(256,120)`（`60*2`）
   - `y_hat = coarse_interp + refine`
4. Stage-2 细化：
   - `refine_head_stage2: Linear(128,256)->GELU->Linear(256,120)`
   - `new_y_hat = y_hat + 0.5 * refine_stage2`
5. 不确定性尺度：
   - `scale_head` 输出 `scal [B,6,60,2]`
   - `refine_scale_head` 输出 `scal_new [B,6,60,2]`
   - 用 `ELU + 1 + 1e-4` 保证尺度为正
6. dense 分支：
   - `dense_head: Linear(128,256)->GELU->Linear(256,120)`
   - 输出 `dense_predict [B,60,2]`

### 4.3 输出接口（与原 DeMo 保持兼容）

`forward(data)` 输出固定字典：

- `y_hat`, `pi`, `scal`
- `dense_predict`
- `y_hat_others`
- `new_y_hat`, `new_pi`, `scal_new`
- `spike_sparsity_loss`, `membrane_stability_loss`

这保证了 trainer/loss/metrics/eval 无需改接口。

---

## 5. 训练目标与优化

实现：`src/model/trainer_forecast.py`

### 5.1 Loss 组成

总损失为以下项加和：

1. 主模态轨迹回归：`agent_reg_loss`（best mode SmoothL1）
2. 主模态分类：`agent_cls_loss`（`pi` 交叉熵）
3. 其他 agent 轨迹回归：`others_reg_loss`
4. Dense state 分支回归：`dense_reg_loss`
5. 二阶段 refine 回归：`new_agent_reg_loss`
6. refine 分类：`new_pi_reg_loss`
7. Laplace NLL（主输出）：`laplace_loss`
8. Laplace NLL（refine 输出）：`laplace_loss_new`
9. SNN 正则：
   - `spike_reg_weight * spike_sparsity_loss`
   - `mem_reg_weight * membrane_stability_loss`

默认正则权重（fast 配置）：

- `spike_reg_weight = 5e-5`
- `mem_reg_weight = 1e-4`

### 5.2 优化器与学习率

- Optimizer: AdamW（按参数类型分 decay/no_decay）
- Scheduler: `WarmupCosLR`
  - warmup: 前 `warmup_epochs=10`
  - cosine 衰减到 `min_lr=1e-5`
- 默认全局：`lr=0.003`, `weight_decay=1e-2`, `epochs=60`

### 5.3 训练器设置（train.py）

- Lightning DDP 多卡训练
- SNN 模型自动把 `precision` 从 `32-true` 切到 `bf16-mixed`
- `SNNModelForecastFast` 会把 `log_every_n_steps` 至少提高到 200

---

## 6. 评估与推理

- 训练：`python train.py model=model_forecast_snn_fast`
- 验证：`python eval.py model=model_forecast_snn_fast`
- 测试提交：`python eval.py model=model_forecast_snn_fast gpus=1 test=true`

指标由 `src/metrics` 计算，核心包括：

- `minADE1/minADE6`
- `minFDE1/minFDE6`
- `MR`
- `brier-minFDE6`

---

## 7. 你这条 SNN Fast 路线的核心特点（总结）

1. SNN 只放在最值钱且开销可控的时序编码（Temporal）
2. 场景交互走稀疏图 + 轻全局补偿 + 1 层轻 Transformer
3. 解码器用两阶段 coarse-to-fine 保留多模态与长时细化能力
4. 全流程 I/O 与原 DeMo 保持兼容，可直接复用原训练与评测框架
