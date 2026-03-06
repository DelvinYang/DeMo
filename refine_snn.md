这说明你这版已经**跨过了“能不能加速”**，现在进入第二阶段：
**你删对了计算量，但删错了信息流。**

速度大幅上升、性能掉 50% 以上，通常不是“小调参”问题，而是当前 `TemporalConvSNNEncoder -> EventSceneGraph -> FastDecoder` 这条链路里，至少有 **1–2 个关键建模能力被砍断了**。对 DeMo 这种任务，最容易被砍坏的就是这三件事：

1. **多模态能力没了**
2. **地图约束/交互约束太弱了**
3. **轻量 decoder 过于“直出”，没有逐步 rollout 能力**

所以现在不要继续只调 `tau / v_th / spike_reg_weight` 这类细节了。
你需要做的是：**保留当前快结构，但把原 DeMo 最值钱的 inductive bias 一部分接回来。**

---

# 先判断掉点来自哪一层

你现在这 50% 以上的掉点，大概率来自下面这个组合：

### 1. EventSceneGraph 太稀

你把全量 transformer 改成稀疏事件图后，**图是快了**，但很可能：

* active agents 选得太少
* lane token 压到 16 后丢了关键拓扑
* 图边只保留局部邻居，缺失长程约束

结果就是 scene 理解能力断崖下降。

### 2. FastDecoder 太轻

原 DeMo 的 time decoder 虽然重，但它确实承担了：

* mode/state 解耦
* 多未来步状态展开
* 模态内动态细化

你现在如果是：

```python
scene_feat -> mode logits
scene_feat -> trajectory MLP
```

这种一把出 60 步，通常会很快，但 motion forecasting 会非常容易掉点。

### 3. TemporalConvSNNEncoder 压缩过头

`50 -> 8` 虽然快，但可能把：

* 微小加减速变化
* lane change precursor
* interaction intent 先兆
  直接压没了。

所以你现在的核心方向不是“再轻量”，而是：

> **把少量高价值能力加回来，但不能把 FLOPs 加回原 DeMo。**

---

# 正确的修复路线：做 “Fast-but-Accurate” 而不是继续极限压缩

我建议你接下来不要推翻现有快架构，而是在现有基础上做 **三处加法**：

## 路线总览

把当前：

```text
TemporalConvSNNEncoder
-> EventSceneGraph
-> FastDecoder
```

改成：

```text
TemporalConvSNNEncoder
-> Dual-Path Scene Encoder
-> Two-Stage Fast Decoder
```

核心思想：

* **主路径保持快**
* **补一个极轻的精度修复路径**
* **让 decoder 从“一步直出”变成“粗到细”**

---

# 一、先不要恢复原 Transformer，而是做 Dual-Path Scene Encoder

这是最重要的一步。

你现在的问题大概率不是“scene graph 不行”，而是**scene graph 太单一路径**。
解决方法不是直接回原 5 层 transformer，而是做：

## 双路径 Scene Backbone

### Path A：当前 EventSceneGraph（保留）

负责速度，继续稀疏图。

### Path B：轻量 global context path（新增）

只做 very cheap 的全局补偿。

例如：

* 对所有 actor/lane token 做一次 very light pooling
* 或做 1 层 low-rank attention
* 或做 1 层 tiny transformer（dim 减半，head 减半，只有 1 层）

然后把两路融合：

[
z_{\text{scene}} = z_{\text{event}} + \alpha z_{\text{global}}
]

这一步的作用很大：
Event graph 很容易只会“局部邻接交互”，但 motion forecasting 往往需要一点点全局语义，比如：

* 车流整体方向
* 远端 lane continuation
* 周边 agent 模态分布

你不需要把全局主干恢复完整，只要补一个 **超轻 global correction branch**，掉点通常会明显回收。

---

## 推荐实现

新增一个：

```python
LightGlobalContext
```

输入仍然是 actor + lane tokens，但只做下面之一：

### 方案 A：全局 pooling MLP

最快，最稳：

```python
global_actor = mean(topk_actor_tokens)
global_lane = mean(topk_lane_tokens)
global_ctx = MLP(concat(global_actor, global_lane))
```

再把它 broadcast 回 ego token。

### 方案 B：单层 low-rank attention

更强一点：

* token 数仍控制很小
* hidden dim 降到原来一半
* 仅 1 layer

### 方案 C：landmark tokens

从 lane tokens 中不是只取最近 16 个，而是：

* 最近 lanes 8 个
* 前向延展 lanes 4 个
* 全局 landmark lanes 4 个

这个会比“纯最近 lane”好很多。

---

# 二、Lane token 压缩不能只按距离，必须改成“结构化保留”

你现在默认压到 16 个 lane token，这个思路没错，但**只按距离 Top-K** 很容易出大问题。

因为最近的不一定最重要。
对轨迹预测来说，更关键的是：

* 当前所在 lane
* 左右相邻 lane
* 前方延伸 lane
* 候选变道 lane
* 冲突 lane / merge lane

所以建议你把 lane selection 从：

```python
topk by ego distance
```

改成：

```python
structured lane selection
```

---

## 推荐的 lane token 配额

16 个 token 可以这样分：

* 4 个：当前 lane chain
* 4 个：前向 lane chain
* 4 个：左右相邻 / merge / split lanes
* 4 个：最近补充 token

这样速度几乎不变，但信息保留会比纯距离法强很多。

如果你愿意稍微放宽一点，我建议直接试：

* `max_lane_tokens = 24`

但这 24 个必须是结构化采样，而不是单纯最近 24 个。
24 对比 16，速度损失通常不大，但精度可能会回来一大截。

---

# 三、FastDecoder 不能纯“一步直出”，要改成 Two-Stage Decoder

这大概率是第二大掉点来源。

原 DeMo 的强项之一就是：
**不是直接从 scene feature 一把回归完整未来轨迹，而是有 mode/state 的结构化解码。**

你现在的 `FastDecoder` 如果太轻，就会缺少：

* 模态分离能力
* 时间一致性
* 长时预测的渐进约束

---

## 推荐改法：Two-Stage Fast Decoder

不要恢复原重型 time decoder，而是做一个轻量的两阶段版。

### Stage 1：coarse trajectory proposal

快速生成 K 个粗轨迹：

```python
scene_feat -> K coarse anchors
scene_feat -> K logits
```

### Stage 2：residual refinement

对每个 coarse proposal 再做一个小 refinement：

```python
refined_traj = coarse_traj + delta_traj
```

其中 `delta_traj` 来自：

* ego feature
* global context
* selected lane context
* mode embedding

这样计算量仍远低于原 decoder，但表达能力会比“一把直出”强很多。

---

## 更具体一点

你可以把 60 步预测拆成：

* 先预测 6 个关键点（每 10 步一个）
* 再插值/上采样到 60 步
* 再用 residual head 微调

这会非常适合快模型，因为：

* 粗结构先确定
* 细节再修复
* 长时误差通常会明显下降

---

# 四、Temporal 压缩别固定 8，改成“自适应压缩 + 残差直通”

`compressed_steps=8` 太可能压狠了。
但也不一定要直接回到 50。

我建议改成：

## 方案：双支路 temporal encoder

### Branch A：压缩后的 SNN 主路径

保持快。

### Branch B：极轻的最近几帧直通残差

例如保留最近 4 帧或 6 帧的连续信息：

```python
recent_residual = MLP(last_4_frames_features)
```

最后融合：

```python
hist_feat = snn_temporal_feat + beta * recent_residual
```

这样做的原因很简单：
很多驾驶意图其实不需要完整 50 帧，但**最后几帧特别关键**。
你用 `50 -> 8` 压缩时，最容易损失的恰恰是最近细粒度动态。

这条支路便宜，但很可能能回收不少精度。

---

# 五、Multi-modal 能力必须明确补回来

如果你现在 `FastDecoder` 的多模态只是简单：

```python
scene_feat -> pi
scene_feat -> y_hat[K]
```

那很可能模态之间没有真正分开。
原 DeMo 的强点之一正是 mode/state 解耦。你可以不完全照搬，但要恢复一点“mode-specific feature”。

---

## 推荐做法

为每个 mode 引入独立 embedding：

```python
mode_embed[k]
traj_k = Decoder(scene_feat + mode_embed[k])
```

然后 refinement head 也是 mode-specific 的，或者至少 conditional。

不要用一个共享 MLP 只在最后一层分 K 路，这通常会导致：

* 各 mode 很像
* minADE/minFDE 非常差

---

# 六、训练上加“蒸馏”，这是最值得做的

你现在已经有一个很快但掉点严重的学生模型。
最有效的精度恢复方式，不是继续堆结构，而是：

## 用原 DeMo 做 teacher，蒸馏到 Fast 模型

这在你这个阶段非常合适，因为：

* teacher 已经有
* student 已经有
* I/O 完全兼容

---

## 蒸馏目标建议

### 1. logit distillation

蒸馏 `pi / new_pi`

### 2. trajectory distillation

蒸馏 `y_hat / new_y_hat`

### 3. feature distillation

蒸馏 scene-level hidden feature 或 decoder input

尤其推荐第 2 条，最直接。

加一个损失：

[
L = L_{\text{task}} + \lambda_1 L_{\text{KD-pi}} + \lambda_2 L_{\text{KD-traj}}
]

这个通常对轻量化 student 很有效，尤其在你允许“输入输出一致但内部大改”的条件下。

如果 teacher 训练成本太高，甚至可以先离线缓存 teacher 输出。

---

# 七、把 `spike_sparsity_loss` 降低甚至先关掉

你现在目标是：

* 先把精度救回来
* 再看 SNN 稀疏性是否值得追求

如果 `spike_sparsity_loss` 权重大，student 很可能：

* 本来表达能力就弱
* 还被迫稀疏
* 最终直接欠拟合

所以我建议当前阶段：

* `spike_reg_weight -> 很小`
* `mem_reg_weight -> 保留但小`
* 甚至先只保留 membrane stability，不强推 spike sparsity

先把模型训会，再谈 SNN 优势展示。

---

# 八、Loss 重加权，优先救核心指标

你说“性能下降超过 50%”，通常先看：

* minADE6
* minFDE6
* MR

对轻量模型，最容易崩的是 FDE 和 MR。
所以你现在可以在 loss 上偏向终点与模态：

## 推荐

加大：

* mode classification loss
* final-step / endpoint loss
* new head 的终点约束

轻量模型很容易轨迹整体趋势错，所以 endpoint supervision 很重要。

---

# 九、一个最现实、最可能成功的修正版

我建议你不要再做大推翻，而是做下面这个版本：

## `SNNModelForecastFastAcc`

结构：

### Temporal

* `TemporalConvSNNEncoder`
* * `RecentFrameResidual`

### Scene

* `EventSceneGraph`
* * `LightGlobalContext`

### Lane

* 结构化 lane selection
* token 数从 16 调到 24

### Decoder

* `TwoStageFastDecoder`

  * coarse proposals
  * residual refinement
  * mode embeddings

### Training

* KD from raw DeMo
* 降低 spike sparsity
* 强化 endpoint / mode loss

这版的特点是：

* 主体仍是快路径
* 计算量只比你当前版多一点
* 但会明显比当前版更“像 DeMo”

---

# 十、按优先级给你一个执行顺序

## 第一优先级：先做这 3 件

1. `FastDecoder -> TwoStageFastDecoder`
2. `lane selection -> structured lane selection`
3. `EventSceneGraph + LightGlobalContext`

这三件最可能直接把掉点从 50% 拉回到可接受区间。

## 第二优先级：再做这 2 件

4. `RecentFrameResidual` 加到 temporal
5. 降低/关闭 spike sparsity reg

## 第三优先级：最后做

6. teacher-student distillation
7. endpoint-aware loss reweight

---

# 我的判断

你现在这版并不是“加速思路错了”，而是：

> **为了速度把 DeMo 中最贵的东西都砍掉了，但同时也把它最值钱的建模偏置砍掉了。**

所以接下来不是回退到原模型，而是：

> **用极小代价把 mode、global context、lane structure、refinement 这几样能力补回来。**

这才最符合你当前目标。

如果你愿意，我下一步可以直接给你出一份 **`SNNModelForecastFastAcc` 的模块级改造清单**，包括每个新类的名字、输入输出张量形状、以及应该替换你当前哪些文件。

---

# 当前代码已落地的实线路径（2026-03-07）

已按本文建议将 SNN 实线路径调整为：

`TemporalConvSNNEncoder(+recent residual) -> EventSceneGraph + LightGlobalContext + HybridBlock(1-layer) -> TwoStageFastDecoder`

对应实现：

- `src/model/layers/temporal_conv_snn.py`
- `src/model/layers/event_scene_graph.py`
- `src/model/layers/light_global_context.py`
- `src/model/layers/two_stage_fast_decoder.py`
- `src/model/model_forecast_snn_fast.py`
- `conf/model/model_forecast_snn_fast.yaml`

已实现的关键点：

- Temporal recent residual：`recent_frames=6`、`recent_residual_weight=0.35`
- Dual-path scene：`EventSceneGraph + LightGlobalContext`
- Hybrid scene refine：默认 `hybrid_scene_depth=1`
- Structured lane selection：`near_ego + near_goal + diverse`
- Two-stage decoder：`coarse-to-fine` 多模态解码
- 默认参数改为保精度方向：`max_lane_tokens=24`、`compressed_steps=10`、降低 spike 稀疏正则权重
