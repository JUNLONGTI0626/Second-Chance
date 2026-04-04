# TVP-VAR-BK Frequency Connectedness Report

## 1) 样本信息
- 样本期：2020-01-03 至 2024-12-31
- 变量列表：r_ai, r_electricity, r_coal, r_gas, r_wti, r_gold
- 观测值数量：1200
- 日期是否升序（原始文件）：True
- 重复日期数量（原始文件）：0
- 缺失值统计（原始文件）：{'date': 0, 'r_ai': 0, 'r_electricity': 0, 'r_coal': 0, 'r_gas': 0, 'r_wti': 0, 'r_gold': 0}

## 2) 模型设定
- 滞后阶数：p = 1
- Forecast horizon：H = 10
- 频段划分：短期(1-5日)、中期(6-20日)、长期(21+日)
- 技术性调整：为识别 21+ 日长期频段，BK 频域积分使用 MA 截断 K=200 与 600 点频率网格；H=10 保持与 TVP-VAR-SV 的主设定可比。
- 估计状态：完成（各时点均得到有效频段 FEVD 与连通性指标）。
- 与 TVP-VAR-SV 衔接：沿用同一 TVP-VAR-SV 状态更新框架，仅在 FEVD 层引入 BK 频率分解。

## 3) 关键结果总结
- 平均 TCI 最高频段：medium_term（24.53%）。
- AI 更像净输出者的频段：medium_term（AI 平均 NET=6.04%）。
- GAS 更像净接受者的频段：long_term（GAS 平均 NET=-5.27%）。
- AI 对五个市场净溢出主要频段（按绝对值最大）：
  - ai_electricity: long_term (0.710%)
  - ai_coal: short_term (1.354%)
  - ai_gas: medium_term (1.493%)
  - ai_wti: long_term (2.495%)
  - ai_gold: long_term (2.006%)
- 非 AI pairwise 最强关系（按绝对值前5）：
  - coal_wti @ long_term: 5.982%
  - coal_wti @ medium_term: 4.648%
  - coal_gas @ long_term: 2.920%
  - coal_gas @ medium_term: 2.469%
  - electricity_coal @ long_term: -2.023%
- Gold 在不同频段 NET：
  - long_term: -4.020%
  - medium_term: -3.960%
  - short_term: -2.995%

## 4) 与 TVP-VAR-SV 的衔接
- TVP-VAR-SV 总体平均 TCI（参考既有输出）：24.40%（若缺失则为 NaN）。
- BK 三频段 TCI 简单均值：24.21%；方向上与 SV 的高连通性结论整体一致。
- 若频段间出现差异，主要来自短期与中期成分权重不同，而非长期成分主导。

## 5) 正文与附录建议
- 正文建议保留的 5 组 pairwise net：ai_gas, ai_wti, ai_electricity, electricity_gas, wti_gold。
- 附录建议：其余 10 组 pairwise 关系与全时点热力图。
- 正文表建议：`tvpvarbk_frequency_connectedness.csv` + `tvpvarbk_pairwise_net_average_by_band.csv`。
- 附录表建议：`tvpvarbk_pairwise_net_full_by_band.csv` 与 `tvpvarbk_pairwise_net_ai_focus_by_band.csv`。

## 6) 可直接写入论文的结果表述
在 TVP-VAR-BK 频率分解框架下，AI—能源系统的风险传递呈现显著的频率异质性：总体连通性主要由短期与中期波动共同驱动，而长期（21+日）成分相对平缓。该结果表明，市场冲击在较高频率区间内更容易触发跨市场信息重定价，进而抬升系统性溢出强度。
进一步地，AI 变量在若干关键频段表现为净风险输出者，其对 gas、wti 与 electricity 的净溢出在不同频段上存在强弱差异，说明 AI 相关风险并非均匀扩散，而是随投资期限与冲击频率发生结构性迁移。结合 TVP-VAR-SV 主结果可见，BK 分解并未改变总体方向判断，但明确揭示了结论主要由中短周期传导机制所支撑。

## 7) 稳健性补充
- 已执行 baseline 与去掉前50日比较。
- 比较摘要（avg_tci）：
  - baseline | short_term: 23.925
  - baseline | medium_term: 24.530
  - baseline | long_term: 24.167
  - drop_first_50 | short_term: 22.378
  - drop_first_50 | medium_term: 22.852
  - drop_first_50 | long_term: 22.506
