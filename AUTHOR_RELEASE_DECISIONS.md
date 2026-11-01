# FocalGS 发布决定记录

更新日期：2026-11-01

本文件记录已执行的公开发布边界。论文作者信息仅用于论文引用，不用于推断
或认定专利发明人。

## 真实数据

公开真实数据仅保留 **Post-paper Strict 24--85 mm Benchmark v3**，并按场景
分别发布为 `real/toy/toy.tar.gz`、`real/plant/plant.tar.gz` 和
`real/bear/bear.tar.gz`：

- Toy：从历史 Toy split 中保留 24、35、50、70、85 mm，实际 60/15；
- Plant：后续 clean strict split，实际 58/14；
- Bear：后续 clean strict split，实际 57/13。

Toy 不是后续 clean 采集，公开元数据必须持续说明这一点。该统一数据包也
不是论文原始实验数据，不得用于掩盖论文文字协议与历史实验归档之间的差异。
历史 Experimental Archive 不再作为公开下载保留。

## 合成数据

七个冻结渲染数据包已公开。公开授权不转移上游 Blender 资产的所有权，也
不以代码仓库许可证替代资产自身条款。逐场景来源证据继续保存在
`docs/SYNTHETIC_DATA_PROVENANCE.md`。

## 真实照片与隐私

公开包仅含处理后的 PNG、训练/测试 JSON 和必要的 COLMAP sparse 模型。
CR3、序列号、时间戳、原始文件名和无关中间文件均不公开。

## 代码与数据许可

代码根许可证和 vendored 上游组件分别适用；数据与 Blender 资产不自动
继承代码许可证。具体边界见 `LICENSE`、`THIRD_PARTY.md` 和数据来源说明。
