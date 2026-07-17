# 拟合结果导出 Wo 参数设计

## 目标

在不改变现有导出文件结构和拟合计算的前提下，将有限长度 Warburg 元件的 `Wo_R`、`Wo_T`、`Wo_P` 加入单次拟合和批量拟合的 `rs_rct` 汇总导出。

## 现状

- `FitOutcome.parameters` 已由拟合流程保存 `Rs`、`Rct`、`Wo_R`、`Wo_T`、`Wo_P`。
- 单次拟合和批量拟合都调用 `build_rs_rct_table()`。
- 单次 `fit_report` 已导出全部拟合参数，因此本次不重复扩展该报告。
- 不包含 Warburg 元件的模型没有 Wo 参数，导出对应单元格为空值。

## 导出字段

`rs_rct` 表固定使用以下列顺序：

```text
label, file, Rs, Rct, Rsei, Wo_R, Wo_T, Wo_P
```

其中 `Wo_R` 为 Warburg 电阻、`Wo_T` 为特征时间、`Wo_P` 为指数参数。字段名沿用拟合结果内部名称，避免单次报告、批量汇总和界面显示出现不同命名。

## 数据流

1. `export_spectrum_bundle()` 将单个 `(SpectrumData, FitOutcome)` 传给 `build_rs_rct_table()`。
2. `export_batch_summary()` 将每个批量结果的 `(SpectrumData, FitOutcome)` 传给同一个函数。
3. `build_rs_rct_table()` 按固定字段顺序读取 `fit.parameters`；缺失拟合结果或缺失参数时写入 `NaN`。
4. 现有 TXT、CSV、XLSX 写入逻辑不变，因此三种格式自然获得相同字段。

## 错误和兼容性

- 不修改拟合算法、参数命名、表名和文件名。
- 不把缺失 Wo 参数推算成其他值；缺失时保持空值，避免误导后续分析。
- 不增加新的降级逻辑或吞掉导出异常。

## 测试验收

- `build_rs_rct_table()` 的列顺序包含三个 Wo 字段，并能读取对应拟合值。
- 单次 TXT 导出的 `rs_rct` 首行包含三个 Wo 字段。
- 批量 TXT 导出的 `rs_rct` 首行包含三个 Wo 字段。
- XLSX 的 `rs_rct` 工作表包含相同列顺序和对应数值。
- 原有导出测试和完整测试集通过。

