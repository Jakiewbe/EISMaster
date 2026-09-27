# 拟合结果导出 Wo 参数 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `Wo_R`、`Wo_T`、`Wo_P` 加入单次和批量拟合的 `rs_rct` 导出表。

**Architecture:** 继续使用现有 `build_rs_rct_table()` 作为单次和批量导出的唯一汇总表构造入口。只扩展固定列顺序和参数读取，不修改拟合结果模型、文件写入层或 UI 调用链。

**Tech Stack:** Python 3.11、pandas、openpyxl、pytest/unittest。

## Global Constraints

- 不创建新分支或 worktree。
- 不修改拟合算法、参数命名、工作簿表名和导出文件名。
- 缺失 Wo 参数写为空值，不推算、不静默替代。
- 先写测试并确认测试失败，再写生产代码。

---

### Task 1: 固化 Wo 汇总导出契约

**Files:**
- Modify: `tests/test_exports.py`

**Interfaces:**
- Consumes: 现有 `build_rs_rct_table()`、`export_spectrum_bundle()`、`export_batch_summary()`。
- Produces: 对 `rs_rct` 列顺序和 Wo 数值的回归约束。

- [ ] **Step 1: 写失败测试**

在 `tests/test_exports.py` 中将 `rs_rct` 的期望列更新为 `label\tfile\tRs\tRct\tRsei\tWo_R\tWo_T\tWo_P`，并让拟合样例包含：

```python
parameters={"Rs": 5.8, "Rct": 120.0, "Wo_R": 15.0, "Wo_T": 2.5, "Wo_P": 0.5}
```

增加断言，确认单次和批量 TXT 的数据行末尾包含 `15.0\t2.5\t0.5`，并在 XLSX `rs_rct` 工作表确认同样的列顺序和值。

- [ ] **Step 2: 运行测试确认失败**

运行：

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_exports.py -q
```

预期：现有实现生成的 `rs_rct` 表只有 `Rs/Rct/Rsei`，新增 Wo 列相关断言失败。

### Task 2: 实现共用 Wo 字段导出

**Files:**
- Modify: `src/eismaster/exporters.py:182-194`

**Interfaces:**
- Consumes: `FitOutcome.parameters` 中的 `Rs`、`Rct`、`Rsei`、`Wo_R`、`Wo_T`、`Wo_P`。
- Produces: `build_rs_rct_table()` 返回固定列顺序 `label,file,Rs,Rct,Rsei,Wo_R,Wo_T,Wo_P` 的 `DataFrame`。

- [ ] **Step 1: 最小实现**

在现有行字典中追加：

```python
"Wo_R": np.nan if fit is None else fit.parameters.get("Wo_R", np.nan),
"Wo_T": np.nan if fit is None else fit.parameters.get("Wo_T", np.nan),
"Wo_P": np.nan if fit is None else fit.parameters.get("Wo_P", np.nan),
```

并保持字典插入顺序，不引入额外字段或 fallback。

- [ ] **Step 2: 运行目标测试确认通过**

运行：

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_exports.py -q
```

预期：导出测试全部通过。

### Task 3: 完成验证和第一性原理复核

**Files:**
- Inspect: `src/eismaster/exporters.py`
- Inspect: `tests/test_exports.py`

- [ ] **Step 1: 运行完整测试集**

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest -q
```

- [ ] **Step 2: 检查差异**

```powershell
git diff -- src/eismaster/exporters.py tests/test_exports.py
```

确认只有 Wo 三个字段和对应测试发生变化，且现有未提交文件未被带入。

- [ ] **Step 3: Review 查 Bug 和第一性原理复核**

确认单次和批量都经过同一个构造函数；没有重复实现、没有改变 `fit_report`、没有对缺失 Wo 参数做错误推算。

