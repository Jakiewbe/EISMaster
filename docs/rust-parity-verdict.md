# Python 与 Rust/Tauri 实现对照结论

## 当前结论

结论为：**暂停替换，保留 Python/PySide6 作为正式主线，Rust/Tauri 作为对照实现**。

Rust 工作区已经具备可运行的核心能力，但本轮没有完成 Python 与 Rust 在同一组输入上的逐项数值对照，也没有完成 Tauri 桌面应用的人工交互验收。因此不能把 Rust/Tauri 标记为 Python 主线的等价替代。

## 已确认事实

| 项目 | Python/PySide6 | Rust/Tauri | 结论 |
| --- | --- | --- | --- |
| 核心单元测试 | `pytest`：80 个通过，0 个失败 | `cargo test --workspace`：79 个通过，0 个失败 | 两侧各自通过 |
| 代码格式 | 全量 Ruff：118 个历史问题，未宣称通过 | `cargo fmt --check` 通过 | Rust 通过；Python 仍需后续清理 |
| CLI | 不适用 | `eismaster-cli --help` 正常返回帮助文本 | Rust CLI 通过 |
| 前端构建 | 不适用 | `npm run build` 通过 | Tauri 前端通过 |
| TXT/BIN 解析数值一致性 | 未做同输入对照 | Rust 自有测试通过 | 未验证跨实现一致性 |
| 质量评估与分段 | 未做同输入对照 | Rust 自有测试通过 | 未验证跨实现一致性 |
| 单弧/双弧拟合 | 未做同输入对照 | Rust 自有测试通过 | 未验证跨实现一致性 |
| 批处理与 Excel 导出 | Python 有现有测试；最终汇总待完成 | Rust 自有测试通过 | 未验证跨实现一致性 |
| MATLAB DRT | Python 已做路径与失败返回校验 | Rust 仅保留相关接口/测试 | 未验证等价性 |
| 打包 | Python PyInstaller 构建通过，资源检查 3/3 | Tauri 桌面包未做完整发布验收 | Python 当前更可靠 |

## 已知警告

Rust 测试通过，但仍有两个编译警告：核心拟合模块存在未使用导入，Tauri 命令模块存在不必要的 `mut`。它们不影响本轮测试结果，但应在后续 Rust 清理任务中处理。

## 后续替换条件

只有在同一批确定性 TXT/BIN fixture 上完成解析、质量评估、分段、单弧拟合、双弧拟合、批处理和导出结果的逐项对照，并完成 Tauri 应用启动与主要流程验收后，才重新评估是否切换正式主线。
