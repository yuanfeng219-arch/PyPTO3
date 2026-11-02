# 作者意见修复记录 · 2026-11-02

范围：独立 UI 优化版；原版仅作术语与交互基准，不修改。保留开工前已有改动。本轮不提交、不发布、不回复或关闭 GitHub Issue。

| Issue | 修复 |
|---|---|
| #159 | 128 个 Task 恢复编号及每项 5 个 Q Head 圆点，保留 Stack 表达。 |
| #160 | 同步路径和依赖路径显式无填充，消除黑色闭合面；同排状态边经独立上方通道进入目标，显示状态关系标签；紧凑布局同样预留分组间隔，避免 Output 与 Attention 框相交。 |
| #161 | Softmax → PV 按 Slot 分配错开的水平通道；移出原控制器容器的返回按钮由父级稳定事件处理。 |
| #162 | PA 计算模块独立于 Task Generation / Block Assignment 与范围切换；恢复 CSA 计算模块菜单，保留原版建设中阶段及其占位。 |
| #163 | 每个 Stack 只对应一个形变容器并进入同身份 QK 节点；原内容随容器过渡，形变代理不在末段淡出，目标显示后再移除，取消与减少动态效果策略保留。 |
| #164 | 模块、路径、对象信息与参数分层；共用透明悬浮布局和全局按钮锚点，避免 Settings 落入主 Tab 后方；菜单使用 SVG 箭头及低对比填充状态。 |

术语：停止运行时翻译原版 UI；恢复 Assignment / Pipeline / Graph、Phase0 / Attention+Output、Task Generation / Block Assignment、Generation / Ordering / Assignment 以及完整场景选项。领域模型和源码术语不改写。

## 验证

- 浏览器检查 PA 的 128 个 Task 均有正确编号及 5 个圆点；模块与视图控制位于独立容器。
- PA Task 0 的 Assignment Block 内 → Pipeline：3 个 Stack 对应 3 个代理，保持 opacity 1，包含源内容；结束后代理归零且无遗留隐藏 OP。返回 Block 间已实际点击成功。
- CSA 模块菜单可展开，选择 Q Proj 显示原版建设中占位，再切回 qk_pv 正常；Pipeline Work Item 0 可进入。
- PA 展开 Softmax 后检查 SVG 非 marker 路径：无填充异常；三个根分组间隔为 32 模型单位，无重叠。
- Node 回归覆盖紧凑分组不重叠、同排状态依赖绕过中间对象并垂直进入目标；变更 JS 语法检查通过。
- 浏览器检查使用 1280×720 的 PA 视口及 1125×1017 的 CSA 视口。未完成所有展开组合与窄屏全量回归，未做设备性能或线上验证。
