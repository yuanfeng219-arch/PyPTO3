# PyPTO Deck 视觉一致性审计

入口：[visual-audit.html](visual-audit.html)。可直接用本地浏览器打开，不依赖服务或网络。

- `visual-tokens.json`：建议的视觉角色、现有 Token 映射与保留例外，尚未应用。
- `visual-audit-inventory.json`：14 项待审阅问题、14 项保留记录；每项含属性、来源、选择器、覆盖判断、建议和影响范围。
- `screenshots/`：13 页当前状态及模式、Task、层级、hover/focus 截图。报告区域标记通过 HTML 叠加，原始截图未加标记。
- `evidence/`：逐页 computed styles、匹配规则、来源清单、状态覆盖、自动检测原始结果与处理说明、交付核验。

## 证据说明

在 1600×900 Chrome 视口下检查当前 Deck，沿用其现有 fit 缩放和页脚。computed 为 CSS 布局值，标记框坐标为屏幕像素。CSSOM 规则保留作用域，原始声明可能存在简写/变量展开为空的情况；请结合 `actualCss` 与问题附带源码选择器理解，不把匹配规则顺序等同于层叠获胜顺序。

P13 为 PNG 预览，不可读取图内 UI 的 computed styles。P11 外部字体表受跨域 CSSOM 限制，已记录。隐藏 Inspector 没有被强行显示；本轮没有把不可见源规则算作可见浮层缺陷。动画截图记录特定相位，不是全时序评测。

此阶段未修改 Deck、既有 CSS/JS、Demo 源文件或设计系统 Token，未新增或运行单元测试。等待用户确认后再实施。后续先确定 Pipeline 共享来源，再确认嵌入适配层，最后讨论近似灰值和控件细节；不要全局查找替换业务图形。
