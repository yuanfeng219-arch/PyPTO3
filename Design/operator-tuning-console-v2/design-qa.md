# V2 Design QA

## Scope

Investigation 工作台默认页：任务队列、任务概览、假设、实验、证据，以及跳转到原控制台的证据视图。

## Static checks

- v2.js syntax check passed.
- V2 entry references the shared original data and workbench runtime.
- Diff whitespace check passed.

## Visual capture

final result: blocked

当前执行环境禁止打开本地 HTML 和 localhost，因此无法捕获并检查实际渲染结果、窄屏布局和交互状态。这里没有将静态检查替代视觉验收。需要在允许本地浏览器访问的环境中，至少验证：

1. 默认 Investigation 页的三栏布局；
2. 切换任务与概览 / 假设 / 实验 / 证据；
3. 从证据行跳转到原五层工作台；
4. 从原层级页签返回 Investigations；
5. case 切换后的任务重建。
