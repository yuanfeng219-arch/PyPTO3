# Matrix Canvas 使用说明

`matrix-canvas` 用于绘制二维 Matrix Tensor，适合展示逐元素数据、行列关系、Load3D A2 逻辑矩阵、写入过程、Padding 来源以及超大矩阵的缩略表达。

矩阵整体保留圆角裁切，内部所有明细格和缩略格共用连续的直角网格线。格子本体没有独立间距、圆角或卡片式边框；聚合格只额外显示一个居中的圆角摘要标记。

它和 `tensor-volume-canvas` 同属 `tensor-visualization` Pattern 家族：

- 二维行列关系使用 `matrix-canvas`。
- 固定视角的三维体素结构使用 `tensor-volume-canvas`。

## 基本用法

### 1. 加载样式和脚本

```html
<link rel="stylesheet" href="pto-design-system/tokens/foundation.css">
<link rel="stylesheet" href="pto-design-system/tokens/semantic.css">
<link rel="stylesheet" href="pto-design-system/tokens/components.css">
<link rel="stylesheet" href="pto-design-system/patterns/matrix-canvas/pattern.css">

<script src="pto-design-system/patterns/matrix-canvas/pattern.js"></script>
```

### 2. 添加 Canvas

Canvas 的父容器必须有明确的宽度和高度。

```html
<div style="width: 720px; height: 480px;">
  <canvas id="matrixCanvas"></canvas>
</div>
```

### 3. 渲染普通矩阵

```js
const scene = {
  extent: {
    rows: 8,
    columns: 12,
  },
  axes: {
    rows: 'M · row',
    columns: 'N · column',
  },
  cells: [
    {
      id: 'cell-2-7',
      row: 2,
      column: 7,
      label: '0.82',
      value: 0.82,
      tone: 'input',
      style: 'value',
      states: ['row-focus', 'column-focus', 'current'],
    },
  ],
};

const controller = window.PtoMatrixCanvas.render(
  document.getElementById('matrixCanvas'),
  scene,
  {
    ariaLabel: '8 行 12 列输入矩阵',
    showAxes: true,
    showGrid: true,
  }
);
```

## Scene 配置

### `extent`

定义源矩阵的真实大小。即使使用缩略格，也必须保留源矩阵的行列数，不能改成缩略网格的大小。

| 字段 | 类型 | 说明 |
|---|---|---|
| `rows` | 正整数 | 源矩阵行数 |
| `columns` | 正整数 | 源矩阵列数 |

### `axes`

定义两个方向的坐标轴名称。

| 字段 | 类型 | 默认值 |
|---|---|---|
| `rows` | string | `row` |
| `columns` | string | `column` |

### `cells`

定义需要绘制的矩阵格。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `row` | number | 是 | 起始行坐标，从 `0` 开始 |
| `column` | number | 是 | 起始列坐标，从 `0` 开始 |
| `id` | string | 否 | 格子标识 |
| `label` | string | 否 | 格子内文字 |
| `value` | number | 否 | 原始数值，用于 tooltip |
| `rowSpan` | 正整数 | 否 | 当前格覆盖的源矩阵行数，默认 `1` |
| `columnSpan` | 正整数 | 否 | 当前格覆盖的源矩阵列数，默认 `1` |
| `tone` | string | 否 | 语义颜色，默认 `neutral` |
| `style` | string | 否 | 格子样式，默认 `value` |
| `states` | string[] | 否 | 可叠加的显示状态 |
| `summary` | object | 聚合格需要 | 聚合范围和统计摘要 |

坐标超出 `extent` 范围时，该格不会绘制。末端格子的 `rowSpan` 和 `columnSpan` 会自动裁剪到源矩阵边界。

## Tone 可选值

| 值 | 用途 |
|---|---|
| `neutral` | 普通矩阵数据 |
| `input` | 输入数据 |
| `output` | 输出数据 |
| `compute` | 计算数据 |
| `reduction` | 归约数据 |
| `fusion` | 融合数据 |

Tone 只表达明确的数据语义或交互强调。没有选中、播放、当前处理或其他语义倾向时，普通格和 `createAggregatedCells` 生成的聚合格都默认使用 `neutral`。业务页面不应仅根据“输入 / 输出对象身份”自动上色，也不应传入自定义颜色或覆盖 Pattern 的共享色板。

## Style 可选值

| 值 | 用途 |
|---|---|
| `value` | 普通有值格 |
| `empty` | 空格或弱内容格 |
| `padding` | Padding 来源格，使用斜线纹理 |
| `broadcast` | 广播生成格；在底层 Tone 上叠加半透明灰色蒙版，边界沿用普通连续网格 |
| `aggregate` | 一个格子代表一片源数据的缩略格 |

### 广播矩阵

`broadcast` 用于区分真实存储的源数据与通过广播语义生成的数据。例如 Bias 从 `[1,16]` 广播为 `[16,16]` 时，第一行保留 `value`，后 15 行标记为 `broadcast`。

```js
const sourceColumns = 16;
const targetRows = 16;
const cells = [];

for (let row = 0; row < targetRows; row += 1) {
  for (let column = 0; column < sourceColumns; column += 1) {
    cells.push({
      row,
      column,
      value: bias[column],
      style: row === 0 ? 'value' : 'broadcast',
    });
  }
}

const scene = {
  extent: { rows: targetRows, columns: sourceColumns },
  axes: { rows: 'broadcast rows', columns: 'bias columns' },
  cells,
};
```

`broadcast` 只表达数据来源，不代表弱化或 Padding。灰色蒙版覆盖在底层 Tone 之上，因此不会丢失输入、输出等语义；它也可以与 `current`、`selected` 等 States 叠加。`current` 保持最高视觉优先级，不叠加灰色蒙版。

## States 可选值

多个状态可以通过 `states` 数组叠加。状态存在覆盖顺序，`current` 的视觉优先级最高。

| 值 | 用途 |
|---|---|
| `row-focus` | 当前行 |
| `column-focus` | 当前列 |
| `written` | 已写入 |
| `selected` | 已选择；聚合格使用完整蓝色填充 |
| `current` | 当前处理格，使用警示色和强化网格边界，不添加内部描边 |
| `muted` | 弱化显示 |

点击格子不会由 Pattern 自动写入 `selected`。Pattern 通过 `onSelect` 把点击结果交给业务，业务更新 Scene 后才会显示选中态。

```js
let currentScene = scene;

const controller = window.PtoMatrixCanvas.render(
  document.getElementById('matrixCanvas'),
  currentScene,
  {
    onSelect(cell) {
      if (!cell) return;

      currentScene = {
        ...currentScene,
        cells: currentScene.cells.map((item) => ({
          ...item,
          states: item.id === cell.id
            ? [...new Set([...(item.states || []), 'selected'])]
            : (item.states || []).filter((state) => state !== 'selected'),
        })),
      };

      controller.update(currentScene);
    },
  }
);
```

## 超大矩阵缩略表达

超大矩阵不需要生成每一个明细格。使用 `createAggregatedCells` 可以把源数据转换为任意数量的缩略格。

### 按缩略网格数量生成

业务注入希望看到的缩略行列数，Pattern 自动计算每格覆盖的源数据范围。

```js
const sourceRows = 240;
const sourceColumns = 420;
const values = new Float32Array(sourceRows * sourceColumns);

const aggregateCells = window.PtoMatrixCanvas.createAggregatedCells(
  {
    values,
    rows: sourceRows,
    columns: sourceColumns,
  },
  {
    thumbnailRows: 12,
    thumbnailColumns: 20,
    tone: 'neutral',
  }
);

const aggregateScene = {
  extent: {
    rows: sourceRows,
    columns: sourceColumns,
  },
  axes: {
    rows: 'source rows',
    columns: 'source columns',
  },
  cells: aggregateCells,
};

const controller = window.PtoMatrixCanvas.render(
  document.getElementById('matrixCanvas'),
  aggregateScene,
  {
    ariaLabel: '240 行 420 列矩阵的缩略视图',
  }
);
```

这里的源矩阵仍是 `240 × 420`，`12 × 20` 只是目标缩略格数量。由于边界取整，最终格数可能少于目标值，但不会超出源矩阵范围。

### 按每格覆盖范围生成

如果业务更关心一个缩略格代表多少源数据，可以直接注入固定跨度。

```js
const aggregateCells = window.PtoMatrixCanvas.createAggregatedCells(
  {
    values,
    rows: sourceRows,
    columns: sourceColumns,
  },
  {
    blockRows: 8,
    blockColumns: 32,
    tone: 'neutral',
  }
);
```

每个缩略格的实际跨度会记录在自身的 `rowSpan`、`columnSpan` 和 `summary` 中，尾部格可能小于指定跨度。

### 使用惰性数据读取

如果源数据不适合放入一维数组，可以提供 `valueAt(row, column)`。

```js
const aggregateCells = window.PtoMatrixCanvas.createAggregatedCells(
  {
    rows: sourceRows,
    columns: sourceColumns,
    valueAt(row, column) {
      return getMatrixValue(row, column);
    },
  },
  {
    thumbnailRows: 16,
    thumbnailColumns: 24,
  }
);
```

### `summary`

聚合格沿用普通矩阵的连续直角网格，并通过一个居中的圆角标记表达聚合强度，不在格子内显示文字。完整范围和统计值通过 hover tooltip 查看。

| 字段 | 类型 | 说明 |
|---|---|---|
| `rows` | 正整数 | 当前格代表的源数据行数 |
| `columns` | 正整数 | 当前格代表的源数据列数 |
| `count` | 正整数 | 参与统计的有效值数量 |
| `min` | number | 最小值 |
| `max` | number | 最大值 |
| `mean` | number | 平均值 |
| `intensity` | 0–1 | 中央标记的强度 |

`createAggregatedCells` 会自动生成这些字段，并根据全部聚合格的 `mean` 归一化计算 `intensity`。业务也可以自行构造 `style: 'aggregate'` 的格子并注入这些字段。

## Options 可配置项

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ariaLabel` | string | `Two-dimensional matrix tensor` | Canvas 的无障碍说明 |
| `showAxes` | boolean | `true` | 是否显示行列坐标轴 |
| `showGrid` | boolean | `true` | 是否在可读尺度显示逻辑网格 |
| `interactive` | boolean | `true` | 是否启用 hover、点击、滚动和键盘交互 |
| `showTooltip` | boolean | `true` | 是否显示 hover tooltip |
| `autoFit` | boolean | `true` | 初始和未改动视图时是否自动适配 |
| `minZoom` | number | `0.015` | 最小缩放比例 |
| `maxZoom` | number | `12` | 最大缩放比例 |
| `scrollSpeed` | number | `1` | 横向/纵向滚动速度，内部限制为 `0.1–4` |
| `padding.top` | number | `38` | 顶部留白 |
| `padding.right` | number | `28` | 右侧留白 |
| `padding.bottom` | number | `42` | 底部留白 |
| `padding.left` | number | `52` | 左侧留白 |
| `onHover(cell, event)` | function | 无 | hover 格变化时回调 |
| `onSelect(cell, event)` | function | 无 | 点击格子时回调 |

## 交互

- 滚轮：在矩阵范围内纵向滚动，到达上下边界后停止。
- 触控板横向手势或 `Shift + 滚轮`：在矩阵范围内横向滚动，到达左右边界后停止。
- 矩阵小于视窗时保持居中，滚轮事件继续交给外层页面。
- `Ctrl/Command + 滚轮`：以指针位置为中心缩放。
- `F` 或 `0`：适配完整矩阵。
- `+` 或 `=`：放大。
- `-` 或 `_`：缩小。
- 鼠标悬停：显示格子范围、数值或聚合统计。
- 鼠标点击：通过 `onSelect` 上报格子。

Pattern 不支持拖拽平移，也不会把矩阵滚出边界形成无限画布。

## 更新、视图控制和销毁

```js
// 更新场景
controller.update(nextScene);

// 同时更新场景和配置
controller.update(nextScene, {
  showAxes: false,
});

// 容器尺寸变化后手动重绘
controller.resize();

// 适配完整矩阵
controller.fit();

// 以画布中心缩放
controller.setZoom(2);

// 以指定画布坐标为锚点缩放
controller.setZoom(2, { x: 320, y: 220 });

// 读取当前视图
const viewState = controller.getViewState();

// 重置视图，当前等价于 fit()
controller.resetView();

// 页面卸载或不再使用时销毁
controller.destroy();
```

调用 `update` 时，如果源矩阵尺寸发生变化，默认会重新适配。传入 `{ preserveView: true }` 可以保留当前视图。

## 文字缩放规则

- 普通格最短边小于 `40px` 时隐藏文字。
- `current` 格最短边达到 `28px` 后可以显示文字。
- 可见文字使用等宽字体，字号限制在 `11–14px`。
- 聚合格在任何缩放级别都不显示文字，详情统一使用 tooltip。

## 使用限制

- 不要复制或修改 Pattern 内部的矩阵几何、圆角裁切、DPR、命中测试、滚轮浏览、缩放、文字 LOD、tooltip 或 `ResizeObserver` 逻辑。
- 不要把缩略网格的尺寸写入 `scene.extent`；`extent` 始终表达真实源矩阵。
- 不要在业务页面覆盖共享格子色板或用页面私有颜色替代 PTO 语义 Token。
- 不要通过 iframe 使用，应直接加载 `pattern.css` 和 `pattern.js`。
- Load3D、Conv、Tiling、播放和选择规则应由业务页面转换成 `tone`、`style` 和 `states`。
- Pattern 只负责绘制，不包含工具栏、Inspector、卡片或播放控件。
