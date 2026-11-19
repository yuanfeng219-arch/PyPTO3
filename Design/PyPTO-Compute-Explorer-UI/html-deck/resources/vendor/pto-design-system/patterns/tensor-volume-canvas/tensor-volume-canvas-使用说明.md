# Tensor Volume Canvas 使用说明

`tensor-volume-canvas` 用于绘制固定视角的三维 Tensor 体素图，适合 NCHW、Load3D、Conv、Tiling 和 Code Recovery 场景。

## 基本用法

### 1. 加载样式和脚本

```html
<link rel="stylesheet" href="pto-design-system/tokens/foundation.css">
<link rel="stylesheet" href="pto-design-system/tokens/semantic.css">
<link rel="stylesheet" href="pto-design-system/tokens/components.css">
<link rel="stylesheet" href="pto-design-system/patterns/tensor-volume-canvas/pattern.css">

<script src="pto-design-system/patterns/tensor-volume-canvas/pattern.js"></script>
```

### 2. 添加 Canvas

Canvas 的父容器必须有明确的宽度和高度。

```html
<div style="width: 640px; height: 420px;">
  <canvas id="tensorCanvas"></canvas>
</div>
```

### 3. 渲染场景

```js
const scene = {
  extent: {
    columns: 4,
    rows: 4,
    depth: 3,
  },
  axes: {
    columns: 'W',
    rows: 'H',
    depth: 'C',
  },
  voxels: [
    {
      id: 'voxel-0',
      column: 0,
      row: 0,
      depth: 0,
      tone: 'input',
      state: 'current',
      label: 'x000',
    },
  ],
};

const controller = window.PtoTensorVolumeCanvas.render(
  document.getElementById('tensorCanvas'),
  scene,
  {
    ariaLabel: '输入 Tensor 体素图',
    showAxes: true,
    autoLabelDensity: true,
  }
);
```

## Scene 配置

### `extent`

定义 Tensor 体素空间的大小。

| 字段 | 类型 | 说明 |
|---|---|---|
| `columns` | 正整数 | 横向体素数量 |
| `rows` | 正整数 | 纵向体素数量 |
| `depth` | 正整数 | 深度方向体素数量 |

### `axes`

定义三个方向的坐标轴名称。

| 字段 | 类型 | 默认值 |
|---|---|---|
| `columns` | string | `W` |
| `rows` | string | `H` |
| `depth` | string | `C` |

### `voxels`

定义需要绘制的体素。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `column` | number | 是 | 横向坐标，从 `0` 开始 |
| `row` | number | 是 | 纵向坐标，从 `0` 开始 |
| `depth` | number | 是 | 深度坐标，从 `0` 开始 |
| `id` | string | 否 | 体素标识 |
| `label` | string | 否 | 体素内文字 |
| `tone` | string | 否 | 语义颜色，默认 `neutral` |
| `state` | string | 否 | 显示状态，默认 `base` |

坐标超出 `extent` 范围时，该体素不会绘制。

## Tone 可选值

| 值 | 用途 |
|---|---|
| `neutral` | 普通体素 |
| `input` | 输入数据 |
| `output` | 输出数据 |
| `compute` | 计算数据 |
| `reduction` | 归约数据 |
| `fusion` | 融合数据 |

`neutral` 和 `padding` 使用 API Visualizer Load3D 固定色板；其他 Tone 使用 PTO 语义 Token。没有选中、播放、当前处理或其他语义倾向时，体素默认使用 `neutral`，不要仅根据“输入 / 输出对象身份”自动上色。不支持业务页面传入自定义颜色。

## State 可选值

| 值 | 用途 |
|---|---|
| `base` | 默认状态 |
| `ghost` | 弱化显示 |
| `padding` | H/W 越界区域或 C 维尾部补齐 |
| `window` | 当前窗口范围 |
| `current` | 当前处理体素 |
| `skipped` | 跳过的体素 |

## Options 可配置项

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `ariaLabel` | string | `Three-dimensional tensor volume` | Canvas 的无障碍说明 |
| `padding.top` | number | `44` | 顶部留白 |
| `padding.right` | number | `38` | 右侧留白 |
| `padding.bottom` | number | `42` | 底部留白 |
| `padding.left` | number | `52` | 左侧留白 |
| `showAxes` | boolean | `true` | 是否显示坐标轴 |
| `autoLabelDensity` | boolean | `true` | 空间不足时是否自动隐藏普通标签 |

`current` 状态的标签不会被自动隐藏。

## 更新和销毁

```js
// 更新场景
controller.update(nextScene);

// 同时更新场景和配置
controller.update(nextScene, {
  showAxes: false,
});

// 容器尺寸变化后手动重绘
controller.resize();

// 页面卸载或不再使用时销毁
controller.destroy();
```

## 使用限制

- 不要复制或修改 Pattern 内部的投影、体素几何、遮挡排序和 DPR 逻辑。
- 不支持旋转、拖拽、平移、滚轮缩放或自定义相机。
- 不要通过 iframe 使用，应直接加载 `pattern.css` 和 `pattern.js`。
- 业务选择、播放、Load3D、Conv 和 Tiling 规则应在业务页面中转换为 `tone` 和 `state`。
- Pattern 只负责绘制，不包含工具栏、Inspector、卡片或播放控件。
