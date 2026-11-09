/**
 * 生成《PyPTO 2027 UX 规划》汇报 PPT（16:9，18 页 = 封面 + 16 页正文 + 收束）。
 *
 * 内容严格取自 Product_Planning/PyPTO_2027_UX规划_五看三定.md，外部数据均为该文
 * 第七节列出的来源，一手／二手标注与原文一致。
 *
 * 设计令牌（PRIMARY / ACCENT / LIGHT / BAND / GREY / RED / GREEN / AMBER、
 * Microsoft YaHei、1.02in 深蓝标题栏 + 底部洞察条）与仓库既有 PPT 生成脚本
 * Competitive_Analysis/generate_pptx.py、
 * Product_Planning/PyPTO3.0_体验设计项目材料包/generate_hn_pptx_v2.py 保持一致，
 * 以维持仓库内 PPT 视觉统一。
 *
 * 为什么是 JS 而不是 python-pptx：撰写本稿的机器上未安装 Python（python 为
 * Windows Store 占位程序），仓库既有两个 python-pptx 生成脚本无法运行，故改用
 * pptxgenjs。如需与既有脚本统一到 python-pptx，可按本文件的页面结构重写。
 *
 * 用法：
 *     npm install pptxgenjs@3.12.0
 *     node Product_Planning/generate_ux2027_pptx.js Product_Planning
 * 省略输出目录参数时写入当前工作目录。
 */
const pptxgen = require("pptxgenjs");
const path = require("path");

// ---------- 设计令牌（与仓库既有 PPT 一致） ----------
const PRIMARY = "1F3864"; // 深蓝
const ACCENT = "2E75B6"; // 蓝
const LIGHT = "F2F4F8"; // 浅灰底
const BAND = "E8EEF6"; // 表头浅蓝
const GREY = "595959";
const WHITE = "FFFFFF";
const RED = "C00000";
const GREEN = "2E7D32";
const AMBER = "B76E00";
const SKY = "BDD3EE";
const DIM = "9DB8D8";
const PALE = "D6E2F1";
const FONT = "Microsoft YaHei";

const DATE = "2026-11-09";
const FOOTER =
  "PyPTO 2027 UX 规划 · 战略讨论稿 · 外部资料核实于 2026-11-09，不代表已交付能力或研发承诺";

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9"; // 13.333 x 7.5 in
pptx.author = "PyPTO UX";
pptx.company = "PyPTO3";
pptx.title = "PyPTO 2027 UX 规划：从「能生成」到「能采信」";
const SW = 13.333;
const SH = 7.5;

let pageNo = 0;

function newSlide() {
  return pptx.addSlide();
}

// 顶部标题栏 + 页脚
function header(s, title, kicker) {
  pageNo += 1;
  s.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: SW, h: 1.02, fill: { color: PRIMARY }, line: { type: "none" },
  });
  s.addShape(pptx.ShapeType.rect, {
    x: 0, y: 1.02, w: SW, h: 0.042, fill: { color: ACCENT }, line: { type: "none" },
  });
  if (kicker) {
    s.addText(kicker, {
      x: 0.5, y: 0.1, w: 12.3, h: 0.3, fontSize: 11, bold: true,
      color: SKY, fontFace: FONT, valign: "middle",
    });
  }
  s.addText(title, {
    x: 0.5, y: kicker ? 0.38 : 0.26, w: 12.3, h: 0.56, fontSize: 22, bold: true,
    color: WHITE, fontFace: FONT, valign: "middle",
  });
  s.addText(FOOTER, {
    x: 0.5, y: SH - 0.33, w: 11.0, h: 0.26, fontSize: 8.5, color: GREY, fontFace: FONT,
  });
  s.addText(`${DATE}  ·  ${pageNo}`, {
    x: SW - 1.9, y: SH - 0.33, w: 1.5, h: 0.26, fontSize: 8.5, color: GREY,
    fontFace: FONT, align: "right",
  });
}

// 底部一句核心判断
function insight(s, text, opts = {}) {
  const y = opts.y !== undefined ? opts.y : 6.3;
  const color = opts.color || ACCENT;
  const h = opts.h || 0.8;
  s.addShape(pptx.ShapeType.rect, {
    x: 0.5, y, w: 12.33, h, fill: { color: BAND }, line: { type: "none" },
  });
  s.addShape(pptx.ShapeType.rect, {
    x: 0.5, y, w: 0.055, h, fill: { color }, line: { type: "none" },
  });
  s.addText(text, {
    x: 0.72, y, w: 12.0, h, fontSize: 12, bold: true, color: PRIMARY,
    fontFace: FONT, valign: "middle", lineSpacingMultiple: 1.15,
  });
}

// 浅色卡片
function card(s, x, y, w, h, title, body, opts = {}) {
  const fill = opts.fill || LIGHT;
  s.addShape(pptx.ShapeType.rect, {
    x, y, w, h, fill: { color: fill }, line: { type: "none" },
  });
  if (opts.bar) {
    s.addShape(pptx.ShapeType.rect, {
      x, y, w: 0.05, h, fill: { color: opts.bar }, line: { type: "none" },
    });
  }
  if (title) {
    s.addText(title, {
      x: x + 0.2, y: y + 0.12, w: w - 0.38, h: 0.3, fontSize: opts.titleSize || 13,
      bold: true, color: opts.titleColor || PRIMARY, fontFace: FONT,
    });
  }
  if (body) {
    s.addText(body, {
      x: x + 0.2, y: y + (title ? 0.46 : 0.14), w: w - 0.38,
      h: h - (title ? 0.58 : 0.26), fontSize: opts.bodySize || 11.5,
      color: opts.bodyColor || GREY, fontFace: FONT, lineSpacingMultiple: 1.22,
      valign: "top",
    });
  }
}

// 大数字卡片
function stat(s, x, y, w, h, value, label, note, color) {
  s.addShape(pptx.ShapeType.rect, {
    x, y, w, h, fill: { color: LIGHT }, line: { type: "none" },
  });
  s.addShape(pptx.ShapeType.rect, {
    x, y, w, h: 0.05, fill: { color: color || ACCENT }, line: { type: "none" },
  });
  s.addText(value, {
    x: x + 0.14, y: y + 0.18, w: w - 0.28, h: 0.62, fontSize: 30, bold: true,
    color: color || ACCENT, fontFace: FONT, align: "center", valign: "middle",
  });
  s.addText(label, {
    x: x + 0.14, y: y + 0.82, w: w - 0.28, h: 0.4, fontSize: 11.5, bold: true,
    color: PRIMARY, fontFace: FONT, align: "center", lineSpacingMultiple: 1.1,
  });
  if (note) {
    s.addText(note, {
      x: x + 0.14, y: y + 1.22, w: w - 0.28, h: h - 1.3, fontSize: 9.5,
      color: GREY, fontFace: FONT, align: "center", lineSpacingMultiple: 1.15,
    });
  }
}

// 表格
function table(s, x, y, w, headers, rows, colW, opts = {}) {
  const bodySize = opts.bodySize || 10.5;
  const headerSize = opts.headerSize || 10.5;
  const head = headers.map((t) => ({
    text: t,
    options: {
      bold: true, color: PRIMARY, fill: { color: BAND }, fontSize: headerSize,
      fontFace: FONT, valign: "middle", align: "left",
    },
  }));
  const body = rows.map((r, i) =>
    r.map((c, j) => {
      const cell = typeof c === "object" && c !== null ? c : { t: c };
      return {
        text: cell.t === null || cell.t === undefined ? "" : String(cell.t),
        options: {
          fontSize: cell.size || bodySize,
          color: cell.color || (opts.colColors && opts.colColors[j]) || GREY,
          bold: cell.bold || false,
          fill: { color: i % 2 === 0 ? WHITE : LIGHT },
          fontFace: FONT, valign: "middle", align: "left",
        },
      };
    })
  );
  s.addTable([head, ...body], {
    x, y, w, colW,
    border: { type: "solid", pt: 0.5, color: "D5DCE6" },
    autoPage: false,
    margin: [3, 5, 3, 5],
    rowH: opts.rowH,
  });
}

// 项目符号
function bullets(s, x, y, w, h, items, opts = {}) {
  const size = opts.size || 12.5;
  s.addText(
    items.map((it) => {
      const [text, level, color, bold] = Array.isArray(it) ? it : [it, 0];
      return {
        text,
        options: {
          bullet: { indent: 14 },
          indentLevel: level || 0,
          fontSize: size - (level || 0) * 1,
          color: color || opts.color || GREY,
          bold: !!bold,
          fontFace: FONT,
          paraSpaceAfter: opts.gap === undefined ? 5 : opts.gap,
          lineSpacingMultiple: 1.18,
        },
      };
    }),
    { x, y, w, h, valign: "top" }
  );
}

function notes(s, text) {
  s.addNotes(text);
}

// =========================================================
// S1 封面
// =========================================================
{
  const s = newSlide();
  s.background = { color: PRIMARY };
  s.addShape(pptx.ShapeType.rect, {
    x: 0, y: 4.42, w: SW, h: 0.042, fill: { color: ACCENT }, line: { type: "none" },
  });
  s.addText("PyPTO 2027 UX 规划", {
    x: 0.8, y: 1.25, w: 11.7, h: 1.0, fontSize: 42, bold: true, color: WHITE,
    fontFace: FONT,
  });
  s.addText("从「能生成」到「能采信」", {
    x: 0.8, y: 2.42, w: 11.7, h: 0.7, fontSize: 24, color: SKY, fontFace: FONT,
  });
  s.addText("五看三定 · 两个重点设计方向 + 一个条件启动方向", {
    x: 0.8, y: 3.3, w: 11.7, h: 0.45, fontSize: 14, color: DIM, fontFace: FONT,
  });
  s.addText(
    "本稿回答三个问题：2027 年 PyPTO 的 UX 主命题是什么、为什么不是「让 Agent 把算子写出来」、以及应当聚焦哪两件事。\n" +
      "外部论据区分一手与二手并标注核实日期；内部判断区分事实、推断与建议。不代表已交付能力、已确定产品边界或研发承诺。",
    {
      x: 0.8, y: 4.78, w: 11.7, h: 1.4, fontSize: 13, color: PALE, fontFace: FONT,
      lineSpacingMultiple: 1.32,
    }
  );
  s.addText("战略讨论稿 · 规划年度 2027 · " + DATE, {
    x: 0.8, y: 6.6, w: 11.7, h: 0.4, fontSize: 12, color: DIM, fontFace: FONT,
  });
  notes(
    s,
    "全文底稿：Product_Planning/PyPTO_2027_UX规划_五看三定.md。汇报时长建议 25 分钟：五看 10 分钟、方向 10 分钟、目标与路径 5 分钟。"
  );
}

// =========================================================
// S2 一页结论
// =========================================================
{
  const s = newSlide();
  header(s, "一页结论：2027 年稀缺的不是生成能力，是验收面", "核心结论");
  s.addShape(pptx.ShapeType.rect, {
    x: 0.5, y: 1.28, w: 12.33, h: 0.78, fill: { color: PRIMARY }, line: { type: "none" },
  });
  s.addText(
    "2027 年 PyPTO 的 UX 主命题，不是「让 Agent 把算子写出来」，而是「让人和组织敢把 Agent 写出来的东西用到生产里」。",
    {
      x: 0.72, y: 1.28, w: 11.9, h: 0.78, fontSize: 15, bold: true, color: WHITE,
      fontFace: FONT, valign: "middle", lineSpacingMultiple: 1.15,
    }
  );
  s.addText("三条已成立的外部事实叠加逼出这个结论：", {
    x: 0.5, y: 2.2, w: 12.3, h: 0.3, fontSize: 12.5, bold: true, color: PRIMARY,
    fontFace: FONT,
  });
  card(
    s, 0.5, 2.58, 3.94, 1.98,
    "① 生成已官方化",
    "HC 2026（9-19）把「从人工算子编程到 Agent 编程与调优」列为五个转变之一，并开源 CANNBot / Model Agent / MindStudio Agent / Solution Agent。\n\n→ 「有没有 Agent」在 2027 不再是差异点。",
    { bar: ACCENT, bodySize: 11 }
  );
  card(
    s, 4.69, 2.58, 3.94, 1.98,
    "② 生成的可信度远低于表观",
    "修正基线后最优模型几何平均加速 1.43× → 0.88×，不再快于 PyTorch；28% 的 kernel 抬高峰值显存；46.6% 的「正确」kernel 慢于 eager。\n\n→ Agent 自称成功不可直接采信。",
    { bar: RED, bodySize: 11 }
  );
  card(
    s, 8.88, 2.58, 3.95, 1.98,
    "③ 信任模式已固化",
    "编码 Agent 以 66% 成为第一大 AI 用例、73% 每日使用；但仅 48% 表示「能轻易验证时才信任」，16% 扩展到非重要决策。\n\n→ human-in-the-loop 是常态，不是降级路径。",
    { bar: AMBER, bodySize: 11 }
  );
  table(
    s, 0.5, 4.72, 12.33,
    ["", "方向", "一句话", "优先级"],
    [
      [
        { t: "方向一", bold: true, color: PRIMARY },
        { t: "可采信的 Agent 作业面", bold: true, color: PRIMARY },
        "把「Agent 交付一个算子」变成一次有契约、有独立验收、有接管出口的工程作业",
        { t: "P0 · 全年主线", bold: true, color: RED },
      ],
      [
        { t: "方向二", bold: true, color: PRIMARY },
        { t: "代际迁移与硬件约束工作台", bold: true, color: PRIMARY },
        "一年一代的硬件节奏下，让「源码可移植」的承诺在体验上被兑现",
        { t: "P0 · 跟随 960 节奏", bold: true, color: RED },
      ],
      [
        { t: "方向三", bold: true, color: PRIMARY },
        { t: "服务目标到算子的证据链", bold: true, color: PRIMARY },
        "SuperPoD / SuperCluster 规模下，从服务结果归因到算子与通信",
        { t: "P1 · 条件启动", bold: true, color: AMBER },
      ],
    ],
    [1.0, 2.9, 7.0, 1.43],
    { rowH: 0.4 }
  );
  insight(
    s,
    "两个 P0 共享同一套对象模型（任务契约 → 候选 → 证据 → 决策），因此不是两个产品，而是同一体验底座的两个入口。",
    { y: 6.44, h: 0.56 }
  );
  notes(
    s,
    "这一页是全稿的唯一必讲页。若只有 5 分钟，讲三张卡片与方向表即可。方向三明确标为条件启动，避免被当作同等投入承诺。"
  );
}

// =========================================================
// S3 五看 · 看市场与趋势
// =========================================================
{
  const s = newSlide();
  header(s, "2027 年有四个确定的时间点，不是趋势判断", "五看 ①｜看市场与趋势");
  table(
    s, 0.5, 1.3, 12.33,
    ["趋势与公开依据", "2027 年的具体形态", "对 PyPTO UX 的直接含义"],
    [
      [
        { t: "硬件一年一代，且 2027 年内有两次代际事件。960DT 定于 2027 Q1（提前三个季度）、960PR 定于 2027 Q3；970/980 分别 2028/2029。【外·二手】", size: 10 },
        { t: "开发者在同一年里面对两次新代际适配，而不是一次", size: 10 },
        { t: "迁移体验从「偶发项目」变成常态工作流，必须有常设入口，而非一次性迁移指南", size: 10 },
      ],
      [
        { t: "PTO ISA 对外开放，并把「跨代兼容、算子源码可移植」作为公开承诺；现含 8 类、120+ 条虚拟指令，支持 Auto 与 Manual 双模式。【外·一手】", size: 10 },
        { t: "承诺已公开，2027 是被开发者实际检验的一年", size: 10 },
        { t: "「可移植」是体验承诺而非只是编译器承诺：性能掉多少、哪一行要改、Auto 退回 Manual 的代价，必须可见", size: 10 },
      ],
      [
        { t: "Agent 编程与调优进入官方工具链（四类工具均已开源）；社区开放万卡级算力与「100 NPU-Hour 计划」，三年投入 50 亿元。【外·一手】", size: 10 },
        { t: "开发者手上同时有多个 Agent 工具，算力门槛显著降低", size: 10 },
        { t: "竞争转向「谁的产出可被采信、谁能承接别家 Agent 的产出」；算力约束减弱后，验证成本成为新瓶颈", size: 10 },
      ],
      [
        { t: "评测口径本身成为基础设施。CANN Bench 已进入官方 CANN 仓：53 算子 / 1060 用例 / 四难度层，编译—正确性—性能三轴加权，以真机硬件锚定上限（HAP）计分，抗 reward hacking。【外·一手】", size: 10 },
        { t: "「我的算子好不好」有了社区公认刻度", size: 10 },
        { t: "产品应对齐这把尺子而不是自造一把；三轴评分可直接作为验收界面的信息结构", size: 10 },
      ],
    ],
    [4.9, 2.9, 4.53],
    { rowH: 0.96 }
  );
  insight(
    s,
    "四条合起来：算力不缺、生成不缺、评测刻度开始统一——缺的是把三者接起来让人做决定的那一层。这是把「验收面」作为主命题的全部依据。",
    { y: 6.2, h: 0.7 }
  );
  notes(
    s,
    "边界：960 系列时间表来自对 HC 2026 的媒体转述，未见独立确认。若被追问，口径是「以官方发布为准，本规划用它做节奏假设而非承诺」。"
  );
}

// =========================================================
// S4 为什么不是「让 Agent 写出来」
// =========================================================
{
  const s = newSlide();
  header(s, "为什么主命题不是「让 Agent 把算子写出来」", "五看 ①｜关键反证");
  s.addText(
    "以下均为 2026 年公开论文的一手数据。它们的共同含义是：任何「Agent 自称成功」的信号都不可直接采信。",
    { x: 0.5, y: 1.3, w: 12.3, h: 0.32, fontSize: 12.5, bold: true, color: PRIMARY, fontFace: FONT }
  );
  stat(s, 0.5, 1.78, 2.96, 1.95, "0.88×", "修正基线后的几何平均加速", "原协议为 1.43×。无模型稳定优于 PyTorch", RED);
  stat(s, 3.63, 1.78, 2.96, 1.95, "46.6%", "「正确」kernel 慢于 eager 基线", "正确性不等于效率；跨硬件加速方差达 21.4×", RED);
  stat(s, 6.76, 1.78, 2.96, 1.95, "72%", "Fusion 任务五种方法全部失败", "量化类任务 0/30 成功", AMBER);
  stat(s, 9.89, 1.78, 2.94, 1.95, "56%", "昇腾 910B 上 fusion 算子可行率", "同口径 elementwise / normalization 为 90–100%", AMBER);
  card(
    s, 0.5, 3.93, 6.08, 1.3,
    "两个容易被误读的现象",
    "① 迭代修复会抬高编译通过率但拉低平均加速（编译率 52.3%→68.8%，平均加速 1.58×→1.44×）。\n② 28% 的生成 kernel 抬高峰值显存——速度与内存的取舍常被整体忽略。",
    { bar: RED, bodySize: 11 }
  );
  card(
    s, 6.76, 3.93, 6.07, 1.3,
    "失效原因不是模型不够强",
    "部分「加速」来自针对窄测试分布硬编码绕过计算；而难区分布（融合、跨层协调、数值精度契约）由任务结构决定——类别对正确性的解释力约为方法的三倍（9.4% vs 3.3%）。",
    { bar: PRIMARY, bodySize: 11 }
  );
  insight(
    s,
    "结论不是「Agent 不可用」，而是「生成能力会迅速商品化，而验证能力不会」。融合与跨层编排恰是 PyPTO 多级 IR 与 MPMD 的主场。",
    { y: 5.42, h: 0.72 }
  );
  s.addText(
    "来源：KernelBench-Verified（arXiv:2607.16241，2026-06-26）、KernelBenchX（arXiv:2605.04956，2026-05-11）、AgenticCANN（arXiv:2607.26661 v3，2026-08-13）。均为一手论文摘要；为特定环境下结果，不可直接迁移。",
    { x: 0.5, y: 6.26, w: 12.33, h: 0.5, fontSize: 9.5, color: AMBER, fontFace: FONT, lineSpacingMultiple: 1.15 }
  );
  notes(
    s,
    "这一页是说服页。若评审质疑「是不是选择性引用」，可指出 KernelBench-Verified 与 KernelBenchX 分属不同团队、不同硬件、不同口径，结论方向一致。"
  );
}

// =========================================================
// S5 看客户
// =========================================================
{
  const s = newSlide();
  header(s, "决策内容变了，决策人没变", "五看 ②｜看客户");
  table(
    s, 0.5, 1.3, 12.33,
    ["角色", "2026 的典型问题", "2027 的典型问题", "由此产生的体验诉求"],
    [
      ["算子工程师", "这个算子怎么写出来、怎么调快", { t: "这三个候选里哪个我敢签字，依据是什么", bold: true, color: PRIMARY }, "候选对比 + 独立验收 + 可下钻到 IR 与 trace"],
      ["模型／推理工程师", "这个算子在我的模型里快了多少", { t: "跨代硬件上我的模型还跑不跑得动、要改哪里", bold: true, color: PRIMARY }, "迁移影响面清单、按模型而非按算子组织的回归"],
      ["编译／后端工程师", "编译器为什么这么选", { t: "Agent 绕过了编译器的哪个假设，是 bug 还是启发式该改", bold: true, color: PRIMARY }, "Agent 决策与编译器决策的对照、可回灌的反例"],
      ["平台／性能团队", "谁在占设备", { t: "Agent 跑掉的算力换回了什么，哪些结果可复用", bold: true, color: PRIMARY }, "实验预算、产出留存、跨团队证据交接"],
      ["技术负责人／评审者", "这个优化有效吗", { t: "这批 Agent 产出的质量门槛在哪，谁担责", bold: true, color: PRIMARY }, "一页式验收摘要 + 明确的未覆盖项与责任边界"],
    ],
    [2.1, 3.2, 3.9, 3.13],
    { rowH: 0.52 }
  );
  card(
    s, 0.5, 4.52, 6.08, 1.6,
    "外部调研与这组推演一致",
    "AI 使用最集中的三个场景都属于「答案可核验」的技术任务：熟悉代码生成 69%、调试与重构 64%、写测试 58%。一旦任务转向难以核验的领域，使用率显著下降。",
    { bar: ACCENT, bodySize: 11.5 }
  );
  card(
    s, 6.76, 4.52, 6.07, 1.6,
    "仓库既有研究的三条主线被放大，而非被解决",
    "自动化的信任赤字 · 抽象是一道单向门 · 工具在最需要的时候失效。\nAgent 介入后生成速度上升，而验证速度没有同步上升。",
    { bar: PRIMARY, bodySize: 11.5 }
  );
  insight(
    s,
    "共同诉求压成一句：用户要的不是更强的 Agent，而是更便宜的「不信」——否掉一个坏候选的成本，决定了他们是否愿意让 Agent 批量产出。",
    { y: 6.3, h: 0.7 }
  );
  notes(
    s,
    "局限：角色与诉求基于仓库既有研究与场景推演，不是 2027 年的新一轮实证用研。方向一第一阶段应补一轮针对「验收行为」的专项访谈——这一条要主动说，不要等被问。"
  );
}

// =========================================================
// S6 看竞争
// =========================================================
{
  const s = newSlide();
  header(s, "竞争面是「验收」与「迁移」，不是 DSL 语法", "五看 ③｜看竞争");
  table(
    s, 0.5, 1.3, 12.33,
    ["对标层次", "代表参照与公开事实", "PyPTO 在 2027 的位置"],
    [
      [
        { t: "昇腾官方 Agent 工具链", bold: true, color: PRIMARY },
        { t: "CANNBot、Model Agent、MindStudio Agent、Solution Agent 均已开源。【外·一手】", size: 10 },
        { t: "不与之争生成，争验收与承接：应能消费这些工具的产出并给出独立结论", size: 10, color: PRIMARY, bold: true },
      ],
      [
        { t: "昇腾侧 Agent 算子研究", bold: true, color: PRIMARY },
        { t: "AgenticCANN 在 910B 六算子上：elementwise / normalization 可行率 90–100%，fusion 仅 56%；知识注入使 elementwise 从 57% 提到 86%。【外·一手】", size: 10 },
        { t: "「越难的算子越不行」是结构性现象。融合与跨层编排正是多级 IR 与 MPMD 的主场，是差异化的真实落点", size: 10, color: PRIMARY, bold: true },
      ],
      [
        { t: "社区评测刻度", bold: true, color: PRIMARY },
        { t: "CANN Bench 三轴评分 + 真机 HAP 上限 + 抗 reward hacking 设计，已版本化进入官方 CANN 仓。【外·一手】", size: 10 },
        { t: "谁定义「好」谁就定义界面。对齐而非另立：把三轴与 HAP 作为验收界面的信息骨架", size: 10, color: PRIMARY, bold: true },
      ],
      [
        { t: "上游生态的收敛动作", bold: true, color: PRIMARY },
        { t: "Helion 于 2026-04-07 被 PyTorch Foundation 接纳，采用 AOT autotuning（约 10 分钟搜索 1200 个候选），后端路线图含 TileIR / Pallas / CuTe DSL。【外·二手】", size: 10 },
        { t: "「一次搜很多候选」正在成为标准动作。候选规模化后，对比与采信界面成为瓶颈", size: 10, color: PRIMARY, bold: true },
      ],
      [
        { t: "头部模型侧的硬件协同", bold: true, color: PRIMARY },
        { t: "DeepSeek 于 2026-09-30 发布 Ascend 版六个仓库；co-design 案例包括为 Ascend 重排 FlashMLA 的 KV-cache 布局、因 Cube L0C 无法原地 rescale 而把累加器留在 Vector UB。【外·二手】", size: 10 },
        { t: "真实的昇腾优化全部发生在「硬件事实 → 实现决策」这一层。把这类决策显式化，是可防御的叙事", size: 10, color: PRIMARY, bold: true },
      ],
    ],
    [2.5, 5.9, 3.93],
    { rowH: 0.8 }
  );
  insight(
    s,
    "不应把差异化押在「tile DSL 更好用」上——这一层已高度同质。可防御的位置有两个：融合场景下的 Agent 可采信性；跨代迁移的体验兑现。",
    { y: 6.18, h: 0.7 }
  );
  notes(
    s,
    "Helion 与 DeepSeek 两条为二手，媒体与分析机构转述，其中「六个新仓库」的计数另有不同口径。汇报时如涉及对外材料，建议只保留一手条目。"
  );
}

// =========================================================
// S7 看自己
// =========================================================
{
  const s = newSlide();
  header(s, "结构优势真实，但兑现依赖四个前提", "五看 ④｜看自己");
  s.addText("可从仓库材料确认的结构性资产", {
    x: 0.5, y: 1.3, w: 6.08, h: 0.3, fontSize: 12.5, bold: true, color: PRIMARY, fontFace: FONT,
  });
  bullets(
    s, 0.5, 1.66, 6.08, 2.0,
    [
      ["Tile 为核心的编程模型", 0],
      ["Tensor → Tile → Block → Execution Graph 多级 IR 与 pass 体系", 0],
      ["CodeGen 到 PTO 虚拟指令、设备侧 MPMD 调度", 0],
      ["面向不同人群的分层抽象（Tensor / Tile / Block 级）", 0],
      ["大量可复用研究沉淀：体验洞察、1463 条 Issue 的失效类型、性能案例、Hardware-Native 问题地图", 0],
    ],
    { size: 11.5, gap: 5 }
  );
  s.addText("兑现它所依赖的四个前提", {
    x: 6.76, y: 1.3, w: 6.07, h: 0.3, fontSize: 12.5, bold: true, color: RED, fontFace: FONT,
  });
  card(
    s, 6.76, 1.66, 6.07, 1.06,
    "① 链路末端必须接回用户源码",
    "诊断只定位到 IR 文本位置时，对人价值有限，对 Agent 价值更低——Agent 无法用一个指向 IR 行的提示去改用户代码。",
    { bar: RED, bodySize: 10.5, titleSize: 11.5 }
  );
  card(
    s, 6.76, 2.8, 6.07, 0.88,
    "② 需要作业级而非 kernel 级的验收对象",
    "否则 MPMD 与跨 kernel 编排的优势无法进入用户感知。",
    { bar: AMBER, bodySize: 10.5, titleSize: 11.5 }
  );
  card(
    s, 6.76, 3.76, 6.07, 0.88,
    "③ 失效类型要转成界面上的检查项",
    "已有分类不能停留在报告里，否则不参与任何一次真实决策。",
    { bar: AMBER, bodySize: 10.5, titleSize: 11.5 }
  );
  card(
    s, 6.76, 4.72, 6.07, 0.92,
    "④ 降级路径（逃生舱）的迁移成本必须低",
    "若从自动层降到手动层等于重写数据流，分层抽象在真实压力下不会被使用。",
    { bar: RED, bodySize: 10.5, titleSize: 11.5 }
  );
  card(
    s, 0.5, 3.78, 6.08, 1.62,
    "这四条的归属",
    "前两条决定方向一能否成立，第四条决定方向二能否成立。\n\n它们都位于体验与架构的交界，无法只靠 UX 侧交付，必须进入 2027 的联合排期。",
    { bar: PRIMARY, bodySize: 11.5 }
  );
  insight(
    s,
    "优势不在「有多级 IR」，而在「语义链路比纯 kernel DSL 更长」；但长链路只有在末端能接回用户写的那一行时才产生用户价值。",
    { y: 5.74, h: 0.62 }
  );
  notes(
    s,
    "这一页承担自我批评功能。建议主动讲前提①，不要等技术评审提出来——它是本规划里唯一一条「不解决则两个方向都只能做一半」的依赖。"
  );
}

// =========================================================
// S8 看机会
// =========================================================
{
  const s = newSlide();
  header(s, "三个机会：价值假设与首要不确定性", "五看 ⑤｜看机会");
  table(
    s, 0.5, 1.4, 12.33,
    ["机会", "用户价值假设", "战略价值假设", "首要不确定性"],
    [
      [
        { t: "可采信的 Agent 作业面", bold: true, color: PRIMARY },
        "把「否掉一个坏候选」的成本从小时级降到分钟级，让专家敢于让 Agent 批量产出",
        "在生成能力商品化后占住验收位；对齐 CANN Bench 刻度即进入社区默认路径",
        { t: "独立验收能否做到足够强（反例生成、隐藏分布、显存与数值同时看），以及验证本身的算力成本", color: AMBER },
      ],
      [
        { t: "代际迁移与硬件约束工作台", bold: true, color: PRIMARY },
        "一年两次代际事件下，把迁移从项目变成例行作业",
        "兑现 PTO ISA 的公开可移植承诺，形成官方叙事的体验载体",
        { t: "960 系列真机与工具链的可得时间；跨代性能回归的数据来源", color: AMBER },
      ],
      [
        { t: "服务目标到算子的证据链", bold: true, color: PRIMARY },
        "让团队知道该优化哪一层、改了是否对真实负载有效",
        "把底层性能与服务价值连接",
        { t: "跨系统数据权限、代表性负载、与服务平台的边界", color: AMBER },
      ],
    ],
    [2.9, 3.3, 3.1, 3.03],
    { rowH: 0.9 }
  );
  card(
    s, 0.5, 5.1, 12.33, 1.1,
    "取舍逻辑",
    "前两个机会的不确定性是「能不能做好」，可以通过设计与试点在 2027 年内回答；第三个机会的不确定性是「数据与边界能不能拿到」，不由本团队决定。因此前两个列为 P0，第三个设为条件启动。",
    { bar: ACCENT, bodySize: 11.5 }
  );
  insight(
    s,
    "三个机会共用同一套对象模型。方向三启动时复用，不另建模型——这是避免产出两套不互通界面的唯一办法。",
    { y: 6.32, h: 0.6 }
  );
}

// =========================================================
// S9 三定总览
// =========================================================
{
  const s = newSlide();
  header(s, "三定总览：两个重点 + 一个条件启动", "三定｜定方向");
  const rows = [
    ["用户目标", "交付一个算子　/　迁移到新代际　/　达成服务目标", PRIMARY, WHITE],
    ["人 + Agent", "定义契约　→　探索候选　→　独立验收　→　签字", ACCENT, WHITE],
    ["共享对象", "任务契约　→　候选　→　证据（IR / trace / 测量条件）　→　决策", BAND, PRIMARY],
    ["既有执行层", "PyPTO 编译与调度　·　CANN　·　官方 Agent 工具　·　服务与训练框架", LIGHT, GREY],
  ];
  let y = 1.42;
  rows.forEach(([label, text, fill, color], i) => {
    s.addShape(pptx.ShapeType.rect, {
      x: 2.2, y, w: 9.0, h: 0.72, fill: { color: fill }, line: { type: "none" },
    });
    s.addText(text, {
      x: 2.4, y, w: 8.6, h: 0.72, fontSize: 13, bold: i < 2, color,
      fontFace: FONT, valign: "middle", align: "center",
    });
    s.addText(label, {
      x: 0.6, y, w: 1.5, h: 0.72, fontSize: 11.5, bold: true, color: PRIMARY,
      fontFace: FONT, valign: "middle", align: "right",
    });
    if (i < rows.length - 1) {
      s.addText("▼", {
        x: 6.4, y: y + 0.74, w: 0.6, h: 0.26, fontSize: 11, color: ACCENT,
        fontFace: FONT, align: "center",
      });
    }
    y += 1.0;
  });
  card(
    s, 0.5, 5.5, 3.94, 1.26,
    "方向一 · P0",
    "可采信的 Agent 作业面\n契约 → 候选 → 独立验收 → 签字",
    { bar: RED, bodySize: 11 }
  );
  card(
    s, 4.69, 5.5, 3.94, 1.26,
    "方向二 · P0",
    "代际迁移与硬件约束工作台\n跟随 960DT（Q1）/ 960PR（Q3）",
    { bar: RED, bodySize: 11 }
  );
  card(
    s, 8.88, 5.5, 3.95, 1.26,
    "方向三 · P1 条件启动",
    "服务目标到算子的证据链\n2027 只做「接口 + 一个故事」",
    { bar: AMBER, bodySize: 11 }
  );
  notes(
    s,
    "共享对象是设计提议，不预设统一平台已存在。方向一与方向二必须共用契约与证据对象，否则会产出两套不互通的界面。"
  );
}

// =========================================================
// S10 方向一 · 问题判断与目标体验
// =========================================================
{
  const s = newSlide();
  header(s, "方向一｜可采信的 Agent 作业面", "定方向 · P0 · 2027 全年主线");
  card(
    s, 0.5, 1.32, 12.33, 1.16,
    "问题判断",
    "瓶颈不在生成。表观加速在修正基线后可能归零甚至为负、正确的 kernel 近半数慢于基线、迭代修复抬高编译率却拉低平均加速、融合类是系统性难区。这些数字的共同含义是：产品必须提供与生成过程解耦的独立验收。",
    { bar: RED, bodySize: 12 }
  );
  card(
    s, 0.5, 2.64, 12.33, 1.34,
    "目标体验",
    "开发者给出一个有边界的目标（算子语义、设备、workload、精度要求、可改范围、算力预算、停止条件），Agent 在授权内探索；\n最终开发者看到的不是一段对话，而是一张可签字的验收页——哪些候选通过了什么检查、在什么硬件上离上限多远、哪些输入没覆盖、花了多少设备时间、哪一步是 Agent 的假设而不是工具的结论。",
    { bar: ACCENT, bodySize: 12 }
  );
  s.addText("用户此刻真正在做的决定", {
    x: 0.5, y: 4.14, w: 12.3, h: 0.3, fontSize: 12.5, bold: true, color: PRIMARY, fontFace: FONT,
  });
  card(s, 0.5, 4.5, 3.94, 0.92, "「我敢不敢签字」", "不是「它快不快」。签字意味着承担后续风险。", { bar: PRIMARY, bodySize: 10.5, titleSize: 12 });
  card(s, 4.69, 4.5, 3.94, 0.92, "「这个候选差在哪」", "需要在分钟级而不是小时级得到否定结论。", { bar: PRIMARY, bodySize: 10.5, titleSize: 12 });
  card(s, 8.88, 4.5, 3.95, 0.92, "「我能不能接手」", "接管是常态路径，不是失败后的补救。", { bar: PRIMARY, bodySize: 10.5, titleSize: 12 });
  insight(
    s,
    "失败即停（kill criteria）：若独立验收无法在不显著增加算力成本的前提下稳定识别出「表观有效但实际无效」的候选，则核心假设不成立，应退回到「证据组织工具」的较小定位。",
    { y: 5.62, h: 0.74, color: RED }
  );
  notes(
    s,
    "kill criteria 要主动讲。它是本规划与一般方向性材料的区别：给出了判定失败的条件，而不是只给成功愿景。"
  );
}

// =========================================================
// S11 方向一 · 五个设计对象
// =========================================================
{
  const s = newSlide();
  header(s, "方向一｜五个关键设计对象", "定方向 · P0 · 设计对象");
  const objs = [
    ["① 任务契约　Task Contract", "目标、数值语义、设备与代际、workload 形状分布、精度门槛、可修改文件范围、算力预算、停止条件。可编辑、可复用、保留来源。", "它同时是 Agent 的输入和验收的判据——没有契约就没有「通过」的定义。", ACCENT],
    ["② 独立验收面　Acceptance Panel", "对齐 CANN Bench 三轴：编译、功能正确性、性能分别独立呈现；性能以真机硬件锚定上限（HAP 式刻度）而非相对倍数作为主刻度。", "追加两条必查项：峰值内存变化（外部数据 28% 的生成 kernel 抬高峰值显存）、数值精度口径（量化路径外部评测 0/30）。", RED],
    ["③ 反例与覆盖　Coverage", "显式呈现测试分布、隐藏分布结果，以及哪些输入形状／数值范围未被覆盖。", "抗 reward hacking 的用户侧对应物：用户不必理解 reward hacking，但需看到「这个候选只在测过的那几个 shape 上成立」。", AMBER],
    ["④ 证据与归因　Evidence Trail", "每个候选绑定代码版本、编译产物、IR 与 pass 决策、trace、测量条件。", "区分三类来源：Agent 的假设、工具的结果、人的决定。这是多级 IR 的体验兑现点，也是与纯 kernel DSL 的真实分野。", PRIMARY],
    ["⑤ 预算与接管　Budget & Handover", "设备时间与调用成本实时可见；支持暂停、接管、从任意候选分叉继续。", "外部调研显示 human-in-the-loop 是主导模式——接管不是降级路径而是常态路径。", GREEN],
  ];
  let y = 1.3;
  objs.forEach(([t, body, note, color]) => {
    s.addShape(pptx.ShapeType.rect, {
      x: 0.5, y, w: 12.33, h: 0.92, fill: { color: LIGHT }, line: { type: "none" },
    });
    s.addShape(pptx.ShapeType.rect, {
      x: 0.5, y, w: 0.05, h: 0.92, fill: { color }, line: { type: "none" },
    });
    s.addText(t, {
      x: 0.68, y: y + 0.08, w: 3.1, h: 0.76, fontSize: 11.5, bold: true,
      color: PRIMARY, fontFace: FONT, valign: "middle", lineSpacingMultiple: 1.1,
    });
    s.addText(body, {
      x: 3.86, y: y + 0.06, w: 4.5, h: 0.8, fontSize: 10, color: GREY,
      fontFace: FONT, valign: "middle", lineSpacingMultiple: 1.16,
    });
    s.addText(note, {
      x: 8.5, y: y + 0.06, w: 4.2, h: 0.8, fontSize: 10, color, fontFace: FONT,
      valign: "middle", lineSpacingMultiple: 1.16, italic: false,
    });
    y += 1.0;
  });
  s.addText("设计对象", { x: 0.68, y: 1.04, w: 3.0, h: 0.22, fontSize: 9, bold: true, color: GREY, fontFace: FONT });
  s.addText("内容", { x: 3.86, y: 1.04, w: 3.0, h: 0.22, fontSize: 9, bold: true, color: GREY, fontFace: FONT });
  s.addText("为什么必须有", { x: 8.5, y: 1.04, w: 3.0, h: 0.22, fontSize: 9, bold: true, color: GREY, fontFace: FONT });
  s.addText(
    "首个设计交付物：三段式交互原型 —— (a) 契约起草（含缺失项提示与预算设定） → (b) 执行与接管（计划、当前动作、预算消耗、失败与分叉） → (c) 验收与签字（三轴 + 覆盖 + 证据下钻 + 一页摘要）；配套验收数据契约草案（最小字段集）。",
    { x: 0.5, y: 6.34, w: 12.33, h: 0.62, fontSize: 11, bold: true, color: PRIMARY, fontFace: FONT, lineSpacingMultiple: 1.18 }
  );
}

// =========================================================
// S12 方向二 · 问题判断与目标体验
// =========================================================
{
  const s = newSlide();
  header(s, "方向二｜代际迁移与硬件约束工作台", "定方向 · P0 · 跟随 960 系列节奏");
  card(
    s, 0.5, 1.32, 12.33, 1.32,
    "问题判断",
    "2027 年内有两次代际事件（960DT 定于 Q1、960PR 定于 Q3），而 PTO ISA 已公开承诺「tile 级抽象实现跨代兼容、算子源码可跨代迁移」并提供 Auto／Manual 双模式。承诺公开意味着它会被检验：源码能编过不等于迁移完成——性能是否保持、哪些手工优化在新代际失效、Auto 不够时降到 Manual 要改多少，都是真实的用户工作。",
    { bar: RED, bodySize: 11.5 }
  );
  card(
    s, 0.5, 2.76, 6.08, 1.5,
    "外部 co-design 案例佐证这类决策的形态",
    "· 为 Ascend 重排 FlashMLA 的 KV-cache 布局，把量化值与 scale 同区存放以减少搬运请求\n· 因 Cube L0C 无法原地 rescale，把累加器留在 Vector UB\n\n都不是「写得对不对」，而是「硬件事实 → 实现决策」。",
    { bar: ACCENT, bodySize: 10.5 }
  );
  card(
    s, 6.76, 2.76, 6.07, 1.5,
    "目标体验",
    "开发者把一组已有算子或一个模型指向新代际，得到一张迁移影响面：哪些可直接复用、哪些能编过但性能回退、哪些依赖的硬件事实已改变、每一条对应要改的那一行源码与可选方案。\n\nAuto 达不到目标时，在同一套写法内逐步接手控制权，而不是重写数据流。",
    { bar: PRIMARY, bodySize: 10.5 }
  );
  s.addText("五个关键设计对象", {
    x: 0.5, y: 4.34, w: 12.3, h: 0.28, fontSize: 12.5, bold: true, color: PRIMARY, fontFace: FONT,
  });
  const o2 = [
    ["① 代际事实差异表", "与开发决策相关的硬件事实（存储层级、搬运与同步、精度格式如 HiF8→HiF4、SIMD 与 SIMT 组合）组织成可引用条目，每条关联「影响哪类决策」"],
    ["② 迁移影响面", "以算子／模型为单位的分类清单：直接复用 / 需调参 / 需改实现 / 不支持；每条给出判据与证据，而非仅给结论"],
    ["③ 约束到源码的回指", "硬件约束被破坏或逼近时，提示指向用户写的那一行，而不是流水线后的 IR 位置（与方向一共用的地基）"],
    ["④ 渐进式逃生舱", "Auto → Manual 的控制权移交应是同一套语法内的标注式降级，而非切换到另一套数据流写法"],
    ["⑤ 跨代回归", "同一套契约在两个代际上的结果并排呈现，测量条件差异显式标注，禁止跨条件直接比较"],
  ];
  let y2 = 4.68;
  o2.forEach(([t, body], i) => {
    s.addShape(pptx.ShapeType.rect, {
      x: 0.5, y: y2, w: 12.33, h: 0.33,
      fill: { color: i % 2 === 0 ? LIGHT : WHITE }, line: { type: "none" },
    });
    s.addText(t, {
      x: 0.62, y: y2, w: 2.3, h: 0.33, fontSize: 10.5, bold: true, color: PRIMARY,
      fontFace: FONT, valign: "middle",
    });
    s.addText(body, {
      x: 2.98, y: y2, w: 9.7, h: 0.33, fontSize: 9.8, color: GREY, fontFace: FONT,
      valign: "middle",
    });
    y2 += 0.35;
  });
  insight(
    s,
    "最高风险一条是④：若 Auto 降到 Manual 等于重写数据流，分层抽象在真实压力下不会被使用。失败即停：960 真机与工具链 2027 年内不可得，则降级为「代际事实知识库 + 迁移检查清单」。",
    { y: 6.48, h: 0.6, color: RED }
  );
  notes(
    s,
    "首个设计交付物：一个代际迁移故事的交互原型——选定一组真实算子（优先取仓库已有案例），走完「指向新代际 → 影响面 → 逐条处置 → 跨代回归 → 迁移摘要」；外加代际事实条目的结构定义（字段、来源、失效期）。"
  );
}

// =========================================================
// S13 方向三 · 条件启动
// =========================================================
{
  const s = newSlide();
  header(s, "方向三｜服务目标到算子的证据链（条件启动）", "定方向 · P1");
  card(
    s, 0.5, 1.32, 12.33, 1.4,
    "问题判断",
    "2027 年的运行环境是 SuperPoD 与 SuperCluster：单套 Atlas 960E SuperPoD 可支持 4096 NPU、8 EFLOPS FP8、至多 1PB HBM；SuperCluster 可互联至 512,000 NPU，多轨拓扑可扩展到百万级；华为并指出常规服务器集群中处理器间通信可占训练时间 40% 以上。仓库既有研究亦指向同一结论：分析的最小单元在真实服务里是链路而不是算子。",
    { bar: ACCENT, bodySize: 11.5 }
  );
  stat(s, 0.5, 2.88, 3.94, 1.6, "40%+", "通信在常规集群中可占训练时间", "算子加速与服务目标达成之间的距离被进一步拉大", AMBER);
  card(
    s, 4.69, 2.88, 8.14, 1.6,
    "为什么是 P1 而不是并列 P0",
    "这一层的体验建设强依赖外部数据与边界——服务平台、调度、框架、权限都不由本团队决定。把它与两个 P0 并列投入，会在拿不到数据时同时拖慢三件事。",
    { bar: AMBER, bodySize: 11.5 }
  );
  card(
    s, 0.5, 4.64, 6.08, 1.6,
    "2027 年的最小动作",
    "不建平台，先建「接口 + 一个故事」：选一个真实的推理或训练链路问题，走通「服务目标 → 分层下钻 → 定位到算子或通信 → 回到方向一的契约 → 验收后交回服务侧」的闭环。\n过程中明确三件事：数据从哪来、谁有权限、结论由谁负责。",
    { bar: PRIMARY, bodySize: 10.8 }
  );
  card(
    s, 6.76, 4.64, 6.07, 1.6,
    "启动条件（三者缺一则保持研究议题）",
    "① 具备一个愿意共建的服务或训练团队\n② 可获得代表性负载与样本 trace\n③ 方向一的契约与证据对象已稳定",
    { bar: GREEN, bodySize: 11.5 }
  );
  insight(
    s,
    "方向三启动时复用方向一与方向二的同一套对象模型，不另建模型。",
    { y: 6.42, h: 0.54 }
  );
}

// =========================================================
// S14 定目标
// =========================================================
{
  const s = newSlide();
  header(s, "定目标：不以单一加速倍数作为价值陈述", "三定｜定目标");
  s.addShape(pptx.ShapeType.rect, {
    x: 0.5, y: 1.3, w: 12.33, h: 0.66, fill: { color: PRIMARY }, line: { type: "none" },
  });
  s.addText(
    "总体目标（2027）：把「一次 Agent 产出从交付到被采信」的周期与成本降到可规模化的水平，同时不降低质量门槛。",
    { x: 0.72, y: 1.3, w: 11.9, h: 0.66, fontSize: 13.5, bold: true, color: WHITE, fontFace: FONT, valign: "middle" }
  );
  table(
    s, 0.5, 2.1, 12.33,
    ["方向", "用户价值指标", "质量与成本护栏", "待验证的成功信号"],
    [
      [
        { t: "方向一 可采信的 Agent 作业面", bold: true, color: PRIMARY },
        "否掉坏候选的时间；从契约到可签字候选的时间；人工介入的原因分布",
        "三轴（编译／正确性／性能）达标率、覆盖缺口数、峰值内存变化、设备时间与调用成本",
        "同批真实任务下用户更快形成可审查结论，且采信结果在真机回归中不翻盘",
      ],
      [
        { t: "方向二 代际迁移工作台", bold: true, color: PRIMARY },
        "迁移中需从零排查的算子比例；迁移影响面的判定准确率",
        "跨代性能回退的发现时点（上线前／后）、降级改动量（行数或改动类型）",
        "代际事件前后，迁移从专项项目变为例行作业",
      ],
      [
        { t: "方向三 服务证据链（条件）", bold: true, color: AMBER },
        "归因到正确层级的比例",
        "测量口径一致性、负载代表性",
        "团队能据此决定优化层级并在代表性负载上验证",
      ],
    ],
    [2.7, 3.2, 3.4, 3.03],
    { rowH: 0.8 }
  );
  card(
    s, 0.5, 5.4, 6.08, 1.48,
    "Q1 必须先做基线测量",
    "现有流程下「否掉一个候选」「完成一次迁移判定」各需多长时间、失败原因分布如何。没有基线的量化目标会在复盘时无法解释。报告时区分离线实验、模拟负载与真实生产，披露样本量与未完成任务。",
    { bar: ACCENT, bodySize: 10.5 }
  );
  card(
    s, 6.76, 5.4, 6.07, 1.48,
    "不承诺的三件事（期望管理）",
    "① 不承诺「Agent 自动写出高性能算子」——外部一手数据不支持，承诺会在第一次真实演示中被戳破；应承诺「每一步可验证、可接管」。\n② 不承诺跨厂商硬件可移植。\n③ 不承诺全算子覆盖；应公开覆盖范围与更新节奏。",
    { bar: RED, bodySize: 10.5 }
  );
}

// =========================================================
// S15 定路径
// =========================================================
{
  const s = newSlide();
  header(s, "定路径：2027 四个阶段，每阶段都有退出条件", "三定｜定路径");
  table(
    s, 0.5, 1.34, 12.33,
    ["阶段", "重点工作", "UX 与技术交付", "退出条件／决策点"],
    [
      [
        { t: "Q1\n定义与基线", bold: true, color: PRIMARY },
        "针对「验收行为」的专项访谈（算子、模型、评审三类角色）；盘点可用真机与数据；确认与官方 Agent 工具链的分工",
        "任务契约与验收面的对象定义；验收数据契约草案；现状基线报告",
        { t: "用户确认「否掉候选的成本」是真问题；契约字段可被至少一类真实任务填满", color: PRIMARY },
      ],
      [
        { t: "Q2\n方向一首个闭环", bold: true, color: PRIMARY },
        "选 1–2 个真实算子任务（优先取融合类，因其为公认难区）打通契约 → 执行 → 验收",
        "三段式交互原型 + 可运行的验收面（对齐三轴与真机上限刻度）",
        { t: "能稳定识别出「表观有效实际无效」的候选；否则触发方向一的 kill criteria", color: RED },
      ],
      [
        { t: "Q2–Q3\n方向二跟随代际", bold: true, color: PRIMARY },
        "结合 960 系列可得性做代际事实条目与迁移影响面；并行推进「约束回指源码」与渐进式逃生舱的架构前置项",
        "代际迁移交互原型；代际事实结构定义；降级路径的设计方案",
        { t: "真机与工具链可得；降级改动量可控；否则降级为知识库 + 检查清单", color: RED },
      ],
      [
        { t: "Q3–Q4\n收敛与条件扩展", bold: true, color: PRIMARY },
        "两个 P0 的对象模型合并固化；视条件启动方向三的一个故事",
        "统一对象模型文档；跨代回归视图；（条件）服务链路故事原型",
        { t: "以真实团队采用率、采信结果的翻盘率、迁移发现时点决定是否规模化", color: PRIMARY },
      ],
    ],
    [1.9, 3.7, 3.3, 3.43],
    { rowH: 0.82 }
  );
  card(
    s, 0.5, 5.56, 12.33, 1.3,
    "需要产品与技术在 Q1 共同拍板的五件事",
    "① 架构前置项排期：约束与诊断能否回指用户源码行；渐进式降级是否在同一语法内实现——这两项不解决，两个方向都只能做到一半。\n" +
      "② 与官方 Agent 工具链的关系：是否承接 CANNBot 等工具的产出并给出独立验收结论（建议：是）。　③ 是否直接采用 CANN Bench 三轴与真机上限刻度作为验收界面骨架。\n" +
      "④ 算力与预算治理：验证消耗的设备时间由谁承担、如何呈现、Agent 自主探索的上限。　⑤ 责任归属：正确性、精度与上线责任如何在人、Agent 与平台之间划分。",
    { bar: RED, bodySize: 10.5 }
  );
}

// =========================================================
// S16 风险
// =========================================================
{
  const s = newSlide();
  header(s, "风险与应对", "收束 ①");
  table(
    s, 0.5, 1.4, 12.33,
    ["风险", "说明", "应对"],
    [
      [
        { t: "验收面自身不可信", bold: true, color: RED },
        "若独立验收的测试分布不够宽，它会复制 reward hacking 的问题，只是换了一层界面",
        "对齐已公开抗 reward hacking 设计的社区基准；覆盖缺口必须显式可见而非静默通过",
      ],
      [
        { t: "架构前置项不落地", bold: true, color: RED },
        "源码回指与渐进降级属于架构工作，UX 侧无法单独交付",
        "Q1 作为联合排期的硬前置项确认，不确认则调整方向定位而非硬推",
      ],
      [
        { t: "代际节奏不确定", bold: true, color: AMBER },
        "960 时间表来自媒体转述，真机与工具链可得时间未知",
        "方向二设置明确降级形态（知识库 + 检查清单）",
      ],
      [
        { t: "与官方工具链定位冲突", bold: true, color: AMBER },
        "官方已开源四类 Agent 工具，若定位不清会被视为重复建设",
        "明确「不争生成、争采信与承接」，并在材料中正面陈述分工",
      ],
      [
        { t: "验证成本超过生成成本", bold: true, color: AMBER },
        "算力门槛下降后，验证成为新的成本中心",
        "把预算与成本做成一等界面对象；验证策略分层（廉价筛除 + 昂贵确认）",
      ],
      [
        { t: "外部数据的时效与口径", bold: true, color: GREY },
        "多条关键依据为 2026 年中的论文与会议材料，2027 年可能演进",
        "每半年复核一次来源；结论随证据更新而非随叙事惯性保留",
      ],
    ],
    [2.7, 4.7, 4.93],
    { rowH: 0.66 }
  );
  insight(
    s,
    "前两条是本规划的真实生死线：验收面不可信、架构前置项不落地，任何一条成立都应调整方向定位，而不是继续推进界面设计。",
    { y: 6.14, h: 0.62, color: RED }
  );
}

// =========================================================
// S17 来源与边界
// =========================================================
{
  const s = newSlide();
  header(s, "信息来源与边界", "收束 ②｜公开资料核实于 2026-11-09");
  s.addText("外部一手", {
    x: 0.5, y: 1.3, w: 6.08, h: 0.28, fontSize: 12, bold: true, color: GREEN, fontFace: FONT,
  });
  bullets(
    s, 0.5, 1.62, 6.08, 2.5,
    [
      ["华为官方新闻：Huawei Advances Agentic Computing…（2026-09-19，HC 2026）— 五个转变、PTO ISA 开放、四类 Agent 工具开源、万卡算力与 100 NPU-Hour、三年 50 亿", 0],
      ["CANN Bench，arXiv:2607.20518（2026-07-08）— 53 算子/1060 用例、三轴评分、真机 HAP 上限、抗 reward hacking", 0],
      ["AgenticCANN，arXiv:2607.26661 v3（2026-08-13）— 910B 六算子；fusion 56%、elementwise 90–100%；知识注入 57%→86%", 0],
      ["KernelBench-Verified，arXiv:2607.16241（2026-06-26）— 修正基线后 0.88×；28% 抬高峰值显存", 0],
      ["KernelBenchX，arXiv:2605.04956 v2（2026-05-11）— 46.6% 正确 kernel 慢于 eager；Fusion 72% 全失败；量化 0/30", 0],
      ["Stack Overflow Developer Survey 2026 · AI — 66% 第一用例、73% 每日使用、48% 仅在易验证时信任", 0],
    ],
    { size: 9.8, gap: 4 }
  );
  s.addText("外部二手（需进一步确认）", {
    x: 6.76, y: 1.3, w: 6.07, h: 0.28, fontSize: 12, bold: true, color: AMBER, fontFace: FONT,
  });
  bullets(
    s, 6.76, 1.62, 6.07, 1.5,
    [
      ["CRN Asia（HC 2026 报道）— 960DT 2027 Q1、960PR 2027 Q3；Atlas 960E / SuperCluster 规模；通信占训练时间 40%+。时间表未见独立确认", 0],
      ["PyTorch Foundation 接纳 Helion（2026-04-07）— AOT autotuning 约 10 分钟搜 1200 候选。多为发布稿与第三方博客", 0],
      ["DeepSeek × Ascend（转述 SemiAnalysis，2026-10-22）— 六仓库、FlashMLA KV-cache 重排、L0C 累加器。「六个新仓库」计数另有不同口径", 0],
    ],
    { size: 9.8, gap: 4 }
  );
  s.addText("内部材料", {
    x: 6.76, y: 3.3, w: 6.07, h: 0.28, fontSize: 12, bold: true, color: ACCENT, fontFace: FONT,
  });
  bullets(
    s, 6.76, 3.62, 6.07, 1.6,
    [
      ["repo/pto/README.md — Tile 模型、多级 IR、MPMD、分层抽象", 0],
      ["Insight/PyPTO开发者体验洞察报告0818.md — 五类角色与三条主线", 0],
      ["Insight/Hardware-Native失效类型_基于1463条Issue_20261019.md", 0],
      ["Insight/PyPTO性能调优典型用户案例_基于GitHub_20260923.md", 0],
      ["Product_Planning/Hardware-Native_开发全流程体验问题地图_20261104.md", 0],
      ["Product_Planning/Agent强参与下的算子开发_流程开发者与UX变化.md", 0],
    ],
    { size: 9.8, gap: 3 }
  );
  card(
    s, 0.5, 4.3, 6.08, 1.74,
    "边界声明",
    "· 公开研究与评测是特定环境下的结果，供趋势判断与设计启发，不代表 PyPTO 或昇腾的实测效果。\n· 其他硬件平台的实验结果不可直接迁移到昇腾。\n· 角色与诉求基于仓库既有研究与场景推演，不是 2027 年的新一轮实证用研。\n· 本稿为战略讨论稿，不代表已交付能力、已确定产品边界或研发承诺。",
    { bar: GREY, bodySize: 10.2 }
  );
  insight(
    s,
    "全文底稿：Product_Planning/PyPTO_2027_UX规划_五看三定.md（含逐条标注的事实／推断／建议区分）。",
    { y: 6.18, h: 0.56 }
  );
}

// =========================================================
// S18 收束
// =========================================================
{
  const s = newSlide();
  s.background = { color: PRIMARY };
  s.addShape(pptx.ShapeType.rect, {
    x: 0, y: 1.02, w: SW, h: 0.042, fill: { color: ACCENT }, line: { type: "none" },
  });
  s.addText("如果 2027 只做两件事", {
    x: 0.8, y: 0.3, w: 11.7, h: 0.62, fontSize: 26, bold: true, color: WHITE, fontFace: FONT,
  });
  s.addText(
    "生成能力在 2027 会迅速商品化——官方工具链已开源，社区算力已开放，评测刻度已统一。\n" +
      "在这个格局里，稀缺的不是「写得出来」，而是「敢用」。",
    {
      x: 0.8, y: 1.4, w: 11.7, h: 0.95, fontSize: 14.5, color: PALE, fontFace: FONT,
      lineSpacingMultiple: 1.35,
    }
  );
  s.addShape(pptx.ShapeType.rect, {
    x: 0.8, y: 2.5, w: 5.6, h: 1.95, fill: { color: "2A4A80" }, line: { type: "none" },
  });
  s.addText("一｜把一次 Agent 产出变成一次可签字的工程作业", {
    x: 1.02, y: 2.68, w: 5.2, h: 0.6, fontSize: 15, bold: true, color: WHITE, fontFace: FONT,
    lineSpacingMultiple: 1.15,
  });
  s.addText(
    "契约 → 候选 → 独立验收 → 证据 → 签字。\n对齐社区刻度，而不是自造一把尺子。",
    { x: 1.02, y: 3.42, w: 5.2, h: 0.9, fontSize: 12.5, color: PALE, fontFace: FONT, lineSpacingMultiple: 1.3 }
  );
  s.addShape(pptx.ShapeType.rect, {
    x: 6.93, y: 2.5, w: 5.6, h: 1.95, fill: { color: "2A4A80" }, line: { type: "none" },
  });
  s.addText("二｜让「源码可移植」的公开承诺在体验上被兑现", {
    x: 7.15, y: 2.68, w: 5.2, h: 0.6, fontSize: 15, bold: true, color: WHITE, fontFace: FONT,
    lineSpacingMultiple: 1.15,
  });
  s.addText(
    "一年两次代际事件下，迁移从专项项目变成例行作业。\n能编过 ≠ 迁移完成。",
    { x: 7.15, y: 3.42, w: 5.2, h: 0.9, fontSize: 12.5, color: PALE, fontFace: FONT, lineSpacingMultiple: 1.3 }
  );
  s.addShape(pptx.ShapeType.rect, {
    x: 0.8, y: 4.72, w: 11.73, h: 1.12, fill: { color: ACCENT }, line: { type: "none" },
  });
  s.addText(
    "两件事共用一套对象模型（契约 → 候选 → 证据 → 决策），并共用同一个架构前置项：诊断与约束能否回指用户写的那一行源码。\n" +
      "这一条不解决，两个方向都只能做到一半——因此它是 Q1 必须拍板的第一件事。",
    {
      x: 1.02, y: 4.72, w: 11.3, h: 1.12, fontSize: 12.5, bold: true, color: WHITE,
      fontFace: FONT, valign: "middle", lineSpacingMultiple: 1.3,
    }
  );
  s.addText("下一步：Q1 专项访谈 + 现状基线 + 五件拍板事项　·　" + DATE, {
    x: 0.8, y: 6.2, w: 11.7, h: 0.4, fontSize: 12, color: DIM, fontFace: FONT,
  });
  notes(
    s,
    "收束页只讲两句：生成会商品化、验收不会；两件事共用一个架构前置项。其余交给讨论。"
  );
}

// ---------- 保存 ----------
const out = path.join(
  process.argv[2] || ".",
  "PyPTO_2027_UX规划_五看三定_汇报.pptx"
);
pptx.writeFile({ fileName: out }).then(() => {
  console.log("SAVED:", out, "slides:", pageNo + 2);
});
