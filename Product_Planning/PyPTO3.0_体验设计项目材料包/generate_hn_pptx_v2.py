# -*- coding: utf-8 -*-
"""生成《昇腾 Hardware-Native 阶段汇报稿 V2》PPT（11 页 = 封面 + HN-01~09 + 收束）。

内容严格取自 Hardware-Native_阶段汇报稿_V2_20261101.md，证据底稿为
Hardware-Native_昇腾硬件事实与开发决策_研究报告_20261031.md。
报错原文、代码与数字均摘自 github_issues/pto/pypto_issues.json 对应 Issue 正文，未经改写。

设计令牌与 helper 复用 Competitive_Analysis/generate_pptx.py，保持仓库内 PPT 视觉一致。

用法：
    python Product_Planning/PyPTO3.0_体验设计项目材料包/generate_hn_pptx_v2.py
依赖：
    pip install python-pptx
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---------- 设计令牌（与 Competitive_Analysis/generate_pptx.py 一致） ----------
PRIMARY = RGBColor(0x1F, 0x38, 0x64)   # 深蓝
ACCENT  = RGBColor(0x2E, 0x75, 0xB6)   # 蓝
LIGHT   = RGBColor(0xF2, 0xF4, 0xF8)   # 浅灰底
BAND    = RGBColor(0xE8, 0xEE, 0xF6)   # 表头浅蓝
GREY    = RGBColor(0x59, 0x59, 0x59)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
RED     = RGBColor(0xC0, 0x00, 0x00)   # 警示 / 错误归因
GREEN   = RGBColor(0x2E, 0x7D, 0x32)   # 证据充分
AMBER   = RGBColor(0xB7, 0x6E, 0x00)   # 待验证 / 证据不足
CODEBG  = RGBColor(0x1B, 0x1F, 0x2A)   # 代码块底
CODEFG  = RGBColor(0xE6, 0xEA, 0xF2)   # 代码块字
FONT    = "Microsoft YaHei"
MONO    = "Consolas"

FOOTER = "昇腾 Hardware-Native 阶段汇报稿 V2 · 报错与代码摘自 Issue 原文，未经改写"
DATE   = "2026-11-01"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def set_font(run, size=14, bold=False, color=GREY, font=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    # 同时设置 latin 与 ea（CJK）字体，确保中文显示
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def add_rect(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, text, size=14, bold=False, color=GREY,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT, line_spacing=1.05):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(1); tf.margin_bottom = Pt(1)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    r = p.add_run(); r.text = text
    set_font(r, size, bold, color, font)
    return tb


def add_bullets(slide, x, y, w, h, items, size=14, gap=6, base_color=GREY):
    """items: list of (text, level) 或 (text, level, color, bold)。

    注意只使用 2 元组或 4 元组——3 元组会被原 helper 误解析。
    """
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(4); tf.margin_right = Pt(4)
    first = True
    for it in items:
        if len(it) == 2:
            text, level = it
            color, bold = base_color, False
        else:
            text, level, color, bold = it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(gap)
        p.line_spacing = 1.08
        marker = "•  " if level == 0 else "–  "
        r = p.add_run(); r.text = marker + text
        set_font(r, size - level, bold, color)
    return tb


def add_code(slide, x, y, w, h, lines, size=10.5, fg=CODEFG, bg=CODEBG):
    """等宽代码 / 报错块。lines 为字符串列表。"""
    add_rect(slide, x, y, w, h, bg)
    tb = slide.shapes.add_textbox(x + Pt(6), y + Pt(4), w - Pt(12), h - Pt(8))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.12
        p.space_after = Pt(0)
        r = p.add_run(); r.text = line
        set_font(r, size, False, fg, MONO)
    return tb


def header(slide, title, kicker=None):
    add_rect(slide, 0, 0, SW, Inches(1.02), PRIMARY)
    add_rect(slide, 0, Inches(1.02), SW, Pt(3), ACCENT)
    if kicker:
        add_text(slide, Inches(0.5), Inches(0.11), Inches(12), Inches(0.3),
                 kicker, size=11, bold=True, color=RGBColor(0xBD, 0xD3, 0xEE))
    add_text(slide, Inches(0.5), Inches(0.37) if kicker else Inches(0.27),
             Inches(12.3), Inches(0.6), title, size=23, bold=True, color=WHITE)
    add_text(slide, Inches(0.5), SH - Inches(0.3), Inches(10.5), Inches(0.24),
             FOOTER, size=9, color=GREY)
    add_text(slide, SW - Inches(1.3), SH - Inches(0.3), Inches(0.9), Inches(0.24),
             DATE, size=9, color=GREY, align=PP_ALIGN.RIGHT)


def insight(slide, text, y=Inches(6.28), color=ACCENT):
    """页面底部的一句核心洞察。"""
    add_rect(slide, Inches(0.5), y, Inches(12.33), Inches(0.82), BAND)
    add_rect(slide, Inches(0.5), y, Pt(4), Inches(0.82), color)
    add_text(slide, Inches(0.72), y + Pt(5), Inches(12.0), Inches(0.72),
             text, size=12, bold=True, color=PRIMARY,
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.15)


def note_line(slide, x, y, w, text, color=AMBER, size=10):
    add_text(slide, x, y, w, Inches(0.3), text, size=size, color=color, bold=True)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def style_table(table, header_fill=BAND, header_color=PRIMARY, body_size=11,
                header_size=11, col_colors=None):
    for j, cell in enumerate(table.rows[0].cells):
        cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Pt(4); cell.margin_right = Pt(4)
        cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                set_font(r, header_size, True, header_color)
    for i in range(1, len(table.rows)):
        for j, cell in enumerate(table.rows[i].cells):
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Pt(4); cell.margin_right = Pt(4)
            cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    c = GREY
                    if col_colors and j in col_colors:
                        c = col_colors[j]
                    set_font(r, body_size, False, c)


def add_table(slide, x, y, w, h, headers, rows, col_widths=None,
              body_size=11, header_size=11, col_colors=None):
    gfx = slide.shapes.add_table(len(rows) + 1, len(headers), x, y, w, h)
    table = gfx.table
    if col_widths:
        total = sum(col_widths)
        for j, cw in enumerate(col_widths):
            table.columns[j].width = Emu(int(w * cw / total))
    for j, htext in enumerate(headers):
        table.cell(0, j).text = htext
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            table.cell(i + 1, j).text = "" if val is None else str(val)
    style_table(table, body_size=body_size, header_size=header_size,
                col_colors=col_colors)
    return table


def new_slide():
    return prs.slides.add_slide(BLANK)


# =========================================================
# S1 — 封面
# =========================================================
s = new_slide()
add_rect(s, 0, 0, SW, SH, PRIMARY)
add_rect(s, 0, Inches(4.5), SW, Pt(3), ACCENT)
add_text(s, Inches(0.8), Inches(1.35), Inches(11.7), Inches(1.1),
         "昇腾 Hardware-Native", size=44, bold=True, color=WHITE)
add_text(s, Inches(0.8), Inches(2.5), Inches(11.7), Inches(0.7),
         "硬件事实如何进入开发决策", size=22, color=RGBColor(0xBD, 0xD3, 0xEE))
add_text(s, Inches(0.8), Inches(3.35), Inches(11.7), Inches(0.5),
         "PyPTO3.0 体验设计项目 · 阶段汇报前段洞察 · V2", size=14,
         color=RGBColor(0x9D, 0xB8, 0xD8))
add_text(s, Inches(0.8), Inches(4.85), Inches(11.7), Inches(1.5),
         "本专题回答两个问题：为什么 Hardware-Native 是本次 UX 项目必须关注的体验方向，"
         "以及从中识别出了哪些工具机会。\n"
         "四个案例的报错原文、代码与数字均摘自社区 Issue 正文，未经改写；证据边界集中在末页前一节。",
         size=13, color=RGBColor(0xD6, 0xE2, 0xF1), line_spacing=1.3)
add_text(s, Inches(0.8), Inches(6.65), Inches(11.7), Inches(0.4),
         "PyPTO3.0 体验设计项目材料包 · " + DATE, size=12,
         color=RGBColor(0x9D, 0xB8, 0xD8))
notes(s, "本稿替代 20261031 版 8 页稿。旧稿 HN-06 整页结论（CSA 第二个 AIV 未分摊有效计算）"
         "经核对已不成立，旧稿与同名 pptx 不应继续使用。")

# =========================================================
# S2 — HN-01 命题
# =========================================================
s = new_slide()
header(s, "昇腾开发的关键挑战：计算怎么写，只决定了一半", "HN-01 ｜ 命题")
add_text(s, Inches(0.5), Inches(1.3), Inches(12.3), Inches(0.4),
         "模型与算子开发从计算意图出发。但在昇腾上，计算表达本身并没有决定它怎么执行。",
         size=14, color=PRIMARY, bold=True)
add_text(s, Inches(0.5), Inches(1.82), Inches(12.3), Inches(0.3),
         "另一半是四个必须落实的决定：", size=13)
add_bullets(s, Inches(0.5), Inches(2.18), Inches(5.9), Inches(1.7), [
    ("这段数据放哪块片上空间", 0),
    ("切多大", 0),
    ("谁来算", 0),
    ("什么时候交接", 0),
], size=13.5, gap=7)
add_rect(s, Inches(6.7), Inches(2.12), Inches(6.13), Inches(1.75), LIGHT)
add_text(s, Inches(6.92), Inches(2.26), Inches(5.7), Inches(0.3),
         "这些规则来自硬件", size=12.5, bold=True, color=PRIMARY)
add_text(s, Inches(6.92), Inches(2.62), Inches(5.7), Inches(1.1),
         "它们写在编译器的校验里、后端的容量表里、ISA 的契约里。\n\n"
         "但它们进入开发者视野的方式，通常是一条编译失败，或者一次设备超时。",
         size=12, line_spacing=1.25)
add_rect(s, Inches(0.5), Inches(4.08), Inches(12.33), Inches(1.0), BAND)
add_rect(s, Inches(0.5), Inches(4.08), Pt(4), Inches(1.0), PRIMARY)
add_text(s, Inches(0.72), Inches(4.2), Inches(2.0), Inches(0.3),
         "工作定义", size=11.5, bold=True, color=PRIMARY)
add_text(s, Inches(0.72), Inches(4.5), Inches(11.9), Inches(0.5),
         "软件系统在抽象、映射、调度和执行时尊重目标硬件的能力、约束与成本；"
         "并且当某条硬件约束必须由开发者参与处理时，系统让这个决定可理解、可操作、可验证。",
         size=12.5, color=PRIMARY, line_spacing=1.2)
add_text(s, Inches(0.5), Inches(5.3), Inches(12.3), Inches(0.6),
         "两句话缺一不可。只有前半句是「硬件亲和优化」，那是编译器的工作；"
         "只有后半句是把复杂度甩给用户。", size=12.5, color=GREY)
insight(s, "核心洞察：我们关注的体验问题是——当硬件约束影响开发选择时，工具能否帮用户理解影响并作出决定。")
notes(s, "定义是本项目的工作定义，不是行业统一标准。\n"
         "如被问「这是不是编译器团队的事」：回到 HN-08 的责任层判据——四个案例里三个的首要责任"
         "确实在技术侧，这一点我们在 HN-08 明确承认并给出了 UX 的正当职责。")

# =========================================================
# S3 — HN-02 硬件
# =========================================================
s = new_slide()
header(s, "先把硬件讲清楚：三个数字，一条不能绕的路", "HN-02 ｜ 背景")
add_text(s, Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35),
         "昇腾的片上存储不是层级缓存，而是一组用途固定的专用空间——这是与 GPU 最大的心智差异。",
         size=13.5, bold=True, color=PRIMARY)
add_table(s, Inches(0.5), Inches(1.75), Inches(7.1), Inches(2.6),
          ["PyPTO 空间", "硬件", "用途", "Ascend910B"],
          [["Left", "L0A", "Matmul 左操作数", "64 KiB"],
           ["Right", "L0B", "Matmul 右操作数", "64 KiB"],
           ["Acc", "L0C", "Matmul 累加器", "128 KiB"],
           ["Vec", "UB", "Vector 计算、Tile load", "188,416 B 有效"],
           ["Mat", "L1", "Cube 操作数暂存", "512 KiB"]],
          col_widths=[1.5, 1.0, 2.7, 1.9], body_size=10.5, header_size=10.5)
add_text(s, Inches(7.9), Inches(1.75), Inches(4.9), Inches(0.3),
         "一条不能绕的路", size=12.5, bold=True, color=PRIMARY)
add_code(s, Inches(7.9), Inches(2.1), Inches(4.93), Inches(0.52),
         ["DDR -> Vec / Mat -> Left / Right / Acc"], size=11.5)
add_text(s, Inches(7.9), Inches(2.76), Inches(4.9), Inches(0.9),
         "tile.load 到不了 L0。一次 matmul 的操作数至少经过两跳，"
         "中间那一跳由编译器自动插入 tile.move。",
         size=11.5, line_spacing=1.25)
add_text(s, Inches(7.9), Inches(3.7), Inches(4.9), Inches(0.3),
         "还有两个数字要记住", size=12.5, bold=True, color=PRIMARY)
add_bullets(s, Inches(7.9), Inches(4.02), Inches(4.93), Inches(1.5), [
    ("Ascend910B 有 48 个物理 AIV 核", 0, PRIMARY, True),
    ("AIC 与 AIV 交接数据要经 GM 中转，预留的缓冲从 UB 的 188,416 里直接扣掉——"
     "用户代码里没有任何一行提到它", 0, GREY, False),
], size=11.5, gap=7)
add_rect(s, Inches(0.5), Inches(4.55), Inches(7.1), Inches(1.5), LIGHT)
add_text(s, Inches(0.72), Inches(4.72), Inches(6.7), Inches(1.2),
         "这一页不是硬件科普。\n\n后面三个案例的报错，全部由这几个数字构成。",
         size=13, color=PRIMARY, bold=True, line_spacing=1.3)
insight(s, "核心洞察：这些不是背景知识，而是开发者每天会撞到的边界。工具要么帮他理解这些边界，要么让他在边界上反复试错。")
notes(s, "容量数字绑定 Ascend910B 及报告时的工具链版本。\n"
         "Ascend950 的 L0C 是 256 KiB、访存粒度从 512 B 变为 128 B——"
         "所以这些数字必须从目标档案查询，不能写死在前端。\n"
         "Vec 的 188,416 B 来自 #2287 报错原文中编译器报告的 platform limit。")

# =========================================================
# S4 — HN-03 框架已承担
# =========================================================
s = new_slide()
header(s, "但大部分硬件复杂性，框架已经承担了", "HN-03 ｜ 排除错误方向")
add_text(s, Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.35),
         "PyPTO 的内存配置不是在某个 API 里一次确定的，而是沿编译流水线逐步落实：",
         size=13.5, bold=True, color=PRIMARY)
add_code(s, Inches(0.5), Inches(1.72), Inches(6.5), Inches(2.95), [
    "Tensor / Tile 语义",
    "  -> InferTileMemorySpace     推导 Tile 所在空间",
    "  -> AutoTileMatmulL0         按 L0 容量自动选 m/n/k",
    "  -> ResolveBackendOpLayouts  修复后端要求的布局",
    "  -> ExpandMixedKernel        划分 AIC/AIV 与跨核边界",
    "  -> LowerPipelineLoops       生成多 stage / ping-pong",
    "  -> MemoryReuse              按生命周期与别名规则复用",
    "  -> AllocateMemoryAddr       分配实际片上地址",
    "  -> codegen / PTOAS",
], size=10.5)
add_text(s, Inches(7.3), Inches(1.72), Inches(5.5), Inches(0.3),
         "框架已经在做的事", size=12.5, bold=True, color=PRIMARY)
add_bullets(s, Inches(7.3), Inches(2.04), Inches(5.53), Inches(2.6), [
    ("按 L0 容量自动选分块", 0),
    ("自动插入跨空间搬运", 0),
    ("自动划分 AIC/AIV 边界", 0),
    ("按生命周期复用缓冲", 0),
    ("维护「哪些算子不能原地复用」的清单", 0),
], size=12, gap=7)
add_rect(s, Inches(0.5), Inches(4.85), Inches(12.33), Inches(1.28), LIGHT)
add_text(s, Inches(0.72), Inches(5.0), Inches(11.9), Inches(0.35),
         "所以体验问题不是「用户需要学会这些」。", size=14, bold=True, color=PRIMARY)
add_text(s, Inches(0.72), Inches(5.42), Inches(11.9), Inches(0.6),
         "真正的体验问题只在剩下那一小部分：框架处理不了、处理错了，"
         "或者处理了但用户必须知道结果的时候。", size=13.5, bold=True, color=RED)
insight(s, "核心洞察：Hardware-Native 做对了的样子，可能是用户看到的硬件细节更少。"
           "衡量标准不是用户理解了多少硬件，而是用户不需要理解多少硬件仍能把事做对。")
notes(s, "这一页直接排除「做一个硬件工作台 / 芯片展板」的方案。\n"
         "如有人提议增加硬件可视化，用 HN-08 要求六回应——先问这件事为什么不能由框架决定。\n"
         "昇腾官方的 Ascend C 用 TPipe/TQue 承载复杂性，也是同一个取向。")

# =========================================================
# S5 — HN-04 归因
# =========================================================
s = new_slide()
header(s, "洞察一：信息不缺，缺的是归因", "HN-04 ｜ 案例 PyPTO #2287")
add_code(s, Inches(0.5), Inches(1.28), Inches(12.33), Inches(1.42), [
    "ValueError: Vec buffer usage (213760 bytes) exceeds platform limit (188416 bytes).",
    "The first 131072 bytes of that space are reserved by system.reserve_buffer, so tiles are",
    "allocated above them - this is the cross-core pipe ring. Lower its depth with",
    "optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the",
    "tile that crosses the cube/vector boundary",
], size=10)
add_text(s, Inches(0.5), Inches(2.82), Inches(6.1), Inches(0.62),
         "先肯定它。用量、上限、超出部分的归属、预留大小、两条修复建议——"
         "按任何诊断设计规范衡量，这都是一条好报错。", size=12, color=GREEN)
add_text(s, Inches(0.5), Inches(3.52), Inches(6.1), Inches(0.3),
         "然后看真相。两条建议都是错的。", size=14, bold=True, color=RED)
add_table(s, Inches(0.5), Inches(3.9), Inches(6.1), Inches(1.0),
          ["方向", "消费者 Tile", "字节"],
          [["V2C（AIV→AIC）", "staged KV tile", "131,072"],
           ["C2V（AIC→AIV）", "qk_raw [32,128] FP32", "16,384"]],
          col_widths=[2.2, 2.6, 1.3], body_size=10.5, header_size=10.5)
add_text(s, Inches(0.5), Inches(5.0), Inches(6.1), Inches(0.95),
         "编译器把两个方向的缓冲都按双向最大值预留。于是 16 KB 的载荷，"
         "在最紧张的 UB 里占了 131,072 字节的坑——8 倍过量。\n"
         "正确尺寸应为 32,768 字节，总占用 115,456 / 188,416，轻松装下。",
         size=11.5, line_spacing=1.25)
add_bullets(s, Inches(6.9), Inches(2.82), Inches(5.93), Inches(1.6), [
    ("建议一「降低环深度」：报告实测，强制最小深度后预留仍是 131,072，依然超 25,344 字节。",
     0, RED, False),
    ("建议二「缩小跨边界的 Tile」：让开发者为编译器的缺陷付出真实性能代价。",
     0, RED, False),
], size=11.5, gap=8)
add_rect(s, Inches(6.9), Inches(4.5), Inches(5.93), Inches(1.45), LIGHT)
add_text(s, Inches(7.12), Inches(4.65), Inches(5.5), Inches(0.3),
         "最刺眼的一点", size=12, bold=True, color=RED)
add_text(s, Inches(7.12), Inches(4.98), Inches(5.5), Inches(0.85),
         "报错里说的「前 131,072 字节被预留」，恰恰就是那个算错的数。"
         "工具把缺陷的产物当作既定事实报告给了用户。",
         size=12, color=PRIMARY, bold=True, line_spacing=1.25)
insight(s, "核心洞察：一条诊断的价值 = 归因正确率 × 可操作性，不是两者之和。"
           "#2287 可操作性接近满分、归因错误，于是乘积为负——它主动把开发者引向了错误的修复。", color=RED)
notes(s, "commit 7d743e8d，Ascend 910B / A2A3，Issue 已关闭为 not planned——关闭不自证修复。\n"
         "产品含义：归因不确定时，降低建议的确定性比提高建议的具体性更重要。\n"
         "交叉验证：CSA 编译产物中 qk_raw 为 [32,128] FP32，32×128×4 = 16,384，与报告一致。")

# =========================================================
# S6 — HN-05 同报错不同责任
# =========================================================
s = new_slide()
header(s, "洞察二：同样的报错，完全不同的责任", "HN-05 ｜ 案例 #630 vs #626")
add_code(s, Inches(0.5), Inches(1.28), Inches(12.33), Inches(0.72), [
    "#630:  Right buffer usage (98304 bytes)  exceeds platform limit (65536 bytes)",
    "#626:  Right buffer usage (131072 bytes) exceeds platform limit (65536 bytes)",
], size=11)
add_text(s, Inches(0.5), Inches(2.12), Inches(12.3), Inches(0.3),
         "同样的空间、同样的上限、同样的措辞、同样的 pass。但责任层完全不同。",
         size=13, bold=True, color=PRIMARY)
# 左：#630
add_rect(s, Inches(0.5), Inches(2.5), Inches(6.1), Inches(2.5), LIGHT)
add_text(s, Inches(0.72), Inches(2.62), Inches(5.7), Inches(0.3),
         "#630 ｜ 真的放不下，但根因是抽象退化", size=12, bold=True, color=AMBER)
add_text(s, Inches(0.72), Inches(2.95), Inches(5.7), Inches(0.42),
         "kv_lora_rank 从 128 改到 512，只改了一个模型维度。", size=11.5)
add_code(s, Inches(0.72), Inches(3.4), Inches(5.66), Inches(0.5),
         ["set_cube_tile_shape([M_TILE, K_TILE, N_TILE])   # PTO 2.0"], size=9.5)
add_text(s, Inches(0.72), Inches(3.98), Inches(5.7), Inches(0.92),
         "PTO 3.0 取消了这个接口，同一件事变成手写三层循环加逐块处理；"
         "RMSNorm 还要自己处理跨 chunk 的 reduce。提报人直接要求恢复一个"
         "声明式接口 @pl.tile_policy(...)。", size=11, line_spacing=1.22)
# 右：#626
add_rect(s, Inches(6.9), Inches(2.5), Inches(5.93), Inches(2.5), LIGHT)
add_text(s, Inches(7.12), Inches(2.62), Inches(5.5), Inches(0.3),
         "#626 ｜ 根本不需要用户改任何东西", size=12, bold=True, color=RED)
add_text(s, Inches(7.12), Inches(2.95), Inches(5.5), Inches(0.42),
         "编译器的容量校验用了不考虑生命周期的高水位算法：", size=11.5)
add_code(s, Inches(7.12), Inches(3.4), Inches(5.49), Inches(0.5),
         ["high_water = max(addr + size)"], size=9.5)
add_text(s, Inches(7.12), Inches(3.98), Inches(5.5), Inches(0.92),
         "把两个生命周期可以错开的 64 KiB 变量直接累加成 131,072，"
         "拒绝了一个合法的 kernel。", size=11, line_spacing=1.22)
add_rect(s, Inches(0.5), Inches(5.12), Inches(12.33), Inches(1.0), BAND)
add_rect(s, Inches(0.5), Inches(5.12), Pt(4), Inches(1.0), RED)
add_text(s, Inches(0.72), Inches(5.24), Inches(11.9), Inches(0.82),
         "开发者没有任何办法从报错里区分这两种情况。\n"
         "面对 #626，一个尽责的开发者会去缩小 Tile。他会成功——因为缩小后确实能通过校验。"
         "代价是白白损失性能，而真正的缺陷留在原地。",
         size=12.5, bold=True, color=PRIMARY, line_spacing=1.22)
insight(s, "核心洞察：归因错误不只浪费时间，它会制造错误的修复并掩盖真正的缺陷。"
           "每条诊断必须携带责任层——确认是用户配置问题、确认是系统问题、或尚不能区分。", color=RED)
notes(s, "#630 仍 open，#626 已关闭。\n"
         "#630 会反驳一个常见直觉——面对容量报错，产品的第一反应是「做更好的容量可视化」，"
         "但提报人要的是把决定还给框架。这正好支撑 HN-03。\n"
         "「尚不能区分」这个状态是诚实的也是有用的：它告诉用户该去提 Issue，而不是继续试参数。")

# =========================================================
# S7 — HN-06 约束前移
# =========================================================
s = new_slide()
header(s, "洞察三：本可以静态检查的，代价是一次设备重置", "HN-06 ｜ 案例 PyPTO #1935")
add_code(s, Inches(0.5), Inches(1.28), Inches(6.1), Inches(1.15), [
    "N = 24   # < 48 物理 AIV -> 部分占用",
    "",
    'pl.system.syncall(core_type="aiv_only")   # HARD barrier',
    "with pl.spmd(N):                          # N=24 -> 部分占用",
], size=10)
add_code(s, Inches(6.9), Inches(1.28), Inches(5.93), Inches(1.15), [
    "[ERROR] AICore error 507018: bounded device",
    "        drain failed (force reset will follow)",
    "RuntimeError: run_prepared failed with code 507018",
], size=10)
add_text(s, Inches(0.5), Inches(2.56), Inches(12.3), Inches(0.3),
         "设备随后不可用，后续任务报 code 13 直到恢复。这是共享开发设备，影响不止一个人。",
         size=12.5, bold=True, color=RED)
add_text(s, Inches(0.5), Inches(2.95), Inches(12.3), Inches(0.58),
         "根因：hard 屏障（FFTS）等待该核型的每一个物理核到达。910B 有 48 个 AIV，"
         "启动了 24 个 block，另外 24 个核永远不会到达，等待永不完成。\n"
         "pl.spmd(24) → 507018；只把 24 改成 48 → PASS。占用度是唯一变量。",
         size=12, line_spacing=1.25)
add_text(s, Inches(0.5), Inches(3.68), Inches(12.3), Inches(0.3),
         "三件让这个案例特别刺眼的事", size=12.5, bold=True, color=PRIMARY)
add_bullets(s, Inches(0.5), Inches(4.0), Inches(12.33), Inches(1.2), [
    ("编译期完全可判定。编译器同时掌握 pl.spmd(N)（写在源码里）和该 SoC 每种核型的物理核数"
     "（在后端参数表里）。这是一次静态整数比较。", 0),
    ("这条知识当时已经在仓库里。它写在 tests/st/runtime/cross_core/test_syncall.py 的 docstring 里"
     "——一个开发者写代码时不会打开的文件。", 0),
    ("替代方案也已经存在。soft 模式在部分占用下工作正常。开发者需要的不是新能力，"
     "只是在选 hard 的那一刻知道它有这个前提。", 0),
], size=11.5, gap=5)
add_rect(s, Inches(0.5), Inches(5.42), Inches(12.33), Inches(0.72), LIGHT)
add_text(s, Inches(0.72), Inches(5.5), Inches(11.9), Inches(0.6),
         "提报人自己写出了理想的诊断：「requires the spmd launch to fill all 48 AIV cores, "
         "but pl.spmd(24) launches 24 blocks. Use mode=\"soft\" for partial occupancy.」"
         "——违反了什么约束、具体数字对比、可执行的替代方案，三件齐全。",
         size=11.5, color=PRIMARY, line_spacing=1.2)
insight(s, "核心洞察：代价不对称到了荒谬的程度——一次静态整数比较，对应一次设备强制重置。"
           "这是时机错误，不是信息缺失。凡是编译期可判定的约束，都应该在编译期拒绝。")
notes(s, "这一页首先是编译器的工作项，不是界面需求——务必主动说明，否则会被正确地质疑。\n"
         "UX 在这里的正当职责有两条：参与设计这条诊断怎么写；"
         "以及让 hard/soft 的契约差异在用户写下 mode= 的时刻可见（编辑器集成），"
         "而不是补一篇文档让用户去记。\n"
         "环境：pypto d598b41b、pto-isa e722679、ptoas 0.48、Ascend 910B。Issue 已关闭。")

# =========================================================
# S8 — HN-07 自证
# =========================================================
s = new_slide()
header(s, "洞察四：连我们自己，也读错了编译产物", "HN-07 ｜ 证据修订")
add_text(s, Inches(0.5), Inches(1.28), Inches(12.3), Inches(0.52),
         "旧稿的主张：据 CSA 编译产物断言 qk_pv_aiv 双 AIV 派发，但第二条分支的 softmax Tile "
         "valid_shape=[0,0]，据此识别出「有效计算未分摊到第二个 AIV」，"
         "并把计算分工列为首个试点任务。", size=12.5, line_spacing=1.25)
add_text(s, Inches(0.5), Inches(1.92), Inches(12.3), Inches(0.3),
         "本轮直接核对那份产物，结论是这个读法站不住：", size=13, bold=True, color=PRIMARY)
add_bullets(s, Inches(0.5), Inches(2.26), Inches(12.33), Inches(1.5), [
    ("dual_aiv_dispatch 属性确实存在（第 2952 行）——这部分属实。", 0, GREEN, False),
    ("但两个 subblock 分支的迭代次数由同一个表达式决定："
     "(qk_items - qk_core + 23) // 24（第 2964、3025 行）。从产物看不出哪一支不干活。",
     0, RED, False),
    ("valid_shape=[0, 0] 在该文件出现 68 次，遍布不同函数、不同分支，包括明显在做实际计算的 Tile。"
     "它是 TileView 的默认哨兵标注，不能按字面读成「有效数据为空」。", 0, RED, False),
], size=11.5, gap=6)
add_rect(s, Inches(0.5), Inches(3.92), Inches(12.33), Inches(0.62), BAND)
add_rect(s, Inches(0.5), Inches(3.92), Pt(4), Inches(0.62), RED)
add_text(s, Inches(0.72), Inches(4.02), Inches(11.9), Inches(0.45),
         "「第二个 AIV 没有分摊有效计算」这个判断，在当前证据下不成立。"
         "它可能仍然为真，但需要别的证据——分支实际执行次数、profiling、"
         "或对 set_validshape 调用链的完整追踪。",
         size=12.5, bold=True, color=PRIMARY, line_spacing=1.2)
add_rect(s, Inches(0.5), Inches(4.72), Inches(12.33), Inches(1.4), LIGHT)
add_text(s, Inches(0.72), Inches(4.86), Inches(11.9), Inches(0.3),
         "为什么把这件事放进汇报", size=12.5, bold=True, color=PRIMARY)
add_text(s, Inches(0.72), Inches(5.2), Inches(11.9), Inches(0.82),
         "读错这个字段的，不是赶工期的开发者。是在专门研究这个主题、有充足时间、"
         "直接打开了编译产物的研究者。",
         size=13.5, bold=True, color=RED, line_spacing=1.22)
insight(s, "核心洞察：如果连我们都会读错，那么「把硬件事实摆出来让开发者自己判断」就不是可行的设计策略。"
           "原始产物是必要的证据底座，但它不能充当界面。", color=RED)
notes(s, "这一页也修正了一个机会点的证据状态——计算分工目前缺少确凿的昇腾案例证据，"
         "因此在 HN-09 的推进顺序里从「首个试点」降到第 4 位。\n"
         "主动讲这个修订会增加整份材料的可信度，不要回避。\n"
         "工具的价值恰在这里：把需要上下文才能解读的字段，翻译成「这个 Tile 的有效范围是 X，依据是 Y」"
         "——或者在无法确定时，明确说「此处该字段不可解读」。")

# =========================================================
# S9 — HN-08 机会点
# =========================================================
s = new_slide()
header(s, "机会点：按「谁来决定」分层，而不是按硬件机制分类", "HN-08 ｜ 机会点")
add_text(s, Inches(0.5), Inches(1.26), Inches(12.3), Inches(0.3),
         "四个案例指向同一个结构问题：一条硬件事实该不该进界面，取决于谁来决定它。",
         size=13, bold=True, color=PRIMARY)
add_table(s, Inches(0.5), Inches(1.62), Inches(12.33), Inches(1.9),
          ["责任层", "含义", "本专题案例"],
          [["L1 系统保证", "系统必须正确处理，不应进入用户视野",
            "#1935 的满占用检查、#626 的生命周期校验、#2287 的分方向预留"],
           ["L2 必须可见", "系统处理了，但用户需要知道结果",
            "自动插入的 tile.move、跨核管道的 UB 预留、自动选中的 m/n/k"],
           ["L3 必须可控", "用户需要表达系统无法推断的意图",
            "#630 要的 tile_policy、#2287 要的 local_slot_num、hard/soft 选择"],
           ["L4 必须可验证", "用户需要确认结果符合预期",
            "改动后的实际分配、数值结果、目标链路耗时"]],
          col_widths=[1.9, 4.0, 6.4], body_size=10.5, header_size=11)
add_rect(s, Inches(0.5), Inches(3.64), Inches(6.1), Inches(1.28), BAND)
add_rect(s, Inches(0.5), Inches(3.64), Pt(4), Inches(1.28), PRIMARY)
add_text(s, Inches(0.72), Inches(3.74), Inches(5.7), Inches(0.3),
         "一条评审可直接使用的判据", size=11.5, bold=True, color=PRIMARY)
add_text(s, Inches(0.72), Inches(4.06), Inches(5.7), Inches(0.78),
         "任何 Hardware-Native 设计提案，先问它属于哪一层。"
         "如果答案是 L1，它是技术任务，不是设计任务。",
         size=12.5, bold=True, color=PRIMARY, line_spacing=1.22)
add_text(s, Inches(6.9), Inches(3.64), Inches(5.93), Inches(1.28),
         "我们主动说明：四个案例里，三个的首要责任在 L1。\n"
         "这不削弱这个方向，而是界定了它。UX 在 L1 问题上的正当工作是：参与设计那条诊断"
         "（说什么、在哪说、给什么出路），以及在修复落地前组织用户的判断——"
         "不是用提示框替代修复。", size=11.5, line_spacing=1.25)
add_table(s, Inches(0.5), Inches(5.02), Inches(12.33), Inches(1.1),
          ["决策对象", "核心问题", "证据状态"],
          [["形态与容量", "数据怎么放", "充分（#630、#626、#2287）"],
           ["执行时序", "什么时候交接", "充分（#1935）"],
           ["数据交接", "数据怎么走", "跨核有强证据；Cube↔Vector 融合缺真实任务"],
           ["计算分工", "谁来算", "不足（见 HN-07）"]],
          col_widths=[2.2, 2.6, 7.5], body_size=10.5, header_size=11,
          col_colors={2: AMBER})
insight(s, "核心洞察：机会点不是四个硬件领域，而是一张「责任层 × 决策对象」的表。"
           "每个格子对应一个具体设计动作——例如「执行时序 × L1」就是「哪些同步约束必须编译期拒绝」。")
notes(s, "这套分类替代了旧稿的九域、六承载形态、四类硬件机制和 O1–O4 四套并行分类。\n"
         "按硬件机制分类是技术视角，它无法回答「该不该进界面」；按「谁来决定」可以。\n"
         "设计要求最强三条：诊断必须标注责任层（#626 vs #630）；"
         "资源占用必须按来源拆分并标明每项由谁决定（#2287）；"
         "先问能不能自动再问怎么呈现（#630）。完整七条见研究报告第五部分。")

# =========================================================
# S10 — HN-09 验证
# =========================================================
s = new_slide()
header(s, "怎么证明这个方向值得投", "HN-09 ｜ 验证计划")
add_text(s, Inches(0.5), Inches(1.26), Inches(12.3), Inches(0.3),
         "可证伪的假设（每条都配了放弃条件）", size=12.5, bold=True, color=PRIMARY)
add_table(s, Inches(0.5), Inches(1.6), Inches(12.33), Inches(2.1),
          ["假设", "验证任务", "什么情况下应当放弃"],
          [["H1 标注责任层能减少错误修复", "给出 #626 与 #630 两条文本相同的报错，让开发者判断该改什么",
            "开发者本来就能区分"],
           ["H2 拆分占用来源能让用户找对对象", "#2287 场景：判断该缩 Tile 还是报 Issue",
            "现有报错已足够"],
           ["H3 约束前移能减少无效设备实验", "#1935 场景：离线判断 hard/soft 是否匹配",
            "编译器加检查后问题消失，界面无额外价值"],
           ["H4 自动化比可视化更有价值", "#630 场景：对比「更好的容量可视化」与「恢复 tile_policy」",
            "用户确实想要手工控制"],
           ["H5 翻译硬件字段能减少误读", "HN-07 场景：从编译产物判断双 AIV 分工",
            "原始字段配文档已够用"]],
          col_widths=[3.3, 5.3, 3.7], body_size=10, header_size=10.5)
add_rect(s, Inches(0.5), Inches(3.82), Inches(6.1), Inches(0.92), BAND)
add_rect(s, Inches(0.5), Inches(3.82), Pt(4), Inches(0.92), ACCENT)
add_text(s, Inches(0.72), Inches(3.92), Inches(5.7), Inches(0.75),
         "H4 最该先做。它直接检验本专题的反直觉结论。如果结果显示用户更想要可视化"
         "而不是自动化，HN-03 和设计要求三都需要重写。",
         size=11.5, bold=True, color=PRIMARY, line_spacing=1.2)
add_rect(s, Inches(6.9), Inches(3.82), Inches(5.93), Inches(0.92), LIGHT)
add_text(s, Inches(7.12), Inches(3.92), Inches(5.5), Inches(0.75),
         "推荐的核心指标：错误归因次数。四个案例里有三个的核心损失都由归因错误造成，"
         "而它恰好是现有工具完全没有度量的东西。",
         size=11.5, bold=True, color=PRIMARY, line_spacing=1.2)
add_table(s, Inches(0.5), Inches(4.86), Inches(12.33), Inches(1.26),
          ["顺序", "内容", "依据"],
          [["0", "Target / Finding / 证据等级的数据模型", "所有案例的共同前提"],
           ["1", "容量与占用来源拆分", "证据最充分"],
           ["2", "约束前移与编译期检查", "收益明确且可度量"],
           ["3", "执行时序与缓冲复用的解释", "与 1、2 共享对象"],
           ["4", "计算分工", "证据不足，需先补案例"]],
          col_widths=[0.8, 6.0, 5.5], body_size=10, header_size=10.5)
insight(s, "不要用的指标：面板点击数、用户看过多少硬件名词、资源填满比例、"
           "用户会不会说「Hardware-Native」。这些都不代表任务价值。")
notes(s, "推进顺序来自证据成熟度与可验证性，不是发生率或 ROI 排名"
         "——四个案例来自定向检索，不构成抽样。\n"
         "对照设计：三个版本（现状 / 加结构化诊断与责任层标注 / 再加局部硬件可视化），"
         "用难度相近的任务和顺序平衡，分别识别「更好的信息组织」和「额外可视化」各自的贡献。\n"
         "安全约束：#1935 这类问题用离线材料或已知测试用例验证，不为研究故意使共享设备挂死。")

# =========================================================
# S11 — 收束
# =========================================================
s = new_slide()
add_rect(s, 0, 0, SW, SH, PRIMARY)
add_rect(s, 0, Inches(1.5), SW, Pt(3), ACCENT)
add_text(s, Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.6),
         "阶段结论", size=30, bold=True, color=WHITE)
add_text(s, Inches(0.8), Inches(1.9), Inches(11.73), Inches(3.4),
         "昇腾上的模型与算子实现，需要同时处理专用存储、核间分工、跨核交接和异步同步。"
         "这些机制直接影响分块、放置、融合和流水选择。\n\n"
         "但真实开发记录显示，问题通常不是硬件信息缺失——报错里的数字往往是齐全的。"
         "问题是归因：同样一条「Right buffer 超限」，可能要改分块，也可能是编译器校验算错；"
         "一条措辞确定的建议，可能把开发者引向错误的修复；"
         "一次本可静态完成的整数比较，代价是设备强制重置。我们自己在读编译产物时也犯了同类错误。",
         size=14, color=RGBColor(0xD6, 0xE2, 0xF1), line_spacing=1.35)
add_rect(s, Inches(0.8), Inches(5.08), Inches(11.73), Inches(1.16), ACCENT)
add_text(s, Inches(1.02), Inches(5.2), Inches(11.3), Inches(0.95),
         "因此本阶段将 Hardware-Native 收敛为一个结构问题——"
         "一条硬件事实该不该进界面，取决于谁来决定它——并按四个责任层组织机会点。"
         "能由系统可靠处理的复杂性应当继续自动承担，且优先于任何界面改进；"
         "需要用户参与的决定，则提供明确的对象、责任归属与证据等级。",
         size=13, bold=True, color=WHITE, line_spacing=1.3)
add_text(s, Inches(0.8), Inches(6.62), Inches(11.7), Inches(0.4),
         "下一步：以容量与占用归属作为第一个试点任务 · " + DATE, size=12,
         color=RGBColor(0x9D, 0xB8, 0xD8))
notes(s, "衔接后续方案：把「模型配置 → 分块 → 物理 Tile → 空间分配 → 容量上限」这条链路显式化，"
         "并在每一环问一次「这件事能不能自动」。通过任务对照验证哪些硬件事实需要显性呈现、"
         "哪些应由系统承担。\n\n"
         "证据边界见文案稿第四节；评审常见质疑与回答见第三节（含「你们自己读错了产物，材料可信吗」）。")

# ---------- 保存 ----------
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Hardware-Native_阶段汇报_洞察与机会_V2_20261101.pptx")
prs.save(out)
print("SAVED:", out, "slides:", len(prs.slides._sldIdLst))
