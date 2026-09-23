
(function () {
  const viewportSelector = ".controlFlowExplorer_codeViewport__3FdyY";
  const tokenPattern = /(#.*$)|((?:[rubfRUBF]{0,2})(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'))|\b(def|class|if|elif|else|for|while|in|return|yield|import|from|as|try|except|finally|with|lambda|and|or|not|is|pass|break|continue|raise|assert|async|await|global|nonlocal|del)\b|\b(True|False|None)\b|\b(0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)(?=\s*\()/g;
  const escapeHtml = (value) => value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  function colorize(source) {
    let result = "";
    let cursor = 0;

    source.replace(
      tokenPattern,
      (match, comment, string, keyword, constant, number, fn, offset) => {
        result += escapeHtml(source.slice(cursor, offset));
        const className = comment
          ? "llvmcfg-syntax-comment"
          : string
            ? "llvmcfg-syntax-string"
            : keyword
              ? "llvmcfg-syntax-keyword"
              : constant
                ? "llvmcfg-syntax-constant"
                : number
                  ? "llvmcfg-syntax-number"
                  : "llvmcfg-syntax-function";
        result += '<span class="' + className + '">' + escapeHtml(match) + "</span>";
        cursor = offset + match.length;
        return match;
      }
    );

    return result + escapeHtml(source.slice(cursor));
  }

  let scheduled = false;

  function enhanceCode() {
    scheduled = false;
    document.querySelectorAll(viewportSelector + " code").forEach((code) => {
      const source = code.textContent || "";
      if (code.dataset.highlightSource === source) return;
      code.dataset.highlightSource = source;
      code.innerHTML = colorize(source);
    });

    enhanceScenarioControls();
    enhanceSemanticGraphCards();
    enhanceEdgeConditionLabels();
    syncFlowVisualComparisons();
  }

  function setText(element, value) {
    if (element && element.textContent !== value) element.textContent = value;
  }

  function setLeadingText(element, value) {
    if (!element) return;
    const textNode = Array.from(element.childNodes).find(
      (node) => node.nodeType === Node.TEXT_NODE
    );
    if (textNode) {
      if (textNode.nodeValue !== value) textNode.nodeValue = value;
      return;
    }
    element.prepend(document.createTextNode(value));
  }

  function makeElement(tagName, className, textContent) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (textContent !== undefined) element.textContent = textContent;
    return element;
  }

  function replaceStateVisual(node, signature, build) {
    let visual = node.querySelector(":scope > .llvmcfg-state-visual");
    if (visual?.dataset.signature === signature) return visual;
    visual?.remove();
    visual = build();
    visual.classList.add("llvmcfg-state-visual");
    visual.dataset.signature = signature;
    node.append(visual);
    return visual;
  }

  function buildScenarioVisual(node, inputLength, factorCount, taskCount, isObserved) {
    const visibleTokenCount = Math.min(inputLength, 32);
    const defaultFooter = isObserved
      ? factorCount + " 种展开规格有执行记录"
      : factorCount + " 种展开规格可参与本次审查";
    const signature = ["scenario", inputLength, factorCount, taskCount, isObserved].join(":");
    const visual = replaceStateVisual(node, signature, () => {
      const root = makeElement("span", "llvmcfg-scenario-visual");
      const hero = makeElement("span", "llvmcfg-scenario-hero");
      hero.append(makeElement("b", "", String(inputLength)));
      hero.append(makeElement("span", "", "TOKEN"));
      root.append(hero);
      root.append(makeElement("code", "llvmcfg-scenario-bound", "t = x_in.shape[0]"));

      const strip = makeElement("span", "llvmcfg-token-strip");
      strip.setAttribute("aria-label", inputLength + " 个待处理 token");
      for (let index = 0; index < visibleTokenCount; index += 1) {
        const cell = makeElement("i", "llvmcfg-token-cell");
        cell.setAttribute("aria-hidden", "true");
        cell.dataset.tokenIndex = String(index);
        strip.append(cell);
      }
      root.append(strip);
      const overflow = inputLength - visibleTokenCount;
      if (overflow > 0) {
        root.append(makeElement("small", "llvmcfg-scenario-footer", "另有 " + overflow + " token 未在缩略条中展开"));
      }
      root.append(makeElement("small", "llvmcfg-scenario-footer llvmcfg-scenario-summary", defaultFooter));
      return root;
    });
    visual.dataset.defaultFooter = defaultFooter;
    visual.dataset.taskEventCount = String(taskCount || 0);
  }

  function formatTaskRate(taskCount, invocationCount) {
    if (!invocationCount) return "0";
    const rate = taskCount / invocationCount;
    return Number.isInteger(rate) ? String(rate) : rate.toFixed(1);
  }

  function buildIterationGroups(inputLength, factor, groupCount, mode) {
    const groups = makeElement("span", "llvmcfg-iteration-groups");
    const visibleGroupCount = Math.min(groupCount, 16);
    groups.style.setProperty("--group-columns", String(Math.min(visibleGroupCount || 1, 4)));

    for (let groupIndex = 0; groupIndex < visibleGroupCount; groupIndex += 1) {
      const group = makeElement("span", "llvmcfg-iteration-group");
      const box = makeElement("span", "llvmcfg-iteration-box");
      const start = groupIndex * factor;
      const end = Math.min(start + factor, inputLength);
      for (let iteration = start; iteration < end; iteration += 1) {
        box.append(makeElement("span", "llvmcfg-iteration-index", String(iteration)));
      }
      group.append(box);
      group.append(
        makeElement(
          "small",
          "llvmcfg-invocation-index",
          mode === "runtime-miss" ? "" : "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯"[groupIndex]
        )
      );
      groups.append(group);
    }
    groups.setAttribute(
      "aria-label",
      inputLength + " 个 iteration 按每组 " + factor + " 个划分为 " + groupCount + " 组"
    );
    return groups;
  }

  function buildFactorVisual(node, factor, invocationCount, taskCount, mode, inputLength) {
    const isObserved = mode === "observed";
    const isMiss = mode === "runtime-miss";
    const groupCount = isObserved ? invocationCount : Math.floor(inputLength / factor);
    const modeLabel = isObserved ? "运行实测" : isMiss ? "本次未执行" : "约束推演";
    const signature = ["factor-groups", factor, invocationCount, taskCount, mode, inputLength].join(":");

    replaceStateVisual(node, signature, () => {
      const root = makeElement("span", "llvmcfg-factor-visual");
      const header = makeElement("span", "llvmcfg-factor-header");
      const identity = makeElement("span", "llvmcfg-factor-identity");
      identity.append(makeElement("b", "llvmcfg-factor-symbol", "×" + factor));
      identity.append(makeElement("small", "llvmcfg-factor-caption", "展开规格"));
      header.append(identity);
      header.append(makeElement("small", "llvmcfg-factor-mode", modeLabel));
      root.append(header);

      const processLabel = makeElement("span", "llvmcfg-factor-process-label");
      processLabel.append(document.createTextNode("一次处理 "));
      processLabel.append(makeElement("b", "", String(factor)));
      processLabel.append(document.createTextNode(" 个 iteration"));
      root.append(processLabel);
      root.append(buildIterationGroups(inputLength, factor, groupCount, mode));

      const summary = makeElement("span", "llvmcfg-factor-summary");
      const calls = makeElement("span");
      calls.append(
        makeElement(
          "strong",
          "",
          isObserved ? invocationCount + " 次调用" : isMiss ? "0 次调用" : groupCount + " 个完整组"
        )
      );
      calls.append(makeElement("small", "", isObserved ? "运行调用次数" : "基于输入长度计算"));
      summary.append(calls);

      const tasks = makeElement("span");
      if (isObserved) {
        tasks.append(makeElement("strong", "", formatTaskRate(taskCount, invocationCount) + " Task/次"));
        tasks.append(makeElement("small", "", taskCount + " Task事件"));
      } else if (isMiss) {
        tasks.append(makeElement("strong", "", "0 Task事件"));
        tasks.append(makeElement("small", "", "本次没有执行记录"));
      } else {
        tasks.append(makeElement("strong", "", "尚无运行数据"));
        tasks.append(makeElement("small", "", "不推算 Task 数量"));
      }
      summary.append(tasks);
      root.append(summary);
      return root;
    });
  }

  function bindFactorInteractions(node) {
    if (node.dataset.llvmcfgFactorInteractionBound === "true") return;
    node.dataset.llvmcfgFactorInteractionBound = "true";
    const showFactor = () => {
      const flow = node.closest(".react-flow");
      if (flow) setFlowFactorLink(flow, Number(node.dataset.unrollFactor));
    };
    const restoreSelection = () => {
      const flow = node.closest(".react-flow");
      if (!flow) return;
      window.requestAnimationFrame(() => setFlowFactorLink(flow, getSelectedFactor(flow)));
    };
    node.addEventListener("pointerenter", showFactor);
    node.addEventListener("pointerleave", restoreSelection);
    node.addEventListener("focus", showFactor);
    node.addEventListener("blur", restoreSelection);
    node.addEventListener("click", () => window.requestAnimationFrame(showFactor));
  }

  function stabilizeStateCard(node, kind, status) {
    node.dataset.llvmcfgCardKind = kind;
    node.dataset.llvmcfgStatus = status;
    node.style.setProperty("height", "auto", "important");
    node.style.setProperty("max-height", "none", "important");
    node.style.setProperty("overflow", "visible", "important");
    [
      node.querySelector(":scope > strong"),
      node.querySelector(":scope > p"),
      node.querySelector(":scope > .review-flow-node__metrics")
    ].forEach((element) => element?.style.setProperty("display", "none", "important"));
  }

  function enhanceSemanticGraphCards() {
    document.querySelectorAll(".review-flow-node").forEach((node) => {
      const eyebrow = node.querySelector(".review-flow-node__eyebrow");
      const eyebrowText = eyebrow?.textContent.trim() || "";
      const title = node.querySelector(":scope > strong");
      const originalTitle = title?.textContent.trim() || "";
      const metrics = node.querySelectorAll(".review-flow-node__metrics > span");
      const isObservedBaseline =
        node.dataset.llvmcfgSemanticType === "observed-baseline" ||
        node.classList.contains("llvmcfg-observed-baseline") ||
                /^observed baseline/i.test(eyebrowText);
      const isRequestScenario =
        node.dataset.llvmcfgSemanticType === "request-scenario" ||
        node.classList.contains("llvmcfg-request-scenario") ||
        /\brequest\b/i.test(eyebrowText) && /what-if review/i.test(originalTitle);
      const isRuntimeScenario =
        node.dataset.llvmcfgSemanticType === "runtime-scenario" ||
        node.classList.contains("llvmcfg-runtime-scenario") ||
        /^observed scenario/i.test(eyebrowText) && /^runtime\b/i.test(originalTitle);
      const isObservedPath =
        node.dataset.llvmcfgSemanticType === "observed-path" ||
        node.classList.contains("llvmcfg-observed-path") ||
        /^(observed execution path|executed path)/i.test(eyebrowText);
      const isRuntimeMiss =
        node.dataset.llvmcfgSemanticType === "runtime-miss" ||
        node.classList.contains("llvmcfg-runtime-miss") ||
        /^not observed/i.test(eyebrowText);
      const isCandidatePath =
        node.dataset.llvmcfgSemanticType === "candidate-path" ||
        node.classList.contains("llvmcfg-candidate-path") ||
        /^(coverage witness|eligible fallback path)/i.test(eyebrowText);

      if (isObservedBaseline || isRuntimeScenario) {
        node.dataset.llvmcfgSemanticType = isRuntimeScenario
          ? "runtime-scenario"
          : "observed-baseline";
        node.classList.add(
          "llvmcfg-semantic-card",
          "llvmcfg-scenario-card",
          isRuntimeScenario ? "llvmcfg-runtime-scenario" : "llvmcfg-observed-baseline"
        );

        const inputLengthMatch = originalTitle.match(/输入长度[：:]\s*(\d+)\s*token/i);
        const inputLength = Number(inputLengthMatch?.[1] || node.dataset.inputLength || "16");
        node.dataset.inputLength = String(inputLength);

        const firstMetric = Number(
          metrics[0]?.querySelector("b")?.textContent.trim() ||
          (isRuntimeScenario ? node.dataset.taskEventCount : node.dataset.factorCount) ||
          "0"
        );
        const secondMetric = Number(
          metrics[1]?.querySelector("b")?.textContent.trim() ||
          (isRuntimeScenario ? node.dataset.factorCount : node.dataset.taskEventCount) ||
          "0"
        );
        const observedFactorCount = isRuntimeScenario ? secondMetric : firstMetric;
        const taskCount = isRuntimeScenario ? firstMetric : secondMetric;
        node.dataset.factorCount = String(observedFactorCount);
        node.dataset.taskEventCount = String(taskCount);
        setLeadingText(eyebrow, "已观测运行");
        node.querySelector(":scope > .llvmcfg-card-identity")?.remove();
        node.setAttribute(
          "aria-label",
          "已观测运行，输入长度 " + inputLength + " token，" +
            observedFactorCount + " 种展开规格有执行记录"
        );
        buildScenarioVisual(node, inputLength, observedFactorCount, taskCount, true);
        stabilizeStateCard(node, "scenario", "observed");
        return;
      }

      if (isRequestScenario) {
        node.dataset.llvmcfgSemanticType = "request-scenario";
        node.classList.add("llvmcfg-semantic-card", "llvmcfg-scenario-card", "llvmcfg-request-scenario");
        const inputLengthMatch = originalTitle.match(/输入长度[：:]\s*(\d+)\s*token/i);
        const inputLength = Number(inputLengthMatch?.[1] || node.dataset.inputLength || "16");
        node.dataset.inputLength = String(inputLength);
        const factorCount = Number(
          metrics[0]?.querySelector("b")?.textContent.trim() || node.dataset.factorCount || "0"
        );
        node.dataset.factorCount = String(factorCount);
        setLeadingText(eyebrow, "待审查场景");
        node.querySelector(":scope > .llvmcfg-card-identity")?.remove();
        node.setAttribute(
          "aria-label",
          "待审查场景，输入长度 " + inputLength + " token，" + factorCount + " 种展开规格可参与"
        );
        buildScenarioVisual(node, inputLength, factorCount, 0, false);
        stabilizeStateCard(node, "scenario", "request");
        return;
      }

      if (isObservedPath || isCandidatePath || isRuntimeMiss) {
        const factorMatch = originalTitle.match(/(?:Unroll|展开规格)\s*[·:]?\s*×\s*(\d+)/i);
        const factor = factorMatch?.[1] || node.dataset.unrollFactor;
        if (!factor) return;
        node.dataset.unrollFactor = factor;
        const mode = isObservedPath ? "observed" : isRuntimeMiss ? "runtime-miss" : "candidate";
        node.dataset.llvmcfgSemanticType = isObservedPath
          ? "observed-path"
          : isRuntimeMiss
            ? "runtime-miss"
            : "candidate-path";
        node.classList.add(
          "llvmcfg-semantic-card",
          "llvmcfg-factor-card",
          isObservedPath
            ? "llvmcfg-observed-path"
            : isRuntimeMiss
              ? "llvmcfg-runtime-miss"
              : "llvmcfg-candidate-path"
        );
        setLeadingText(
          eyebrow,
          isObservedPath ? "已执行" : isRuntimeMiss ? "本次未执行" : "待审查"
        );
        node.querySelector(":scope > .llvmcfg-card-identity")?.remove();
        const invocationCount = Number(
          metrics[0]?.querySelector("b")?.textContent.trim() || node.dataset.invocationCount || "0"
        );
        node.dataset.invocationCount = String(invocationCount);
        let taskCount = 0;
        if (isObservedPath) {
          const taskMatch = node.querySelector(":scope > p")?.textContent.match(/(\d+)\s*task events/i);
          const taskMetric = metrics[1]?.querySelector("small")?.textContent.match(/task events/i)
            ? metrics[1]?.querySelector("b")?.textContent.trim()
            : null;
          taskCount = Number(taskMatch?.[1] || taskMetric || node.dataset.taskEventCount || "0");
        }
        node.dataset.taskEventCount = String(taskCount);
        node.setAttribute(
          "aria-label",
          "展开规格 ×" + factor + "，一次展开 " + factor + " 个 iteration，" +
            (isObservedPath
              ? "本次运行调用 " + invocationCount + " 次，关联 " + taskCount + " 个 Task 事件"
              : isRuntimeMiss
                ? "本次运行未执行"
                : "当前仅为约束推演，尚无真实运行记录")
        );
        const flow = node.closest(".react-flow");
        const scenario = flow?.querySelector('[data-llvmcfg-card-kind="scenario"]');
        const inputLength = Number(
          scenario?.dataset.inputLength ||
          (isObservedPath && invocationCount ? Number(factor) * invocationCount : 16)
        );
        buildFactorVisual(node, Number(factor), invocationCount, taskCount, mode, inputLength);
        stabilizeStateCard(node, "factor", mode);
        bindFactorInteractions(node);
      }
    });
  }

  function getSelectedFactor(flow) {
    const selected = flow.querySelector(
      ".review-flow-node[data-llvmcfg-card-kind=\"factor\"].is-selected, " +
      ".react-flow__node.selected .review-flow-node[data-llvmcfg-card-kind=\"factor\"]"
    );
    return selected ? Number(selected.dataset.unrollFactor) : null;
  }

  function setFlowFactorLink(flow, factor) {
    const validFactor = Number.isFinite(factor) && factor > 0 ? factor : null;
    flow.querySelectorAll('.review-flow-node[data-llvmcfg-card-kind="factor"]').forEach((card) => {
      const isLinked = Number(card.dataset.unrollFactor) === validFactor;
      card.dataset.llvmcfgLinked = String(isLinked);
      card.classList.toggle("is-linked-factor", isLinked);
    });
    flow.querySelectorAll(".react-flow__edge-text.llvmcfg-condition-edge-label").forEach((label) => {
      label.classList.toggle("is-linked-condition", Number(label.dataset.unrollFactor) === validFactor);
    });

    flow.querySelectorAll('.review-flow-node[data-llvmcfg-card-kind="scenario"]').forEach((scenario) => {
      const inputLength = Number(scenario.dataset.inputLength || "0");
      const cells = scenario.querySelectorAll(".llvmcfg-token-cell");
      scenario.dataset.llvmcfgLinked = String(Boolean(validFactor));
      scenario.classList.toggle("is-linked-scenario", Boolean(validFactor));
      cells.forEach((cell, index) => {
        const groupIndex = validFactor ? Math.floor(index / validFactor) : 0;
        const fullGroupLimit = validFactor ? Math.floor(inputLength / validFactor) * validFactor : 0;
        cell.classList.toggle("is-group-alt", Boolean(validFactor) && groupIndex % 2 === 1);
        cell.classList.toggle("is-group-start", Boolean(validFactor) && index % validFactor === 0);
        cell.classList.toggle(
          "is-group-end",
          Boolean(validFactor) && (index + 1) % validFactor === 0
        );
        cell.classList.toggle("is-remainder", Boolean(validFactor) && index >= fullGroupLimit);
      });

      const summary = scenario.querySelector(".llvmcfg-scenario-summary");
      const visual = scenario.querySelector(":scope > .llvmcfg-state-visual");
      if (!summary || !visual) return;
      if (!validFactor) {
        setText(summary, visual.dataset.defaultFooter || "");
        return;
      }
      const fullGroups = Math.floor(inputLength / validFactor);
      const remainder = inputLength % validFactor;
      setText(
        summary,
        inputLength + " token → " + fullGroups + " 个完整组 × " + validFactor +
          (remainder ? "，另有 " + remainder + " token" : "")
      );
    });
  }

  function syncFlowVisualComparisons() {
    document.querySelectorAll(".react-flow").forEach((flow) => {
      setFlowFactorLink(flow, getSelectedFactor(flow));
    });
  }

  function getEdgeConditionHelp(factor) {
    return "×" + factor + " 表示选择一次展开 " + factor +
      " 个 iteration 的路径。它是 loop_unroll 的展开规格，不是源码循环的基础步长 step=1。";
  }

  function getEdgeTooltip() {
    let tooltip = document.getElementById("llvmcfg-edge-tooltip");
    if (tooltip) return tooltip;
    tooltip = document.createElement("div");
    tooltip.id = "llvmcfg-edge-tooltip";
    tooltip.className = "llvmcfg-edge-tooltip";
    tooltip.setAttribute("role", "tooltip");
    tooltip.hidden = true;
    document.body.append(tooltip);
    return tooltip;
  }

  function showEdgeTooltip(label) {
    const tooltip = getEdgeTooltip();
    const bounds = label.getBoundingClientRect();
    const center = bounds.left + bounds.width / 2;
    tooltip.textContent = label.dataset.llvmcfgConditionHelp || "";
    tooltip.style.left = Math.max(176, Math.min(window.innerWidth - 176, center)) + "px";
    tooltip.style.top = Math.max(16, bounds.top) + "px";
    tooltip.hidden = false;
  }

  function hideEdgeTooltip() {
    const tooltip = document.getElementById("llvmcfg-edge-tooltip");
    if (tooltip) tooltip.hidden = true;
  }

  function getConditionTarget(label) {
    const edge = label.closest(".react-flow__edge");
    const edgeMarker = edge?.querySelector('[data-testid^="rf__edge-"]');
    const edgeId = Number(edgeMarker?.getAttribute("data-testid")?.replace("rf__edge-", ""));
    if (!Number.isFinite(edgeId) || edgeId < 1000) return null;

    const targetId = String(edgeId - 999);
    const flow = label.closest(".react-flow");
    const targetNode = flow?.querySelector('.react-flow__node[data-id="' + targetId + '"]');
    const targetCard = targetNode?.querySelector(".review-flow-node");
    const factor =
      targetCard?.dataset.unrollFactor ||
      targetCard?.querySelector(":scope > strong")?.textContent.match(/×\s*(\d+)/)?.[1];
    return targetNode && factor ? { targetNode, factor } : null;
  }

  function positionConditionLabel(label, targetNode) {
    const wrapper = label.closest(".react-flow__edge-textwrapper");
    const coordinateRoot = wrapper?.parentElement;
    const matrix = coordinateRoot?.getScreenCTM?.();
    const svg = label.ownerSVGElement;
    if (!wrapper || !matrix || !svg) return;

    const targetBounds = targetNode.getBoundingClientRect();
    const screenPoint = svg.createSVGPoint();
    screenPoint.x = targetBounds.left + targetBounds.width / 2 - 26;
    screenPoint.y = targetBounds.top - 24;
    const localPoint = screenPoint.matrixTransform(matrix.inverse());
    const textBounds = label.getBBox();
    wrapper.setAttribute(
      "transform",
      "translate(" + (localPoint.x - textBounds.width / 2) + " " +
        (localPoint.y - textBounds.height / 2) + ")"
    );
  }

  function enhanceEdgeConditionLabels() {
    document.querySelectorAll(".react-flow__edge-text").forEach((label) => {
      const factorMatch = label.textContent.trim().match(/^×\s*(\d+)$/);
      if (!factorMatch) return;
      const target = getConditionTarget(label);
      const factor = target?.factor || factorMatch[1];
      const help = getEdgeConditionHelp(factor);
      setText(label, "×" + factor);
      if (target) positionConditionLabel(label, target.targetNode);
      getEdgeTooltip();
      label.classList.add("llvmcfg-condition-edge-label");
      label.setAttribute("tabindex", "0");
      label.setAttribute("aria-describedby", "llvmcfg-edge-tooltip");
      label.setAttribute("aria-label", help);
      label.dataset.unrollFactor = String(factor);
      label.dataset.llvmcfgConditionHelp = help;
      if (label.dataset.llvmcfgTooltipBound === "true") return;
      label.dataset.llvmcfgTooltipBound = "true";
      label.addEventListener("pointerenter", () => showEdgeTooltip(label));
      label.addEventListener("pointerleave", hideEdgeTooltip);
      label.addEventListener("focus", () => showEdgeTooltip(label));
      label.addEventListener("blur", hideEdgeTooltip);
    });
  }

  function enhanceScenarioControls() {
    const shapeInput = document.getElementById("shape-t");
    const controlGroup = shapeInput?.closest(".controlFlowExplorer_controlGroup__sDVKd");
    if (!controlGroup) return;

    const heading = controlGroup.querySelector(".controlFlowExplorer_controlHeading__hW2W9");
    const shapeLabel = heading?.querySelector('label[for="shape-t"]');
    setText(shapeLabel, "输入长度");
    shapeLabel?.setAttribute(
      "aria-label",
      "输入长度 t，等于 x_in.shape[0]。t 是输入张量第 0 维的动态长度；在当前算子中，对应本次需要处理的 token 数。"
    );
    heading?.querySelector(".controlFlowExplorer_scenarioBadge__mcVLm")?.remove();

    if (heading && !controlGroup.querySelector(".llvmcfg-source-definition")) {
      const sourceDefinition = document.createElement("div");
      sourceDefinition.className = "llvmcfg-source-definition";
      sourceDefinition.innerHTML =
        '<span>源码中的动态边界</span>' +
        '<pre aria-label="输入长度对应的 Python 源码"><code>' +
        't = x_in.shape[0]\n' +
        'pypto.<span class="llvmcfg-syntax-function">loop_unroll</span>(' +
        '<span class="llvmcfg-syntax-number">0</span>, t, ' +
        '<span class="llvmcfg-syntax-number">1</span>, ...)' +
        "</code></pre>" +
        "<p>t 是输入张量第 0 维的动态长度；在当前算子中，对应本次需要处理的 token 数。</p>";
      heading.after(sourceDefinition);
    }

    const description = Array.from(controlGroup.children).find(
      (element) => element.tagName === "P"
    );
    setText(
      description,
      "循环处理索引 0 到 t−1。当前只有输入长度为 16 token 的场景具备真实运行证据。"
    );

    const reviewControls = controlGroup.parentElement;
    const readonlyGroup = reviewControls?.querySelector(
      ".controlFlowExplorer_readonlyGroup__yKqX7"
    );
    if (!readonlyGroup) return;

    const readonlyTitle = Array.from(readonlyGroup.children).find(
      (element) => element.tagName === "SPAN"
    );
    setText(readonlyTitle, "循环如何运行");

    let summary = readonlyGroup.querySelector(".llvmcfg-loop-summary");
    if (!summary) {
      summary = document.createElement("p");
      summary.className = "llvmcfg-loop-summary";
      readonlyGroup.querySelector("dl")?.before(summary);
    }
    setText(
      summary,
      "源码循环基础步长为 1，即逻辑上遍历 0 … t-1；后续 loop_unroll 会改变一次循环实际展开多少个 iteration。"
    );

    const rows = readonlyGroup.querySelectorAll("dl > div");
    const copy = [
      ["处理范围", "第 0 个到第 t−1 个 token"],
      ["源码基础步长", "1 个 iteration"],
      ["对应源码", "第 252–253 行"]
    ];
    rows.forEach((row, index) => {
      const item = copy[index];
      if (!item) return;
      setText(row.querySelector("dt"), item[0]);
      setText(row.querySelector("dd"), item[1]);
    });
  }

  function scheduleEnhancement() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(enhanceCode);
  }

  const root = document.getElementById("root");
  if (root) {
    new MutationObserver(scheduleEnhancement).observe(root, {
      subtree: true,
      childList: true,
      characterData: true
    });
    scheduleEnhancement();
  }
})();
