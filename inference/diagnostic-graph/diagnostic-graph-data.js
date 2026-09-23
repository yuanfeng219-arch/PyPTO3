/* Diagnostic graph data. Values are migrated from the former graph and backbone views. */
(function () {
  const rankLatency = [12.4, 12.7, 17.8, 12.5, 12.9, 12.6, 12.3, 12.8];
  const routeLoad = [.94, 1.02, 1.65, .96, 1.01, .93, 1.08, .98];

  window.DiagnosticGraphData = {
    nodes: [
      {id: 'finding', kind: 'finding', title: 'Throughput', signal: '−27%', summary: '1000 → 730 tok/s · TPOT +32%', priority: 'anchor'},
      {id: 'decode', kind: 'reasoning', title: 'Decode', signal: '+33%', summary: '贡献 82% 的新增时间', evidenceRefs: ['phase']},
      {id: 'rank2', kind: 'entity', title: 'Rank 2', signal: '+41%', summary: '17.8 ms · 组中位数 12.6 ms', evidenceRefs: ['rank-latency']},
      {id: 'device', kind: 'cause', state: 'normal', title: 'Device', signal: '✓', summary: '时钟、利用率处于基线范围'},
      {id: 'memory', kind: 'cause', state: 'normal', title: 'Memory / KV', signal: '✓', summary: '无驱逐或缺页'},
      {id: 'routing', kind: 'cause', state: 'abnormal', title: 'Routing', signal: '1.65×', summary: 'Rank 2 / EP 组中位数负载', evidenceRefs: ['route-load', 'expert-traffic', 'placement']},
      {id: 'kernel', kind: 'cause', state: 'normal', title: 'Kernel', signal: '✓', summary: '同 shape 最大差异 +3.4%'},
      {id: 'communication', kind: 'cause', state: 'unresolved', title: 'Communication', signal: '+11%', summary: '1.9 → 2.1 ms · 不足以解释 +41%'},
      {id: 'hypothesis', kind: 'hypothesis', title: '热点 Expert 共置可能造成 Routing 负载不均', signal: '待验证', summary: '由 3 份独立证据支持 · High confidence'},
      {
        id: 'diagnosis',
        kind: 'diagnosis',
        title: 'Rank 2 是当前 Decode 的关键路径',
        signal: '高置信',
        summary: '长尾时延与热点负载在 Rank 2 同位',
        action: {
          eyebrow: 'Experiment suggestion',
          title: '验证 Expert 共置解释',
          summary: '只改变 E19 的 Placement，其他条件保持不变',
          label: '前往验证'
        }
      }
    ],
    edges: [
      {id: 'finding-decode', source: 'finding', target: 'decode', relation: 'locate', label: '82% 新增时间'},
      {id: 'decode-rank2', source: 'decode', target: 'rank2', relation: 'promote', label: '唯一 Rank 离群'},
      {id: 'rank2-device', source: 'rank2', target: 'device', relation: 'checked'},
      {id: 'rank2-memory', source: 'rank2', target: 'memory', relation: 'checked'},
      {id: 'rank2-kernel', source: 'rank2', target: 'kernel', relation: 'checked'},
      {id: 'rank2-communication', source: 'rank2', target: 'communication', relation: 'unresolved'},
      {id: 'rank2-routing', source: 'rank2', target: 'routing', relation: 'continue', label: '最强关联'},
      {id: 'routing-hypothesis', source: 'routing', target: 'hypothesis', relation: 'support', label: '3 份证据'},
      {id: 'hypothesis-diagnosis', source: 'hypothesis', target: 'diagnosis', relation: 'diagnose', label: '综合研判'}
    ],
    evidenceById: {
      phase: {
        title: '阶段时延 / ms',
        type: 'table',
        rows: [['Queue', '3.0', '3.2'], ['Prefill', '46.0', '47.0'], ['Decode', '12.6', '16.8'], ['Output', '1.3', '1.3']],
        note: '只有 Decode 进入下一层调查。'
      },
      'rank-latency': {
        title: 'Rank 解码时延 / ms', type: 'bars', labels: rankLatency.map((_, index) => `R${index}`), values: rankLatency, hotIndex: 2,
        note: 'Rank 2 是唯一显著离群项。'
      },
      'route-load': {
        attachmentTitle: 'Routed tokens', title: 'Rank 路由 Token / 中位数倍数', type: 'bars', labels: routeLoad.map((_, index) => `R${index}`), values: routeLoad, hotIndex: 2,
        note: 'R2 承担 1.65× 中位数负载；其他 Rank 均接近参考范围。'
      },
      'expert-traffic': {
        attachmentTitle: 'Expert traffic', title: 'Expert 流量 / 平均值倍数', type: 'bars', labels: ['E0', 'E1', 'E7', 'E3', 'E4', 'E5', 'E19', 'E8'], values: [.31, .44, 1, .39, .48, .36, .88, .43], hotIndices: [2, 6],
        note: 'E7 2.4×、E19 2.1×；其他 Expert 大多处于平均流量的 0.7–1.3×。'
      },
      placement: {
        attachmentTitle: 'Expert placement', title: 'Expert → Rank', type: 'table', rows: [['E7', 'Rank 2', 'Rank 2'], ['E19', 'Rank 6', 'Rank 2']],
        note: '基线将两个最热 Expert 分别放在 Rank 2 和 Rank 6。'
      }
    },
    experiment: {
      hypothesis: 'Hot expert co-location causes the Rank 2 long pole.',
      change: 'E19: Rank 2 → Rank 6；E7 保持 Rank 2',
      fixed: 'Workload · Concurrency · EP = 8 · 其他 Serving 配置',
      expected: 'Rank skew ↓ · Decode latency ↓ · TPOT ↓ · Throughput ↑'
    }
  };
}());
