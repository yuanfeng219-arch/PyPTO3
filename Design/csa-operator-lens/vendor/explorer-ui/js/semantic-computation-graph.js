/* Semantic contract. Layout and presentation remain separate projections. */
(function (root) {
  'use strict';
  const schemaVersion = 2;
  const TYPES = new Set(['data', 'operation', 'state', 'composite']);
  const EDGES = new Set(['data', 'computation', 'state', 'control', 'feedback']);
  const PROVENANCE = new Set(['fact', 'resolved', 'compiled', 'estimated', 'measured']);
  const RELATIONS = new Set(['identity', 'view', 'slice', 'aggregate', 'control']);
  const UNKNOWN = /^(?:unknown|待确认|未知|tbd|todo|n\/a|\?)$/i;
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const fail = (path, message) => { throw new Error(`${path}: ${message}`); };
  const object = (value, path) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) fail(path, 'expected object');
    return value;
  };
  const string = (value, path) => {
    if (typeof value !== 'string' || !value.trim() || UNKNOWN.test(value.trim())) fail(path, 'expected known nonempty string');
  };
  const list = (value, path) => { if (!Array.isArray(value)) fail(path, 'expected array'); return value; };
  function noUnknown(value, path) {
    if (value == null) fail(path, 'null/undefined is not a known value');
    if (typeof value === 'string' && UNKNOWN.test(value.trim())) fail(path, 'unknown placeholder');
    if (Array.isArray(value)) value.forEach((v, i) => noUnknown(v, `${path}[${i}]`));
    else if (typeof value === 'object') Object.entries(value).forEach(([k, v]) => noUnknown(v, `${path}.${k}`));
  }
  function source(value, path, files) {
    object(value, path); string(value.file, `${path}.file`);
    if (files && !files.has(value.file)) fail(path, `source file ${value.file} is not declared`);
    list(value.ranges, `${path}.ranges`).forEach((range, i) => {
      object(range, `${path}.ranges[${i}]`);
      if (!Number.isInteger(range.start) || !Number.isInteger(range.end) || range.start < 1 || range.end < range.start) fail(path, 'invalid source range');
      const count = files?.get(value.file);
      if (count !== undefined && range.end > count) fail(path, 'source range exceeds lineCount');
    });
    list(value.symbols, `${path}.symbols`).forEach((symbol, i) => string(symbol, `${path}.symbols[${i}]`));
    if (own(value, 'primaryRange') && (!Number.isInteger(value.primaryRange) || value.primaryRange < 0 || value.primaryRange >= value.ranges.length)) fail(path, 'invalid primary source range');
  }
  function provenance(value, path, requiredFields) {
    const covered = new Set();
    list(value, path).forEach((entry, i) => {
      const at = `${path}[${i}]`; object(entry, at);
      if (!PROVENANCE.has(entry.kind)) fail(at, 'invalid provenance kind');
      string(entry.origin, `${at}.origin`);
      list(entry.fields, `${at}.fields`).forEach((field) => {
        string(field, `${at}.fields`);
        if (requiredFields.length && !requiredFields.includes(field) && !requiredFields.some((key) => field.startsWith(`${key}.`))) fail(at, `field ${field} is not populated`);
        covered.add(field);
      });
      if (own(entry, 'note')) string(entry.note, `${at}.note`);
    });
    requiredFields.forEach((field) => {
      if (![...covered].some((item) => item === field || field.startsWith(`${item}.`))) fail(path, `missing provenance for ${field}`);
    });
  }
  function ports(value, path) {
    object(value, path);
    for (const side of ['inputs', 'outputs']) {
      object(value[side], `${path}.${side}`);
      Object.entries(value[side]).forEach(([name, port]) => {
        string(name, `${path}.${side} name`); object(port, `${path}.${side}.${name}`);
        if (own(port, 'entity')) string(port.entity, `${path}.${side}.${name}.entity`);
        if (own(port, 'description')) string(port.description, `${path}.${side}.${name}.description`);
        for (const key of ['dtype', 'granularity']) if (own(port, key)) string(port[key], `${path}.${side}.${name}.${key}`);
      });
    }
  }
  function payload(node, path) {
    const value = object(node[node.type], `${path}.${node.type}`);
    if (node.type === 'data') {
      string(value.role, `${path}.${node.type}.role`); string(value.granularity, `${path}.${node.type}.granularity`);
      for (const key of ['dtype', 'layout', 'symbol', 'entity']) if (own(value, key)) string(value[key], `${path}.data.${key}`);
      if (own(value, 'shape')) {
        object(value.shape, `${path}.${node.type}.shape`);
        if (!own(value.shape, 'allocated') && !own(value.shape, 'valid')) fail(path, 'empty shape');
        for (const key of ['allocated', 'valid'].filter((key) => own(value.shape, key))) list(value.shape[key], `${path}.${node.type}.shape.${key}`).forEach((dim) => {
          if ((typeof dim !== 'string' || !dim.trim()) && (!Number.isInteger(dim) || dim < 0)) fail(path, `invalid ${key} dimension`);
        });
        if (value.shape.allocated && value.shape.valid && value.shape.allocated.length !== value.shape.valid.length) fail(path, 'shape rank mismatch');
      }
      if (own(value, 'relations')) list(value.relations, `${path}.${node.type}.relations`).forEach((relation) => {
        object(relation, `${path}.${node.type}.relations`);
        string(relation.kind, `${path}.${node.type}.relations.kind`);
        string(relation.entity, `${path}.${node.type}.relations.entity`);
        string(relation.description, `${path}.${node.type}.relations.description`);
      });
      if (own(value, 'addressing')) {
        object(value.addressing, `${path}.${node.type}.addressing`);
        Object.entries(value.addressing).forEach(([key, expression]) => { string(key, `${path}.${node.type}.addressing key`); string(expression, `${path}.${node.type}.addressing.${key}`); });
      }
      if (own(value, 'storage')) { object(value.storage, `${path}.${node.type}.storage`); string(value.storage.space, `${path}.${node.type}.storage.space`); }
      for (const key of ['producer', 'consumers']) if (own(value, key)) {
        const ids = key === 'producer' ? [value[key]] : list(value[key], `${path}.${node.type}.${key}`);
        ids.forEach((id) => string(id, `${path}.${node.type}.${key}`));
      }
    } else if (node.type === 'state') {
      string(value.initial, `${path}.${node.type}.initial`); string(value.update, `${path}.${node.type}.update`);
      if (own(value, 'feedback')) string(value.feedback, `${path}.${node.type}.feedback`);
      if (own(value, 'symbols')) list(value.symbols, `${path}.${node.type}.symbols`).forEach((symbol) => string(symbol, `${path}.${node.type}.symbols`));
      for (const key of ['dtype', 'granularity', 'entity']) if (own(value, key)) string(value[key], `${path}.state.${key}`);
    } else {
      string(value.semantics, `${path}.${node.type}.semantics`); string(value.category, `${path}.${node.type}.category`);
      list(value.api, `${path}.${node.type}.api`).forEach((api) => string(api, `${path}.${node.type}.api`));
      if (node.type === 'composite') {
        object(value.subgraph, `${path}.${node.type}.subgraph`);
        object(value.portMappings, `${path}.${node.type}.portMappings`);
      }
    }
    return value;
  }
  function mapping(node, byId, path) {
    const { subgraph, portMappings } = node.composite;
    for (const side of ['inputs', 'outputs']) {
      const declared = node.ports[side];
      const maps = object(portMappings[side], `${path}.portMappings.${side}`);
      if (Object.keys(maps).sort().join('|') !== Object.keys(declared).sort().join('|')) fail(path, `${side} mappings must match declared ports`);
      Object.entries(maps).forEach(([name, map]) => {
        const at = `${path}.portMappings.${side}.${name}`; object(map, at);
        string(map.node, `${at}.node`); string(map.port, `${at}.port`);
        if (!RELATIONS.has(map.relation)) fail(at, 'invalid relation');
        if (['slice', 'aggregate'].includes(map.relation)) string(map.description, `${at}.description`);
        const child = byId.get(map.node);
        if (!child || !own(child.ports[side], map.port)) fail(at, 'missing child endpoint port');
        const parentPort = declared[name], childPort = child.ports[side][map.port];
        const childEntity = childPort.entity || child.data?.entity || child.state?.entity;
        if (['identity', 'view'].includes(map.relation) && parentPort.entity && childEntity && parentPort.entity !== childEntity) fail(at, 'entity mismatch');
        if (map.relation !== 'control') {
          const parent = parentPort, inner = childPort;
          for (const key of ['dtype', 'granularity']) {
            if (key === 'granularity' && map.relation !== 'identity') continue;
            const a = parent[key] || node[node.type][key], b = inner[key] || child[child.type][key];
            if (a && b && a !== b) fail(at, `${key} mismatch`);
          }
        }
      });
    }
    if (own(subgraph, 'inputs') || own(subgraph, 'outputs')) fail(path, 'legacy subgraph ports are not allowed');
  }
  function validate(model) {
    noUnknown(model, 'graph');
    if (model.schemaVersion !== schemaVersion) fail('graph', `schemaVersion must be ${schemaVersion}`);
    function visit(graph, path, files) {
      object(graph, path); string(graph.id, `${path}.id`);
      if (graph.schemaVersion !== undefined && graph.schemaVersion !== schemaVersion) fail(path, 'invalid schemaVersion');
      if (own(graph, 'sourceFiles')) {
        files = new Map();
        const entries = Array.isArray(graph.sourceFiles) ? graph.sourceFiles : Object.entries(object(graph.sourceFiles, `${path}.sourceFiles`)).map(([file, value]) => ({ file, lineCount: typeof value === 'number' ? value : value?.lineCount }));
        entries.forEach((entry) => {
          object(entry, `${path}.sourceFiles`); string(entry.file, `${path}.sourceFiles.file`);
          if (!Number.isInteger(entry.lineCount) || entry.lineCount < 1) fail(path, 'invalid lineCount');
          if (files.has(entry.file)) fail(path, 'duplicate source file');
          files.set(entry.file, entry.lineCount);
        });
      }
      const nodes = list(graph.nodes, `${path}.nodes`), edges = list(graph.edges, `${path}.edges`);
      const byId = new Map(), allIds = new Set();
      nodes.forEach((node) => {
        object(node, `${path}.node`); string(node.id, `${path}.node.id`);
        if (node.id.includes('/')) fail(path, 'node id cannot contain /');
        for (const key of ['lines', 'sourceLine', 'sourceRanges', 'semanticType', 'title', 'sub', 'kind', 'badge', 'api', 'detail', 'evidence']) if (own(node, key)) fail(path, `derived field ${key} is not semantic schema`);
        if (allIds.has(node.id)) fail(path, `duplicate id ${node.id}`); allIds.add(node.id);
        const at = `${path}/${node.id}`;
        if (!TYPES.has(node.type)) fail(at, 'invalid node type');
        string(node.name, `${at}.name`); string(node.description, `${at}.description`);
        source(node.source, `${at}.source`, files); ports(node.ports, `${at}.ports`);
        const value = payload(node, at);
        const fields = node.type === 'data' ? Object.keys(value).filter((key) => !['producer', 'consumers'].includes(key)).map((key) => `data.${key}`) :
          node.type === 'state' ? Object.keys(value).map((key) => `state.${key}`) : [];
        provenance(node.provenance, `${at}.provenance`, fields);
        if (!node.provenance.length) fail(at, 'missing provenance');
        byId.set(node.id, node);
      });
      nodes.forEach((node) => {
        const at = `${path}/${node.id}`;
        if (node.type === 'composite') {
          const children = visit(node.composite.subgraph, at, files);
          mapping(node, children, at);
        }
        if (node.type === 'data') {
          const value = node[node.type];
          for (const key of ['producer', 'consumers']) if (own(value, key)) {
            const ids = key === 'producer' ? [value[key]] : value[key];
            ids.forEach((id) => { if (!byId.has(id)) fail(at, `unknown ${key} ${id}`); });
            ids.forEach((id) => {
              if (!edges.some((edge) => key === 'producer' ? edge.s === id && edge.t === node.id : edge.s === node.id && edge.t === id)) fail(at, `${key} does not match dependency graph`);
            });
          }
        }
      });
      edges.forEach((edge) => {
        object(edge, `${path}.edge`); string(edge.id, `${path}.edge.id`);
        if (allIds.has(edge.id)) fail(path, `duplicate id ${edge.id}`); allIds.add(edge.id);
        const at = `${path}.edge.${edge.id}`;
        if (!EDGES.has(edge.type)) fail(at, 'invalid edge type');
        const s = byId.get(edge.s), t = byId.get(edge.t);
        if (!s || !t) fail(at, 'broken endpoint');
        string(edge.sourcePort, `${at}.sourcePort`); string(edge.targetPort, `${at}.targetPort`);
        if (!own(s.ports.outputs, edge.sourcePort) || !own(t.ports.inputs, edge.targetPort)) fail(at, 'missing edge port');
        const sourcePort = s.ports.outputs[edge.sourcePort], targetPort = t.ports.inputs[edge.targetPort];
        if (sourcePort.entity && targetPort.entity && sourcePort.entity !== targetPort.entity) fail(at, 'edge entity mismatch');
        if (sourcePort.dtype && targetPort.dtype && sourcePort.dtype !== targetPort.dtype) fail(at, 'edge dtype mismatch');
        if (own(edge, 'route')) string(edge.route, `${at}.route`);
      });
      // Loop dependencies are semantic feedback, independent of route/style.
      const degree = new Map(nodes.map((node) => [node.id, 0]));
      const successors = new Map(nodes.map((node) => [node.id, []]));
      edges.filter((edge) => edge.type !== 'feedback').forEach((edge) => {
        degree.set(edge.t, degree.get(edge.t) + 1); successors.get(edge.s).push(edge.t);
      });
      const ready = nodes.filter((node) => degree.get(node.id) === 0).map((node) => node.id);
      let visited = 0;
      while (ready.length) {
        const id = ready.shift(); visited += 1;
        successors.get(id).forEach((target) => { degree.set(target, degree.get(target) - 1); if (!degree.get(target)) ready.push(target); });
      }
      if (visited !== nodes.length) fail(path, 'forward dependency cycle; mark feedback semantically');
      list(graph.groups || [], `${path}.groups`).forEach((group) => {
        object(group, `${path}.group`); string(group.id, `${path}.group.id`);
        if (allIds.has(group.id)) fail(path, `duplicate id ${group.id}`); allIds.add(group.id);
        if (group.type !== 'region') fail(path, 'invalid group type');
        string(group.name, `${path}.group.name`); string(group.description, `${path}.group.description`);
        source(group.source, `${path}.group.source`, files);
        provenance(group.provenance, `${path}.group.provenance`, []);
        list(group.members, `${path}.group.members`).forEach((id) => { if (!byId.has(id)) fail(path, `unknown group member ${id}`); });
        if (own(group, 'ports')) ports(group.ports, `${path}.group.ports`);
      });
      return byId;
    }
    visit(model, 'graph');
    return model;
  }
  function materialize(model, expandedIds, options) {
    validate(model);
    return root.PTO_COMPUTATION_LAYOUT.layout(model, expandedIds, options);
  }
  root.PTO_SEMANTIC_COMPUTATION_GRAPH = { schemaVersion, validate, materialize };
})(typeof window === 'undefined' ? globalThis : window);
