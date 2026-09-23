export interface NodeFlow {
  id: string
  data: { id: string; label: string; name: string }
  type: string
  position: { x: number; y: number }
  isSame: string
}

export interface EdgeFlow {
  id: string
  source: string
  target: string
  type: string
  label?: string
  labelStyle: object
}

export interface GraphNode {
  _gvid: number
  name: string
  label: string
  shape: string
  metadata?: BlockMetadata
}

export interface ReviewNodePresentation {
  selectionId: string
  eyebrow: string
  title: string
  description: string
  confidence?: string
  tone?: 'neutral' | 'info' | 'success' | 'warning' | 'muted'
  diffStatus?: 'same' | 'added' | 'removed' | 'changed'
  tags?: string[]
  metrics?: Array<{ label: string; value: string }>
}

export interface BlockMetadata {
  block_id: string
  source_lines: number[]
  instruction_count: number
  opcodes: string[]
  terminator: string
  predecessors: string[]
  successors: Array<{ block_id: string; label?: string }>
}

export interface SourceData {
  dev: string
  host: string
  dev_path: string
  host_path: string
}

export interface GraphEdge {
  _gvid: number
  tail: number
  head: number
  tailport?: string
  label?: string
}

export interface GraphJSON {
  name: string
  directed: boolean
  strict: boolean
  label: string
  _subgraph_cnt: number
  objects: GraphNode[]
  edges: GraphEdge[]
}

export interface GraphState {
  before_json: GraphJSON
  before_output?: Array<string>
  after_json: GraphJSON
  after_output?: Array<string>
  file_pass?: string
  isReady?: boolean
  filterID?: number
  beforeg_data?: string
  afterg_data?: string
  source_data?: SourceData
}
