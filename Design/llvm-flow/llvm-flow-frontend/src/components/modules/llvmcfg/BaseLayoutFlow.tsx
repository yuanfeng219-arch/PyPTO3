import { useCallback, useEffect, useRef, useState } from 'react'
import ReactFlow, {
  addEdge,
  ConnectionLineType,
  useNodesState,
  useEdgesState,
  MiniMap,
  Background,
  Node,
  Edge,
  MarkerType,
  Connection,
  Position,
  ReactFlowInstance,
} from 'react-flow-renderer'
import dagre from 'dagre'
import CustomNode from './CustomNode'
import './LayoutFlow.scss'
import {
  GraphJSON,
  GraphNode,
  GraphEdge,
  ReviewNodePresentation,
} from '@/types/graph'

const nodeTypes = {
  selectorNode: CustomNode,
}

interface LayoutConfig {
  nodeWidth: number
  nodeHeight: number
  defaultPosition: [number, number]
  minZoom: number
  labelType: 'simple' | 'detail' | 'review'
}

export interface BaseLayoutFlowProps {
  llvmJson: GraphJSON
  llvmJson_compare: GraphJSON
  llvmOutput?: string[]
  title: string
  paneLabel?: string
  layoutConfig: LayoutConfig
  selectedBlockId?: string
  onSelectBlock?: (side: 'dev' | 'host', blockId: string) => void
  readOnly?: boolean
  showMinimap?: boolean
  nodePresentations?: Record<number, ReviewNodePresentation>
}

interface ExtendedNode extends GraphNode {
  isSame: 'yes' | 'no'
}

interface FlowNodeData {
  type: string
  label: string[]
  name: string
  id: string
  isSame: 'yes' | 'no'
  block_id: string
  layoutConfig: LayoutConfig
  isSelected: boolean
  presentation?: ReviewNodePresentation
  selectionId: string
}

const BaseLayoutFlow = ({
  llvmJson,
  llvmJson_compare,
  llvmOutput = [],
  title,
  paneLabel,
  layoutConfig,
  selectedBlockId,
  onSelectBlock,
  readOnly = false,
  showMinimap = true,
  nodePresentations = {},
}: BaseLayoutFlowProps) => {
  const [layoutDirection, setLayoutDirection] = useState<'TB' | 'LR'>('TB')
  const [legendOpen, setLegendOpen] = useState<boolean>(false)
  const nodeinitial = llvmJson.objects
  const node = nodeinitial.map((object: GraphNode) => {
    return { ...object, isSame: 'no' as const }
  })
  const edge = llvmJson.edges || [
    {
      _gvid: 0,
      tail: 0,
      head: 0,
    },
  ]

  const numberOfNode = llvmJson.objects.length
  const numberOfEdge = edge.length

  const position = { x: 0, y: 0 }

  // step2) basicblock id 지정 (ex. %210)
  const blockID = node.map(({ label }: ExtendedNode) =>
    label.replace(/[{}]/g, '').split(/\\l/)[0].slice(0, -1),
  )

  // step3) 같은 basicblock 찾기 + entry basic block의 경우 따로 비교
  function checkSameBlock(
    json: string[],
    output: string[],
    data: ExtendedNode[],
  ) {
    for (let i = 0; i < json.length; i++) {
      for (let j = 0; j < output.length; j++) {
        if (json[i] === output[j]) {
          data[i].isSame = 'yes'
        }
      }
    }
  }
  checkSameBlock(blockID, llvmOutput, node)

  function checkSameEntryBlock(
    json: GraphNode[],
    json_compare: GraphNode[],
    data: ExtendedNode[],
  ) {
    if (json[0].label === json_compare[0].label) {
      data[0].isSame = 'yes'
    }
  }
  checkSameEntryBlock(llvmJson.objects, llvmJson_compare.objects, node)

  // step*) edge tailport와 node 정보 연결하기
  function connectTailport(
    tailport: string | undefined,
    tail: number,
  ): string | null {
    if (tailport) {
      const tailLabel = node.find(
        (node: ExtendedNode) => node._gvid === tail,
      )?.label
      if (!tailLabel) return null
      const text = tailLabel.substring(tailLabel.indexOf(tailport) + 3)
      if (text.includes('|')) {
        return text.substring(0, text.indexOf('|'))
      } else {
        return text.substring(0, text.indexOf('}'))
      }
    } else {
      return null
    }
  }

  // setp**) 위에서 아래로 target이 되는 경우, targetHandleID 설정
  function setTargetHandleID(
    tail: number,
    head: number,
  ): 'a' | 'b' | undefined {
    const tailNode = node.find((node: ExtendedNode) => node._gvid === tail)
    const headNode = node.find((node: ExtendedNode) => node._gvid === head)
    if (!tailNode || !headNode) return undefined

    const tailLabel = tailNode.label
      .replace(/[{}]/g, '')
      .split(/\\l/)[0]
      .slice(0, -1)

    const headLabel = headNode.label
      .replace(/[{}]/g, '')
      .split(/\\l/)[0]
      .slice(0, -1)

    if (tailLabel > headLabel) {
      return 'b'
    } else if (tailLabel < headLabel) {
      return 'a'
    }
    return undefined
  }

  // step4) node, edge 정의
  const initialNode: Node<FlowNodeData>[] = node.map(
    ({ _gvid, name, label, isSame }: ExtendedNode) => ({
      id: _gvid.toString(),
      data: {
        type: title,
        label: label.replace(/[{}]/g, '').split(/\\l/),
        name: name.replace('Node', ''),
        id: _gvid.toString(),
        isSame: isSame,
        block_id:
          title + label.replace(/[{}]/g, '').split(/\\l/)[0].slice(0, -1),
        layoutConfig: layoutConfig,
        isSelected: false,
        presentation: nodePresentations[_gvid],
        selectionId:
          nodePresentations[_gvid]?.selectionId ??
          label.replace(/[{}]/g, '').split(/\\l/)[0].slice(0, -1),
      },
      type: 'selectorNode',
      position: position,
    }),
  )

  const initialEdge = edge.map(
    ({ _gvid, tail, head, tailport, label }: GraphEdge) => ({
      id: _gvid.toString(),
      source: tail.toString(),
      target: head.toString(),
      type: 'smoothstep',
      animated: false,
      targetHandle: setTargetHandleID(tail, head),
      sourceHandle: setTargetHandleID(tail, head),
      label: label ?? connectTailport(tailport, tail),
      labelStyle: {
        fontFamily: 'var(--font-mono)',
        fontSize: '12px',
        fontWeight: 500,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
      },
    }),
  )

  // step5) react-flow 설정
  // dagre 레이아웃 적용
  const dagreGraph = new dagre.graphlib.Graph()
  dagreGraph.setDefaultEdgeLabel(() => ({}))
  const getLayoutedElements = (
    nodes: Node<FlowNodeData>[],
    edges: Edge[],
    direction: 'TB' | 'LR' = 'TB',
  ) => {
    const isHorizontal = direction === 'LR'
    dagreGraph.setGraph({ rankdir: direction })
    nodes.forEach((node) => {
      dagreGraph.setNode(node.id, {
        width: layoutConfig.nodeWidth,
        height: layoutConfig.nodeHeight,
      })
    })
    edges.forEach((edge) => {
      dagreGraph.setEdge(edge.source, edge.target)
    })
    dagre.layout(dagreGraph)
    nodes.forEach((node) => {
      const nodeWithPosition = dagreGraph.node(node.id)
      node.targetPosition = isHorizontal ? Position.Left : Position.Top
      node.sourcePosition = isHorizontal ? Position.Right : Position.Bottom
      node.position = {
        x: nodeWithPosition.x - layoutConfig.nodeWidth / 2,
        y: nodeWithPosition.y - layoutConfig.nodeHeight / 2,
      }
      return node
    })
    return { nodes, edges }
  }
  const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
    initialNode,
    initialEdge,
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges)
  const flowInstance = useRef<ReactFlowInstance | null>(null)

  useEffect(() => {
    setNodes((currentNodes) =>
      currentNodes.map((flowNode) => ({
        ...flowNode,
        data: {
          ...flowNode.data,
          isSelected:
            flowNode.data.selectionId === selectedBlockId ||
            flowNode.data.block_id.endsWith(selectedBlockId || '__none__'),
        },
      })),
    )

    if (!selectedBlockId || !flowInstance.current) return
    const target = nodes.find(
      (flowNode) =>
        flowNode.data.selectionId === selectedBlockId ||
        flowNode.data.block_id.endsWith(selectedBlockId),
    )
    if (target) {
      flowInstance.current.setCenter(
        target.position.x + layoutConfig.nodeWidth / 2,
        target.position.y + layoutConfig.nodeHeight / 2,
        { zoom: 0.9, duration: 450 },
      )
    }
  }, [selectedBlockId, layoutConfig.nodeHeight, layoutConfig.nodeWidth])

  const nodeColor = (node: Node<FlowNodeData>) => {
    switch (node.data.isSame) {
      case 'yes':
        return 'var(--primary)'
      case 'no':
        return 'var(--foreground-muted)'
      default:
        return 'var(--foreground-muted)'
    }
  }

  const onConnect = useCallback(
    (params: Connection) =>
      setEdges((eds) =>
        addEdge(
          { ...params, type: ConnectionLineType.SmoothStep, animated: false },
          eds,
        ),
      ),
    [],
  )

  const onLayout = useCallback(
    (direction: 'TB' | 'LR') => {
      const { nodes: layoutedNodes, edges: layoutedEdges } =
        getLayoutedElements(nodes, edges, direction)
      setNodes([...layoutedNodes])
      setEdges([...layoutedEdges])
    },
    [nodes, edges],
  )

  return (
    <section className="cfg-frame" aria-label={`${title} control-flow graph`}>
      <header className="pto-ide-frame__pane-header cfg-frame__header">
        <div>
          <span className="pto-ide-frame__pane-title">
            {paneLabel ?? title}
          </span>
          <span className="pto-ide-frame__pane-meta">
            {numberOfNode} blocks · {numberOfEdge} edges
          </span>
        </div>
        {!readOnly ? (
          <div className="cfg-frame__actions">
            <div
              className="tab-control cfg-layout-tabs"
              role="tablist"
              aria-label={`${title} layout`}
            >
              <button
                type="button"
                className={`tab-control-item ${
                  layoutDirection === 'TB' ? 'is-selected' : ''
                }`}
                onClick={() => {
                  onLayout('TB')
                  setLayoutDirection('TB')
                }}
                role="tab"
                aria-selected={layoutDirection === 'TB'}
              >
                Vertical
              </button>
              <button
                type="button"
                className={`tab-control-item ${
                  layoutDirection === 'LR' ? 'is-selected' : ''
                }`}
                onClick={() => {
                  onLayout('LR')
                  setLayoutDirection('LR')
                }}
                role="tab"
                aria-selected={layoutDirection === 'LR'}
              >
                Horizontal
              </button>
            </div>
            <button
              type="button"
              className={`btn btn-icon btn-ghost cfg-legend-toggle ${
                legendOpen ? 'is-active' : ''
              }`}
              onClick={() => setLegendOpen((open) => !open)}
              aria-expanded={legendOpen}
              aria-label="图例说明"
              title="图例"
            >
              <svg
                viewBox="0 0 24 24"
                width="16"
                height="16"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="9" />
                <path d="M12 11.5v4.5" />
                <circle
                  cx="12"
                  cy="8"
                  r="0.5"
                  fill="currentColor"
                  stroke="none"
                />
              </svg>
            </button>
          </div>
        ) : null}
      </header>
      <div className="cfg-frame__canvas">
        {legendOpen && (
          <div className="cfg-legend" aria-label="图例">
            <div className="cfg-legend__row">
              <span
                className="cfg-legend__box cfg-legend__box--outline"
                aria-hidden="true"
              />
              <span>
                基本块：顺序执行的一段指令，方框 <code>%N</code> 是编号
              </span>
            </div>
            <div className="cfg-legend__row">
              <span className="cfg-legend__glyph" aria-hidden="true">
                →
              </span>
              <span>跳转：箭头表示控制流流向</span>
            </div>
            <div className="cfg-legend__row">
              <span className="cfg-legend__glyph" aria-hidden="true">
                T / F
              </span>
              <span>分支：T = 条件为真，F = 条件为假</span>
            </div>
            <div className="cfg-legend__row">
              <span className="cfg-legend__glyph" aria-hidden="true">
                ↩
              </span>
              <span>回边：指回上方的边，构成循环</span>
            </div>
            <div className="cfg-legend__row">
              <span
                className="cfg-legend__box cfg-legend__box--same"
                aria-hidden="true"
              />
              <span>与对侧一致（未变）</span>
            </div>
            <div className="cfg-legend__row">
              <span
                className="cfg-legend__box cfg-legend__box--diff"
                aria-hidden="true"
              />
              <span>本侧独有（新增 / 删除）</span>
            </div>
          </div>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          connectionLineType={ConnectionLineType.SmoothStep}
          defaultPosition={layoutConfig.defaultPosition}
          defaultZoom={0.5}
          minZoom={layoutConfig.minZoom}
          onInit={(instance) => {
            flowInstance.current = instance
          }}
          onNodeClick={(_event, flowNode) =>
            onSelectBlock?.(title as 'dev' | 'host', flowNode.data.selectionId)
          }
          fitView
          fitViewOptions={{ padding: 0.2 }}
          nodesConnectable={false}
          nodesDraggable={!readOnly}
        >
          <Background gap={20} size={1} />
          {showMinimap ? (
            <MiniMap nodeColor={nodeColor} nodeStrokeWidth={3} />
          ) : null}
        </ReactFlow>
      </div>
    </section>
  )
}

export default BaseLayoutFlow
