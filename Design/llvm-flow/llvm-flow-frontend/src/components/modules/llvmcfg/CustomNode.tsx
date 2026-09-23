/* eslint-disable camelcase */
import { memo, useState } from 'react'
import './CustomNode.scss'
import { Handle, Position } from 'react-flow-renderer'
import { COLORS } from '@/const/color'
import { ReviewNodePresentation } from '@/types/graph'

interface NodeData {
  block_id: string;
  type: string;
  label: string[];
  isSame: 'yes' | 'no';
  isSelected: boolean;
  presentation?: ReviewNodePresentation;
}

interface CustomNodeProps {
  data: NodeData & {
    layoutConfig: {
      nodeWidth: number;
      nodeHeight: number;
      labelType: 'simple' | 'detail' | 'review';
    };
  };
}

// eslint-disable-next-line react/display-name
export default memo(({ data }: CustomNodeProps) => {
  const [localFullMode, setLocalFullMode] = useState<boolean>(false)
  const { layoutConfig } = data

  if (layoutConfig.labelType === 'review' && data.presentation) {
    const presentation = data.presentation
    return (
      <>
        <Handle type="target" position={Position.Top} isConnectable={false} />
        <button
          type="button"
          className={`review-flow-node review-flow-node--${
            presentation.tone ?? 'neutral'
          } review-flow-node--diff-${
            presentation.diffStatus ?? 'same'
          } ${data.isSelected ? 'is-selected' : ''}`}
          id={data.block_id}
          name={data.type}
          aria-label={`${presentation.eyebrow}: ${presentation.title}`}
          style={{ width: layoutConfig.nodeWidth, height: layoutConfig.nodeHeight }}
        >
          <span className="review-flow-node__eyebrow">
            {presentation.eyebrow}
            {presentation.confidence ? <em>{presentation.confidence}</em> : null}
          </span>
          <strong>{presentation.title}</strong>
          <p>{presentation.description}</p>
          {presentation.metrics?.length ? (
            <span className="review-flow-node__metrics">
              {presentation.metrics.map((metric) => (
                <span key={metric.label}>
                  <b>{metric.value}</b>
                  <small>{metric.label}</small>
                </span>
              ))}
            </span>
          ) : null}
          {presentation.tags?.length ? (
            <span className="review-flow-node__tags">
              {presentation.tags.map((tag) => <i key={tag}>{tag}</i>)}
            </span>
          ) : null}
        </button>
        <Handle type="source" position={Position.Bottom} isConnectable={false} />
      </>
    )
  }

  const handleFull = () => {
    setLocalFullMode(!localFullMode)
  }

  // 전체 모드가 detail이면 기본적으로 detail 모드
  // 전체 모드가 simple이면 기본적으로 simple 모드
  // localFullMode로 개별 노드의 모드를 토글
  const isDetailMode = layoutConfig.labelType === 'detail' 
    ? !localFullMode  // detail 모드에서는 localFullMode가 true일 때 simple로
    : localFullMode   // simple 모드에서는 localFullMode가 true일 때 detail로

  // detail 모드일 때는 큰 크기, simple 모드일 때는 작은 크기
  const nodeWidth = isDetailMode ? 350 : 45
  const nodeHeight = isDetailMode ? 300 : 45

  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        id="a"
        isConnectable={false}
        style={{
          left: '20%',
          right: 'auto',
          background: 'transparent',
          border: 'transparent',
        }}
      />
      <Handle
        type="source"
        position={Position.Top}
        id="b"
        isConnectable={false}
        style={
          `${data.isSame}` === 'yes'
            ? {
                right: '20%',
                left: 'auto',
                background: COLORS.PURPLE,
                border: 'none',
                borderRadius: '0px',
                height: '3.5px',
              }
            : {
                right: '20%',
                left: 'auto',
                background: COLORS.GRAY,
                border: 'none',
                borderRadius: '0px',
                height: '3.5px',
              }
        }
      />
      <button
        className={`${data.isSame} ${data.isSelected ? 'is-selected' : ''}`}
        onDoubleClick={handleFull}
        id={data.block_id}
        name={data.type}
        aria-label={`${data.type} Basic Block ${data.block_id.replace(data.type, '')}`}
        style={{ 
          width: `${nodeWidth}px`,
          height: `${nodeHeight}px`,
          transition: 'all 0.3s ease',
          overflow: 'auto',
          whiteSpace: isDetailMode ? 'normal' : 'nowrap',
          padding: isDetailMode ? 'var(--space-2)' : '0',
          textAlign: isDetailMode ? 'left' : 'center'
        }}
      >
        {isDetailMode ? (
          <div>
            {data.label.map(function (item: string, i: number) {
              return <p key={i}>{item}</p>
            })}
          </div>
        ) : (
          <>{data.block_id.substring(data.block_id.indexOf('%'))}</>
        )}
      </button>
      <Handle
        type="source"
        position={Position.Bottom}
        id="a"
        isConnectable={false}
        style={
          `${data.isSame}` === 'yes'
            ? {
                right: '20%',
                left: 'auto',
                background: COLORS.PURPLE,
                border: 'none',
                borderRadius: '0px',
                height: '3.5px',
              }
            : {
                right: '20%',
                left: 'auto',
                background: COLORS.GRAY,
                border: 'none',
                borderRadius: '0px',
                height: '3.5px',
              }
        }
      />
      <Handle
        type="target"
        position={Position.Left}
        id="b"
        isConnectable={false}
        style={{
          background: 'transparent',
          border: 'transparent',
        }}
      />
      <Handle
        type="target"
        position={Position.Right}
        id="b"
        isConnectable={false}
        style={{
          background: 'transparent',
          border: 'transparent',
        }}
      />
    </>
  )
})
