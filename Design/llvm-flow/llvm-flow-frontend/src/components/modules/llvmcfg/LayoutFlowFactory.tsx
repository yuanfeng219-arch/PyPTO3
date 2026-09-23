import BaseLayoutFlow from './BaseLayoutFlow'
import { GraphJSON, ReviewNodePresentation } from '@/types/graph'

interface LayoutFlowFactoryProps {
  llvmJson: GraphJSON
  llvmJson_compare: GraphJSON
  llvmOutput?: string[]
  title: string
  paneLabel?: string
  variant: 'simpleSmall' | 'detailLarge' | 'reviewLarge'
  selectedBlockId?: string
  onSelectBlock?: (side: 'dev' | 'host', blockId: string) => void
  readOnly?: boolean
  showMinimap?: boolean
  nodePresentations?: Record<number, ReviewNodePresentation>
}

const LayoutFlowFactory = ({
  variant,
  ...baseProps
}: LayoutFlowFactoryProps) => {
  const variants = {
    simpleSmall: {
      nodeWidth: 45,
      nodeHeight: 45,
      defaultPosition: [150, 150] as [number, number],
      minZoom: 0.1,
      labelType: 'simple' as const,
    },
    detailLarge: {
      nodeWidth: 350,
      nodeHeight: 300,
      defaultPosition: [150, 0] as [number, number],
      minZoom: 0.05,
      labelType: 'detail' as const,
    },
    reviewLarge: {
      nodeWidth: 286,
      nodeHeight: 204,
      defaultPosition: [0, 0] as [number, number],
      minZoom: 0.2,
      labelType: 'review' as const,
    },
  }

  const layoutConfig = variants[variant]

  return <BaseLayoutFlow {...baseProps} layoutConfig={layoutConfig} />
}

export default LayoutFlowFactory
