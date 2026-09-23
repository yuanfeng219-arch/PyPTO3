import React from 'react'
import { createRoot } from 'react-dom/client'
import LLVMcfg from '@/components/pages/llvmcfg/llvmcfg'
import './llvmcfg-standalone.scss'

const container = document.getElementById('root')

if (!container) {
  throw new Error('Missing #root mount point')
}

createRoot(container).render(
  <React.StrictMode>
    <LLVMcfg />
  </React.StrictMode>,
)
