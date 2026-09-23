/* eslint-disable @typescript-eslint/no-var-requires */
const CracoAlias = require('craco-alias')
const path = require('path')
const webpack = require('webpack')

const isLlvmCfgStandalone = process.env.LLVM_CFG_STANDALONE === '1'

module.exports = {
  plugins: [
    {
      plugin: CracoAlias,
      options: {
        source: 'tsconfig',
        tsConfigPath: 'tsconfig.json',
      },
    },
  ],
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.module.rules.push({
        test: /\.py$/i,
        type: 'asset/source',
      })

      if (!isLlvmCfgStandalone) return webpackConfig

      webpackConfig.entry = path.resolve(
        __dirname,
        'src/llvmcfg-standalone.tsx',
      )
      webpackConfig.output.filename = 'static/js/llvmcfg.js'
      webpackConfig.output.chunkFilename = 'static/js/llvmcfg.[id].js'
      webpackConfig.optimization.runtimeChunk = false
      webpackConfig.optimization.splitChunks = false
      webpackConfig.plugins.push(
        new webpack.optimize.LimitChunkCountPlugin({ maxChunks: 1 }),
      )

      return webpackConfig
    },
  },
}
