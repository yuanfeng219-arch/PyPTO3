FROM node:22-bookworm-slim

WORKDIR /app
COPY . .
RUN npm ci --omit=dev --registry=https://registry.npmmirror.com

ENV NODE_ENV=production
ENV PORT=8787
ENV MCP_REMOTE_MODE=1
EXPOSE 8787

CMD ["node", "standard_http_server.mjs"]
