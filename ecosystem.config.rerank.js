module.exports = {
  apps: [
    {
      name: 'rerank-api',
      script: '.venv/bin/uvicorn',
      args: 'main:app --host 0.0.0.0 --port 8004',
      interpreter: 'none',
      cwd: __dirname,
      autorestart: true,
      watch: false,
      max_memory_restart: '4G',
      env: {
        RERANK_MODEL: 'BAAI/bge-reranker-v2-m3',
        TRANSFORMERS_OFFLINE: '1',
        HF_DATASETS_OFFLINE: '1',
      },
    },
  ],
};
