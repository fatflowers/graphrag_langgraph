# GraphRAG LangGraph Layer

This project provides a LangGraph-based orchestration layer around the official
Microsoft GraphRAG implementation installed from the local repo at
`/Users/simon/learn/graphrag`.

## Generating `settings.yaml`

GraphRAG uses a YAML config file (commonly called `settings.yaml` or `graphrag.yml`)
that describes models, input/output storage, prompts, and query behaviour.

To generate a starter config:

1. **Go to the official GraphRAG repo** (the upstream library):

   ```bash
   cd /Users/simon/learn/graphrag
   ```

2. **Run the official init command** to scaffold a new project:

   ```bash
   uv run python -m graphrag init --root /Users/simon/learn/graphrag_langgraph
   ```

   This will:

   - Create a `settings.yaml` file in `/Users/simon/learn/graphrag_langgraph`
   - Create default `input/`, `output/`, `cache/`, `logs/`, `prompts/` directories
   - Populate model, storage, and prompt defaults compatible with GraphRAG

3. **Adjust the generated `settings.yaml`** in
   `/Users/simon/learn/graphrag_langgraph/settings.yaml` as needed:

   - Set `models.default_chat_model.api_key` and `models.default_embedding_model.api_key`
     via `${GRAPHRAG_API_KEY}` in the file and define `GRAPHRAG_API_KEY` in `.env`
   - Ensure `input.storage.base_dir` points to your corpus directory (default `input`)
   - Ensure `output.base_dir` points to your desired index directory (default `output`)

4. **Point the LangGraph CLI at this config**:

   ```bash
   cd /Users/simon/learn/graphrag_langgraph
   . .env 2>/dev/null || true

   # Standard indexing:
   uv run graphrag-langgraph index --config settings.yaml --method standard

   # Global query:
   uv run graphrag-langgraph query --config settings.yaml --mode global --question "What is GraphRAG?"
   ```

The LangGraph layer reads the same `settings.yaml` as the official GraphRAG CLI, so
you can reuse a single configuration file for both implementations.

