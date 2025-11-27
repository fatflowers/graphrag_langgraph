import shutil
from pathlib import Path

from graphrag_langgraph.config import IndexConfig
from graphrag_langgraph.indexing.pipeline import build_index_graph, run_indexing
from graphrag_langgraph.types import Document


def test_indexing_pipeline_creates_artifacts(tmp_path: Path):
    docs = [
        Document(id="doc1", title="Story", text="Alpha builds a city.\n\nBeta explores the forest."),
        Document(id="doc2", title="Report", text="Gamma meets Alpha in the market."),
    ]
    config = IndexConfig(vector_store_dir=tmp_path / "index")
    graph = build_index_graph(config)
    final_state = run_indexing(graph, docs, config)

    assert final_state.text_units, "Text units should be produced"
    assert final_state.entities, "Entities should be extracted"
    assert final_state.index_store is not None
    assert final_state.communities, "Communities should be detected"

    if config.persist_graph:
        assert (config.vector_store_dir / "entities.json").exists()

    # cleanup
    shutil.rmtree(config.vector_store_dir)
