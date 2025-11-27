from graphrag_langgraph.config import IndexConfig, QueryConfig
from graphrag_langgraph.engine import GraphRAGEngine
from graphrag_langgraph.indexing.pipeline import build_index_graph, run_indexing
from graphrag_langgraph.types import Document


def build_index_store():
    docs = [
        Document(id="doc1", title="World", text="The city of Solis thrives near the river.\n\nNova leads the council."),
        Document(id="doc2", title="Relations", text="Solis trades with the mountain town of Kora."),
    ]
    index_config = IndexConfig(persist_graph=False)
    graph = build_index_graph(index_config)
    state = run_indexing(graph, docs, index_config)
    return state.index_store


def test_global_and_local_modes():
    index_store = build_index_store()
    engine = GraphRAGEngine(index_config=IndexConfig(persist_graph=False), query_config=QueryConfig())
    engine.index_store = index_store

    answer_global = engine.answer("Provide a global overview of this world", mode="global")
    assert "Answer" in answer_global

    answer_local = engine.answer("Who leads Solis?", mode="local")
    assert "Answer" in answer_local


def test_basic_mode():
    index_store = build_index_store()
    engine = GraphRAGEngine(index_config=IndexConfig(persist_graph=False), query_config=QueryConfig())
    engine.index_store = index_store

    answer = engine.answer("Tell me about the river", mode="basic")
    assert "Answer" in answer
