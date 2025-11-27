"""Minimal demo for indexing and querying."""
from pathlib import Path

from graphrag_langgraph.config import IndexConfig, QueryConfig
from graphrag_langgraph.engine import GraphRAGEngine
from graphrag_langgraph.types import Document


def main() -> None:
    docs = [
        Document(
            id="1",
            title="Chronicles",
            text=(
                "Aurora founded the city of Lumen on the coast.\n\n"
                "The council of Lumen collaborates with the island research guild.\n\n"
                "Trade routes connect Lumen with the northern port of Crest."
            ),
        ),
        Document(
            id="2",
            title="Relationships",
            text="Aurora mentors a young explorer named Kiri who studies ocean currents.",
        ),
    ]

    index_config = IndexConfig(vector_store_dir=Path(".demo_index"))
    query_config = QueryConfig()
    engine = GraphRAGEngine(index_config=index_config, query_config=query_config)

    print("Indexing corpus...")
    engine.index(docs)
    print("Index built. Communities:")
    for cid, summary in engine.index_store.community_summaries.items():
        print(f"- {cid}: {summary.summary_text}")

    questions = [
        "Give a global overview of the setting",
        "Who is Aurora mentoring?",
        "How is Lumen connected to other places?",
    ]
    for q in questions:
        print("\nQuestion:", q)
        answer = engine.answer(q)
        print(answer)


if __name__ == "__main__":
    main()
