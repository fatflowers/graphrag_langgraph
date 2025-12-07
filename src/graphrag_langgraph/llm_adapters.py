"""
LangChain adapters that wrap the upstream GraphRAG model configuration and prompts.

These helpers are intentionally thin: they reuse upstream prompt templates and model
configuration, but present LangChain-friendly primitives for LangGraph nodes.
"""

from __future__ import annotations

from typing import Any, Sequence

from graphrag.config.defaults import DEFAULT_CHAT_MODEL_ID, DEFAULT_EMBEDDING_MODEL_ID
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.language_model.manager import ModelManager
from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT
from graphrag.prompts.index.extract_claims import EXTRACT_CLAIMS_PROMPT
from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT
from graphrag.prompts.query.drift_search_system_prompt import (
    DRIFT_LOCAL_SYSTEM_PROMPT,
    DRIFT_REDUCE_PROMPT,
)
from graphrag.prompts.query.global_search_knowledge_system_prompt import (
    GENERAL_KNOWLEDGE_INSTRUCTION,
)
from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
from graphrag.prompts.query.global_search_reduce_system_prompt import (
    REDUCE_SYSTEM_PROMPT,
)
from graphrag.prompts.query.local_search_system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)
from graphrag.prompts.query.question_gen_system_prompt import QUESTION_SYSTEM_PROMPT
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field

__all__ = [
    "get_llm",
    "get_embeddings",
    "PromptBundle",
    "load_index_prompts",
    "load_query_prompts",
]


class PromptBundle(dict):
    """Namespace for related prompt templates."""

    def to_prompt_template(self) -> PromptTemplate:
        """Convert a single template to a LangChain PromptTemplate if possible."""
        if "template" in self:
            return PromptTemplate.from_template(self["template"])
        msg = "PromptBundle does not contain a 'template' key; expand this helper as needed."
        raise KeyError(msg)


def get_llm(config: GraphRagConfig, model_id: str | None = None) -> BaseLanguageModel:
    """
    Return a LangChain `BaseLanguageModel` aligned with a GraphRAG model definition.

    Wraps the upstream `ModelManager` ChatModel instance in a minimal LangChain
    `BaseChatModel` adapter so LangGraph nodes can compose it with other LC tools.
    """
    model_key = model_id or DEFAULT_CHAT_MODEL_ID
    model_settings = config.get_language_model_config(model_key)
    chat_model = ModelManager().get_or_create_chat_model(
        name=model_key,
        model_type=model_settings.type,
        config=model_settings,
    )
    return _ChatModelAdapter(chat_model, model_settings.model)


def get_embeddings(
    config: GraphRagConfig, model_id: str | None = None
) -> Embeddings:
    """
    Return LangChain `Embeddings` aligned with a GraphRAG embedding model definition.

    As with `get_llm`, this wraps upstream embedding providers (OpenAI/Azure/etc.)
    while presenting the LangChain interface.
    """
    model_key = model_id or DEFAULT_EMBEDDING_MODEL_ID
    embed_settings = config.get_language_model_config(model_key)
    embedding_model = ModelManager().get_or_create_embedding_model(
        name=model_key,
        model_type=embed_settings.type,
        config=embed_settings,
    )
    return _EmbeddingAdapter(embedding_model)


def load_index_prompts() -> dict[str, PromptBundle]:
    """Expose upstream indexing prompts for LangChain nodes."""
    return {
        "extract_graph": PromptBundle(template=GRAPH_EXTRACTION_PROMPT),
        "extract_claims": PromptBundle(template=EXTRACT_CLAIMS_PROMPT),
        "community_report": PromptBundle(template=COMMUNITY_REPORT_PROMPT),
        "summarize_descriptions": PromptBundle(template=SUMMARIZE_PROMPT),
    }


def load_query_prompts() -> dict[str, PromptBundle]:
    """Expose upstream query prompts for LangChain nodes."""
    return {
        "global_map": PromptBundle(template=MAP_SYSTEM_PROMPT),
        "global_reduce": PromptBundle(template=REDUCE_SYSTEM_PROMPT),
        "global_knowledge": PromptBundle(template=GENERAL_KNOWLEDGE_INSTRUCTION),
        "local": PromptBundle(template=LOCAL_SEARCH_SYSTEM_PROMPT),
        "question_gen": PromptBundle(template=QUESTION_SYSTEM_PROMPT),
        "basic": PromptBundle(template=BASIC_SEARCH_SYSTEM_PROMPT),
        "drift_local": PromptBundle(template=DRIFT_LOCAL_SYSTEM_PROMPT),
        "drift_reduce": PromptBundle(template=DRIFT_REDUCE_PROMPT),
    }


def run_template_prompt(prompt: PromptTemplate, **kwargs: Any) -> str:
    """
    Format a prompt template without executing it.

    This helper is mainly for debugging and parity checks; actual LLM calls should
    be wired through LangChain models returned by `get_llm`.
    """
    return prompt.format(**kwargs)


class _ChatModelAdapter(BaseChatModel):
    """LangChain ChatModel wrapper around the upstream GraphRAG ChatModel."""

    chat_model: Any
    model_name: str = Field(default="graphrag-chat")

    class Config:
        """Pydantic config."""

        extra = Extra.allow
        arbitrary_types_allowed = True

    def __init__(self, chat_model: Any, model_name: str):
        super().__init__(chat_model=chat_model, model_name=model_name)
        self.chat_model = chat_model
        self.model_name = model_name

    def _messages_to_prompt(self, messages: Sequence[BaseMessage]) -> str:
        """Flatten LC chat messages into a single prompt string."""
        system_parts = []
        user_parts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_parts.append(m.content)
            elif isinstance(m, HumanMessage):
                user_parts.append(m.content)
            else:
                # Fallback for other message types
                user_parts.append(getattr(m, "content", str(m)))
        parts = []
        if system_parts:
            parts.append("\n\n".join(system_parts))
        if user_parts:
            parts.append("\n\n".join(user_parts))
        return "\n\n".join(parts)

    def _generate(
        self, messages: Sequence[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        prompt = self._messages_to_prompt(messages)
        response = self.chat_model.chat(prompt, history=None, **kwargs)
        content = getattr(response, "output", None)
        if content is not None and hasattr(content, "content"):
            text = content.content
        else:
            text = str(response)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self, messages: Sequence[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        prompt = self._messages_to_prompt(messages)
        response = await self.chat_model.achat(prompt, history=None, **kwargs)
        content = getattr(response, "output", None)
        if content is not None and hasattr(content, "content"):
            text = content.content
        else:
            text = str(response)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return self.model_name


class _EmbeddingAdapter(Embeddings):
    """LangChain Embeddings wrapper around the upstream GraphRAG EmbeddingModel."""

    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.embed_batch(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embedding_model.aembed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedding_model.embed(text)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.embedding_model.aembed(text)
