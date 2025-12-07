from typing import Callable, Awaitable

from graphrag_langgraph2.loaders.text import load_text
from graphrag_langgraph2.models.input import InputConfig, InputFileType
from graphrag_langgraph2.storage.pipeline_storage import PipelineStorage

from langchain_core.documents import Document

loaders: dict[str, Callable[..., Awaitable[list[Document]]]] = {
    InputFileType.text: load_text,
    # InputFileType.csv: load_csv,
    # InputFileType.json: load_json,
}

async def load_input(
    config: InputConfig,
    storage: PipelineStorage,
) -> list[Document]:
    loader = loaders[config.file_type]
    if loader is None:
        msg = f"No loader found for {config.file_type}"
        raise ValueError(msg)
    return await loader(config, storage)