import re
from graphrag_langgraph2.models.input import InputConfig
from graphrag_langgraph2.storage.pipeline_storage import PipelineStorage
from pathlib import Path
from graphrag_langgraph2.utils.hash import gen_sha512_hash
from langchain_core.documents import Document

async def load_text(
    config: InputConfig,
    storage: PipelineStorage,
) -> list[Document]:
    async def load_file(path: str) -> Document:
        text = await storage.get(path, encoding=config.encoding)
        return Document(
            id=gen_sha512_hash(text),
            page_content=text,
            metadata={
                "source": path,
                "creation_date": await storage.get_creation_date(path),
            },
        )
        
    files = list(
        storage.find(
            re.compile(config.file_pattern),
            file_filter=config.file_filter,
        )
    )
    
    if len(files) == 0:
        msg = f"No {config.file_type} files found"
        raise ValueError(msg)

    return [await load_file(file[0]) for file in files]