from pydantic import BaseModel, Field
from pathlib import Path
from .models.input import InputFileType

class GraphRagConfig(BaseModel):
    file_type: InputFileType = Field(
        description="The input file type to use.",
        default=InputFileType.text,
    )
    input_dir: Path = Field(
        description="The input directory to use.",
        default=Path("input"),
    )