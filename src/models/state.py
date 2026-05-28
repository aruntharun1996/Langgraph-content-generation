from typing import Optional
from pydantic import BaseModel, Field


class ContentState(BaseModel):
   
    user_request: str = Field(description="The original content request from the user.")

    generated_content: Optional[str] = Field(
        default=None,
        description="The most recently generated content.",
    )
    iteration: int = Field(
        default=0,
        description="How many generation attempts have been made.",
    )

    is_approved: Optional[bool] = Field(
        default=None,
        description="True if the evaluator approved the content.",
    )
    evaluation_feedback: Optional[str] = Field(
        default=None,
        description="Structured feedback from the evaluator (used on regeneration).",
    )
    evaluation_score: Optional[float] = Field(
        default=None,
        description="Quality score 0.0 – 10.0 assigned by the evaluator.",
    )

    previous_contents: list[str] = Field(
        default_factory=list,
        description="All previously generated versions, oldest first.",
    )
    previous_feedbacks: list[str] = Field(
        default_factory=list,
        description="All evaluator feedbacks, aligned with previous_contents.",
    )

    final_content: Optional[str] = Field(
        default=None,
        description="The approved, final content returned to the user.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Any error message captured during execution.",
    )

    class Config:
        arbitrary_types_allowed = True
