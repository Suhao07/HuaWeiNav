"""Semantic room choice shared by backend exploration adapters."""
from pydantic import BaseModel

from llm_utils.cognav_llm_adapter import get_client_and_model
from prompting.templates import ROOM_PROMPT


class RoomReasoningStep(BaseModel):
    explanation: str
    output: str


class RoomSelection(BaseModel):
    steps: list[RoomReasoningStep]
    final_answer: int
    reason: str


def choose_semantic_room(prompt, candidate_ids, *, vlm):
    client, model = get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": ROOM_PROMPT},
                  {"role": "user", "content": prompt}],
        response_format=RoomSelection, trace_label="room_selection")
    answer = completion.choices[0].message.parsed
    if answer is None or answer.final_answer not in candidate_ids:
        raise ValueError("Room selection must name a currently available room")
    return answer
