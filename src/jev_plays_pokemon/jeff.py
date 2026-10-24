"""Jeff: a local small model (Qwen3.5-2B on llama-server) that answers Jev's
Choice questions, so it drops in wherever a `decision.JevClient` goes.

Each Choice becomes one chat call whose answer is a single letter label. The
labels' first-token log probabilities, renormalized over the offered labels,
give each option its share, and the top share is the answer's confidence.
The options are shuffled every call to cancel the small model's bias toward
the first option.
"""

import json
import math
import random
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI
from typesafe_sdk import Choice, JSONContent

from jev_plays_pokemon.dialog_vision import VisionClient

# Every letter is one token in Qwen's tokenizer, and a full bag in battle
# offers up to 30 options, more than one case of letters holds.
_LABELS = string.ascii_uppercase + string.ascii_lowercase

_TOP_LOGPROBS = 20

_SYSTEM_PROMPT = (
    "You play Pokemon Red. You get the game state and a question with "
    "lettered options. Reply with the letter of the best option only."
)


class _TopLogprob(Protocol):
    @property
    def token(self) -> str: ...

    @property
    def logprob(self) -> float: ...


class _TokenLogprob(Protocol):
    @property
    def top_logprobs(self) -> Sequence[_TopLogprob]: ...


class _Logprobs(Protocol):
    @property
    def content(self) -> Sequence[_TokenLogprob]: ...


class _Choice(Protocol):
    @property
    def logprobs(self) -> _Logprobs: ...


class _ChatCompletion(Protocol):
    @property
    def choices(self) -> Sequence[_Choice]: ...


class JeffNoAnswer(Exception):
    """None of the offered labels is among Jeff's top first tokens."""


@dataclass(frozen=True)
class JeffAnswer:
    choice: str
    confidence: float


@dataclass(frozen=True)
class JeffResult:
    choices: Mapping[str, JeffAnswer]


class JeffClient:
    def __init__(
        self,
        client: VisionClient,
        *,
        shuffle: Callable[[list[str]], None] = random.shuffle,
    ) -> None:
        self._client = client
        self._shuffle = shuffle

    @classmethod
    def from_url(cls, base_url: str) -> "JeffClient":
        return cls(
            OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="not-required")
        )

    def system_one(
        self, state: JSONContent, questions: Mapping[str, Choice]
    ) -> JeffResult:
        return JeffResult(
            {
                question_id: self._answer(state, choice)
                for question_id, choice in questions.items()
            }
        )

    def _answer(self, state: JSONContent, choice: Choice) -> JeffAnswer:
        actions = list(choice.criteria)
        if len(actions) > len(_LABELS):
            raise ValueError(
                f"Jeff can label {len(_LABELS)} options, got {len(actions)}"
            )
        self._shuffle(actions)
        labelled = dict(zip(_LABELS, actions, strict=False))
        options = "\n".join(
            f"{label}: {choice.criteria[action] or action}"
            for label, action in labelled.items()
        )

        response: _ChatCompletion = self._client.chat.completions.create(
            model="jeff",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"State: {json.dumps(state)}\n\n"
                    f"{choice.instructions}\n{options}",
                },
            ],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=_TOP_LOGPROBS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        shares = {
            labelled[entry.token]: math.exp(entry.logprob)
            for entry in response.choices[0].logprobs.content[0].top_logprobs
            if entry.token in labelled
        }
        if not shares:
            raise JeffNoAnswer(
                f"no offered label among Jeff's top {_TOP_LOGPROBS} tokens"
            )
        best = max(shares, key=shares.__getitem__)
        return JeffAnswer(choice=best, confidence=shares[best] / sum(shares.values()))
