import math
from types import SimpleNamespace

import pytest
from typesafe_sdk import Choice

from jev_plays_pokemon.jeff import JeffClient, JeffNoAnswer


class _FakeCompletions:
    def __init__(self, top_logprobs: dict[str, float]) -> None:
        self._top_logprobs = top_logprobs
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        top = [
            SimpleNamespace(token=token, logprob=math.log(probability))
            for token, probability in self._top_logprobs.items()
        ]
        content = [SimpleNamespace(top_logprobs=top)]
        return SimpleNamespace(
            choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))]
        )


def _jeff(top_logprobs: dict[str, float]) -> tuple[JeffClient, _FakeCompletions]:
    completions = _FakeCompletions(top_logprobs)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return JeffClient(client, shuffle=lambda actions: actions.reverse()), completions


_QUESTION = Choice(
    instructions="Which single action should be taken next?",
    criteria={"up": "Move up", "a": "Confirm", "start": "Open the start menu"},
)


def test_answers_with_the_shuffled_action_behind_the_top_label():
    jeff, completions = _jeff({"A": 0.6, "B": 0.2, "The": 0.2})

    answer = jeff.system_one(
        {"map_name": "PALLET TOWN"}, {"action": _QUESTION}
    ).choices["action"]

    assert answer.choice == "start"
    assert answer.confidence == pytest.approx(0.75)
    prompt = completions.last_kwargs["messages"][1]["content"]
    assert "A: Open the start menu\nB: Confirm\nC: Move up" in prompt


def test_raises_when_no_offered_label_is_among_the_top_tokens():
    jeff, _ = _jeff({"The": 0.5, "D": 0.5})

    with pytest.raises(JeffNoAnswer):
        jeff.system_one({}, {"action": _QUESTION})
