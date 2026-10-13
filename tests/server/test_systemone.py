"""TypeSafe-compatible POST /v1/systemone: mapping onto /decide, the harness's response contract, and limits."""
import math

import pytest
from app.systemone import choice_confidence, score_confidence

from tests.server.test_server import artifacts, client  # noqa: F401  (module fixtures reused)

STATE = "Policy: refunds need a receipt and purchase within 30 days. Bought 12 days ago, no receipt."
CHOICE = {"type": "choice", "instructions": "Which action is requested?",
          "criteria": {"refund": "Return money already charged", "cancel": "End a subscription",
                       "status": "Learn delivery progress"}}
NOUL = {"type": "noul", "instructions": "Is the refund permitted?",
        "criteria": {"false": "A condition is missing.", "true": "Every condition is met."}}
SCORE = {"type": "score", "instructions": "How severe is the problem?",
         "criteria": ["Cosmetic only", "One user impaired", "Many users blocked", "Data loss"]}


def post(client, questions, state=STATE):  # noqa: F811
    return client.post("/v1/systemone", json={"model": "mayaz", "state": state, "questions": questions})


def decide(client, qtype, question, options):  # noqa: F811
    return client.post("/decide", json={"type": qtype, "state": STATE, "question": question,
                                        "options": [{"key": k, "description": d} for k, d in options]}).json()


def test_choice_matches_decide_and_harness_contract(client):  # noqa: F811
    r = post(client, {"decision": CHOICE})
    assert r.status_code == 200
    body = r.json()
    a = body["answers"]["decision"]
    assert a["type"] == "choice" and a["choice"] in CHOICE["criteria"]
    assert set(a["probabilities"]) == set(CHOICE["criteria"]) and abs(sum(a["probabilities"].values()) - 1) < 1e-3
    ref = decide(client, "choice", CHOICE["instructions"], list(CHOICE["criteria"].items()))
    assert a["choice"] == ref["choice"]
    assert all(math.isclose(a["probabilities"][k], v, abs_tol=1e-6) for k, v in ref["probabilities"].items())
    assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] == 0
    assert body["model"] and {"server_ms", "rss_mb", "hwm_mb"} <= set(body["runtime"])


def test_noul_returns_p_yes_as_a_float(client):  # noqa: F811
    a = post(client, {"decision": NOUL}).json()["answers"]["decision"]
    ref = decide(client, "noul", NOUL["instructions"], [("true", NOUL["criteria"]["true"]),
                                                       ("false", NOUL["criteria"]["false"])])
    assert a["type"] == "noul" and isinstance(a["noul"], float) and 0 <= a["noul"] <= 1
    assert math.isclose(a["noul"], ref["probabilities"]["true"], abs_tol=1e-6)


def test_noul_without_criteria_uses_the_training_defaults(client):  # noqa: F811
    a = post(client, {"decision": {"type": "noul", "instructions": "The customer is angry."}}).json()
    assert 0 <= a["answers"]["decision"]["noul"] <= 1


def test_score_levels_are_string_indices_and_score_is_the_expectation(client):  # noqa: F811
    a = post(client, {"decision": SCORE}).json()["answers"]["decision"]
    assert a["type"] == "score" and set(a["probabilities"]) == {"0", "1", "2", "3"}
    exp = sum(int(k) * v for k, v in a["probabilities"].items())
    assert math.isclose(a["score"], exp, abs_tol=1e-9) and 0 <= a["confidence"] <= 1


def test_several_questions_share_one_state(client):  # noqa: F811
    one = {q: post(client, {q: body}).json()["answers"][q] for q, body in (("c", CHOICE), ("n", NOUL))}
    both = post(client, {"c": CHOICE, "n": NOUL}).json()["answers"]
    assert both["c"]["choice"] == one["c"]["choice"] and math.isclose(both["n"]["noul"], one["n"]["noul"])


def test_json_state_is_accepted(client):  # noqa: F811
    assert post(client, {"decision": NOUL}, state={"order": {"days": 12, "receipt": False}}).status_code == 200


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "x"},                                         # no criteria
    {"type": "choice", "instructions": "x", "criteria": {"a": "only one"}},           # 1 label
    {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},  # > 10 levels
    {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}},                # unknown noul key
    {"type": "predict", "instructions": "x"},                                         # unknown type
    {"type": "choice", "instructions": "x", "criteria": {"<OPT>": "a", "b": "b"}},    # reserved token
])
def test_malformed_questions_return_422(client, q):  # noqa: F811
    assert post(client, {"decision": q}).status_code == 422


def test_too_many_questions_return_422(client):  # noqa: F811
    assert post(client, {f"q{i}": NOUL for i in range(51)}).status_code == 422


def test_confidence_formulas_match_typesafe_docs():
    assert choice_confidence(0.6, 3) == pytest.approx(0.4)
    assert score_confidence([0.0, 0.57, 0.43]) == pytest.approx(1 - 0.43 / (2 / 3))


def test_runtime_reports_the_execution_provider_and_defaults_to_cpu(client):  # noqa: F811
    assert post(client, {"decision": NOUL}).json()["runtime"]["ep"] == "CPUExecutionProvider"
