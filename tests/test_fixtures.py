"""The test fixture is synthetic and hand-written; it must keep covering every type and option-count bucket."""
from mayaz.runtime import bucket
from tests.conftest import FIXTURES, load_jsonl


def test_fixture_items_are_synthetic_and_cover_types_and_buckets():
    items = load_jsonl(FIXTURES / "items.jsonl")
    assert len(items) >= 20 and len({it["id"] for it in items}) == len(items)
    assert {it["source"] for it in items} == {"synthetic"}
    assert {it["type"] for it in items} == {"choice", "noul", "score"}
    assert {bucket(len(it["options"])) for it in items} == {"2", "3-5", "6-20", "21+"}
    assert max(len(it["options"]) for it in items) == 64
    for it in items:
        keys = [o["key"] for o in it["options"]]
        assert len(set(keys)) == len(keys) and it["gold"] in keys
        assert it["meta"]["bucket"] == bucket(len(keys)) and it["meta"]["n_options"] == len(keys)
