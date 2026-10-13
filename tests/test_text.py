from mayaz.text import humanize_label, option_text, state_text


def test_humanize_label():
    assert humanize_label("card_arrival") == "card arrival"
    assert humanize_label("Top_Up-by_card") == "top up by card"
    assert humanize_label("EducationalInstitution") == "educational institution"


def test_state_text_json_is_compact_and_unicode():
    assert state_text({"a": "é", "b": [1, 2]}) == '{"a": "é", "b": [1, 2]}'
    assert state_text("  plain text  ") == "plain text"


def test_option_text_choice_uses_humanized_key_and_desc():
    assert option_text("choice", "human_review", "Queue this trace.") == "human review: Queue this trace."
    assert option_text("choice", "card_arrival", None) == "card arrival"


def test_option_text_score_and_noul_never_show_key_words():
    assert option_text("score", "2", "Elevated; within the week.") == "Elevated; within the week."
    t = option_text("noul", "false", "No human attention is warranted.")
    assert t == "No human attention is warranted." and "false" not in t.lower()


def test_option_text_noul_without_desc_uses_neutral_defaults():
    assert option_text("noul", "true", None) == "Yes, the statement holds."
    assert option_text("noul", "false", None) == "No, the statement does not hold."
