"""Credential scanning retains strings while accepting numeric token scores."""
import json

from check_release_safety import findings


def test_numeric_token_score_is_not_a_credential():
    text = json.dumps({"top": {"Bearer": -12.345678901234567, "bearer": -17.123456789012345}})
    assert list(findings(text)) == []


def test_credential_strings_and_integers_remain_flagged():
    for value in ("-12.345678901234567", "fixtureCredentialValue", 1234567890123456):
        assert (1, "credential-assignment") in findings(json.dumps({"api_key": value}))


def test_numeric_prefix_does_not_hide_a_credential():
    assert (1, "credential-assignment") in findings('bearer=-12.345678901234suffix')
    assert (1, "credential-assignment") in findings('{"Bearer": -12.345678901234suffix}')


def test_other_findings_on_same_line_are_retained():
    text = json.dumps({"top": {"Bearer": -12.345678901234567}, "secret": "fixtureCredentialValue"})
    assert (1, "credential-assignment") in findings(text)
