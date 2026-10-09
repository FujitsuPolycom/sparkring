"""Credential scanning retains strings while accepting numeric token scores."""
import json

from check_release_safety import findings

SCORE = -12.345678901234567

def panel(top):
    return dict(model='fixture', label='fixture', k=2, panel='fixture.json', n_records=1,
                wall_s=1, records=[dict(w=0, p=0, top=top, argmax='token')])


def test_numeric_token_score_is_not_a_credential():
    text = json.dumps(panel({"Bearer": SCORE, "bearer": SCORE - 5}))
    assert list(findings(text)) == []


def test_credential_strings_and_integers_remain_flagged():
    for value in ("-12.345678901234567", "fixtureCredentialValue", 1234567890123456):
        assert (1, "credential-assignment") in findings(json.dumps({"api_key": value}))


def test_numeric_prefix_does_not_hide_a_credential():
    value = "-12.345678901234suffix"
    assert (1, "credential-assignment") in findings('bearer=' + value)
    assert (1, "credential-assignment") in findings('{"Bearer": ' + value + '}')


def test_other_findings_on_same_line_are_retained():
    value = "fixtureCredentialValue"
    fixture = panel({"Bearer": SCORE})
    fixture['secret'] = value
    text = json.dumps(fixture)
    assert (1, "credential-assignment") in findings(text)


def test_non_panel_numbers_and_malformed_assignments_remain_flagged():
    value = '-12.345678901234'
    for text in ('password: ' + value + ',', json.dumps({'Bearer': float(value)}),
                 json.dumps({'prose': 'password: ' + value + ','})):
        assert (1, 'credential-assignment') in findings(text)


def test_panel_string_scores_and_duplicate_keys_remain_flagged():
    value = 'fixtureCredentialValue'
    assert (1, 'credential-assignment') in findings(json.dumps(panel({'Bearer': value})))
    raw = json.dumps(panel({'Bearer': SCORE}))
    duplicate = raw[:-1] + ', "secret": ' + json.dumps(value) + ', "secret": 0}'
    assert (1, 'credential-assignment') in findings(duplicate)


def test_panel_normalization_preserves_other_numeric_assignments():
    raw = json.dumps(panel({'Bearer': SCORE}))
    value = '-0.000000000001'
    outside = raw[:-1] + ', "password": ' + value + '}'
    inside = raw.replace('"top": {', '"top": {"secret": ' + value + ', ')
    for text in (outside, inside):
        assert (1, 'credential-assignment') in findings(text)


def test_local_windows_user_paths_are_findings():
    # Assembled from pieces so that this file holds no such path.
    drive, users = "C:", "Users"
    for text in (drive + "\\" + users + "\\someone\\work", json.dumps({"lead": drive + "\\" + users + "\\someone"}),
                 drive + "/" + users + "/someone", "/mnt/c/" + users + "/someone", "Local " + "AppData" + "\\Temp"):
        assert (1, "local-user-path") in findings(text), text
    for text in ("C: drive", "the Users guide", "AppData is a directory name", "/mnt/data/users"):
        assert (1, "local-user-path") not in findings(text), text
