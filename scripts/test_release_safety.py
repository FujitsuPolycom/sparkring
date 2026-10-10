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


def test_site_values_from_an_untracked_file_are_findings(tmp_path):
    from check_release_safety import site_patterns
    assert site_patterns(tmp_path / "absent.txt") == []
    values = tmp_path / "site-values.txt"
    # Synthetic values in the shapes of a site's: a host name and a locally administered MAC address.
    values.write_text("# site\nspark-zz99\n02:00:00:12:34:56  # rank 0, port 0\n00:00:00:00:00:00\n", encoding="utf-8")
    site = site_patterns(values)
    assert len(site) == 2
    for text in ("ssh spark-zz99", "SPARK-ZZ99:", "mac 02:00:00:12:34:56", "02-00-00-12-34-56",
                 "fe80::ff:fe12:3456%enp1s0f0np0"):
        assert (1, "site-value") in findings(text, site), text
    for text in ("spark-zz990", "my-spark-zz99-copy", "02:00:00:12:34:57", "fe80::ff:fe12:3457"):
        assert (1, "site-value") not in findings(text, site), text
    assert list(findings("ssh spark-zz99")) == []


def test_the_site_value_allowlist_names_only_frozen_inputs():
    from check_release_safety import SITE_VALUE_ALLOWED
    from pathlib import Path
    frozen = (Path(__file__).resolve().parents[1] / "runtime/releases/preserved-inputs.json").read_text(encoding="utf-8")
    for path, lines in SITE_VALUE_ALLOWED.items():
        assert f'"{path}"' in frozen and all(isinstance(line, int) and reason for line, reason in lines.items())
