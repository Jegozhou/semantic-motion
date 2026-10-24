import pytest

from semantic_motion.anchors import AnchorError, Anchors, parse_anchor
from semantic_motion.asr import build_transcript


def _words(spec):
    """[(text, start, end), ...] -> raw whisper-like words"""
    return [{"text": t, "start": s, "end": e, "probability": 0.95} for t, s, e in spec]


@pytest.fixture()
def zh_tr():
    raw = _words([("大家好，", 0.0, 0.6), ("团队", 1.0, 1.3), ("用了", 1.3, 1.5), ("三个月，", 1.5, 2.0),
                  ("把", 2.2, 2.3), ("成本", 2.3, 2.6), ("降低了", 2.6, 2.9), ("40%。", 2.9, 3.6),
                  ("日活", 4.2, 4.5), ("从", 4.5, 4.6), ("2万", 4.6, 5.0), ("增长到", 5.0, 5.4), ("8万5千。", 5.4, 6.2)])
    return build_transcript(raw, "zh", None, refine=False)


def test_parse_anchor_syntaxes():
    assert parse_anchor("3:两千多张") == {"s": 3, "phrase": "两千多张"}
    assert parse_anchor("3:两千多张@end") == {"edge": "end", "s": 3, "phrase": "两千多张"}
    assert parse_anchor("3:门票+0.4")["offset"] == pytest.approx(0.4)
    assert parse_anchor("w0012") == {"w": "w0012"}
    assert parse_anchor("12.5s") == {"t": 12.5}
    assert parse_anchor(7.25) == {"t": 7.25}
    assert parse_anchor({"sentence": 2, "text": "x"}) == {"s": 2, "phrase": "x"}
    with pytest.raises(AnchorError):
        parse_anchor(None)


def test_phrase_resolves_to_word_onset(zh_tr):
    a = Anchors(zh_tr)
    r = a.resolve("成本")
    assert r["t"] == pytest.approx(2.3)
    assert r["score"] == 1.0
    assert a.resolve("成本@end")["t"] == pytest.approx(2.6)
    # character interpolation inside a word: '三个月' spoken 1.5-2.0, '月' is the 3rd of 3 chars
    t_yue = a.resolve("月")["t"]
    assert 1.5 < t_yue < 2.0


def test_spoken_numbers_match_asr_digits(zh_tr):
    a = Anchors(zh_tr)
    assert a.resolve("百分之四十")["t"] == pytest.approx(2.9)
    assert a.resolve("两万")["t"] == pytest.approx(4.6)
    assert a.resolve("八万五千")["t"] == pytest.approx(5.4)


def test_fuzzy_and_errors(zh_tr):
    a = Anchors(zh_tr)
    r = a.resolve("成本降底了")        # one wrong char -> fuzzy match with a warning
    assert r["score"] < 1.0 and "fuzzy" in r.get("warning", "")
    with pytest.raises(AnchorError):
        a.resolve("完全不存在的短语内容")
    with pytest.raises(AnchorError):
        a.resolve({"s": 99})


def test_corrections_map_brand_names():
    raw = _words([("我们", 0.0, 0.3), ("今天", 0.3, 0.6), ("聊", 0.6, 0.8), ("OpenCloud", 0.9, 1.6), ("。", 1.6, 1.6)])
    tr = build_transcript(raw, "zh", None, refine=False)
    a = Anchors(tr, corrections={"OpenCloud": "OpenClaw"})
    assert a.resolve("OpenClaw")["t"] == pytest.approx(0.9)
