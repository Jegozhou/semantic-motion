from semantic_motion.text import cn_to_number, find_numbers, norm, numeric_variants, en_words_to_number, visual_len


def test_cn_numbers():
    assert cn_to_number("两千") == 2000
    assert cn_to_number("八万五千") == 85000
    assert cn_to_number("一点五") == 1.5
    assert cn_to_number("十二") == 12
    assert cn_to_number("3万") == 30000
    assert cn_to_number("abc") is None


def test_find_numbers_zh():
    ns = find_numbers("成本降低了百分之四十，日活从2万增长到8万5千")
    vals = [(n["value"], n["suffix"]) for n in ns]
    assert (40.0, "%") in vals
    assert (20000.0, "") in vals
    # filler words are not numbers
    assert find_numbers("我们一起看一下") == []


def test_en_words_to_number():
    assert en_words_to_number("twenty thousand") == 20000
    assert en_words_to_number("fifty-five million") == 55_000_000
    assert en_words_to_number("one hundred and twenty") == 120
    assert en_words_to_number("thousand") is None


def test_numeric_variants():
    assert "40" in numeric_variants("百分之四十")
    assert "2万" in numeric_variants("两万")
    assert "8万5千" in numeric_variants("八万五千")
    assert "40" in numeric_variants("forty percent")
    assert "20000" in numeric_variants("twenty thousand")
    assert numeric_variants("第一") == []          # ordinal-ish single digit is left alone
    assert numeric_variants("the numbers") == []


def test_norm_and_width():
    assert norm("Hello, 世界！") == "hello世界"
    assert norm("龍蝦") == norm("龙虾")             # traditional -> simplified
    assert visual_len("中文") == 2.0
    assert visual_len("ab") < visual_len("中文")
