from benchmarks.offline import LeadSummarizer, lead_summary


def test_lead_summary_keeps_first_n_sentences():
    article = "First fact. Second fact! Third fact? Fourth fact."
    assert lead_summary(article, 3) == "First fact. Second fact! Third fact?"
    assert lead_summary(article, 1) == "First fact."


def test_short_article_is_returned_whole():
    assert lead_summary("Only one sentence.", 3) == "Only one sentence."


def test_adapter_is_deterministic_and_free():
    adapter = LeadSummarizer()
    a, b = adapter.run("One. Two. Three. Four."), adapter.run("One. Two. Three. Four.")
    assert a.answer == b.answer == "One. Two. Three."
    assert a.cost_usd == 0.0 and a.context is None
