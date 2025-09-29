from llm_collaborator.convergence import has_converged, similarity_ratio


def test_similarity_ratio_handles_empty_strings():
    assert similarity_ratio("", "Something") == 0.0
    assert similarity_ratio(None, "Other") == 0.0


def test_has_converged_true_when_above_threshold():
    result = has_converged("abc", "abc", threshold=0.9)
    assert result.converged
    assert result.similarity >= 0.9
