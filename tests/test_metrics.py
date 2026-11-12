from src.evaluation.metrics import precision_recall_ndcg_at_k, intra_list_diversity


def test_precision_recall_ndcg_at_k():
    precision, recall, ndcg = precision_recall_ndcg_at_k([10, 20, 30], {10, 30}, 3)
    assert precision == 2 / 3
    assert recall == 1.0
    assert 0.0 < ndcg <= 1.0


def test_diversity_jaccard_distance():
    genres = {1: {"Action", "Drama"}, 2: {"Action"}, 3: {"Comedy"}}
    value = intra_list_diversity([1, 2, 3], genres)
    assert 0.0 < value <= 1.0


def test_single_item_diversity_is_zero():
    assert intra_list_diversity([1], {1: {"Drama"}}) == 0.0
