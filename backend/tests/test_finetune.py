"""Phase 6 tests: dataset split balance and contrastive pair construction."""
from __future__ import annotations

from backend.app.finetune.dataset import LabeledItem, build_pairs, split


def _items():
    items = [LabeledItem(i, f"useful text {i}", "useful") for i in range(20)]
    items += [LabeledItem(100 + i, f"skipped text {i}", "skipped") for i in range(8)]
    return items


def test_split_keeps_both_classes_in_train_and_test():
    train, test = split(_items(), test_ratio=0.3, seed=1)
    assert {i.label for i in train} == {"useful", "skipped"}
    assert {i.label for i in test} == {"useful", "skipped"}
    # deterministic + disjoint
    train2, test2 = split(_items(), test_ratio=0.3, seed=1)
    assert [i.article_id for i in test] == [i.article_id for i in test2]
    ids_train = {i.article_id for i in train}
    ids_test = {i.article_id for i in test}
    assert ids_train.isdisjoint(ids_test)


def test_build_pairs_has_positive_and_negative_labels():
    train, _ = split(_items(), test_ratio=0.3, seed=1)
    pairs = build_pairs(train, max_pairs=200, seed=1)
    labels = set(pairs["label"])
    assert labels == {0.0, 1.0}
    assert len(pairs["sentence1"]) == len(pairs["sentence2"]) == len(pairs["label"])
    assert len(pairs["label"]) > 0
