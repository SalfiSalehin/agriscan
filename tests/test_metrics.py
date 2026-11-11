from ml.metrics import accuracy, confusion_matrix, macro_f1, per_class_prf


def test_confusion_matrix_and_scores():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    precision, recall, f1, support = per_class_prf(cm)
    assert support.tolist() == [2, 2, 1]
    assert abs(precision[1] - 2 / 3) < 1e-9
    assert abs(recall[1] - 1.0) < 1e-9
    assert recall[2] == 0.0
    assert abs(accuracy(cm) - 0.6) < 1e-9
    assert abs(macro_f1(cm) - (0.5 + 0.8 + 0.0) / 3) < 1e-9


def test_macro_f1_ignores_classes_without_samples():
    cm = confusion_matrix([0, 0], [0, 0], 3)
    assert macro_f1(cm) == 1.0
