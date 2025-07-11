import torch
from torchvision import transforms

from autolightning.auto_data_module import build_transform, compose_if_list


class SimpleTransform:
    """Simple transform that adds a constant value to tensors."""

    def __init__(self, add_value=1):
        self.add_value = add_value
        self.tot = transforms.ToTensor()

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.tot(x)
        return x + self.add_value


class SimpleTargetTransform:
    """Simple transform that adds a constant value to target."""

    def __init__(self, add_value=10):
        self.add_value = add_value

    def __call__(self, y):
        return y + self.add_value


def test_compose_if_list():
    # Test empty list
    assert compose_if_list([]) is None

    # Test single transform
    transform = SimpleTransform(1)
    assert compose_if_list([transform]) == transform

    # Test multiple transforms
    tf1 = SimpleTransform(1)
    tf2 = SimpleTransform(2)
    composed = compose_if_list([tf1, tf2])

    # Should be a Compose object
    assert isinstance(composed, transforms.Compose)
    assert len(composed.transforms) == 2
    assert composed.transforms[0] == tf1
    assert composed.transforms[1] == tf2

    # Test non-list
    assert compose_if_list(tf1) == tf1
    assert compose_if_list(None) is None


def test_build_transform_simple():
    # Test simple transform
    transform = SimpleTransform(1)
    assert build_transform("train", transform) == transform

    # Test None
    assert build_transform("train", None) is None


def test_build_transform_list():
    # Test list of transforms
    tf_list = [SimpleTransform(1), SimpleTransform(2)]
    built_tf = build_transform("train", tf_list)
    assert isinstance(built_tf, transforms.Compose)
    assert len(built_tf.transforms) == 2
    assert built_tf.transforms[0] == tf_list[0]
    assert built_tf.transforms[1] == tf_list[1]

    # Test the composed transform works
    x = torch.tensor([1, 2, 3])
    assert torch.all(built_tf(x) == torch.tensor([4, 5, 6]))  # 1+1+2


def test_build_transform_dict():
    # Test dictionary with pre, stage, post
    tf_dict = {
        "pre": SimpleTransform(1),
        "train": SimpleTransform(2),
        "post": SimpleTransform(3),
    }

    built_tf = build_transform("train", tf_dict)
    assert isinstance(built_tf, transforms.Compose)

    # Test the composed transform works
    x = torch.tensor([1, 2, 3])
    assert torch.all(built_tf(x) == torch.tensor([7, 8, 9]))  # +1+2+3

    # Test val stage
    built_val_tf = build_transform("val", tf_dict)
    assert isinstance(built_val_tf, transforms.Compose)
    result = built_val_tf(torch.tensor([1, 2, 3]))
    assert torch.all(result == torch.tensor([5, 6, 7]))  # +1+3 (no val stage)


def test_build_transform_complex_dict():
    # Test dictionary with lists
    tf_dict = {
        "pre": [SimpleTransform(1), SimpleTransform(1)],
        "train": SimpleTransform(3),
        "post": [SimpleTransform(2), SimpleTransform(2)],
    }

    built_tf = build_transform("train", tf_dict)
    assert isinstance(built_tf, transforms.Compose)

    # Test the composed transform works
    x = torch.tensor([1, 2, 3])
    assert torch.all(built_tf(x) == torch.tensor([10, 11, 12]))  # +1+1+3+2+2
