import pytest
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

from autolightning import AutoDataModule
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
    """Tests that compose_if_list correctly handles empty lists, single transforms, and multiple transforms."""
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
    """Tests that build_transform correctly handles simple transforms and None values."""
    # Test simple transform
    transform = SimpleTransform(1)
    assert build_transform("train", transform) == transform
    
    # Test None
    assert build_transform("train", None) is None


def test_build_transform_list():
    """Tests that build_transform correctly composes lists of transforms."""
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
    """Tests that build_transform correctly processes dictionaries with pre, stage, and post transforms."""
    # Test dictionary with pre, stage, post
    tf_dict = {
        "pre": SimpleTransform(1),
        "train": SimpleTransform(2),
        "post": SimpleTransform(3)
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
    """Tests that build_transform correctly handles dictionaries with lists of transforms in each stage."""
    # Test dictionary with lists
    tf_dict = {
        "pre": [SimpleTransform(1), SimpleTransform(1)],
        "train": SimpleTransform(3),
        "post": [SimpleTransform(2), SimpleTransform(2)]
    }
    
    built_tf = build_transform("train", tf_dict)
    assert isinstance(built_tf, transforms.Compose)
    
    # Test the composed transform works
    x = torch.tensor([1, 2, 3])
    assert torch.all(built_tf(x) == torch.tensor([10, 11, 12]))  # +1+1+3+2+2


def test_datamodule_simple_transforms():
    """Tests that AutoDataModule correctly applies simple transforms to both data and targets."""
    transform = SimpleTransform(5)
    target_transform = SimpleTargetTransform(10)
    
    data = AutoDataModule(
        dataset=CIFAR10("data", train=True, download=True),
        transforms=transform,
        target_transforms=target_transform
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Get transformed dataset
    train_ds = data.get_transformed_dataset("train")
    
    # Original dataset item
    original_ds = CIFAR10("data", train=True, download=True)

    assert len(train_ds) == len(original_ds)
    
    # Transformed dataset item
    for i in range(15):
        img, label = original_ds[i]
        transformed_img, transformed_label = train_ds[i]

        tot = transforms.ToTensor()
        
        # Image should be original + 5, label should be original + 10
        transformed_img_expected = tot(img) + 5
        transformed_label_expected = label + 10
        
        assert torch.all(transformed_img == transformed_img_expected)
        assert transformed_label == transformed_label_expected


def test_datamodule_dict_transforms():
    """Tests that AutoDataModule correctly applies dictionary transforms with phase-specific configurations."""
    transforms_dict = {
        "pre": SimpleTransform(1),
        "train": SimpleTransform(2),
        "val": SimpleTransform(3),
        "post": SimpleTransform(1)
    }
    
    target_transforms_dict = {
        "train": SimpleTargetTransform(10),
        "val": SimpleTargetTransform(20)
    }
    
    data = AutoDataModule(
        dataset={
            "train": CIFAR10("data", train=True, download=True),
            "val": CIFAR10("data", train=False, download=True)
        },
        transforms=transforms_dict,
        target_transforms=target_transforms_dict
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Get transformed datasets
    train_ds = data.get_transformed_dataset("train")
    val_ds = data.get_transformed_dataset("val")
    
    # Original dataset items
    train_original = CIFAR10("data", train=True, download=True)
    val_original = CIFAR10("data", train=False, download=True)
    
    train_img, train_label = train_original[0]
    val_img, val_label = val_original[0]
    
    # Transformed dataset items
    transformed_train_img, transformed_train_label = train_ds[0]
    transformed_val_img, transformed_val_label = val_ds[0]

    tot = transforms.ToTensor()
    
    # Train: pre(1) + train(2) + post(1) = +4 for images, +10 for labels
    # Val: pre(1) + val(3) + post(1) = +5 for images, +20 for labels
    assert torch.allclose(transformed_train_img, tot(train_img) + 4)
    assert transformed_train_label == train_label + 10

    assert torch.allclose(transformed_val_img, tot(val_img) + 5)
    assert transformed_val_label == val_label + 20


def test_datamodule_pre_loaded_transforms():
    """Tests that AutoDataModule correctly applies pre_load transforms before creating in-memory datasets."""
    transforms_dict = {
        "pre_load": SimpleTransform(1),
        "train": SimpleTransform(2)
    }
    
    data = AutoDataModule(
        dataset=CIFAR10("data", train=True, download=True),
        transforms=transforms_dict,
        pre_load=True
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Get transformed dataset
    train_ds = data.get_transformed_dataset("train")
    
    # Original dataset
    original_ds = CIFAR10("data", train=True, download=True)
    img, label = original_ds[0]
    
    # Transformed dataset
    transformed_img, transformed_label = train_ds[0]
    
    # Should have pre_load(+1) and then train(+2) applied
    assert torch.allclose(transformed_img, transforms.ToTensor()(img) + 3)
    assert transformed_label == label  # No target transform


def test_pre_load_target_transforms():
    """Tests that pre_load target transforms are applied correctly before loading datasets into memory."""
    transforms_dict = {
        "pre_load": SimpleTransform(1),
        "train": SimpleTransform(2)
    }
    
    target_transforms_dict = {
        "pre_load": SimpleTargetTransform(5),
        "train": SimpleTargetTransform(10)
    }
    
    data = AutoDataModule(
        dataset=CIFAR10("data", train=True, download=True),
        transforms=transforms_dict,
        target_transforms=target_transforms_dict,
        pre_load=True
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Get transformed dataset
    train_ds = data.get_transformed_dataset("train")
    
    # Original dataset
    original_ds = CIFAR10("data", train=True, download=True)
    img, label = original_ds[0]
    
    # Transformed dataset
    transformed_img, transformed_label = train_ds[0]
    
    # Should have pre_load(+1) and then train(+2) applied to image
    # Should have pre_load(+5) and then train(+10) applied to label
    assert torch.allclose(transformed_img, transforms.ToTensor()(img) + 3)
    assert transformed_label == label + 15  # 5 + 10


def test_pre_and_post_transforms_only():
    """Tests that transforms function correctly when only pre and post transforms are specified."""
    transforms_dict = {
        "pre": SimpleTransform(2),
        "post": SimpleTransform(3)
    }
    
    target_transforms_dict = {
        "pre": SimpleTargetTransform(5),
        "post": SimpleTargetTransform(7)
    }
    
    data = AutoDataModule(
        dataset=CIFAR10("data", train=True, download=True),
        transforms=transforms_dict,
        target_transforms=target_transforms_dict
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Get transformed dataset
    train_ds = data.get_transformed_dataset("train")
    
    # Original dataset
    original_ds = CIFAR10("data", train=True, download=True)
    img, label = original_ds[0]
    
    # Transformed dataset
    transformed_img, transformed_label = train_ds[0]
    
    # Should have pre(+2) and post(+3) applied to image = +5
    # Should have pre(+5) and post(+7) applied to label = +12
    assert torch.allclose(transformed_img, transforms.ToTensor()(img) + 5)
    assert transformed_label == label + 12


def test_multiple_pre_post_transforms():
    """Tests the correct application of multiple pre and post transforms in sequence."""
    transforms_dict = {
        "pre": [SimpleTransform(1), SimpleTransform(1)],
        "post": [SimpleTransform(2), SimpleTransform(3)]
    }
    
    data = AutoDataModule(
        dataset=CIFAR10("data", train=True, download=True),
        transforms=transforms_dict
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Get transformed dataset
    train_ds = data.get_transformed_dataset("train")
    
    # Original dataset
    original_ds = CIFAR10("data", train=True, download=True)
    img, label = original_ds[0]
    
    # Transformed dataset
    transformed_img, transformed_label = train_ds[0]
    
    # Should have pre(+1+1) and post(+2+3) applied to image = +7
    assert torch.allclose(transformed_img, transforms.ToTensor()(img) + 7)
    assert transformed_label == label  # No target transform


def test_all_transform_stages():
    """Tests the correct application of all transform stages: pre_load, pre, phase-specific, and post."""
    transforms_dict = {
        "pre_load": SimpleTransform(1),
        "pre": SimpleTransform(2),
        "train": SimpleTransform(3),
        "val": SimpleTransform(4),
        "post": SimpleTransform(5)
    }
    
    target_transforms_dict = {
        "pre_load": SimpleTargetTransform(10),
        "pre": SimpleTargetTransform(20),
        "train": SimpleTargetTransform(30),
        "val": SimpleTargetTransform(40),
        "post": SimpleTargetTransform(50)
    }
    
    data = AutoDataModule(
        dataset={
            "train": CIFAR10("data", train=True, download=True),
            "val": CIFAR10("data", train=False, download=True)
        },
        transforms=transforms_dict,
        target_transforms=target_transforms_dict,
        pre_load=True
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Get transformed datasets
    train_ds = data.get_transformed_dataset("train")
    val_ds = data.get_transformed_dataset("val")
    
    # Original datasets
    train_original = CIFAR10("data", train=True, download=True)
    val_original = CIFAR10("data", train=False, download=True)
    
    # Test train dataset
    img, label = train_original[0]
    transformed_img, transformed_label = train_ds[0]
    
    # Image: pre_load(+1) + pre(+2) + train(+3) + post(+5) = +11
    # Label: pre_load(+10) + pre(+20) + train(+30) + post(+50) = +110
    assert torch.allclose(transformed_img, transforms.ToTensor()(img) + 11)
    assert transformed_label == label + 110
    
    # Test val dataset
    img, label = val_original[0]
    transformed_img, transformed_label = val_ds[0]
    
    # Image: pre_load(+1) + pre(+2) + val(+4) + post(+5) = +12
    # Label: pre_load(+10) + pre(+20) + val(+40) + post(+50) = +120
    assert torch.allclose(transformed_img, transforms.ToTensor()(img) + 12)
    assert transformed_label == label + 120


def test_batch_transforms():
    """Tests that batch transforms are correctly applied during batch transfer."""
    batch_transforms = {
        "before": SimpleTransform(5),
        "after": SimpleTransform(10)
    }
    
    data = AutoDataModule(
        dataset=None,
        batch_transforms=batch_transforms,
        target_batch_transforms=None
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Create a dummy batch (X, y)
    batch = (torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    
    # Test on_before_batch_transfer
    before_batch = data.on_before_batch_transfer(batch, 0)
    assert torch.allclose(before_batch[0], torch.tensor([6, 7, 8]))  # X + 5
    assert torch.allclose(before_batch[1], torch.tensor([0, 1, 2]))  # y unchanged
    
    # Test on_after_batch_transfer
    after_batch = data.on_after_batch_transfer(batch, 0)
    assert torch.allclose(after_batch[0], torch.tensor([11, 12, 13]))  # X + 10
    assert torch.allclose(after_batch[1], torch.tensor([0, 1, 2]))  # y unchanged


def test_batch_and_target_batch_transforms():
    """Tests that both batch and target batch transforms are correctly applied during batch transfer."""
    batch_transforms = {
        "before": SimpleTransform(5),
        "after": SimpleTransform(10)
    }
    
    target_batch_transforms = {
        "before": SimpleTargetTransform(1),
        "after": SimpleTargetTransform(2)
    }
    
    data = AutoDataModule(
        dataset=None,
        batch_transforms=batch_transforms,
        target_batch_transforms=target_batch_transforms
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Create a dummy batch (X, y)
    batch = (torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    
    # Test on_before_batch_transfer
    before_batch = data.on_before_batch_transfer(batch, 0)
    assert torch.allclose(before_batch[0], torch.tensor([6, 7, 8]))  # X + 5
    assert torch.allclose(before_batch[1], torch.tensor([1, 2, 3]))  # y + 1
    
    # Test on_after_batch_transfer
    after_batch = data.on_after_batch_transfer(batch, 0)
    assert torch.allclose(after_batch[0], torch.tensor([11, 12, 13]))  # X + 10
    assert torch.allclose(after_batch[1], torch.tensor([2, 3, 4]))  # y + 2


def test_combined_target_batch_transforms():
    """Tests that combined mode applies the same batch transforms to both inputs and targets."""
    batch_transforms = {
        "before": SimpleTransform(5),
    }
    
    data = AutoDataModule(
        dataset=None,
        batch_transforms=batch_transforms,
        target_batch_transforms="combine"
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Create a dummy batch (a tensor, not a tuple)
    batch = torch.tensor([1, 2, 3])
    
    # With combine mode, the entire batch should be transformed
    before_batch = data.on_before_batch_transfer(batch, 0)
    
    # Since the transform adds 5 to everything
    assert torch.allclose(before_batch, torch.tensor([6, 7, 8]))


def test_error_pre_load_without_enabling():
    """Tests that an error is raised when pre_load transform is specified but pre_load is disabled."""
    transforms_dict = {
        "pre_load": SimpleTransform(1),
        "train": SimpleTransform(2)
    }
    
    data = AutoDataModule(
        dataset=CIFAR10("data", train=True, download=True),
        transforms=transforms_dict,
        pre_load=False  # Not enabled
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Should raise an error when trying to get the transformed dataset
    with pytest.raises(ValueError, match="Pre-load transform specified.*but pre-load is not enabled"):
        data.get_transformed_dataset("train")


def test_phase_specific_pre_load():
    """Tests that phase-specific pre_load settings are correctly applied."""
    transforms_dict = {
        "pre_load": SimpleTransform(1),
        "train": SimpleTransform(2),
        "val": SimpleTransform(3)
    }
    
    # Enable pre_load only for train phase
    data = AutoDataModule(
        dataset={
            "train": CIFAR10("data", train=True, download=True),
            "val": CIFAR10("data", train=False, download=True)
        },
        transforms=transforms_dict,
        pre_load={"train": True, "val": False}
    )
    
    data.prepare_data()
    data.setup('fit')
    
    # Get train dataset - should work fine with pre_load
    _ = data.get_transformed_dataset("train")
    
    # Get val dataset - should raise error
    with pytest.raises(ValueError, match="Pre-load transform specified.*but pre-load is not enabled"):
        _ = data.get_transformed_dataset("val")
