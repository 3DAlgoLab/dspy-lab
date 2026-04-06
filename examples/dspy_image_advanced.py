"""
Advanced Examples: DSPy Image Optimization and Advanced Features

This module demonstrates advanced usage of dspy.Image including
optimization, custom adapters, and error handling.
"""

import tempfile
from typing import Any

from dspy.utils.dummies import DummyLM

import dspy
from examples.dspy_image_signatures import ImageCaptionSignature


def example1_few_shot_learning():
    """
    Example 1: Few-shot learning with images using LabeledFewShot.

    Demonstrates how to compile a predictor with few-shot examples
    containing images.
    """

    # Define training examples with images
    examples = [
        dspy.Example(
            image=dspy.Image(
                "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
            ),
            caption="A Great Dane standing in a field",
        ).with_inputs("image"),
        dspy.Example(
            image=dspy.Image("https://images.dog.ceo/breeds/pug/n02085620_1588.jpg"),
            caption="A P Pug sitting down",
        ).with_inputs("image"),
        dspy.Example(
            image=dspy.Image(
                "https://images.dog.ceo/breeds/beagle/n02088364_10564.jpg"
            ),
            caption="A Beagle lying on the ground",
        ).with_inputs("image"),
    ]

    # Set up the predictor with few-shot learning
    lm = DummyLM([{"caption": "A large dog breed standing"}] * 20)
    dspy.configure(lm=lm)

    student_predictor = dspy.Predict(ImageCaptionSignature)

    optimizer = dspy.teleprompt.LabeledFewShot(k=2)
    compiled_predictor = optimizer.compile(student=student_predictor, trainset=examples)

    # Test the compiled predictor
    test_image = dspy.Image(
        "https://images.dog.ceo/breeds/chihuahua/n02085620_1588.jpg"
    )
    result = compiled_predictor(image=test_image)

    print("=== Few-Shot Learning with Images ===")
    print(f"Number of training examples: {len(examples)}")
    print(f"Compiled predictor output: {result.caption}")
    print(f"Predictor was optimized with LabeledFewShot (k=2)")
    print()


def example2_optimization_with_validation():
    """
    Example 2: Optimization with custom metric validation.

    Demonstrate using a custom validation metric for image tasks.
    """

    def custom_image_metric(example, prediction, trace=None):
        """
        Custom metric that validates caption quality.

        Args:
            example: Original example
            prediction: Model prediction
            trace: Optional trace information

        Returns:
            bool: True if prediction meets criteria
        """
        # Simple validation: caption should be non-empty and have at least one word
        return bool(prediction.caption) and len(prediction.caption.split()) > 1

    examples = [
        dspy.Example(
            image=dspy.Image(
                "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
            ),
            caption="A Great Dane dog",
        ).with_inputs("image"),
        dspy.Example(
            image=dspy.Image("https://images.dog.ceo/breeds/pug/n02085620_1588.jpg"),
            caption="A P Pug puppy",
        ).with_inputs("image"),
    ]

    lm = DummyLM([{"caption": "A dog in the image"}] * 20)
    dspy.configure(lm=lm)

    predictor = dspy.Predict(ImageCaptionSignature)

    optimizer = dspy.teleprompt.LabeledFewShot(k=1, metric=custom_image_metric)
    compiled_predictor = optimizer.compile(
        student=predictor, trainset=examples, sample=False
    )

    result = compiled_predictor(
        image=dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")
    )

    print("=== Optimization with Custom Metric ===")
    print(f"Metric validation: caption has {len(result.caption.split())} words")
    print(f"Output: {result.caption}")
    print()


def example3_error_handling():
    """
    Example 3: Error handling for invalid image inputs.

    Demonstrate proper error handling for invalid image sources.
    """

    print("=== Error Handling Demo ===")

    # Valid cases
    try:
        valid_image = dspy.Image(
            "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
        )
        print(f"✓ Valid URL: {repr(valid_image)}")
    except Exception as e:
        print(f"✗ URL failed: {e}")

    # Invalid string format
    try:
        invalid_image = dspy.Image("this_is_not_a_valid_image_source_12345")
        print(f"✗ Should have raised error for invalid format")
    except ValueError as e:
        print(f"✓ Invalid format caught: {type(e).__name__}")

    # Non-existent file
    try:
        nonexistent_image = dspy.Image("/nonexistent/file.png")
        print(f"✗ Should have raised error for nonexistent file")
    except Exception as e:
        print(f"✓ Nonexistent file caught: {type(e).__name__}")

    print()


def example4_image_download_handling():
    """
    Example 4: Handling image downloads with various MIME types.

    Demonstrate downloading images and supporting different file types.
    """

    print("=== Image Download Handling ===")

    # Download image and check the encoded format
    image_url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    image_downloaded = dspy.Image(image_url, download=True)

    print(f"Original URL: {image_url}")
    print(
        f"Downloaded image starts with 'data:': {image_downloaded.url.startswith('data:')}"
    )
    print(f"Contains mime type 'image/jpeg': {'image/jpeg' in image_downloaded.url}")
    print(
        f"Base64 length: {len(image_downloaded.url.split('base64,')[1][:50])} chars (truncated)"
    )

    # Test with data URI (should use as-is)
    image_from_uri = dspy.Image(data_uri_example())
    print(f"\nData URI preserved: {image_from_uri.url.startswith('data:image/png')}")

    print()


def data_uri_example():
    """Helper to create a sample data URI."""
    import base64
    from io import BytesIO
    from PIL import Image

    pil_image = Image.new("RGB", (50, 50), color="red")
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def example5_save_load_compiled_predictor():
    """
    Example 5: Saving and loading compiled predictors.

    Demonstrate persisting compiled predictors with image signatures.
    """

    examples = [
        dspy.Example(
            image=dspy.Image(
                "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
            ),
            caption="Training example 1",
        ).with_inputs("image"),
    ]

    lm = DummyLM([{"caption": "Saved prediction"}] * 20)
    dspy.configure(lm=lm)

    # Compile predictor
    student = dspy.Predict(ImageCaptionSignature)
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled = optimizer.compile(student=student, trainset=examples, sample=False)

    # Save to file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
        temp_path = f.name

    compiled.save(temp_path)
    print(f"=== Save/Load Compiled Predictor ===")
    print(f"Compiled predictor saved to: {temp_path}")

    # Load predictor
    loaded_predictor = dspy.Predict(ImageCaptionSignature)
    loaded_predictor.load(temp_path)

    # Test with loaded predictor
    test_image = dspy.Image(
        "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    )
    result = loaded_predictor(image=test_image)

    print(f"Loaded predictor output: {result.caption}")
    print(f"Both save and load successful!")

    # Clean up
    import os

    os.remove(temp_path)
    print()


def example6_custom_adapter_usage():
    """
    Example 6: Using different adapters with image tasks.

    Demonstrate how different adapters handle image inputs.
    """

    # Available adapters
    from dspy.adapters import ChatAdapter, JSONAdapter, XMLAdapter

    print("=== Adapter Types for Image Tasks ===")

    adapters_to_try = [
        ("ChatAdapter", ChatAdapter),
        ("JSONAdapter", JSONAdapter),
        ("XMLAdapter", XMLAdapter),
    ]

    for name, adapter_cls in adapters_to_try:
        print(f"{name}:")

        # Create an instance
        adapter = adapter_cls()

        # Define a simple signature
        class TestSignature(dspy.Signature):
            image: dspy.Image = dspy.InputField()
            result: str = dspy.OutputField()

        # See if adapter can handle it
        try:
            # This would format the message for the adapter
            test_image = dspy.Image("https://example.com/test.jpg")
            example = dspy.Example(image=test_image, result="test").with_inputs(
                "image", "result"
            )

            # Format for adapter
            formatted = adapter.format(example, TestSignature)
            print(f"  ✓ Handles images: {type(formatted).__name__}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()


def example7_batch_processing():
    """
    Example 7: Batch processing multiple images.

    Demonstrate efficient processing of multiple images
    with the same signature.
    """

    lm = DummyLM([{"caption": f"Caption {i}"} for i in range(10)])
    dspy.configure(lm=lm)

    predictor = dspy.Predict(ImageCaptionSignature)

    # Batch of image URLs
    image_urls = [
        "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg",
        "https://images.dog.ceo/breeds/pug/n02085620_1588.jpg",
        "https://images.dog.ceo/breeds/beagle/n02088364_10564.jpg",
        "https://images.dog.ceo/breeds/chihuahua/n02085620_1588.jpg",
    ]

    print("=== Batch Processing ===")
    print(f"Processing {len(image_urls)} images in batch...\n")

    results = []
    for i, url in enumerate(image_urls, 1):
        image = dspy.Image(url)
        result = predictor(image=image)
        results.append(result)
        print(f"Image {i}: {result.caption}")

    print(f"\nBatch completed: {len(results)} predictions made")
    print()


def example8_nested_structures():
    """
    Example 8: Images in nested structures (Pydantic models, lists, dicts).

    Demonstrate using images in complex nested data structures.
    """

    import pydantic

    print("=== Nested Image Structures ===")

    class NestedImageData(pydantic.BaseModel):
        """Model with multiple image fields."""

        primary_image: dspy.Image
        secondary_images: list[dspy.Image] | None = None
        metadata: dict[str, str] = {}

        class Config:
            protected_namespaces = ()

    # Create instance with nested images
    image1 = dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")
    image2 = dspy.Image("https://images.dog.ceo/breeds/pug/n02085620_1588.jpg")
    image3 = dspy.Image("https://images.dog.ceo/breeds/beagle/n02088364_10564.jpg")

    nested_data = NestedImageData(
        primary_image=image1,
        secondary_images=[image2, image3],
        metadata={"description": "Test nested data"},
    )

    print(f"Created nested data model with:")
    print(f"  - 1 primary image")
    print(f"  - {len(nested_data.secondary_images)} secondary images")
    print(f"  - {len(nested_data.metadata)} metadata fields")

    # Use in signature
    class NestedSignature(dspy.Signature):
        data: NestedImageData = dspy.InputField()
        summary: str = dspy.OutputField()

    lm = DummyLM([{"summary": "Summary of nested images"}])
    dspy.configure(lm=lm)

    predictor = dspy.Predict(NestedSignature)
    result = predictor(data=nested_data)

    print(f"\nPrediction result: {result.summary}")
    print()


def example9_compare_with_and_without_download():
    """
    Example 9: Compare behavior with and without download.

    Show the difference between URL-only and downloaded images.
    """

    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"

    # Without download (URL only)
    image_no_download = dspy.Image(url)

    # With download (base64 encoded)
    image_with_download = dspy.Image(url, download=True)

    print("=== URL vs Download Comparison ===")
    print(f"Original URL: {url}")
    print()
    print("Without download:")
    print(f"  URL type: {type(image_no_download.url)}")
    print(f"  Is data URI: {image_no_download.url.startswith('data:')}")
    print(f"  Contains original URL: {'images.dog.ceo' in image_no_download.url}")
    print()
    print("With download:")
    print(f"  URL type: {type(image_with_download.url)}")
    print(f"  Is data URI: {image_with_download.url.startswith('data:')}")
    print(f"  Has mime type: {'image/jpeg' in image_with_download.url}")
    print(f"  Base64 length: ~{len(image_with_download.url) - 30} chars")
    print()


def example9_run_all():
    """Helper to run all examples."""

    print("=" * 60)
    print("DSPy Image - Advanced Feature Examples")
    print("=" * 60)
    print()

    print("Example 1: Few-Shot Learning")
    print("-" * 40)
    example1_few_shot_learning()

    print("Example 2: Optimization with Validation")
    print("-" * 40)
    example2_optimization_with_validation()

    print("Example 3: Error Handling")
    print("-" * 40)
    example3_error_handling()

    print("Example 4: Image Download Handling")
    print("-" * 40)
    example4_image_download_handling()

    print("Example 5: Save/Load Compiled Predictor")
    print("-" * 40)
    # Save/load example would run, output shown
    print("See example5_save_load_compiled_predictor() for full demo")
    print()

    print("Example 6: Custom Adapters")
    print("-" * 40)
    example6_custom_adapter_usage()

    print("Example 7: Batch Processing")
    print("-" * 40)
    example7_batch_processing()

    print("Example 8: Nested Structures")
    print("-" * 40)
    example8_nested_structures()

    print("Example 9: Compare Download Options")
    print("-" * 40)
    example9_compare_with_and_without_download()


if __name__ == "__main__":
    example9_run_all()

    print("=" * 60)
    print("All advanced examples completed!")
    print("=" * 60)
