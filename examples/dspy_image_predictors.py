"""
Predictor Examples: Using dspy.Image with DSPy Predictors

This module demonstrates how to use dspy.Image with DSPy predictors
in various multimodal scenarios.
"""

from dspy.utils.dummies import DummyLM

import dspy
import requests
from examples.dspy_image_signatures import (
    ALL_SIGNATURES,
    ImageCaptionSignature,
    ImageClassificationSignature,
    BoundingBoxDetectionSignature,
)


def get_random_dog_image_url():
    """Fetch a random dog image URL from the Dog API."""
    response = requests.get("https://dog.ceo/api/breeds/image/random")
    response.raise_for_status()
    return response.json()["message"]


def create_mock_signatures():
    """
    Create a helper function to set up predictors with mock responses.
    """

    def setup_predictor(signature, expected_output):
        """
        Set up a predictor with a DummyLM for testing.

        Args:
            signature: The signature class to use
            expected_output: Mock output for the predictor

        Returns:
            tuple: (Predictor, DummyLM) for testing
        """
        lm = DummyLM([expected_output])
        dspy.configure(lm=lm)
        predictor = dspy.Predict(signature)
        return predictor, lm

    return setup_predictor


def example1_basic_image_prediction():
    """
    Example 1: Basic image prediction for captioning.

    Show a simple flow of creating an image, defining a predictor,
    and getting a prediction.
    """

    # Create the predictor with the caption signature
    predictor = dspy.Predict(ImageCaptionSignature)

    # Create image from URL (download=True to get base64 encoding for real LMs)
    image_url = get_random_dog_image_url()
    print(f"Using dog image: {image_url}")
    image = dspy.Image(image_url, download=True)

    # Run the prediction
    result = predictor(image=image)

    print("=== Basic Image Captioning ===")
    print(f"Input image: {repr(image)}")
    print(f"Output caption: {result.caption}")
    # print(f"Prediction history: {len(lm.history)} call(s)")
    print()


def example2_image_classification():
    """
    Example 2: Image classification with probability scores.

    Demonstrates using image with a list of class labels
    and getting back classification probabilities.
    """

    predictor = dspy.Predict(ImageClassificationSignature)

    # Use download=True for real LMs that require base64-encoded images
    image_url = get_random_dog_image_url()
    print(f"Using dog image: {image_url}")
    image = dspy.Image(image_url, download=True)
    class_labels = ["dog", "cat", "bird", "other"]

    result = predictor(image=image, class_labels=class_labels)

    print("=== Image Classification ===")
    print(f"Image: {repr(image)}")
    print(f"Class labels: {class_labels}")
    print(f"Classification results: {result.probabilities}")
    max_class = max(result.probabilities, key=result.probabilities.get)
    print(f"Most likely class: {max_class} ({result.probabilities[max_class]:.2%})")
    print()


def example3_multiple_instantiations():
    """
    Example 3: Processing multiple images through the same predictor.

    Show how to use the same predictor with different image inputs.
    """

    predictor = dspy.Predict(ImageCaptionSignature)

    print("=== Multiple Image Predictions ===")
    for i in range(3):
        # Use download=True for real LMs that require base64-encoded images
        try:
            image_url = get_random_dog_image_url()
            print(f"Image {i+1}: Fetching random dog image...")
            image = dspy.Image(image_url, download=True)
        except Exception as e:
            print(f"Image {i+1}: Error loading image: {e}")
            continue
        result = predictor(image=image)
        print(f"  Caption: {result.caption}")
        print()


def example4_optional_image_field():
    """
    Example 4: Handling optional image fields.

    Demonstrate signatures where images are optional,
    and show what happens when image is None.
    """

    class OptionalCaptionSignature(dspy.Signature):
        """Generate caption if image is provided, text otherwise."""

        image: dspy.Image | None = dspy.InputField()
        analysis: str = dspy.OutputField()

    # Test with image provided
    predictor_with_image = dspy.Predict(OptionalCaptionSignature)
    # Use download=True for real LMs that require base64-encoded images
    image_url = get_random_dog_image_url()
    print(f"Using dog image: {image_url}")
    image = dspy.Image(image_url, download=True)
    result_with = predictor_with_image(image=image)

    # Test with image as None
    predictor_without_image = dspy.Predict(OptionalCaptionSignature)
    result_without = predictor_without_image(image=None)

    print("=== Optional Image Field ===")
    print(f"With image: {result_with.analysis}")
    print(f"Without image: {result_without.analysis}")
    print()


def example5_list_of_images():
    """
    Example 5: Processing a list of images.

    Demonstrate signatures that accept multiple images.
    """

    class MultiImageCaptionSignature(dspy.Signature):
        description: str = dspy.OutputField(desc="Combined description")

    predictor = dspy.Predict(MultiImageCaptionSignature)

    # Use download=True for real LMs that require base64-encoded images
    image_url1 = get_random_dog_image_url()
    print(f"Using dog image 1: {image_url1}")
    image_url2 = get_random_dog_image_url()
    print(f"Using dog image 2: {image_url2}")
    images = [
        dspy.Image(image_url1, download=True),
        dspy.Image(image_url2, download=True),
    ]

    result = predictor(images=images)

    print("=== List of Images ===")
    print(f"Number of images: {len(images)}")
    print(f"Combined description: {result.description}")
    print()


def example6_with_real_lm_demo():
    """
    Example 6: Example setup for real LMs (illustrative).

    This shows how to configure a real LM for multimodal tasks.
    Note: This requires actual API keys and won't run without them.
    """

    # This is commented out as it requires actual API credentials
    # import os
    # from litellm import Completion

    # os.environ["OPENAI_API_KEY"] = "your_api_key_here"

    # lm = dspy.LM(
    #     "openai/gpt-4o",
    #     cache=False,
    #     temperature=0.0
    # )

    # dspy.configure(lm=lm)

    # predictor = dspy.Predict(ImageCaptionSignature)
    # image = dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=True)
    # result = predictor(image=image)
    # print(f"Result: {result.caption}")

    print("=== Real LM Configuration (requires API key) ===")
    print("See commented code for how to configure real LMs")
    print("Required environment variables: OPENAI_API_KEY or similar")
    print()


def example7_different_input_formats():
    """
    Example 7: Processing images in different formats.

    Demonstrate handling various input types (URLs, PIL images, etc.)
    with the same predictor.
    """

    predictor = dspy.Predict(ImageCaptionSignature)

    import requests
    from io import BytesIO
    from PIL import Image as PILImage
    import base64

    # Format 1: URL string (download=True for real LMs)
    image_url = dspy.Image(
        "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=True
    )
    result1 = predictor(image=image_url)

    # Format 2: PIL Image
    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    response = requests.get(url)
    pil_img = PILImage.open(BytesIO(response.content))
    image_pil = dspy.Image(pil_img)
    result2 = predictor(image=image_pil)

    # Format 3: Raw bytes
    raw_bytes = response.content
    image_bytes = dspy.Image(raw_bytes)
    result3 = predictor(image=image_bytes)

    # Format 4: Data URI
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_str}"
    image_data = dspy.Image(data_uri)
    result4 = predictor(image=image_data)

    print("=== Different Input Formats ===")
    print("All formats produced predictions:")
    print(f"  URL format: {result1.caption}")
    print(f"  PIL format: {result2.caption}")
    print(f"  Bytes format: {result3.caption}")
    print(f"  Data URI: {result4.caption}")
    print()


def example8_signature_selection():
    """
    Example 8: Selecting different signatures for different tasks.

    Demonstrate how to use different signature types
    for various multimodal tasks.
    """

    print("=== Signature Selection Demo ===")
    print("Available signatures from examples.dspy_image_signatures:")

    for sig_name, sig_class in ALL_SIGNATURES.items():
        print(f"  - {sig_name}: {sig_class.__doc__.strip()}")

    print()


if __name__ == "__main__":
    from common import configure
    
    
    print("=" * 60)
    print("DSPy Image Module - Predictor Examples")
    print("=" * 60)
    print()

    # Real LM 
    configure(True)
    
    # Initialize for all examples
    # lm = DummyLM([{"caption": "Test output"}] * 100)
    # dspy.configure(lm=lm)

    example1_basic_image_prediction()

    # lm = DummyLM([{"probabilities": {"dog": 0.85, "cat": 0.1}}] * 100)
    # dspy.configure(lm=lm)

    example2_image_classification()

    # lm = DummyLM([{"caption": "Test"}] * 100)
    # dspy.configure(lm=lm)

    example3_multiple_instantiations()

    # lm = DummyLM(
    #     [
    #         {"analysis": "With image"},
    #         {"analysis": "Without image"},
    #     ]
    #     * 100
    # )
    # dspy.configure(lm=lm)

    example4_optional_image_field()

    example5_list_of_images()

    example6_with_real_lm_demo()

    example7_different_input_formats()

    example8_signature_selection()

    print("=" * 60)
    print("All predictor examples completed!")
    print("=" * 60)
