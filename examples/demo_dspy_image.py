"""
Complete Demo: DSPy Image Module

This is a comprehensive demonstration of the dspy.Image module,
covering all major features and use cases.

Run with:
    python demo_dspy_image.py

Requirements:
    dspy-ai
    Pillow
    requests
"""

from dspy.utils.dummies import DummyLM

import dspy
from dspy.adapters import ChatAdapter, JSONAdapter, XMLAdapter

# =============================================================================
# SECTION 1: SIGNATURE DEFINITIONS
# =============================================================================


class ImageCaptionSignature(dspy.Signature):
    """Generate a caption for an image."""

    image: dspy.Image = dspy.InputField(desc="The image to caption")
    caption: str = dspy.OutputField(desc="Caption describing the image content")


class ImageClassificationSignature(dspy.Signature):
    """Classify an image into categories."""

    image: dspy.Image = dspy.InputField()
    class_labels: list[str] = dspy.InputField(desc="Possible categories")
    probabilities: dict[str, float] = dspy.OutputField(desc="Confidence scores")


class VisualQAASignature(dspy.Signature):
    """Answer questions about an image."""

    image: dspy.Image = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class MultiImageSignature(dspy.Signature):
    """Process multiple images."""

    images: list[dspy.Image] = dspy.InputField(desc="List of images")
    descriptions: list[str] = dspy.OutputField(desc="Descriptions for each image")


# =============================================================================
# SECTION 2: BASIC IMAGE CREATION
# =============================================================================


def section1_basic_image_creation():
    """Demonstrate creating dspy.Image objects from various sources."""
    print("=" * 70)
    print("SECTION 1: Basic Image Creation")
    print("=" * 70)
    print()

    # 1. From URL (default behavior - no download)
    url_image = dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")
    print("1. From URL (URL only, no download):")
    print(f"   {repr(url_image)}")
    print(f"   Is data URI: {url_image.url.startswith('data:')}")
    print()

    # 2. From URL (with download and encoding)
    downloaded_image = dspy.Image(
        "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=True
    )
    print("2. From URL (downloaded and encoded):")
    print(f"   Is data URI: {downloaded_image.url.startswith('data:')}")
    print(f"   Contains 'image/jpeg': {'image/jpeg' in downloaded_image.url}")
    print(f"   Length: ~{len(downloaded_image.url)} chars")
    print()

    # 3. From PIL Image
    from PIL import Image as PILImage
    from io import BytesIO
    import base64

    pil_img = PILImage.new("RGB", (100, 100), color="red")
    pil_image_obj = dspy.Image(pil_img)
    print("3. From PIL Image object:")
    print(f"   Created from RGB image")
    print(f"   Is data URI: {pil_image_obj.url.startswith('data:')}")
    print()

    # 4. From bytes
    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    import requests

    response = requests.get(url)
    bytes_image = dspy.Image(response.content)
    print("4. From raw bytes:")
    print(f"   Created from binary data")
    print(f"   Is data URI: {bytes_image.url.startswith('data:')}")
    print()

    # 5. From data URI (preserved as-is)
    # Create a sample PNG as data URI
    pil_small = PILImage.new("RGB", (50, 50), color="blue")
    buffered = BytesIO()
    pil_small.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_str}"

    data_uri_image = dspy.Image(data_uri)
    print("5. From data URI (preserved):")
    print(f"   Original data URI length: ~{len(data_uri)} chars")
    print(f"   Preserved as-is: {data_uri_image.url == data_uri}")
    print()

    print("✓ All basic image creation methods demonstrated\n")


# =============================================================================
# SECTION 3: SIMPLE PREDICTOR USAGE
# =============================================================================


def section2_simple_prediction():
    """Demonstrate basic predictor usage with images."""
    print("=" * 70)
    print("SECTION 2: Simple Prediction Examples")
    print("=" * 70)
    print()

    # Set up mock LM
    lm = DummyLM([{"caption": "A Great Dane dog in a field"}] * 100)
    dspy.configure(lm=lm)

    # Create predictor
    predictor = dspy.Predict(ImageCaptionSignature)

    # Create image input
    image = dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")

    # Run prediction
    result = predictor(image=image)

    print("Simple Image Captioning:")
    print(f"  Input: dspy.Image from URL")
    print(f"  Output: {result.caption}")
    print(f"  Prediction successful: {bool(result.caption)}")
    print()

    print("✓ Simple prediction completed\n")


# =============================================================================
# SECTION 4: VARIOUS SIGNATURE TYPES
# =============================================================================


def section3_signature_variations():
    """Demonstrate different signature types with images."""
    print("=" * 70)
    print("SECTION 3: Signature Variations")
    print("=" * 70)
    print()

    lm = DummyLM(
        [
            {"probabilities": {"dog": 0.85, "cat": 0.1, "other": 0.05}},
            {"answer": "The image contains a dog"},
            {"descriptions": ["Caption for image 1", "Caption for image 2"]},
        ]
    )
    dspy.configure(lm=lm)

    # 1. Classification signature
    print("1. Classification with probability outputs:")
    print("-" * 40)
    classifier = dspy.Predict(ImageClassificationSignature)
    image = dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")
    result = classifier(image=image, class_labels=["dog", "cat", "other"])
    print(f"   Classes: {list(result.probabilities.keys())}")
    print(
        f"   Highest confidence: {max(result.probabilities, key=result.probabilities.get)}"
    )
    print()

    # 2. Visual QA signature
    print("2. Visual Question Answering:")
    print("-" * 40)
    vqa = dspy.Predict(VisualQAASignature)
    result = vqa(
        image=dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"),
        question="What animal is in the image?",
    )
    print(f"   Question: 'What animal is in the image?'")
    print(f"   Answer: {result.answer}")
    print()

    # 3. Multi-image signature
    print("3. Multiple image processing:")
    print("-" * 40)
    multi = dspy.Predict(MultiImageSignature)
    results = multi(
        images=[
            dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"),
            dspy.Image("https://images.dog.ceo/breeds/pug/n02085620_1588.jpg"),
        ]
    )
    print(f"   Input images: {len(results.descriptions)}")
    print(f"   Descriptions: {results.descriptions}")
    print()

    print("✓ Signature variations demonstrated\n")


# =============================================================================
# SECTION 5: OPTIONAL AND NESTED IMAGES
# =============================================================================


def section4_optional_nested_images():
    """Demonstrate optional images and nested structures."""
    print("=" * 70)
    print("SECTION 4: Optional and Nested Images")
    print("=" * 70)
    print()

    # Optional image signature
    class OptionalImageSignature(dspy.Signature):
        optional: dspy.Image | None = dspy.InputField()
        output: str = dspy.OutputField()

    lm = DummyLM(
        [
            {"output": "Image analysis completed"},
            {"output": "No image, text analysis only"},
        ]
    )
    dspy.configure(lm=lm)

    predictor = dspy.Predict(OptionalImageSignature)

    # With image
    result_with = predictor(optional=dspy.Image("https://example.com/image.jpg"))

    # Without image (None)
    result_without = predictor(optional=None)

    print("Optional Image Field:")
    print(f"  With image: {result_with.output}")
    print(f"  Without image: {result_without.output}")
    print()

    # Pydantic model with images
    import pydantic

    class ImageModel(pydantic.BaseModel):
        primary: dspy.Image
        additional: list[dspy.Image] | None = None

        model_config = pydantic.ConfigDict(protected_namespaces=())

    lm = DummyLM([{"output": "Analysis complete"}])
    dspy.configure(lm=lm)

    class PydanticSignature(dspy.Signature):
        model_data: ImageModel = dspy.InputField()
        output: str = dspy.OutputField()

    predictor = dspy.Predict(PydanticSignature)

    model_input = ImageModel(
        primary=dspy.Image("https://example.com/primary.jpg"),
        additional=[
            dspy.Image("https://example.com/add1.jpg"),
            dspy.Image("https://example.com/add2.jpg"),
        ],
    )

    result = predictor(model_data=model_input)

    print("Nested Pydantic Model with Images:")
    print(f"  Primary image: included")
    print(f"  Additional images: {len(model_input.additional)}")
    print(f"  Output: {result.output}")
    print()

    print("✓ Optional and nested images demonstrated\n")


# =============================================================================
# SECTION 6: ADAPTERS AND FORMATTING
# =============================================================================


def section5_adapters():
    """Demonstrate different adapters with image tasks."""
    print("=" * 70)
    print("SECTION 5: Adapter Formats")
    print("=" * 70)
    print()

    adapters = [
        ("ChatAdapter", ChatAdapter),
        ("JSONAdapter", JSONAdapter),
        ("XMLAdapter", XMLAdapter),
    ]

    print("Available adapters for multimodal tasks:")
    for name, cls in adapters:
        adapter = cls()
        print(f"  - {name}: {type(adapter).__name__}")

    print()

    # Format an example with ChatAdapter
    adapter = ChatAdapter()

    class SimpleSig(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        result: str = dspy.OutputField()

    # Create a simple example
    example = dspy.Example(
        image=dspy.Image("https://example.com/test.jpg"), result="test output"
    ).with_inputs("image")

    # Format for the adapter (signature, demos, inputs)
    formatted = adapter.format(SimpleSig, [example], {"image": example.image})

    print(f"Sample formatted output type: {type(formatted).__name__}")
    if isinstance(formatted, list) and formatted:
        print(f"  Message count: {len(formatted)}")
        print(
            f"  Content includes image: {any(isinstance(c, dict) and 'image_url' in c for c in formatted)}"
        )

    print()
    print("✓ Adapter demonstration completed\n")


# =============================================================================
# SECTION 7: FEW-SHOT LEARNING
# =============================================================================


def section6_few_shot_learning():
    """Demonstrate few-shot learning with image examples."""
    print("=" * 70)
    print("SECTION 6: Few-Shot Learning")
    print("=" * 70)
    print()

    # Create training examples with images
    training_examples = [
        dspy.Example(
            image=dspy.Image(
                "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
            ),
            caption="A Great Dane standing",
        ).with_inputs("image"),
        dspy.Example(
            image=dspy.Image("https://images.dog.ceo/breeds/pug/n02085620_1588.jpg"),
            caption="A P Pug puppy sitting",
        ).with_inputs("image"),
        dspy.Example(
            image=dspy.Image("https://images.dog.ceo/breeds/beagle/n02088364_10564.jpg"),
            caption="A Beagle lying down",
        ).with_inputs("image"),
    ]

    print("Training Examples:")
    for i, example in enumerate(training_examples, 1):
        print(f"  {i}. Image: dspy.Image from URL")
        print(f"     Caption: {example.caption}")

    print()

    # Compile with LabeledFewShot
    lm = DummyLM([{"caption": "A dog in the scene"}] * 20)
    dspy.configure(lm=lm)

    predictor = dspy.Predict(ImageCaptionSignature)

    optimizer = dspy.teleprompt.LabeledFewShot(k=2)
    compiled = optimizer.compile(student=predictor, trainset=training_examples)

    print(f"Compiled with LabeledFewShot(k={optimizer.k})")
    print(f"Training examples used: {len(training_examples)}")
    print()

    # Test compiled predictor
    test_img = dspy.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")
    result = compiled(image=test_img)

    print(f"Test prediction: {result.caption}")
    print()

    print("✓ Few-shot learning demonstrated\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Run all demo sections."""

    print("\n")
    print("=" * 70)
    print("DSPy Image Module - Comprehensive Demo")
    print("=" * 70)
    print()
    print("This demo covers:")
    print("  - Creating dspy.Image from various sources")
    print("  - Using images in predictor signatures")
    print("  - Different signature types and variations")
    print("  - Optional and nested image fields")
    print("  - Adapters for model formatting")
    print("  - Few-shot learning with images")
    print()

    # Run all sections
    section1_basic_image_creation()
    section2_simple_prediction()
    section3_signature_variations()
    section4_optional_nested_images()
    section5_adapters()
    section6_few_shot_learning()

    # Summary
    print("=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print()
    print("✓ All sections completed successfully!")
    print()
    print("Key takeaways:")
    print(
        "  1. dspy.Image supports multiple input types (URL, file, PIL, bytes, data URI)"
    )
    print("  2. Images work with any dspy.Signature InputField")
    print("  3. Download parameter controls URL vs base64 encoding")
    print("  4. Supports single, multiple, and optional image fields")
    print("  5. Integrates with all DSPy adapters")
    print("  6. Compatible with compiled optimizers (e.g., LabeledFewShot)")
    print()
    print("For more examples, see:")
    print("  - examples/dspy_image_basic.py")
    print("  - examples/dspy_image_signatures.py")
    print("  - examples/dspy_image_predictors.py")
    print("  - examples/dspy_image_advanced.py")
    print()
    print("=" * 70)
    print("END OF DEMO")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
