"""
Basic Examples: Creating dspy.Image Objects

This module demonstrates the various ways to create dspy.Image objects
from different input sources.
"""

import base64
from io import BytesIO

import requests
from PIL import Image as PILImage

import dspy
from examples.common import configure


def pause():
    pass


def example1_url_from_string():
    """
    Example 1: Create dspy.Image from a URL string.

    Image URLs are passed directly to the constructor.
    The URL is NOT downloaded by default, so it remains as a string URL.
    """
    image_url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    image = dspy.Image(image_url)

    print("URL Image String:", str(image))
    print("Repr:", repr(image))
    print("Image URL:", image.url)
    pause()


def example2_url_with_download():
    """
    Example 2: Download and encode an image from URL.

    Using download=True fetches the image data, determines the MIME type,
    and converts it to a base64 data URI.
    """
    image_url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    image = dspy.Image(image_url, download=True)

    print("Downloaded Image:", str(image)[:100])
    print("Starts with data:", image.url.startswith("data:"))
    print("Has base64:", "base64," in image.url)
    pause()


def example3_from_file_path():
    """
    Example 3: Create dspy.Image from a local file path.

    File paths are automatically encoded to base64 data URIs.
    """
    # Create a temporary image file for demonstration
    pil_image = PILImage.new("RGB", (100, 100), color="red")
    temp_file = "/tmp/test_image.png"
    pil_image.save(temp_file)

    image = dspy.Image(temp_file)

    print("File Image:", str(image))
    print("MIME type in URL:", "image/png" in image.url)
    pause()


def example3_1_from_file_path():
    """
    Example 3: Create dspy.Image from a local file path.

    File paths are automatically encoded to base64 data URIs.
    """
    # Create a temporary image file for demonstration
    # pil_image = PILImage.new("RGB", (100, 100), color="red")
    # temp_file = "/tmp/test_image.png"
    # pil_image.save(temp_file)

    image = dspy.Image("data/nana.jpg")

    print("File Image:", str(image)[:256])
    print("MIME type in URL:", "image/png" in image.url)
    pause()


def example4_from_pil_image():
    """
    Example 4: Create dspy.Image from a PIL Image object.

    PIL Image objects are automatically encoded to base64.
    """
    # Create a sample PIL image
    pil_image = PILImage.new("RGB", (100, 100), color="blue")

    # Convert to RGB if it has alpha channel
    if pil_image.mode in ("RGBA", "LA", "P"):
        pil_image = pil_image.convert("RGB")

    image = dspy.Image(pil_image)

    print("PIL Image:", str(image))
    print("Has base64:", "base64," in image.url)
    pause()


def example5_from_pil_url():
    """
    Example 5: Load PIL from a URL and convert to dspy.Image.

    First download and open as PIL, then wrap in dspy.Image.
    """
    image_url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    response = requests.get(image_url)
    response.raise_for_status()

    pil_image = PILImage.open(BytesIO(response.content))
    dspy_image = dspy.Image(pil_image)

    print("PIL from URL:", str(dspy_image))
    pause()


def example6_from_bytes():
    """
    Example 6: Create dspy.Image from raw bytes.

    Raw image bytes are encoded to a data URI.
    """
    image_url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    response = requests.get(image_url)
    response.raise_for_status()

    raw_bytes = response.content
    image = dspy.Image(raw_bytes)

    print("From Bytes:", str(image))
    print("Has base64:", "base64," in image.url)
    pause()


def example7_from_data_uri():
    """
    Example 7: Create dspy.Image from a data URI.

    Data URIs are used as-is without re-encoding.
    """
    # Create a simple base64-encoded PNG
    pil_image = PILImage.new("RGB", (50, 50), color="green")
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_str}"

    image = dspy.Image(data_uri)

    print("Data URI Image:", str(image))
    print("Original data URI:", data_uri[:50], "...")
    pause()


def example8_in_signature_context():
    """
    Example 8: Using dspy.Image within signature inputs.

    Images can be passed directly to predictors as input fields.
    """

    # Define a signature that takes an image
    class CaptionSignature(dspy.Signature):
        """Generate a caption for the provided image."""

        image: dspy.Image = dspy.InputField()
        caption: str = dspy.OutputField()

    image = dspy.Image(
        url="https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=True
    )
    print("Signature Field for 'image':", image)

    ai = dspy.Predict(CaptionSignature)
    result = ai(image=image)
    print(result.caption)

    pause()


def example9_optional_image():
    """
    Example 9: Optional image fields.

    Images can be optional in signatures.
    """

    class OptionalCaptionSignature(dspy.Signature):
        """If only image is provided, generate a caption for answer.
        If query is provided, answer property with image & query.
        """

        image: dspy.Image | None = dspy.InputField(default=None)
        query: dspy.Image | None = dspy.InputField(default=None)
        answer: str = dspy.OutputField()

    ai = dspy.Predict(OptionalCaptionSignature)

    print("\nCase 1: Only Image")
    print("-" * 100)
    result = ai(image=dspy.Image("data/nana.jpg"))  # working !!! [IMPORTANT]
    print(result.answer)

    print("\nCase 1.5: Only Image(just path)")
    print("-" * 100)
    result = ai(image="data/nana.jpg")  # NOT working !!! [IMPORTANT]
    print(result.answer)

    print("\nCase 2: Only Query")
    print("-" * 100)
    result = ai(query="Why is the sky blue?")
    print(result.answer)

    print("\nCase 3: Image & Query")
    print("-" * 100)
    result = ai(
        image=dspy.Image("data/nana.jpg"), query="What's color of the animal?"
    )  # working !!! [IMPORTANT]
    print(result.answer)


def example10_list_of_images():
    """
    Example 10: Lists of images.

    Signatures can accept multiple images as a list.
    """

    class MultiImageCaptionSignature(dspy.Signature):
        """Generate description for multiple images."""

        images: list[dspy.Image] = dspy.InputField(desc="Multiple images to analyze")
        description: str = dspy.OutputField(
            desc="Description on inputted multiple images"
        )

    images = [
        dspy.Image(
            "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=True
        ),
        dspy.Image(
            "https://images.dog.ceo/breeds/terrier-lakeland/n02095570_6471.jpg",
            download=True,
        ),
    ]

    print("List of Images:", images)
    print("Number of images:", len(images))

    predictor = dspy.Predict(MultiImageCaptionSignature)
    result = predictor(images=images)
    print(result.description)
    pause()


def full_test():
    print("=" * 60)
    print("DSPy Image Module - Basic Examples")
    print("=" * 60)
    pause()

    print("Example 1: URL from String")
    print("-" * 40)
    example1_url_from_string()

    print("Example 2: Download and Encode URL")
    print("-" * 40)
    example2_url_with_download()

    print("Example 3: From File Path")
    print("-" * 40)
    example3_from_file_path()

    print("Example 4: From PIL Image")
    print("-" * 40)
    example4_from_pil_image()

    print("Example 5: PIL from URL")
    print("-" * 40)
    example5_from_pil_url()

    print("Example 6: From Raw Bytes")
    print("-" * 40)
    example6_from_bytes()

    print("Example 7: From Data URI")
    print("-" * 40)
    example7_from_data_uri()

    print("Example 8: Signature Context")
    print("-" * 40)
    example8_in_signature_context()
    # exit()

    print("Example 9: Optional Image")
    print("-" * 40)
    example9_optional_image()

    print("Example 10: List of Images")
    print("-" * 40)
    example10_list_of_images()

    print("=" * 60)
    print("All basic examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    configure()
    # full_test()
    # example8_in_signature_context() # pass
    # example3_1_from_file_path()
    # example9_optional_image()
    example10_list_of_images()
