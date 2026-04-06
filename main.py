"""
DSPy Image Lab - Main Entry Point

This is the most representative example of using dspy.Image with a real LM.
It demonstrates image captioning using random dog images from the Dog API.
"""

import dspy
import requests
from examples.common import configure


def get_random_dog_image_url():
    """Fetch a random dog image URL from the Dog API."""
    response = requests.get("https://dog.ceo/api/breeds/image/random")
    response.raise_for_status()
    return response.json()["message"]


class ImageCaption(dspy.Signature):
    """Generate a descriptive caption for the given image."""
    image: dspy.Image = dspy.InputField(desc="The image to caption")
    caption: str = dspy.OutputField(desc="A detailed description of the image")


def main():
    print("=" * 60)
    print("DSPy Image Lab - Image Captioning Example")
    print("=" * 60)
    print()

    # Configure the Language Model from .env
    print("Configuring LM...")
    configure()
    print()

    # Fetch a random dog image URL
    print("Fetching random dog image from Dog API...")
    image_url = get_random_dog_image_url()
    print(f"Image URL: {image_url}")
    print()

    # Create the image with download=True (required for real LMs)
    print("Loading image...")
    image = dspy.Image(image_url, download=True)
    print(f"Image loaded: {len(image.url)} characters")
    print()

    # Create predictor and generate caption
    print("Generating caption with LM...")
    print("-" * 60)
    predictor = dspy.Predict(ImageCaption)
    result = predictor(image=image)

    # Display results
    print("CAPTION:")
    print(result.caption)
    print("-" * 60)
    print()
    print("Done! Run again for a different dog image.")
    print("=" * 60)


if __name__ == "__main__":
    main()
