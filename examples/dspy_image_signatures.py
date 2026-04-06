"""
Signature Definitions for DSPy Image Examples

This module defines various signature classes that use dspy.Image
for different multimodal tasks.
"""

import pydantic

import dspy


class ImageCaptionSignature(dspy.Signature):
    """Generate a detailed caption for an image.

    This signature takes a single image and produces a descriptive caption.
    """

    image: dspy.Image = dspy.InputField(desc="The input image to analyze")
    caption: str = dspy.OutputField(
        desc="A detailed caption describing the image content"
    )


class ImageClassificationSignature(dspy.Signature):
    """Classify an image into predefined categories.

    This signature takes an image and class labels, returning classification
    probabilities for each class.
    """

    image: dspy.Image = dspy.InputField(desc="The input image to classify")
    class_labels: list[str] = dspy.InputField(
        desc="List of possible class labels for classification"
    )
    probabilities: dict[str, float] = dspy.OutputField(
        desc="Confidence scores for each class label"
    )


class BoundingBoxDetectionSignature(dspy.Signature):
    """Detect and return bounding boxes for objects in an image.

    This signature analyzes an image and returns coordinates of detected objects.
    """

    image: dspy.Image = dspy.InputField(desc="The input image for object detection")
    bboxes: list[tuple[int, int, int, int]] = dspy.OutputField(
        desc="List of bounding boxes as (x_min, y_min, x_max, y_max) coordinates"
    )
    labels: list[str] = dspy.OutputField(
        desc="Labels corresponding to each bounding box"
    )


class MultiImageCaptionSignature(dspy.Signature):
    """Generate captions for multiple images.

    This signature accepts a list of images and generates captions for each.
    """

    images: list[dspy.Image] = dspy.InputField(desc="A list of images to analyze")
    image_descriptions: list[str] = dspy.OutputField(
        desc="A list of captions, one for each input image"
    )


class OptionalImageSignature(dspy.Signature):
    """Process an optional image with fallback text.

    This signature demonstrates handling optional image fields.
    """

    image: dspy.Image | None = dspy.InputField(
        desc="Optional image input. If None, provide text-based analysis"
    )
    analysis: str = dspy.OutputField(
        desc="Analysis result based on image (if provided) or text"
    )


class ImageToCodeSignature(dspy.Signature):
    """Generate HTML/CSS code based on a UI screenshot.

    This signature takes a screenshot and generates corresponding code.
    """

    screenshot: dspy.Image = dspy.InputField(
        desc="UI screenshot/image to convert to code"
    )
    target_language: str = dspy.InputField(
        desc="Target programming language (e.g., 'HTML', 'CSS')"
    )
    generated_code: str = dspy.OutputField(desc="Generated code matching the UI design")


class VisualQAASignature(dspy.Signature):
    """Answer questions about image content.

    This signature performs visual question answering on an image.
    """

    image: dspy.Image = dspy.InputField(desc="The image to analyze")
    question: str = dspy.InputField(desc="The question to answer about the image")
    answer: str = dspy.OutputField(desc="The answer to the question based on the image")


class PydanticImageSignature(dspy.Signature):
    """Demonstrate using dspy.Image within Pydantic models.

    This signature shows images nested inside a Pydantic model.
    """

    # Nested model containing image fields
    model_input: pydantic.BaseModel = dspy.InputField()
    analysis: str = dspy.OutputField()


class OCRSignature(dspy.Signature):
    """Extract and transcribe text from an image.

    This signature performs optical character recognition (OCR).
    """

    document_image: dspy.Image = dspy.InputField(
        desc="Image containing text to transcribe"
    )
    extracted_text: str = dspy.OutputField(desc="Text extracted from the image")


class DocumentAnalysisSignature(dspy.Signature):
    """Analyze document images (PDFs, scans).

    This signature can handle various document types including PDFs.
    """

    document: dspy.Image = dspy.InputField(desc="Document image or PDF to analyze")
    summary: str = dspy.OutputField(desc="Summary of the document content")


class SceneDescriptionSignature(dspy.Signature):
    """Describe a scene in detail.

    This signature provides comprehensive scene understanding.
    """

    scene_image: dspy.Image = dspy.InputField(desc="Image of the scene to describe")
    elements: list[str] = dspy.OutputField(
        desc="List of elements detected in the scene"
    )
    spatial_relations: list[dict[str, str]] = dspy.OutputField(
        desc="Spatial relationships between elements"
    )
    mood: str = dspy.OutputField(desc="Overall mood or atmosphere of the scene")


class ImageCaptionSignatureV2(ImageCaptionSignature):
    """Enhanced caption signature with additional metadata.

    A more detailed version with additional output fields.
    """

    image: dspy.Image = dspy.InputField(desc="The input image to analyze")
    caption: str = dspy.OutputField(
        desc="A detailed caption describing the image content"
    )
    detected_objects: list[str] = dspy.OutputField(
        desc="List of objects detected in the image"
    )
    scene_type: str = dspy.OutputField(desc="Type of scene (e.g., indoor, outdoor)")
    confidence_score: float = dspy.OutputField(
        desc="Confidence score of the prediction (0.0 to 1.0)"
    )


def create_pydantic_model_signature():
    """
    Helper function to create a custom Pydantic model for images.
    """

    class ImageAnalysisModel(pydantic.BaseModel):
        """Model for image analysis that can contain multiple images."""

        primary_image: dspy.Image
        reference_images: list[dspy.Image] | None = None
        image_description: str = "Initial description placeholder"

        class Config:
            protected_namespaces = ()

    return ImageAnalysisModel


# Alternative: Create model dynamically
def get_pydantic_model_signature():
    """
    Alternative way to create image model signature.
    """

    class AnalysisModel(pydantic.BaseModel):
        image_1: dspy.Image = dspy.InputField()
        image_2: dspy.Image | None = None
        description: str = dspy.OutputField()

    return AnalysisModel


class ColorAnalysisSignature(dspy.Signature):
    """Analyze colors in an image.

    This signature identifies dominant colors and palette.
    """

    image: dspy.Image = dspy.InputField(desc="Image to analyze for colors")
    dominant_colors: list[str] = dspy.OutputField(
        desc="List of dominant colors in the image"
    )
    color_palette: dict[str, str] = dspy.OutputField(
        desc="Palette with approximate CSS color values"
    )


class TextBasedDetectionSignature(dspy.Signature):
    """Detect objects in image with text-based guidance.

    This signature allows filtering detections by text description.
    """

    image: dspy.Image = dspy.InputField(desc="Image to analyze")
    search_query: str = dspy.InputField(
        desc="Text description of what to look for in the image"
    )
    results: dict[str, list[tuple[int, int, int, int]]] = dspy.OutputField(
        desc="Detected objects with bounding boxes"
    )


# Collection of all available signatures
ALL_SIGNATURES = {
    "caption": ImageCaptionSignature,
    "classification": ImageClassificationSignature,
    "bbox_detection": BoundingBoxDetectionSignature,
    "multi_image": MultiImageCaptionSignature,
    "optional_image": OptionalImageSignature,
    "code_generation": ImageToCodeSignature,
    "vqa": VisualQAASignature,
    "pydantic": PydanticImageSignature,
    "ocr": OCRSignature,
    "document_analysis": DocumentAnalysisSignature,
    "scene_description": SceneDescriptionSignature,
    "caption_v2": ImageCaptionSignatureV2,
    "color_analysis": ColorAnalysisSignature,
    "text_based_detection": TextBasedDetectionSignature,
}
