# DSPy Image Module - Example Code

This directory contains comprehensive examples demonstrating how to use the `dspy.Image` module for multimodal tasks with DSPy.

## Overview

The `dspy.Image` module allows you to pass image data to language models through DSPy signatures. Images can be sourced from URLs, local files, PIL Image objects, raw bytes, or data URIs. They are automatically encoded and formatted according to the OpenAI API image_url format.

## Requirements

```bash
pip install dspy-ai Pillow requests
```

## Example Files

### 1. [demo_dspy_image.py](demo_dspy_image.py) - Complete Demo Script

A comprehensive demonstration covering all major features of `dspy.Image`:

```bash
python demo_dspy_image.py
```

This script demonstrates:
- Basic image creation from various sources
- Simple predictor usage
- Different signature types
- Optional and nested image fields
- Adapter formats
- Few-shot learning with images

### 2. [dspy_image_basic.py](dspy_image_basic.py) - Basic Image Creation

Focuses on the fundamental ways to create `dspy.Image` objects:

```bash
python dspy_image_basic.py
```

Includes examples of:
- Creating images from URL strings
- Downloading and encoding URLs (`download=True`)
- Creating from local file paths
- Creating from PIL Image objects
- Creating from raw bytes
- Using data URIs
- Image fields in signature contexts

### 3. [dspy_image_signatures.py](dspy_image_signatures.py) - Signature Definitions

Defines various reusable signature classes for image tasks:

```python
from dspy_image_signatures import (
    ImageCaptionSignature,
    ImageClassificationSignature,
    BoundingBoxDetectionSignature,
    VisualQAASignature,
    MultiImageSignature,
    OptionalImageSignature,
    OCRSignature,
    # ... and more
)
```

Available signatures:
- `ImageCaptionSignature` - Generate captions for images
- `ImageClassificationSignature` - Classify images into categories
- `BoundingBoxDetectionSignature` - Detect objects with bounding boxes
- `MultiImageSignature` - Process lists of images
- `VisualQAASignature` - Visual question answering
- `OptionalImageSignature` - Handle optional image inputs
- `OCRSignature` - Text extraction from images
- `ImageToCodeSignature` - Generate code from UI screenshots

### 4. [dspy_image_predictors.py](dspy_image_predictors.py) - Predictor Examples

Shows how to use the signatures with actual predictors:

```bash
python dspy_image_predictors.py
```

Demonstrates:
- Basic image prediction for captioning
- Image classification with probability scores
- Multiple image instantiations
- Optional image fields
- List of images processing
- Different input format handling

### 5. [dspy_image_advanced.py](dspy_image_advanced.py) - Advanced Features

Covers advanced usage patterns:

```bash
python dspy_image_advanced.py
```

Topics include:
- Few-shot learning with images
- Custom validation metrics
- Error handling for invalid inputs
- Image download handling with different MIME types
- Save/load compiled predictors
- Custom adapter usage
- Batch processing
- Nested structures with Pydantic models
- Comparison of download vs URL-only modes

## Quick Start Examples

### Basic Usage

```python
import dspy

# Create an image from a URL
image = dspy.Image("https://example.com/image.jpg")

# Use in a predictor
class CaptionSignature(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    caption: str = dspy.OutputField()

predictor = dspy.Predict(CaptionSignature)
result = predictor(image=image)
print(result.caption)
```

### Download and Encode

```python
# Download and encode the image to base64
image = dspy.Image("https://example.com/image.jpg", download=True)

# Now it's a data URI
print(image.url)  # data:image/jpeg;base64,/9j/4AAQSkZJRg...
```

### Multiple Images

```python
class MultiImageSignature(dspy.Signature):
    images: list[dspy.Image] = dspy.InputField()
    descriptions: list[str] = dspy.OutputField()

predictor = dspy.Predict(MultiImageSignature)
result = predictor(
    images=[
        dspy.Image("https://example.com/image1.jpg"),
        dspy.Image("https://example.com/image2.jpg"),
    ]
)
print(result.descriptions)
```

### Optional Images

```python
class OptionalImageSignature(dspy.Signature):
    image: dspy.Image | None = dspy.InputField()
    analysis: str = dspy.OutputField()

# With image
predictor = dspy.Predict(OptionalImageSignature)
result = predictor(image=some_image)

# Without image
result = predictor(image=None)
```

### Few-Shot Learning

```python
from dspy.teleprompt import LabeledFewShot

# Create training examples
examples = [
    dspy.Example(
        image=dspy.Image("https://example.com/train1.jpg"),
        caption="Training caption 1"
    ).with_inputs("image"),
    
    dspy.Example(
        image=dspy.Image("https://example.com/train2.jpg"),
        caption="Training caption 2"
    ).with_inputs("image"),
]

# Compile predictor
student = dspy.Predict(CaptionSignature)
optimizer = LabeledFewShot(k=2)
compiled = optimizer.compile(student, trainset=examples)

# Use compiled predictor
result = compiled(image=test_image)
```

### From PIL Image

```python
from PIL import Image
from io import BytesIO
import dspy

# Create PIL image
pil_img = Image.new("RGB", (100, 100), color="red")

# Convert to dspy.Image
image = dspy.Image(pil_img)
```

### From Bytes

```python
import requests
import dspy

# Download image as bytes
response = requests.get("https://example.com/image.jpg")
image = dspy.Image(response.content)
```

## Common Use Cases

### Image Captioning
```python
class Caption(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    caption: str = dspy.OutputField()
```

### Image Classification
```python
class Classification(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    class_labels: list[str] = dspy.InputField()
    probabilities: dict[str, float] = dspy.OutputField()
```

### Visual Question Answering
```python
class VQA(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

### Object Detection
```python
class BBoxDetection(dspy.Signature):
    image: dspy.Image = dspy.InputField()
    bboxes: list[tuple[int, int, int, int]] = dspy.OutputField()
    labels: list[str] = dspy.OutputField()
```

## Notes

- By default, `dspy.Image` keeps URLs as-is without downloading. Use `download=True` to fetch and encode as base64.
- Images in signatures should be referenced as input or output fields.
- Optional image fields use `dspy.Image | None` annotation.
- Multiple images are passed as `list[dspy.Image]`.
- Images are automatically formatted for the underlying LM adapter.

## Documentation

For more information about the `dspy.Image` module:
- Source code: `ref/dspy/dspy/adapters/types/image.py`
- Tests: `ref/dspy/tests/signatures/test_adapter_image.py`
- Base type: `ref/dspy/dspy/adapters/types/base_type.py`

## License

Examples are provided under the same license as DSPy.
