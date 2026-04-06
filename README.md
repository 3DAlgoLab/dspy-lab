# DSPy Image Lab 🐕

A hands-on experimentation environment for learning and exploring DSPy's multimodal capabilities, specifically the `dspy.Image` module.

## Overview

This project provides a comprehensive set of examples demonstrating how to use `dspy.Image` with DSPy predictors for various multimodal AI tasks, including:

- **Image Captioning**: Generate descriptive captions for images
- **Image Classification**: Classify images with confidence scores
- **Visual Question Answering**: Answer questions about image content
- **Multi-Image Processing**: Handle multiple images in a single request
- **Optional Image Fields**: Handle cases where images may or may not be provided

## Project Structure

```
dspy-lab/
├── examples/                    # Example scripts demonstrating dspy.Image usage
│   ├── demo_dspy_image.py       # Complete comprehensive demo (all features)
│   ├── dspy_image_basic.py      # Basic image creation examples
│   ├── dspy_image_signatures.py # Signature definitions for image tasks
│   ├── dspy_image_predictors.py # Predictor usage with images
│   ├── dspy_image_advanced.py   # Advanced patterns and use cases
│   ├── optional_signature_example.py  # Optional image field handling
│   └── common.py                # Shared utilities (LM configuration)
├── docs/                        # Documentation
│   ├── plan01-dspy-image-example.md   # Implementation plan
│   └── optional-signature-example.md  # Optional fields documentation
├── data/                        # Data directory
├── ref/                         # Reference DSPy source code
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for package management
- API access to an LLM (OpenAI, local LM Studio, etc.)

### Setup

1. **Clone the repository** (if applicable)

2. **Set up environment variables** in `.env`:
   ```env
   LLM_MODEL=openai/gpt-4o-mini  # or your preferred model
   LLM_API_BASE=https://api.openai.com/v1  # or your local endpoint
   LLM_API_KEY=your-api-key-here
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Run the examples**:
   ```bash
   # Run the comprehensive demo
   uv run examples/demo_dspy_image.py
   
   # Run basic examples
   uv run examples/dspy_image_basic.py
   
   # Run predictor examples (uses real LM)
   uv run examples/dspy_image_predictors.py
   
   # Run signature examples
   uv run examples/dspy_image_signatures.py
   ```

## Key Features

### 1. Creating dspy.Image Objects

```python
import dspy

# From URL (no download - passes URL to LM)
image = dspy.Image("https://example.com/image.jpg")

# From URL with download (base64 encodes for LM compatibility)
image = dspy.Image("https://example.com/image.jpg", download=True)

# From PIL Image
from PIL import Image as PILImage
pil_img = PILImage.open("local_image.jpg")
image = dspy.Image(pil_img)

# From bytes
with open("image.jpg", "rb") as f:
    image = dspy.Image(f.read())
```

### 2. Using Images in Signatures

```python
import dspy

class ImageCaptionSignature(dspy.Signature):
    """Generate a caption for an image."""
    image: dspy.Image = dspy.InputField(desc="The image to caption")
    caption: str = dspy.OutputField(desc="Caption describing the image content")
```

### 3. Using Images with Predictors

```python
# Create predictor
predictor = dspy.Predict(ImageCaptionSignature)

# Create image with download=True for real LMs
image = dspy.Image("https://dog.ceo/api/breeds/image/random", download=True)

# Get prediction
result = predictor(image=image)
print(result.caption)
```

### 4. Optional Image Fields

```python
class OptionalImageSignature(dspy.Signature):
    image: dspy.Image | None = dspy.InputField()
    analysis: str = dspy.OutputField()

predictor = dspy.Predict(OptionalImageSignature)

# Works with image
result_with = predictor(image=dspy.Image("dog.jpg", download=True))

# Works without image
result_without = predictor(image=None)
```

### 5. Using the Dog API for Dynamic Images

The examples use the [Dog CEO API](https://dog.ceo/dog-api/) to fetch random dog images:

```python
import requests

def get_random_dog_image_url():
    """Fetch a random dog image URL from the Dog API."""
    response = requests.get("https://dog.ceo/api/breeds/image/random")
    response.raise_for_status()
    return response.json()["message"]

# Use in your code
image_url = get_random_dog_image_url()
image = dspy.Image(image_url, download=True)
```

## Examples Guide

| Example | Description |
|---------|-------------|
| `demo_dspy_image.py` | Complete demo covering all features |
| `dspy_image_basic.py` | Creating Image objects from various sources |
| `dspy_image_signatures.py` | Signature definitions for image tasks |
| `dspy_image_predictors.py` | Using predictors with real LMs |
| `dspy_image_advanced.py` | Advanced patterns and edge cases |
| `optional_signature_example.py` | Optional image field handling |

## Important Notes

### Using with Real LMs

When using real Language Models (like GPT-4V, Claude, or local models via LM Studio):

1. **Always use `download=True`** for URL images:
   ```python
   image = dspy.Image("https://example.com/image.jpg", download=True)
   ```
   This downloads and base64-encodes the image, which most LMs require.

2. **Configure your LM** via `common.py` or directly:
   ```python
   from examples.common import configure
   configure()  # Uses .env variables
   ```

3. **Handle image loading errors** when fetching from external APIs:
   ```python
   try:
       image = dspy.Image(url, download=True)
   except Exception as e:
       print(f"Error loading image: {e}")
   ```

### Using with DummyLM (Testing)

For testing without API calls, DSPy provides `DummyLM`:

```python
from dspy.utils.dummies import DummyLM

lm = DummyLM([{"caption": "A dog in a field"}])
dspy.configure(lm=lm)
```

## Troubleshooting

### "No LM is loaded" Error

Run `configure()` from `examples.common` before using predictors:

```python
from examples.common import configure
configure()
```

### "url field must be a base64 encoded image" Error

Use `download=True` when creating images from URLs:

```python
image = dspy.Image(url, download=True)  # Required for real LMs
```

### 404 Errors with Dog API Images

Some dog.ceo URLs may return 404. The examples handle this gracefully with try/except blocks.

## Dependencies

- `dspy>=3.1.3` - DSPy framework
- `pillow>=12.2.0` - Image processing
- `requests` - HTTP requests (for API calls)
- `pydantic` - Data validation

## License

This project is for educational purposes. The DSPy library has its own license.

## Resources

- [DSPy Documentation](https://dspy.ai/)
- [Dog CEO API](https://dog.ceo/dog-api/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
