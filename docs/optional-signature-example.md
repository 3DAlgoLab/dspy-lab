# DSPy Optional Signature Example

## Overview

DSPy does not natively support optional fields in Signatures. However, you can work around this limitation using Python's `typing.Optional` type annotation with default values.

## Implementation Approaches

### Approach 1: Using `Optional` Type with Default Values (Recommended)

```python
import dspy
from typing import Optional

class QuestionAnswering(dspy.Signature):
    """Answer questions based on the given context.
    
    Given a question and optional context, provide an accurate answer.
    If no context is provided, use your general knowledge.
    """
    question: str = dspy.InputField(desc="The question to answer")
    context: Optional[str] = dspy.InputField(
        desc="Context or background information (optional)", 
        default=None
    )
    # Note: Output fields need a default value for optional signatures
    answer: str = dspy.OutputField(default="", desc="The answer to the question")

# Usage examples
# With context
example_with_context = QuestionAnswering(question="What is DSPy?", context="DSPy is a framework for programming LLMs.")
print(example_with_context)

# Without context (context will be None)
example_without_context = QuestionAnswering(question="What is the capital of France?")
print(example_without_context.context)  # Output: None
```

### Approach 2: Using Union Type with Multiple Signatures

For more complex scenarios, you can create multiple signature variants:

```python
import dspy
from typing import Optional

class QuestionAnsweringWithContext(dspy.Signature):
    """Answer questions when context is available."""
    question: str = dspy.InputField(desc="The question to answer")
    context: str = dspy.InputField(desc="Context or background information")
    answer: str = dspy.OutputField(desc="The answer to the question")

class QuestionAnsweringWithoutContext(dspy.Signature):
    """Answer questions using general knowledge."""
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="The answer to the question")

# Usage
def answer_question(question: str, context: Optional[str] = None):
    if context:
        signature = QuestionAnsweringWithContext
    else:
        signature = QuestionAnsweringWithoutContext
    
    example = signature(question=question, context=context)
    return example
```

### Approach 3: Custom Field with Validation

For more control over optional field behavior:

```python
import dspy
from typing import Optional
from pydantic import Field as PydanticField

class OptionalInputField(dspy.InputField):
    """Custom InputField that supports optional values."""
    
    def __init__(self, desc: str = None, default=None, **kwargs):
        super().__init__(desc=desc, **kwargs)
        self.default = default
    
    @classmethod
    def create(cls, desc: str = None, default=None, **kwargs):
        """Factory method to create an optional input field."""
        return cls(desc=desc, default=default, **kwargs)

class FlexibleSignature(dspy.Signature):
    """A signature with flexible optional fields."""
    
    question: str = dspy.InputField(desc="The main query")
    metadata: Optional[dict] = PydanticField(
        default=None,
        json_schema_extra={
            "__dspy_field_type": "input",
            "desc": "Additional metadata (optional)",
            "prefix": "Metadata:"
        }
    )
    answer: str = dspy.OutputField(desc="The response")

# Usage
example1 = FlexibleSignature(question="What is DSPy?", metadata={"source": "documentation"})
example2 = FlexibleSignature(question="What is Python?")  # metadata will be None
```

## Important Notes

1. **Type Safety**: Using `Optional` provides better type hints and IDE support.
2. **Default Values Required**: Pydantic requires default values for all optional fields (both input and output).
3. **Output Fields**: Output fields must have a default value (e.g., `default=""`) to allow instantiation without providing them.
4. **Prompt Generation**: When generating prompts for LLMs, you may need to handle None values explicitly in your prompt templates.

## Best Practices

- Use `Optional[Type]` with `default=None` for truly optional fields
- Document the optional nature of fields in their descriptions
- Consider creating separate signature classes if optional behavior significantly changes the task logic
- Test both with and without optional field values to ensure robustness
