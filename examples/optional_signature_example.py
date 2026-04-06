"""
DSPy Optional Signature Examples

This module demonstrates how to work with optional fields in DSPy Signatures.
Note: DSPy does not natively support optional fields, but we can use Python's
typing.Optional with default values as a workaround.
"""

import dspy
from typing import Optional

# ============================================================================
# Example 1: Basic Optional Field Using Optional Type
# ============================================================================


class QuestionAnswering(dspy.Signature):
    """Answer questions based on the given context.

    Given a question and optional context, provide an accurate answer.
    If no context is provided, use your general knowledge.
    """

    question: str = dspy.InputField(desc="The question to answer")
    context: Optional[str] = dspy.InputField(
        desc="Context or background information (optional)", default=None
    )
    answer: str = dspy.OutputField(default="", desc="The answer to the question")


def test_basic_optional():
    """Test basic optional field usage."""
    print("=" * 60)
    print("Example 1: Basic Optional Field")
    print("=" * 60)

    # With context
    example_with_context = QuestionAnswering(
        question="What is DSPy?", context="DSPy is a framework for programming LLMs."
    )
    
    print(f"\nWith context:")
    print(f"  Question: {example_with_context.question}")
    print(f"  Context: {example_with_context.context}")

    # Without context (context will be None)
    example_without_context = QuestionAnswering(
        question="What is the capital of France?"
    )
    print(f"\nWithout context:")
    print(f"  Question: {example_without_context.question}")
    print(f"  Context: {example_without_context.context}")

    # Check if field exists in signature
    print(f"\nInput fields: {list(QuestionAnswering.input_fields.keys())}")
    print(f"Output fields: {list(QuestionAnswering.output_fields.keys())}")


# ============================================================================
# Example 2: Multiple Optional Fields
# ============================================================================


class FlexibleQA(dspy.Signature):
    """Flexible question answering with multiple optional parameters."""

    question: str = dspy.InputField(desc="The main question")
    context: Optional[str] = dspy.InputField(
        desc="Context information (optional)", default=None
    )
    hints: Optional[list[str]] = dspy.InputField(
        desc="Additional hints or constraints (optional)", default=[]
    )
    answer: str = dspy.OutputField(default="", desc="The final answer")


def test_multiple_optional():
    """Test multiple optional fields."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Optional Fields")
    print("=" * 60)

    # With all fields
    full_example = FlexibleQA(
        question="How do I get started with DSPy?",
        context="DSPy is a Python framework for LLM programming.",
        hints=["Keep it simple", "Mention key features"],
    )
    print(f"\nFull example:")
    print(f"  Question: {full_example.question}")
    print(f"  Context: {full_example.context}")
    print(f"  Hints: {full_example.hints}")

    # With only required field
    minimal_example = FlexibleQA(question="What is Python?")
    print(f"\nMinimal example:")
    print(f"  Question: {minimal_example.question}")
    print(f"  Context: {minimal_example.context}")
    print(f"  Hints: {minimal_example.hints}")


# ============================================================================
# Example 3: Using Union Type for Complex Optional Types
# ============================================================================


class AdvancedSignature(dspy.Signature):
    """Advanced signature with complex optional types."""

    input_text: str = dspy.InputField(desc="The text to process")
    options: Optional[dict] = dspy.InputField(
        desc="Processing options (optional)", default=None
    )
    result: str = dspy.OutputField(default="", desc="The processed result")


def test_union_type():
    """Test Union type for optional fields."""
    print("\n" + "=" * 60)
    print("Example 3: Complex Optional Types")
    print("=" * 60)

    # With options dict
    with_options = AdvancedSignature(
        input_text="Hello, world!", options={"uppercase": True, "prefix": "---"}
    )
    print(f"\nWith options:")
    print(f"  Input: {with_options.input_text}")
    print(f"  Options: {with_options.options}")

    # Without options
    without_options = AdvancedSignature(input_text="Hello!")
    print(f"\nWithout options:")
    print(f"  Input: {without_options.input_text}")
    print(f"  Options: {without_options.options}")


# ============================================================================
# Example 4: Runtime Field Selection Based on Availability
# ============================================================================


class QuestionAnsweringWithContext(dspy.Signature):
    """Answer questions when context is available."""

    question: str = dspy.InputField(desc="The question to answer")
    context: str = dspy.InputField(desc="Context or background information")
    answer: str = dspy.OutputField(default="", desc="The answer to the question")


class QuestionAnsweringWithoutContext(dspy.Signature):
    """Answer questions using general knowledge."""

    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(default="", desc="The answer to the question")


def adaptive_question_answering(question: str, context: Optional[str] = None):
    """Adaptively choose signature based on context availability."""

    if context:
        print(f"\nUsing signature WITH context...")
        signature = QuestionAnsweringWithContext
    else:
        print(f"\nUsing signature WITHOUT context...")
        signature = QuestionAnsweringWithoutContext

    example = signature(question=question, context=context)
    return example


def test_adaptive():
    """Test adaptive signature selection."""
    print("\n" + "=" * 60)
    print("Example 4: Adaptive Signature Selection")
    print("=" * 60)

    # With context - uses QuestionAnsweringWithContext
    result1 = adaptive_question_answering(
        "What is DSPy?", context="DSPy is a framework for programming LLMs."
    )
    print(f"Result: {result1}")

    # Without context - uses QuestionAnsweringWithoutContext
    result2 = adaptive_question_answering("What is Python?")
    print(f"Result: {result2}")


# ============================================================================
# Example 5: Using Signature Methods with Optional Fields
# ============================================================================


def test_signature_methods():
    """Test DSPy signature methods with optional fields."""
    print("\n" + "=" * 60)
    print("Example 5: Signature Methods")
    print("=" * 60)

    # Using .prepend() to add a required field
    original = QuestionAnswering
    print(f"\nOriginal signature:")
    print(f"  Input fields: {list(original.input_fields.keys())}")

    new_sig = original.prepend("topic", dspy.InputField(desc="Topic category"))
    print(f"\nAfter prepend 'topic':")
    print(f"  Input fields: {list(new_sig.input_fields.keys())}")

    # Using .delete() to remove optional field
    deleted_sig = new_sig.delete("context")
    print(f"\nAfter delete 'context':")
    print(f"  Input fields: {list(deleted_sig.input_fields.keys())}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    test_basic_optional()
    test_multiple_optional()
    test_union_type()
    test_adaptive()
    test_signature_methods()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
