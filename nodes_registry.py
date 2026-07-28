import re
from functools import wraps
from typing import Callable, Optional, Type

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODES_DISPLAY_NAME_PREFIX = "🅛🅣🅧"
EXPERIMENTAL_DISPLAY_NAME_PREFIX = "(Experimental 🧪)"
DEPRECATED_DISPLAY_NAME_PREFIX = "(Deprecated 🚫)"
DEFAULT_CATEGORY_NAME = "Lightricks"

# ComfyUI groups hosted API nodes under its own top-level root; relocating one
# of those would hide it from that section, so leave them where they are.
EXTERNAL_CATEGORY_ROOTS = ("api node",)


def normalize_category(category: Optional[str]) -> str:
    """Force every node into one coherent menu tree.

    ComfyUI groups the node browser by exact category string, so
    ``lightricks/LTXV`` and ``Lightricks/latents`` render as two separate
    top-level folders, and a bare ``sampling`` scatters nodes into ComfyUI's
    own menus. Both were happening across this package.
    """
    if not category:
        return DEFAULT_CATEGORY_NAME

    root = category.split("/", 1)[0]
    if root.casefold() == DEFAULT_CATEGORY_NAME.casefold():
        # Re-case the root only; the rest of the path is left alone.
        return DEFAULT_CATEGORY_NAME + category[len(root) :]
    if root.casefold() in {r.casefold() for r in EXTERNAL_CATEGORY_ROOTS}:
        return category
    return f"{DEFAULT_CATEGORY_NAME}/{category}"


def register_node(node_class: Type, name: str, description: str) -> None:
    """
    Register a ComfyUI node class to ComfyUI's global nodes' registry.

    Args:
        node_class (Type): The class of the node to be registered.
        name (str): The name of the node.
        description (str): The short user-friendly description of the node.

    Raises:
        ValueError: If `node_class` is not a class, or `class_name` or `display_name` is not a string.
    """

    if not isinstance(node_class, type):
        raise ValueError("`node_class` must be a class")

    if not isinstance(name, str):
        raise ValueError("`name` must be a string")

    if not isinstance(description, str):
        raise ValueError("`description` must be a string")

    NODE_CLASS_MAPPINGS[name] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[name] = description


def _is_v3_node(node_class: Type) -> bool:
    """Check if the node class is a v3 node (has define_schema method)."""
    return hasattr(node_class, "define_schema") and callable(
        getattr(node_class, "define_schema")
    )


def _wrap_define_schema(node_class: Type, display_name: str) -> None:
    """Wrap define_schema to inject the display_name and normalize the category."""
    original_define_schema = node_class.define_schema

    @classmethod
    @wraps(original_define_schema.__func__)
    def wrapped_define_schema(cls):
        schema = original_define_schema.__func__(cls)
        # Only set display_name if not already set in the schema
        if schema.display_name is None:
            schema.display_name = display_name
        schema.category = normalize_category(getattr(schema, "category", None))
        return schema

    node_class.define_schema = wrapped_define_schema


def comfy_node(
    node_class: Optional[Type] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    experimental: bool = False,
    deprecated: bool = False,
    skip: bool = False,
) -> Callable:
    """
    Decorator for registering a node class with optional name, description, and status flags.

    Args:
        node_class (Type): The class of the node to be registered.
        name (str, optional): The name of the class. If not provided, the class name will be used.
        description (str, optional): The description of the class.
          If not provided, an auto-formatted description will be used based on the class name.
        experimental (bool): Flag indicating if the class is experimental. Defaults to False.
        deprecated (bool): Flag indicating if the class is deprecated. Defaults to False.
        skip (bool): Flag indicating if the node registration should be skipped. Defaults to False.
          This is useful for conditionally registering nodes based on certain conditions
          (e.g. unavailability of certain dependencies).

    Returns:
        Callable: The decorator function.

    Raises:
        ValueError: If `node_class` is not a class.
    """

    def decorator(node_class: Type) -> Type:
        if skip:
            return node_class

        if not isinstance(node_class, type):
            raise ValueError("`node_class` must be a class")

        nonlocal name, description
        if name is None:
            name = node_class.__name__

            # Remove possible "Node" suffix from the class name, e.g. "EditImageNode -> EditImage"
            if name is not None and name.endswith("Node"):
                name = name[:-4]

        description = _format_description(description, name, experimental, deprecated)

        # For v3 nodes, wrap define_schema to inject the display_name
        if _is_v3_node(node_class):
            _wrap_define_schema(node_class, description)
        else:
            node_class.CATEGORY = normalize_category(
                getattr(node_class, "CATEGORY", None)
            )

        register_node(node_class, name, description)
        return node_class

    # If the decorator is used without parentheses
    if node_class is None:
        return decorator
    else:
        return decorator(node_class)


def _format_description(
    description: str, class_name: str, experimental: bool, deprecated: bool
) -> str:
    """Format nodes display name to a standard format"""

    # If description is not provided, auto-generate one based on the class name
    if description is None:
        description = camel_case_to_spaces(class_name)

    # Strip the prefix if it's already there
    prefix_len = len(NODES_DISPLAY_NAME_PREFIX)
    if description.startswith(NODES_DISPLAY_NAME_PREFIX):
        description = description[prefix_len:].lstrip()

    # Add the deprecated / experimental prefixes
    if deprecated:
        description = f"{DEPRECATED_DISPLAY_NAME_PREFIX} {description}"
    elif experimental:
        description = f"{EXPERIMENTAL_DISPLAY_NAME_PREFIX} {description}"

    # Add the prefix
    description = f"{NODES_DISPLAY_NAME_PREFIX} {description}"

    return description


def camel_case_to_spaces(text: str) -> str:
    # Add space before each capital letter except the first one
    spaced_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    # Handle sequences of uppercase letters followed by a lowercase letter
    spaced_text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced_text)
    # Handle sequences of uppercase letters not followed by a lowercase letter
    spaced_text = re.sub(r"(?<=[A-Z])(?=[A-Z][A-Z][a-z])", " ", spaced_text)
    return spaced_text
