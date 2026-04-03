"""
Image processing stub - DISABLED for faster deployment

All image processing has been removed to:
- Speed up PDF processing
- Reduce deployment complexity
- Eliminate Vision API costs
- Simplify dependencies

If you need images later, restore from backup.
"""

# All image processing is disabled
# This module is kept as a stub for backward compatibility

async def generate_image_descriptions(images, openai_api_key, context_map=None):
    """Stub: returns empty list (no image processing)"""
    return []

def process_image_for_embedding(image_metadata, description=None, context=None):
    """Stub: returns None (no image processing)"""
    return None
