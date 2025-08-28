import httpx
import asyncio
import structlog
import numpy as np

from aiolimiter import AsyncLimiter

from google import genai
from google.genai import types

from typing import List, Optional

from io import BytesIO
from PIL import Image

from config import get_settings

logger = structlog.get_logger()
settings = get_settings()

class EmbeddingClient:
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=60.0)

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
        )

        self.video_limiter = AsyncLimiter(20, 90)

    async def embed_image(self, image: Image.Image) -> np.ndarray:
        """Get embedding for single image using SigLIP service"""
        try:
            img_bytes = BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            files = {"file": ("image.png", img_bytes, "image/png")}
            response = await self.http_client.post(
                f"{settings.embedding_service_url}/embed",
                files=files
            )
            response.raise_for_status()

            result = response.json()
            return np.array(result["embedding"])

        except Exception as e:
            logger.error(f"Failed to get image embedding: {e}")
            raise

    async def embed_images_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Get embeddings for multiple images"""
        try:
            files = []
            for i, image in enumerate(images):
                img_bytes = BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                files.append(("files", (f"image_{i}.png", img_bytes, "image/png")))

            response = await self.http_client.post(
                f"{settings.embedding_service_url}/embed-batch",
                files=files
            )
            response.raise_for_status()

            result = response.json()
            return [np.array(emb) for emb in result["embeddings"]]

        except Exception as e:
            logger.error(f"Failed to get batch embeddings: {e}")
            raise

    async def embed_text(self, text: str, max_retries: int = 3, delay: float = 1.0) -> np.ndarray:
        """Get embedding for text using Gemini, with simple retry"""
        for attempt in range(1, max_retries + 1):
            try:
                result = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=settings.gemini_embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=settings.text_embedding_dimension,
                        task_type="SEMANTIC_SIMILARITY"
                    )
                )
                return np.array(result.embeddings[0].values)

            except Exception as e:
                logger.error(f"[Attempt {attempt}/{max_retries}] Failed to get text embedding: {e}")

                if attempt == max_retries:
                    raise

                await asyncio.sleep(delay)

    # async def embed_texts_batch(self, texts: List[str]) -> List[np.ndarray]:
    #     """Get embeddings for multiple texts"""
    #     try:
    #         # Gemini supports batch embedding
    #         results = await asyncio.to_thread(
    #             genai.embed_content,
    #             model=settings.gemini_embedding_model,
    #             content=texts,
    #             task_type="RETRIEVAL_DOCUMENT"
    #         )
    #         return [np.array(emb) for emb in results['embedding']]

    #     except Exception as e:
    #         logger.error(f"Failed to get batch text embeddings: {e}")
    #         raise

    async def analyze_video_with_gemini(self, video_url: str, start_offset: str = None, end_offset: str = None) -> dict:
        """Analyze video using Gemini with retry mechanism"""
        max_retries = 3

        for attempt in range(max_retries):
            async with self.video_limiter:
                try:
                    prompt = """
                    Analyze this video and determine if it's essentially a static image video.

                    Consider:
                    1. Is the main content static (unchanging image)?
                    2. Are there only minor effects like zooming, panning, or text overlays?
                    3. Is there any meaningful motion or scene changes?

                    A video is considered "static" if:
                    - It shows the same image throughout
                    - Only has minor camera movements or effects
                    - Has no meaningful content changes

                    Respond in JSON format:
                    {
                        "is_static": true/false,
                        "confidence": 0.0-1.0,
                        "reasoning": "explanation",
                        "detected_elements": {
                            "has_motion": true/false,
                            "motion_type": "none/minor/significant",
                            "has_scene_changes": true/false,
                            "has_effects": true/false,
                            "effect_types": []
                        }
                    }
                    """

                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=settings.gemini_model,
                        contents=types.Content(
                            parts=[
                                types.Part(
                                    file_data=types.FileData(file_uri=video_url),
                                    video_metadata=types.VideoMetadata(
                                        start_offset='10s' if start_offset is None else start_offset,
                                        end_offset='30s' if end_offset is None else end_offset
                                    )
                                ),
                                types.Part(text=prompt)
                            ]
                        )
                    )

                    # Parse JSON response
                    import json
                    result_text = response.text.strip()
                    if result_text.startswith("```json"):
                        result_text = result_text[7:-3]

                    result = json.loads(result_text)
                    return result

                except Exception as e:
                    if attempt < max_retries - 1:  # Not the last attempt
                        logger.warning(f"Gemini video analysis failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    else:
                        logger.error(f"Gemini video analysis failed after {max_retries} attempts: {e}")
                        raise


# Global instance
embedding_client = EmbeddingClient()
