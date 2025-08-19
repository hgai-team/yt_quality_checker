import asyncio
import httpx
import hashlib
import structlog
import numpy as np
import uuid
from uuid import uuid4
from PIL import Image
from typing import List, Dict

from io import BytesIO

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, and_
from sqlalchemy.engine import Result

from models.database import Video
from core.embedding_client import embedding_client
from core.qdrant_client import vector_store
from config import get_settings


logger = structlog.get_logger()
settings = get_settings()


class ThumbnailAnalyzer:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def aclose(self):
        await self.client.aclose()

    async def text_to_uuid(self, text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    async def process_channel_thumbnails(
        self,
        db: AsyncSession,
        channel_id: str
    ) -> Dict:
        """Process all thumbnails for a channel using SigLIP embeddings"""

        # Get videos without embeddings
        result = await db.exec(
            select(Video).where(
                and_(
                    Video.channel_id == channel_id,
                    Video.thumbnail_embedding_id.is_(None),
                    Video.thumbnail_url.isnot(None)
                )
            )
        )
        videos = result.all()

        if not videos:
            return {
                "channel_id": str(channel_id),
                "processed": 0,
                "errors": 0,
                "total": 0
            }

        processed = 0
        errors = 0

        # Process in batches for efficiency
        batch_size = 10
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]

            try:
                # Download images
                images = []
                valid_videos = []
                thumbnail_urls = []

                for video in batch:
                    try:
                        image = await self._download_image(video.thumbnail_url)
                        images.append(image)
                        valid_videos.append(video)
                        thumbnail_urls.append(video.thumbnail_url)
                    except Exception as e:
                        logger.error(f"Failed to download thumbnail for {video.video_id}: {e}")
                        errors += 1

                if not images:
                    continue

                # Get embeddings in batch
                if len(images) > 1:
                    embeddings = await embedding_client.embed_images_batch(images)
                else:
                    embedding = await embedding_client.embed_image(images[0])
                    embeddings = [embedding]

                # Store embeddings
                for video, image, thumbnail_url, embedding in zip(valid_videos, images, thumbnail_urls, embeddings):
                    # Generate hash
                    image_hash = self._compute_hash(image)
                    video.thumbnail_hash = image_hash

                    # Store in Qdrant
                    point_id = await self.text_to_uuid(video.video_id)
                    await vector_store.upsert_embedding(
                        "youtube_thumbnails",
                        point_id,
                        embedding,
                        {
                            "video_id": video.video_id,
                            "channel_id": str(video.channel_id),
                            "thumbnail_url": thumbnail_url
                        }
                    )

                    video.thumbnail_embedding_id = point_id
                    processed += 1

            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                errors += len(batch)

        await db.commit()

        return {
            "channel_id": str(channel_id),
            "processed": processed,
            "errors": errors,
            "total": len(videos)
        }

    async def analyze_channel_thumbnails(
        self,
        db: AsyncSession,
        channel_id: str
    ) -> Dict:
        """Find duplicate thumbnails using vector similarity"""

        # Get channel videos with embeddings
        result = await db.exec(
            select(Video).where(
                and_(
                    Video.channel_id == channel_id,
                    Video.thumbnail_embedding_id.isnot(None)
                )
            )
        )
        videos = result.all()

        n = len(videos)
        if n < 2:
            return {
                "thumbnail_duplicates": {
                    "ratio": 0.0,
                    "total_videos": n,
                    "duplicate_pairs": [],
                    "threshold_exceeded": False
                }
            }

        dup_pairs = await vector_store.find_duplicates_in_channel(
            "youtube_thumbnails",
            channel_id,
            threshold=settings.image_similarity_threshold
        )

        total_pairs = n * (n - 1) / 2
        duplicate_ratio = (len(dup_pairs) / total_pairs) if total_pairs else 0.0

        return {
            "thumbnail_duplicates": {
                "ratio": duplicate_ratio,
                "total_videos": n,
                "duplicate_pairs": dup_pairs,
                "threshold_exceeded": duplicate_ratio > settings.duplicate_threshold
            }
        }

    async def _download_image(self, url: str) -> Image.Image:
        """Download image from URL"""
        response = await self.client.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')

    def _compute_hash(self, image: Image.Image) -> str:
        """Compute SHA256 hash of image"""
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        return hashlib.sha256(img_bytes.getvalue()).hexdigest()
