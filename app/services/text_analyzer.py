import structlog
import asyncio
import uuid
import numpy as np

from typing import List, Dict

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, and_
from sqlalchemy.engine import Result

from models.database import Video
from core.embedding_client import embedding_client
from core.qdrant_client import vector_store
from config import get_settings
from core.database import PostgresEngineManager as PEM



logger = structlog.get_logger()
settings = get_settings()


class TextAnalyzer:
    def __init__(self):
        pass

    async def text_to_uuid(self, text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    async def process_channel_text(
        self,
        channel_id: str
    ) -> Dict:
        """Process text embeddings for channel"""

        async def run_titles():
            async with PEM.get_session() as db:
                return await self._process_text_field(
                    db, channel_id, "title", "youtube_titles"
                )

        async def run_descriptions():
            async with PEM.get_session() as db:
                return await self._process_text_field(
                    db, channel_id, "description", "youtube_descriptions"
                )

        title_result, desc_result = await asyncio.gather(
            run_titles(),
            run_descriptions()
        )

        return {
            "channel_id": str(channel_id),
            "titles_processed": title_result["processed"],
            "descriptions_processed": desc_result["processed"],
            "errors": title_result["errors"] + desc_result["errors"]
        }

    async def _process_text_field(
        self,
        db: AsyncSession,
        channel_id: str,
        field_name: str,
        collection_name: str
    ) -> Dict:
        """Process embeddings for a text field"""

        # Get videos without embeddings for this field
        embedding_field = f"{field_name}_embedding_id"

        query = select(Video).where(
            and_(
                Video.channel_id == channel_id,
                getattr(Video, embedding_field).is_(None),
                getattr(Video, field_name).isnot(None)
            )
        )

        result = await db.exec(query)
        videos = result.all()

        processed = 0
        errors = 0

        # Batch process for efficiency
        batch_size = settings.batch_size
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]
            texts = [getattr(v, field_name) for v in batch]

            try:
                # Generate embeddings in batch
                tasks = [
                    embedding_client.embed_text(text) for text in texts
                ]

                embeddings = await asyncio.gather(*tasks, return_exceptions=True)

                # Store each embedding
                for video, embedding in zip(batch, embeddings):
                    if isinstance(embedding, Exception):
                        logger.error(f"Embedding failed for video {video.video_id}: {embedding}")
                        errors += 1
                        continue

                    try:
                        point_id = await self.text_to_uuid(video.video_id)
                        await vector_store.upsert_embedding(
                            collection_name,
                            point_id,
                            embedding,
                            {
                                "video_id": video.video_id,
                                "channel_id": str(video.channel_id),
                                "text": getattr(video, field_name)
                            }
                        )
                        setattr(video, embedding_field, point_id)
                        processed += 1

                    except Exception as e:
                        logger.error(f"Failed to upsert embedding for video {video.video_id}: {e}")
                        errors += 1

            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                errors += len(batch)

        await db.commit()

        return {
            "processed": processed,
            "errors": errors,
            "total": len(videos)
        }

    async def analyze_channel_text(
        self,
        db: AsyncSession,
        channel_id: str
    ) -> Dict:
        """Analyze text duplicates"""

        title_task = vector_store.find_duplicates_in_channel(
            "youtube_titles",
            str(channel_id),
            threshold=settings.text_similarity_threshold,
            extra_key="text",
            text_preview=True
        )
        desc_task = vector_store.find_duplicates_in_channel(
            "youtube_descriptions",
            str(channel_id),
            threshold=settings.text_similarity_threshold,
            extra_key="text",
            text_preview=True
        )

        title_duplicates, desc_duplicates = await asyncio.gather(title_task, desc_task)

        result_titles = await db.exec(
            select(Video).where(
                and_(Video.channel_id == channel_id, Video.title_embedding_id.isnot(None))
            )
        )
        videos_titles = result_titles.all()
        n_titles = len(videos_titles)
        total_pairs_titles = n_titles * (n_titles - 1) / 2 if n_titles else 0
        ratio_titles = (len(title_duplicates) / total_pairs_titles) if total_pairs_titles else 0.0

        result_desc = await db.exec(
            select(Video).where(
                and_(Video.channel_id == channel_id, Video.description_embedding_id.isnot(None))
            )
        )
        videos_desc = result_desc.all()
        n_desc = len(videos_desc)
        total_pairs_desc = n_desc * (n_desc - 1) / 2 if n_desc else 0
        ratio_desc = (len(desc_duplicates) / total_pairs_desc) if total_pairs_desc else 0.0

        return {
            "title_duplicates": {
                "ratio": ratio_titles,
                "total_videos": n_titles,
                "duplicate_pairs": title_duplicates,
                "threshold_exceeded": ratio_titles > settings.duplicate_threshold
            },
            "description_duplicates": {
                "ratio": ratio_desc,
                "total_videos": n_desc,
                "duplicate_pairs": desc_duplicates,
                "threshold_exceeded": ratio_desc > settings.duplicate_threshold
            }
        }
