from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SearchRequest, Filter, FieldCondition, MatchValue, HasIdCondition
)
from typing import List, Dict, Optional
import numpy as np
from config import get_settings

settings = get_settings()

def _preview_text(s: Optional[str], maxlen: int = 140) -> Optional[str]:
    if not s:
        return s
    s = s.strip().replace("\n", " ")
    return (s[:maxlen] + "…") if len(s) > maxlen else s

class VectorStore:
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collections = [
            {
                "name": "youtube_thumbnails",
                "embed_dim": settings.image_embedding_dimension
            },
            {
                "name": "youtube_titles",
                "embed_dim": settings.text_embedding_dimension
            },
            {
                "name": "youtube_descriptions",
                "embed_dim": settings.text_embedding_dimension
            }
        ]

    async def count(self, collection_name: str) -> int:
        return (
            await self.client.count(
                collection_name=collection_name, exact=True
            )
        ).count

    async def init_collections(self):
        """Initialize Qdrant collections"""
        for collection in self.collections:
            if not await self.client.collection_exists(collection["name"]):
                await self.client.create_collection(
                    collection_name=collection["name"],
                    vectors_config=VectorParams(
                        size=collection["embed_dim"],
                        distance=Distance.COSINE
                    )
                )

    async def upsert_embedding(
        self,
        collection_name: str,
        point_id: str,
        embedding: np.ndarray,
        metadata: Dict
    ):
        """Store embedding in Qdrant"""

        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=metadata
        )

        await self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        return point_id

    async def search_similar(
        self,
        collection_name: str,
        point: PointStruct,
        channel_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.8,
        allowed_ids: Optional[set[str]] = None
    ) -> List[Dict]:
        """Search for similar embeddings"""
        
        must_conds = [
            FieldCondition(
                key="channel_id",
                match=MatchValue(value=channel_id)
            )
        ]
        if allowed_ids:
            must_conds.append(
                HasIdCondition(has_id=list(allowed_ids))
            )

        filter_conditions = None
        if channel_id:
            filter_conditions = Filter(
                must=must_conds,
                must_not=[
                    HasIdCondition(
                        has_id=[point.id]
                    )
                ]
            )

        results = (
                await self.client.query_points(
                collection_name=collection_name,
                query=point.vector,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
        ).points

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]

    async def find_duplicates_in_channel(
        self,
        collection_name: str,
        channel_id: str,
        threshold: float = 0.92,
        extra_key: str = "thumbnail_url",
        text_preview: bool = False,
        allowed_ids: Optional[set[str]] = None
    ) -> List[tuple]:
        """Find all duplicate pairs within a channel"""
        must_conds = [
            FieldCondition(
                key="channel_id",
                match=MatchValue(value=channel_id)
            )
        ]
        
        if allowed_ids:
            must_conds.append(
                HasIdCondition(has_id=list(allowed_ids))
            )
        
        total = await self.count(collection_name)

        # Get all points for channel
        points, _ = await self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=must_conds
            ),
            limit=total,
            with_payload=True,
            with_vectors=True
        )

        duplicates: List[dict] = []
        seen = set()
        for point in points:
            p_pl = point.payload or {}
            p_vid = p_pl.get("video_id")
            if not p_vid:
                continue
            
            _allowed_ids = None
            if allowed_ids:
                _allowed_ids = allowed_ids.copy()
                _allowed_ids.discard(point.id)

            # lấy extra theo key
            p_extra = p_pl.get(extra_key)
            if text_preview and p_extra:
                p_extra = _preview_text(p_extra)

            similar = await self.search_similar(
                collection_name=collection_name,
                point=point,
                channel_id=channel_id,
                limit=20,
                score_threshold=threshold,
                allowed_ids=_allowed_ids
            )

            for match in similar:
                m_pl = match["payload"] or {}
                m_vid = m_pl.get("video_id")
                if not m_vid or m_vid == p_vid:
                    continue

                key = tuple(sorted((p_vid, m_vid)))
                if key in seen:
                    continue
                seen.add(key)

                m_extra = m_pl.get(extra_key)
                if text_preview and m_extra:
                    m_extra = _preview_text(m_extra)

                duplicates.append({
                    "video_id_1": p_vid,
                    "video_id_2": m_vid,
                    # đổi tên field extra theo key để không phá UI cũ
                    "thumbnail_1": p_extra if extra_key == "thumbnail_url" else None,
                    "thumbnail_2": m_extra if extra_key == "thumbnail_url" else None,
                    "text_1": p_extra if extra_key == "text" else None,
                    "text_2": m_extra if extra_key == "text" else None,
                    "score": float(match["score"]),
                    "watch_url_1": f"https://www.youtube.com/watch?v={p_vid}",
                    "watch_url_2": f"https://www.youtube.com/watch?v={m_vid}",
                })

        return duplicates

vector_store = VectorStore()
