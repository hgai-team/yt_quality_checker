from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, func

from core.database import PostgresEngineManager as PEM
from models.database import Channel, Video, AnalysisResult
from services.data_fetcher import IncrementalDataFetcher
from services.thumbnail_analyzer import ThumbnailAnalyzer
from services.text_analyzer import TextAnalyzer
# from services.video_analyzer import AdvancedVideoAnalyzer
# from services.channel_monitor import ChannelMonitor
# from services.business_rules import BusinessRulesEngine

from config import get_settings
import structlog
import asyncio

router = APIRouter(prefix="/channels", tags=["channels"])
logger = structlog.get_logger()
settings = get_settings()


class ChannelRequest(BaseModel):
    channel_id: str
    project_name: Optional[str] = None
    force_refresh: bool = False


class AnalysisRequest(BaseModel):
    channel_id: str
    analysis_types: List[str] = ["all"]
    sample_size: int = 10


@router.post("/analyze")
async def analyze_channel(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks = None
):
    """Analyze channel thumbnails and text"""
    thumb_analyzer = ThumbnailAnalyzer()
    text_analyzer = TextAnalyzer()

    async def run_analyze_thumbnails():
        async with PEM.get_session() as db:
            return await thumb_analyzer.analyze_channel_thumbnails(db, request.channel_id)

    async def run_analyze_text():
        async with PEM.get_session() as db:
            return await text_analyzer.analyze_channel_text(db, request.channel_id)

    # chạy song song, mỗi task có session riêng
    thumb_result, text_result = await asyncio.gather(
        run_analyze_thumbnails(),
        run_analyze_text()
    )

    return {
        "status": "success",
        "channel_id": request.channel_id,
        "thumbnail_analysis": thumb_result,
        "text_analysis": text_result
    }


@router.post("/sync")
async def sync_channel(
    request: ChannelRequest,
    background_tasks: BackgroundTasks = None
):
    """Sync channel data incrementally"""

    fetcher = IncrementalDataFetcher()

    # Fetch new videos
    result = await fetcher.fetch_channel_videos(
        request.channel_id,
        request.force_refresh
    )

    # Update project if provided
    if request.project_name:
        async with PEM.get_session() as db:
            channel = await db.exec(
                select(Channel).where(Channel.channel_id == request.channel_id)
            )
            channel = channel.first()
            channel.project_name = request.project_name

    # Trigger background processing for embeddings
    if background_tasks and result["new_videos"] > 0:
        background_tasks.add_task(
            process_channel_embeddings,
            request.channel_id
        )

    return {
        "status": "success",
        "channel_id": request.channel_id,
        "total_videos": result["total_videos"],
        "new_videos": result["new_videos"],
        "existing_videos": result["existing_videos"]
    }

async def process_channel_embeddings(channel_id: str):
    """Background task to process embeddings"""
    thumb_analyzer = ThumbnailAnalyzer()
    text_analyzer  = TextAnalyzer()

    async def run_thumbnails():
        async with PEM.get_session() as db:
            return await thumb_analyzer.process_channel_thumbnails(db, channel_id)

    async def run_text():
        return await text_analyzer.process_channel_text(channel_id)

    # chạy song song, mỗi task có session riêng
    thumb_result, text_result = await asyncio.gather(
        run_thumbnails(),
        run_text()
    )

    # (tùy chọn) đóng http client nếu bạn thêm aclose()
    await thumb_analyzer.aclose()

    return {
        "thumbnails": thumb_result,
        "text": text_result
    }
