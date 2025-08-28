import re
from typing import List, Optional, Dict
from datetime import datetime, timezone

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, and_
from models.database import Channel, Video
from config import get_settings
import structlog

logger = structlog.get_logger()
settings = get_settings()


class BusinessRulesEngine:

    def __init__(self):
        self.project_keywords = settings.project_keywords

    async def validate_channel(
        self,
        db: AsyncSession,
        channel_id: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> Dict:
        """Validate business rules for channel"""

        # Get channel
        result = await db.exec(
            select(Channel).where(Channel.channel_id == channel_id)
        )
        channel = result.first()

        if not channel:
            return {"error": "Channel not found"}

        conds = [Video.channel_id == channel_id]
        if from_date:
            if from_date.tzinfo is not None:
                from_date = from_date.replace(tzinfo=None)
            conds.append(Video.upload_date >= from_date)
        if to_date:
            if to_date.tzinfo is not None:
                to_date = to_date.replace(tzinfo=None)
            conds.append(Video.upload_date <= to_date)

        # Get all videos
        videos_result = await db.exec(
            select(Video).where(and_(*conds))
        )
        videos = videos_result.all()

        # Check MBT usage
        mbt_result = self._check_mbt_usage(videos)

        # Check project compliance
        project_result = self._check_project_compliance(
            channel.project_name,
            videos
        )

        return {
            "channel_id": str(channel_id),
            "uses_mbt": mbt_result["uses_mbt"],
            "mbt_ratio": mbt_result["ratio"],
            "mbt_videos": mbt_result["count"],
            "non_mbt_videos": mbt_result["non_mbt_videos"],
            "project_compliant": project_result["compliant"],
            "project_compliance_ratio": project_result["ratio"],
            "non_compliant_videos": project_result["non_compliant"]  # Trả về toàn bộ
        }

    def _check_mbt_usage(self, videos: List[Video]) -> Dict:
        """Check MBT usage in videos based on HGED* code in description only."""
        if not videos:
            return {"uses_mbt": False, "ratio": 0.0, "count": 0, "non_mbt_videos": []}

        # "Chuỗi liền" bắt đầu bằng HGED, cho phép chữ, số, và _ tiếp theo.
        # Ví dụ khớp: HGED, HGED_ABC, HGED123_ABC
        # Dùng \b để tránh ăn vào ký tự chữ/số liền kề ngoài chuỗi.
        hged_pattern = re.compile(r'\bHGED[A-Za-z0-9_]+\b')

        mbt_count = 0
        non_mbt_videos = []

        for video in videos:
            desc = (video.description or "")
            # Kiểm tra CHỈ trong description
            if hged_pattern.search(desc):
                mbt_count += 1
            else:
                non_mbt_videos.append({
                    "video_id": video.video_id,
                    "title": video.title
                })

        return {
            "uses_mbt": mbt_count > 0,
            "ratio": mbt_count / len(videos),
            "count": mbt_count,
            "non_mbt_videos": non_mbt_videos
        }


    def _check_project_compliance(
        self,
        project_name: str,
        videos: List[Video]
    ) -> Dict:
        """Check if videos comply with project"""

        if not project_name or not videos:
            return {"compliant": True, "ratio": 1.0, "non_compliant": []}

        if project_name.lower() not in self.project_keywords:
            return {"compliant": True, "ratio": 1.0, "non_compliant": []}

        keywords = self.project_keywords[project_name.lower()]
        compliant_count = 0
        non_compliant = []

        for video in videos:
            text = f"{video.title or ''} {video.description or ''}".lower()

            is_compliant = any(keyword in text for keyword in keywords)

            if is_compliant:
                compliant_count += 1
            else:
                non_compliant.append({
                    "video_id": video.video_id,
                    "title": video.title
                })

        ratio = compliant_count / len(videos)

        return {
            "compliant": ratio >= 0.6,
            "ratio": ratio,
            "non_compliant": non_compliant
        }
