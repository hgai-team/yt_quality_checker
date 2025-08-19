import asyncio
import json
import subprocess

from typing import List, Dict, Optional
from datetime import datetime, timezone

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, and_
from sqlalchemy.engine import Result

from models.database import Channel, Video
from config import get_settings
from core.database import PostgresEngineManager as PEM

settings = get_settings()

class IncrementalDataFetcher:
    """Fetch only new/updated data from YouTube"""

    async def fetch_channel_videos(
        self,
        channel_id: str,
        force_refresh: bool = False
    ) -> Dict:
        """Fetch videos incrementally"""

        max_results = 7
        # Get existing channel
        async with PEM.get_session() as db:
            stmt = select(Channel).where(Channel.channel_id == channel_id)
            result: Result = await db.exec(stmt)
            channel: Channel = result.first()

            if not channel:
                # Create new channel
                channel = await self._create_channel(db, channel_id)
                max_results = 20

            # # Get existing video IDs
            existing_videos: Result = await db.exec(
                select(Video.video_id).where(Video.channel_id == channel_id)
            )
            existing_ids = {v for v in existing_videos.all()}

            # Fetch video list from YouTube
            all_videos = await self._fetch_youtube_videos(channel_id, max_results)

            # Filter new videos
            new_videos = [
                v for v in all_videos
                if v["video_id"] not in existing_ids or force_refresh
            ]

            # Save new videos
            saved_count = 0
            for video_data in new_videos:
                video = await self._save_video(db, channel_id, video_data)
                if video:
                    saved_count += 1

            # Update channel stats
            channel.last_checked = datetime.now(timezone.utc)
            channel.last_video_count = len(all_videos)

            await db.commit()

            return {
                "channel_id": channel_id,
                "total_videos": len(all_videos),
                "existing_videos": len(existing_ids),
                "new_videos": saved_count,
                "videos": all_videos
            }

    async def _create_channel(self, db: AsyncSession, channel_id: str) -> Channel:
        """Create new channel entry"""
        channel_url = f"https://www.youtube.com/channel/{channel_id}"

        cmd = [
            "yt-dlp",
            "--dump-json",
            "--playlist-end", "1",
            channel_url
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                raise Exception(f"yt-dlp failed: {error_msg}")

            output = stdout.decode().strip()
            if not output:
                raise Exception("Empty output from yt-dlp")

            info = json.loads(output)

            channel_name = (
                info.get("channel") or
                info.get("uploader") or
                info.get("channel_name") or
                info.get("uploader_id") or
                f"Channel_{channel_id}"
            )

            channel = Channel(
                channel_id=channel_id,
                channel_name=channel_name,
                channel_url=channel_url
            )

            db.add(channel)
            await db.flush()
            await db.refresh(channel)
            return channel

        except Exception as e:
            print(f"Error creating channel: {e}")
            raise

    async def _fetch_youtube_videos(
        self,
        channel_id: str,
        max_results: int = 10
    ) -> List[Dict]:
        """Fetch video metadata from YouTube"""
        channel_url = f"https://www.youtube.com/channel/{channel_id}"

        cmd = [
            "yt-dlp",
            "--dump-json",
            "--playlist-items", f"1:{max_results}",
            channel_url
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            videos = []
            for line in stdout.decode().strip().split('\n'):
                if line:
                    video = json.loads(line)
                    videos.append({
                        "video_id": video.get("id"),
                        "title": video.get("title"),
                        "description": video.get("description", ""),
                        "thumbnail_url": video.get("thumbnail"),
                        "upload_date": self._parse_date(video.get("upload_date")),
                        "duration": video.get("duration"),
                        "view_count": video.get("view_count"),
                        "like_count": video.get("like_count"),
                        "tags": video.get("tags", [])
                    })

            return videos

        except Exception as e:
            return []

    async def _save_video(
        self,
        db: AsyncSession,
        channel_id: str,
        video_data: Dict
    ) -> Optional[Video]:
        """Save or update video in database"""
        try:
            # Check if exists
            result = await db.exec(
                select(Video).where(Video.video_id == video_data["video_id"])
            )
            video = result.first()

            if not video:
                video = Video(
                    video_id=video_data["video_id"],
                    channel_id=channel_id
                )
                db.add(video)

            # Update fields
            video.title = video_data.get("title")
            video.description = video_data.get("description")
            video.thumbnail_url = video_data.get("thumbnail_url")
            video.upload_date = video_data.get("upload_date")
            video.duration = video_data.get("duration")
            video.view_count = video_data.get("view_count")
            video.like_count = video_data.get("like_count")
            video.tags = video_data.get("tags", [])
            video.updated_at = datetime.now(timezone.utc)

            await db.flush()
            return video

        except Exception:
            await db.rollback()
            return None


    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse YouTube date string"""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except:
            return None
