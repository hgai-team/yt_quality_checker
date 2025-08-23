import asyncio
from typing import Dict
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime, timezone

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, and_

from models.database import Video
from core.embedding_client import embedding_client
from config import get_settings
import structlog

logger = structlog.get_logger()
settings = get_settings()

class VideoAnalyzer:

    async def analyze_channel_videos(
        self,
        db: AsyncSession,
        channel_id: str,
        sample_size: int = 10
    ) -> Dict:
        """Analyze videos for static content using Gemini"""

        # Get unanalyzed videos
        result = await db.exec(
            select(Video).where(
                and_(
                    Video.channel_id == channel_id
                )
            )
        )
        videos = result.all()

        if not videos:
            return {
                "channel_id": str(channel_id),
                "analyzed": 0,
                "static_count": 0,
                "static_ratio": 0.0
            }

        static_count = 0
        analyzed_count = 0
        results = []

        for video in videos:
            try:
                if video.is_static_video is None:
                    video_url = f"https://www.youtube.com/watch?v={video.video_id}"

                    try:
                        if video.duration < 100:
                            time_segment = video.duration // 2
                        elif video.duration < 1000:
                            time_segment = video.duration // 20
                        elif video.duration < 10000:
                            time_segment = video.duration // 200
                        else:
                            time_segment = video.duration // 2000
                    except Exception as e:
                        logger.error(f"Failed to determine time segment for video {video.video_id}: {e}")
                        time_segment = 10

                    start_offset = f"{int(time_segment)}s"
                    end_offset = f"{int(time_segment * 2)}s"

                    # Analyze with Gemini
                    analysis = await embedding_client.analyze_video_with_gemini(video_url, start_offset, end_offset)


                    is_static = analysis.get("is_static", False)
                    confidence = analysis.get("confidence", 0.0)


                    method = "gemini"

                    # Update database
                    video.is_static_video = is_static
                    video.static_confidence = confidence
                    video.static_analysis_method = method
                    video.static_analysis_details = analysis
                    video.analyzed_at = datetime.now(timezone.utc)

                    if is_static:
                        static_count += 1

                    analyzed_count += 1
                    results.append({
                        "video_id": video.video_id,
                        "title": video.title,
                        "is_static": is_static,
                        "confidence": confidence,
                        "method": method
                    })
                else:
                    # Already analyzed, skip
                    results.append({
                        "video_id": video.video_id,
                        "title": video.title,
                        "is_static": video.is_static_video,
                        "confidence": video.static_confidence,
                        "method": video.static_analysis_method
                    })
                    analyzed_count += 1
                    if video.is_static_video:
                        static_count += 1

            except Exception as e:
                logger.error(f"Failed to analyze video {video.video_id}: {e}")

        await db.commit()

        static_ratio = static_count / analyzed_count if analyzed_count > 0 else 0

        return {
            "channel_id": str(channel_id),
            "analyzed": analyzed_count,
            "static_count": static_count,
            "static_ratio": static_ratio,
            "results": results
        }

    def _analyze_with_opencv(self, video_path: str) -> Dict:
        """Fallback OpenCV analysis for when Gemini fails"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {
                "is_static": False,
                "confidence": 0.0,
                "error": "Cannot open video",
                "method": "opencv"
            }

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames evenly
        sample_count = min(10, frame_count)
        sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)

        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(gray, (160, 120)))

        cap.release()

        if len(frames) < 2:
            return {
                "is_static": False,
                "confidence": 0.0,
                "error": "Not enough frames",
                "method": "opencv"
            }

        # Calculate frame differences
        differences = []
        for i in range(len(frames) - 1):
            diff = cv2.absdiff(frames[i], frames[i + 1])
            differences.append(np.mean(diff))

        max_diff = np.max(differences)

        # Determine if static based on motion threshold
        if max_diff < 2.0:
            confidence = 0.95
            is_static = True
        elif max_diff < 5.0:
            confidence = 0.7
            is_static = True
        else:
            confidence = min(1.0 / max_diff, 0.3)
            is_static = False

        return {
            "is_static": is_static,
            "confidence": confidence,
            "max_difference": float(max_diff),
            "frame_count": len(frames),
            "method": "opencv"
        }

    async def _download_video_segment(
        self,
        video_url: str,
        duration: int = 20
    ) -> str:
        """Download video segment using yt-dlp"""
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "video.mp4")

        cmd = [
            "yt-dlp",
            "-f", "worst[height>=360]",
            "--no-playlist",
            "-o", output_path,
            "--download-sections", f"*0-{duration}",
            "--quiet",
            video_url
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"Failed to download video: {stderr.decode()}")

        return output_path
