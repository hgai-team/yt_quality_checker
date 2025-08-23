from datetime import datetime, timedelta, timezone
from typing import Dict

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, func

from models.database import Channel, Video
from config import get_settings

import structlog

logger = structlog.get_logger()
settings = get_settings()

class ChannelMonitor:
    
    async def check_activity(
        self,
        db: AsyncSession,
        channel_id: str
    ) -> Dict:
        """Check channel activity status"""
        
        # Get latest video upload date
        result = await db.exec(
            select(func.max(Video.upload_date))
            .where(Video.channel_id == channel_id)
        )
        last_upload = result.first()
        
        if last_upload:
            days_inactive = (datetime.now() - last_upload).days
            is_active = days_inactive <= settings.inactive_days
        else:
            days_inactive = 999
            is_active = False
        
        # Update channel status
        channel_result = await db.exec(
            select(Channel).where(Channel.channel_id == channel_id)
        )
        channel = channel_result.first()
        if channel:
            channel.is_active = is_active
        
        return {
            "is_active": is_active,
            "days_inactive": days_inactive,
            "last_upload": last_upload.isoformat() if last_upload else None,
            "threshold_days": settings.inactive_days
        }