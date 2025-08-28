from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, func, and_

from core.database import PostgresEngineManager as PEM
from models.database import Channel, Video, AnalysisResult
from services.data_fetcher import IncrementalDataFetcher
from services.thumbnail_analyzer import ThumbnailAnalyzer
from services.text_analyzer import TextAnalyzer
from services.video_analyzer import VideoAnalyzer
from services.channel_monitor import ChannelMonitor
from services.business_rules import BusinessRulesEngine

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

    # New optional fields from Excel
    networks: Optional[str] = None
    department: Optional[str] = None
    employee_name: Optional[str] = None
    employee_id: Optional[str] = None

class AnalysisRequest(BaseModel):
    channel_id: str
    analysis_types: List[str] = ["all"]
    sample_size: int = 10
    # bộ lọc thời gian (tuỳ chọn). None = toàn bộ video
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

    # Custom thresholds - khi có thì không dùng cache
    image_similarity_threshold: Optional[float] = None
    text_similarity_threshold: Optional[float] = None
    duplicate_threshold: Optional[float] = None

def _same_range(a: Optional[datetime], b: Optional[datetime]) -> bool:
    return (a is None and b is None) or (a and b and a == b)

def _has_custom_params(request: AnalysisRequest) -> bool:
    """Check if request has custom thresholds or date filters"""
    return any([
        request.image_similarity_threshold is not None,
        request.text_similarity_threshold is not None,
        request.duplicate_threshold is not None,
        request.from_date is not None,
        request.to_date is not None
    ])

def _get_effective_thresholds(request: AnalysisRequest) -> dict:
    """Get effective threshold values (custom or default)"""
    return {
        "image_similarity_threshold": request.image_similarity_threshold or settings.image_similarity_threshold,
        "text_similarity_threshold": request.text_similarity_threshold or settings.text_similarity_threshold,
        "duplicate_threshold": request.duplicate_threshold or settings.duplicate_threshold
    }

def _compute_severity(d: dict, thresholds: dict) -> float:
    # d là payload tổng hợp (thumb/text/video/monitor/business)
    dup_thres = thresholds["duplicate_threshold"]
    static_thres = getattr(settings, "static_video_threshold", 0.5)  # fallback
    score = 0.0

    # Duplicates
    for key in ("thumbnail_analysis",):
        r = d.get(key, {}).get("thumbnail_duplicates", {}).get("ratio", 0.0)
        if r > dup_thres: score += min(25.0 * (r - dup_thres) / max(1e-6, 1 - dup_thres), 25.0)

    r_title = d.get("text_analysis", {}).get("title_duplicates", {}).get("ratio", 0.0)
    if r_title > dup_thres: score += min(20.0 * (r_title - dup_thres) / max(1e-6, 1 - dup_thres), 20.0)
    r_desc  = d.get("text_analysis", {}).get("description_duplicates", {}).get("ratio", 0.0)
    if r_desc  > dup_thres: score += min(15.0 * (r_desc  - dup_thres) / max(1e-6, 1 - dup_thres), 15.0)

    # Static
    static_ratio = d.get("video_analysis", {}).get("static_ratio", 0.0)
    if static_ratio > static_thres:
        score += min(25.0 * (static_ratio - static_thres) / max(1e-6, 1 - static_thres), 25.0)

    # Inactive
    days_inactive = d.get("channel_monitor", {}).get("days_inactive", 0)
    if days_inactive > settings.inactive_days:
        score += 10.0

    # Project compliance
    pr_ok = d.get("business_rules", {}).get("project_compliant", True)
    if not pr_ok:
        ratio = d.get("business_rules", {}).get("project_compliance_ratio", 0.0)
        score += min(25.0 * (1.0 - ratio), 25.0)

    return float(max(0.0, min(100.0, score)))

async def _save_snapshot(db: AsyncSession, channel_id: str, payload: dict,
                         source_last_checked: Optional[datetime],
                         f_from: Optional[datetime], f_to: Optional[datetime],
                         thresholds: dict) -> AnalysisResult:
    ar = AnalysisResult(
        channel_id=channel_id,
        # Duplicates
        thumbnail_duplicate_ratio = payload["thumbnail_analysis"]["thumbnail_duplicates"]["ratio"],
        thumbnail_duplicate_pairs = payload["thumbnail_analysis"]["thumbnail_duplicates"]["duplicate_pairs"],
        title_duplicate_ratio      = payload["text_analysis"]["title_duplicates"]["ratio"],
        title_duplicate_pairs      = payload["text_analysis"]["title_duplicates"]["duplicate_pairs"],
        description_duplicate_ratio= payload["text_analysis"]["description_duplicates"]["ratio"],
        description_duplicate_pairs= payload["text_analysis"]["description_duplicates"]["duplicate_pairs"],
        # Video analysis
        static_video_ratio = payload["video_analysis"]["static_ratio"],
        static_video_ids   = [r["video_id"] for r in payload["video_analysis"].get("results", []) if r.get("is_static")],
        # Activity
        days_inactive      = payload["channel_monitor"]["days_inactive"],
        last_upload_date   = (payload["channel_monitor"]["last_upload"]
                              and datetime.fromisoformat(payload["channel_monitor"]["last_upload"])),
        # Business rules
        uses_mbt                 = payload["business_rules"]["uses_mbt"],
        mbt_usage_ratio          = payload["business_rules"]["mbt_ratio"],
        non_mbt_videos           = payload["business_rules"]["non_mbt_videos"],
        project_compliance       = payload["business_rules"]["project_compliant"],
        project_compliance_ratio = payload["business_rules"]["project_compliance_ratio"],
        non_compliant_videos     = payload["business_rules"]["non_compliant_videos"],

        # Summary
        total_issues = None,  # tuỳ bạn muốn tính
        issues       = None,  # hoặc build list text, có thể bổ sung sau
        severity_score = _compute_severity(payload, thresholds),

        # Snapshot meta
        source_last_checked = source_last_checked,
        filter_from = f_from,
        filter_to   = f_to,
    )
    db.add(ar)
    await db.commit()
    await db.refresh(ar)
    return ar

async def _build_video_issues_list(
    db: AsyncSession,
    channel_id: str,
    payload: dict,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[dict]:
    """Build list of all videos with their specific issues"""
    conds = [Video.channel_id == channel_id]
    if from_date:
        if from_date.tzinfo is not None:
            from_date = from_date.replace(tzinfo=None)
        conds.append(Video.upload_date >= from_date)
    if to_date:
        if to_date.tzinfo is not None:
            to_date = to_date.replace(tzinfo=None)
        conds.append(Video.upload_date <= to_date)

    result = await db.exec(
        select(Video).where(and_(*conds)).order_by(Video.upload_date.desc())
    )
    videos = result.all()

    # Get duplicate pairs data
    thumbnail_pairs = payload.get("thumbnail_analysis", {}).get("thumbnail_duplicates", {}).get("duplicate_pairs", [])
    title_pairs = payload.get("text_analysis", {}).get("title_duplicates", {}).get("duplicate_pairs", [])
    desc_pairs = payload.get("text_analysis", {}).get("description_duplicates", {}).get("duplicate_pairs", [])
    static_video_ids = set(payload.get("video_analysis", {}).get("static_video_ids", []))
    non_compliant_videos = {v["video_id"]: v for v in payload.get("business_rules", {}).get("non_compliant_videos", [])}
    non_mbt_videos = {v["video_id"]: v for v in payload.get("business_rules", {}).get("non_mbt_videos", [])}

    # Build lookup dicts for duplicates with details
    thumbnail_duplicates = {}
    for pair in thumbnail_pairs:
        vid1, vid2 = pair["video_id_1"], pair["video_id_2"]
        if vid1 not in thumbnail_duplicates:
            thumbnail_duplicates[vid1] = []
        if vid2 not in thumbnail_duplicates:
            thumbnail_duplicates[vid2] = []
        thumbnail_duplicates[vid1].append({
            "duplicate_with": vid2,
            "score": pair["score"],
            "thumbnail_url": pair.get("thumbnail_2")
        })
        thumbnail_duplicates[vid2].append({
            "duplicate_with": vid1,
            "score": pair["score"],
            "thumbnail_url": pair.get("thumbnail_1")
        })

    title_duplicates = {}
    for pair in title_pairs:
        vid1, vid2 = pair["video_id_1"], pair["video_id_2"]
        if vid1 not in title_duplicates:
            title_duplicates[vid1] = []
        if vid2 not in title_duplicates:
            title_duplicates[vid2] = []
        title_duplicates[vid1].append({
            "duplicate_with": vid2,
            "score": pair["score"],
            "text": pair.get("text_2")
        })
        title_duplicates[vid2].append({
            "duplicate_with": vid1,
            "score": pair["score"],
            "text": pair.get("text_1")
        })

    desc_duplicates = {}
    for pair in desc_pairs:
        vid1, vid2 = pair["video_id_1"], pair["video_id_2"]
        if vid1 not in desc_duplicates:
            desc_duplicates[vid1] = []
        if vid2 not in desc_duplicates:
            desc_duplicates[vid2] = []
        desc_duplicates[vid1].append({
            "duplicate_with": vid2,
            "score": pair["score"],
            "text": pair.get("text_2")
        })
        desc_duplicates[vid2].append({
            "duplicate_with": vid1,
            "score": pair["score"],
            "text": pair.get("text_1")
        })

    video_list = []
    for video in videos:
        issues = {}

        if video.video_id in thumbnail_duplicates:
            issues["thumbnail_duplicate"] = {
                "count": len(thumbnail_duplicates[video.video_id]),
                "duplicates": thumbnail_duplicates[video.video_id]
            }

        if video.video_id in title_duplicates:
            issues["title_duplicate"] = {
                "count": len(title_duplicates[video.video_id]),
                "duplicates": title_duplicates[video.video_id]
            }

        if video.video_id in desc_duplicates:
            issues["description_duplicate"] = {
                "count": len(desc_duplicates[video.video_id]),
                "duplicates": desc_duplicates[video.video_id]
            }

        if video.video_id in static_video_ids:
            issues["static_video"] = {
                "confidence": video.static_confidence,
                "method": video.static_analysis_method,
                "details": video.static_analysis_details
            }

        if video.video_id in non_compliant_videos:
            issues["project_non_compliant"] = {
                "project_name": payload.get("project_name"),
                "title": non_compliant_videos[video.video_id].get("title")
            }

        if video.video_id in non_mbt_videos:
            issues["not_using_mbt"] = {
                "title": non_mbt_videos[video.video_id].get("title"),
                "reason": "No HGED code found in description"
            }

        video_list.append({
            "video_id": video.video_id,
            "title": video.title,
            "upload_date": video.upload_date.isoformat() if video.upload_date else None,
            "thumbnail_url": video.thumbnail_url,
            "watch_url": f"https://www.youtube.com/watch?v={video.video_id}",
            "issues": issues,
            "has_issues": len(issues) > 0,
            "issue_count": len(issues)
        })

    return video_list

def _format_analysis_response(
    channel: Channel,
    payload: dict,
    thresholds: dict,
    source: str = "fresh",
    analysis_date: str = None,
    severity_score: float = None,
    snapshot_id: str = None,
    video_list: List[dict] = None
) -> dict:
    """Format consistent analysis response"""

    base_response = {
        "status": "success",
        "channel_id": payload["channel_id"],
        "channel_name": payload["channel_name"],
        "project_name": payload["project_name"],
        "networks": payload["networks"],
        "department": payload["department"],
        "employee_name": payload["employee_name"],
        "source": source,
        "analysis_date": analysis_date,
        "severity_score": severity_score,
        "thresholds_used": thresholds,

        "thumbnail_analysis": payload["thumbnail_analysis"],
        "text_analysis": payload["text_analysis"],
        "video_analysis": {
            "static_ratio": payload["video_analysis"]["static_ratio"],
            "static_video_ids": [r["video_id"] for r in payload["video_analysis"].get("results", []) if r.get("is_static")] if "results" in payload["video_analysis"] else payload["video_analysis"].get("static_video_ids", [])
        },
        "channel_monitor": payload["channel_monitor"],
        "business_rules": payload["business_rules"]
    }

    if snapshot_id:
        base_response["snapshot_id"] = snapshot_id

    if video_list:
        base_response["videos"] = video_list
        base_response["total_videos"] = len(video_list)
        base_response["videos_with_issues"] = len([v for v in video_list if v["has_issues"]])

    return base_response

def _collect_static_lists(rows: List[Channel]) -> dict:
    projects = []
    departments = []
    networks = []
    for ch in rows:
        if ch.project_name and ch.project_name not in projects:
            projects.append(ch.project_name)
        if getattr(ch, "department", None) and ch.department not in departments:
            departments.append(ch.department)
        if getattr(ch, "networks", None) and ch.networks not in networks:
            networks.append(ch.networks)
    return {
        "all_projects": projects,
        "all_departments": departments,
        "all_networks": networks,
    }

@router.post("/analyze")
async def analyze_channel(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks = None
):
    """Analyze channel thumbnails and text"""
    thresholds = _get_effective_thresholds(request)
    use_cache = not _has_custom_params(request)

    thumb_analyzer = ThumbnailAnalyzer()
    text_analyzer = TextAnalyzer()
    channel_monitor = ChannelMonitor()
    video_analyzer = VideoAnalyzer()
    business_rules = BusinessRulesEngine()

    async with PEM.get_session() as db:
        ch_res = await db.exec(select(Channel).where(Channel.channel_id == request.channel_id))
        channel = ch_res.first()
        if not channel:
            raise HTTPException(status_code=404, detail="Channel not found")

        cached = None
        if use_cache:
            snap_q = (
                select(AnalysisResult)
                .where(
                    AnalysisResult.channel_id == request.channel_id,
                )
                .order_by(AnalysisResult.analysis_date.desc())
                .limit(10)
            )
            snap_res = await db.exec(snap_q)
            snapshots = snap_res.all()

            for s in snapshots:
                if _same_range(s.filter_from, request.from_date) and _same_range(s.filter_to, request.to_date):
                    if s.source_last_checked == channel.last_checked:
                        cached = s
                        break

    if cached and use_cache:
        # Build video list for cached results
        async with PEM.get_session() as db:
            cached_payload = {
                "channel_id": request.channel_id,
                "channel_name": channel.channel_name,
                "project_name": channel.project_name,
                "networks": channel.networks,
                "department": channel.department,
                "employee_name": channel.employee_name,
                "thumbnail_analysis": {
                    "thumbnail_duplicates": {
                        "ratio": cached.thumbnail_duplicate_ratio,
                        "duplicate_pairs": cached.thumbnail_duplicate_pairs,
                    }
                },
                "text_analysis": {
                    "title_duplicates": {
                        "ratio": cached.title_duplicate_ratio,
                        "duplicate_pairs": cached.title_duplicate_pairs,
                    },
                    "description_duplicates": {
                        "ratio": cached.description_duplicate_ratio,
                        "duplicate_pairs": cached.description_duplicate_pairs,
                    }
                },
                "video_analysis": {
                    "static_ratio": cached.static_video_ratio,
                    "static_video_ids": cached.static_video_ids,
                },
                "channel_monitor": {
                    "days_inactive": cached.days_inactive,
                    "last_upload": cached.last_upload_date and cached.last_upload_date.isoformat(),
                    "threshold_days": settings.inactive_days,
                },
                "business_rules": {
                    "uses_mbt": cached.uses_mbt,
                    "mbt_ratio": cached.mbt_usage_ratio,
                    "non_mbt_videos": cached.non_mbt_videos,
                    "project_compliant": cached.project_compliance,
                    "project_compliance_ratio": cached.project_compliance_ratio,
                    "non_compliant_videos": cached.non_compliant_videos,
                }
            }

            video_list = await _build_video_issues_list(
                db, request.channel_id, cached_payload,
                request.from_date, request.to_date
            )

            return _format_analysis_response(
                channel=channel,
                payload=cached_payload,
                thresholds={
                    "image_similarity_threshold": settings.image_similarity_threshold,
                    "text_similarity_threshold": settings.text_similarity_threshold,
                    "duplicate_threshold": settings.duplicate_threshold
                },
                source="cache",
                analysis_date=cached.analysis_date.isoformat(),
                severity_score=cached.severity_score,
                snapshot_id=str(cached.id),
                video_list=video_list
            )

    async def run_analyze_thumbnails():
        async with PEM.get_session() as db:
            return await thumb_analyzer.analyze_channel_thumbnails(
                db, request.channel_id, request.from_date, request.to_date,
                image_similarity_threshold=thresholds["image_similarity_threshold"], duplicate_threshold=thresholds["duplicate_threshold"]
            )

    async def run_analyze_text():
        async with PEM.get_session() as db:
            return await text_analyzer.analyze_channel_text(
                db, request.channel_id, request.from_date, request.to_date,
                text_similarity_threshold=thresholds["text_similarity_threshold"], duplicate_threshold=thresholds["duplicate_threshold"]
            )

    async def run_channel_monitor():
        async with PEM.get_session() as db:
            return await channel_monitor.check_activity(db, request.channel_id)

    async def run_video_analyzer():
        async with PEM.get_session() as db:
            return await video_analyzer.analyze_channel_videos(db, request.channel_id, request.from_date, request.to_date)

    async def run_business_rules():
        async with PEM.get_session() as db:
            return await business_rules.validate_channel(db, request.channel_id, request.from_date, request.to_date)

    # chạy song song, mỗi task có session riêng
    thumb_result, text_result, monitor_result, video_result, rules_result   = await asyncio.gather(
        run_analyze_thumbnails(),
        run_analyze_text(),
        run_channel_monitor(),
        run_video_analyzer(),
        run_business_rules()
    )

    payload = {
        "channel_id": request.channel_id,
        "channel_name": channel.channel_name,
        "project_name": channel.project_name,
        "networks": channel.networks,
        "department": channel.department,
        "employee_name": channel.employee_name,
        "thumbnail_analysis": thumb_result,
        "text_analysis": text_result,
        "channel_monitor": monitor_result,
        "video_analysis": video_result,
        "business_rules": rules_result
    }

    # Only save snapshot if using default thresholds
    snapshot = None
    if use_cache:
        async with PEM.get_session() as db:
            ch_res = await db.exec(select(Channel).where(Channel.channel_id == request.channel_id))
            channel = ch_res.first()
            snapshot = await _save_snapshot(
                db, request.channel_id, payload,
                source_last_checked=channel.last_checked,
                f_from=request.from_date, f_to=request.to_date,
                thresholds=thresholds
            )

    # Format video analysis for consistency
    payload["video_analysis"] = {
        "static_ratio": video_result["static_ratio"],
        "static_video_ids": [r["video_id"] for r in video_result.get("results", []) if r.get("is_static")]
    }

    # Build video list with issues
    async with PEM.get_session() as db:
        video_list = await _build_video_issues_list(
            db, request.channel_id, payload,
            request.from_date, request.to_date
        )

    return _format_analysis_response(
        channel=channel,
        payload=payload,
        thresholds=thresholds,
        source="fresh",
        analysis_date=snapshot.analysis_date.isoformat() if snapshot else datetime.now().isoformat(),
        severity_score=_compute_severity(payload, thresholds),
        snapshot_id=str(snapshot.id) if snapshot else None,
        video_list=video_list
    )


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

    # Update channel information if provided
    if (request.project_name or request.networks or request.department or
        request.employee_name or request.employee_id):
        async with PEM.get_session() as db:
            channel = await db.exec(
                select(Channel).where(Channel.channel_id == request.channel_id)
            )
            channel = channel.first()

            if channel:
                # Update existing fields
                if request.project_name:
                    channel.project_name = request.project_name

                # Update new fields
                if request.networks is not None:
                    channel.networks = request.networks
                if request.department is not None:
                    channel.department = request.department
                if request.employee_name is not None:
                    channel.employee_name = request.employee_name
                if request.employee_id is not None:
                    channel.employee_id = request.employee_id

                await db.commit()
                await db.refresh(channel)

    # Trigger background processing for embeddings
    if background_tasks:
        background_tasks.add_task(
            process_channel_embeddings,
            request.channel_id
        )

    return {
        "status": "success",
        "channel_id": request.channel_id,
        "total_videos": result["total_videos"],
        "new_videos": result["new_videos"],
        "existing_videos": result["existing_videos"],
        "updated_fields": {
            "project_name": request.project_name,
            "networks": request.networks,
            "department": request.department,
            "employee_name": request.employee_name,
            "employee_id": request.employee_id
        }
    }

async def process_channel_embeddings(channel_id: str):
    """Background task to process embeddings"""
    thumb_analyzer = ThumbnailAnalyzer()
    text_analyzer  = TextAnalyzer()
    video_analyzer = VideoAnalyzer()

    async def run_thumbnails():
        async with PEM.get_session() as db:
            return await thumb_analyzer.process_channel_thumbnails(db, channel_id)

    async def run_text():
        return await text_analyzer.process_channel_text(channel_id)

    async def run_video():
        async with PEM.get_session() as db:
            return await video_analyzer.analyze_channel_videos(db, channel_id)

    # chạy song song, mỗi task có session riêng
    thumb_result, text_result, video_result = await asyncio.gather(
        run_thumbnails(),
        run_text(),
        run_video()
    )

    # (tùy chọn) đóng http client nếu bạn thêm aclose()
    await thumb_analyzer.aclose()

    return {
        "thumbnails": thumb_result,
        "text": text_result,
        "video": video_result
    }



@router.get("/issues")
async def list_channels_with_issues(
    min_severity: float = Query(1.0, ge=0.0, le=100.0),
    limit: int = Query(50, ge=1, le=200),
    page: int = Query(1, ge=1),
    search: Optional[str] = Query(None),
    project_name: Optional[str] = Query(None),
    networks: Optional[str] = Query(None),
    department: Optional[str] = Query(None),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None),
    # Custom thresholds - khi có thì không dùng cache
    image_similarity_threshold: Optional[float] = Query(None),
    text_similarity_threshold: Optional[float] = Query(None),
    duplicate_threshold: Optional[float] = Query(None),
):
    """Lấy snapshot mới nhất mỗi kênh và lọc kênh đang 'vi phạm' (theo severity hoặc quy tắc)."""

    # Get effective thresholds
    thresholds = {
        "image_similarity_threshold": image_similarity_threshold or settings.image_similarity_threshold,
        "text_similarity_threshold": text_similarity_threshold or settings.text_similarity_threshold,
        "duplicate_threshold": duplicate_threshold or settings.duplicate_threshold
    }

    use_cache = not any([
        image_similarity_threshold is not None,
        text_similarity_threshold is not None,
        duplicate_threshold is not None,
        from_date is not None,
        to_date is not None
    ])

    async with PEM.get_session() as db:
        # Get all channels first
        channel_query = select(Channel)
        where_clauses = []

        if search:
            s = f"%{search.lower()}%"
            where_clauses.append(
                func.lower(Channel.channel_name).like(s) | func.lower(Channel.channel_id).like(s)
            )
        if project_name:
            where_clauses.append(Channel.project_name == project_name)
        if networks:
            where_clauses.append(Channel.networks == networks)
        if department:
            where_clauses.append(Channel.department == department)

        if where_clauses:
            channel_query = channel_query.where(and_(*where_clauses))

        channels_res = await db.exec(channel_query)
        channels = channels_res.all()

        # For each channel, find or create analysis
        items = []
        channels_analyzed = 0

        for channel in channels:
            ar = None

            if use_cache:
                # Try to find existing analysis result
                ar_query = select(AnalysisResult).where(
                    AnalysisResult.channel_id == channel.channel_id
                )

                # Filter by date range if provided
                if from_date is not None or to_date is not None:
                    ar_query = ar_query.where(
                        and_(
                            AnalysisResult.filter_from == from_date,
                            AnalysisResult.filter_to == to_date
                        )
                    )
                else:
                    # Get latest analysis with no date filter (both None)
                    ar_query = ar_query.where(
                        and_(
                            AnalysisResult.filter_from.is_(None),
                            AnalysisResult.filter_to.is_(None)
                        )
                    )

                ar_query = ar_query.order_by(AnalysisResult.analysis_date.desc()).limit(1)
                ar_res = await db.exec(ar_query)
                ar = ar_res.first()

            if not ar:
                # Need to analyze this channel - reuse existing analyze_channel logic
                request = AnalysisRequest(
                    channel_id=channel.channel_id,
                    from_date=from_date,
                    to_date=to_date,
                    image_similarity_threshold=image_similarity_threshold,
                    text_similarity_threshold=text_similarity_threshold,
                    duplicate_threshold=duplicate_threshold
                )

                try:
                    result = await analyze_channel(request)
                    channels_analyzed += 1

                    # Create temporary AR object from result for consistency
                    class TempAR:
                        def __init__(self, result_data):
                            self.severity_score = result_data.get("severity_score", 0)
                            self.channel_id = result_data.get("channel_id")
                            self.analysis_date = datetime.fromisoformat(result_data["analysis_date"].replace("Z", "+00:00"))

                            thumb = result_data.get("thumbnail_analysis", {}).get("thumbnail_duplicates", {})
                            text = result_data.get("text_analysis", {})
                            video = result_data.get("video_analysis", {})
                            monitor = result_data.get("channel_monitor", {})
                            business = result_data.get("business_rules", {})

                            self.thumbnail_duplicate_ratio = thumb.get("ratio", 0)
                            self.title_duplicate_ratio = text.get("title_duplicates", {}).get("ratio", 0)
                            self.description_duplicate_ratio = text.get("description_duplicates", {}).get("ratio", 0)
                            self.static_video_ratio = video.get("static_ratio", 0)
                            self.days_inactive = monitor.get("days_inactive", 0)
                            self.mbt_usage_ratio = business.get("mbt_ratio", 0)
                            self.project_compliance_ratio = business.get("project_compliance_ratio", 0)

                    ar = TempAR(result)
                except Exception as e:
                    logger.error(f"Failed to analyze channel {channel.channel_id}: {e}")
                    continue

            if ar:
                # Check if violates rules using current thresholds
                violates = (
                    ar.severity_score >= min_severity or
                    (ar.thumbnail_duplicate_ratio or 0) > thresholds["duplicate_threshold"] or
                    (ar.title_duplicate_ratio or 0) > thresholds["duplicate_threshold"] or
                    (ar.description_duplicate_ratio or 0) > thresholds["duplicate_threshold"] or
                    (ar.static_video_ratio or 0) > getattr(settings, "static_video_threshold", 0.5) or
                    (ar.days_inactive or 0) > settings.inactive_days or
                    ar.mbt_usage_ratio < 1 or
                    ar.project_compliance_ratio < 1
                )

                if violates:
                    items.append({
                        "channel_id": ar.channel_id,
                        "channel_name": channel.channel_name,
                        "project_name": channel.project_name,
                        "networks": channel.networks,
                        "department": channel.department,
                        "employee_name": channel.employee_name,
                        "severity_score": ar.severity_score,
                        "last_analyzed": ar.analysis_date.isoformat(),
                        "last_synced_at": channel.last_checked and channel.last_checked.isoformat(),
                        "highlights": {
                            "dup_thumbnail": ar.thumbnail_duplicate_ratio,
                            "dup_title": ar.title_duplicate_ratio,
                            "dup_desc": ar.description_duplicate_ratio,
                            "static_ratio": ar.static_video_ratio,
                            "inactive_days": ar.days_inactive,
                            "uses_mbt": ar.mbt_usage_ratio,
                            "project_compliant": ar.project_compliance_ratio,
                        }
                    })

        # Sort by severity score
        items.sort(key=lambda x: x["severity_score"], reverse=True)

        # Apply pagination
        total_items = len(items)
        start = (page - 1) * limit
        end = start + limit
        paginated_items = items[start:end]

        # Pagination meta
        items_per_page = limit
        total_pages = (total_items + items_per_page - 1) // items_per_page if items_per_page > 0 else 1

        # Collect lists for filter UI
        res_all = await db.exec(select(Channel.project_name, Channel.department, Channel.networks))
        tuple_rows = res_all.all()
        class _R: pass
        ch_objs = []
        for p, d, n in tuple_rows:
            o = _R(); o.project_name = p; o.department = d; o.networks = n; ch_objs.append(o)
        lists_payload = _collect_static_lists(ch_objs)

        return {
            "status": "success",
            "count": len(paginated_items),
            "items": paginated_items,
            "page": page,
            "totalPages": total_pages,
            "totalItems": total_items,
            "itemsPerPage": items_per_page,
            "from_date": from_date.isoformat() if from_date else None,
            "to_date": to_date.isoformat() if to_date else None,
            "channels_analyzed": channels_analyzed,
            "thresholds_used": thresholds,
            **lists_payload
        }

@router.post("/reanalyze")
async def reanalyze_channel(
    request: AnalysisRequest,
):
    video_analyzer = VideoAnalyzer()
    async def run_video():
        async with PEM.get_session() as db:
            return await video_analyzer.analyze_channel_videos(db, request.channel_id, reanalyze=True)

    video_result = await asyncio.gather(
        run_video()
    )

    return {
        "video": video_result
    }
