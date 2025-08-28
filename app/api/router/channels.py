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

def _same_range(a: Optional[datetime], b: Optional[datetime]) -> bool:
    return (a is None and b is None) or (a and b and a == b)

def _compute_severity(d: dict) -> float:
    # d là payload tổng hợp (thumb/text/video/monitor/business)
    dup_thres = settings.duplicate_threshold
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
                         f_from: Optional[datetime], f_to: Optional[datetime]) -> AnalysisResult:
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
        project_compliance       = payload["business_rules"]["project_compliant"],
        project_compliance_ratio = payload["business_rules"]["project_compliance_ratio"],
        non_compliant_videos     = payload["business_rules"]["non_compliant_videos"],

        # Summary
        total_issues = None,  # tuỳ bạn muốn tính
        issues       = None,  # hoặc build list text, có thể bổ sung sau
        severity_score = _compute_severity(payload),

        # Snapshot meta
        source_last_checked = source_last_checked,
        filter_from = f_from,
        filter_to   = f_to,
    )
    db.add(ar)
    await db.commit()
    await db.refresh(ar)
    return ar

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

        snap_q = (
            select(AnalysisResult)
            .where(
                AnalysisResult.channel_id == request.channel_id,
                # cùng phạm vi thời gian (None == None)
                # Lưu ý: so sánh None trong SQL hơi khác, nên tải về rồi lọc thêm trong Python
            )
            .order_by(AnalysisResult.analysis_date.desc())
            .limit(10)
        )
        snap_res = await db.exec(snap_q)
        snapshots = snap_res.all()

        cached = None
        for s in snapshots:
            if _same_range(s.filter_from, request.from_date) and _same_range(s.filter_to, request.to_date):
                if s.source_last_checked == channel.last_checked:
                    cached = s
                    break

    if cached:
        return {
            "status": "success",
            "channel_id": request.channel_id,
            "channel_name": channel.channel_name,
            "project_name": channel.project_name,
            "networks": channel.networks,
            "department": channel.department,
            "employee_name": channel.employee_name,
            "source": "cache",
            "analysis_date": cached.analysis_date.isoformat(),
            "severity_score": cached.severity_score,
            "snapshot_id": str(cached.id),
            # Trả gọn các trường đủ dùng cho UI; nếu cần full thì bổ sung
            "thumbnail_analysis": {
                "thumbnail_duplicates": {
                    "ratio": cached.thumbnail_duplicate_ratio,
                    "duplicate_pairs": cached.thumbnail_duplicate_pairs,
                    "total_videos": None, "threshold_exceeded": None
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
                "project_compliant": cached.project_compliance,
                "project_compliance_ratio": cached.project_compliance_ratio,
                "non_compliant_videos": cached.non_compliant_videos,
            }
        }

    async def run_analyze_thumbnails():
        async with PEM.get_session() as db:
            return await thumb_analyzer.analyze_channel_thumbnails(db, request.channel_id, request.from_date, request.to_date)

    async def run_analyze_text():
        async with PEM.get_session() as db:
            return await text_analyzer.analyze_channel_text(db, request.channel_id, request.from_date, request.to_date)

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

    async with PEM.get_session() as db:
        ch_res = await db.exec(select(Channel).where(Channel.channel_id == request.channel_id))
        channel = ch_res.first()
        snapshot = await _save_snapshot(
            db, request.channel_id, payload,
            source_last_checked=channel.last_checked,
            f_from=request.from_date, f_to=request.to_date
        )

    payload["video_analysis"] = {
        "static_ratio": video_result["static_ratio"],
        "static_video_ids": [r["video_id"] for r in video_result.get("results", []) if r.get("is_static")]
    }

    return {
        "status": "success",
        "source": "fresh",
        "analysis_date": snapshot.analysis_date.isoformat(),
        "severity_score": snapshot.severity_score,
        **payload
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
):
    """Lấy snapshot mới nhất mỗi kênh và lọc kênh đang 'vi phạm' (theo severity hoặc quy tắc)."""
    async with PEM.get_session() as db:
        # Subquery: latest analysis per channel
        subq = (
            select(
                AnalysisResult.channel_id,
                func.max(AnalysisResult.analysis_date).label("maxd")
            )
            .group_by(AnalysisResult.channel_id)
            .subquery()
        )

        # Base join of latest analysis with channel
        base = (
            select(AnalysisResult, Channel)
            .join(subq, and_(
                AnalysisResult.channel_id == subq.c.channel_id,
                AnalysisResult.analysis_date == subq.c.maxd
            ))
            .join(Channel, Channel.channel_id == AnalysisResult.channel_id)
        )

        # Violation predicate in SQL
        violate_pred = (
            (AnalysisResult.severity_score >= min_severity)
            | (func.coalesce(AnalysisResult.thumbnail_duplicate_ratio, 0) > settings.duplicate_threshold)
            | (func.coalesce(AnalysisResult.title_duplicate_ratio, 0) > settings.duplicate_threshold)
            | (func.coalesce(AnalysisResult.description_duplicate_ratio, 0) > settings.duplicate_threshold)
            | (func.coalesce(AnalysisResult.static_video_ratio, 0) > getattr(settings, "static_video_threshold", 0.5))
            | (func.coalesce(AnalysisResult.days_inactive, 0) > settings.inactive_days)
            | (AnalysisResult.project_compliance == False)
        )

        where_clauses = [violate_pred]

        # Optional filters
        if search:
            s = f"%{search.lower()}%"
            where_clauses.append(
                func.lower(Channel.channel_name).like(s) | func.lower(Channel.channel_id).like(s)
            )
        if project_name:
            where_clauses.append((Channel.project_name == project_name))
        if networks:
            where_clauses.append((Channel.networks == networks))
        if department:
            where_clauses.append((Channel.department == department))

        # Compose filtered query
        filtered_q = base.where(and_(*where_clauses))

        # Count total items
        count_q = select(func.count()).select_from(filtered_q.subquery())
        total_items = (await db.exec(count_q)).one()

        # Apply order and pagination
        paged_q = (
            filtered_q
            .order_by(AnalysisResult.severity_score.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        )

        res = await db.exec(paged_q)
        rows = res.all()

        items = []
        for ar, ch in rows:
            items.append({
                "channel_id": ar.channel_id,
                "channel_name": ch.channel_name,
                "project_name": ch.project_name,
                "networks": ch.networks,
                "department": ch.department,
                "employee_name": ch.employee_name,
                "severity_score": ar.severity_score,
                "last_analyzed": ar.analysis_date.isoformat(),
                "last_synced_at": ch.last_checked and ch.last_checked.isoformat(),
                "highlights": {
                    "dup_thumbnail": ar.thumbnail_duplicate_ratio,
                    "dup_title": ar.title_duplicate_ratio,
                    "dup_desc": ar.description_duplicate_ratio,
                    "static_ratio": ar.static_video_ratio,
                    "inactive_days": ar.days_inactive,
                    "project_compliant": ar.project_compliance,
                }
            })

        # Pagination meta
        items_per_page = limit
        total_pages = (total_items + items_per_page - 1) // items_per_page if items_per_page > 0 else 1

        # collect lists for filter UI
        res_all = await db.exec(select(Channel.project_name, Channel.department, Channel.networks))
        tuple_rows = res_all.all()
        class _R: pass
        ch_objs = []
        for p, d, n in tuple_rows:
            o = _R(); o.project_name = p; o.department = d; o.networks = n; ch_objs.append(o)
        lists_payload = _collect_static_lists(ch_objs)

        return {
            "status": "success",
            "count": len(items),
            "items": items,
            "page": page,
            "totalPages": total_pages,
            "totalItems": total_items,
            "itemsPerPage": items_per_page,
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
