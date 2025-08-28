# app/sync_service.py
import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

import pandas as pd
import httpx
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# ----------------------------
# Config & Logging
# ----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("yt-sync-scheduler")

TZ_NAME = os.getenv("TZ", "Asia/Bangkok")  # UTC+7
TZ = ZoneInfo(TZ_NAME)

DEV_YT_BASE_URL = os.getenv("DEV_YT_BASE_URL", "http://dev-yt:8000")
SYNC_ENDPOINT = os.getenv("SYNC_ENDPOINT", "/api/v1/channels/sync")
ANALYZE_ENDPOINT = os.getenv("ANALYZE_ENDPOINT", "/api/v1/channels/analyze")
REANALYZE_ENDPOINT = os.getenv("REANALYZE_ENDPOINT", "/api/v1/channels/reanalyze")

EXCEL_PATH = os.getenv("EXCEL_PATH", "channels_detail.xlsx")
_EXCEL_SHEET_RAW = os.getenv("EXCEL_SHEET", "").strip()
if _EXCEL_SHEET_RAW == "":
    EXCEL_SHEET = 0
else:
    EXCEL_SHEET = int(_EXCEL_SHEET_RAW) if _EXCEL_SHEET_RAW.isdigit() else _EXCEL_SHEET_RAW

# Excel column mappings - all optional to avoid errors
EXCEL_ID_COL = os.getenv("EXCEL_ID_COL", "Id kênh")
EXCEL_PROJECT_COL = os.getenv("EXCEL_PROJECT_COL", "Dự án")
EXCEL_FORCE_COL = os.getenv("EXCEL_FORCE_COL", "force_refresh")

# New additional columns - all optional
EXCEL_NETWORKS_COL = os.getenv("EXCEL_NETWORKS_COL", "Networks")
EXCEL_DEPARTMENT_COL = os.getenv("EXCEL_DEPARTMENT_COL", "Phòng")
EXCEL_EMPLOYEE_NAME_COL = os.getenv("EXCEL_EMPLOYEE_NAME_COL", "Họ và tên NPT")
EXCEL_EMPLOYEE_ID_COL = os.getenv("EXCEL_EMPLOYEE_ID_COL", "Mã Nhân viên NPT")

# Nếu muốn chạy ngay khi khởi động (ngoài các mốc giờ), set 1
RUN_ON_STARTUP = os.getenv("RUN_ON_STARTUP", "0") == "1"

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))  # seconds
CONCURRENCY = int(os.getenv("CONCURRENCY", "4"))
RETRY = int(os.getenv("RETRY", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.5"))  # exponential backoff factor

# ----------------------------
# Data loading
# ----------------------------
def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def load_channels_from_excel() -> List[Dict[str, Any]]:
    """Đọc Excel và trả về list dict: {channel_id, project_name, force_refresh, networks, department, employee_name, employee_id}"""
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    raw = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET)

    # Chọn DataFrame từ raw
    if isinstance(raw, dict):
        # Cố gắng chọn sheet có cột channel_id (case-insensitive)
        picked_df = None
        picked_name = None
        for name, df_candidate in raw.items():
            cols_map = {str(c).lower().strip(): c for c in df_candidate.columns}
            if EXCEL_ID_COL.lower() in cols_map:
                picked_df = df_candidate
                picked_name = name
                break
        if picked_df is None:
            # Không tìm thấy -> lấy sheet đầu tiên
            picked_name = next(iter(raw))
            picked_df = raw[picked_name]
        logger.info(f"Selected sheet: {picked_name}")
        df = picked_df
    else:
        df = raw

    # Chuẩn hoá tên cột
    cols_map = {str(c).lower().strip(): c for c in df.columns}

    def pick(colname: str) -> Optional[str]:
        return cols_map.get(colname.lower())

    # Required column - channel_id
    id_col = pick(EXCEL_ID_COL) or EXCEL_ID_COL

    # Optional columns - all safe to handle missing
    prj_col = pick(EXCEL_PROJECT_COL) if EXCEL_PROJECT_COL else None
    frc_col = pick(EXCEL_FORCE_COL) if EXCEL_FORCE_COL else None

    # New optional columns
    networks_col = pick(EXCEL_NETWORKS_COL) if EXCEL_NETWORKS_COL else None
    dept_col = pick(EXCEL_DEPARTMENT_COL) if EXCEL_DEPARTMENT_COL else None
    emp_name_col = pick(EXCEL_EMPLOYEE_NAME_COL) if EXCEL_EMPLOYEE_NAME_COL else None
    emp_id_col = pick(EXCEL_EMPLOYEE_ID_COL) if EXCEL_EMPLOYEE_ID_COL else None

    if id_col not in df.columns:
        raise ValueError(
            f"Missing required column '{EXCEL_ID_COL}' in Excel. "
            f"Found columns: {list(df.columns)}"
        )

    # lọc NA, strip, unique theo channel_id
    df[id_col] = df[id_col].astype(str).str.strip()
    df = df[df[id_col] != ""].dropna(subset=[id_col]).drop_duplicates(subset=[id_col])

    payloads: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        payload = {
            "channel_id": row[id_col],
            "project_name": (str(row[prj_col]).strip() if prj_col and pd.notna(row[prj_col]) else None),
            "force_refresh": (_to_bool(row[frc_col]) if frc_col and frc_col in df.columns else False),
        }

        # Add new optional fields safely
        if networks_col and networks_col in df.columns and pd.notna(row[networks_col]):
            payload["networks"] = str(row[networks_col]).strip()

        if dept_col and dept_col in df.columns and pd.notna(row[dept_col]):
            payload["department"] = str(row[dept_col]).strip()

        if emp_name_col and emp_name_col in df.columns and pd.notna(row[emp_name_col]):
            payload["employee_name"] = str(row[emp_name_col]).strip()

        if emp_id_col and emp_id_col in df.columns and pd.notna(row[emp_id_col]):
            payload["employee_id"] = str(row[emp_id_col]).strip()

        payloads.append(payload)

    return payloads

# ----------------------------
# HTTP sync with retries
# ----------------------------
async def post_with_retries(client: httpx.AsyncClient, url: str, json: Dict[str, Any]) -> Dict[str, Any]:
    last_exc = None
    for attempt in range(1, RETRY + 1):
        try:
            resp = await client.post(url, json=json, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return {"ok": True, "data": resp.json()}
        except Exception as e:
            last_exc = e
            wait = (RETRY_BACKOFF ** (attempt - 1))
            logger.warning(f"[{json.get('channel_id')}] Attempt {attempt}/{RETRY} failed: {e}. Backoff {wait:.1f}s")
            await asyncio.sleep(wait)
    return {"ok": False, "error": str(last_exc)}

async def sync_one_channel(client: httpx.AsyncClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Gửi request sync với tất cả các trường từ Excel"""
    url = f"{DEV_YT_BASE_URL.rstrip('/')}{SYNC_ENDPOINT}"
    # payload giờ có thể chứa: channel_id, project_name, force_refresh, networks, department, employee_name, employee_id
    return await post_with_retries(client, url, payload)

async def sync_all_channels() -> Dict[str, Any]:
    """Đọc Excel và sync tất cả channel."""
    started_at = datetime.now(TZ).isoformat()
    logger.info(f"Starting sync at {started_at} ({TZ_NAME})")

    try:
        channels = load_channels_from_excel()
        total = len(channels)
        logger.info(f"Loaded {total} channel(s) from Excel")

        # Log sample of loaded data for debugging
        if total > 0:
            sample_channel = channels[0]
            logger.info(f"Sample channel data: {sample_channel}")

        if total == 0:
            return {"status": "no_channels"}

        results = {"ok": 0, "fail": 0, "details": []}

        limits = asyncio.Semaphore(CONCURRENCY)
        async with httpx.AsyncClient() as client:
            async def worker(ch_payload):
                async with limits:
                    res = await sync_one_channel(client, ch_payload)
                    if res.get("ok"):
                        results["ok"] += 1
                    else:
                        results["fail"] += 1
                    results["details"].append({"channel_id": ch_payload["channel_id"], **res})

            await asyncio.gather(*(worker(p) for p in channels))

        logger.info(f"Sync done. OK={results['ok']}, FAIL={results['fail']}")
        return {"status": "done", **results}
    except Exception as e:
        logger.exception("Sync failed due to unexpected error")
        return {"status": "error", "error": str(e)}

async def analyze_one_channel(client: httpx.AsyncClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{DEV_YT_BASE_URL.rstrip('/')}{ANALYZE_ENDPOINT}"
    return await post_with_retries(client, url, payload)

async def analyze_all_channels() -> Dict[str, Any]:
    """Đọc Excel và analyze tất cả channel."""
    started_at = datetime.now(TZ).isoformat()
    logger.info(f"Starting analyze at {started_at} ({TZ_NAME})")

    try:
        channels = load_channels_from_excel()
        total = len(channels)
        logger.info(f"Loaded {total} channel(s) from Excel")

        # Log sample of loaded data for debugging
        if total > 0:
            sample_channel = channels[0]
            logger.info(f"Sample channel data: {sample_channel}")

        if total == 0:
            return {"status": "no_channels"}

        results = {"ok": 0, "fail": 0, "details": []}

        limits = asyncio.Semaphore(CONCURRENCY)
        async with httpx.AsyncClient() as client:
            async def worker(ch_payload):
                async with limits:
                    res = await analyze_one_channel(client, {"channel_id": ch_payload["channel_id"]})
                    if res.get("ok"):
                        results["ok"] += 1
                    else:
                        results["fail"] += 1
                    results["details"].append({"channel_id": ch_payload["channel_id"], **res})

            await asyncio.gather(*(worker(p) for p in channels))

        logger.info(f"Analze done. OK={results['ok']}, FAIL={results['fail']}")
        return {"status": "done", **results}
    except Exception as e:
        logger.exception("Analze failed due to unexpected error")
        return {"status": "error", "error": str(e)}

async def reanalyze_one_channel(client: httpx.AsyncClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{DEV_YT_BASE_URL.rstrip('/')}{REANALYZE_ENDPOINT}"
    return await post_with_retries(client, url, payload)

async def reanalyze_all_channels() -> Dict[str, Any]:
    """Đọc Excel và reanalyze tất cả channel."""
    started_at = datetime.now(TZ).isoformat()
    logger.info(f"Starting reanalyze at {started_at} ({TZ_NAME})")

    try:
        channels = load_channels_from_excel()
        total = len(channels)
        logger.info(f"Loaded {total} channel(s) from Excel")

        # Log sample of loaded data for debugging
        if total > 0:
            sample_channel = channels[0]
            logger.info(f"Sample channel data: {sample_channel}")

        if total == 0:
            return {"status": "no_channels"}

        results = {"ok": 0, "fail": 0, "details": []}

        limits = asyncio.Semaphore(12)
        async with httpx.AsyncClient() as client:
            async def worker(ch_payload):
                async with limits:
                    res = await reanalyze_one_channel(client, {"channel_id": ch_payload["channel_id"]})
                    if res.get("ok"):
                        results["ok"] += 1
                    else:
                        results["fail"] += 1
                    results["details"].append({"channel_id": ch_payload["channel_id"], **res})

            await asyncio.gather(*(worker(p) for p in channels))

        logger.info(f"Reanalyze done. OK={results['ok']}, FAIL={results['fail']}")
        return {"status": "done", **results}
    except Exception as e:
        logger.exception("Reanalyze failed due to unexpected error")
        return {"status": "error", "error": str(e)}

# ----------------------------
# APScheduler + FastAPI Lifespan
# ----------------------------
scheduler = AsyncIOScheduler(timezone=str(TZ))

def add_cron_jobs():
    # 01:00, 07:00, 13:00, 19:00 (Asia/Bangkok)
    for hour in (7, 15, 23):
        scheduler.add_job(
            sync_all_channels,
            trigger=CronTrigger(hour=hour, minute=0),
            id=f"sync_{hour:02d}_00",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600
        )
        logger.info(f"Scheduled daily sync at {hour:02d}:00 {TZ_NAME}")

    for hour in (6, 14, 22):
        scheduler.add_job(
            analyze_all_channels,
            trigger=CronTrigger(hour=hour, minute=0),
            id=f"sync_{hour:02d}_00",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600
        )
        logger.info(f"Scheduled daily sync at {hour:02d}:00 {TZ_NAME}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    add_cron_jobs()
    scheduler.start()
    logger.info("Scheduler started")
    if RUN_ON_STARTUP:
        asyncio.create_task(sync_all_channels())
    try:
        yield
    finally:
        # shutdown
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

app = FastAPI(title="YT Sync Scheduler", version="1.0.0", lifespan=lifespan)

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now(TZ).isoformat(), "tz": TZ_NAME}

@app.post("/sync-now")
async def sync_now():
    """Cho phép gọi tay để sync ngay (tuỳ chọn)."""
    return await sync_all_channels()

@app.post("/analyze-now")
async def analyze_now():
    """Cho phép gọi tay để analyze ngay (tuỳ chọn)."""
    return await analyze_all_channels()

@app.post("/reanalyze-now")
async def reanalyze_now():
    """Cho phép gọi tay để reanalyze ngay (tuỳ chọn)."""
    return await reanalyze_all_channels()
