# yt_quality_checker

## Overview

YouTube Quality Checker - Hệ thống tự động kiểm tra chất lượng YouTube channels sử dụng AI/ML.

## New Excel Fields Support

### Added Fields

Project now supports additional fields from Excel file `channels_detail.xlsx`:

- **`Id kênh`** (required): YouTube Channel ID
- **`Dự án`** (optional): Project name
- **`Networks`** (optional): Network/company information
- **`Phòng`** (optional): Department responsible
- **`Họ và tên NPT`** (optional): Employee full name
- **`Mã Nhân viên NPT`** (optional): Employee ID
- **`force_refresh`** (optional): Force refresh flag

### Database Migration

After updating the code, run the migration to add new fields to the database:

```bash
# Add new fields
python app/models/migration_add_excel_fields.py

# Rollback if needed
python app/models/migration_add_excel_fields.py down
```

### Environment Variables

You can customize Excel column names via environment variables:

```bash
EXCEL_NETWORKS_COL="Networks"
EXCEL_DEPARTMENT_COL="Phòng"
EXCEL_EMPLOYEE_NAME_COL="Họ và tên NPT"
EXCEL_EMPLOYEE_ID_COL="Mã Nhân viên NPT"
```

## Architecture

- **Main App** (Port 8000): FastAPI backend, database operations
- **Scheduler** (Port 8002): Automated Excel reading and API calls
- **SigLIP Embedder** (Port 8001): Image similarity analysis
- **Database**: PostgreSQL + Qdrant vector store
