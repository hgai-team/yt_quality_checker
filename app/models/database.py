from sqlalchemy import (
    Column, String, DateTime, Boolean, Float, JSON, 
    ForeignKey, Text, Integer, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


class Channel(Base):
    __tablename__ = "channels"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id = Column(String, unique=True, nullable=False, index=True)
    channel_name = Column(String)
    channel_url = Column(String)
    project_name = Column(String, index=True)
    
    # Tracking
    created_at   = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_checked = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_video_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    videos = relationship("Video", back_populates="channel", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="channel")
    
    __table_args__ = (
        Index('idx_channel_project', 'project_name'),
    )


class Video(Base):
    __tablename__ = "videos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(String, unique=True, nullable=False, index=True)
    channel_id = Column(UUID(as_uuid=True), ForeignKey("channels.id", ondelete="CASCADE"))
    
    # Metadata
    title = Column(Text)
    description = Column(Text)
    thumbnail_url = Column(String)
    upload_date = Column(DateTime)
    duration = Column(Float)
    view_count = Column(Integer)
    like_count = Column(Integer)
    tags = Column(ARRAY(String))
    
    # Processed data
    thumbnail_hash = Column(String, index=True)
    thumbnail_embedding_id = Column(String)  # Qdrant point ID
    title_embedding_id = Column(String)  # Qdrant point ID
    description_embedding_id = Column(String)  # Qdrant point ID
    
    # Analysis results
    is_static_video = Column(Boolean)
    static_confidence = Column(Float)
    static_analysis_method = Column(String)
    static_analysis_details = Column(JSONB)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    analyzed_at = Column(DateTime)
    
    # Relationships
    channel = relationship("Channel", back_populates="videos")
    
    __table_args__ = (
        Index('idx_video_channel', 'channel_id'),
        Index('idx_video_upload', 'upload_date'),
        Index('idx_video_analyzed', 'analyzed_at'),
    )


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id = Column(UUID(as_uuid=True), ForeignKey("channels.id"))
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    # Duplicate metrics
    thumbnail_duplicate_ratio = Column(Float)
    thumbnail_duplicate_pairs = Column(JSONB)
    title_duplicate_ratio = Column(Float)
    title_duplicate_pairs = Column(JSONB)
    description_duplicate_ratio = Column(Float)
    description_duplicate_pairs = Column(JSONB)
    
    # Video analysis
    static_video_ratio = Column(Float)
    static_video_ids = Column(ARRAY(String))
    
    # Activity metrics
    days_inactive = Column(Integer)
    last_upload_date = Column(DateTime)
    
    # Business rules
    uses_mbt = Column(Boolean)
    mbt_usage_ratio = Column(Float)
    project_compliance = Column(Boolean)
    project_compliance_ratio = Column(Float)
    non_compliant_videos = Column(JSONB)
    
    # Summary
    total_issues = Column(Integer)
    issues = Column(JSONB)
    recommendations = Column(JSONB)
    severity_score = Column(Float)  # 0-100
    
    # Relationships
    channel = relationship("Channel", back_populates="analysis_results")
    
    __table_args__ = (
        Index('idx_analysis_date', 'analysis_date'),
        Index('idx_analysis_channel_date', 'channel_id', 'analysis_date'),
    )


class EmbeddingCache(Base):
    """Cache for expensive embeddings"""
    __tablename__ = "embedding_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String, unique=True, nullable=False, index=True)
    content_type = Column(String)  # 'image', 'text'
    embedding = Column(JSONB)
    model_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)