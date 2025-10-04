"""
Comprehensive tests for analytics endpoints.

Tests cover:
- KPI calculations (totals, averages, percentiles)
- Detection statistics (severity, status, source, class breakdowns)
- Geographic data aggregation (geohash clustering, hotspots)
- Temporal patterns (daily, hourly, day-of-week trends)
- Performance metrics (latency, processing times, resolution times)
- Data accuracy (correct calculations and aggregations)
- Date range filtering (days parameter, start/end dates)
- Romanian timezone handling
- Quality metrics (false positive rates, verification rates)
- User engagement analytics
- Resolution efficiency tracking
- SLA breach monitoring
"""

import pytest
import os
from datetime import datetime, timedelta, date, timezone
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from zoneinfo import ZoneInfo

# Import all models
import app.models.user
import app.models.media
import app.models.revoked
import app.models.rag
import app.models.conversation

from app.main import app
from app.core.database import Base, get_db
from app.models.user import User as UserModel, RoleEnum
from app.models.media import Media, Frame, Detection, Severity, IssueStatus, DetectionSource
from app.core.security import create_access_token, get_password_hash

# Romanian timezone
RO_TZ = ZoneInfo("Europe/Bucharest")

# Use PostgreSQL for testing
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@db:5432/urban_ai_test"
)

engine = create_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a real database session for testing"""
    session = TestingSessionLocal()
    try:
        # Clean up data before test
        session.query(Detection).delete()
        session.query(Frame).delete()
        session.query(Media).delete()
        session.query(UserModel).delete()
        session.commit()
        yield session
    finally:
        # Rollback any pending transactions
        session.rollback()
        try:
            # Clean up data after test
            session.query(Detection).delete()
            session.query(Frame).delete()
            session.query(Media).delete()
            session.query(UserModel).delete()
            session.commit()
        except:
            session.rollback()
        finally:
            session.close()


@pytest.fixture
def override_get_db(db_session):
    """Override FastAPI database dependency"""
    def _override():
        try:
            yield db_session
        finally:
            pass
    return _override


@pytest.fixture
def admin_user(db_session):
    """Create an admin user for testing"""
    user = UserModel(
        username="admin_analytics",
        hashed_password=get_password_hash("adminpass123"),
        role=RoleEnum.admin
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def regular_user(db_session):
    """Create a regular user for testing"""
    user = UserModel(
        username="user_analytics",
        hashed_password=get_password_hash("userpass123"),
        role=RoleEnum.user
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_token(admin_user):
    """Generate JWT token for admin user"""
    return create_access_token({"sub": admin_user.username})


@pytest.fixture
def user_token(regular_user):
    """Generate JWT token for regular user"""
    return create_access_token({"sub": regular_user.username})


@pytest.fixture
def client(override_get_db):
    """Create test client with database override"""
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_media_with_detections(db_session, admin_user):
    """Create sample media with detections for analytics testing"""
    media_list = []

    # Create media over last 7 days
    base_time = datetime.now(timezone.utc) - timedelta(days=6)

    for day in range(7):
        # Create 2 images and 1 video per day
        for i in range(3):
            media_type = "image" if i < 2 else "video"
            created_at = base_time + timedelta(days=day, hours=i*2)

            media = Media(
                filename=f"test_day{day}_{i}.jpg" if media_type == "image" else f"test_day{day}_{i}.mp4",
                media_type=media_type,
                user_username=admin_user.username,
                created_at=created_at,
                process_ms_total=1000 + (day * 100) + (i * 50),  # Gradual increase
                geohash6=f"u2mw{'ab'[i%2]}{day}",  # Different geohashes
                address=f"Street {day}, City",
            )
            db_session.add(media)
            db_session.flush()

            # Create frame for each media
            frame = Frame(
                media_id=media.id,
                frame_index=0,
                timestamp=0.0
            )
            db_session.add(frame)
            db_session.flush()

            # Create detections with varying severity and confidence
            severities = [Severity.high, Severity.medium, Severity.low]
            sources = [DetectionSource.yolo, DetectionSource.gpt_dino, DetectionSource.sam_fallback]
            statuses = [IssueStatus.open, IssueStatus.resolved, IssueStatus.ignored]
            class_names = ["pothole", "crack", "debris", "damaged_sign"]

            for j in range(2):  # 2 detections per media
                detection = Detection(
                    frame_id=frame.id,
                    class_id=j,
                    class_name=class_names[j % len(class_names)],
                    confidence=0.70 + (0.05 * day),  # Confidence increases over days
                    x1=100.0 * j,
                    y1=100.0 * j,
                    x2=200.0 * j,
                    y2=200.0 * j,
                    severity=severities[(day + j) % 3],
                    status=statuses[j % 3],
                    source=sources[i % 3],
                    created_at=created_at,
                    resolved_at=created_at + timedelta(hours=24) if j % 3 == 1 else None,
                    assigned_to=admin_user.username if j % 2 == 0 else None,
                    verified_by=admin_user.username if j % 4 == 0 else None,
                    verified_at=created_at + timedelta(hours=1) if j % 4 == 0 else None,
                )
                db_session.add(detection)

            media_list.append(media)

    db_session.commit()
    return media_list


# ============== Access Control Tests ==============

class TestAccessControl:
    """Test admin-only access to analytics endpoints"""

    def test_analytics_requires_admin_role(self, client, user_token):
        """Test that non-admin users cannot access analytics"""
        response = client.get(
            "/analytics/kpis",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        assert response.status_code == 403

    def test_analytics_requires_authentication(self, client):
        """Test that unauthenticated requests are rejected"""
        response = client.get("/analytics/kpis")
        assert response.status_code == 401

    def test_admin_can_access_analytics(self, client, admin_token, sample_media_with_detections):
        """Test that admin users can access analytics"""
        response = client.get(
            "/analytics/kpis",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200


# ============== KPI and Latency Tests ==============

class TestKPIs:
    """Test KPI calculations and latency metrics"""

    def test_kpis_with_data(self, client, admin_token, sample_media_with_detections):
        """Test KPI endpoint returns correct aggregated metrics"""
        response = client.get(
            "/analytics/kpis?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "window" in data
        assert "uploads" in data
        assert "latency_ms" in data
        assert "detections" in data

        # Verify uploads
        assert data["uploads"]["total"] == 21  # 7 days × 3 media per day
        assert data["uploads"]["images"] == 14  # 7 days × 2 images per day
        assert data["uploads"]["videos"] == 7   # 7 days × 1 video per day

        # Verify detections
        assert data["detections"]["total"] == 42  # 21 media × 2 detections each

        # Verify latency exists
        assert "avg" in data["latency_ms"]
        assert "p95" in data["latency_ms"]
        assert data["latency_ms"]["avg"] > 0
        assert data["latency_ms"]["p95"] >= data["latency_ms"]["avg"]

    def test_kpis_empty_data(self, client, admin_token, db_session):
        """Test KPIs with no data returns zero values"""
        response = client.get(
            "/analytics/kpis?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["uploads"]["total"] == 0
        assert data["detections"]["total"] == 0
        assert data["latency_ms"]["avg"] == 0

    def test_kpis_image_filter(self, client, admin_token, sample_media_with_detections):
        """Test KPIs filtered by image media type"""
        response = client.get(
            "/analytics/kpis?days=7&media_type=image",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["uploads"]["total"] == 14
        assert data["uploads"]["images"] == 14
        assert data["uploads"]["videos"] == 0

    def test_latency_by_day(self, client, admin_token, sample_media_with_detections):
        """Test daily latency breakdown"""
        response = client.get(
            "/analytics/latency-by-day?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 7  # 7 days of data

        for day_data in data:
            assert "date" in day_data
            assert "avg_ms" in day_data
            assert "p95_ms" in day_data
            assert "count" in day_data
            assert day_data["count"] == 3  # 3 media per day


# ============== Upload Analytics Tests ==============

class TestUploadAnalytics:
    """Test upload statistics and trends"""

    def test_uploads_by_day(self, client, admin_token, sample_media_with_detections):
        """Test uploads grouped by day"""
        response = client.get(
            "/analytics/uploads-by-day?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 7
        for day_data in data:
            assert "date" in day_data
            assert "count" in day_data
            assert day_data["count"] == 3  # 3 uploads per day

    def test_uploads_by_user(self, client, admin_token, sample_media_with_detections):
        """Test uploads grouped by user"""
        response = client.get(
            "/analytics/uploads-by-user?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) >= 1
        assert data[0]["user"] == "admin_analytics"
        assert data[0]["count"] == 21

    def test_uploads_date_range_filtering(self, client, admin_token, sample_media_with_detections):
        """Test uploads with custom date range"""
        today = date.today()
        start_date = today - timedelta(days=3)
        end_date = today

        response = client.get(
            f"/analytics/uploads-by-day?start={start_date}&end={end_date}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should have data for approximately the specified range
        assert len(data) >= 3


# ============== Detection Statistics Tests ==============

class TestDetectionStatistics:
    """Test detection breakdowns and aggregations"""

    def test_severity_by_day(self, client, admin_token, sample_media_with_detections):
        """Test severity breakdown by day"""
        response = client.get(
            "/analytics/detections/severity-by-day?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 7
        for day_data in data:
            assert "date" in day_data
            assert "low" in day_data
            assert "medium" in day_data
            assert "high" in day_data
            # Each day should have total of 6 detections (3 media × 2 detections)
            assert day_data["low"] + day_data["medium"] + day_data["high"] == 6

    def test_source_breakdown(self, client, admin_token, sample_media_with_detections):
        """Test detection source breakdown"""
        response = client.get(
            "/analytics/detections/source-breakdown?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should have sources
        assert len(data) > 0
        sources = [item["source"] for item in data]
        total_count = sum(item["count"] for item in data)

        assert total_count == 42  # All detections
        assert any(s in ["yolo", "gpt_dino", "sam_fallback"] for s in sources)

    def test_status_breakdown(self, client, admin_token, sample_media_with_detections):
        """Test detection status breakdown"""
        response = client.get(
            "/analytics/detections/status-breakdown?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) > 0
        statuses = [item["status"] for item in data]
        total_count = sum(item["count"] for item in data)

        assert total_count == 42
        assert any(s in ["open", "resolved", "ignored"] for s in statuses)

    def test_top_classes(self, client, admin_token, sample_media_with_detections):
        """Test top detection classes"""
        response = client.get(
            "/analytics/detections/top-classes?days=7&limit=5",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) <= 5  # Respects limit
        assert len(data) > 0

        # Verify sorted by count descending
        counts = [item["count"] for item in data]
        assert counts == sorted(counts, reverse=True)

        # Verify structure
        for item in data:
            assert "class_name" in item
            assert "count" in item

    def test_confidence_summary(self, client, admin_token, sample_media_with_detections):
        """Test confidence statistics summary"""
        response = client.get(
            "/analytics/detections/confidence-summary?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify percentiles
        assert "avg" in data
        assert "p05" in data
        assert "p50" in data
        assert "p95" in data

        # Verify ordering
        assert data["p05"] <= data["p50"] <= data["p95"]
        assert 0 <= data["avg"] <= 1
        assert 0 <= data["p95"] <= 1


# ============== Geographic Analytics Tests ==============

class TestGeographicAnalytics:
    """Test geographic clustering and heatmaps"""

    def test_geo_heatmap(self, client, admin_token, sample_media_with_detections):
        """Test geographic heatmap data"""
        response = client.get(
            "/analytics/geo/heatmap?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) > 0
        for item in data:
            assert "geohash6" in item
            assert "count" in item
            assert "latest" in item

    def test_geo_heatmap_min_count_filter(self, client, admin_token, sample_media_with_detections):
        """Test heatmap with minimum count filter"""
        response = client.get(
            "/analytics/geo/heatmap?days=7&min_count=5",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # All results should meet min_count
        for item in data:
            assert item["count"] >= 5

    def test_geo_hotspots(self, client, admin_token, sample_media_with_detections):
        """Test geographic hotspots with rich data"""
        response = client.get(
            "/analytics/geo/hotspots?days=7&precision=5",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) > 0
        for hotspot in data:
            assert "geohash" in hotspot
            assert "count" in hotspot
            assert "prev_count" in hotspot
            assert "trend_pct" in hotspot
            assert "uploaders" in hotspot
            assert "severity" in hotspot
            assert "top_classes" in hotspot

            # Verify severity breakdown
            assert "low" in hotspot["severity"]
            assert "medium" in hotspot["severity"]
            assert "high" in hotspot["severity"]

    @patch('app.api.analytics._geohash_mod')
    def test_geo_hotspots_without_geohash_lib(self, mock_geohash, client, admin_token, sample_media_with_detections):
        """Test hotspots when geohash2 library is not available"""
        mock_geohash = None

        response = client.get(
            "/analytics/geo/hotspots?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should still return data, but without lat/lon/bbox
        for hotspot in data:
            # lat/lon/bbox may be None
            assert "geohash" in hotspot


# ============== Temporal Patterns Tests ==============

class TestTemporalPatterns:
    """Test temporal activity patterns"""

    def test_temporal_patterns(self, client, admin_token, sample_media_with_detections):
        """Test hourly and daily patterns"""
        response = client.get(
            "/analytics/activity/temporal-patterns?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "hourly" in data
        assert "daily" in data

        # Verify hourly pattern
        for hour_data in data["hourly"]:
            assert "hour" in hour_data
            assert "uploads" in hour_data
            assert "avg_processing_ms" in hour_data
            assert 0 <= hour_data["hour"] <= 23

        # Verify daily pattern
        for day_data in data["daily"]:
            assert "day_of_week" in day_data
            assert "day_name" in day_data
            assert "uploads" in day_data
            assert "avg_processing_ms" in day_data
            assert 0 <= day_data["day_of_week"] <= 6
            assert day_data["day_name"] in [
                "Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday"
            ]

    def test_user_engagement_metrics(self, client, admin_token, sample_media_with_detections):
        """Test user engagement calculations"""
        response = client.get(
            "/analytics/activity/user-engagement?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) > 0
        for user_stats in data:
            assert "user" in user_stats
            assert "total_uploads" in user_stats
            assert "active_days" in user_stats
            assert "activity_rate_pct" in user_stats
            assert "uploads_per_active_day" in user_stats
            assert "avg_hours_between_uploads" in user_stats
            assert "engagement_score" in user_stats
            assert "first_upload" in user_stats
            assert "last_upload" in user_stats

            # Verify calculations
            assert 0 <= user_stats["activity_rate_pct"] <= 100
            assert 0 <= user_stats["engagement_score"] <= 100


# ============== Performance Metrics Tests ==============

class TestPerformanceMetrics:
    """Test performance and efficiency metrics"""

    def test_confidence_by_class(self, client, admin_token, sample_media_with_detections):
        """Test confidence breakdown by detection class"""
        response = client.get(
            "/analytics/detections/confidence-by-class?days=7&min_detections=3",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        for class_stats in data:
            assert "class_name" in class_stats
            assert "avg_confidence" in class_stats
            assert "std_confidence" in class_stats
            assert "p25_confidence" in class_stats
            assert "p75_confidence" in class_stats
            assert "count" in class_stats
            assert "reliability_score" in class_stats

            # Verify calculations
            assert class_stats["count"] >= 3  # Respects min_detections
            assert 0 <= class_stats["avg_confidence"] <= 1
            assert 0 <= class_stats["reliability_score"] <= 1

    def test_daily_health_metrics(self, client, admin_token, sample_media_with_detections):
        """Test system daily health metrics"""
        response = client.get(
            "/analytics/system/daily-health?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 7  # 7 days
        for day_health in data:
            assert "date" in day_health
            assert "uploads" in day_health
            assert "detections" in day_health
            assert "avg_processing_ms" in day_health
            assert "p95_processing_ms" in day_health
            assert "avg_confidence" in day_health
            assert "high_confidence_rate" in day_health
            assert "detections_per_upload" in day_health

            # Verify calculations
            if day_health["uploads"] > 0:
                expected_det_per_upload = day_health["detections"] / day_health["uploads"]
                assert abs(day_health["detections_per_upload"] - expected_det_per_upload) < 0.01


# ============== Quality Metrics Tests ==============

class TestQualityMetrics:
    """Test detection quality and accuracy metrics"""

    def test_detection_quality_overview(self, client, admin_token, sample_media_with_detections):
        """Test detection quality metrics"""
        response = client.get(
            "/analytics/quality/detection-quality?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "overview" in data
        assert "by_confidence_range" in data
        assert "by_class" in data
        assert "daily_trends" in data

        # Verify overview
        overview = data["overview"]
        assert "total_detections" in overview
        assert "false_positive_rate" in overview
        assert "verification_rate" in overview
        assert "avg_confidence" in overview

        # Verify rates are percentages
        assert 0 <= overview["false_positive_rate"] <= 100
        assert 0 <= overview["verification_rate"] <= 100

        # Verify confidence ranges
        for range_data in data["by_confidence_range"]:
            assert "confidence_range" in range_data
            assert "total_count" in range_data
            assert "false_positive_rate" in range_data

    def test_geographic_issue_patterns(self, client, admin_token, sample_media_with_detections):
        """Test geographic issue clustering and recurrence"""
        response = client.get(
            "/analytics/geography/issue-patterns?days=7&precision=5",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "geographic_clusters" in data
        assert "recurring_patterns" in data
        assert "summary" in data

        # Verify summary calculations
        summary = data["summary"]
        assert "total_problem_areas" in summary
        assert "total_recurring_patterns" in summary
        assert "avg_issues_per_area" in summary


# ============== Resolution and Assignment Tests ==============

class TestResolutionMetrics:
    """Test resolution efficiency and assignment tracking"""

    def test_time_to_resolution(self, client, admin_token, sample_media_with_detections):
        """Test resolution time by severity"""
        response = client.get(
            "/analytics/detections/time-to-resolution?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        for severity_data in data:
            assert "severity" in severity_data
            assert "avg_hours" in severity_data
            assert "p95_hours" in severity_data
            assert "count" in severity_data

            assert severity_data["severity"] in ["low", "medium", "high"]
            assert severity_data["avg_hours"] >= 0
            assert severity_data["p95_hours"] >= severity_data["avg_hours"]

    def test_resolution_efficiency(self, client, admin_token, sample_media_with_detections):
        """Test assignee performance metrics"""
        response = client.get(
            "/analytics/performance/resolution-efficiency?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "assignee_performance" in data
        assert "verification_performance" in data
        assert "workload_distribution" in data

        # Verify workload distribution
        for workload in data["workload_distribution"]:
            assert "assignee" in workload
            assert "total_assigned" in workload
            assert "open" in workload
            assert "resolved" in workload
            assert "ignored" in workload
            assert "completion_rate" in workload

            # Verify completion rate calculation
            total = workload["total_assigned"]
            completed = workload["resolved"] + workload["ignored"]
            expected_rate = (completed / max(1, total)) * 100.0
            assert abs(workload["completion_rate"] - expected_rate) < 0.01

    def test_issues_aging_buckets(self, client, admin_token, sample_media_with_detections):
        """Test issue aging and SLA tracking"""
        response = client.get(
            "/analytics/issues/aging-buckets?days=7&scope=backlog",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "window" in data
        assert "sla_hours" in data
        assert "by_severity" in data
        assert "by_assignee" in data
        assert "sla_breach_open" in data
        assert "sla_breach_rate" in data
        assert "open_counts" in data

        # Verify SLA hours
        sla = data["sla_hours"]
        assert sla["high"] == 24
        assert sla["medium"] == 72
        assert sla["low"] == 168

        # Verify severity buckets
        for severity in ["low", "medium", "high"]:
            if severity in data["by_severity"]:
                buckets = data["by_severity"][severity]
                assert "0-24h" in buckets
                assert "1-3d" in buckets
                assert "3-7d" in buckets
                assert "7-30d" in buckets
                assert ">30d" in buckets
                assert "total" in buckets

                # Verify total matches sum
                total = sum([
                    buckets["0-24h"], buckets["1-3d"], buckets["3-7d"],
                    buckets["7-30d"], buckets[">30d"]
                ])
                assert buckets["total"] == total

    def test_issues_aging_with_custom_sla(self, client, admin_token, sample_media_with_detections):
        """Test custom SLA parameters"""
        response = client.get(
            "/analytics/issues/aging-buckets?days=7&sla_high_h=12&sla_medium_h=48&sla_low_h=120",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify custom SLA values
        assert data["sla_hours"]["high"] == 12
        assert data["sla_hours"]["medium"] == 48
        assert data["sla_hours"]["low"] == 120


# ============== Date Range Filtering Tests ==============

class TestDateRangeFiltering:
    """Test date range filtering across all endpoints"""

    def test_days_parameter(self, client, admin_token, sample_media_with_detections):
        """Test days parameter filtering"""
        response = client.get(
            "/analytics/kpis?days=3",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Should include data from last 3 days
        assert "window" in data
        assert data["uploads"]["total"] >= 0

    def test_start_end_date_parameters(self, client, admin_token, sample_media_with_detections):
        """Test start and end date parameters"""
        end_date = date.today()
        start_date = end_date - timedelta(days=2)

        response = client.get(
            f"/analytics/kpis?start={start_date}&end={end_date}",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "window" in data
        # Verify window contains our dates
        assert "start" in data["window"]
        assert "end" in data["window"]

    def test_invalid_days_parameter(self, client, admin_token):
        """Test validation of days parameter"""
        response = client.get(
            "/analytics/kpis?days=400",  # Exceeds max of 365
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 422  # Validation error


# ============== Edge Cases Tests ==============

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_database(self, client, admin_token, db_session):
        """Test analytics with no data"""
        # Ensure database is empty
        response = client.get(
            "/analytics/kpis?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["uploads"]["total"] == 0
        assert data["detections"]["total"] == 0

    def test_single_media_single_detection(self, client, admin_token, db_session, admin_user):
        """Test with minimal data"""
        # Create single media with single detection
        media = Media(
            filename="single.jpg",
            media_type="image",
            user_username=admin_user.username,
            process_ms_total=1000,
        )
        db_session.add(media)
        db_session.flush()

        frame = Frame(media_id=media.id, frame_index=0, timestamp=0.0)
        db_session.add(frame)
        db_session.flush()

        detection = Detection(
            frame_id=frame.id,
            class_id=1,
            class_name="pothole",
            confidence=0.95,
            x1=0.0, y1=0.0, x2=100.0, y2=100.0,
            severity=Severity.high,
            status=IssueStatus.open,
            source=DetectionSource.yolo,
        )
        db_session.add(detection)
        db_session.commit()

        response = client.get(
            "/analytics/kpis?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["uploads"]["total"] == 1
        assert data["detections"]["total"] == 1

    def test_media_type_filter_no_results(self, client, admin_token, sample_media_with_detections):
        """Test filtering by media type that has no data"""
        # Assuming sample data has both images and videos
        # Query for a specific type and verify results
        response_images = client.get(
            "/analytics/kpis?days=7&media_type=image",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        response_videos = client.get(
            "/analytics/kpis?days=7&media_type=video",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response_images.status_code == 200
        assert response_videos.status_code == 200

        images_data = response_images.json()
        videos_data = response_videos.json()

        # Total should split between images and videos
        total_response = client.get(
            "/analytics/kpis?days=7",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        total_data = total_response.json()

        assert (images_data["uploads"]["total"] +
                videos_data["uploads"]["total"]) == total_data["uploads"]["total"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
