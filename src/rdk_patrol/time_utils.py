from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


try:
    BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    # Beijing time has no daylight-saving transition.  This fallback keeps
    # configuration checks and offline ledger generation working on minimal
    # Windows/Python installations that do not bundle the IANA tz database.
    BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
BEIJING_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def beijing_datetime(epoch_seconds: float | None = None) -> datetime:
    if epoch_seconds is None:
        return datetime.now(BEIJING_TIMEZONE)
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).astimezone(
        BEIJING_TIMEZONE
    )


def format_beijing_time(epoch_seconds: float | None = None) -> str:
    return beijing_datetime(epoch_seconds).strftime(BEIJING_TIME_FORMAT)
