from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
from typing import Sequence

from .alarms import AlarmRepository
from .config import load_config
from .logging_setup import configure_logging
from .preflight import run_preflight
from .web import export_offline_html


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "system.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rdk-patrol",
        description="RDK S100 四功能统一巡检运行时",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="启动统一检测、录像和只读网页")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument(
        "--data-dir",
        default=os.environ.get("RDK_PATROL_DATA_DIR", ""),
        help="覆盖运行数据根目录；板端通常为 /var/lib/rdk-patrol",
    )
    run.add_argument("--source", choices=("ros2", "video"), default="ros2")
    run.add_argument("--video", default="", help="回放组合双目视频")
    run.add_argument("--topic", default="", help="临时覆盖 ROS 图像话题")
    run.add_argument(
        "--message-type",
        choices=("auto", "sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image"),
        default="auto",
    )
    run.add_argument("--max-frames", type=int, default=0)
    run.add_argument("--duration", type=float, default=0.0)
    run.add_argument("--no-web", action="store_true")
    run.add_argument("--no-recording", action="store_true")

    preflight = commands.add_parser("preflight", help="只读部署预检")
    preflight.add_argument("--config", default=str(DEFAULT_CONFIG))
    preflight.add_argument(
        "--data-dir",
        default=os.environ.get("RDK_PATROL_DATA_DIR", ""),
    )
    preflight.add_argument("--board", action="store_true")
    preflight.add_argument("--json", action="store_true")
    preflight.add_argument(
        "--strict",
        action="store_true",
        help="把现场未标定等警告也作为失败",
    )

    validate = commands.add_parser("validate-config", help="校验 system.yaml")
    validate.add_argument("--config", default=str(DEFAULT_CONFIG))

    ledger = commands.add_parser(
        "export-ledger",
        help="生成可离线双击查看的自包含报警台账",
    )
    ledger.add_argument(
        "--data-dir",
        required=True,
        help="包含 alarms/ 的运行数据目录",
    )
    ledger.add_argument("--output", required=True)
    ledger.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "validate-config":
        config = load_config(args.config)
        print(
            json.dumps(
                {
                    "ok": True,
                    "schema_version": config["schema_version"],
                    "config": str(Path(args.config).resolve()),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "export-ledger":
        return _export_ledger(args)
    if args.command == "run":
        return _run(args)
    raise AssertionError(args.command)


def _preflight(args: argparse.Namespace) -> int:
    report = run_preflight(
        args.config,
        board_checks=bool(args.board),
        data_dir=args.data_dir or None,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        symbols = {"ok": "[ OK ]", "warning": "[WARN]", "error": "[FAIL]"}
        for result in report.results:
            print(
                "{} {:<26} {}".format(
                    symbols[result.status],
                    result.name,
                    result.message,
                )
            )
        print(
            f"\nerrors={report.errors} warnings={report.warnings} "
            f"ok={str(report.ok).lower()}"
        )
    return 0 if report.ok and not (args.strict and report.warnings) else 1


def _export_ledger(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    repository = AlarmRepository(data_dir / "alarms")
    output = Path(args.output).resolve()
    export_offline_html(
        output,
        repository,
        limit=None if int(args.limit) <= 0 else int(args.limit),
    )
    print(str(output))
    return 0


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    configure_logging(
        PROJECT_ROOT,
        data_dir=data_dir,
    )
    if args.source == "video" and not args.video:
        raise SystemExit("--source video requires --video")
    if args.video and args.source != "video":
        args.source = "video"

    # Imported only for the run command: preflight and ledger export remain
    # available on development computers without ROS/HBM board libraries.
    from .factory import build_application

    application = build_application(
        config_path,
        data_dir=data_dir,
        source=args.source,
        video_path=args.video or None,
        topic_override=args.topic or None,
        message_type_override=args.message_type,
        web_enabled_override=False if args.no_web else None,
        recording_enabled_override=False if args.no_recording else None,
    )
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, request_stop)
        except (OSError, ValueError):
            pass
    summary = application.run(
        stop_event=stop_event,
        max_frames=max(0, int(args.max_frames)),
        duration_seconds=max(0.0, float(args.duration)),
    )
    print(
        json.dumps(
            {
                "processed_frames": summary.processed_frames,
                "alarms_saved": summary.alarms_saved,
                "elapsed_seconds": round(summary.elapsed_seconds, 3),
                "reason": summary.reason,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
