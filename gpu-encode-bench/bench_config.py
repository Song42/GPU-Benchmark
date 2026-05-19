"""Configuration and constants for the GPU encode benchmark."""

from __future__ import annotations

CODEC_ENCODERS = {
    "h264": "h264_nvenc",
    "hevc": "hevc_nvenc",
    "av1": "av1_nvenc",
}

CODEC_LABELS = tuple(CODEC_ENCODERS.keys())

BITRATE_MATRIX = {
    "uhd": [10, 7],
    "fhd": [5, 3],
}

CQ_VALUE = 23
PRESET = "slow"
PARALLEL_JOBS_DEFAULT = [1, 2, 4, 8, 9, 10, 11, 12, 13]

AXIS_BITRATE = "bitrate"
AXIS_CQ23 = "cq23"

STATUS_SUCCESS = "SUCCESS"
STATUS_FAIL_UNSUPPORTED_ENCODER = "FAIL_UNSUPPORTED_ENCODER"
STATUS_FAIL_PRESET_UNSUPPORTED = "FAIL_PRESET_UNSUPPORTED"
STATUS_FAIL_FFMPEG_ERROR = "FAIL_FFMPEG_ERROR"
STATUS_FAIL_TIMEOUT = "FAIL_TIMEOUT"
STATUS_FAIL_OUTPUT_MISSING = "FAIL_OUTPUT_MISSING"
STATUS_FAIL_VMAF = "FAIL_VMAF"
STATUS_LIMIT_REACHED = "LIMIT_REACHED"
STATUS_SKIPPED_PRECHECK_FAILED = "SKIPPED_PRECHECK_FAILED"

TERMINAL_FAILURE_STATUSES = {
    STATUS_FAIL_UNSUPPORTED_ENCODER,
    STATUS_FAIL_PRESET_UNSUPPORTED,
    STATUS_FAIL_FFMPEG_ERROR,
    STATUS_FAIL_TIMEOUT,
    STATUS_FAIL_OUTPUT_MISSING,
    STATUS_SKIPPED_PRECHECK_FAILED,
}


def bitrate_bufsize_mbps(target_bitrate_mbps: int | float) -> int | float:
    return target_bitrate_mbps * 2


def single_timeout_sec(input_duration_sec: float | None) -> int:
    duration = input_duration_sec or 0
    return int(min(420, max(180, duration * 8)))


def parallel_timeout_sec(input_duration_sec: float | None) -> int:
    duration = input_duration_sec or 0
    return int(min(900, max(300, duration * 15)))


def scaling_verdict(
    scaling_efficiency: float | None,
    status_counts: dict[str, int],
    total_jobs: int,
) -> str:
    failed_jobs = total_jobs - status_counts.get(STATUS_SUCCESS, 0)
    if (
        status_counts.get(STATUS_FAIL_TIMEOUT, 0)
        or status_counts.get(STATUS_FAIL_UNSUPPORTED_ENCODER, 0) == total_jobs
        or failed_jobs >= max(1, total_jobs // 2)
    ):
        return STATUS_LIMIT_REACHED

    if scaling_efficiency is None:
        return ""
    if scaling_efficiency >= 0.75:
        return "GOOD_SCALE"
    if scaling_efficiency >= 0.40:
        return "PARTIAL_SCALE"
    return "POOR_SCALE"
