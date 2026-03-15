"""
Video and telemetry endpoints.

Provides the MJPEG camera stream, raw angle telemetry, and a health probe.
All computer-vision state is owned by the CVPipeline singleton; these
endpoints are pure read-through with no side effects.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from cv.pipeline import pipeline

router = APIRouter(tags=["Video"])


@router.get("/video", summary="MJPEG pose stream")
async def video_stream():
    """Streams the live camera feed with MediaPipe skeleton and safety overlays.

    Returns:
        A ``multipart/x-mixed-replace`` streaming response. Clients should
        render this in an ``<img>`` tag or equivalent video consumer.
        Frame rate is capped at ~15 fps (every other camera frame is
        processed by MediaPipe to bound CPU usage).
    """
    return StreamingResponse(
        pipeline.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/angles", summary="Current joint angles snapshot")
async def get_angles():
    """Returns the most recently computed joint angles from the CV pipeline.

    Returns:
        Dict mapping joint name → angle in degrees. Returns an empty dict
        if no pose has been detected since server start.
    """
    return pipeline.get_angles()


@router.get("/health", summary="Liveness probe")
async def health():
    """Liveness probe for load balancers and container orchestrators.

    Returns:
        ``{"status": "ok"}`` with HTTP 200.
    """
    return {"status": "ok"}
