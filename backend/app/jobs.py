"""In-process job queue.

Inference is blocking and GPU-bound, which forces two things on any async server:

1. **It must not run on the event loop.** A 20-second RRDBNet pass on the loop
   thread would freeze every other request, including the progress polls for the
   job that is running. So work happens in a thread pool.

2. **It must not run concurrently with itself.** Two 4x passes on a 4GB card do not
   fit; they would OOM, or thrash the tiler down to 64px tiles and take longer than
   running in sequence. So the executor has exactly one worker and jobs queue.

This is the local-first shape of the "workers + queue" in the architecture. The
public surface (submit -> poll/stream -> fetch result) is deliberately the same one
a RabbitMQ-backed cloud deployment would expose, so swapping the implementation
does not change the API or the frontend.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from visionsr.core.errors import VisionSRError
from visionsr.core.types import EnhanceOptions, EnhanceResult

from .schemas import JobEvent, JobState, JobStatus, ResultSchema, result_to_schema

log = logging.getLogger(__name__)

#: Finished jobs hold their result bytes in memory. Evict them after this long so a
#: long-running server does not grow without bound.
JOB_TTL_SECONDS = 30 * 60


@dataclass
class Job:
    id: str
    options: EnhanceOptions
    source: bytes
    filename: str

    status: JobStatus = JobStatus.QUEUED
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None

    result: ResultSchema | None = None
    output: bytes | None = None
    media_type: str = "image/png"

    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    #: Subscribers to the SSE stream. A list, not a single queue: the UI may have
    #: several tabs open on one job.
    listeners: list[asyncio.Queue[JobEvent]] = field(default_factory=list)

    def to_state(self) -> JobState:
        return JobState(
            job_id=self.id,
            status=self.status,
            stage=self.stage,
            progress=round(self.progress, 4),
            error=self.error,
            result=self.result,
        )

    def to_event(self) -> JobEvent:
        return JobEvent(
            status=self.status,
            stage=self.stage,
            progress=round(self.progress, 4),
            error=self.error,
        )


class JobManager:
    """Owns the job table and the single inference worker."""

    def __init__(self, max_workers: int = 1) -> None:
        # One worker. See the module docstring — this is a correctness constraint on
        # a small card, not a tuning choice.
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="infer")
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    async def submit(self, source: bytes, filename: str, options: EnhanceOptions) -> Job:
        job = Job(id=uuid.uuid4().hex, options=options, source=source, filename=filename)

        async with self._lock:
            self._evict_expired()
            self._jobs[job.id] = job

        # The task drives the job to completion and updates it in place; callers poll
        # or stream. But the event loop only holds a *weak* reference to a task, so a
        # fire-and-forget `create_task(...)` whose return value is dropped can be
        # garbage-collected mid-flight — the job would simply stop, with no error and
        # no terminal event, leaving the client's progress bar frozen forever. Holding
        # a strong reference until it finishes is what prevents that.
        task = asyncio.create_task(self._run(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        log.info("job %s queued (%s, %d KB)", job.id, filename, len(source) // 1024)
        return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def delete(self, job_id: str) -> bool:
        async with self._lock:
            return self._jobs.pop(job_id, None) is not None

    async def stream(self, job: Job) -> AsyncIterator[JobEvent]:
        """Yield progress events until the job reaches a terminal state."""
        queue: asyncio.Queue[JobEvent] = asyncio.Queue()
        job.listeners.append(queue)

        try:
            # Send the current state immediately: a client that connects after the
            # job finished must still get an answer rather than hanging.
            yield job.to_event()
            if job.status.is_terminal:
                return

            while True:
                event = await queue.get()
                yield event
                if event.status.is_terminal:
                    return
        finally:
            if queue in job.listeners:
                job.listeners.remove(queue)

    async def _run(self, job: Job) -> None:
        loop = asyncio.get_running_loop()

        def on_progress(stage: str, fraction: float) -> None:
            """Called from the worker thread — must hop back to the loop."""
            job.stage = stage
            job.progress = fraction
            loop.call_soon_threadsafe(self._publish, job)

        try:
            job.status = JobStatus.RUNNING
            self._publish(job)

            result, output, media_type = await loop.run_in_executor(
                self._executor, _enhance_blocking, job, on_progress
            )

            job.result = result_to_schema(result)
            job.output = output
            job.media_type = media_type
            job.status = JobStatus.DONE
            job.stage = "done"
            job.progress = 1.0

            log.info(
                "job %s done in %.1fs (%s)",
                job.id,
                result.total_duration_ms / 1000,
                " + ".join(r.model_id for r in result.runs),
            )

        except VisionSRError as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            log.warning("job %s failed: %s", job.id, exc)

        except Exception as exc:
            job.status = JobStatus.FAILED
            # Do not leak internals to the client, but do record them.
            job.error = "Internal error during enhancement."
            log.exception("job %s crashed: %s", job.id, exc)

        finally:
            job.finished_at = time.monotonic()
            # Release the input as soon as the run is over — a batch of 20MB uploads
            # would otherwise sit in memory for the whole TTL.
            if job.status == JobStatus.FAILED:
                job.source = b""
            self._publish(job)

    def _publish(self, job: Job) -> None:
        event = job.to_event()
        for queue in list(job.listeners):
            # put_nowait, never await: a slow or dead SSE client must not be able to
            # stall the inference worker.
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                log.debug("dropping progress event for a saturated listener")

    def _evict_expired(self) -> None:
        now = time.monotonic()
        stale = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None and now - job.finished_at > JOB_TTL_SECONDS
        ]
        for job_id in stale:
            del self._jobs[job_id]
        if stale:
            log.debug("evicted %d expired job(s)", len(stale))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _enhance_blocking(
    job: Job, on_progress: Callable[[str, float], None]
) -> tuple[EnhanceResult, bytes, str]:
    """The actual work. Runs on a worker thread."""
    from visionsr.inference.engine import get_engine
    from visionsr.preprocessing.io import encode_image, load_image

    image, metadata = load_image(job.source)
    result = get_engine().enhance(image, job.options, progress=on_progress, metadata=metadata)

    fmt = job.options.output_format
    output = encode_image(
        result.image,
        format=fmt,
        quality=job.options.output_quality,
        metadata=metadata if job.options.preserve_exif else None,
    )

    return result, output, f"image/{'jpeg' if fmt == 'jpeg' else fmt}"


_manager: JobManager | None = None


def get_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
