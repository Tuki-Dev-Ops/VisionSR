"""API contract tests.

These run the real engine through the real HTTP layer — no mocks. Mocking the
engine here would test only that FastAPI can route, which was never in doubt; what
is worth pinning down is the contract the frontend is written against, and that a
job actually reaches ``done`` and hands back decodable image bytes.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytestmark = pytest.mark.weights


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def upload(portrait_image) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(portrait_image).save(buffer, format="PNG")
    return buffer.getvalue()


def _post_job(client, upload: bytes, **fields) -> str:
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("portrait.png", upload, "image/png")},
        data=fields,
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def _await_job(client, job_id: str) -> dict:
    """Drain the SSE stream, then read the final state.

    Also the test that the stream *terminates*: if the terminal event were never
    published, this would hang rather than fail — so the iteration is bounded.
    """
    events = []
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")

        for line in stream.iter_lines():
            if not line.startswith("data: "):
                continue
            events.append(json.loads(line.removeprefix("data: ")))
            if len(events) > 500:
                pytest.fail("the SSE stream did not close after the job finished")

    assert events, "no progress events were emitted"
    assert events[-1]["status"] in ("done", "failed"), (
        f"the stream closed on a non-terminal event: {events[-1]}"
    )

    state = client.get(f"/api/v1/jobs/{job_id}").json()
    state["_events"] = events
    return state


# -- read endpoints ---------------------------------------------------------


def test_health(client):
    body = client.get("/api/v1/health").json()

    assert body["status"] == "ok"
    assert "cpu" in body["backends"]
    assert body["version"]


def test_models_lists_installed_weights(client):
    body = client.get("/api/v1/models").json()
    models = {m["id"]: m for m in body["models"]}

    assert models, "the registry is empty"
    assert all(m["installed"] for m in models.values()), "some checkpoints are missing"

    # The frontend renders these directly; a rename would break it silently.
    for model in models.values():
        assert set(model) >= {
            "id",
            "name",
            "task",
            "architecture",
            "scale",
            "content_types",
            "description",
            "installed",
            "backends",
            "precisions",
        }


def test_analyze_reports_what_it_sees(client, upload):
    body = client.post(
        "/api/v1/analyze", files={"file": ("portrait.png", upload, "image/png")}
    ).json()

    assert body["content_type"] == "portrait"
    assert len(body["faces"]) == 1
    assert body["recommended_model"]
    assert body["recommended_face_model"] is not None

    # The fixture is a JPEG-50 of a blurred downscale, so the analyser had better say so.
    assert body["compression_level"] > 0.3
    assert 0.0 <= body["quality_score"] <= 1.0


# -- job lifecycle ----------------------------------------------------------


def test_job_runs_to_completion_and_returns_an_image(client, upload):
    job_id = _post_job(client, upload, scale=4, face_restore="true")
    state = _await_job(client, job_id)

    assert state["status"] == "done", state.get("error")
    assert state["progress"] == 1.0

    result = state["result"]
    assert result["width"] == 512 and result["height"] == 768  # 128x192 at x4
    assert len(result["models_used"]) == 2, "face restoration should have run"
    assert result["tiles"] >= 1

    response = client.get(f"/api/v1/jobs/{job_id}/result")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    image = Image.open(io.BytesIO(response.content))
    assert image.size == (512, 768)


def test_progress_events_advance_through_the_pipeline(client, upload):
    job_id = _post_job(client, upload, scale=2, face_restore="false")
    state = _await_job(client, job_id)

    stages = [e["stage"] for e in state["_events"]]
    assert any("upscal" in s for s in stages), f"no upscaling stage reported: {stages}"

    progress = [e["progress"] for e in state["_events"]]
    assert all(0.0 <= p <= 1.0 for p in progress)


def test_face_restore_can_be_disabled(client, upload):
    job_id = _post_job(client, upload, scale=2, face_restore="false")
    state = _await_job(client, job_id)

    assert state["result"]["models_used"] == [state["result"]["analysis"]["recommended_model"]]


def test_source_is_retrievable_for_the_before_after_view(client, upload):
    job_id = _post_job(client, upload, scale=2)
    _await_job(client, job_id)

    response = client.get(f"/api/v1/jobs/{job_id}/source")
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.content)).size == (128, 192)


def test_delete_removes_the_job(client, upload):
    job_id = _post_job(client, upload, scale=2)
    _await_job(client, job_id)

    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404


# -- failure modes ----------------------------------------------------------


def test_unknown_job_is_404(client):
    assert client.get("/api/v1/jobs/does-not-exist").status_code == 404


def test_result_before_completion_is_409(client, upload):
    job_id = _post_job(client, upload, scale=4)

    # Racy by nature: on a fast machine the job may already be done. Only assert the
    # contract when we actually caught it mid-flight.
    response = client.get(f"/api/v1/jobs/{job_id}/result")
    if response.status_code != 200:
        assert response.status_code == 409

    _await_job(client, job_id)


def test_an_unsupported_scale_is_rejected_with_a_useful_message(client, upload):
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("portrait.png", upload, "image/png")},
        data={"scale": 3},
    )
    assert response.status_code == 422
    assert "2, 4, 8, 16" in response.json()["detail"]


def test_a_non_image_upload_fails_cleanly(client):
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    # Accepted into the queue, then the job itself fails — the decode happens on the
    # worker. Either way the client must get a clear answer, never a hang.
    if response.status_code == 202:
        state = _await_job(client, response.json()["job_id"])
        assert state["status"] == "failed"
        assert state["error"]
    else:
        assert response.status_code == 400


def test_an_empty_upload_is_rejected(client):
    response = client.post(
        "/api/v1/jobs", files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400
