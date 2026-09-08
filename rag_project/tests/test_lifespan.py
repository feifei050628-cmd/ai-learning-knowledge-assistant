from fastapi.testclient import TestClient

from rag_project.api import app


class LifespanFakePipeline:
    pass


def test_lifespan_loads_and_releases_pipeline(
    monkeypatch,
):
    monkeypatch.setattr(
        "rag_project.api.RAGPipeline",
        LifespanFakePipeline,
    )

    with TestClient(app) as client:
        assert isinstance(
            app.state.pipeline,
            LifespanFakePipeline,
        )

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["pipeline_loaded"] is True

    assert app.state.pipeline is None
