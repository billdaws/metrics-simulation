import atexit
import pathlib
import time
import uuid
from types import TracebackType

import docker
import docker.errors
import docker.models.containers
import requests

_IMAGE = "graphiteapp/graphite-statsd:latest"
_LABEL = "metrics-simulation.role=graphite"
_READY_TIMEOUT_SEC = 30
_CARBON_CONF = pathlib.Path(__file__).parent / "graphite" / "carbon.conf"


class GraphiteServer:
    """Manages a Graphite container for simulation use.

    Writes go through Carbon (TCP) and queries go through the render API (HTTP),
    matching the production data path exactly. The container is started and
    stopped programmatically rather than via docker-compose, so the full data
    lifecycle is under Python's control. Call reset() between simulation runs
    to stop the container, discard all whisper data, and start fresh — no
    stale metrics from previous runs.

    Each instance gets a unique container name to avoid naming conflicts. A
    shared label lets start() find and remove orphaned containers left behind
    by previous sessions that exited uncleanly. An atexit handler covers
    graceful shutdown (e.g. kernel restart).

    Docker is not contacted until start() is called, so instances can be
    created freely without a running daemon (e.g. in tests).
    """

    def __init__(self, http_port: int = 8080, carbon_port: int = 2003) -> None:
        self.http_port = http_port
        self.carbon_port = carbon_port
        self._name = f"metrics-simulation-graphite-{uuid.uuid4().hex[:8]}"
        self._container: docker.models.containers.Container | None = None

    @property
    def graphite_url(self) -> str:
        return f"http://localhost:{self.http_port}"

    @property
    def carbon_host(self) -> str:
        return "localhost"

    def start(self) -> None:
        client = docker.from_env()
        self._purge_orphans(client)
        self._container = client.containers.run(
            _IMAGE,
            name=self._name,
            labels={"metrics-simulation.role": "graphite"},
            ports={
                "80/tcp": self.http_port,
                "2003/tcp": self.carbon_port,
            },
            volumes={
                str(_CARBON_CONF): {
                    "bind": "/opt/graphite/conf/carbon.conf",
                    "mode": "ro",
                },
            },
            detach=True,
        )
        atexit.register(self.stop)
        self._wait_ready()

    def stop(self) -> None:
        if self._container is not None:
            try:
                self._container.remove(force=True, v=True)
            except Exception:
                pass
            self._container = None

    def reset(self) -> None:
        """Stop, discard all stored data, and start fresh."""
        self.stop()
        self.start()

    def _purge_orphans(self, client: docker.DockerClient) -> None:
        for container in client.containers.list(
            all=True, filters={"label": _LABEL}
        ):
            try:
                container.remove(force=True, v=True)
            except Exception:
                pass

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + _READY_TIMEOUT_SEC
        while time.monotonic() < deadline:
            try:
                resp = requests.get(
                    f"{self.graphite_url}/render",
                    params={"target": "warmup", "format": "json"},
                    timeout=1,
                )
                if resp.status_code in (200, 400):
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError(
            f"Graphite did not become ready within {_READY_TIMEOUT_SEC}s. "
            "Check that Docker is running and the port is not already in use."
        )

    def __enter__(self) -> "GraphiteServer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()
