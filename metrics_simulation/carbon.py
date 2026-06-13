import socket
from collections.abc import Sequence


def write_series(
    metric: str,
    points: Sequence[tuple[float, int]],
    host: str = "localhost",
    port: int = 2003,
) -> None:
    payload = "".join(f"{metric} {value} {timestamp}\n" for value, timestamp in points)
    with socket.create_connection((host, port)) as sock:
        sock.sendall(payload.encode())


def write_batch(
    series: Sequence[tuple[str, Sequence[tuple[float, int]]]],
    host: str = "localhost",
    port: int = 2003,
) -> None:
    lines: list[str] = []
    for metric, points in series:
        for value, timestamp in points:
            lines.append(f"{metric} {value} {timestamp}\n")
    with socket.create_connection((host, port)) as sock:
        sock.sendall("".join(lines).encode())
