from dataclasses import dataclass

import requests


@dataclass
class Datapoint:
    value: float | None
    timestamp: int


@dataclass
class Series:
    target: str
    datapoints: list[Datapoint]


def query(
    target: str,
    from_time: int,
    until_time: int,
    base_url: str = "http://localhost:8080",
) -> list[Series]:
    resp = requests.get(
        f"{base_url}/render",
        params={
            "target": target,
            "from": from_time,
            "until": until_time,
            "format": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [
        Series(
            target=item["target"],
            datapoints=[Datapoint(value=v, timestamp=t) for v, t in item["datapoints"]],
        )
        for item in resp.json()
    ]
