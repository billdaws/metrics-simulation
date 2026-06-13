from unittest.mock import MagicMock, patch

from metrics_simulation.graphite import Datapoint, Series, query


def _mock_response(data: list) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = data
    mock.raise_for_status.return_value = None
    return mock


def test_query_parses_single_series() -> None:
    payload = [{"target": "sim.test", "datapoints": [[0.5, 1000], [0.8, 1060]]}]
    with patch("metrics_simulation.graphite.requests.get", return_value=_mock_response(payload)):
        result = query("sim.test", from_time=1000, until_time=1060)

    assert len(result) == 1
    assert result[0].target == "sim.test"
    assert result[0].datapoints == [Datapoint(value=0.5, timestamp=1000), Datapoint(value=0.8, timestamp=1060)]


def test_query_handles_null_datapoints() -> None:
    payload = [{"target": "sim.test", "datapoints": [[None, 1000], [0.3, 1060]]}]
    with patch("metrics_simulation.graphite.requests.get", return_value=_mock_response(payload)):
        result = query("sim.test", from_time=1000, until_time=1060)

    assert result[0].datapoints[0].value is None
    assert result[0].datapoints[1].value == 0.3


def test_query_returns_empty_list_when_no_series() -> None:
    with patch("metrics_simulation.graphite.requests.get", return_value=_mock_response([])):
        result = query("sim.nothing", from_time=1000, until_time=1060)

    assert result == []


def test_query_passes_correct_params() -> None:
    mock = _mock_response([])
    with patch("metrics_simulation.graphite.requests.get", return_value=mock) as mock_get:
        query("sim.test", from_time=1000, until_time=2000, base_url="http://example:8080")

    _, kwargs = mock_get.call_args
    params = kwargs["params"]
    assert params["target"] == "sim.test"
    assert params["from"] == 1000
    assert params["until"] == 2000
    assert params["format"] == "json"
