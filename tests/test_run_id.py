import datetime
import re

from smart_inspection.io.run_id import get_run_id


# Without a tag, the run id is only the timestamp YYYYMMDD_HHMMSS.
def test_get_run_id_without_tag_is_a_timestamp():
    assert re.fullmatch(r"\d{8}_\d{6}", get_run_id())


# With a tag, the run id is "<tag>_<timestamp>".
def test_get_run_id_with_tag_is_prefixed():
    assert re.fullmatch(r"my_tag_\d{8}_\d{6}", get_run_id(tag="my_tag"))


# An empty tag behaves like no tag (no leading underscore).
def test_get_run_id_with_empty_tag_is_a_timestamp():
    assert re.fullmatch(r"\d{8}_\d{6}", get_run_id(tag=""))


# The timestamp part is a real date/time that can be parsed back.
def test_get_run_id_timestamp_is_a_valid_datetime():
    timestamp = get_run_id()

    parsed = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

    assert abs((datetime.datetime.now() - parsed).total_seconds()) < 5
