
import os
import pytest
import requests
from unittest.mock import patch, mock_open
from docker.python_recover_data.recover_data import download_and_save_file

@pytest.fixture
def mock_response():
    class MockResponse:
        def __init__(self, content, status_code):
            self.content = content
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code != 200:
                raise requests.exceptions.HTTPError(f"{self.status_code} Error")

    return MockResponse

@patch("python_recover_data.recover_data.requests.get")
@patch("python_recover_data.recover_data.open", new_callable=mock_open)
@patch("python_recover_data.recover_data.os.makedirs")
def test_download_and_save_file(mock_makedirs, mock_open, mock_get, mock_response):
    url = "http://example.com/"
    raw_data_relative_path = "/fake/path"
    filenames = ['links.csv', 'movies.csv', 'ratings.csv']

    mock_get.side_effect = [mock_response(b"file content", 200) for _ in filenames]

    download_and_save_file(url, raw_data_relative_path)

    mock_makedirs.assert_called_once_with(raw_data_relative_path, exist_ok=True)
    assert mock_get.call_count == len(filenames)
    assert mock_open.call_count == len(filenames)
    for filename in filenames:
        mock_open.assert_any_call(os.path.join(raw_data_relative_path, filename), 'wb')