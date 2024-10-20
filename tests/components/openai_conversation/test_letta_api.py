import pytest
from unittest.mock import MagicMock, patch


# Assume HomeAssistant is the main class to be tested
class HomeAssistant:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.is_running = False

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False


@pytest.fixture
def mock_hass():
    """Simple synchronous fixture for HomeAssistant."""
    hass = HomeAssistant(config_dir="/fake/path")
    hass.start()
    yield hass
    hass.stop()


def test_home_assistant_start_stop(mock_hass):
    """Test that HomeAssistant starts and stops correctly."""
    assert mock_hass.is_running == True
    mock_hass.stop()
    assert mock_hass.is_running == False


def test_home_assistant_config_dir(mock_hass):
    """Test the configuration directory is set correctly."""
    assert mock_hass.config_dir == "/fake/path"
