import pytest

from ai_augmented.research_assistant.pipeline import ResearchAssistant


@pytest.fixture(scope="module")
def assistant():
    return ResearchAssistant()


def test_relevant_query_is_grounded(assistant):
    result = assistant.ask("What is driving Apple's services margin?")
    assert result.grounded
    assert len(result.citations) > 0
    assert "aapl" in result.citations[0].lower()


def test_out_of_scope_query_triggers_fallback(assistant):
    result = assistant.ask("Tell me about cryptocurrency regulation")
    assert not result.grounded
    assert result.citations == []


def test_weather_query_is_out_of_scope(assistant):
    result = assistant.ask("What's the weather like in Singapore today")
    assert not result.grounded


def test_macro_query_retrieves_macro_notes(assistant):
    result = assistant.ask("What has driven recent market volatility?")
    assert result.grounded
    assert any("macro" in c.lower() or "volatility" in c.lower() for c in result.citations)


def test_mock_mode_never_calls_live_api(assistant):
    result = assistant.ask("What is the outlook for Google advertising revenue?", mode="mock")
    assert result.mode_used == "mock"
    assert "mock mode" in result.answer.lower()
