from churn_platform.app.config import AppConfig
from churn_platform.app.services.cqrs import QueryHandler, CommandHandler


def test_app_config_exposes_default_settings():
    config = AppConfig()
    assert config.app_name == "AI-Powered Customer Churn Prediction Platform"
    assert config.python_version == "3.11"
    assert config.model_version == "v1.0.0"


def test_cqrs_base_classes_are_abstract_contracts():
    assert issubclass(QueryHandler, object)
    assert issubclass(CommandHandler, object)
