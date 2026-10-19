import sys

sys.path.insert(0, r"C:\Users\tushar.anand\Downloads\MCA Project")

from churn_platform.app.config import AppConfig
from churn_platform.app.services.cqrs import CommandHandler, QueryHandler

config = AppConfig()
assert config.app_name == "AI-Powered Customer Churn Prediction Platform"
assert config.python_version == "3.11"
assert config.model_version == "v1.0.0"
assert issubclass(QueryHandler, object)
assert issubclass(CommandHandler, object)
print("contract validation passed")
