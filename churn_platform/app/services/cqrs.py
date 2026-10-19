from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List


class QueryHandler(ABC):
    """Base class for read-oriented query operations."""

    @abstractmethod
    def handle(self, query: Dict[str, Any]) -> Any:
        raise NotImplementedError


class CommandHandler(ABC):
    """Base class for write-oriented command operations."""

    @abstractmethod
    def handle(self, command: Dict[str, Any]) -> Any:
        raise NotImplementedError


class CustomerQueryHandler(QueryHandler):
    def handle(self, query: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "query": query,
            "result": "Customer query executed successfully.",
        }


class InterventionCommandHandler(CommandHandler):
    def handle(self, command: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "command": command,
            "result": "Intervention command accepted for human review.",
        }
