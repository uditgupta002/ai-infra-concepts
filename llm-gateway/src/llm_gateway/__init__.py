"""LLM Gateway - Production-grade multi-provider LLM routing.

This package provides a unified interface for interacting with multiple
LLM providers with intelligent routing, failover, and caching capabilities.
"""

from llm_gateway.gateway import LLMGateway
from llm_gateway.config import GatewaySettings, get_gateway_settings

__version__ = "0.1.0"
__all__ = ["LLMGateway", "GatewaySettings", "get_gateway_settings"]
