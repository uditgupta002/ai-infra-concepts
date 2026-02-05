# Shared Infrastructure

Common utilities and abstractions used across all modules in this repository.

## Components

- **config/** - Configuration management with Pydantic Settings
- **logging/** - Structured JSON logging with correlation IDs
- **models/** - Shared Pydantic models and base classes

## Usage

```python
from shared.config import get_settings
from shared.logging import get_logger, with_correlation_id
from shared.models import BaseRequest, BaseResponse
```
