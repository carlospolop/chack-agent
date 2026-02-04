# chack-tools

Reusable tools package extracted from Chack so multiple bots can share the same tooling.

## Install

```bash
pip install chack-tools
```

or from GitHub:

```bash
pip install "git+https://github.com/carlospolop/chack-tools.git@main"
```

## Usage

```python
from chack_tools import Toolset
from chack_tools.config import ToolsConfig

tools = Toolset(ToolsConfig(), tool_profile="telegram").tools
```

## Publishing

Publishing to PyPI is automated by GitHub Actions on pushes to `master`.
Configure these repo secrets:

- `PYPI_API_TOKEN`: PyPI API token for this package.
