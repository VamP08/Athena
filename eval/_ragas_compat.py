"""
eval/_ragas_compat.py
Compatibility shim — import this BEFORE importing ragas.

Why this exists (2026-07):
  ragas 0.4.3 does a top-level `from langchain_community.chat_models.vertexai
  import ChatVertexAI` (and `from langchain_community.llms import VertexAI`) in
  ragas/llms/base.py. langchain-community was sunset in May 2026 and its 0.4.x
  releases removed those modules, so *any* `import ragas` crashes with
  ModuleNotFoundError — even though Athena never touches VertexAI.

  This shim registers stub modules for exactly the missing import targets so the
  ragas import succeeds. The stubs are inert placeholder classes; ragas only uses
  these symbols in isinstance checks for Vertex-specific behaviour we never hit.

Remove when ragas ships a release compatible with the post-sunset ecosystem
(track: https://github.com/vibrantlabsai/ragas).
"""

import importlib
import sys
import types

_STUBS = {
    "langchain_community": [],
    "langchain_community.chat_models": [],
    "langchain_community.chat_models.vertexai": ["ChatVertexAI"],
    "langchain_community.llms": ["VertexAI"],
}


def apply() -> None:
    for module_name, attrs in _STUBS.items():
        try:
            module = importlib.import_module(module_name)
            # Module imports fine — only patch attributes that are missing
            for attr in attrs:
                if not hasattr(module, attr):
                    setattr(module, attr, type(attr, (), {}))
        except Exception:
            stub = types.ModuleType(module_name)
            for attr in attrs:
                setattr(stub, attr, type(attr, (), {}))
            sys.modules[module_name] = stub


apply()
