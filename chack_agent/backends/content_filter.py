from __future__ import annotations


class ProviderContentFilterError(RuntimeError):
    """The provider's content filter stopped the model's response.

    Its message always contains ``content_filter`` so callers can tell this
    policy refusal apart from an empty answer.
    """

    def __init__(self, detail: str = "finish_reason=content_filter") -> None:
        super().__init__(f"Provider content filter stopped the response ({detail})")
