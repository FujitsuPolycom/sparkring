"""Actionable installer input requests, shared by terminal and JSON callers."""


class NeedsInput(ValueError):
    def __init__(self, message, *, field, details=None):
        super().__init__(message)
        self.field, self.details = field, details or {}

    def document(self):
        return {"state": "needs_input", "field": self.field, "message": str(self), "details": self.details}
