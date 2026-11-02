"""Errors raised by model-neutral deployment mechanics."""


class DeploymentContractError(ValueError):
    """A Docker inspection value cannot satisfy the deployment contract."""
