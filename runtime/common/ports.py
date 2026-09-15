"""Probe Linux serving addresses using the TCP listener's restart semantics."""
import ipaddress
import socket


def check_tcp_bind(address, port):
    """Reject active listeners while allowing reuse after a completed connection."""
    family = socket.AF_INET6 if ipaddress.ip_address(address).version == 6 else socket.AF_INET
    with socket.socket(family) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((address, port))
        except OSError as error:
            raise ValueError(f"Serving address is unavailable: {address}:{port}") from error
