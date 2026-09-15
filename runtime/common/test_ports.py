"""Linux socket states distinguish active listeners from completed connections."""
import errno
import socket
import sys

import pytest

from runtime.common.ports import check_tcp_bind

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux serving socket semantics")


def test_active_listener_remains_unavailable():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(ValueError, match="Serving address is unavailable"):
            check_tcp_bind(*listener.getsockname())


def test_completed_connection_does_not_block_a_restarted_listener():
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        address = listener.getsockname()
        client.settimeout(5)
        client.connect(address)
        server, _ = listener.accept()
        with server:
            server.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b""
            client.close()
    with socket.socket() as plain:
        with pytest.raises(OSError) as blocked:
            plain.bind(address)
        assert blocked.value.errno == errno.EADDRINUSE
    check_tcp_bind(*address)
