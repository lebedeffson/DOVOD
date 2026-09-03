from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .procedure import Graph, action_set


@dataclass(frozen=True)
class AuthorizationCertificate:
    schema: str
    graph_digest: str
    action: int
    prerequisites: tuple[int, ...]
    authorized: bool


def graph_digest(graph: Graph) -> str:
    payload = json.dumps(graph, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def issue_certificate(state: list[int], graph: Graph, action: int) -> AuthorizationCertificate:
    return AuthorizationCertificate(
        schema="dovod-certificate-v1",
        graph_digest=graph_digest(graph),
        action=int(action),
        prerequisites=tuple(sorted(graph.get(int(action), []))),
        authorized=int(action) in action_set(state, graph),
    )


def verify_certificate(state: list[int], graph: Graph, certificate: AuthorizationCertificate) -> bool:
    expected = issue_certificate(state, graph, certificate.action)
    return asdict(expected) == asdict(certificate)
