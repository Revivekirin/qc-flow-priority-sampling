from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.acfql import ACFQLAgent
from agents.acfql_logging import ACFQLAgent_LOG

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acfql_ptr=ACFQLAgent,
    acfql_log=ACFQLAgent_LOG,
)
