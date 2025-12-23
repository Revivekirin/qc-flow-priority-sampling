from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.acfql import ACFQLAgent
from agents.pars import PARSACFQLAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acfql_ptr=ACFQLAgent,
    pars=PARSACFQLAgent,
)
