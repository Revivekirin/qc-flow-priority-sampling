from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.acfql import ACFQLAgent
from agents.acfql_pars import ACFQLAgentPARS

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acfql_ptr=ACFQLAgent,
    acfql_pars=ACFQLAgentPARS,
)
