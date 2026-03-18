"""Tests for Agentsmith."""
from src.core import Agentsmith
def test_init(): assert Agentsmith().get_stats()["ops"] == 0
def test_op(): c = Agentsmith(); c.detect(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Agentsmith(); [c.detect() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Agentsmith(); c.detect(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Agentsmith(); r = c.detect(); assert r["service"] == "agentsmith"
