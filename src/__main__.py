"""CLI for agentsmith."""
import sys, json, argparse
from .core import Agentsmith

def main():
    parser = argparse.ArgumentParser(description="The Agent Smith Problem — Detecting emergent self-preservation and self-replication drives in autonomous AI agents.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Agentsmith()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.detect(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"agentsmith v0.1.0 — The Agent Smith Problem — Detecting emergent self-preservation and self-replication drives in autonomous AI agents.")

if __name__ == "__main__":
    main()
