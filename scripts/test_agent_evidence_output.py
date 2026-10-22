from __future__ import annotations

from pprint import pprint

from phishing_detection.agents.base import AgentContext
from phishing_detection.agents.email_agent import EmailAnalysisAgent
from phishing_detection.agents.sender_agent import SenderAnalysisAgent


def main() -> None:
    email_text = """
Urgent Account Verification Required

Dear Customer,

We detected unusual activity on your account.

For your security, you must verify your account immediately.
Failure to complete verification within 24 hours may result
in temporary account suspension.

Please verify your account immediately.

Thank you,
Account Security Team
""".strip()

    context = AgentContext(
        input_text=email_text,
        sender="security@paypa1-support.com",
        metadata={
            "receiver": "user@example.com",
            "frontend": False,
            "verification": "agent_evidence_output",
        },
    )

    print("=" * 80)
    print("SPECIALIZED AGENT EXECUTION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # EMAIL ANALYSIS AGENT
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EMAIL ANALYSIS AGENT")
    print("=" * 80)

    email_agent = EmailAnalysisAgent()
    email_result = email_agent.run(context)

    print("\nAgentResult:")
    pprint(email_result.to_dict())

    print("\nEvidence:")
    for evidence in email_result.evidence:
        pprint(evidence.to_dict())

    # ------------------------------------------------------------------
    # SENDER ANALYSIS AGENT
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SENDER ANALYSIS AGENT")
    print("=" * 80)

    sender_agent = SenderAnalysisAgent()
    sender_result = sender_agent.run(context)

    print("\nAgentResult:")
    pprint(sender_result.to_dict())

    print("\nEvidence:")
    for evidence in sender_result.evidence:
        pprint(evidence.to_dict())

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRUCTURED EVIDENCE SUMMARY")
    print("=" * 80)

    print(f"\nEmail Agent:")
    print(f"  Label              : {email_result.label_name}")
    print(f"  Phishing Probability: {email_result.phishing_probability:.4f}")
    print(f"  Confidence         : {email_result.confidence:.4f}")
    print(f"  Uncertainty        : {email_result.uncertainty:.4f}")
    print(f"  Evidence Count     : {len(email_result.evidence)}")

    print(f"\nSender Agent:")
    print(f"  Label              : {sender_result.label_name}")
    print(f"  Phishing Probability: {sender_result.phishing_probability:.4f}")
    print(f"  Confidence         : {sender_result.confidence:.4f}")
    print(f"  Uncertainty        : {sender_result.uncertainty:.4f}")
    print(f"  Evidence Count     : {len(sender_result.evidence)}")

    print("\n" + "=" * 80)
    print("AGENT EVIDENCE OUTPUT VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()