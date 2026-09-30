# Seventh-launch pending-slot verification: EXP-18 only

**PASS: `18037/L/slot_00` completed at expected attempt 2 with its original evidence
preserved.** The watch began at 2026-09-11 20:39:18 UTC and final identity checks
passed at 20:48:07.351042 UTC (`1789159687351042446` ns), within 20 minutes.
The response completion timestamp is `1789159552823912015` ns.

All six original files match [resume audit 06](resume_audit_06.md): request,
generation start, propagated feedback, original context, started attempt 1 and
failed attempt 1. The checks passed before and after response completion.
Started/attempt numbering is contiguous `[1, 2]`; the new response matches the
completed attempt's provider ID, and exactly one response representation exists.
No additional transport retry or completed-response replacement occurred.

The existing static response verifier confirmed exact source/response integrity:
`parse_status=parsed`, `source_status=valid`, `finish_reason=stop`.
**Static acceptance is not trajectory validity or optimization performance.**
No trajectory result, candidate behavior or comparative efficacy was inspected.

The failed first attempt retains
`possible_remote_completion_or_duplicate_billing=true` and has no reported token,
usage or cost fields. Its remote completion and billing remain unknown; the new
successful response does not resolve them or justify treating them as zero.

| New completed evidence | SHA-256 |
| --- | --- |
| `response.json` | `c415245a6672d678d45a49b2eff558b9d5e8ee91c6a3179ffb5881e2ced63311` |
| `attempt_2.json` | `c4eb5fd7a0a8fd71ed9fc7250d659263c7595da15ea541dde9d71f8431f542c1` |
| Exact source | `7e414b3a544758c929a746bed91a00828b8d93d4a01fb40f40b4e3a3a53f1931` |

The root operator separately reports that all 64,350 files in incident 010 remained
unchanged after this response. Its [resume_verification_007.json](resume_verification_007.json)
has reported SHA-256
`5cae663fe47bdbf9aa471ca9949b629be95e9f95061367054fb96f974d02df57`.
That whole-run inventory was not repeated here; this independent review covers
only the resumed EXP-18 slot. EXP-17 was not inspected or interfered with.

No objective/candidate/model/network or metadata call, credential load, process
intervention, frozen edit or commit occurred. A tool-output serialization issue
required repeating a read-only identity check; no scientific execution was repeated.
Verification used hashing, attempt/response identity checks and the existing static
response verifier. No regression suite was rerun because no implementation changed.
