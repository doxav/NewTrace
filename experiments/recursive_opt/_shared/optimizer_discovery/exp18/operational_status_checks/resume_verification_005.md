# Fifth-launch pending-slot verification

**PASS: both previously uncompleted slots received their first completed response
at the expected resumed attempt, with no replacement of completed evidence.**
The read-only watch began at 2026-09-11 12:15:43 UTC and final checks completed at
12:25:46.851137 UTC (`1789129546851136903` ns), within the 20-minute limit.

| Study / previously pending slot | Completed attempt | Completion timestamp, ns | Preserved original files |
| --- | ---: | --- | ---: |
| EXP-17 `17009/C/slot_04` | 2 | `1789128923505922433` | 5 / 5 |
| EXP-18 `18037/M/slot_01` | 3 | `1789129425789450957` | 8 / 8 |

The original requests, generation starts, propagated feedback, failed attempts and
EXP-18 memory context retain their exact hashes from
[resume audit 04](resume_audit_04.md). Started and recorded attempt numbers are
contiguous: `[1, 2]` and `[1, 2, 3]`. Each new response identifies the matching
completed attempt and provider ID, and each slot has exactly one response file.
No additional transport retry was needed after the resumed attempts began.
The EXP-17 response was also unchanged between its first authentication and the
final check after EXP-18 completed.

Both responses have `finish_reason=stop`, `parse_status=parsed` and static
`source_status=valid`; the existing response verifier confirmed exact generated
source integrity. **Static source acceptance is not trajectory execution validity.**
No TRAIN/validation/audit trajectory result, comparative metric or candidate behavior
was inspected. These checks establish successful transport resumption and artifact
identity, not optimization performance or scientific completion of either study.

The earlier one EXP-17 and two EXP-18 failed attempts remain byte-identical,
including `possible_remote_completion_or_duplicate_billing=true`. They contain no
reported token/cost/usage fields. The successful new responses do not resolve
whether those failed attempts completed remotely or were billed. Preserve that
unknown usage separately from any reported usage of the completed responses.

| New completed evidence | SHA-256 |
| --- | --- |
| EXP-17 `response.json` | `714d10a7e0f8dad696d8402fe39056e47ff03ea02d3fc9c155cde76302f47db4` |
| EXP-17 `attempt_2.json` | `52fcab7545c8aba49d8626d6b69f2557c0a3d97a48c248bbf479179e66ea6b15` |
| EXP-17 exact source | `f7aef8b74f461e52a9153d469c2179a7d8b09a9c914e0741e48b48856473eb73` |
| EXP-18 `response.json` | `f2eee623f33c80e2d0a36cea83bfb10acf74c91691a85ceea864e8e379bb658d` |
| EXP-18 `attempt_3.json` | `a497ee4e4a321cb91be438c9c1ed77b5597179e81c0d97e6b6471f6dd86b8edb` |
| EXP-18 exact source | `1c1cbcde608f4d38928b2fe9713dceb67b374e2e7108cd9ab202ca694543cb3a` |

This independent check was limited to those two slots. The root operator separately
reports preservation of all 51,222 EXP-17 and 40,894 EXP-18 files inventoried in
incidents 006/007, recorded in [resume_verification_005.json](resume_verification_005.json)
with reported SHA-256
`9ae21d1286ffe807ce2ec1a44f6379a093230275db941d23f77e1d20c6afc642`.
That full inventory check was not repeated here.

No model, provider metadata, network, objective or candidate-execution call was
made; no process was interrupted or restarted, and no frozen source, prior evidence
or commit was changed. Verification used read-only hashing, attempt identity checks
and the existing static response verifier. No regression suite was rerun because
no implementation changed.
