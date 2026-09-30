# Sixth-launch pending-slot verification

**PASS for transport resumption and evidence preservation.** Both pending slots
received their first completed response at the expected resumed attempt. The
read-only watch ran from 2026-09-11 17:11:15 UTC to the final identity check at
17:16:59.620961 UTC (`1789147019620961161` ns), within the 20-minute limit.

| Study / previously pending slot | Completed attempt | Completion timestamp, ns | Static source status |
| --- | ---: | --- | --- |
| EXP-17 `17011/C/slot_05` | 3 | `1789146771776849751` | `syntax_error` |
| EXP-18 `18037/PM/slot_10` | 2 | `1789146940999437119` | `valid` |

Both responses have `parse_status=parsed` and `finish_reason=stop`. The EXP-17
response contains an invalid generated program, consumes its completed proposal
slot and must remain preserved without repair or replacement. It is not a further
transport failure. EXP-18 passes static source checks; **this does not establish
valid trajectory execution or optimization performance**. No trajectory result,
comparative metric or candidate behavior was inspected in either study.

All seven original files per study match the hashes authenticated in
[resume audit 05](resume_audit_05.md), including requests, original generation starts,
propagated feedback, prior starts/failures, and EXP-18's original PM memory context
and Pareto decision. The checks passed before and after response completion.
Started/attempt numbering is contiguous: `[1, 2, 3]` and `[1, 2]`. Each completed
response matches its completed attempt/provider ID and is the slot's sole response
file. No additional transport retry was needed after the resumed attempts began.
The EXP-17 completed response remained byte-identical between its initial
verification and the final check after EXP-18 completed.

The earlier two EXP-17 and one EXP-18 failed attempts remain unchanged, including
`possible_remote_completion_or_duplicate_billing=true`. They have no reported
usage/token/cost fields. The new completed responses do not resolve whether those
failed attempts completed remotely or incurred charges; retain that unknown usage
separately from usage reported for the completed responses.

| Completed evidence | SHA-256 |
| --- | --- |
| EXP-17 `response.json` | `edcad2a368ef6f266a9f120a9312faef15e4b84628c60de097ec9a5bf605b0c8` |
| EXP-17 `attempt_3.json` | `146263e9a3ae703250320c79c03a0d09e3f7eee7379c8abdf8445df9efbedb8f` |
| EXP-17 exact source | `8c677034379ba41c5eac341af9cecdd41f22e3b71696da1a6bc031ef0ab9bfd1` |
| EXP-18 `response.json` | `efe3731bf55f0b488dc2df947d12af21bda4d6d5fbd8c22d2b6ab578a8a1958a` |
| EXP-18 `attempt_2.json` | `83f9042f0ded99c00ac29f0e20c079f6a2fe04417d0e5e9be83b8ec30dcd2002` |
| EXP-18 exact source | `a6be05f3e36f6022701c4e12894e7edd9fc6a9453eb1ee8fc277823d62edee5b` |

This independent verification is limited to the two resumed slots. The root
operator owns the separate post-response rehash of all files in incident-008/009;
that whole-run inventory was not repeated here. These checks establish neither
scientific completion of the studies nor superiority of either search procedure.

Verification used read-only hashing, attempt/response identity checks and the
existing static response verifier. No objective/candidate/model/network or metadata
call, credential load, process intervention, frozen edit or commit occurred.
No regression suite was rerun because no implementation changed.

Operator completion update: the root reports that its post-response rehash preserved
all 68,454 EXP-17 and 53,699 EXP-18 files from incident-008/009. Its independently
saved [resume_verification_006.json](resume_verification_006.json) has reported SHA-256
`8f65957b44e87ffcb102cfcd50f1cd4215ce2efa29e20501a8e79f21bd8a7112`.
This full inventory result is attributed to the root and was not repeated here.
