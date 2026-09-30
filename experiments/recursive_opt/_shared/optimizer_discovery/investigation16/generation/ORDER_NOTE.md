# G1 request-order clarification (before completion of the first block)

The frozen request list is authoritative and remains unchanged. Inspection of its
order shows that the implementation counterbalances cap order **between contexts**:
I always runs 8000 then 32000; L always runs 32000 then 8000. Context order alternates
between blocks. Thus each cap runs first in six of the twelve context/block pairs.

The protocol's phrase “alternate cap-first order within each context/block” could
be read as alternating within the same context across blocks. That stronger
counterbalance is not implemented. This clarification changes no request, value,
or decision. It was recorded after the first I/8000 response (empty, length) and
while its I/32000 partner was in flight, before examining any paired cap result.

Consequences: pooled cap comparisons have balanced first/second ordering across
contexts, but within-context estimates remain confounded with request order.
Upstream routing is stochastic and receipts must be reported. G1 is a diagnostic,
not a proof that cap alone explains EXP-15's relative A2 performance. Later
factorial studies should balance order within each context as well.
