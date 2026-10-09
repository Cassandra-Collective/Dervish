# Where the quantum half of the project lives

Dervish is a classical optimizer; its connection to quantum computing is in the
mathematics it reuses, and this file says exactly how far that connection goes.

## 1 · Rotations as gates

A Dervish step moves the frame `Q` to `Q R` with `R` orthogonal
(`docs/DESIGN.md` section 3). Nothing about the rows' lengths changes, only
their directions, exactly as a quantum gate changes a state's direction in
Hilbert space and never its norm. Unitary recurrent networks (Arjovsky, Shah
and Bengio, 2016) made this analogy first and for the same reason: a weight
matrix that is kept unitary cannot explode or vanish along any direction. The
spectrum `s` is the part of `W` that has no quantum analogue, and it is also
the only part the optimizer must police; keeping those two roles in separate
variables is the whole design.

## 2 · The nearest unitary is a polar factor

In a quantum noise simulator, calibration drift means a gate that should be unitary is
measured as a matrix that is not. The standard repair, projecting a matrix onto
the nearest unitary in Frobenius norm, is its polar factor `U` from
`A = U P`. That is the same object Muon's and Dervish's Newton-Schulz
iterations compute, and `dervish.polar` computes it for a complex gate without
any complex arithmetic by working on the realification

    rho(A) = [[Re A, -Im A], [Im A, Re A]]

because `rho` respects products and transposes (`rho(A)^T = rho(A^H)`), so the
polar factor of `rho(A)` is `rho(U)`. `examples/nearest_unitary.py` drifts a
CNOT by an interval-valued perturbation (a small random matrix whose norm is
drawn from an interval, in the spirit of the engine's interval-valued noise
ensembles), recovers the nearest unitary with the optimizer's own routine, and
reports the process fidelity before and after. Run it:

    python examples/nearest_unitary.py

## 3 · Intervals

Dervish's spectrum lives in a hard interval `[scale_floor, scale_cap]`. Every
model it produces has all hidden singular values inside that interval, which is
the kind of set-valued statement an interval-valued noise model makes about noise
parameters. This is an analogy, not a theorem that connects the two systems,
and it is stated here so that nobody later mistakes it for one.

## 4 · What is not here

No quantum circuit is simulated, no Qiskit code is called, and no claim is
made that the optimizer is "quantum". The honest summary: one numerical
routine (the polar factor by Newton-Schulz) does useful work in both quantum
noise modelling and optimisation, and this project puts it to work in the second.
