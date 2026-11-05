/-
  Mechanised definitions for "Explicit Positional Encodings Turn Length
  Generalisation into Lookup".

  What is formalised here: the definitions of *positional rule*,
  *length-coupled*, *length-decoupled* and *position-free*, and a
  machine-checked classification of the five tasks in the paper.

  What is NOT formalised here: the empirical claims (the cliff, where it sits,
  the padded protocol). Those are measurements, not theorems, and no proof
  assistant can discharge them.

  The formalisation is deliberately dependency-free: core Lean 4 only, no
  Mathlib, so it builds in seconds on any machine and in CI.
-/

namespace PositionalLookup

/-- The five tasks of the paper. -/
inductive Task where
  | copy
  | prefixXor
  | sort
  | swapHalves
  | reverse
  deriving DecidableEq, Repr

/-- The positional rule `ρ(i, L)`: the input positions that output token `i`
    reads, given that the sequence has length `L`.

    `sort` has no positional rule (which input supplies output `i` is decided by
    the values, not by position); it is represented by the empty list, and
    `sort_positionFree` below records that this is deliberate. -/
def rule : Task → Nat → Nat → List Nat
  | .copy,       i, _ => [i]
  | .prefixXor,  i, _ => List.range (i + 1)
  | .sort,       _, _ => []
  | .swapHalves, i, L => [(i + L / 2) % L]
  | .reverse,    i, L => [L - 1 - i]

/-- A task is *length-coupled* when its rule can differ between two lengths:
    Definition 2 of the paper, existential form. -/
def Coupled (t : Task) : Prop :=
  ∃ i L L', rule t i L ≠ rule t i L'

/-- A task is *length-decoupled* when its rule is the same at every length:
    the universal form. -/
def Decoupled (t : Task) : Prop :=
  ∀ i L L', rule t i L = rule t i L'

/-- A task is *position-free* when it has no positional rule at all. -/
def PositionFree (t : Task) : Prop :=
  ∀ i L, rule t i L = []

/-! ### Decoupled tasks: the rule never mentions `L` -/

theorem copy_decoupled : Decoupled .copy := by
  intro i L L'
  rfl

theorem prefixXor_decoupled : Decoupled .prefixXor := by
  intro i L L'
  rfl

/-! ### Coupled tasks: two lengths give different answers for the same index -/

theorem reverse_coupled : Coupled .reverse := by
  refine ⟨0, 4, 6, ?_⟩
  decide

theorem swapHalves_coupled : Coupled .swapHalves := by
  refine ⟨0, 4, 6, ?_⟩
  decide

/-! ### Position-free task -/

theorem sort_positionFree : PositionFree .sort := by
  intro i L
  rfl

/-! ### The classification is not vacuous

  `Decoupled` and `Coupled` are not complements in general (a rule could depend
  on `L` only for some indices). The two results below show that at least one
  task inhabits each class, which is what makes the taxonomy in the paper
  non-empty.
-/

theorem exists_decoupled : ∃ t, Decoupled t := ⟨.copy, copy_decoupled⟩

theorem exists_coupled : ∃ t, Coupled t := ⟨.reverse, reverse_coupled⟩

/-- The paper's claim that `reverse` is *not* decoupled. This is the direction
    that matters for the taxonomy: a coupled task must fail the universal form. -/
theorem reverse_not_decoupled : ¬ Decoupled .reverse := by
  intro h
  have h1 := h 0 4 6
  exact absurd h1 (by decide)

end PositionalLookup
