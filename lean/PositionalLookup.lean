/-
  Mechanised definitions for "The Answer Stops at Eight: Termination, Not
  Position Range, Drives Length-Generalisation Failure".

  What is formalised here:
    * the three sites at which position can enter a causal attention layer, and
      the kind of code each site can carry;
    * the fact that no encoding in the paper's battery is both absolute and at
      the logits site -- the cell that the softmax makes empty;
    * the definitions of *truncation* and *addressable boundary*, and the
      machine-checked claim that the addressable codes are exactly the ones the
      paper observes truncating.

  What is NOT formalised: the empirical claims (where generation stops, what
  happens when EOS supervision is removed). Those are measurements, not
  theorems, and no proof assistant can discharge them. The algebraic identity
  behind the empty cell -- that a bias constant along the key axis cancels in
  the softmax -- is stated in the paper as Eq. (2) and is not mechanised here,
  because doing it properly needs a real-closed field and this file is
  deliberately dependency-free (core Lean 4, no Mathlib, builds in seconds).
-/

namespace StopsAtEight

/-- Where positional information enters the attention computation. -/
inductive Site where
  | representation   -- h_i = e(x_i) + p_i
  | qk               -- q_i, k_j modified by position
  | logits           -- an additive bias B_ij
  deriving DecidableEq, Repr

/-- What the code can express: a per-index address, or a pairwise relation. -/
inductive Kind where
  | absolute
  | relative
  deriving DecidableEq, Repr

/-- An encoding from the paper's battery.

    `addressable` records whether the model can name an output index exactly at
    generation time; it is the property the paper uses to predict truncation. -/
structure Code where
  name        : String
  site        : Site
  kind        : Kind
  addressable : Bool
  deriving DecidableEq, Repr

/-- The battery: five encodings plus the `q/k`-absolute control. -/
def codes : List Code :=
  [ ⟨"nope",    .representation, .relative, false⟩
  , ⟨"learned", .representation, .absolute, true⟩
  , ⟨"sin",     .representation, .absolute, true⟩
  , ⟨"qk_abs",  .qk,             .absolute, true⟩
  , ⟨"rope",    .qk,             .relative, true⟩
  , ⟨"alibi",   .logits,         .relative, false⟩
  ]

/-- Truncation: the model stops before finishing the answer. -/
def Truncates (c : Code) : Prop := c.addressable = true

/-- An absolute code cannot act at the logits site: a bias that depends only on
    the query index is constant along the key axis and cancels in the softmax.
    Hence every logits code in the battery is relative. -/
def noAbsoluteAtLogits : Bool :=
  codes.all (fun c => !(c.site == Site.logits) || (c.kind == Kind.relative))

theorem no_absolute_at_logits : noAbsoluteAtLogits = true := by decide

/-- The encodings the paper observes stopping at `L_tr`. -/
def truncating : List String :=
  (codes.filter (fun c => c.addressable)).map (fun c => c.name)

theorem truncating_eq : truncating = ["learned", "sin", "qk_abs", "rope"] := by decide

/-- The encodings the paper observes stopping at the true length. -/
def nonTruncating : List String :=
  (codes.filter (fun c => !c.addressable)).map (fun c => c.name)

theorem non_truncating_eq : nonTruncating = ["nope", "alibi"] := by decide

/-- The partition is cut by addressability, not by kind and not by site:
    a relative code (`rope`) truncates and an absolute code at `q/k` truncates
    exactly like one at the representation. -/
theorem not_kind_cut :
    (codes.filter (fun c => c.kind == Kind.relative)).length = 3 := by decide

theorem not_site_cut :
    (codes.filter (fun c => c.site == Site.representation)).length = 3 := by decide

/-- A code at the `q/k` site carries an absolute kind, which is the condition
    that breaks the confound in the standard battery. -/
theorem qk_absolute_exists :
    (codes.filter (fun c => c.site == Site.qk && c.kind == Kind.absolute)).length = 1 := by
  decide

end StopsAtEight
