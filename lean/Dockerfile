# Lean 4 environment for the mechanised definitions.
#
#   docker build -t poslookup-lean lean/
#   docker run --rm poslookup-lean
#
# No Mathlib: the formalisation uses core Lean only, so this builds in seconds.
FROM leanprover/lean4:v4.35.0

WORKDIR /lean
COPY . .
RUN lake build
