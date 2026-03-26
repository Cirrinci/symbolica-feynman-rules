#!/bin/bash

MODEL="code/model_symbolica.py"

CALL_OUT="viz/outputs/call_graph"
MATH_OUT="viz/outputs/math_flow"

CALL_DOT="$CALL_OUT/call_graph_focus_contract_to_full_expression.dot"
CALL_PDF="$CALL_OUT/call_graph_focus_contract_to_full_expression.pdf"

MATH_DOT="$MATH_OUT/math_flow_contract_to_full_expression.dot"
MATH_PDF="$MATH_OUT/math_flow_contract_to_full_expression.pdf"

mkdir -p "$CALL_OUT"
mkdir -p "$MATH_OUT"

python viz/tools/generate_model_diagrams_v2.py "$MODEL" \
  --focus contract_to_full_expression \
  --depth 2 \
  --hide-isolated \
  --out-dir "$CALL_OUT" \
  --only-calls

[ -f "$CALL_DOT" ] && dot -Tpdf "$CALL_DOT" -o "$CALL_PDF"

python viz/tools/generate_math_flow_diagrams.py "$MODEL" \
  --focus contract_to_full_expression \
  --out-dir "$MATH_OUT"

[ -f "$MATH_DOT" ] && dot -Tpdf "$MATH_DOT" -o "$MATH_PDF" || true

[ -f "$CALL_PDF" ] && open "$CALL_PDF"
[ -f "$MATH_PDF" ] && open "$MATH_PDF"