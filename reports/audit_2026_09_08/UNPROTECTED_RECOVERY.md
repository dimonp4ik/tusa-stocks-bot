# Unprotected position recovery

A failed protection order followed by failed emergency closes previously returned without recording the position. The local fix persists the remaining position with no protection ID. Reconciliation attempts closure regardless of whether the parent signal is still open. New entries for that user are refused while an unprotected record remains open. Successful reconciliation marks the record UNPROTECTED_CLOSED and does not submit a cancellation for an empty protection ID.

Tests exercise full mocked entry/protection/close failure and subsequent reconciliation. Network/DB availability remains required; persistence failure is critically logged and the existing urgent user message still asks for manual closure. No live deployment occurred.
