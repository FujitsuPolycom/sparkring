#pragma once

namespace spark_transport::tiled_prefill_research::detail {

// Reserve ownership before posting. first_post must submit one WR and guarantee
// that throwing means it was not posted (VerbsEndpoint::write has this contract).
// Once it returns, any final_post failure retains the reservation: an earlier
// unsignaled payload may still access registered storage without a CQE.
template <typename Pending, typename Counter, typename Record,
          typename FirstPost, typename FinalPost>
void post_reserved_work(Pending& pending, Counter& outstanding,
                        const Record& record, FirstPost first_post,
                        FinalPost final_post) {
  pending.push_back(record);  // Allocation may fail before any WR is submitted.
  outstanding += record.wqe_span;
  try {
    first_post();
  } catch (...) {
    outstanding -= record.wqe_span;
    pending.pop_back();
    throw;
  }
  final_post();
}

}  // namespace spark_transport::tiled_prefill_research::detail
