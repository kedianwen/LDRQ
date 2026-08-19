// Assembles the policy's 425-dim observation vector from per-cycle sensor frames.
//
// This class exists because of one detail that is very easy to get wrong and
// impossible to notice from the outside: Isaac Lab flattens a stacked
// observation history **per term**, not per frame.
//
// With 5 frames and terms (ang_vel 3, gravity 3, command 3, joint_pos 26,
// joint_vel 26, prev_action 24) the vector the policy was trained on is
//
//     [ ang_vel   t-4 t-3 t-2 t-1 t ]   15
//     [ gravity   t-4 t-3 t-2 t-1 t ]   15
//     [ command   t-4 t-3 t-2 t-1 t ]   15
//     [ joint_pos t-4 t-3 t-2 t-1 t ]  130
//     [ joint_vel t-4 t-3 t-2 t-1 t ]  130
//     [ action    t-4 t-3 t-2 t-1 t ]  120
//                                      ---
//                                      425
//
// and NOT five consecutive 85-float frames. Both layouts have the right length,
// so a frame-major implementation loads, runs, and produces confident garbage.
// Oldest sample first within each term, newest last -- matching Isaac Lab's
// CircularBuffer, whose `buffer` property documents "most recent entry at the
// end and oldest entry at the beginning".
//
// The per-term widths are not hard-coded here; they are read from the interface
// descriptor emitted by tools/dump_interface.py so that the C++ side cannot
// drift away from the trained policy without the mismatch being detected.

#ifndef R1_POLICY_RUNNER__OBS_ASSEMBLER_HPP_
#define R1_POLICY_RUNNER__OBS_ASSEMBLER_HPP_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace r1_policy_runner
{

/// One observation term: a name and how many floats it contributes per frame.
struct ObsTermSpec
{
  std::string name;
  std::size_t width = 0;
};

/// Ring buffer over the last `history` frames, emitting the term-major layout
/// described above. Allocation-free after construction.
class ObsAssembler
{
public:
  ObsAssembler(std::vector<ObsTermSpec> terms, std::size_t history)
  : terms_(std::move(terms)), history_(history)
  {
    if (history_ == 0) {throw std::invalid_argument("history must be >= 1");}
    for (const auto & t : terms_) {
      if (t.width == 0) {throw std::invalid_argument("term '" + t.name + "' has zero width");}
      frame_dim_ += t.width;
    }
    if (frame_dim_ == 0) {throw std::invalid_argument("no observation terms");}
    ring_.assign(frame_dim_ * history_, 0.0f);
  }

  std::size_t frame_dim() const noexcept {return frame_dim_;}
  std::size_t history() const noexcept {return history_;}
  std::size_t obs_dim() const noexcept {return frame_dim_ * history_;}
  bool warm() const noexcept {return filled_ >= history_;}
  const std::vector<ObsTermSpec> & terms() const noexcept {return terms_;}

  /// Clears history. Call on every transition into an actively-controlled
  /// state: a policy fed a history that straddles an idle period sees a
  /// discontinuity it never saw in training.
  void Reset() noexcept
  {
    std::fill(ring_.begin(), ring_.end(), 0.0f);
    head_ = 0;
    filled_ = 0;
  }

  /// Appends one frame of `frame_dim()` floats, laid out as the terms in order.
  void Push(const float * frame)
  {
    float * slot = ring_.data() + head_ * frame_dim_;
    std::copy(frame, frame + frame_dim_, slot);
    head_ = (head_ + 1) % history_;
    if (filled_ < history_) {++filled_;}
  }

  /// Writes the term-major observation vector into @p out (`obs_dim()` floats).
  ///
  /// Before `history` frames have been pushed the not-yet-written slots read as
  /// zero. Check warm() and hold the robot still until it returns true rather
  /// than feeding the policy a partially-zero history.
  void Fill(float * out) const
  {
    // head_ points at the slot the NEXT push will overwrite, i.e. the oldest.
    std::size_t offset = 0;
    std::size_t term_off = 0;
    for (const auto & term : terms_) {
      for (std::size_t k = 0; k < history_; ++k) {
        const std::size_t slot = (head_ + k) % history_;
        const float * src = ring_.data() + slot * frame_dim_ + term_off;
        std::copy(src, src + term.width, out + offset);
        offset += term.width;
      }
      term_off += term.width;
    }
  }

private:
  std::vector<ObsTermSpec> terms_;
  std::size_t history_;
  std::size_t frame_dim_ = 0;
  std::vector<float> ring_;
  std::size_t head_ = 0;
  std::size_t filled_ = 0;
};

}  // namespace r1_policy_runner

#endif  // R1_POLICY_RUNNER__OBS_ASSEMBLER_HPP_
