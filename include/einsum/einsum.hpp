#pragma once

// The header-only part of the library: the compile-time entry point
// einsum<"ij,jk->ik">(), the object it answers, the views, the lowering and the
// kernels.
//
// einsum(std::string_view), which parses a subscript that is not known until
// run time, lives in <einsum/rt/parse.hpp> and needs libeinsum_rt linked.
#include "einsum/core/einsum_object.hpp"
#include "einsum/core/execute.hpp"
#include "einsum/core/kernels.hpp"
#include "einsum/core/kind.hpp"
#include "einsum/core/limits.hpp"
#include "einsum/core/lower.hpp"
#include "einsum/core/owned.hpp"
#include "einsum/core/path.hpp"
#include "einsum/core/plan.hpp"
#include "einsum/core/view.hpp"
#include "einsum/ct/einsum.hpp"
#include "einsum/ct/subscripts.hpp"
#include "einsum/util/concepts.hpp"
#include "einsum/util/error.hpp"
#include "einsum/util/fixed_string.hpp"
#include "einsum/util/fixed_vec.hpp"
#include "einsum/util/ranges.hpp"
