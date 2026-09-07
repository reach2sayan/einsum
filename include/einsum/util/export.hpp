#pragma once

// libeinsum_rt is built with hidden visibility, so the handful of symbols a
// caller links against have to say so.  Everything else in the project is
// header-only and wears nothing.
#if defined(_MSC_VER)
#if defined(EINSUM_BUILDING)
#define EINSUM_API __declspec(dllexport)
#else
#define EINSUM_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define EINSUM_API __attribute__((visibility("default")))
#else
#define EINSUM_API
#endif
