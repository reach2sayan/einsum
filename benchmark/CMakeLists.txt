einsum_use_googlebenchmark()

add_executable(einsum_benchmark benchmark.cpp)
target_link_libraries(einsum_benchmark PRIVATE einsum::einsum einsum::rt benchmark::benchmark)
einsum_target_flags(einsum_benchmark)
einsum_target_pch(einsum_benchmark)
einsum_target_trace(einsum_benchmark)
einsum_runtime_deps(einsum_benchmark)

# `cmake --build ... --target benchmark_json` leaves a machine-readable run
# beside the binary, which is what a before/after comparison reads.
add_custom_target(benchmark_json
        COMMAND einsum_benchmark --benchmark_format=json
        --benchmark_out=${CMAKE_CURRENT_BINARY_DIR}/benchmark.json
        DEPENDS einsum_benchmark
        COMMENT "Running einsum_benchmark into benchmark.json"
        VERBATIM)
