using SimpleDiffEq, BenchmarkTools
using LinearAlgebra, StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Lotka-Volterra
function lotka!(du, u, p, t)
    du[1] = 1.5u[1] - u[1] * u[2]
    du[2] = u[1] * u[2] - 3.0u[2]
    return nothing
end
lotka_prob = ODEProblem(lotka!, [1.0, 1.0], (0.0, 10.0))

# Slightly larger system (linear, skew-symmetric → bounded)
const N = 30
A = randn(rng, N, N)
A = A - A'
u0_N = randn(rng, N)
function lin!(du, u, p, t)
    mul!(du, p, u)
    return nothing
end
lin_prob = ODEProblem(lin!, u0_N, (0.0, 1.0), A)

# Scalar ODE (out-of-place path)
scalar_prob = ODEProblem((u, p, t) -> -u + sin(t), 1.0, (0.0, 10.0))

# =============================================================================
# Fixed-step explicit solvers
# =============================================================================

SUITE["fixed_step"] = BenchmarkGroup()

SUITE["fixed_step"]["SimpleRK4_small"] = @benchmarkable solve(
    $lotka_prob, SimpleRK4(); dt = 0.01
)
SUITE["fixed_step"]["SimpleTsit5_small"] = @benchmarkable solve(
    $lotka_prob, SimpleTsit5(); dt = 0.01
)
SUITE["fixed_step"]["SimpleRK4_medium"] = @benchmarkable solve(
    $lin_prob, SimpleRK4(); dt = 0.001
)
SUITE["fixed_step"]["SimpleTsit5_medium"] = @benchmarkable solve(
    $lin_prob, SimpleTsit5(); dt = 0.001
)

# =============================================================================
# Adaptive solver
# =============================================================================

SUITE["adaptive"] = BenchmarkGroup()

SUITE["adaptive"]["SimpleATsit5_scalar"] = @benchmarkable solve(
    $scalar_prob, SimpleATsit5()
)
SUITE["adaptive"]["SimpleATsit5_small"] = @benchmarkable solve(
    $lotka_prob, SimpleATsit5()
)

# =============================================================================
# Discrete map
# =============================================================================

SUITE["discrete"] = BenchmarkGroup()

logistic(u, p, t) = p * u * (1 - u)
map_prob = DiscreteProblem(logistic, 0.5, (0, 1000), 3.7)
SUITE["discrete"]["SimpleFunctionMap"] = @benchmarkable solve(
    $map_prob, SimpleFunctionMap()
)
