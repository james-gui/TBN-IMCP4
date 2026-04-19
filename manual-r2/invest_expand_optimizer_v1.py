from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

TOTAL_BUDGET = 50_000
MAX_ALLOCATION = 100
NUM_SPEED_CHOICES = MAX_ALLOCATION + 1
SPEED_CHOICES = list(range(NUM_SPEED_CHOICES))
DEFAULT_STRATEGIC_SCENARIOS = (0.95, 0.97, 0.98)


@dataclass(frozen=True)
class AllocationChoice:
    research: int
    scale: int
    speed: int
    speed_multiplier: float
    pnl: float
    budget_used: float
    expected_rank: float


@dataclass(frozen=True)
class PopulationModelResult:
    name: str
    strategy: list[float]
    converged: bool
    iterations: int
    family: str = "strategic"


@dataclass(frozen=True)
class MixtureScenarioResult:
    strategic_share: float
    noise_share: float
    model_weights: list[tuple[str, float]]
    market_strategy: list[float]


class PayoffModel:
    """Budget allocator with endogenous rank-based speed."""

    def __init__(self, num_players: int = 20_000) -> None:
        self.num_players = num_players
        self._products: list[list[float]] = []
        self._costs: list[list[float]] = []
        self._research: list[list[int]] = []
        self._scale: list[list[int]] = []
        self._build_envelopes()

    @staticmethod
    def research_value(research_pct: int) -> float:
        return 200_000.0 * math.log1p(research_pct) / math.log(101.0)

    @staticmethod
    def scale_multiplier(scale_pct: int) -> float:
        return 0.07 * scale_pct

    @staticmethod
    def budget_used(research_pct: int, scale_pct: int, speed_pct: int) -> float:
        return TOTAL_BUDGET * (research_pct + scale_pct + speed_pct) / MAX_ALLOCATION

    @staticmethod
    def speed_multiplier_from_cdf(cdf_value: float) -> float:
        # Exact in expectation because the rank multiplier is affine in rank.
        return 0.1 + 0.8 * cdf_value

    def _build_envelopes(self) -> None:
        for speed in SPEED_CHOICES:
            products: list[float] = []
            costs: list[float] = []
            research_allocs: list[int] = []
            scale_allocs: list[int] = []
            remaining = MAX_ALLOCATION - speed

            for research in range(remaining + 1):
                research_value = self.research_value(research)
                for scale in range(remaining - research + 1):
                    products.append(research_value * self.scale_multiplier(scale))
                    costs.append(self.budget_used(research, scale, speed))
                    research_allocs.append(research)
                    scale_allocs.append(scale)

            self._products.append(products)
            self._costs.append(costs)
            self._research.append(research_allocs)
            self._scale.append(scale_allocs)

    def best_choice_for_speed(
        self, speed: int, speed_multiplier: float, opponent_cdf: float
    ) -> AllocationChoice:
        best_pnl = -float("inf")
        best_idx = 0
        products = self._products[speed]
        costs = self._costs[speed]

        for idx, product in enumerate(products):
            pnl = product * speed_multiplier - costs[idx]
            if pnl > best_pnl:
                best_pnl = pnl
                best_idx = idx

        higher_share = 1.0 - opponent_cdf
        expected_rank = 1.0 + (self.num_players - 1) * higher_share
        return AllocationChoice(
            research=self._research[speed][best_idx],
            scale=self._scale[speed][best_idx],
            speed=speed,
            speed_multiplier=speed_multiplier,
            pnl=best_pnl,
            budget_used=costs[best_idx],
            expected_rank=expected_rank,
        )

    def expected_choices(self, speed_strategy: list[float]) -> tuple[list[float], list[AllocationChoice]]:
        pnls: list[float] = []
        choices: list[AllocationChoice] = []
        cdf = 0.0

        for speed, share in enumerate(speed_strategy):
            cdf += share
            multiplier = self.speed_multiplier_from_cdf(cdf)
            choice = self.best_choice_for_speed(speed, multiplier, cdf)
            pnls.append(choice.pnl)
            choices.append(choice)

        return pnls, choices

    def best_response_distribution(
        self, speed_strategy: list[float], tolerance: float = 1e-9
    ) -> list[float]:
        pnls, _ = self.expected_choices(speed_strategy)
        best_value = max(pnls)
        support = [idx for idx, value in enumerate(pnls) if abs(value - best_value) <= tolerance]
        response = [0.0] * NUM_SPEED_CHOICES
        mass = 1.0 / len(support)
        for idx in support:
            response[idx] = mass
        return response

    def top_choices_against(self, speed_strategy: list[float], limit: int = 5) -> list[AllocationChoice]:
        _, choices = self.expected_choices(speed_strategy)
        ranking = sorted(choices, key=lambda choice: choice.pnl, reverse=True)
        return ranking[:limit]


def uniform_strategy() -> list[float]:
    return [1.0 / NUM_SPEED_CHOICES] * NUM_SPEED_CHOICES


def normalize(strategy: list[float]) -> list[float]:
    total = sum(strategy)
    if total <= 0.0:
        return uniform_strategy()
    return [value / total for value in strategy]


def weighted_average(strategies: list[list[float]], weights: list[float]) -> list[float]:
    combined = [0.0] * NUM_SPEED_CHOICES
    for strategy, weight in zip(strategies, weights):
        for speed in SPEED_CHOICES:
            combined[speed] += weight * strategy[speed]
    return normalize(combined)


def max_abs_delta(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def blend(left: list[float], right: list[float], weight: float) -> list[float]:
    return [(1.0 - weight) * a + weight * b for a, b in zip(left, right)]


def softmax(values: list[float], temperature: float) -> list[float]:
    peak = max(values)
    weights = [math.exp(max(-700.0, min(700.0, temperature * (value - peak)))) for value in values]
    return normalize(weights)


def softmax_by_profit(values: list[float], intensity: float, profit_scale: float) -> list[float]:
    peak = max(values)
    shifted = [intensity * (value - peak) / profit_scale for value in values]
    weights = [math.exp(max(-700.0, min(700.0, value))) for value in shifted]
    return normalize(weights)


def poisson_weights(tau: float, max_level: int) -> list[float]:
    weights = [
        math.exp(-tau) * tau**level / math.factorial(level) for level in range(max_level + 1)
    ]
    return normalize(weights)


def top_speed_support(strategy: list[float], limit: int = 5) -> list[tuple[int, float]]:
    ranking = sorted(enumerate(strategy), key=lambda pair: pair[1], reverse=True)
    return [(speed, share) for speed, share in ranking if share > 1e-4][:limit]


def strategy_line(strategy: list[float], limit: int = 5) -> str:
    support = top_speed_support(strategy, limit=limit)
    if not support:
        return "none"
    return ", ".join(f"{speed}% ({share:.1%})" for speed, share in support)


def random_walk_strategy(
    center: int = 50,
    bias: float = 0.075,
    iterations: int = 600,
) -> list[float]:
    distribution = uniform_strategy()

    for _ in range(iterations):
        updated = [0.0] * NUM_SPEED_CHOICES
        for speed, mass in enumerate(distribution):
            if speed < center:
                up_probability = 0.5 + bias
            elif speed > center:
                up_probability = 0.5 - bias
            else:
                up_probability = 0.5

            down_probability = 1.0 - up_probability
            up_speed = min(MAX_ALLOCATION, speed + 1)
            down_speed = max(0, speed - 1)
            updated[up_speed] += mass * up_probability
            updated[down_speed] += mass * down_probability

        distribution = normalize(updated)

    return distribution


def intuitive_noise_strategy() -> list[float]:
    # "No idea" players look uniform; intuitive players tend to avoid extreme corners.
    uniform = uniform_strategy()
    walker = random_walk_strategy()
    return weighted_average([uniform, walker], [0.5, 0.5])


def solve_mean_field_best_response(
    model: PayoffModel,
    max_iterations: int = 800,
    tolerance: float = 1e-10,
) -> PopulationModelResult:
    strategy = uniform_strategy()
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        response = model.best_response_distribution(strategy)
        step = 2.0 / (iteration + 2.0)
        updated = normalize(blend(strategy, response, step))

        if (
            max_abs_delta(updated, strategy) < tolerance
            or max_abs_delta(strategy, response) < 1e-6
        ) and iteration > 150:
            strategy = updated
            converged = True
            break

        strategy = updated

    if not converged:
        response = model.best_response_distribution(strategy)
        converged = max_abs_delta(strategy, response) < 1e-4

    return PopulationModelResult(
        name="Mean-field best response",
        strategy=strategy,
        converged=converged,
        iterations=iteration,
    )


def solve_logit_qre(
    model: PayoffModel,
    lambda_qre: float = 12.0,
    damping: float = 0.35,
    max_iterations: int = 1_500,
    tolerance: float = 1e-10,
) -> PopulationModelResult:
    strategy = uniform_strategy()
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        pnls, _ = model.expected_choices(strategy)
        response = softmax(pnls, lambda_qre)
        updated = normalize(blend(strategy, response, damping))

        if max_abs_delta(updated, strategy) < tolerance and iteration > 150:
            strategy = updated
            converged = True
            break

        strategy = updated

    return PopulationModelResult(
        name="Logit QRE",
        strategy=strategy,
        converged=converged,
        iterations=iteration,
    )


def solve_heterogeneous_qre(
    model: PayoffModel,
    lambdas: tuple[float, ...] = (4.0, 12.0, 28.0),
    lambda_weights: tuple[float, ...] = (0.25, 0.45, 0.30),
    damping: float = 0.35,
    max_iterations: int = 1_500,
    tolerance: float = 1e-10,
) -> PopulationModelResult:
    strategy = uniform_strategy()
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        pnls, _ = model.expected_choices(strategy)
        subtype_responses = [softmax(pnls, lambda_value) for lambda_value in lambdas]
        response = weighted_average(subtype_responses, list(lambda_weights))
        updated = normalize(blend(strategy, response, damping))

        if max_abs_delta(updated, strategy) < tolerance and iteration > 150:
            strategy = updated
            converged = True
            break

        strategy = updated

    return PopulationModelResult(
        name="Heterogeneous QRE",
        strategy=strategy,
        converged=converged,
        iterations=iteration,
    )


def solve_cognitive_hierarchy(
    model: PayoffModel,
    tau: float = 1.5,
    max_level: int = 8,
) -> PopulationModelResult:
    level_weights = poisson_weights(tau, max_level)
    level_strategies = [uniform_strategy()]

    for level in range(1, max_level + 1):
        lower_weights = normalize(level_weights[:level])
        belief = [0.0] * NUM_SPEED_CHOICES

        for lower_level in range(level):
            for speed in SPEED_CHOICES:
                belief[speed] += lower_weights[lower_level] * level_strategies[lower_level][speed]

        level_strategies.append(model.best_response_distribution(belief))

    aggregate = weighted_average(level_strategies, level_weights)
    return PopulationModelResult(
        name="Cognitive hierarchy",
        strategy=aggregate,
        converged=True,
        iterations=max_level,
    )


def solve_ewa_learning(
    model: PayoffModel,
    choice_temperature: float = 1.8,
    payoff_scale: float = 25_000.0,
    phi: float = 0.92,
    rho: float = 0.90,
    delta: float = 0.70,
    max_iterations: int = 1_200,
    tolerance: float = 1e-10,
) -> PopulationModelResult:
    attractions = [0.0] * NUM_SPEED_CHOICES
    strategy = uniform_strategy()
    experience = 1.0
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        pnls, _ = model.expected_choices(strategy)
        normalized_payoffs = [pnl / payoff_scale for pnl in pnls]
        next_experience = rho * experience + 1.0
        updated_attractions = [0.0] * NUM_SPEED_CHOICES

        for speed in SPEED_CHOICES:
            reinforcement = (delta + (1.0 - delta) * strategy[speed]) * normalized_payoffs[speed]
            updated_attractions[speed] = (
                phi * experience * attractions[speed] + reinforcement
            ) / next_experience

        updated_strategy = softmax(updated_attractions, choice_temperature)

        if max_abs_delta(updated_strategy, strategy) < tolerance and iteration > 100:
            attractions = updated_attractions
            strategy = updated_strategy
            experience = next_experience
            converged = True
            break

        attractions = updated_attractions
        strategy = updated_strategy
        experience = next_experience

    return PopulationModelResult(
        name="EWA learning",
        strategy=strategy,
        converged=converged,
        iterations=iteration,
    )


def strategic_model_weights(
    model: PayoffModel,
    strategic_results: list[PopulationModelResult],
    selection_intensity: float,
    profit_scale: float,
) -> list[tuple[str, float]]:
    theoretical_pnls = [
        model.top_choices_against(result.strategy, limit=1)[0].pnl for result in strategic_results
    ]
    weights = softmax_by_profit(theoretical_pnls, selection_intensity, profit_scale)
    return [(result.name, weight) for result, weight in zip(strategic_results, weights)]


def build_mixture_scenario(
    model: PayoffModel,
    strategic_results: list[PopulationModelResult],
    strategic_share: float,
    selection_intensity: float,
    profit_scale: float,
) -> MixtureScenarioResult:
    noise_share = 1.0 - strategic_share
    model_weights = strategic_model_weights(
        model,
        strategic_results,
        selection_intensity=selection_intensity,
        profit_scale=profit_scale,
    )
    strategic_mix = weighted_average(
        [result.strategy for result in strategic_results],
        [weight for _, weight in model_weights],
    )
    noise_mix = intuitive_noise_strategy()
    market_strategy = normalize(
        [
            strategic_share * strategic_mix[speed] + noise_share * noise_mix[speed]
            for speed in SPEED_CHOICES
        ]
    )
    return MixtureScenarioResult(
        strategic_share=strategic_share,
        noise_share=noise_share,
        model_weights=model_weights,
        market_strategy=market_strategy,
    )


def format_choice(choice: AllocationChoice, rank_denominator: int) -> str:
    return (
        f"R={choice.research} S={choice.scale} V={choice.speed} | "
        f"PnL={choice.pnl:,.2f} | "
        f"mult={choice.speed_multiplier:.4f} | "
        f"rank={choice.expected_rank:.1f}/{rank_denominator}"
    )


def describe_result(model: PayoffModel, result: PopulationModelResult, limit: int = 5) -> str:
    best_choices = model.top_choices_against(result.strategy, limit=limit)
    status = "converged" if result.converged else "max-iter"
    lines = [
        result.name,
        f"  status: {status} after {result.iterations} iterations",
        f"  population speed mix: {strategy_line(result.strategy, limit=limit)}",
        "  top allocations:",
    ]
    for index, choice in enumerate(best_choices, start=1):
        lines.append(f"    {index}. {format_choice(choice, model.num_players)}")
    return "\n".join(lines)


def describe_mixture_scenario(
    model: PayoffModel,
    scenario: MixtureScenarioResult,
    limit: int = 5,
) -> str:
    best_choices = model.top_choices_against(scenario.market_strategy, limit=limit)
    weight_line = ", ".join(
        f"{name}={weight:.1%}" for name, weight in sorted(scenario.model_weights, key=lambda item: item[1], reverse=True)
    )
    lines = [
        f"Heterogeneous market ({scenario.strategic_share:.0%} strategic / {scenario.noise_share:.0%} noise)",
        f"  strategic model weights: {weight_line}",
        f"  market speed mix: {strategy_line(scenario.market_strategy, limit=limit)}",
        "  top allocations:",
    ]
    for index, choice in enumerate(best_choices, start=1):
        lines.append(f"    {index}. {format_choice(choice, model.num_players)}")
    return "\n".join(lines)


def parse_share_scenarios(text: str) -> list[float]:
    values = [float(chunk.strip()) for chunk in text.split(",") if chunk.strip()]
    valid = []
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError("Strategic share scenarios must be between 0 and 1.")
        valid.append(value)
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize research / scale / speed with endogenous rank-based speed models "
            "and a heterogeneous strategic-vs-noise population."
        )
    )
    parser.add_argument(
        "--players",
        type=int,
        default=20_000,
        help="Population size used for expected rank reporting.",
    )
    parser.add_argument(
        "--qre-lambda",
        type=float,
        default=12.0,
        help="Precision parameter for homogeneous logit QRE.",
    )
    parser.add_argument(
        "--ch-tau",
        type=float,
        default=1.5,
        help="Average reasoning depth for cognitive hierarchy.",
    )
    parser.add_argument(
        "--ch-levels",
        type=int,
        default=8,
        help="Maximum cognitive hierarchy level to include.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top speed masses and top allocations to print.",
    )
    parser.add_argument(
        "--strategic-scenarios",
        type=str,
        default="0.95,0.97,0.98",
        help="Comma-separated strategic population shares. Noise share is 1 minus this value.",
    )
    parser.add_argument(
        "--selection-intensity",
        type=float,
        default=2.0,
        help="How strongly players switch toward models with higher theoretical PnL.",
    )
    parser.add_argument(
        "--selection-profit-scale",
        type=float,
        default=25_000.0,
        help="PnL scale used in the evolutionary model-weight softmax.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PayoffModel(num_players=args.players)
    strategic_results = [
        solve_mean_field_best_response(model),
        solve_logit_qre(model, lambda_qre=args.qre_lambda),
        solve_heterogeneous_qre(model),
        solve_cognitive_hierarchy(model, tau=args.ch_tau, max_level=args.ch_levels),
        solve_ewa_learning(model),
    ]

    for result in strategic_results:
        print(describe_result(model, result, limit=args.top_k))
        print()

    for strategic_share in parse_share_scenarios(args.strategic_scenarios):
        scenario = build_mixture_scenario(
            model,
            strategic_results,
            strategic_share=strategic_share,
            selection_intensity=args.selection_intensity,
            profit_scale=args.selection_profit_scale,
        )
        print(describe_mixture_scenario(model, scenario, limit=args.top_k))
        print()


if __name__ == "__main__":
    main()
