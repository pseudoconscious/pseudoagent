import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def _probability_to_ratio(probability, reverse_probability):
    """
    Convert a pairwise probability judgment into an AHP ratio.
    """

    epsilon = 0.001

    probability = max(
        epsilon,
        min(1.0 - epsilon, probability)
    )

    reverse_probability = max(
        epsilon,
        min(1.0 - epsilon, reverse_probability)
    )

    return probability / reverse_probability


def _solve_ahp(matrix):
    """
    Solve an AHP pairwise-comparison matrix using the
    geometric-mean method.

    Returns normalized weights.
    """

    n = len(matrix)

    if n == 1:
        return [1.0]

    geometric_means = []

    for row in matrix:
        product = 1.0

        for value in row:
            product *= value

        geometric_means.append(
            product ** (1.0 / n)
        )

    total = sum(geometric_means)

    return [
        value / total
        for value in geometric_means
    ]


def _make_pairwise_question(
    llm,
    data,
    left_name,
    left_description,
    right_name,
    right_description,
    context,
):
    """
    Ask one binary pairwise AHP question.

    The probability returned by choose_probability is
    converted into an AHP ratio.
    """

    prompt = f"""
You are making ONE simple pairwise judgment.

Current data:
{json.dumps(data, indent=4)}

{context}

Compare ONLY these two choices:

A: {left_name}
Description: {left_description}

B: {right_name}
Description: {right_description}

Which choice is more appropriate?

Return the probability that A is more appropriate than B.
Do not reason about any other choices.
"""

    result = llm.choose_probability(
        prompt,
        [left_name, right_name],
        [left_description, right_description],
    )

    probabilities = result["probabilities"]

    left_probability = probabilities.get(
        left_name,
        0
    )

    right_probability = probabilities.get(
        right_name,
        0
    )

    return _probability_to_ratio(
        left_probability,
        right_probability,
    )


def choose_ahp(
    llm,
    data,
    tools,
    criterias,
):
    """
    Select the most appropriate tool using AHP.

    1. Compare criteria against each other.
    2. Compare tools against each other for every criterion.
    3. Run independent pairwise questions concurrently.
    4. Solve the AHP matrices.
    5. Combine criterion weights with tool weights.
    6. Return the highest-scoring tool.
    """

    criterion_names = list(criterias.keys())
    tool_names = list(tools.keys())

    # -----------------------------------------------------
    # 1. Criteria importance matrix
    # -----------------------------------------------------

    criteria_matrix = [
        [1.0 for _ in criterion_names]
        for _ in criterion_names
    ]

    criteria_questions = []

    for i in range(len(criterion_names)):
        for j in range(i + 1, len(criterion_names)):

            left = criterion_names[i]
            right = criterion_names[j]

            criteria_questions.append(
                (
                    i,
                    j,
                    left,
                    right,
                    criterias[left],
                    criterias[right],
                )
            )

    def run_criteria_question(question):
        i, j, left, right, left_desc, right_desc = question

        ratio = _make_pairwise_question(
            llm,
            data,
            left,
            left_desc,
            right,
            right_desc,
            context=(
                "The criteria are being compared for their importance "
                "when selecting the single most appropriate tool."
            ),
        )

        return i, j, ratio

    with ThreadPoolExecutor(
        max_workers=len(criteria_questions) or 1
    ) as executor:

        futures = [
            executor.submit(
                run_criteria_question,
                question
            )
            for question in criteria_questions
        ]

        for future in as_completed(futures):
            i, j, ratio = future.result()

            criteria_matrix[i][j] = ratio
            criteria_matrix[j][i] = 1.0 / ratio

    criterion_weights = _solve_ahp(
        criteria_matrix
    )

    # -----------------------------------------------------
    # 2. Tool pairwise matrices for every criterion
    # -----------------------------------------------------

    tool_weights_by_criterion = {}

    for criterion_index, criterion_name in enumerate(
        criterion_names
    ):

        criterion_description = criterias[
            criterion_name
        ]

        tool_matrix = [
            [1.0 for _ in tool_names]
            for _ in tool_names
        ]

        tool_questions = []

        for i in range(len(tool_names)):
            for j in range(i + 1, len(tool_names)):

                left = tool_names[i]
                right = tool_names[j]

                tool_questions.append(
                    (
                        i,
                        j,
                        left,
                        right,
                    )
                )

        def run_tool_question(question):
            i, j, left, right = question

            ratio = _make_pairwise_question(
                llm,
                data,
                left,
                tools[left],
                right,
                tools[right],
                context=(
                    f"Criterion: {criterion_name}\n"
                    f"Criterion description: "
                    f"{criterion_description}\n\n"
                    "Judge only how well each tool satisfies this "
                    "criterion in the current state."
                ),
            )

            return i, j, ratio

        with ThreadPoolExecutor(
            max_workers=len(tool_questions) or 1
        ) as executor:

            futures = [
                executor.submit(
                    run_tool_question,
                    question
                )
                for question in tool_questions
            ]

            for future in as_completed(futures):
                i, j, ratio = future.result()

                tool_matrix[i][j] = ratio
                tool_matrix[j][i] = 1.0 / ratio

        tool_weights = _solve_ahp(
            tool_matrix
        )

        tool_weights_by_criterion[
            criterion_name
        ] = tool_weights

    # -----------------------------------------------------
    # 3. Combine all AHP layers
    # -----------------------------------------------------

    final_scores = {
        tool: 0.0
        for tool in tool_names
    }

    for criterion_index, criterion_name in enumerate(
        criterion_names
    ):

        criterion_weight = criterion_weights[
            criterion_index
        ]

        tool_weights = tool_weights_by_criterion[
            criterion_name
        ]

        for tool_index, tool_name in enumerate(
            tool_names
        ):

            final_scores[tool_name] += (
                criterion_weight *
                tool_weights[tool_index]
            )

    # -----------------------------------------------------
    # 4. Select highest-scoring tool
    # -----------------------------------------------------

    selected_tool = max(
        final_scores,
        key=final_scores.get
    )

    print("### AHP CRITERIA WEIGHTS ###")
    print(
        json.dumps(
            dict(
                zip(
                    criterion_names,
                    criterion_weights
                )
            ),
            indent=4
        )
    )

    print("### AHP TOOL SCORES ###")
    print(
        json.dumps(
            final_scores,
            indent=4
        )
    )

    print("### AHP SELECTED TOOL ###")
    print(selected_tool)

    return selected_tool