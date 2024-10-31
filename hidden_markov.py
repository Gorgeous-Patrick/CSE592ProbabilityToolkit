from factor import Factor
from sympy import symbols, solve, Eq


class MarkovFactor(Factor):
    """A Markov Factor is a Factor whose variable is called 'state'."""

    def __init__(self, table):
        """
        Initialize the Markov Factor.

        Args:
            table: Dictionary where keys are states (e.g., 0), and values are the corresponding probabilities.
        """
        super().__init__(["state"], table)


class MarkovTransitionFactor(Factor):
    """A Markov Transition Factor is a Factor whose variables are 'prev_state' and 'current_state'."""

    def __init__(self, table):
        """
        Initialize the Markov Transition Factor.

        Args:
            table: Dictionary where keys are tuples (prev_state, current_state), and values are the corresponding probabilities.
        """
        super().__init__(["prev_state", "current_state"], table)


class MarkovEmissionFactor(Factor):
    """A Markov Emission Factor is a Factor whose variables are 'state' and 'observation'."""

    def __init__(self, table):
        """
        Initialize the Markov Emission Factor.

        Args:
            table: Dictionary where keys are tuples (state, observation), and values are the corresponding probabilities.
        """
        super().__init__(["state", "observation"], table)


class HiddenMarkovModel:
    def __init__(
        self, states, observations, start_probs, transition_probs, emission_probs
    ):
        """
        Initialize the Hidden Markov Model.

        Args:
            states: List of possible hidden states.
            observations: List of possible observations.
            start_probs: Initial probabilities as a Factor over states.
            transition_probs: Transition probabilities as a Factor over (previous_state, current_state).
            emission_probs: Emission probabilities as a dictionary of Factors, one for each observation.
        """
        self.states = states
        self.observations = observations
        self.start_probs = start_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs

    def stationary_distribution(self):
        """
        Compute the stationary distribution of the Markov chain.

        Returns:
            The stationary distribution as a Factor over states.
        """
        # Initialize the stationary distribution as the start probabilities
        symbols_list = [symbols(f"state_{i}") for i in range(len(self.states))]
        # Create a dict that maps states to symbols
        distro = {(state,): symbol for state, symbol in zip(self.states, symbols_list)}
        # Create a factor with the stationary distribution
        distro_factor = Factor(["prev_state"], distro)
        new_distro_factor = (
            self.transition_probs.multiply(distro_factor)
            .sum_out("prev_state")
            .rename_variable("current_state", "state")
        )
        # Construct the equations for the stationary distribution
        equations = [Eq(sum(symbols_list), 1)]
        for state in self.states:
            equation = Eq(distro[(state,)], new_distro_factor.table[(state,)])
            equations.append(equation)
        solution = solve(equations, symbols_list)
        # Construct the factor stationary distribution from the solution
        return Factor(
            ["state"], {state: solution[symbol] for state, symbol in distro.items()}
        )

    def filtering(self, observed_sequence):
        """
        Perform the Forward algorithm to compute the probability of the observation sequence.

        Args:
            observed_sequence: List of observed symbols.

        Returns:
            The probability of the observed sequence.
        """
        forward_message = self.start_probs
        # Initialize forward messages with start probabilities and first observation
        for obs in observed_sequence:
            forward_message = (
                forward_message.rename_variable("state", "prev_state")
                .multiply(self.transition_probs)
                .sum_out("prev_state")
                .rename_variable("current_state", "state")
                .multiply(self.emission_probs.restrict("observation", obs))
                .normalize()
            )
        return forward_message

    def predict(self, observed_seq, target_idx):
        """
        Predict the probability of the target state at a given time step.

        Args:
            observed_seq: List of observed symbols.
            target_idx: The time step to predict the target state.

        Returns:
            The probability of the target state at the given time step.
        """
        forward_message = self.filtering(observed_seq)
        # Iterate over the part that is not observed
        for t in range(len(observed_seq), target_idx):
            forward_message = (
                forward_message.rename_variable("state", "prev_state")
                .multiply(self.transition_probs)
                .sum_out("prev_state")
                .rename_variable("current_state", "state")
                .normalize()
            )
        return forward_message

    def backward(self, observed_sequence):
        """
        Perform the Backward algorithm to compute the probability of the observation sequence.

        Args:
            observed_sequence: List of observed symbols.

        Returns:
            The probability of the observed sequence.
        """
        # Initialize backward messages to all ones (since we're looking at the end of the sequence)
        backward_message = Factor(["state"], {(s,): 1.0 for s in self.states})

        for t in reversed(range(len(observed_sequence) - 1)):
            # Update with the observation evidence
            evidence_factor = self.emission_probs[observed_sequence[t + 1]]
            backward_message = backward_message.multiply(evidence_factor)

            # Predict: Multiply by transition probabilities and sum out next state
            backward_message = self.transition_probs.multiply(backward_message)
            backward_message = backward_message.sum_out("next_state")

        # Multiply with start probabilities and first observation evidence
        initial_evidence = self.emission_probs[observed_sequence[0]]
        start_prob = self.start_probs.multiply(initial_evidence)
        backward_message = backward_message.multiply(start_prob)

        # Sum out the initial states to get the probability of the entire sequence
        sequence_prob = sum(backward_message.table.values())
        return sequence_prob

    def viterbi(self, observed_sequence):
        """
        Perform the Viterbi algorithm to find the most likely sequence of hidden states.

        Args:
            observed_sequence: List of observed symbols.

        Returns:
            The most likely sequence of states.
        """
        viterbi_table = [{}]
        backpointer = [{}]

        # Initialize base cases (t == 0)
        initial_message = self.start_probs.multiply(
            self.emission_probs[observed_sequence[0]]
        )
        for state in self.states:
            viterbi_table[0][state] = initial_message.table[(state,)]
            backpointer[0][state] = None

        # Run Viterbi for t > 0
        for t in range(1, len(observed_sequence)):
            viterbi_table.append({})
            backpointer.append({})
            observation = observed_sequence[t]

            for state in self.states:
                max_prob, max_state = max(
                    (
                        viterbi_table[t - 1][prev_state]
                        * self.transition_probs.table[(prev_state, state)]
                        * self.emission_probs[observation].table[(state,)],
                        prev_state,
                    )
                    for prev_state in self.states
                )
                viterbi_table[t][state] = max_prob
                backpointer[t][state] = max_state

        # Find the optimal path
        max_final_prob, max_final_state = max(
            (viterbi_table[-1][state], state) for state in self.states
        )
        best_path = [max_final_state]

        # Backtrace the path
        for t in range(len(observed_sequence) - 1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        return best_path
