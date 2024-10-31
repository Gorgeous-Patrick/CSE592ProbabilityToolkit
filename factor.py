from sympy import symbols, solveset, Eq


class Factor:
    def __init__(self, variables, table):
        """
        Initialize the factor.

        Args:
            variables: List of variable names (e.g., ['A', 'B']).
            table: Dictionary where keys are tuples representing assignments (e.g., (0, 1)),
                   and values are the corresponding probabilities.
        """
        self.variables = variables  # List of variable names
        self.table = table  # Probability table as a dictionary

    def __repr__(self):
        """
        String representation for pretty printing.
        """
        repr_str = f"Factor({', '.join(self.variables)}):\n"
        for assignment, prob in self.table.items():
            assignment_str = ", ".join(
                [f"{var}={val}" for var, val in zip(self.variables, assignment)]
            )
            repr_str += f"  P({assignment_str}) = {prob}\n"
        return repr_str

    def restrict(self, variable, value):
        """
        Apply evidence by restricting a variable to a specific value.

        Args:
            variable: The variable to restrict (e.g., 'A').
            value: The value to restrict it to (e.g., 1).

        Returns:
            A new factor with the variable restricted.
        """
        if variable not in self.variables:
            return self  # If the variable is not in the factor, return the factor as is

        var_index = self.variables.index(variable)
        new_table = {
            assignment: prob
            for assignment, prob in self.table.items()
            if assignment[var_index] == value
        }

        # Remove the restricted variable from the variable list and table entries
        new_variables = [v for v in self.variables if v != variable]
        new_table = {
            tuple(val for i, val in enumerate(assignment) if i != var_index): prob
            for assignment, prob in new_table.items()
        }

        return Factor(new_variables, new_table)

    def multiply(self, other):
        """
        Multiply this factor with another factor.

        Args:
            other: Another Factor to multiply with.

        Returns:
            A new factor representing the product of the two factors.
        """
        common_vars = list(set(self.variables).intersection(other.variables))
        all_vars = self.variables + [
            var for var in other.variables if var not in common_vars
        ]

        result_table = {}

        # Iterate over all combinations of the factors' tables
        for assignment1, prob1 in self.table.items():
            for assignment2, prob2 in other.table.items():
                # Check if the common variables have the same value
                if self._consistent(assignment1, assignment2, common_vars, other):
                    merged_assignment = self._merge_assignments(
                        assignment1, assignment2, self.variables, other.variables
                    )
                    result_table[merged_assignment] = prob1 * prob2

        return Factor(all_vars, result_table)

    def sum_out(self, variable):
        """
        Eliminate a variable by summing it out.

        Args:
            variable: The variable to eliminate.

        Returns:
            A new factor where the variable is summed out.
        """
        if variable not in self.variables:
            return self  # If the variable is not in the factor, return it as is

        var_index = self.variables.index(variable)
        new_variables = [v for v in self.variables if v != variable]
        result_table = {}

        for assignment, prob in self.table.items():
            reduced_assignment = tuple(
                val for i, val in enumerate(assignment) if i != var_index
            )
            if reduced_assignment not in result_table:
                result_table[reduced_assignment] = 0
            result_table[reduced_assignment] += prob

        return Factor(new_variables, result_table)

    def _consistent(self, assignment1, assignment2, common_vars, other):
        """
        Check if two assignments are consistent for common variables.
        """
        for var in common_vars:
            index1 = self.variables.index(var)
            index2 = other.variables.index(var)
            if assignment1[index1] != assignment2[index2]:
                return False
        return True

    def _merge_assignments(self, assignment1, assignment2, vars1, vars2):
        """
        Merge two assignments into one, respecting common variables.
        """
        merged_assignment = []
        for var in vars1:
            merged_assignment.append(assignment1[vars1.index(var)])
        for var in vars2:
            if var not in vars1:
                merged_assignment.append(assignment2[vars2.index(var)])
        return tuple(merged_assignment)

    def normalize(self):
        """
        Normalize the factor so that the sum of all probabilities equals 1.

        Returns:
            A new factor that is normalized.
        """
        total_prob = sum(self.table.values())
        if total_prob == 0:
            return self  # If the total probability is zero, return the factor as is to avoid division by zero.

        normalized_table = {
            assignment: prob / total_prob for assignment, prob in self.table.items()
        }
        return Factor(self.variables, normalized_table)

    def normalize_over(self, normalize_vars):
        """
        Normalize the factor over a subset of variables.

        Args:
            normalize_vars: A list of variables to normalize over.

        Returns:
            A new factor that is normalized over the specified variables.
        """
        normalize_indices = [self.variables.index(var) for var in normalize_vars]
        other_indices = [
            i for i in range(len(self.variables)) if i not in normalize_indices
        ]

        # Group entries based on the "other" variables (those not being normalized)
        grouped_entries = {}
        for assignment, prob in self.table.items():
            other_assignment = tuple(assignment[i] for i in other_indices)
            if other_assignment not in grouped_entries:
                grouped_entries[other_assignment] = 0
            grouped_entries[other_assignment] += prob

        # Normalize each entry based on the group
        normalized_table = {}
        for assignment, prob in self.table.items():
            other_assignment = tuple(assignment[i] for i in other_indices)
            total_prob = grouped_entries[other_assignment]
            if total_prob == 0:
                raise "div by 0"
            else:
                normalized_table[assignment] = prob / total_prob

        return Factor(self.variables, normalized_table)

    def rename_variable(self, old_name, new_name):
        """
        Rename a variable in the factor.

        Args:
            old_name: The old variable name.
            new_name: The new variable name.

        Returns:
            A new factor with the variable renamed.
        """
        if old_name not in self.variables:
            return self
        self.variables[self.variables.index(old_name)] = new_name
        return self


if __name__ == "__main__":
    # Example usage

    # Define two factors using the new Factor class
    factor1 = Factor(["A", "B"], {(0, 0): 0.1, (0, 1): 0.9, (1, 0): 0.7, (1, 1): 0.3})

    factor2 = Factor(["B", "C"], {(0, 0): 0.4, (0, 1): 0.6, (1, 0): 0.8, (1, 1): 0.2})

    # Print the initial factors
    print("Factor 1:")
    print(factor1)

    print("\nFactor 2:")
    print(factor2)

    # Multiply the factors
    product_factor = factor1.multiply(factor2)
    print("\nAfter multiplication:")
    print(product_factor)

    # Eliminate variable B
    eliminated_factor = product_factor.sum_out("B")
    print("\nAfter eliminating B:")
    print(eliminated_factor)

    # Restrict variable C to 1 (e.g., C = 1 is evidence)
    restricted_factor = eliminated_factor.restrict("C", 1)
    print("\nAfter restricting C to 1:")
    print(restricted_factor)

    x = symbols("x")

    # Define two factors using the new Factor class
    factor1 = Factor(["A", "B"], {(0, 0): x, (0, 1): 1 - x, (1, 0): 0.7, (1, 1): 0.3})
    factor2 = Factor(["B", "C"], {(0, 0): 0.4, (0, 1): 0.6, (1, 0): 0.8, (1, 1): 0.2})

    factor = factor1.multiply(factor2).sum_out("B").normalize_over("A")
    expr = factor.table[(0, 0)]
    print(factor)
    print(solveset(Eq(expr, 1 / 3), x))
