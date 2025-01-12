def linear_schedule(initial_value, final_value=0.0):
    """
    Returns a scheduler function for linearly decreasing a value over time.

    Args:
        initial_value (float): The starting value.
        final_value (float): The final value to reach. Defaults to 0.0.

    Returns:
        function: A scheduler function that takes `progress` (a float from 0.0 to 1.0)
                  and returns the interpolated value.
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
    if isinstance(final_value, str):
        final_value = float(final_value)

    assert initial_value > final_value, "Initial value must be greater than final value."

    def scheduler(progress: float) -> float:
        """
        Compute the value based on the current progress.

        Args:
            progress (float): A float between 0.0 (start) and 1.0 (end).

        Returns:
            float: The interpolated value.
        """
        return final_value + progress * (initial_value - final_value)

    return scheduler
