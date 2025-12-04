def calculate_BI(rej_power: float, dt: float) -> float:
    """
    Calculate the background interference (BI) using given parameters.

    The formula used to calculate the BI is based on the activity of a source,
    the rejection power, the exposure time, and other constants. The formula is:

    BI = (correction_factor * Probability * (1 - rej_power) * Activity^2) /
        (LMO_MASS * second_in_year) * dt / milliseconds_in_second

    Parameters
    ----------
    rej_power : float
        The rejection power (fraction of rejection).
    dt : float
        The exposure time in seconds.

    Returns
    -------
    float
        The calculated background interference (BI) value in appropriate units.
    """
    # Constants used in the BI calculation
    Activity = 2.93E-03  # Activity in Bq (N/s)
    LMO_MASS = 0.28  # Mass of LMO in kg
    second_in_year = 3600 * 24 * 365  # Number of seconds in a year
    Probability = 3.41E-04  # Given probability factor
    correction_factor = 1  # Factor for corrections
    milliseconds_in_second = 1000  # Conversion factor from seconds to milliseconds
    K = (correction_factor * Probability * Activity * Activity /
            LMO_MASS * second_in_year * dt / milliseconds_in_second)
    # BI calculation formula based on provided parameters
    return K *  (1 - rej_power)

Activity = 2.93E-03  # Activity in Bq (N/s)
LMO_MASS = 0.28  # Mass of LMO in kg
second_in_year = 3600 * 24 * 365  # Number of seconds in a year
Probability = 3.41E-04  # Given probability factor
correction_factor = 1  # Factor for corrections
dt = 0.8  # Example exposure time in seconds
milliseconds_in_second = 1000  # Conversion factor from seconds to milliseconds
K = (correction_factor * Probability * Activity * Activity /
        LMO_MASS * second_in_year * dt / milliseconds_in_second)