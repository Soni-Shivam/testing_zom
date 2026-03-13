"""
Feature Domain 3: Contextual Features
========================================
Context encoder that transforms environmental variables into
neural-network-ready representations:
  - Cyclical time encoding (hour, day-of-week)
  - Day type one-hot (Weekday/Weekend/Holiday/Festive)
  - Weather proxy (hot/cold/rainy)
"""

import numpy as np
from typing import Dict

# Canonical day-type categories
DAY_TYPES = ["Weekday", "Weekend", "Holiday", "Festive"]
DAY_TYPE_TO_IDX = {dt: i for i, dt in enumerate(DAY_TYPES)}

# Canonical weather categories
WEATHER_TYPES = ["hot", "cold", "rainy"]
WEATHER_TO_IDX = {w: i for i, w in enumerate(WEATHER_TYPES)}

# City-to-weather mapping (simplified proxy based on Indian climate zones)
CITY_WEATHER_MAP = {
    "Delhi-NCR":  "hot",
    "Mumbai":     "rainy",
    "Chennai":    "hot",
    "Hyderabad":  "hot",
    "Bangalore":  "cold",
}


class ContextEncoder:
    """
    Encodes environmental context variables into fixed-dimensional
    feature vectors. Designed for sub-millisecond online computation.

    Total output: 11 features
      - Cyclical hour:  2-dim
      - Cyclical day:   2-dim
      - Day type:       4-dim (one-hot)
      - Weather proxy:  3-dim (one-hot)
    """

    def compute_all(
        self,
        hour_of_day: int,
        day_of_week: int,
        is_weekend: bool = False,
        is_holiday: bool = False,
        is_festive: bool = False,
        city: str = "Delhi-NCR",
    ) -> Dict[str, np.ndarray]:
        """
        Compute all contextual features.

        Args:
            hour_of_day: Hour (0-23).
            day_of_week: Day of week (0=Monday, 6=Sunday).
            is_weekend: Whether it's a weekend.
            is_holiday: Whether it's a public holiday.
            is_festive: Whether it's a festive period (Diwali, etc.).
            city: City for weather proxy lookup.

        Returns:
            Dictionary of named feature arrays.
        """
        return {
            "cyclical_hour":  self.cyclical_hour_encoding(hour_of_day),
            "cyclical_day":   self.cyclical_day_encoding(day_of_week),
            "day_type":       self.day_type_onehot(is_weekend, is_holiday, is_festive),
            "weather_proxy":  self.weather_proxy(city),
        }

    @staticmethod
    def cyclical_hour_encoding(hour: int) -> np.ndarray:
        """
        Encode hour h as a 2D cyclical vector:
          (sin(2πh/24), cos(2πh/24))

        This preserves the circular nature of time — hour 23 is
        close to hour 0 in the encoding space.

        Args:
            hour: Hour of day (0-23).

        Returns:
            NumPy array of shape (2,).
        """
        theta = 2.0 * np.pi * hour / 24.0
        return np.array([np.sin(theta), np.cos(theta)])

    @staticmethod
    def cyclical_day_encoding(day_of_week: int) -> np.ndarray:
        """
        Encode day-of-week d as a 2D cyclical vector:
          (sin(2πd/7), cos(2πd/7))

        Args:
            day_of_week: Day (0=Monday, 6=Sunday).

        Returns:
            NumPy array of shape (2,).
        """
        theta = 2.0 * np.pi * day_of_week / 7.0
        return np.array([np.sin(theta), np.cos(theta)])

    @staticmethod
    def day_type_onehot(
        is_weekend: bool = False,
        is_holiday: bool = False,
        is_festive: bool = False,
    ) -> np.ndarray:
        """
        One-hot encode the day type over {Weekday, Weekend, Holiday, Festive}.

        Priority: Festive > Holiday > Weekend > Weekday.

        Returns:
            NumPy array of shape (4,) one-hot encoded.
        """
        vec = np.zeros(len(DAY_TYPES), dtype=np.float64)

        if is_festive:
            vec[DAY_TYPE_TO_IDX["Festive"]] = 1.0
        elif is_holiday:
            vec[DAY_TYPE_TO_IDX["Holiday"]] = 1.0
        elif is_weekend:
            vec[DAY_TYPE_TO_IDX["Weekend"]] = 1.0
        else:
            vec[DAY_TYPE_TO_IDX["Weekday"]] = 1.0

        return vec

    @staticmethod
    def weather_proxy(city: str) -> np.ndarray:
        """
        Map city to a weather category {hot, cold, rainy}
        and return as one-hot vector.

        Args:
            city: Delivery zone / city name.

        Returns:
            NumPy array of shape (3,) one-hot encoded.
        """
        weather = CITY_WEATHER_MAP.get(city, "hot")
        vec = np.zeros(len(WEATHER_TYPES), dtype=np.float64)
        idx = WEATHER_TO_IDX.get(weather, 0)
        vec[idx] = 1.0
        return vec
