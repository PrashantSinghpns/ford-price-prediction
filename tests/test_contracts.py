"""Contract tests for Ford data validation and cleaning."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Import the local training module without installing a package.
from train import load_and_clean  # Test the public data-loading contract.


class FordDataContractTests(unittest.TestCase):
    """Verify that quality checks protect the regression workflow."""

    def test_removes_invalid_price_and_future_year(self):
        rows = pd.DataFrame(
            [
                {"model": " Focus ", "year": 2019, "price": 12000, "transmission": "Manual", "mileage": 20000, "fuelType": "Petrol", "tax": 145, "mpg": 55.4, "engineSize": 1.0},  # Valid record with whitespace to clean.
                {"model": "Fiesta", "year": 2060, "price": 9000, "transmission": "Manual", "mileage": 30000, "fuelType": "Petrol", "tax": 145, "mpg": 55.4, "engineSize": 1.0},  # Implausible future year.
                {"model": "Fiesta", "year": 2018, "price": 0, "transmission": "Manual", "mileage": 30000, "fuelType": "Petrol", "tax": 145, "mpg": 55.4, "engineSize": 1.0},  # Invalid target.
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ford.csv"  # Create an isolated temporary input file.
            rows.to_csv(path, index=False)  # Write the controlled test data.
            cleaned = load_and_clean(path)  # Run the production cleaning function.
        self.assertEqual(len(cleaned), 1)  # Keep only the defensible record.
        self.assertEqual(cleaned.iloc[0]["model"], "Focus")  # Confirm label whitespace is removed.


if __name__ == "__main__":
    unittest.main()  # Allow this file to run directly during debugging.
