from pydantic import BaseModel
from typing import List, Optional

class JobSearchParams(BaseModel):
    cities: List[tuple[str, str]]  # List of (city, state) tuples
    search_terms: List[str]
    output_file: str = "jobs.csv"
    max_workers: int = 10
    hours_old: int = 24
    results_wanted: int = 100