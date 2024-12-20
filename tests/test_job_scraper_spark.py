import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd

# Import the transformation logic (adjust path as needed)
from transforms.job_scraper_transform import (
    normalize_spark_dataframe,
    validate_spark_dataframe,
    JOB_SCHEMA
)

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return (SparkSession.builder
            .master("local[*]")
            .appName("JobScraperTest")
            .getOrCreate())

@pytest.fixture
def sample_cities_df(spark):
    """Create a sample cities DataFrame."""
    data = [("San Francisco",), ("New York",), ("Seattle",)]
    return spark.createDataFrame(data, ["city"])

@pytest.fixture
def sample_search_terms_df(spark):
    """Create a sample search terms DataFrame."""
    data = [("software engineer",), ("data scientist",)]
    return spark.createDataFrame(data, ["term"])

def test_job_scraping_pipeline(spark, sample_cities_df, sample_search_terms_df):
    """Test the entire job scraping pipeline."""
    from jobspy import scrape_jobs
    
    # Get test data
    city = sample_cities_df.first().city
    term = sample_search_terms_df.first().term
    
    # Fetch some real jobs
    raw_jobs = scrape_jobs(
        site_name=["indeed", "glassdoor", "google", "linkedin"],
        search_term=term,
        location=city,
        results_wanted=5000,  # Small number for testing
        hours_old=7200
    )
    
    # Convert pandas DataFrame to Spark DataFrame
    jobs_df = spark.createDataFrame(raw_jobs, schema=JOB_SCHEMA)
    
    # Test normalization
    normalized_df = normalize_spark_dataframe(jobs_df)
    
    # Assertions
    assert normalized_df.count() > 0, "No jobs were found"
    assert validate_spark_dataframe(normalized_df), "DataFrame validation failed"
    
    # Test specific columns
    assert "title" in normalized_df.columns
    assert "company" in normalized_df.columns
    
    # Print sample results
    print("\nSample jobs found:")
    normalized_df.select("title", "company", "location").show(5, truncate=False)

def test_normalize_spark_dataframe(spark):
    """Test the normalization function with mock data."""
    # Create sample data
    sample_data = [
        {
            "title": "  Senior Engineer  ",
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "min_amount": "100000",
            "max_amount": "150000",
            "currency": None
        }
    ]
    
    # Convert to Spark DataFrame
    input_df = spark.createDataFrame(sample_data)
    
    # Apply normalization
    result_df = normalize_spark_dataframe(input_df)
    
    # Convert to pandas for easier testing
    result = result_df.toPandas()
    
    # Assertions
    assert result.iloc[0]["title"] == "Senior Engineer"  # Check whitespace cleaning
    assert result.iloc[0]["currency"] == "USD"  # Check default currency
    assert isinstance(result.iloc[0]["min_amount"], float)  # Check type conversion

def test_validate_spark_dataframe(spark):
    """Test the validation function."""
    # Valid DataFrame
    valid_data = [
        {
            "title": "Software Engineer",
            "company": "Tech Corp",
            "location": "San Francisco",
            "description": "Great job opportunity",
            "salary_min": 100000.0,
            "salary_max": 150000.0,
            "currency": "USD"
        }
    ]
    valid_df = spark.createDataFrame(valid_data)
    assert validate_spark_dataframe(valid_df)
    
    # Invalid DataFrame (missing required columns)
    invalid_data = [
        {
            "title": "Software Engineer",
            "company": "Tech Corp"
        }
    ]
    invalid_df = spark.createDataFrame(invalid_data)
    assert not validate_spark_dataframe(invalid_df)

if __name__ == "__main__":
    # Create Spark session
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("JobScraperTest") \
        .getOrCreate()
    
    try:
        # Create test fixtures
        sample_cities_df = spark.createDataFrame([("Kentucky","KY")], ["city"])
        sample_search_terms_df = spark.createDataFrame([("Manufacturing Engineer", "Industrial Designer", "Process Engineer", "Mechanical Engineer", "Quality Assurance Specialist", "Production Manager", "Automation Technician", "CNC Machinist", "Supply Chain Analyst", "Lean Manufacturing Consultant")], ["term"])
        
        # Run tests
        print("Running job scraping pipeline test...")
        test_job_scraping_pipeline(spark, sample_cities_df, sample_search_terms_df)
        
        print("\nRunning normalization test...")
        test_normalize_spark_dataframe(spark)
        
        print("\nRunning validation test...")
        test_validate_spark_dataframe(spark)
        
        print("\nAll tests completed successfully!")
        
    finally:
        spark.stop()