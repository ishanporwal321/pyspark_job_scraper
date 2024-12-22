from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Spark
spark = SparkSession.builder \
    .appName("JobScraper") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.session.timeZone", "UTC") \
    .getOrCreate()

# Define input schema for cities
cities_schema = StructType([
    StructField("city", StringType(), False),
    StructField("state", StringType(), False)
])

search_terms_schema = StructType([
    StructField("term", StringType(), False)
])

# Create dynamic input data for cities
cities_data = [
    ("Louisville", "KY"),
    ("Lexington", "KY"),
    ("Bowling Green", "KY"),
    ("Owensboro", "KY"),
    ("Covington", "KY"),
]


manufacturing_terms = [
    "Manufacturing Engineer",
    "Industrial Designer",
    "Process Engineer",
    "Mechanical Engineer",
    "Quality Assurance Specialist",
    "Production Manager",
    "Automation Technician",
    "CNC Machinist",
    "Supply Chain Analyst",
    "Lean Manufacturing Consultant"
]

# Create DataFrames
cities_df = spark.createDataFrame(cities_data, schema=cities_schema)
search_terms_df = spark.createDataFrame(
    [(term,) for term in manufacturing_terms], 
    schema=search_terms_schema
)

# Import and use the transform
from transforms.job_scraper_transform import fetch_jobs_transform

try:
    # Create output directory if it doesn't exist
    output_dir = "job_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run the transform
    results_df = fetch_jobs_transform(
        spark=spark,
        cities=cities_df,
        search_terms=search_terms_df,
        hours_old=7200,
        results_wanted=5000,
        max_workers=10
    )

    # Save results to CSV
    csv_path = f"{output_dir}/jobs_{timestamp}.csv"
    results_df.toPandas().to_csv(csv_path, index=False)
    print(f"\nResults saved to CSV: {csv_path}")

    # Show results and statistics
    print("\nJobs found:")
    results_df.select("title", "company", "location", "salary_min", "salary_max").show(5, truncate=False)
    
    total_jobs = results_df.count()
    print(f"\nTotal jobs found: {total_jobs}")
    
    # Group by location and job title
    print("\nJobs by location:")
    results_df.groupBy("location").count().show(truncate=False)
    
    print("\nTop job titles:")
    results_df.groupBy("title").count().orderBy("count", ascending=False).show(5, truncate=False)

    # Save detailed statistics to a separate CSV
    stats_df = spark.createDataFrame([
        ("Total Jobs", total_jobs),
        ("Unique Companies", results_df.select("company").distinct().count()),
        ("Unique Locations", results_df.select("location").distinct().count()),
        ("Unique Job Titles", results_df.select("title").distinct().count())
    ], ["Metric", "Value"])
    
    stats_path = f"{output_dir}/stats_{timestamp}.csv"
    stats_df.toPandas().to_csv(stats_path, index=False)
    print(f"\nStatistics saved to: {stats_path}")

    # Save company-wise breakdown
    company_stats_path = f"{output_dir}/company_stats_{timestamp}.csv"
    results_df.groupBy("company").count().orderBy("count", ascending=False) \
        .toPandas().to_csv(company_stats_path, index=False)
    print(f"Company-wise statistics saved to: {company_stats_path}")

except Exception as e:
    print(f"Error running job transform: {str(e)}")
    raise

finally:
    spark.stop()