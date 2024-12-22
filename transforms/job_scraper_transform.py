from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf, lit, when, regexp_replace, trim, upper
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, 
    BooleanType, TimestampType
)
import logging
from typing import List
from datetime import datetime
import pandas as pd
import time
import csv
import random


logger = logging.getLogger(__name__)
_total_jobs_scraped = 0  # Global counter for total jobs scraped

# Updated schema to match jobspy output
JOB_SCHEMA = StructType([
    StructField("title", StringType(), True),
    StructField("company", StringType(), True),
    StructField("location", StringType(), True),
    StructField("description", StringType(), True),
    StructField("job_url", StringType(), True),
    StructField("date_posted", StringType(), True),
    StructField("site_name", StringType(), True),
    StructField("salary_min", FloatType(), True),
    StructField("salary_max", FloatType(), True),
    StructField("salary_period", StringType(), True)
])

def fetch_jobs_transform(
    spark: SparkSession, 
    cities: DataFrame,
    search_terms: DataFrame,
    output: DataFrame = None,
    hours_old: int = 7200,
    results_wanted: int = 5000,
    max_workers: int = 10
) -> DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from jobspy import scrape_jobs
    import pandas as pd
    
    # Convert input DataFrames to lists
    city_list = [f"{row.city}, {row.state}" for row in cities.collect()]
    search_term_list = [row.term for row in search_terms.collect()]
    
    logger.info(f"Processing {len(city_list)} cities and {len(search_term_list)} search terms")
    
    def fetch_jobs_for_params(city, search_term):
        """Helper function to fetch jobs for a specific city and search term"""
        try:
            time.sleep(2)

            jobs = scrape_jobs(
                site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
                search_term=search_term,
                location=city,
                results_wanted=results_wanted,
                hours_old=hours_old,
                country_indeed='USA'
            )
            global _total_jobs_scraped
            _total_jobs_scraped += len(jobs)
            logger.info(f"Found {len(jobs)} jobs for '{search_term}' in {city} (Total jobs scraped: {_total_jobs_scraped})")
            return jobs
        except Exception as e:
            logger.error(f"Error fetching jobs for {city} - {search_term}: {str(e)}")
            return pd.DataFrame()

    all_jobs = []
    
    # Use ThreadPoolExecutor for concurrent fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create all combinations of cities and search terms
        future_to_params = {
            executor.submit(fetch_jobs_for_params, city, term): (city, term)
            for city in city_list
            for term in search_term_list
        }
        
        # Process completed futures
        for future in as_completed(future_to_params):
            city, term = future_to_params[future]
            try:
                jobs_df = future.result()
                if not jobs_df.empty:
                    all_jobs.append(jobs_df)
            except Exception as e:
                logger.error(f"Error processing results for {city} - {term}: {str(e)}")
    
    # If no jobs found, return empty DataFrame with schema
    if not all_jobs:
        logger.warning("No jobs found for any city/search term combination")
        return spark.createDataFrame([], schema=JOB_SCHEMA)
    
    # Combine all results
    try:
        combined_df = pd.concat(all_jobs, ignore_index=True)
        logger.info(f"Total jobs found across all searches: {len(combined_df)}")
        
        # Print the actual columns for debugging
        logger.info(f"Available columns: {combined_df.columns.tolist()}")
        
        # Create a mapping of required columns
        column_mapping = {
            'title': 'title',
            'company': 'company',
            'location': 'location',
            'description': 'description',
            'job_url': 'job_url',
            'date_posted': 'date_posted',
            'site_name': 'site',  # Map site_name to site
            'salary_min': 'min_amount',
            'salary_max': 'max_amount',
            'salary_period': 'interval'
        }
        
        # Select and rename columns
        selected_columns = []
        for target_col, source_col in column_mapping.items():
            if source_col in combined_df.columns:
                combined_df[target_col] = combined_df[source_col]
                selected_columns.append(target_col)
            else:
                # Add empty column if it doesn't exist
                combined_df[target_col] = None
                selected_columns.append(target_col)
        
        # Select only the mapped columns
        final_df = combined_df[selected_columns]
        
        # Remove duplicates
        final_df = final_df.drop_duplicates(subset=['job_url'])
        
        # Convert pandas DataFrame to Spark DataFrame with explicit schema
        jobs_df = spark.createDataFrame(final_df, schema=JOB_SCHEMA)
        
        # Apply normalization
        jobs_df = normalize_spark_dataframe(jobs_df)
        
        # Add metadata columns
        jobs_df = jobs_df.withColumn(
            "fetch_timestamp", 
            lit(datetime.now().isoformat())
        )
        
        # Log success
        logger.info(f"Successfully processed {jobs_df.count()} unique jobs (Total jobs scraped: {_total_jobs_scraped})")        
        return jobs_df
        
    except Exception as e:
        logger.error(f"Error processing combined results: {str(e)}")
        raise

def normalize_spark_dataframe(df: DataFrame) -> DataFrame:
    """
    Normalize the Spark DataFrame by cleaning and standardizing the data.
    """
    # Clean text fields
    text_columns = ["title", "company", "description", "location"]
    for col_name in text_columns:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                trim(regexp_replace(col(col_name), r'\s+', ' '))
            )
    
    # Clean and standardize date_posted
    if "date_posted" in df.columns:
        df = df.withColumn(
            "date_posted",
            trim(col("date_posted"))
        )

    
    # Ensure salary fields are numeric
    for col_name in ["salary_min", "salary_max"]:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                col(col_name).cast(FloatType())
            )
    
    return df
