package edu.ucr.cs.cs167.jjin060

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object App {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Wildfire Temporal Analysis")
      .config("spark.sql.parquet.enableVectorizedReader", "false")
      .getOrCreate()

    import spark.implicits._

    // Hardcoded start and end dates
    val startDate = "01-01-2020"
    val endDate = "12-31-2020"

    // Load county dataset and filter for California (STATEFP = "06")
    val countyDF = spark.read.format("parquet").load("/path/to/tl_2018_us_county.parquet")
      .filter($"STATEFP" === "06")
      .select($"GEOID", $"NAME".alias("county_name"))

    // Load wildfire dataset
    val wildfireDF = spark.read.format("parquet").load("/path/to/wildfiredb_10k.parquet")
      .withColumn("acq_date", to_date(col("acq_date"), "yyyy-MM-dd"))
      .filter(col("acq_date").between(to_date(lit(startDate), "MM-dd-yyyy"), to_date(lit(endDate), "MM-dd-yyyy")))

    // Perform an equi-join on GEOID = County
    val joinedDF = wildfireDF.join(countyDF, wildfireDF("County") === countyDF("GEOID"))

    // Aggregate total fire intensity per county
    val resultDF = joinedDF.groupBy("county_name")
      .agg(sum("frp").alias("total_fire_intensity"))
      .orderBy(desc("total_fire_intensity"))

    // Save the result as CSV
    resultDF.coalesce(1).write.option("header", "true").csv("/output/wildfires_California.csv")

    // Display results
    resultDF.show(10, false)

    spark.stop()
  }
}