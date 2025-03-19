package edu.ucr.cs.cs167.jjin060;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class App {
    public static void main(String[] args) {
        // Hardcoded start and end dates
        String startDate = "01-01-2022";
        String endDate = "12-31-2022";    

        SparkSession spark = SparkSession.builder()
                .appName("Wildfire Temporal Analysis")
                .config("spark.sql.parquet.enableVectorizedReader", "false")
                .getOrCreate();

        // Load county dataset and filter for California (STATEFP = "06")
        Dataset<Row> countyDF = spark.read().format("parquet")
                .load("/path/to/tl_2018_us_county.parquet")
                .filter("STATEFP = '06'")
                .select("GEOID", "NAME");

        // Load wildfire dataset
        Dataset<Row> wildfireDF = spark.read().format("parquet")
                .load("/path/to/wildfiredb_10k.parquet")
                .withColumn("acq_date", functions.to_date(functions.col("acq_date"), "yyyy-MM-dd"))
                .filter(functions.col("acq_date").between(
                        functions.to_date(functions.lit(startDate), "MM-dd-yyyy"),
                        functions.to_date(functions.lit(endDate), "MM-dd-yyyy"))
                );

        // Perform an equi-join on GEOID = County
        Dataset<Row> joinedDF = wildfireDF.join(countyDF, wildfireDF.col("County").equalTo(countyDF.col("GEOID")));

        // Aggregate total fire intensity per county
        Dataset<Row> resultDF = joinedDF.groupBy("NAME")
                .agg(functions.sum("frp").alias("total_fire_intensity"))
                .orderBy(functions.desc("total_fire_intensity"));

        // Save the result as CSV
        resultDF.coalesce(1)
                .write()
                .option("header", "true")
                .csv("/output/wildfires_California.csv");

        // Display results
        resultDF.show(10, false);

        spark.stop();
    }
}
