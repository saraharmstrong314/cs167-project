package edu.ucr.cs.cs167.pyoko002

import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._

object WildfireSpatioTemporalAnalysis {
  def main(args: Array[String]): Unit = {

    if (args.length != 3) {
      println("Usage: WildfireSpatioTemporalAnalysis <start_date> <end_date> <output_directory>")
      sys.exit(1)
    }

    val startDate = args(0)  // "01/01/2016"
    val endDate = args(1)    // "12/31/2017"
    val outputDir = args(2)  // Output directory for Shapefile

    val conf = new SparkConf().setAppName("WildfireSpatioTemporalAnalysis").setMaster("local[*]")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    SparkSQLRegistration.registerUDT()
    SparkSQLRegistration.registerUDF(spark)

    // Load wildfire dataset
    val wildfireDF = spark.read.parquet("wildfiredb_sample.parquet")
    wildfireDF.createOrReplaceTempView("wildfire_data")

    // Load county dataset (Shapefile)
    val countyDF = spark.read.format("shapefile").load("tl_2018_us_county.zip")
    countyDF.createOrReplaceTempView("counties")

    // Query to filter and sum fire intensity
    val filteredWildfires = spark.sql(
      s"""
         |SELECT county, SUM(frp) AS fire_intensity
         |FROM wildfire_data
         |WHERE to_date(acq_date, 'yyyy-MM-dd')
         |      BETWEEN to_date('$startDate', 'MM/dd/yyyy')
         |      AND to_date('$endDate', 'MM/dd/yyyy')
         |GROUP BY county
       """.stripMargin
    )
    filteredWildfires.createOrReplaceTempView("fire_intensity_data")

    // Ensure `county` matches `NAME` or `GEOID` in `counties`
    val joinedDF = spark.sql(
      s"""
         |SELECT counties.GEOID, counties.NAME, counties.geometry AS g, fire_intensity
         |FROM fire_intensity_data
         |JOIN counties ON fire_intensity_data.county = counties.NAME
       """.stripMargin
    )

    // Output to Shapefile for QGIS visualization
    joinedDF.coalesce(1).write.format("shapefile").save(outputDir)

    println(s"Choropleth map data saved to $outputDir")

    spark.stop()
  }
}
