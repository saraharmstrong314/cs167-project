package edu.ucr.cs.cs167.sarms013

import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.Map

object BeastScala {
  def main(args: Array[String]): Unit = {
    // Initialize Spark context
    val conf = new SparkConf().setAppName("Beast Example")

    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")

    val spark: SparkSession.Builder = SparkSession.builder().config(conf)
    val sparkSession: SparkSession = spark.getOrCreate()
    val sparkContext = sparkSession.sparkContext
    SparkSQLRegistration.registerUDT
    SparkSQLRegistration.registerUDF(sparkSession)

    //val operation: String = args(0)
    val inputfile: String = args(0)
    val countyName : String = args(1)
    val countyFile = "tl_2018_us_county.zip"
    try {
      // Import Beast features
      import edu.ucr.cs.bdlab.beast._
      val t1 = System.nanoTime()
      var validOperation = true

      // TODO: Temporal Analysis
      var df : DataFrame = sparkSession.read.parquet(inputfile)
      df.createOrReplaceTempView("wildfire_data")

      var counties : DataFrame = sparkSession.read.format("shapefile").load(countyFile)
      counties.createOrReplaceTempView("counties")

      val countyGEOID = sparkSession.sql(
          s"""
            SELECT GEOID
            FROM counties
            WHERE NAME = "$countyName" AND STATEFP = "06"
          """).collect().head.getString(0)

      // acq_date column is of type string, need to be date type and put leading 0 for months < 10
      val cleanedData = df.withColumn("acq_date", to_date(col("acq_date"), "yyyy-MM-dd"))
        .withColumn("year", year(col("acq_date")))
        .withColumn("month", month(col("acq_date")))
        .withColumn("month", lpad(col("month").cast("string"), 2, "0"))
        .withColumn("year_month", concat_ws("-", col("year"), col("month")))

      cleanedData.createOrReplaceTempView("wildfire_data")

      val result = sparkSession.sql(
        s"""
           SELECT year, month, SUM(frp) AS total_fire_intensity
           FROM wildfire_data
           WHERE County = '$countyGEOID'
           GROUP BY year, month
           ORDER BY year, month
        """)

      val formattedResult = result.withColumn(
        "year_month",
        concat_ws("-", col("year"), lpad(col("month").cast("string"), 2, "0"))
      ).select("year_month", "total_fire_intensity")

      val outputFile = s"wildfires_$countyName.csv"
      formattedResult.write.option("header", "true").csv(outputFile)

      val t2 = System.nanoTime()
      if (validOperation)
        println(s"Operation on file '$inputfile' took ${(t2 - t1) * 1E-9} seconds")
      else
        Console.err.println(s"Invalid operation")
    } finally {
      sparkSession.stop()
    }
  }
}