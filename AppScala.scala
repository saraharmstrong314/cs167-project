package edu.ucr.cs.cs167.sdera006

import edu.ucr.cs.bdlab.beast.JavaSpatialSparkContext
import org.apache.spark.beast.{CRSServer, SparkSQLRegistration}
import org.apache.spark.{SparkConf}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._

object AppScala {

  def main(args: Array[String]): Unit = {
    if (args.length < 3) {
      System.err.println("Usage: WildfireDataPreparation <wildfireCSV> <countyShapefileZip> <outputParquet>")
      System.exit(1)
    }

    val wildfireCSV = args(0)
    val countyZip = args(1)
    val outputParquet = args(2)

    // 1. Create SparkConf and SparkSession
    val conf = new SparkConf()
      .setAppName("Wildfire Data Preparation (Spark + Beast, Scala)")
      .setMaster("local[*]")

    val spark: SparkSession = SparkSession.builder()
      .config(conf)
      .getOrCreate()

    // 2. Create Beast JavaSpatialSparkContext (old Beast style)
    val sparkContext = new JavaSpatialSparkContext(spark.sparkContext)

    // 3. (Optional) Start the CRS server if you need transformations
    CRSServer.startServer(sparkContext)

    // 4. Register geometry UDT and UDFs (ST_Point, ST_Contains, etc.)
    SparkSQLRegistration.registerUDT()
    SparkSQLRegistration.registerUDF(spark)

    // 5. Load Wildfire CSV
    //    If it's tab-delimited, we specify "delimiter" -> "\t".
    val wildfiresRaw: Dataset[Row] = spark.read
      .option("header", "true")
      .option("inferSchema", "false")
      .option("delimiter", "\t")
      .csv(wildfireCSV)

    // 6. Keep only the needed columns
    var wildfires = wildfiresRaw.select(
      col("x"),
      col("y"),
      col("acq_date"),
      col("frp"),
      col("acq_time"),
      col("ELEV_mean"),
      col("SLP_mean"),
      col("EVT_mean"),
      col("EVH_mean"),
      col("CH_mean"),
      col("TEMP_ave"),
      col("TEMP_min"),
      col("TEMP_max")
    )

    // 7. Fix the 'frp' column: "12.3,45.7" -> take "12.3" -> cast to double
    wildfires = wildfires.withColumn(
      "frp",
      expr("double(split(frp, ',')[0])")
    )

    // 8. Cast numeric columns to double
    val numericCols = Array(
      "x", "y", "frp",
      "ELEV_mean", "SLP_mean", "EVT_mean", "EVH_mean",
      "CH_mean", "TEMP_ave", "TEMP_min", "TEMP_max"
    )

    numericCols.foreach { c =>
      wildfires = wildfires.withColumn(c, col(c).cast("double"))
    }

    // 9. Create geometry column using ST_CreatePoint(x, y) (if your version has it)
    //    If you get an error about ST_CreatePoint, try ST_Point or ST_GeomFromText
    wildfires = wildfires.withColumn(
      "geometry",
      expr("ST_CreatePoint(x, y)")
    )

    // 10. Load the county shapefile
    val counties = spark.read
      .format("shapefile")
      .load(countyZip)
      .select(
        col("geometry").alias("county_geom"),
        col("GEOID")
      )

    // 11. Create temp views for SQL
    wildfires.createOrReplaceTempView("wildfires")
    counties.createOrReplaceTempView("counties")

    // 12. Spatial join with ST_Contains
    val joinSQL =
      s"""
         |SELECT
         |  w.x, w.y, w.acq_date, w.frp, w.acq_time,
         |  w.ELEV_mean, w.SLP_mean, w.EVT_mean, w.EVH_mean,
         |  w.CH_mean, w.TEMP_ave, w.TEMP_min, w.TEMP_max,
         |  c.GEOID AS County, w.geometry
         |FROM wildfires w JOIN counties c
         |ON ST_Contains(c.county_geom, w.geometry)
         |""".stripMargin

    val joined = spark.sql(joinSQL)

    // 13. Drop geometry column
    val finalDF = joined.drop("geometry")

    // 14. Write Parquet
    finalDF.write.mode("overwrite").parquet(outputParquet)

    println(s"Done! Created Parquet at: $outputParquet")

    spark.stop()
  }
}
