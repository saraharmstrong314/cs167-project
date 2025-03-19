package edu.ucr.cs.cs167.sarms013

// /**
//  * @author ${user.name}
//  */


// General
import org.apache.spark.SparkConf


// Task4 4: Machine Learning Imports
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.CrossValidator
// import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.classification.{LogisticRegression, LinearSVCModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{FeatureHasher, StandardScaler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import edu.ucr.cs.bdlab.beast.JavaSpatialSparkContext
import org.apache.spark.beast.{CRSServer, SparkSQLRegistration}
import org.apache.spark.{SparkConf}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._

object BeastScala {

    def main(args : Array[String]) {
        


        /*
            Spark configuration; local or online.
            Should be the same across tasks.
        */
        val conf = new SparkConf
        if (!conf.contains("spark.master"))
            conf.setMaster("local[*]")
        println(s"Using Spark master '${conf.get("spark.master")}'")

        /*
            SparkSession; enables usage of Spark apis.
            Should be the same across tasks.
        */
        val spark = SparkSession
            .builder()
            .appName("CS167 Project")
            .config(conf)
            .getOrCreate()

        /*


            Note:
            All tasks except Task 1 load the input file as parquet.
            This might mean that we can't generalize reading the input
            in this initial setup.
        */
        val taskNumber = args(0)

        try {
            taskNumber match {

                // TODO: Task 1: Preparation
                case "1" =>

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

                // TODO: Task 2: Spatio-temporal analysis
                case "2" =>

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

                    //wildfireDF.show(5);

                    // Load county dataset (Shapefile)
                    val countyDF = spark.read.format("shapefile").load("tl_2018_us_county.zip")
                    countyDF.createOrReplaceTempView("counties")

                    //countyDF.show(5);

                    // Query to filter and sum fire intensity
                    val filteredWildfires = spark.sql(
                        s"""
                           |SELECT County AS county_name, SUM(frp) AS fire_intensity
                           |FROM wildfire_data
                           |WHERE to_date(acq_date, 'yyyy-MM-dd')
                           |      BETWEEN to_date('$startDate', 'MM/dd/yyyy')
                           |      AND to_date('$endDate', 'MM/dd/yyyy')
                           |GROUP BY County
                        """.stripMargin
                    )
                    filteredWildfires.createOrReplaceTempView("fire_intensity_data")

                    //filteredWildfires.show(5);

                    // Ensure `County` in wildfire_data matches `NAME` or `GEOID` in `counties`
                    val joinedDF = spark.sql(
                        s"""
                           |SELECT counties.GEOID, counties.NAME, counties.geometry AS g, fire_intensity
                           |FROM fire_intensity_data
                           |JOIN counties ON fire_intensity_data.county_name = counties.GEOID
                        """.stripMargin
                    )

                    //joinedDF.show(5);
                    // Output to Shapefile for QGIS visualization
                    joinedDF.coalesce(1).write.format("shapefile").save(outputDir)

                    println(s"Choropleth map data saved to $outputDir")

                    spark.stop()


                // TODO: Task 3: Temporal analysis
                case "3" =>

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


                // TODO: Task 4: Fire Intensity Prediction
                case "4" =>
                    /*
                        Command-line arguments. 
                        Might need extra arguments depending on the task.
                    */

                    val inputFile = args(1)

                    val enableDebug = false
                    val enableDBOut = false
                    val enableBestO = false

                    /*
                        Setup DF
                     */
                    val startTime = System.nanoTime

                    if (enableDebug) { System.out.println("Starting task 4...") }
                    var df = spark.read.parquet(inputFile)
                    var originalDF = df
                    if (enableDebug) { System.out.println("[ReadParquet] Success!") } 

                    df.createOrReplaceTempView("CurrentData")
                    if (enableDBOut) { df.printSchema() }
                    if (enableDBOut) { df.show(5) }
                    if (enableDebug) { System.out.println("[CreateOrReplaceTempView] Success!") }

                    df = spark.sql(
                        s"""
                           | SELECT
                           |    AVG(A.ELEV_mean) AS ELEV_mean,
                           |    AVG(A.SLP_mean) AS SLP_mean,
                           |    AVG(A.EVT_mean) AS EVT_mean,
                           |    AVG(A.EVH_mean) AS EVH_mean,
                           |    AVG(A.CH_mean) AS CH_mean,
                           |    AVG(A.TEMP_ave) AS TEMP_ave,
                           |    AVG(A.TEMP_min) AS TEMP_min,
                           |    AVG(A.TEMP_max) AS TEMP_max,
                           |    SUM(A.frp) AS fire_intensity,
                           |    (SELECT YEAR(A.acq_date)) AS Year,
                           |    (SELECT MONTH(A.acq_date)) AS Month,
                           |    A.County
                           | FROM CurrentData A
                           | GROUP BY
                           |    A.County,
                           |    Year,
                           |    Month
                           |""".stripMargin)
                    if (enableDBOut) { df.printSchema() }
                    if (enableDBOut) { df.show(5) }
                    if (enableDebug) { System.out.println("[SQL] Success!")}


                    /*
                        Transformations
                     */
                    val inputColumns = Array(
                        "ELEV_mean", "SLP_mean", "EVT_mean", 
                        "EVH_mean", "CH_mean", "TEMP_ave", 
                        "TEMP_min", "TEMP_max"
                    )
                    val assembledVector = new VectorAssembler()
                        .setInputCols(inputColumns)
                        .setOutputCol("attributes")
                    val standardScaler = new StandardScaler()
                        .setInputCol("attributes")
                        .setOutputCol("features")
                    if (enableDebug) { System.out.println("[CreatedTransforms] Success!") }

                    val linReg = new LinearRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("fire_intensity")
                        .setMaxIter(1000)
                    if (enableDebug) { System.out.println("[CreatedLogisticRegressionClassifier] Success!") }


                    /*
                        Pipeline
                    */
                    val pipeline = new Pipeline()
                        .setStages(Array(
                            assembledVector,
                            standardScaler,
                            linReg
                        ))
                    if (enableDebug) { System.out.println("[CreatedPipeline] Success!") }


                    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
                        .addGrid(linReg.fitIntercept, Array(true,false))
                        .addGrid(linReg.elasticNetParam, Array(0.0, 0.3, 0.8, 1.0))
                        .addGrid(linReg.regParam,	Array(0.01, 0.0001))
                        .addGrid(linReg.maxIter,	Array(10, 100))
                        .addGrid(linReg.tol,	Array(0.0001, 0.01))
                        .build()
                    if (enableDebug) { System.out.println("[CreatedParamGrid] Success!")}


                    /*
                        Evaluators
                     */
                    val regressionEvaluator = new RegressionEvaluator()
                        .setLabelCol("fire_intensity")
                        .setPredictionCol("prediction")
                        .setMetricName("rmse")
                    val chosenEvaluator = regressionEvaluator
                    if (enableDebug) { System.out.println("[CreatedAndChoseEvaluator] Success!")}


                    /*
                        Validators
                     */
                    val trainValidator = new TrainValidationSplit()
                        .setEstimator(pipeline)
                        .setEvaluator(chosenEvaluator)
                        .setEstimatorParamMaps(paramGrid)
                        .setTrainRatio(0.8)
                        .setParallelism(2)
                    val crossValidator = new CrossValidator()
                        .setEstimator(pipeline)
                        .setEvaluator(chosenEvaluator)
                        .setEstimatorParamMaps(paramGrid)
                        .setNumFolds(5)
                        .setParallelism(2)
                    val chosenValidator = crossValidator
                    if (enableDebug) { System.out.println("[CreatedAndChoseValidator] Success!")}

                    if (enableDBOut) { df.show(5) }
                    val Array(trainingData: Dataset[Row], testData: Dataset[Row]) = df.randomSplit(Array(0.8, 0.2))
                    if (enableDebug) { System.out.println(trainingData)}
                    if (enableDebug) { System.out.println("[SplitData] Success!")}

                    val model = chosenValidator.fit(trainingData)
                    if (enableDebug) { System.out.println("[RanCrossValidation] Success!")}


                    /*
                        Best Parameters
                    */
                    if (enableBestO) {
                        val fitIntercept = model.bestModel
                            .asInstanceOf[PipelineModel]
                            .stages(2)
                            .asInstanceOf[LinearRegressionModel]
                            .fitIntercept
                        val netParam = model.bestModel
                            .asInstanceOf[PipelineModel]
                            .stages(2)
                            .asInstanceOf[LinearRegressionModel]
                            .elasticNetParam
                        val regParam = model.bestModel
                            .asInstanceOf[PipelineModel]
                            .stages(2)
                            .asInstanceOf[LinearRegressionModel]
                            .regParam
                        val maxIter = model.bestModel
                            .asInstanceOf[PipelineModel]
                            .stages(2)
                            .asInstanceOf[LinearRegressionModel]
                            .maxIter
                        val tol = model.bestModel
                            .asInstanceOf[PipelineModel]
                            .stages(2)
                            .asInstanceOf[LinearRegressionModel]
                            .tol


                        System.out.println("--Parameters of the best Model--")
                        System.out.println("fitIntercept: ", fitIntercept)
                        System.out.println("elasticNetParam: ", netParam)
                        System.out.println("regParam: ", regParam)
                        System.out.println("maxIter: ", maxIter)
                        System.out.println("Param: ", tol)
                        System.out.println()
                    }
                    if (enableBestO) { System.out.println("[ChoosingBestModel] Success!")}


                    /*
                        Running and Evaluation
                    */
                    val finalRegressionEvaluator = new RegressionEvaluator()
                        .setLabelCol("fire_intensity")
                        .setPredictionCol("prediction")
                        .setMetricName("rmse")

                    val predictions: DataFrame = model.transform(testData)
                    predictions.select(
                        "ELEV_mean", "SLP_mean", "EVT_mean",
                        "EVH_mean", "CH_mean", "TEMP_ave",
                        "TEMP_min", "TEMP_max", "fire_intensity", "prediction"
                    ).show()
                    if (enableDebug) { System.out.println("# of predictions: ", predictions.count()) }
                    if (enableDBOut) { System.out.println("Original DF: "); originalDF.show() }
                    if (enableDBOut) { System.out.println("Scaled DF: "); df.show() }

                    val rsmeResult: Double = finalRegressionEvaluator.evaluate(predictions)
                    System.out.println(s"RMSE of the test set is $rsmeResult")

                    val endTime = System.nanoTime
                    System.out.println("Total Execution Time: " + (endTime - startTime) * 1E-9 + " seconds")


                // TODO: Task 5: Temporal analysis - 2
                case "5" =>

                    val spark = SparkSession.builder()
                      .appName("Wildfire Temporal Analysis")
                      .master("local[*]") // Run Spark locally with all available cores
                      .config("spark.sql.parquet.enableVectorizedReader", "false")
                      .getOrCreate()

                    import spark.implicits._

                    // Hardcoded start and end dates
                    val startDate = "01-01-2020"
                    val endDate = "12-31-2020"

                    // Load county dataset and filter for California (STATEFP = "06")
                    val df = spark.read.parquet("C:/Users/jjin1/CS167/workspace/jjin060_project2/tl_2018_us_county.parquet")
                      .filter($"STATEFP" === "06")
                      .select($"GEOID", $"NAME".alias("county_name"))

                    // Load wildfire dataset
                    val wildfireDF = spark.read.format("parquet").load("C:/Users/jjin1/CS167/workspace/jjin060_project2/wildfiredb_10k.parquet")
                      .withColumn("acq_date", to_date(col("acq_date"), "yyyy-MM-dd"))
                      .filter(col("acq_date").between(to_date(lit(startDate), "MM-dd-yyyy"), to_date(lit(endDate), "MM-dd-yyyy")))

                    // Perform an equi-join on GEOID = County
                    val countyDF = spark.read.parquet("C:/Users/jjin1/CS167/workspace/jjin060_project2/tl_2018_us_county.parquet")
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
        
    }


  

}
