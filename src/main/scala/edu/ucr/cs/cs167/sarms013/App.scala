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

object App {

    def main(args : Array[String]) {
        
        /*
            Command-line arguments. 
            Might need extra arguments depending on the task.
        */
        val taskNumber = args(0)
        val inputFile = args(1)

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


        try {
            taskNumber match {

                // TODO: Task 1: Preparation
                case "1" =>




                // TODO: Task 2: Spatio-temporal analysis
                case "2" =>




                // TODO: Task 3: Temporal analysis
                case "3" =>




                // TODO: Task 4: Fire Intensity Prediction
                case "4" =>

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
            }
        }
        
    }


  

}
