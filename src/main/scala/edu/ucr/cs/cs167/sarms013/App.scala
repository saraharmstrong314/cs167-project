package edu.ucr.cs.cs167.sarms013

// /**
//  * @author ${user.name}
//  */


// General
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler

// Task4 4: Machine Learning Imports
import org.apache.spark.ml.classification.{LogisticRegression, LinearSVCModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer, StandardScaler}
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

                    val enableDebug = true
                    val enableDBOut = true

                    /*
                        Setup DF
                     */
                    if (enableDebug) { System.out.println("Starting task 4...") }

                    var df = spark.read.parquet(inputFile)
                    if (enableDebug) { System.out.println("[ReadParquet] Success!") } 

                    df.createOrReplaceTempView("CurrentData")
                    if (enableDBOut) { df.printSchema() }
                    if (enableDebug) { System.out.println("[CreateOrReplaceTempView] Success!") }

                    /*
                        Transformations
                     */
                    val inputColumns = Array(
                        "ELEV_mean", "SLP_mean", "EVT_mean", 
                        "EVH_mean", "CH_mean", "TEMP_ave", 
                        "TEMP_min", "TEMP_max"
                    )
                    val elevScaler = new StandardScaler()
                        .setInputCol(inputColumns(0))
                        .setOutputCol(inputColumns(0))
                    val slpScaler = new StandardScaler()
                        .setInputCol(inputColumns(1))
                        .setOutputCol(inputColumns(1))
                    val evtScaler = new StandardScaler()
                        .setInputCol(inputColumns(2))
                        .setOutputCol(inputColumns(2))
                    val evhScaler = new StandardScaler()
                        .setInputCol(inputColumns(3))
                        .setOutputCol(inputColumns(3))
                    val chScaler = new StandardScaler()
                        .setInputCol(inputColumns(4))
                        .setOutputCol(inputColumns(4))
                    val tempAveScaler = new StandardScaler()
                        .setInputCol(inputColumns(5))
                        .setOutputCol(inputColumns(5))
                    val tempMinScaler = new StandardScaler()
                        .setInputCol(inputColumns(6))
                        .setOutputCol(inputColumns(6))
                    val tempMaxScaler = new StandardScaler()
                        .setInputCol(inputColumns(7))
                        .setOutputCol(inputColumns(7))
                    if (enableDebug) { System.out.println("[CreatedTransforms] Success!") }

                    
                    /*
                        Pipeline
                    */
                    val pipeline = new Pipeline()
                      .setStages(Array(
                          elevScaler, slpScaler, evtScaler,
                          evhScaler, chScaler, tempAveScaler,
                          tempMinScaler, tempMaxScaler
                      ))
                    if (enableDebug) { System.out.println("[CreatedPipeline] Success!") }

                    val logReg = new LogisticRegression()
                    if (enableDebug) { System.out.println("[CreatedLogisticRegressionClassifier] Success!") }

                    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
                        //.addGrid(hashingTF.numFeatures, Array(1024, 2048))
                        .addGrid(logReg.fitIntercept, Array(true,false))
                        .addGrid(logReg.regParam,	Array(0.01, 0.0001))
                        .addGrid(logReg.maxIter,	Array(10, 15))
                        .addGrid(logReg.threshold,	Array(0, 0.25))
                        .addGrid(logReg.tol,	Array(0.0001, 0.01))
                        .build()
                    if (enableDebug) { System.out.println("[CreatedParamGrid] Success!")}

                    val cv = new TrainValidationSplit()
                        .setEstimator(pipeline)
                        .setEvaluator(new BinaryClassificationEvaluator())
                        .setEstimatorParamMaps(paramGrid)
                        .setTrainRatio(0.8)
                        .setParallelism(2)
                    if (enableDebug) { System.out.println("[CreatedTrainVaidationSplit] Success!")}

                    val Array(trainingData: Dataset[Row], testData: Dataset[Row]) = df.randomSplit(Array(0.8, 0.2))
                    val model: TrainValidationSplitModel = cv.fit(trainingData)
                    if (enableDebug) { System.out.println("[RanCrossValidation] Success!")}


                    /*
                        Best Parameters
                    */
                    val numFeatures: Int = model.bestModel
                        .asInstanceOf[PipelineModel]
                        .stages(1)
                        .asInstanceOf[HashingTF]
                        .getNumFeatures
                    val fitIntercept: Boolean = model.bestModel
                        .asInstanceOf[PipelineModel]
                        .stages(3)
                        .asInstanceOf[LogisticRegression]
                        .getFitIntercept
                    val regParam: Double = model.bestModel
                        .asInstanceOf[PipelineModel]
                        .stages(3)
                        .asInstanceOf[LogisticRegression]
                        .getRegParam
                    val maxIter: Double = model.bestModel
                        .asInstanceOf[PipelineModel]
                        .stages(3)
                        .asInstanceOf[LogisticRegression]
                        .getMaxIter
                    val threshold: Double = model.bestModel
                        .asInstanceOf[PipelineModel]
                        .stages(3)
                        .asInstanceOf[LogisticRegression]
                        .getThreshold
                    val tol: Double = model.bestModel
                        .asInstanceOf[PipelineModel]
                        .stages(3)
                        .asInstanceOf[LogisticRegression]
                        .getTol

                    println("--Parameters of the best Model--")
                    println("numFeatures: ", numFeatures)
                    println("fitIntercept: ", fitIntercept)
                    println("regParam: ", regParam)
                    println("maxIter: ", maxIter)
                    println("threshold: ", threshold)
                    println("tol: ", tol)
                    println()

                    /*
                        Running and Evaluation
                    */
                    val predictions: DataFrame = model.transform(testData)
                    predictions.select("text", "sentiment", "label", "prediction").show()

                    val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
                        .setLabelCol("label")
                        .setRawPredictionCol("prediction")

                    val accuracy: Double = binaryClassificationEvaluator.evaluate(predictions)
                    println(s"Accuracy of the test set is $accuracy")


                // TODO: Task 5: Temporal analysis - 2
                case "5" =>
            }
        }
        
    }


  

}
