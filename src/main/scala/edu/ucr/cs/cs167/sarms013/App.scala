package edu.ucr.cs.cs167.sarms013

/**
 * @author ${user.name}
 */


// General
import org.apache.spark.SparkConf

// Task4 4: Machine Learning Imports
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object App {

    def main(args : Array[string]) {
        
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
                case 1 =>




                // TODO: Task 2: Spatio-temporal analysis
                case 2 =>




                // TODO: Task 3: Temporal analysis
                case 3 =>




                // TODO: Task 4: Fire Intensity Prediction
                case 4 =>




                // TODO: Task 5: Temporal analysis - 2
                case 5 =>
            }
        }
        
    }


  

}
